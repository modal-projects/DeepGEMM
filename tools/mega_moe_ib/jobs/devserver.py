#!/usr/bin/env python3
"""Mega MoE dev server: hold the cluster warm and answer A/B requests (see ../devclient.py).

Processes, per node (devserver.sh runs `python devserver.py supervise` on every node):
  supervisor  no CUDA. Starts the worker, restarts it when it dies or a request times out, and keeps the nodes in
              step through the Dict key dev/<cluster>/gen (any node bumps it; every node restarts its worker on a
              bump). Node 0 also owns the request slot: it pops dev/<cluster>/req, hands it to the worker through a
              local file, and writes dev/<cluster>/res/<id> (the worker's result, or a failure with the last log
              lines when the worker crashed, hung or timed out).
  worker      `python devserver.py worker`: torch.multiprocessing.spawn of one rank per GPU. Ranks init NCCL once,
              keep the symmetric buffers, DeepEP reference buffers and weights per (shape, tokens) case, and loop:
              rank 0 polls the inbox file and broadcasts each request. Per request: write the request's kernel
              headers over the private install, JIT-compile every (arm, size) variant in parallel over the local
              ranks (compile-only host hook, DG_DEV_COMPILE_ONLY), DeepEP correctness check per arm and size, then
              kineto timing with arms interleaved per round (A B C D A B C D ...).

Per-launch timing reproduces bench_kineto (deep_gemm/testing/bench.py): before every launch a 256 MB L2 flush,
torch.cuda._sleep(2e7) and the DeepEP barrier; the duration is the kernel's CUPTI duration from the chrome trace,
as tests/test_mega_moe.py reports it. `median` = median over (rank, round) of the per-rank mean over the timed
launches (the test's headline: the median of its per-rank "us" over repetitions).

Dict keys (fused-ib-state), K = dev/<cluster>/:
  K+req       request slot (client writes with skip_if_exists; node 0 pops)
  K+res/<id>  result          K+status  supervisor status (state, gen, request, phase)
  K+server    server info     K+gen     {'job', 'gen'}       K+stop  {'time'}: stop when newer than the server
"""
import ctypes
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

KP = 'DG_MEGA_MOE_'
# The knob names in the three lists below are switches of the reference kernels that preceded the in-repo one (the
# reference column of the tables is measured with them); the in-repo kernel reads none of them
DEFAULT_KNOBS = ('DG_MEGA_MOE_IB_SKIP_CREDIT=1,DG_MEGA_MOE_IB_HDR_CTX=1,DG_MEGA_MOE_IB_CONTEXTS=8,DG_MEGA_MOE_IB_FRONT=1,'
                 'DG_MEGA_MOE_IB_GPU_ACQ=1,DG_MEGA_MOE_IB_STREAM_COMBINE=1,DG_MEGA_MOE_PULL_META_EARLY=1')
# Read when the process starts or a buffer is allocated (Python / NCCL setup): fixed for the server's lifetime, so an
# arm may repeat the server's value but not change it (restart the server with another DEV_BASE_KNOBS instead)
STATIC_KNOBS = {'DG_MEGA_MOE_IB', 'DG_MEGA_MOE_IB_PCOMB_COPIES', 'DG_MEGA_MOE_IB_CONTEXTS', 'DG_MEGA_MOE_IB_SHARING', 'DG_MEGA_MOE_IB_EXCLUSIVE',
                'DG_MEGA_MOE_IB_RAIL', 'DG_MEGA_MOE_IB_GIN_TYPE', 'DG_MEGA_MOE_IB_VIRTUAL_LSA', 'DG_MEGA_MOE_IB_FORCE',
                'DG_MEGA_MOE_IB_NVL_REF', 'DG_MEGA_MOE_IB_TOLERANCE',
                # SKIP_CREDIT: ib.py sizes the QP depth from it at import, and kernels with and without
                # ncclGinOptFlagsMaySkipCreditCheck cannot share Gin contexts: interleaving them hung at a header
                # timeout (b300_2n_b, 2026-11-05 23:22). Restart the server to compare it.
                'DG_MEGA_MOE_IB_SKIP_CREDIT'}
# Knobs whose host line only checks presence (any value, "0" included, defines NAME 1); the server maps "0" to unset
PRESENCE_ONLY = {'DG_MEGA_MOE_IB_KEEP_REGS', 'DG_MEGA_MOE_IB_TRACE', 'DG_MEGA_MOE_IB_BF16_COMBINE',
                 'DG_MEGA_MOE_IB_FLAG_REDUCE', 'DG_MEGA_MOE_IB_NO_META_FIRST', 'DG_MEGA_MOE_IB_BACKOFF',
                 'DG_MEGA_MOE_IB_HDR_CTX', 'DG_MEGA_MOE_IB_SKIP_CREDIT', 'DG_MEGA_MOE_IB_RET_DB', 'DG_MEGA_MOE_IB_FRONT',
                 'DG_MEGA_MOE_IB_GPU_ACQ', 'DG_MEGA_MOE_PULL_META_EARLY', 'DG_MEGA_MOE_IB_COUNTS_EARLY',
                 'DG_MEGA_MOE_IB_LOCAL_FIRST', 'DG_MEGA_MOE_IB_SPLIT_SCAN', 'DG_MEGA_MOE_IB_COUNTS_IN_META',
                 'DG_MEGA_MOE_IB_SENDER_TABLES', 'DG_MEGA_MOE_IB_PULL_SKIP_SCAN', 'DG_MEGA_MOE_IB_FWD_DYN',
                 'DG_MEGA_MOE_L2_COLMAJOR', 'DG_MEGA_MOE_IB_DEBUG_D0', 'DG_MEGA_MOE_IB_META_FIRST_ALL',
                 'DG_MEGA_MOE_NVL_TRACE'}
# Headers the host extension also compiles in (layout constants): edits reach the kernel but not the host side
HOST_INCLUDED = ('deep_gemm/layout/', 'deep_gemm/scheduler/', 'deep_gemm/common/')
SHAPES = {
    'flash': dict(hidden=5120, inter=2304, experts=384, topk=6, shared=1),
    'megaseek': dict(hidden=5120, inter=2304, experts=1216, topk=19, shared=1),
}
# The kernel's trace and hang lines start with DG_IB_; OPUS_ is the prefix of reference trees older than that rename
TRACE_RE = re.compile(r'DG_IB_|OPUS_|NVL_TRACE|DeepGEMM|timeout|Timeout|FAIL_DETAIL')
STAGE_RE = re.compile(r'DeepGEMM STAGE|grid sync|barrier .*timeout|DG_IB_HANG|OPUS_HANG')
ALERT_RE = re.compile(r'DG_IB_HANG|OPUS_HANG|DeepGEMM STAGE|DeepGEMM .*timeout|grid sync|timeout|Timeout|trap|illegal|Traceback|Error|FAIL_DETAIL')

ROOT = os.environ.get('DEV_ROOT', '/tmp/devserver')
PLANNED_EXIT = 75  # a rank exits with this after an ORACLE request: restart for fresh buffers, not a failure
INBOX, OUTBOX = f'{ROOT}/inbox', f'{ROOT}/outbox'
PREP_OUT, JOBS_IN = f'{ROOT}/prep_out', f'{ROOT}/jobs_in'
PROGRESS = f'{ROOT}/progress.json'


# ----------------------------------------------------------------------------------------------------------------------
# Pure helpers (unit-tested locally: python devserver.py selftest)

def parse_knobs(s):
    """'K=V,K2=V2 -K3' -> [(K, V), (K2, V2), (K3, None)]; None = unset. Separators: commas or whitespace."""
    ops = []
    for tok in re.split(r'[,\s]+', (s or '').strip()):
        if not tok:
            continue
        if tok.startswith('-'):
            k, v = tok[1:], None
        elif '=' in tok:
            k, v = tok.split('=', 1)
            v = None if v == '' else v
        else:
            k, v = tok, '1'
        if not re.fullmatch(r'DG_MEGA_MOE_[A-Z0-9_]+', k):
            raise ValueError(f'knob {k!r}: arms may only set DG_MEGA_MOE_* names (A-Z, 0-9, _)')
        if v is not None and not re.fullmatch(r'[A-Za-z0-9_.+\-]+', v):
            raise ValueError(f'knob {k}={v!r}: value must be a plain token')
        ops.append((k, v))
    return ops


def arm_env(base, arm, static):
    """Knob env of one arm: base overlaid with arm. Returns (env without static knobs, notes); raises ValueError."""
    env, notes = {}, []
    for k, v in parse_knobs(base) + parse_knobs(arm):
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    for k in list(env):
        if k in PRESENCE_ONLY and env[k] == '0':
            del env[k]
            notes.append(f'{k}=0 treated as unset (the host line only checks presence, so "=0" would mean ON)')
        elif k in STATIC_KNOBS:
            if env[k] != static.get(k):
                raise ValueError(f'{k}={env[k]} differs from the server\'s {static.get(k)!r}: it is read at '
                                 f'allocation/import time, so restart the server with DEV_BASE_KNOBS changed')
            del env[k]
    for k in STATIC_KNOBS:
        if k in static and k not in env and k in [x for x, _ in parse_knobs(arm)]:
            raise ValueError(f'{k} cannot be unset per arm (allocation-time knob)')
    return env, notes


def is_oracle(req):
    return any(v not in (None, '0') for _, knobs in req.get('arms', []) for k, v in _safe_knobs(knobs)
               if k == 'DG_MEGA_MOE_IB_ORACLE')


def _safe_knobs(s):
    try:
        return parse_knobs(s)
    except ValueError:
        return []


def header_tag(headers):
    """Stable tag of a header set (relpath -> text); '' when empty (pristine install, no extra define)."""
    if not headers:
        return ''
    h = hashlib.sha1()
    for rel in sorted(headers):
        h.update(rel.encode() + b'\0' + headers[rel].encode() + b'\0')
    return '0x' + h.hexdigest()[:12]


def localise_failure(torch, y, ref, topk_idx, topk_weights, epr, lsa, rank, prev=(), hyps=None, max_tokens=8, thr_floor=0.08):
    """Where a failing fused output differs from the reference (one rank, one test). Pure tensor math, bounded output.
    y / ref: [T, H]; topk_idx: [T, K]; prev: [(label, y_prev, ref_prev)] of earlier tests on this rank (newest first);
    hyps: {name: [T, H] float tensor = what y would be under that hypothesis} (e.g. reference minus one node's part).
    Per bad token: relative error of the row and of its 4 quarters (the forward-combine item split), which 128-column
    blocks (one fp8 return-record scale each) and how many 16-byte units (8 bf16) are off, the token's picks per node,
    the 'missing / stale contribution' signature (corr = <err, ref> / |ref|^2 and e2 = |err|^2 / |ref|^2: a contribution c
    that is MISSING gives corr ~ -|c|^2/|ref|^2 and e2 ~ |c|^2/|ref|^2; a STALE one gives the same corr and e2 twice that),
    and the best-matching hypothesis per row and per quarter (residual |y - hyp| / |ref|)."""
    yf, rf = y.float(), ref.float()
    T, H = rf.shape
    err = yf - rf
    rn = rf.norm(dim=1).clamp_min(1e-30)
    rel = err.norm(dim=1) / rn
    finite = torch.isfinite(yf).all(dim=1)
    rel = torch.where(finite, rel, torch.full_like(rel, float('inf')))
    med = float(rel[finite].median()) if bool(finite.any()) else 0.0
    # Absolute threshold on |err| / |ref| per row: a passing row sits near 0.02 (calc_diff 2e-4 ~ rel^2 / 2) and the check
    # tolerance 1e-3 is rel 0.045. Not scaled by the median: when many rows are wrong the median is wrong too.
    thr = thr_floor
    bad = torch.nonzero(rel > thr).flatten().tolist()
    nodes = max(1, (int(topk_idx.max().item()) // epr) // lsa + 1) if topk_idx.numel() else 1
    my_node = rank // lsa
    hyp = {'ref': rf}
    for lab, yp, rp in prev:
        if yp is not None and yp.shape == yf.shape:
            hyp[f'{lab}_out'] = yp.float()
        if rp is not None and rp.shape == yf.shape:
            hyp[f'{lab}_ref'] = rp.float()
    hyp['zero'] = torch.zeros_like(rf)
    for k, v in (hyps or {}).items():
        hyp[k] = v.float()
    q = H // 4
    out = dict(rank=rank, node=my_node, tokens=T, bad_tokens=len(bad), thr=round(thr, 4), median_row_rel=round(med, 5),
               bad_idx=bad[:64], rows=[])
    for t in bad[:max_tokens]:
        e, r, yy = err[t], rf[t], yf[t]
        row = dict(token=t, rel=round(float(rel[t]), 4) if bool(finite[t]) else 'nonfinite')
        qv = [float(e[i * q:(i + 1) * q].norm() / r[i * q:(i + 1) * q].norm().clamp_min(1e-30)) for i in range(4)]
        row['quarters'] = [round(v, 4) if v == v and v != float('inf') else 'nonfinite' for v in qv]  # strict-JSON safe
        nb = H // 128
        eb = e[:nb * 128].view(nb, 128).norm(dim=1) / r[:nb * 128].view(nb, 128).norm(dim=1).clamp_min(1e-30)
        bb = (eb > thr) | ~torch.isfinite(eb)
        row['bad_blocks128'] = f'{int(bb.sum())}/{nb}'
        row['block_mask'] = ''.join('X' if b else '.' for b in bb.tolist())
        nu = H // 8
        eu = e[:nu * 8].view(nu, 8).norm(dim=1)
        bu = (eu > thr * float(rn[t]) / (nu ** 0.5)) | ~torch.isfinite(eu)
        ul = bu.tolist()
        runs = sum(1 for i, b in enumerate(ul) if b and (i == 0 or not ul[i - 1]))
        idx = [i for i, b in enumerate(ul) if b]
        row['bad_units16B'] = f'{len(idx)}/{nu}'
        row['unit_span'] = [idx[0], idx[-1], runs] if idx else []
        r2 = float((r * r).sum().clamp_min(1e-30))
        row['corr'] = round(float((e * r).sum()) / r2, 4) if bool(finite[t]) else None
        row['e2'] = round(float((e * e).sum()) / r2, 4) if bool(finite[t]) else None
        ex = topk_idx[t].tolist()
        w = topk_weights[t].tolist() if topk_weights is not None else [0.0] * len(ex)
        per_node = [0] * nodes
        wn = [0.0] * nodes
        for x, ww in zip(ex, w):
            if x >= 0:
                per_node[min(nodes - 1, (x // epr) // lsa)] += 1
                wn[min(nodes - 1, (x // epr) // lsa)] += abs(ww)
        row['picks_per_node'] = per_node
        row['absw_per_node'] = [round(v, 3) for v in wn]
        row['experts'] = [[x, x // epr] for x in ex if x >= 0]
        if bool(finite[t]):
            res = {k: float((yy - v[t]).norm() / rn[t]) for k, v in hyp.items()}
            best = min(res, key=res.get)
            row['best'] = [best, round(res[best], 4)]
            row['hyp'] = {k: round(v, 4) for k, v in sorted(res.items(), key=lambda kv: kv[1])[:6]}
            bq = []
            for i in range(4):
                sl = slice(i * q, (i + 1) * q)
                rq = r[sl].norm().clamp_min(1e-30)
                rr = {k: float((yy[sl] - v[t][sl]).norm() / rq) for k, v in hyp.items()}
                b = min(rr, key=rr.get)
                bq.append([b, round(rr[b], 3)])
            row['best_per_quarter'] = bq
        out['rows'].append(row)
    r0 = out['rows'][0] if out['rows'] else None
    out['line'] = (f'FAIL_DETAIL rank={rank} node={my_node} bad_tokens={len(bad)}/{T} thr={thr:.3f}' +
                   (f' first: token={r0["token"]} rel={r0["rel"]} quarters={r0["quarters"]} blocks128={r0["bad_blocks128"]} '
                    f'units16B={r0["bad_units16B"]} picks/node={r0["picks_per_node"]} corr={r0["corr"]} e2={r0["e2"]} '
                    f'best={r0.get("best")} per_quarter={[b[0] for b in r0.get("best_per_quarter", [])]}' if r0 else
                    ' (no row above the threshold: the failure is spread over all rows, or stats / nonfinite only)'))
    return out


def quantile(sorted_vals, p):
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))] if sorted_vals else None


def parse_cpulist(text):
    out = set()
    for part in text.strip().split(','):
        if part:
            a, _, b = part.partition('-')
            out |= set(range(int(a), int(b or a) + 1))
    return out


def gin_partition(root=None, proc='/proc', apply=True):
    """Give every NCCL GIN progress thread of this container a core of its own and keep everything else off those cores.

    Why (measured 2026-11-12): a GIN put is rung on the NIC by the communicator's CPU progress thread, a loop of
    'sweep the QPs, sched_yield()'. Whenever any non-yielding thread (python housekeeping, a heartbeat, a compiler, a
    rank's main thread) is placed on the core that thread runs on, its next yield hands the core away for up to a
    scheduler slice (2.8 ms) and every rank waits that long for the late put: the ms-scale 'stalled launches'. The
    container has no CAP_SYS_NICE, so priorities are not available; affinity is.

    GIN threads are recognised by name ('NCCL GIN P..': the ranks run with NCCL_SET_THREAD_NAME=1) or, failing that, as
    threads of the rank processes that NCCL narrowed to one NUMA node and that have spun (user-heavy) all their life. Threads of the cases
    of the request about to be timed ({root}/gin_active.<pid>) get their cores first; if a NUMA node runs out, the
    remaining (idle-case) GIN threads share reserved cores among themselves (two yielding loops on one core cost
    microseconds, not milliseconds). Everything else in the container is confined to the unreserved CPUs.
    Idempotent; called by the pool at the start of every quiet window. DEV_PIN=0 switches it off. Returns a summary."""
    if os.environ.get('DEV_PIN', '1') == '0' or (apply and os.environ.get('DEV_FAKE_WORKER')):
        return {'on': False}
    t0 = time.time()
    try:
        with open('/sys/fs/cgroup/cpuset.cpus.effective') as f:
            full = parse_cpulist(f.read())
    except OSError:
        full = set()
    nodes = []
    try:
        for d in sorted(os.listdir('/sys/devices/system/node')):
            if d.startswith('node') and d[4:].isdigit():
                with open(f'/sys/devices/system/node/{d}/cpulist') as f:
                    nodes.append(parse_cpulist(f.read()))
    except OSError:
        pass
    tids = {}     # tid -> (pid, comm, affinity, is_rank)
    for pid in os.listdir(proc):
        if not pid.isdigit():
            continue
        try:
            with open(f'{proc}/{pid}/cmdline') as f:
                is_rank = 'spawn_main' in f.read()
            for t in os.listdir(f'{proc}/{pid}/task'):
                with open(f'{proc}/{pid}/task/{t}/comm') as f:
                    comm = f.read().strip()
                tids[int(t)] = (int(pid), comm, os.sched_getaffinity(int(t)), is_rank)
        except OSError:
            continue
    if not full:
        full = set().union(*[a for _, _, a, _ in tids.values()]) if tids else set()
    nodes = [n & full for n in nodes if n & full] or [set(full)]
    named = any(c.startswith('NCCL ') for _, c, _, _ in tids.values())
    if named:
        gin = [t for t, (_, c, _, _) in tids.items() if c.startswith('NCCL GIN')]
    else:
        # no names (ranks started without NCCL_SET_THREAD_NAME): threads of the rank processes that NCCL narrowed to one
        # NUMA node (or that an earlier call pinned) and that have spun for most of their life. Lifetime CPU, not a
        # sample: the thread that matters most is the one being starved right now.
        hz = os.sysconf('SC_CLK_TCK')
        with open(f'{proc}/uptime') as f:
            up = float(f.read().split()[0])
        gin = []
        for t, (pid, _, a, r) in tids.items():
            if not (r and a != full and any(a <= n for n in nodes)):
                continue
            try:
                with open(f'{proc}/{pid}/task/{t}/stat') as f:
                    x = f.read().rsplit(') ', 1)[1].split()
            except (OSError, IndexError):
                continue
            if (int(x[11]) + int(x[12])) / hz / max(up - int(x[19]) / hz, 1.0) > 0.2 and int(x[11]) >= int(x[12]):
                gin.append(t)
    active = set()
    if root:
        for f in os.listdir(root):
            if f.startswith('gin_active.') and f[11:].isdigit():
                try:
                    if os.path.exists(f'{proc}/{f[11:]}'):
                        with open(f'{root}/{f}') as fh:
                            active |= set(json.load(fh))
                    else:
                        os.unlink(f'{root}/{f}')
                except (OSError, ValueError):
                    pass
    gin.sort(key=lambda t: (t not in active, -t))    # this request's cases first, then the newest
    keep = int(os.environ.get('DEV_PIN_KEEP', 6))     # CPUs per NUMA node that are never reserved
    # ... and those are the LOWEST ones: the first CPUs of the cpuset (13 / 64 on the EPYC hosts) are where tasks from
    # outside the container's control land (b300_2n_a, 2026-11-12 21:0x: a GIN thread pinned to CPU 13 lost 2-7 ms of every
    # 10 ms for ~50 ms every 2 s, with nothing of ours on that core; cores 14.. were clean). Reserve from the top.
    reserved, plan, shared = {i: [] for i in range(len(nodes))}, {}, 0
    for t in gin:
        aff = tids[t][2]
        nd = max(range(len(nodes)), key=lambda i: len(nodes[i] & aff))
        free = sorted(nodes[nd] - set(reserved[nd]))
        if len(free) > keep:
            reserved[nd].append(free[-1])
            plan[t] = free[-1]
        elif reserved[nd]:
            plan[t] = reserved[nd][-1 - shared % len(reserved[nd])]   # share with the lowest-priority reserved cores
            shared += 1
    res_all = {c for v in reserved.values() for c in v}
    rest = full - res_all
    moved = failed = 0
    if rest and apply and gin:   # no GIN thread found: touch nothing
        for t, (_, _, aff, _) in tids.items():
            want = {plan[t]} if t in plan else rest
            if aff != want:
                try:
                    os.sched_setaffinity(t, want)
                    moved += 1
                except OSError:
                    failed += 1
    return {'on': True, 'named': named, 'gin': len(gin), 'gin_active': len([t for t in gin if t in active]),
            'reserved': len(res_all), 'shared': shared, 'rest_cpus': len(rest), 'threads': len(tids), 'moved': moved,
            'failed': failed, 's': round(time.time() - t0, 2),
            'cores': {str(i): sorted(v) for i, v in reserved.items() if v} if not apply else None}


def summarize_timing(per_rank):
    """per_rank: {rank: {round: [durations us]}} -> stats dict (see module docstring for `median`)."""
    rank_means, by_round, launches = [], {}, []
    steps = {}
    for r, rounds in per_rank.items():
        for rd, durs in rounds.items():
            if not durs:
                continue
            m = sum(durs) / len(durs)
            rank_means.append(m)
            by_round.setdefault(rd, []).append(m)
            launches += durs
            for i, d in enumerate(durs):
                steps[(rd, i)] = max(steps.get((rd, i), 0), d)
    if not rank_means:
        return {'n': 0}
    L = sorted(launches)
    S = sorted(steps.values())
    med = statistics.median(L)
    # Stalled launches: a launch STEP (same round, same index, all ranks) is stalled when its slowest rank
    # is above 1.5x the median launch. Reported per 10k steps, with the excess of each stalled step (us over the median,
    # first 64), how many ranks were slow in it, and `median_clean` = the headline median recomputed without those steps.
    stalled = {k for k, v in steps.items() if v > 1.5 * med}
    clean_means, slow_ranks = [], {}
    for r, rounds in per_rank.items():
        for rd, durs in rounds.items():
            keep = [d for i, d in enumerate(durs) if (rd, i) not in stalled]
            if keep:
                clean_means.append(sum(keep) / len(keep))
            for i, d in enumerate(durs):
                if (rd, i) in stalled and d > 1.5 * med:
                    slow_ranks[(rd, i)] = slow_ranks.get((rd, i), 0) + 1
    nranks = len(per_rank)
    stall = {
        'steps': len(steps), 'stall_steps': len(stalled),
        'stalls_per_10k': round(1e4 * len(stalled) / max(len(steps), 1), 1),
        'stall_excess_us': [round(steps[k] - med) for k in sorted(stalled)][:64],
        'stall_all_ranks': sum(1 for v in slow_ranks.values() if v == nranks),
        'stall_one_rank': sum(1 for v in slow_ranks.values() if v == 1),
        'median_clean': round(statistics.median(clean_means), 2) if clean_means else None,
    }
    return {
        **stall,
        'median': round(statistics.median(rank_means), 2),
        'rank_mean_min': round(min(rank_means), 2), 'rank_mean_max': round(max(rank_means), 2),
        'round_medians': [round(statistics.median(by_round[k]), 2) for k in sorted(by_round)],
        'p10': quantile(L, 0.1), 'p50': quantile(L, 0.5), 'p90': quantile(L, 0.9), 'p99': quantile(L, 0.99),
        'max': L[-1], 'step_p50': quantile(S, 0.5), 'step_max': S[-1], 'n': len(L),
        'over_1_5x': sum(x > 1.5 * med for x in L),
    }


def atomic_write_json(path, obj):
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def log_tail(path, n=40, pattern=None):
    try:
        with open(path, errors='replace') as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    lines = [l for l in lines if 'SyncActivityProfiler' not in l and 'USDT' not in l]
    if pattern is not None:
        hits = [l for l in lines if pattern.search(l)]
        return hits[-n:]
    return lines[-n:]


# ----------------------------------------------------------------------------------------------------------------------
# Compile pipeline (DEV_PIPELINE=1, the default; DEV_PIPELINE=0 = the old path: every node's ranks run nvcc inside the
# request).
#
#   store    {STORE}/obj/<key>/{kernel.cubin, ok | fail}: compiled objects by CONTENT. key = sha256(kernel.cu text, nvcc
#            flags with the include dir replaced by @INC@, digest of the deep_gemm include tree). Per node, survives
#            server restarts (STORE = /opt/devserver/store). {STORE}/prov/<cubin sha>.<digest16>.<src16> = "this cubin
#            was built from exactly this source and this include tree".
#   wrapper  The JIT's nvcc is {ROOT}/nvcc (DG_JIT_NVCC_COMPILER) -> `devserver.py nvccwrap`. With DEV_INC_DIGEST unset
#            it execs the real nvcc. Otherwise it computes the key of the compile the JIT asked for and copies the
#            store's cubin (seconds), waits for it while the node's pool has the job pending, replays a stored compile
#            error, or (nothing known) runs the real nvcc itself and stores the result. With DEV_NVCC_CAPTURE=<file> it
#            only records (source, flags) and fails the compile: that is how rank 0 learns the exact source of a
#            variant without compiling it. The JIT's own cache key / commit protocol is untouched.
#   pool     A thread in every node's supervisor: runs the compile jobs node 0 assigns to it (each distinct variant
#            compiled ONCE in the cluster, DEV_POOL_JOBS at a time per node, own process group, nice DEV_POOL_NICE,
#            against a private snapshot of the include tree with that request's headers), publishes the cubin through
#            the Dict, installs every finished cubin into the local store. SIGSTOPs all its compilers while the ranks
#            check / time ({ROOT}/quiet), so background compiles cannot perturb a measurement.
#   ahead    Node 0's supervisor takes up to DEV_LOOKAHEAD requests from the queue, has idle ranks capture their
#            sources ('prep' message) and starts their compiles while earlier requests run; a request is handed to the
#            ranks when its cubins are ready, so its own compile phase is store hits.

STORE = os.environ.get('DEV_STORE') or ('/opt/devserver/store' if os.path.isdir('/opt/devserver') else
                                        os.path.join(os.environ.get('DEV_ROOT', '/tmp/devserver'), 'store'))
INC_MARK = '@INC@'


# DEV_STALL_DIAG=1 (default off = unchanged behaviour): diagnostics for the "seconds per launch in the timing phase" events:
# per-step host timing of every launch in a timing block (launches over 1 s are logged with the step
# that took the time), a Python stack dump from a rank that sits 30 s inside a timing block, the rank logs copied to the
# results volume when a request's timing phase exceeds 120 s, and a pool-job timeout that does not count quiet windows.
STALL_DIAG = os.environ.get('DEV_STALL_DIAG', '0') == '1'

# DEV_HDR_FRESH=1 (default off = unchanged behaviour; set it on clean-ABI hosts): ONE HEADER SET PER RANK PROCESS.
# The JIT keys a kernel by its source text and by the hashes of its includes, which it caches per process; on the
# clean-ABI host kernel.cu no longer carries DG_MEGA_MOE_DEV_HDR, so after a header-only change the same plan hits the
# cubin of the old headers (memory or disk cache), and a miss files the NEW code under a key computed from the OLD
# include hashes (an entry that outlives the process). With the switch on: every worker generation starts without the
# warm case and with its own empty JIT cache directory, compiles nothing before its first request has installed its
# headers, and a request whose header set differs from the one the live ranks hold gets fresh ranks first. Cubins
# still come from the content-keyed store (source, flags, include digest) in seconds.
HDR_FRESH = os.environ.get('DEV_HDR_FRESH', '0') == '1'


def pipeline_on():
    return os.environ.get('DEV_PIPELINE', '1') != '0'


def sha256_hex(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
        h.update(b'\0')
    return h.hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def tree_digest(root):
    """Digest of every file under <root>/deep_gemm (relative path + bytes): the identity of a header set as the
    compiler sees it (cute / cutlass never change for a build)."""
    h = hashlib.sha256()
    base = os.path.join(root, 'deep_gemm')
    for d, dirs, fs in os.walk(base):
        dirs.sort()
        for f in sorted(fs):
            p = os.path.join(d, f)
            h.update(os.path.relpath(p, root).encode() + b'\0')
            with open(p, 'rb') as fh:
                h.update(fh.read())
            h.update(b'\0')
    return h.hexdigest()[:32]


def norm_flags(flags, live_inc):
    """nvcc flags after `--output-file X`, with the live include dir replaced by a marker (one string)."""
    return ' '.join(INC_MARK if f == live_inc else f for f in flags)


def job_key(src, flags_norm, digest):
    return sha256_hex(src, flags_norm, digest)[:40]


def cubin_code_sha(path):
    """sha256 over every ELF section of a cubin except string / symbol tables. nvcc puts a per-invocation hash into
    internal symbol NAMES (`_INTERNAL_<hash>_kernel_cu_<hash>_<pid>`), so two compiles of the same source are not
    byte-identical; code, constants, relocations and nv.info are."""
    import struct
    with open(path, 'rb') as f:
        b = f.read()
    try:
        if b[:4] != b'\x7fELF' or b[4] != 2:
            return None
        shoff, = struct.unpack_from('<Q', b, 0x28)
        shentsize, shnum = struct.unpack_from('<HH', b, 0x3A)
        h = hashlib.sha256()
        for i in range(shnum):
            _, typ, _, _, off, size = struct.unpack_from('<IIQQQQ', b, shoff + i * shentsize)
            if typ in (2, 3, 8):  # SYMTAB, STRTAB, NOBITS
                continue
            h.update(struct.pack('<IQ', typ, size) + b[off:off + size])
        return h.hexdigest()[:32]
    except Exception:  # noqa: BLE001
        return None


SCREEN_KNOB = 'DG_MEGA_MOE_DEV_PTXAS_O'
SRC_MARK = '#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>'  # kernel.cu = knob #defines, this line, instantiation


def entry_src_flags(path, live_inc):
    """(kernel.cu text, normalized nvcc flags) of a JIT cache entry."""
    with open(f'{path}/kernel.cu') as f:
        src = f.read()
    with open(f'{path}/meta.json') as f:
        cmd = json.load(f)['command'].split(' ')
    return src, norm_flags(cmd[cmd.index('--output-file') + 2:], live_inc)



def screening_flags(src):
    """SCREENING mode, per arm: the knob DG_MEGA_MOE_DEV_PTXAS_O=2 (or 1) reaches kernel.cu as a #define the kernel
    ignores (so the variant has its own JIT key and store key); the pipeline then compiles that variant with
    `ptxas -O<n>` instead of the default -O3: about half the compile time at EP32, slightly different machine code.
    Never the default; results say so. No effect with DEV_PIPELINE=0."""
    m = re.search(rf'^#define {SCREEN_KNOB} ([12])\s*$', src, re.M)
    return [f'--ptxas-options=-O{m.group(1)}'] if m else []


def obj_dir(key):
    return f'{STORE}/obj/{key}'


def obj_state(key):
    """'ok' | 'fail' | None for a store object."""
    d = obj_dir(key)
    return 'ok' if os.path.exists(f'{d}/ok') else 'fail' if os.path.exists(f'{d}/fail') else None


def prov_path(cubin_sha, digest, src_sha):
    return f'{STORE}/prov/{cubin_sha[:40]}.{digest[:16]}.{src_sha[:16]}'


def store_install(key, cubin_path=None, cubin_bytes=None, meta=None, fail=None):
    """Atomically publish a store object (first writer wins). Returns the object's state."""
    d = obj_dir(key)
    if obj_state(key):
        return obj_state(key)
    os.makedirs(f'{STORE}/obj', exist_ok=True)
    os.makedirs(f'{STORE}/prov', exist_ok=True)
    tmp = f'{d}.tmp{os.getpid()}.{time.time_ns()}'
    os.makedirs(tmp)
    meta = dict(meta or {}, key=key, time=time.time())
    if fail is not None:
        atomic_write_json(f'{tmp}/fail', dict(meta, output=fail))
    else:
        if cubin_bytes is not None:
            with open(f'{tmp}/kernel.cubin', 'wb') as f:
                f.write(cubin_bytes)
        else:
            shutil.copyfile(cubin_path, f'{tmp}/kernel.cubin')
        meta['cubin_sha'] = file_sha(f'{tmp}/kernel.cubin')
        if meta.get('digest') and meta.get('src_sha'):
            open(prov_path(meta['cubin_sha'], meta['digest'], meta['src_sha']), 'w').close()
        atomic_write_json(f'{tmp}/ok', meta)
    try:
        os.rename(tmp, d)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
    return obj_state(key)


def obj_meta(key):
    st = obj_state(key)
    if not st:
        return None
    try:
        with open(f'{obj_dir(key)}/{st}') as f:
            return dict(json.load(f), state=st)
    except (OSError, ValueError):
        return None


def pool_alive(root):
    try:
        return time.time() - os.path.getmtime(f'{root}/pool.alive') < 30
    except OSError:
        return False


def find_real_nvcc():
    home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if not home:
        w = shutil.which('nvcc')
        home = os.path.dirname(os.path.dirname(os.path.realpath(w))) if w else '/usr/local/cuda'
    return os.environ.get('DEV_REAL_NVCC') or os.path.join(home, 'bin', 'nvcc')


def write_nvcc_wrapper(root):
    """{root}/nvcc: what the JIT runs instead of nvcc. `--version` (hashed into every JIT key) and any compile
    without DEV_INC_DIGEST go straight to the real compiler."""
    real = find_real_nvcc()
    path = f'{root}/nvcc'
    with open(path + '.tmp', 'w') as f:
        f.write(f'#!/bin/sh\n# dev server JIT compiler wrapper (devserver.py, compile pipeline)\n'
                f'if [ -z "$DEV_INC_DIGEST" ] || [ "$1" = "--version" ]; then exec {real} "$@"; fi\n'
                f'DEV_REAL_NVCC={real} exec {sys.executable} {os.path.abspath(__file__)} nvccwrap "$@"\n')
    os.chmod(path + '.tmp', 0o755)
    os.replace(path + '.tmp', path)
    return path, real


def nvccwrap(argv):
    """See the pipeline comment above. argv = the JIT's nvcc arguments: kernel.cu --cubin --output-file X flags..."""
    real = os.environ['DEV_REAL_NVCC']
    if len(argv) < 4 or argv[1] != '--cubin' or argv[2] != '--output-file' or not argv[0].endswith('.cu'):
        os.execv(real, [real] + argv)
    src_path, out = argv[0], argv[3]
    with open(src_path) as f:
        src = f.read()
    flags = norm_flags(argv[4:], os.environ.get('DEV_LIVE_INC', ''))
    cap = os.environ.get('DEV_NVCC_CAPTURE')
    if cap and 'mega_moe_impl' not in src:
        # Not the kernel a capture is after (an empty JIT cache also compiles the locality-domain probe and the
        # layout kernels on the first launch: recording one of those gave the request a wrong source)
        os.execv(real, [real] + argv)
    if cap:
        atomic_write_json(cap, {'src': src, 'flags': flags})
        print('DEV_NVCC_CAPTURE: source recorded, not compiled')
        return 3
    digest = os.environ['DEV_INC_DIGEST']
    root = os.environ.get('DEV_ROOT', '/tmp/devserver')
    key = job_key(src, flags, digest)
    src_sha = sha256_hex(src)
    t0, waited = time.time(), False
    limit = float(os.environ.get('DEV_WRAP_WAIT', 3600))
    while True:
        st = obj_state(key)
        if st == 'ok':
            m = obj_meta(key) or {}
            shutil.copyfile(f'{obj_dir(key)}/kernel.cubin', out)
            if file_sha(out) != m.get('cubin_sha'):
                print(f'DEVPIPE: store object {key} is corrupt (sha mismatch)')
                return 1
            os.makedirs(f'{STORE}/prov', exist_ok=True)
            open(prov_path(m['cubin_sha'], digest, src_sha), 'w').close()
            print(f'DEVPIPE: served {key} from the store ({m.get("how")}, node {m.get("node")}, waited {time.time() - t0:.1f} s)')
            return 0
        if st == 'fail':
            m = obj_meta(key) or {}
            print(m.get('output', 'compile failed (no output stored)'))
            return 1
        # ... but not while the pool is stopped for a check / timing phase ({root}/quiet): a launch-time compile (the
        # first-launch rendezvous compiles every kernel of the buffer) would wait for a job that cannot run
        if os.path.exists(f'{root}/pend/{key}') and pool_alive(root) and time.time() - t0 < limit \
                and not os.path.exists(f'{root}/quiet'):
            waited = True
            time.sleep(0.5)
            continue
        break
    # Not in the store and no pool job: compile here (the old path), then store it for the other local ranks
    p = subprocess.run([real] + argv + screening_flags(src), capture_output=True, text=True)
    output = p.stdout + p.stderr
    sys.stdout.write(output)
    if p.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 0:
        store_install(key, cubin_path=out, meta=dict(how='inline', node=os.environ.get('NODE_RANK', '?'), digest=digest,
                                                     src_sha=src_sha, secs=round(time.time() - t0, 1), waited=waited))
        os.makedirs(f'{STORE}/prov', exist_ok=True)
        open(prov_path(file_sha(out), digest, src_sha), 'w').close()
    return p.returncode


class Snapshots:
    """Private include trees per header set: {root}/snap/<tag>/ = pristine deep_gemm + the request's headers, with
    cute / cutlass symlinked to the pristine install."""

    def __init__(self, root, include_orig):
        self.root, self.orig, self.digests = f'{root}/snap', include_orig, {}

    def get(self, tag, headers):
        """-> (dir, digest). `headers` (relpath -> text) is needed the first time a tag is seen."""
        name = tag or 'pristine'
        d = f'{self.root}/{name}'
        if name in self.digests:
            return d, self.digests[name]
        if headers is None and tag:
            raise KeyError(f'header set {tag} unknown')
        tmp = f'{d}.tmp{os.getpid()}.{time.time_ns()}'
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        for e in os.listdir(self.orig):
            if e == 'deep_gemm':
                shutil.copytree(f'{self.orig}/{e}', f'{tmp}/{e}')
            else:
                os.symlink(f'{self.orig}/{e}', f'{tmp}/{e}')
        for rel, text in (headers or {}).items():
            dst = os.path.normpath(os.path.join(tmp, rel))
            if not dst.startswith(tmp + '/') or not rel.startswith('deep_gemm/'):
                raise ValueError(f'header path {rel!r} must be under deep_gemm/')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, 'w') as f:
                f.write(text)
        shutil.rmtree(d, ignore_errors=True)
        os.rename(tmp, d)
        self.digests[name] = tree_digest(d)
        return d, self.digests[name]


class Pool:
    """Per-node compile pool (threads inside the supervisor). Protocol, Dict keys under K + 'pipe/':
         jobs          node 0: {'job', 'jobs': [{key, node, src, flags, tag, digest}], 'drop': [keys]}  (order = priority)
         hdr/<tag>     node 0: the header set of a tag (relpath -> text)
         done/<node>   every node: {'job', 'res': {key: {ok, cubin_sha, secs, node, err, infra}}}
         cubin/<key>   the compiling node: zlib(kernel.cubin)"""

    def __init__(self, sup):
        import threading
        self.sup, self.node, self.K = sup, sup.node, sup.K + 'pipe/'
        self.root = ROOT
        self.max_jobs = int(os.environ.get('DEV_POOL_JOBS', 16))
        self.nice = int(os.environ.get('DEV_POOL_NICE', 10))
        self.timeout = float(os.environ.get('DEV_POOL_TIMEOUT', 2400))
        self.real = find_real_nvcc()
        self.live_inc = os.environ.get('DEV_LIVE_INC', '')
        self.snaps = Snapshots(ROOT, os.environ.get('DEV_INCLUDE_ORIG', ''))
        self.lock = threading.Lock()
        self.running = {}      # key -> dict(proc, t0, dir, job)
        self.done = {}         # key -> result (this node's compiles)
        self.seen = {}         # key -> result (any node's), installed locally
        self.doc = {'jobs': [], 'drop': []}   # node 0 sets this directly
        self.doc_ver = 0
        self.quiet = False
        self.quiet_total, self.quiet_t0 = 0.0, None   # DEV_STALL_DIAG: seconds spent in quiet windows
        # One supervisor per node index: heartbeat under pipe/owner/<node> (seen once: a second runner incarnation's
        # node 1 compiled and published every node-1 job a second time; node 0 installed its cubins)
        self.t0 = time.time()
        self.owner_id = f"{os.environ.get('MODAL_TASK_ID') or os.uname().nodename}:{os.getpid()}"
        self.dup = None        # id of an older live supervisor with our node index: we then compile / publish nothing
        self.dup_seen = set()  # younger ones we have shouted about
        self.stop = False
        self.stats = {}
        for d in ('pend', 'pool'):
            shutil.rmtree(f'{self.root}/{d}', ignore_errors=True)
            os.makedirs(f'{self.root}/{d}', exist_ok=True)
        for f in ('quiet', 'quiet.ack'):
            try:
                os.unlink(f'{self.root}/{f}')
            except OSError:
                pass
        # stored compile ERRORS do not outlive a server (a transient failure must not stick); cubins do
        for k in os.listdir(f'{STORE}/obj') if os.path.isdir(f'{STORE}/obj') else []:
            if os.path.exists(f'{STORE}/obj/{k}/fail') or '.tmp' in k:
                shutil.rmtree(f'{STORE}/obj/{k}', ignore_errors=True)
        self.st = make_state()  # own handle for the worker thread
        self.threads = [threading.Thread(target=self.guard, args=(fn,), daemon=True) for fn in (self.quiet_loop, self.loop)]
        for t in self.threads:
            t.start()

    def say(self, msg):
        print(f'[pool node{self.node} {time.strftime("%H:%M:%S")}] {msg}', flush=True)

    def guard(self, fn):
        while not self.stop:
            try:
                fn()
            except Exception:  # noqa: BLE001
                self.say(f'{fn.__name__} error:\n{traceback.format_exc()}')
                time.sleep(2)

    # -- quiet: no background compiler runs while the ranks check / time
    def signal_all(self, sig):
        with self.lock:
            procs = [j['proc'] for j in self.running.values()]
        for p in procs:
            try:
                os.killpg(p.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

    def quiet_loop(self):
        qf, af = f'{self.root}/quiet', f'{self.root}/quiet.ack'
        while not self.stop:
            try:
                with open(qf) as f:
                    nonce = f.read().strip()
            except OSError:
                nonce = None
            if nonce and not (self.quiet and self.quiet == nonce):
                if STALL_DIAG and not self.quiet:
                    self.quiet_t0 = time.time()
                self.quiet = nonce
                self.signal_all(signal.SIGSTOP)
                # flush what the compiles wrote (cubins, PTX temporaries) now, not during the timing: on these nodes
                # every core runs a busy-polling NCCL proxy thread, so even kernel writeback shows up as launch stalls
                t = time.time()
                os.sync()
                if time.time() - t > 2:
                    self.say(f'sync before the quiet window took {time.time() - t:.1f} s')
                # every GIN progress thread on its own core, everything else off those cores (see gin_partition)
                try:
                    pin = gin_partition(self.root)
                except Exception:  # noqa: BLE001
                    pin = {'on': True, 'error': traceback.format_exc()[-800:]}
                    self.say(f'gin_partition failed:\n{pin["error"]}')
                if pin.get('moved'):
                    self.say(f'gin_partition: {pin}')
                atomic_write_json(f'{self.root}/pin.json', dict(pin, nonce=nonce))
                with open(af + '.tmp', 'w') as f:
                    f.write(nonce)
                os.replace(af + '.tmp', af)
            elif not nonce and self.quiet:
                if STALL_DIAG and self.quiet_t0 is not None:
                    self.quiet_total += time.time() - self.quiet_t0
                    self.quiet_t0 = None
                self.quiet = False
                self.signal_all(signal.SIGCONT)
            with open(f'{self.root}/pool.alive', 'w'):
                pass
            time.sleep(0.05)

    # -- jobs
    def start(self, job):
        key = job['key']
        d = f'{self.root}/pool/{key}'
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        try:
            hdr = None
            name = job['tag'] or 'pristine'
            if job['tag'] and name not in self.snaps.digests:
                hdr = safe(self.st.get, f'{self.K}hdr/{job["tag"]}')
                if hdr is None:
                    raise RuntimeError(f'header set {job["tag"]} not published')
            inc, digest = self.snaps.get(job['tag'], hdr)
            if digest != job['digest']:
                raise RuntimeError(f'include digest {digest} != {job["digest"]} for header set {name}')
            if job_key(job['src'], job['flags'], digest) != key:
                raise RuntimeError('job key does not match its content')
        except Exception as e:  # noqa: BLE001
            self.finish(key, dict(ok=False, infra=True, err=f'{type(e).__name__}: {e}', node=self.node, secs=0))
            return
        with open(f'{d}/kernel.cu', 'w') as f:
            f.write(job['src'])
        flags = [inc if x == INC_MARK else x for x in job['flags'].split(' ')]
        if self.live_inc and not os.environ.get('DEV_FAKE_WORKER'):
            # __FILE__ strings (assert messages) name the live install, as in a compile inside the ranks: the cubin's
            # code AND data sections are then identical to the old path's (checked on b300_2n_a, 2026-11-12)
            flags.append(f'--compiler-options=-fmacro-prefix-map={inc}={self.live_inc}')
        flags += screening_flags(job['src'])
        cmd = ['nice', '-n', str(self.nice), self.real, f'{d}/kernel.cu', '--cubin', '--output-file', f'{d}/kernel.cubin'] + flags
        env = {k: v for k, v in os.environ.items() if not k.startswith('DEV_INC_DIGEST')}
        with self.lock:
            if self.quiet:   # never start a compiler inside a quiet window
                return
            logf = open(f'{d}/log', 'w')
            proc = subprocess.Popen(cmd, cwd=d, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
            self.running[key] = dict(proc=proc, t0=time.time(), dir=d, job=job, logf=logf)
            if STALL_DIAG:
                self.running[key]['q0'] = self.quiet_total

    def finish(self, key, res):
        self.done[key] = res
        self.seen[key] = res
        self.publish = True

    def reap(self):
        with self.lock:
            items = list(self.running.items())
        for key, j in items:
            rc = j['proc'].poll()
            now = time.time()
            if STALL_DIAG and rc is None and now - j['t0'] - (self.quiet_total - j.get('q0', self.quiet_total)) <= self.timeout:
                continue   # its timeout clock does not run during quiet windows (the compilers are SIGSTOPped there)
            if rc is None and now - j['t0'] > self.timeout and not self.quiet:
                try:
                    os.killpg(j['proc'].pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                rc = -9
            if rc is None:
                continue
            j['logf'].close()
            with self.lock:
                self.running.pop(key, None)
            secs = round(now - j['t0'], 1)
            out = ''
            try:
                with open(f'{j["dir"]}/log', errors='replace') as f:
                    out = f.read()
            except OSError:
                pass
            cubin = f'{j["dir"]}/kernel.cubin'
            job = j['job']
            meta = dict(how='pool', node=self.node, digest=job['digest'], src_sha=sha256_hex(job['src']), secs=secs)
            if rc == 0 and os.path.isfile(cubin) and os.path.getsize(cubin) > 0:
                store_install(key, cubin_path=cubin, meta=meta)
                m = obj_meta(key) or {}
                import zlib
                with open(f'{obj_dir(key)}/kernel.cubin', 'rb') as f:
                    blob = zlib.compress(f.read(), 1)
                ok = safe(self.st.put, f'{self.K}cubin/{key}', blob, default=False) is not False
                res = dict(ok=True, cubin_sha=m.get('cubin_sha'), secs=secs, node=self.node, digest=job['digest'],
                           src_sha=meta['src_sha'], shipped=ok)
                self.say(f'job {key[:10]} ok in {secs} s ({len(blob) >> 10} KiB shipped)')
            elif rc is not None and (rc < 0 or rc > 128 or j.get('killed')):
                res = dict(ok=False, infra=True, err=f'compiler ended with code {rc} after {secs} s', node=self.node, secs=secs)
                self.say(f'job {key[:10]} INFRA failure: {res["err"]}')
            else:
                lines = [l for l in out.splitlines() if 'Vector Type not specified' not in l]
                text = '\n'.join(lines)[-6000:]
                store_install(key, fail=text, meta=meta)
                res = dict(ok=False, err=text, secs=secs, node=self.node, digest=job['digest'], src_sha=meta['src_sha'])
                self.say(f'job {key[:10]} compile error after {secs} s')
            shutil.rmtree(j['dir'], ignore_errors=True)
            self.finish(key, res)
            try:
                os.unlink(f'{self.root}/pend/{key}')
            except OSError:
                pass

    def publish_existing(self, key):
        m = obj_meta(key) or {}
        if m.get('state') == 'ok':
            import zlib
            with open(f'{obj_dir(key)}/kernel.cubin', 'rb') as f:
                blob = zlib.compress(f.read(), 1)
            safe(self.st.put, f'{self.K}cubin/{key}', blob)
            self.finish(key, dict(ok=True, cubin_sha=m.get('cubin_sha'), secs=0, node=self.node, digest=m.get('digest'),
                                  src_sha=m.get('src_sha'), stored=m.get('how')))
        else:
            self.finish(key, dict(ok=False, err=m.get('output', ''), secs=0, node=self.node, digest=m.get('digest'),
                                  src_sha=m.get('src_sha'), stored=m.get('how')))

    def install_remote(self, key, res):
        """Another node compiled it: fetch the cubin (or the compile error) into the local store."""
        if res.get('infra'):
            return True
        if res.get('ok'):
            blob = safe(self.st.get, f'{self.K}cubin/{key}')
            if blob is None:
                return False
            import zlib
            data = zlib.decompress(blob)
            if hashlib.sha256(data).hexdigest() != res.get('cubin_sha'):
                self.say(f'job {key[:10]}: shipped cubin does not match its sha; ignoring it')
                return True
            store_install(key, cubin_bytes=data, meta=dict(how='shipped', node=res.get('node'), digest=res.get('digest'),
                                                           src_sha=res.get('src_sha'), secs=res.get('secs')))
        else:
            store_install(key, fail=res.get('err', ''), meta=dict(how='shipped', node=res.get('node'), digest=res.get('digest'),
                                                                  src_sha=res.get('src_sha'), secs=res.get('secs')))
        return True

    def loop(self):
        last, self.publish = 0, False
        others = {}
        while not self.stop:
            time.sleep(0.2)
            self.reap()
            now = time.time()
            if self.publish and not self.dup:
                self.publish = False
                safe(self.st.put, f'{self.K}done/{self.node}', {'job': self.sup.job, 'res': self.done})
            if now - last >= float(os.environ.get('DEV_POOL_POLL', 2)):
                last = now
                self.check_owner(now)
                if self.node != 0:
                    doc = safe(self.st.get, self.K + 'jobs') or {}
                    self.doc = doc if doc.get('job') == self.sup.job else {'jobs': [], 'drop': []}
                for n in range(self.sup.nnodes):
                    if n != self.node:
                        d = safe(self.st.get, f'{self.K}done/{n}') or {}
                        others[n] = d.get('res', {}) if d.get('job') == self.sup.job else {}
            doc = self.doc
            live = set()
            for job in doc.get('jobs', []):
                key = job['key']
                if key in self.seen:
                    continue
                res = next((o[key] for o in others.values() if key in o), None)
                if res is not None:
                    if obj_state(key) or self.install_remote(key, res):
                        self.seen[key] = res
                    else:
                        live.add(key)
                    continue
                if obj_state(key):
                    # already in this node's store (an earlier server, or a rank compiled it inline): the owner
                    # publishes it, the others wait for the owner's result
                    if job['node'] == self.node:
                        self.publish_existing(key)
                    continue
                live.add(key)
                if not os.path.exists(f'{self.root}/pend/{key}'):
                    open(f'{self.root}/pend/{key}', 'w').close()
                if job['node'] == self.node and job.get('state', 'queued') == 'queued' and key not in self.running and \
                        key not in self.done and len(self.running) < self.max_jobs and not self.quiet and not self.dup:
                    self.start(job)
            for key in os.listdir(f'{self.root}/pend'):
                if key not in live:
                    try:
                        os.unlink(f'{self.root}/pend/{key}')
                    except OSError:
                        pass
            for key in list(self.running):
                if key in doc.get('drop', []) or key not in live:
                    j = self.running.get(key)
                    if j and not j.get('killed'):
                        j['killed'] = True
                        try:
                            os.killpg(j['proc'].pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
            self.stats = {'running': len(self.running), 'waiting': sum(1 for j in doc.get('jobs', []) if j['node'] == self.node
                                                                      and j['key'] in live and j['key'] not in self.running)}

    def check_owner(self, now):
        """Detect a second live supervisor with this job and node index. The older one keeps the index; the younger one
        stops compiling and publishing and says so on every pass (its log line is the alarm)."""
        try:
            k = f'{self.K}owner/{self.node}'
            own = safe(self.st.get, k) or {}
            other = own.get('id') not in (None, self.owner_id) and own.get('job') == self.sup.job and now - own.get('time', 0) < 60
            if other and (own.get('t0', 0), str(own.get('id'))) < (self.t0, self.owner_id):
                if self.dup != own['id']:
                    self.dup = own['id']
                self.say(f'DUPLICATE SUPERVISOR for node {self.node} of {self.sup.job}: {own["id"]} (started {self.t0 - own.get("t0", 0):.0f} s '
                         f'before us) owns this node index; this one ({self.owner_id}) compiles and publishes NOTHING. Stop this container.')
                return
            if other and own['id'] not in self.dup_seen:
                self.dup_seen.add(own['id'])
                self.say(f'DUPLICATE SUPERVISOR for node {self.node} of {self.sup.job}: a younger one ({own["id"]}) is publishing under '
                         f'this node index; cubins shipped from this node since it started may be its builds. Stop that container.')
            self.dup = None
            safe(self.st.put, k, {'job': self.sup.job, 'id': self.owner_id, 't0': self.t0, 'time': now})
        except Exception:  # noqa: BLE001
            self.say(f'check_owner error:\n{traceback.format_exc()}')

    def results(self):
        """Node 0: every result known here (own compiles + other nodes')."""
        return self.seen

    def shutdown(self):
        self.stop = True
        self.signal_all(signal.SIGCONT)
        self.signal_all(signal.SIGKILL)


# ----------------------------------------------------------------------------------------------------------------------
# State (Modal Dict, or a directory for local tests: DEV_DICT_DIR)

class FileDict:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _p(self, k):
        return os.path.join(self.root, k.replace('/', '%'))

    def get(self, k, default=None):
        try:
            with open(self._p(k) + '.bin', 'rb') as f:
                return f.read()
        except OSError:
            pass
        try:
            with open(self._p(k)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return default

    def put(self, k, v, skip_if_exists=False):
        if isinstance(v, bytes):
            with open(self._p(k) + '.bin.tmp', 'wb') as f:
                f.write(v)
            os.replace(self._p(k) + '.bin.tmp', self._p(k) + '.bin')
            return True
        if skip_if_exists:
            try:
                fd = os.open(self._p(k) + '.lock', os.O_CREAT | os.O_EXCL)
                os.close(fd)
            except FileExistsError:
                return False
            if os.path.exists(self._p(k)):
                os.unlink(self._p(k) + '.lock')
                return False
        atomic_write_json(self._p(k), v)
        if skip_if_exists:
            os.unlink(self._p(k) + '.lock')
        return True

    def pop(self, k, default=None):
        try:
            tmp = self._p(k) + f'.pop{os.getpid()}'
            os.replace(self._p(k), tmp)
        except OSError:
            return default
        with open(tmp) as f:
            v = json.load(f)
        os.unlink(tmp)
        return v

    def __setitem__(self, k, v):
        self.put(k, v)

    def contains(self, k):
        return os.path.exists(self._p(k))


def make_state():
    if os.environ.get('DEV_DICT_DIR'):
        return FileDict(os.environ['DEV_DICT_DIR'])
    import modal
    return modal.Dict.from_name('fused-ib-state', environment_name=os.environ.get('DEV_MODAL_ENV') or None)


class ReqQueue:
    """FIFO of requests shared by every client: Modal Queue fused-ib-devq, partition <cluster> (a directory with
    DEV_DICT_DIR). Clients put; node 0's supervisor takes one at a time."""

    def __init__(self, cluster):
        self.cluster = cluster
        if os.environ.get('DEV_DICT_DIR'):
            self.dir = os.path.join(os.environ['DEV_DICT_DIR'], f'queue_{cluster}')
            os.makedirs(self.dir, exist_ok=True)
            self.q = None
        else:
            import modal
            self.q = modal.Queue.from_name('fused-ib-devq', environment_name=os.environ.get('DEV_MODAL_ENV') or None,
                                           create_if_missing=True)

    def put(self, req):
        if self.q is not None:
            self.q.put(req, partition=self.cluster)
        else:
            atomic_write_json(os.path.join(self.dir, f'{time.time():.6f}-{req["id"]}.json'), req)

    def get_nowait(self):
        if self.q is not None:
            import queue
            try:
                return self.q.get(block=False, partition=self.cluster)
            except queue.Empty:
                return None
        for f in sorted(os.listdir(self.dir)):
            if not f.endswith('.json'):
                continue
            try:
                os.replace(os.path.join(self.dir, f), os.path.join(self.dir, f + '.taken'))
            except OSError:
                continue
            if f.endswith('.json'):
                with open(os.path.join(self.dir, f + '.taken')) as fh:
                    return json.load(fh)
        return None

    def len(self):
        if self.q is not None:
            return self.q.len(partition=self.cluster)
        return len([f for f in os.listdir(self.dir) if f.endswith('.json')])

    def put_front(self, reqs):
        """Give requests back to the head of the queue (server stopping with requests it had taken ahead)."""
        if not reqs:
            return
        if self.q is not None:
            rest = self.q.get_many(1000, block=False, partition=self.cluster)
            self.q.put_many(list(reqs) + rest, partition=self.cluster)
        else:
            for i, r in enumerate(reqs):
                atomic_write_json(os.path.join(self.dir, f'0000000000.{i:06d}-{r["id"]}.json'), r)

    def peek(self):
        """The queued requests in order, without taking them."""
        if self.q is not None:
            return list(self.q.iterate(partition=self.cluster))
        out = []
        for f in sorted(os.listdir(self.dir)):
            if f.endswith('.json'):
                with open(os.path.join(self.dir, f)) as fh:
                    out.append(json.load(fh))
        return out

    def remove(self, match):
        """Take out every queued request whose id satisfies match(id); returns them. A Modal Queue has no delete:
        drain and put the rest back in order (a request another client puts during the ~1 s window lands before
        the re-put ones: order changes, nothing is lost; the server may take the head meanwhile, which is fine)."""
        if self.q is None:
            out = []
            for f in sorted(os.listdir(self.dir)):
                if f.endswith('.json'):
                    with open(os.path.join(self.dir, f)) as fh:
                        r = json.load(fh)
                    if match(r['id']):
                        os.unlink(os.path.join(self.dir, f))
                        out.append(r)
            return out
        if not any(match(r['id']) for r in self.peek()):
            return []
        items = self.q.get_many(1000, block=False, partition=self.cluster)
        keep = [r for r in items if not match(r['id'])]
        if keep:
            self.q.put_many(keep, partition=self.cluster)
        return [r for r in items if match(r['id'])]


def safe(fn, *a, default=None, **kw):
    """Dict calls retried briefly: a transient control-plane error must not kill the supervisor."""
    for i in range(3):
        try:
            return fn(*a, **kw)
        except Exception as e:  # noqa: BLE001
            print(f'[sup] state call failed ({e}); retry {i + 1}', flush=True)
            time.sleep(2)
    return default


# ----------------------------------------------------------------------------------------------------------------------
# Supervisor

def gpu_pids():
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:  # noqa: BLE001
        return []


class Supervisor:
    def __init__(self):
        self.cluster = os.environ.get('CLUSTER', 'local')
        self.job = os.environ.get('JOB_ID', 'local')
        self.node = int(os.environ.get('NODE_RANK', 0))
        self.nnodes = int(os.environ.get('NNODES', 1))
        self.K = f'dev/{self.cluster}/'
        self.st = make_state()
        self.rq = ReqQueue(self.cluster) if self.node == 0 else None
        self.t_start = time.time()
        self.deadline = self.t_start + float(os.environ.get('DEV_HOURS', 5)) * 3600
        self.gen, self.proc, self.restarts = 0, None, []
        self.current, self.cur_deadline, self.cur_t0 = None, 0, 0
        self.worker_t0 = 0
        self.base_port = int(os.environ.get('DEV_BASE_PORT', 29600))
        self.ready_timeout = float(os.environ.get('DEV_READY_TIMEOUT', 2400))
        for d in (INBOX, OUTBOX, PREP_OUT, JOBS_IN):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        # compile pipeline (see the comment above class Pool)
        self.pipe = pipeline_on()
        self.pool = None
        self.lookahead = max(1, int(os.environ.get('DEV_LOOKAHEAD', 6))) if self.pipe else 1
        self.pending = []           # node 0: requests taken from the queue, in order: {req, state, keys, tries, t_in}
        self.prep = None            # node 0: the outstanding 'prep' message {rid, t0}
        self.expect_restart = False  # the last result announced a planned rank restart: hand nothing over until then
        self.jobs = {}              # node 0: key -> job (key, node, src, flags, tag, digest, state, rids, t, t_done)
        self.hdr_published = set()
        self.last_cancel_poll = 0
        if self.pipe:
            inc = next((os.path.join(p, 'deep_gemm', 'include') for p in os.environ.get('PYTHONPATH', '').split(':')
                        if p and os.path.isdir(os.path.join(p, 'deep_gemm', 'include'))), os.environ.get('DEV_LIVE_INC', ''))
            os.environ['DEV_LIVE_INC'] = inc
            wrapper, real = write_nvcc_wrapper(ROOT)
            os.environ['DG_JIT_NVCC_COMPILER'] = wrapper
            os.environ['DEV_STORE'] = STORE
            self.say(f'compile pipeline ON: wrapper {wrapper} -> {real}, store {STORE}, include {inc}, '
                     f'lookahead {self.lookahead}, pool {os.environ.get("DEV_POOL_JOBS", 16)} jobs/node')
        else:
            self.say('compile pipeline OFF (DEV_PIPELINE=0): requests compile inside the ranks, one request at a time')

    def say(self, msg):
        print(f'[sup node{self.node} {time.strftime("%H:%M:%S")}] {msg}', flush=True)

    # -- compile pipeline, node 0
    def entry(self, rid):
        return next((e for e in self.pending if e['req']['id'] == rid), None)

    def header_set(self, headers):
        """-> (tag, digest) of a request's header set; builds the snapshot and publishes the set for the other nodes."""
        tag = header_tag(headers or {})
        _, digest = self.pool.snaps.get(tag, headers or {})
        if tag and tag not in self.hdr_published and self.nnodes > 1:
            safe(self.st.put, f'{self.K}pipe/hdr/{tag}', headers)
        self.hdr_published.add(tag)
        return tag, digest

    def add_jobs(self, rid, tag, digest, items):
        """Register compile jobs ({src, flags} each) for a request; returns their keys. A job goes to the node with
        the fewest unfinished jobs."""
        keys = []
        for it in items:
            key = job_key(it['src'], it['flags'], digest)
            keys.append(key)
            j = self.jobs.get(key)
            if j is not None and j['state'] != 'abandoned':
                j['rids'].add(rid)
                continue
            load = [sum(1 for x in self.jobs.values() if x['state'] == 'queued' and x['node'] == n) for n in range(self.nnodes)]
            node = min(range(self.nnodes), key=lambda n: (load[n], n))
            self.jobs[key] = dict(key=key, node=node, src=it['src'], flags=it['flags'], tag=tag, digest=digest,
                                  state='queued', rids={rid}, t=time.time(), t_done=None, res=None)
        self.publish_jobs()
        return keys

    def publish_jobs(self):
        now = time.time()
        for k in [k for k, j in self.jobs.items() if j['state'] != 'queued' and now - (j['t_done'] or now) > 1800]:
            del self.jobs[k]
            safe(self.st.pop, f'{self.K}pipe/cubin/{k}', None)
        doc = {'job': self.job, 'time': now, 'drop': [k for k, j in self.jobs.items() if j['state'] == 'abandoned'],
               'jobs': [{f: j[f] for f in ('key', 'node', 'src', 'flags', 'tag', 'digest', 'state')}
                        for j in self.jobs.values() if j['state'] in ('queued', 'ok', 'fail')]}
        self.pool.doc = doc
        if self.nnodes > 1:
            safe(self.st.put, self.K + 'pipe/jobs', doc)

    def poll_jobs(self):
        res, changed = self.pool.results(), False
        for key, j in self.jobs.items():
            if j['state'] == 'queued' and key in res:
                r = res[key]
                j['state'] = 'ok' if r.get('ok') else 'abandoned' if r.get('infra') else 'fail'
                j['res'], j['t_done'], changed = r, time.time(), True
                if j['state'] == 'abandoned':
                    self.say(f'job {key[:10]} abandoned ({r.get("err")}): the ranks compile it themselves')
        if changed:
            self.publish_jobs()

    def jobs_ready(self, e):
        return all(self.jobs.get(k, {}).get('state', 'abandoned') != 'queued' for k in e['keys'])

    def cancel_entry(self, e, why='cancelled by the client (held by the server, not started)'):
        rid = e['req']['id']
        self.pending.remove(e)
        for k in e['keys']:
            j = self.jobs.get(k)
            if j:
                j['rids'].discard(rid)
                if not j['rids'] and j['state'] == 'queued':
                    j['state'], j['t_done'] = 'abandoned', time.time()
        self.publish_jobs()
        safe(self.st.put, f'{self.K}res/{rid}', dict(id=rid, ok=False, cancelled=True, error=why))
        safe(self.st.pop, f'{self.K}cancel/{rid}', None)
        self.say(f'request {rid} cancelled')

    def requeue_pending(self):
        """Stopping: give the requests taken ahead back to the queue head, in order."""
        if self.pending and self.rq is not None:
            reqs = [e['req'] for e in self.pending]
            safe(self.rq.put_front, reqs)
            self.say(f'returned {len(reqs)} held request(s) to the queue: {[r["id"] for r in reqs]}')
            self.pending = []

    def intake(self, ready):
        """Take requests from the queue: up to DEV_LOOKAHEAD with the pipeline, else one when the ranks are free."""
        if len(self.pending) >= self.lookahead or (not self.pipe and (self.current is not None or not ready)):
            return
        req = safe(self.rq.get_nowait) or (safe(self.st.pop, self.K + 'req', None) if not self.pending else None)
        if not req:
            return
        e = dict(req=req, state='new', keys=[], tries=0, t_in=time.time(), t_prep=None, t_ready=None)
        if not self.pipe or is_oracle(req) or req.get('no_ahead'):
            e['state'] = 'noprep'
        self.pending.append(e)
        if self.pipe:
            self.say(f'request {req["id"]} taken from the queue ({len(self.pending)} held)')

    def pipeline_tick(self, now, ready, alive):
        """Node 0, every pass: cancels, prep replies, job requests from a running request, job results."""
        if now - self.last_cancel_poll >= 3 and self.pending:
            self.last_cancel_poll = now
            for e in list(self.pending):
                if safe(self.st.get, f'{self.K}cancel/{e["req"]["id"]}'):
                    if self.prep and self.prep['rid'] == e['req']['id']:
                        continue  # its sources are being captured: cancel on the next pass
                    self.cancel_entry(e)
        if self.prep:
            e, f = self.entry(self.prep['rid']), f'{PREP_OUT}/{self.prep["rid"]}.json'
            if e is None:
                self.prep = None
            elif os.path.exists(f):
                with open(f) as fh:
                    out = json.load(fh)
                os.unlink(f)
                t0p, self.prep = self.prep['t0'], None
                items = [v for v in out.get('variants', []) if 'src' in v]
                if out.get('nocase'):
                    # its token sizes need cases beyond DEV_MAX_CASES: capture when it is the head (the cases are then
                    # created as its own run would, and the ranks restart after it as before)
                    e['state'] = 'wait_head'
                    e['gen_prep'], e['ncases_prep'] = self.gen, out.get('ncases', 0)
                elif out.get('error'):
                    e['state'] = 'noprep'
                    self.say(f'request {e["req"]["id"]}: no compile-ahead ({out.get("error")})')
                else:
                    try:
                        tag, digest = self.header_set(e['req'].get('headers'))
                        e['keys'] = self.add_jobs(e['req']['id'], tag, digest, items)
                        e['state'], e['t_prep'], e['predicted'] = 'prepped', now, out.get('predicted') or []
                        e['gen_prep'] = self.gen   # 'predicted' = live cases + this request's new sizes > DEV_MAX_CASES in THIS gen
                        self.say(f'request {e["req"]["id"]}: sources captured in {now - t0p:.1f} s, '
                                 f'{len(e["keys"])} compile job(s) ({len(out.get("variants", []))} variants)'
                                 + (f'; sources for token sizes {out["predicted"]} predicted (no live case)' if out.get('predicted') else ''))
                    except Exception as ex:  # noqa: BLE001
                        e['state'] = 'noprep'
                        self.say(f'request {e["req"]["id"]}: no compile-ahead ({type(ex).__name__}: {ex})')
            elif now - self.prep['t0'] > 180 or not alive:
                e['tries'] += 1
                e['state'] = 'new' if e['tries'] < 2 else 'noprep'
                self.prep = None
        for f in sorted(os.listdir(JOBS_IN)):
            if not f.endswith('.json'):
                continue
            with open(f'{JOBS_IN}/{f}') as fh:
                jr = json.load(fh)
            os.unlink(f'{JOBS_IN}/{f}')
            ack = {'ok': False}
            warm = str(jr.get('rid', '')).startswith('warmup')  # the ranks' own warmup request: pristine headers
            if warm or (self.current is not None and jr.get('rid') == self.current['id']):
                try:
                    tag, digest = self.header_set({} if warm else self.current.get('headers'))
                    if digest != jr.get('digest'):
                        raise RuntimeError(f'include digest of the ranks ({jr.get("digest")}) differs from the snapshot ({digest})')
                    ack = {'ok': True, 'keys': self.add_jobs(jr['rid'], tag, digest, jr.get('items', []))}
                    ahead = None if warm else getattr(self, 'cur_ahead', None)
                    if ahead is not None:
                        late = [k for k in ack['keys'] if k not in ahead[0]]
                        if isinstance(self.cur_info.get('ahead'), dict):
                            self.cur_info['ahead'].update(predicted_sizes=ahead[1], not_ahead=len(late))
                        if late:
                            self.say(f'request {jr["rid"]}: {len(late)} of {len(ack["keys"])} variants were NOT compiled ahead '
                                     f'(their source differs from the one captured ahead; predicted sizes {ahead[1]})')
                except Exception as ex:  # noqa: BLE001
                    ack = {'ok': False, 'error': f'{type(ex).__name__}: {ex}'}
                    self.say(f'request {jr.get("rid")}: pool refused its jobs ({ack["error"]}); the ranks compile themselves')
            atomic_write_json(f'{JOBS_IN}/{jr.get("rid")}.ack', ack)
        self.poll_jobs()

    def pipe_status(self):
        if not self.pipe:
            return {}
        return dict(pending=[dict(id=e['req']['id'], state=e['state'], jobs=len(e['keys']),
                                  jobs_done=sum(1 for k in e['keys'] if self.jobs.get(k, {}).get('state', 'x') != 'queued'))
                             for e in self.pending],
                    pool=dict(self.pool.stats, queued=sum(1 for j in self.jobs.values() if j['state'] == 'queued')) if self.pool else {})

    def gen_dir(self, g=None):
        return f'{ROOT}/g{self.gen if g is None else g}'

    def start_worker(self, nowarm=False):
        d = self.gen_dir()
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ, DEV_GEN=str(self.gen), DEV_GEN_DIR=d, MASTER_PORT=str(self.base_port + 37 * self.gen),
                   WORLD_SIZE=str(self.nnodes), RANK=str(self.node))
        if nowarm or HDR_FRESH:
            env['DEV_WARM'] = ''  # ORACLE request next: its kernel must make the first launches on fresh buffers
        if HDR_FRESH:
            env['DG_JIT_CACHE_DIR'] = f'{ROOT}/jitgen/{self.gen}'
            shutil.rmtree(f'{ROOT}/jitgen', ignore_errors=True)
            os.makedirs(env['DG_JIT_CACHE_DIR'], exist_ok=True)
            self.live_tag = None  # the header set of this generation: that of the first request handed to it
        cmd = [sys.executable, os.path.abspath(__file__), 'worker']
        if os.environ.get('DEV_FAKE_WORKER'):
            cmd = [sys.executable, os.path.abspath(__file__), 'fakeworker']
        self.logf = open(f'{d}/worker.log', 'w')
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.logf, stderr=subprocess.STDOUT, start_new_session=True)
        self.worker_t0 = time.time()
        self.say(f'worker gen {self.gen} started (pid {self.proc.pid}, port {env["MASTER_PORT"]})')

    def kill_worker(self, why):
        """SIGKILL the worker's whole process group (spawned ranks, nvcc), even if the spawn parent already exited."""
        if self.proc is None:
            return
        self.say(f'stopping worker gen {self.gen}: {why}')
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.say('worker did not exit within 60 s of SIGKILL')
        self.proc = None
        for f in ('quiet', 'quiet.ack'):  # dead ranks cannot end their quiet window: let the pool's compilers run again
            try:
                os.unlink(f'{ROOT}/{f}')
            except OSError:
                pass
        for d in (INBOX, PREP_OUT, JOBS_IN):
            for f in os.listdir(d):
                if f.startswith('prep-') or d != INBOX:
                    try:
                        os.unlink(f'{d}/{f}')
                    except OSError:
                        pass
        if self.prep:  # its capture died with the ranks: ask again once they are back
            e = self.entry(self.prep['rid'])
            if e:
                e['tries'] += 1
                e['state'] = 'new' if e['tries'] < 2 else 'noprep'
            self.prep = None
        self.expect_restart = False

    def wait_gpus_free(self):
        # GPU contexts are released when the processes are gone; wait so the next worker gets clean GPUs
        if os.environ.get('DEV_FAKE_WORKER'):
            return
        t = time.time()
        while gpu_pids() and time.time() - t < 90:
            time.sleep(2)
        left = gpu_pids()
        if left:
            self.say(f'GPU processes still present after 90 s: {left}')

    def node_tail(self, g):
        d = self.gen_dir(g)
        out = {'worker': log_tail(f'{d}/worker.log', 25)}
        alerts = []
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.startswith('rank') and f.endswith('.log'):
                out[f[:-4]] = log_tail(f'{d}/{f}', 12)
                alerts += [f'{f[:-4]}: {l}' for l in log_tail(f'{d}/{f}', 40, ALERT_RE) if 'STAGE' not in l]
                out.setdefault('stage', []).extend(f'{f[:-4]}: {l}' for l in log_tail(f'{d}/{f}', 300, STAGE_RE))
            if f.startswith('crash'):
                out[f] = open(f'{d}/{f}', errors='replace').read()[-3000:].splitlines()
        out['alerts'] = alerts[-400:]
        try:  # the fault address of a device fault (Xid 31: "MMU Fault ... at address"), where the container may read the kernel log
            p = subprocess.run('dmesg 2>/dev/null | grep -iE "Xid|NVRM" | tail -20', shell=True, capture_output=True, text=True, timeout=5)
            out['xid'] = p.stdout.splitlines()
        except Exception:  # noqa: BLE001
            out['xid'] = []
        return out

    def set_status(self, state, **kw):
        prog = {}
        try:
            with open(PROGRESS) as f:
                prog = json.load(f)
        except (OSError, ValueError):
            pass
        safe(self.st.put, self.K + 'status', dict(
            job=self.job, gen=self.gen, state=state, time=time.time(), restarts=len(self.restarts),
            queued=(safe(self.rq.len) or 0) + len(self.pending) if self.rq else None,
            label=os.environ.get('DEV_LABEL', '').replace('_', ' '), pipeline=self.pipe,
            request=self.current['id'] if self.current else None, progress=prog, **dict(self.pipe_status(), **kw)))

    def planned_exit(self):
        """The spawn parent exits non-zero whenever a rank exits; a planned exit leaves planned_r* markers."""
        d = self.gen_dir()
        return any(f.startswith('planned') for f in os.listdir(d)) if os.path.isdir(d) else False

    def bump_gen(self, why, nowarm=False):
        cur = safe(self.st.get, self.K + 'gen') or {}
        new = max(self.gen + 1, cur.get('gen', 0) if cur.get('job') == self.job else 0)
        safe(self.st.put, self.K + 'gen', {'job': self.job, 'gen': new, 'why': f'node{self.node}: {why}', 'nowarm': nowarm})
        return new

    def fail_current(self, why, g):
        """Node 0: post a failure result for the in-flight request with both nodes' log tails."""
        if self.current is None:
            return
        tails = {f'node{self.node}': self.node_tail(g)}
        t = time.time()
        while self.nnodes > 1 and time.time() - t < 15:
            others = {n: safe(self.st.get, f'{self.K}tail/{self.job}/{g}/node{n}') for n in range(1, self.nnodes)}
            if all(others.values()):
                break
            time.sleep(1)
        for n in range(1, self.nnodes):
            tails[f'node{n}'] = safe(self.st.get, f'{self.K}tail/{self.job}/{g}/node{n}') or 'not received'
        prog = {}
        try:
            with open(PROGRESS) as f:
                prog = json.load(f)
        except (OSError, ValueError):
            pass
        res = dict(id=self.current['id'], ok=False, error=why, gen=g, server_job=self.job, progress=prog,
                   elapsed_s=round(time.time() - self.cur_t0, 1), logs=tails)
        safe(self.st.put, f'{self.K}res/{self.current["id"]}', res)
        self.say(f'request {self.current["id"]} FAILED: {why}')
        self.current = None

    def restart(self, why, nowarm=False, planned=False):
        g = self.gen
        self.kill_worker(why)
        if self.node != 0:
            safe(self.st.put, f'{self.K}tail/{self.job}/{g}/node{self.node}', self.node_tail(g))
        elif not planned:
            self.fail_current(why, g)
        self.wait_gpus_free()
        if not planned:
            self.restarts = [t for t in self.restarts if time.time() - t < 1200] + [time.time()]
        if len(self.restarts) > int(os.environ.get('DEV_MAX_RESTARTS', 20)):
            self.say('too many restarts in 20 min: giving up')
            if self.node == 0:
                self.set_status('dead', why=why)
            return False
        cur = safe(self.st.get, self.K + 'gen') or {}
        target = cur.get('gen', 0) if cur.get('job') == self.job else 0
        self.gen = max(self.gen + 1, target)
        self.start_worker(nowarm=nowarm)
        return True

    def collect_results(self, now):
        """Node 0: publish the results the ranks left in the outbox."""
        for f in sorted(os.listdir(OUTBOX)):
            if not f.endswith('.json'):
                continue
            with open(f'{OUTBOX}/{f}') as fh:
                res = json.load(fh)
            os.unlink(f'{OUTBOX}/{f}')
            mine = self.current is not None and res.get('id') == self.current['id']
            res['elapsed_s'] = round(now - self.cur_t0, 1) if mine else None
            if mine:
                res.update(self.cur_info)
                self.current = None
            if res.get('restarting'):
                # the ranks leave for a planned restart right after this result: hand them nothing until they are back
                self.expect_restart = now
            safe(self.st.put, f'{self.K}res/{res["id"]}', res)
            self.say(f'result {res["id"]}: ok={res.get("ok")} {res.get("summary", "")}')
            if res['id'].startswith('warmup'):
                safe(self.st.put, self.K + 'warmup', res)

    def dispatch(self, e, now):
        req = e['req']
        self.current, self.cur_t0 = req, now
        self.cur_deadline = now + float(req.get('timeout', 900))
        req['accepted'] = now
        jobs = [self.jobs[k] for k in e['keys'] if k in self.jobs]
        self.cur_ahead = (set(e['keys']), e.get('predicted') or []) if e['state'] == 'prepped' else None
        self.cur_info = dict(
            queue_s=round(now - req.get('submitted', now), 1), held_s=round(now - e['t_in'], 1),
            ahead=dict(prepped=e['state'] == 'prepped', jobs=len(e['keys']),
                       compile_s=[j['res'].get('secs') for j in jobs if j.get('res')],
                       nodes=sorted({j['node'] for j in jobs}),
                       states=sorted({j['state'] for j in jobs})) if self.pipe else None)
        req['pipeline'] = self.pipe
        atomic_write_json(f'{INBOX}/{req["id"]}.json', {k: v for k, v in req.items()})
        self.say(f'request {req["id"]} accepted ({len(req.get("arms", []))} arms, tokens {req.get("tokens")}'
                 + (f', {len(e["keys"])} job(s) compiled ahead, held {now - e["t_in"]:.0f} s' if e['state'] == 'prepped' else '') + ')')

    def run(self):
        if self.pipe:
            self.pool = Pool(self)
        try:
            return self.loop()
        finally:
            if self.node == 0:
                self.requeue_pending()
            if self.pool:
                self.pool.shutdown()

    def loop(self):
        K = self.K
        if self.node == 0:
            safe(self.st.put, K + 'gen', {'job': self.job, 'gen': 0, 'why': 'start'})
            safe(self.st.put, K + 'server', dict(
                job=self.job, started=self.t_start, deadline=self.deadline, nodes=self.nnodes,
                base_commit=os.environ.get('DEV_BASE_COMMIT', ''), patch=os.environ.get('PATCH', ''),
                base_knobs=os.environ.get('DEV_BASE_KNOBS', DEFAULT_KNOBS), pipeline=self.pipe, lookahead=self.lookahead,
                static={k: v for k, v in os.environ.items() if k in STATIC_KNOBS}, warm=os.environ.get('DEV_WARM', ''),
                label=os.environ.get('DEV_LABEL', '').replace('_', ' ')))
            if self.pipe:
                self.publish_jobs()
            self.set_status('starting')
        else:
            t = time.time()
            while time.time() - t < 600:
                g = safe(self.st.get, K + 'gen') or {}
                if g.get('job') == self.job:
                    self.gen = g['gen']
                    break
                time.sleep(2)
        self.start_worker()
        last_poll, last_status, ready_seen = 0, 0, False
        stop_sent = False
        self.cur_info = {}
        while True:
            time.sleep(0.5 if not self.pipe else 0.25)
            now = time.time()
            if now > self.deadline:
                self.say('DEV_HOURS reached: stopping')
                self.kill_worker('deadline')
                if self.node == 0:
                    self.fail_current('server reached its deadline', self.gen)
                    self.requeue_pending()
                    self.set_status('stopped', why='deadline')
                return 0
            if now - last_poll >= 2:
                last_poll = now
                stop = safe(self.st.get, K + 'stop')
                if self.node == 0 and not getattr(self, 'draining', False):
                    dr = safe(self.st.get, K + 'drain')
                    if dr and dr.get('time', 0) > self.t_start:
                        self.draining = True
                        self.say('drain requested: no new request is started; stopping when the running one is done')
                if self.node == 0 and getattr(self, 'draining', False) and self.current is None and not stop_sent:
                    stop_sent = True
                    safe(self.st.put, K + 'stop', {'time': time.time(), 'why': 'drain'})
                    stop = {'time': time.time()}
                if stop and stop.get('time', 0) > self.t_start:
                    self.kill_worker('stop requested')
                    if self.node == 0:
                        self.fail_current('server stopped', self.gen)
                        self.requeue_pending()
                        self.set_status('stopped', why='stop requested')
                    self.say('stopped')
                    return 0
                g = safe(self.st.get, K + 'gen') or {}
                if g.get('job') == self.job and g.get('gen', 0) > self.gen:
                    w = str(g.get('why'))
                    if not self.restart(f'gen bump ({w})', nowarm=bool(g.get('nowarm')),
                                        planned='planned' in w or 'ORACLE' in w):
                        return 1
                    ready_seen = False
                    continue
            ready = os.path.exists(f'{self.gen_dir()}/ready')
            if ready and not ready_seen:
                ready_seen = True
                self.say(f'worker gen {self.gen} ready after {now - self.worker_t0:.0f} s')
            rc = self.proc.poll()
            if rc is not None and self.node == 0 and self.current is not None:  # drain on ANY exit (the result may be written)
                # Race (2026-11-06): the ranks write the result to the outbox, then exit with PLANNED_EXIT; collect the
                # result before deciding, or a completed request is reported as 'worker exited with code 1'
                self.collect_results(now)
            if rc is not None and self.current is None and self.planned_exit():
                self.say('planned worker restart (fresh ranks: case cap, ORACLE or a failed check)')
                self.bump_gen('planned restart')
                if not self.restart('planned restart', planned=True):
                    return 1
                ready_seen = False
                continue
            if rc is not None and self.node == 0 and self.current is not None and self.planned_exit():
                # Race (2026-11-12): the previous request's result was collected, the NEXT request was accepted in the
                # same pass, and the ranks then left for their planned restart (case cap / ORACLE) without ever reading
                # it. That used to fail the new request ('worker exited with code 1'). Re-deliver it to the new ranks.
                # (Since the pipeline change a result announces the restart ('restarting') and nothing is handed over
                # until the ranks are back, so this is a backstop.)
                req = self.current
                self.say(f'planned worker restart with request {req["id"]} accepted but not started: re-delivering it')
                self.bump_gen('planned restart')
                if not self.restart('planned restart', planned=True):
                    return 1
                ready_seen = False
                for f in os.listdir(INBOX):
                    if f.startswith(req['id']):
                        os.unlink(f'{INBOX}/{f}')
                now = time.time()
                self.cur_t0 = now
                self.cur_deadline = now + float(req.get('timeout', 900))
                req['accepted'] = now
                atomic_write_json(f'{INBOX}/{req["id"]}.json', req)
                continue
            ready_limit = self.ready_timeout if self.gen == 0 else min(self.ready_timeout, 900)
            if rc is not None or (not ready and now - self.worker_t0 > ready_limit) or \
                    (self.current is not None and now > self.cur_deadline):
                why = (f'worker exited with code {rc}' if rc is not None else
                       'worker not ready in time' if not ready else
                       f'request timed out after {now - self.cur_t0:.0f} s')
                self.bump_gen(why)
                if not self.restart(why):
                    return 1
                ready_seen = False
                continue
            if self.node != 0:
                continue
            # node 0: results, then new requests
            self.collect_results(now)
            self.intake(ready)
            if self.pipe:
                self.pipeline_tick(now, ready, True)
            if self.expect_restart and now - self.expect_restart > 180:
                self.say('the announced planned restart did not happen within 180 s: resuming')
                self.expect_restart = False
            if self.current is None and ready and self.prep is None and not self.expect_restart and self.pending \
                    and not getattr(self, 'draining', False):
                head = self.pending[0]
                if HDR_FRESH:
                    tag = header_tag(head['req'].get('headers') or {})
                    if self.live_tag is not None and tag != self.live_tag:
                        self.say(f'request {head["req"]["id"]}: header set {tag or "pristine"} differs from the live ranks\' '
                                 f'({self.live_tag or "pristine"}): fresh ranks first')
                        self.bump_gen('planned restart: header set changed, fresh ranks before the request', nowarm=True)
                        if not self.restart('planned restart: header set changed, fresh ranks before the request', nowarm=True, planned=True):
                            return 1
                        ready_seen = False
                        continue
                    if self.live_tag is None and head['state'] == 'new':
                        head['state'] = 'noprep'  # no capture before its headers are installed in this process
                max_wait = float(os.environ.get('DEV_AHEAD_MAX_WAIT', 3600))
                if head['state'] == 'prepped' and head['t_ready'] is None and self.jobs_ready(head):
                    head['t_ready'] = now
                head_go = head['state'] == 'noprep' or (head['state'] == 'prepped' and (head['t_ready'] or now - head['t_prep'] > max_wait))
                if not head_go and os.environ.get('DEV_READY_FIRST', '1') != '0':
                    # The head still waits for its compiles: run a later request whose cubins are all ready instead
                    # of idling (order of results changes, nothing else; the head goes as soon as it is ready).
                    nxt = next((x for x in self.pending[1:] if x['state'] == 'prepped' and self.jobs_ready(x)), None)
                    if nxt is not None:
                        self.pending.remove(nxt)
                        self.pending.insert(0, nxt)
                        head, head_go = nxt, True
                        self.say(f'request {nxt["req"]["id"]} is ready and runs ahead of the waiting head')
                if head_go:
                    self.pending.pop(0)
                    req = head['req']
                    if is_oracle(req):
                        # ORACLE assumes its own kernel made the first launches (epochs 1-2 fill both parities): restart
                        # every node without the warmup so the request runs first on fresh buffers
                        self.say(f'request {req["id"]} has ORACLE arms: fresh ranks without warmup')
                        self.bump_gen('ORACLE request: fresh buffers', nowarm=True)
                        if not self.restart('ORACLE request: fresh buffers', nowarm=True, planned=True):
                            return 1
                        ready_seen = False
                        now = time.time()
                    elif (head['state'] == 'prepped' and head.get('predicted') and head.get('gen_prep') == self.gen
                          and os.environ.get('DEV_FRESH_BEFORE', '1') != '0'):
                        # Its new token sizes do not fit the case budget of the live ranks (that is why their sources
                        # were predicted): timing it now would run 8 x (4 x cases + 1) busy threads per node on 115
                        # CPUs. Fresh ranks first, without the warm case, as for ORACLE.
                        self.say(f'request {req["id"]}: token sizes {head["predicted"]} exceed the case budget of the '
                                 'live ranks: fresh ranks first')
                        self.bump_gen('planned restart: case budget, fresh ranks before the request', nowarm=True)
                        if not self.restart('planned restart: case budget, fresh ranks before the request', nowarm=True, planned=True):
                            return 1
                        ready_seen = False
                        now = time.time()
                    if HDR_FRESH:
                        self.live_tag = header_tag(req.get('headers') or {})
                    self.dispatch(head, now)
                else:
                    if (head['state'] == 'wait_head' and head.get('ncases_prep') and head.get('gen_prep') == self.gen
                            and os.environ.get('DEV_FRESH_BEFORE', '1') != '0'):
                        # over the case budget and its sources could not be predicted: same remedy, fresh ranks first;
                        # the forced prep below then creates its cases on empty ranks
                        self.say(f'request {head["req"]["id"]}: over the case budget of the live ranks: fresh ranks first')
                        self.bump_gen('planned restart: case budget, fresh ranks before the request', nowarm=True)
                        if not self.restart('planned restart: case budget, fresh ranks before the request', nowarm=True, planned=True):
                            return 1
                        ready_seen = False
                        continue
                    e = head if head['state'] == 'wait_head' else next((x for x in self.pending if x['state'] == 'new' and (
                        not HDR_FRESH or header_tag(x['req'].get('headers') or {}) == self.live_tag)), None)
                    if e is not None:
                        r = e['req']
                        msg = dict(kind='prep', id=r['id'], arms=r['arms'], tokens=r.get('tokens', [128, 1024]),
                                   shape=r.get('shape', 'flash'), trace=bool(r.get('trace', False)),
                                   tag=header_tag(r.get('headers') or {}),
                                   # forced (= create the cases whatever the budget) only once the unforced prep said
                                   # 'nocase'; an over-budget head is first predicted, then gets fresh ranks (below / at dispatch)
                                   force=e is head and (e['state'] == 'wait_head' or os.environ.get('DEV_FRESH_BEFORE', '1') == '0'))
                        if 'base' in r:
                            msg['base'] = r['base']
                        atomic_write_json(f'{INBOX}/prep-{r["id"]}.json', msg)
                        e['state'], self.prep = 'prepping', {'rid': r['id'], 't0': now}
            if now - last_status >= 5 or (self.current is not None or self.pending) and now - last_status >= 2:
                last_status = now
                self.set_status('busy' if self.current or self.pending else ('idle' if ready else 'starting'))


# ----------------------------------------------------------------------------------------------------------------------
# Worker ranks

class Rank:
    def __init__(self, local_rank):
        import torch
        import torch.distributed as dist
        self.torch, self.dist = torch, dist
        self.local_rank = local_rank
        self.gen_dir = os.environ['DEV_GEN_DIR']
        self.num_local = int(os.environ.get('GPUS_PER_NODE', 8))
        self.node = int(os.environ.get('RANK', 0))
        self.grank = self.node * self.num_local + local_rank
        # Every rank logs to its own file (device printf included: fds 1 and 2 point at it)
        self.log_path = f'{self.gen_dir}/rank{self.grank}.log'
        self.orig_out = os.dup(1)
        fd = os.open(self.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        sys.stdout = open(1, 'w', buffering=1, closefd=False)
        sys.stderr = sys.stdout
        self.libc = ctypes.CDLL(None)
        self.log_off = 0
        self.t_init = time.time()

        import datetime
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
            world_size=int(os.environ['WORLD_SIZE']) * self.num_local, rank=self.grank,
            timeout=datetime.timedelta(minutes=45), device_id=torch.device(f'cuda:{local_rank}'))
        torch.set_default_device('cuda')
        self.group = dist.new_group(list(range(dist.get_world_size())), timeout=datetime.timedelta(minutes=45))  # default 10 min < a Grace compile
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        # deep_gemm.utils.dist keeps the local rank for dist_print
        import deep_gemm.utils.dist as dgd
        dgd._local_rank = local_rank

        import deep_gemm
        self.dg = deep_gemm
        deep_gemm.use_deterministic_algorithms(True)
        self.include = os.path.join(os.path.dirname(deep_gemm.__file__), 'include')
        self.include_orig = os.environ.get('DEV_INCLUDE_ORIG', '')
        self.static = {k: v for k, v in os.environ.items() if k in STATIC_KNOBS}
        self.tol = float(os.environ.get('DG_MEGA_MOE_IB_TOLERANCE', '1e-3'))
        self.nvl_ref = os.environ.get('DG_MEGA_MOE_IB_NVL_REF', '0') == '1'
        self.legacy = os.environ.get('NO_LEGACY', '0') != '1'
        self.deep_ep = self.tilelang_ops = None
        if self.legacy:
            import importlib.util
            import deep_ep
            self.deep_ep = deep_ep
            spec = importlib.util.spec_from_file_location(
                'tilelang_ops', os.path.join(os.environ['DEV_DG_SRC'], 'third-party', 'tilelang_ops', '__init__.py'))
            self.tilelang_ops = importlib.util.module_from_spec(spec)
            sys.modules['tilelang_ops'] = self.tilelang_ops
            spec.loader.exec_module(self.tilelang_ops)
        self.alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(self.alignment)
        self.cases = {}
        self.cur_headers = {}
        self.pipe = pipeline_on() and bool(os.environ.get('DG_JIT_NVCC_COMPILER'))
        try:
            with open('/opt/devserver/py.sum') as f:
                self.build_id = f.read().strip()
        except OSError:
            self.build_id = os.environ.get('PATCH', '')
        self.live_digest = None
        self.quiet_nonce = None
        self.say(f'rank {self.rank}/{self.world} up (gen {os.environ.get("DEV_GEN")}, legacy={self.legacy}, nvl_ref={self.nvl_ref})')

    # -- utilities
    def say(self, msg, console=False):
        line = f'[r{self.rank} {time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        if console:
            os.write(self.orig_out, (line + '\n').encode())

    def progress(self, rid, phase, detail=''):
        if self.rank == 0:
            atomic_write_json(PROGRESS, {'id': rid, 'phase': phase, 'detail': detail, 'time': time.time()})

    def gather(self, obj):
        out = [None] * self.world
        self.dist.all_gather_object(out, obj, group=self.group)
        return out

    def bcast(self, obj):
        o = [obj]
        self.dist.broadcast_object_list(o, src=0, group=self.group)
        return o[0]

    def barrier(self):
        self.dist.barrier(group=self.group)

    def new_log_lines(self):
        self.libc.fflush(None)
        sys.stdout.flush()
        with open(self.log_path, errors='replace') as f:
            f.seek(self.log_off)
            data = f.read()
            self.log_off = f.tell()
        return data.splitlines()

    def apply_env(self, env):
        for k in list(os.environ):
            if k.startswith(KP) and k not in STATIC_KNOBS:
                del os.environ[k]
        os.environ.update(env)

    # -- headers
    def apply_headers(self, headers):
        """Local rank 0 restores the pristine include tree, then writes the request's headers (relpath under
        deep_gemm/include). Every rank then waits; JIT keys get DG_MEGA_MOE_DEV_HDR so a changed header recompiles
        (deep_jit caches include hashes per process)."""
        err = None
        if self.local_rank == 0:
            try:
                restore = set(self.cur_headers) | set(headers)
                for rel in restore:
                    dst = os.path.join(self.include, rel)
                    src = os.path.join(self.include_orig, rel)
                    if os.path.exists(src):
                        shutil.copyfile(src, dst)
                    elif os.path.exists(dst) and rel not in headers:
                        os.unlink(dst)
                for rel, text in headers.items():
                    dst = os.path.normpath(os.path.join(self.include, rel))
                    if not dst.startswith(self.include + '/') or not rel.startswith('deep_gemm/'):
                        raise ValueError(f'header path {rel!r} must be under deep_gemm/include/deep_gemm/')
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with open(dst, 'w') as f:
                        f.write(text)
            except Exception as e:  # noqa: BLE001
                err = f'{type(e).__name__}: {e}'
        self.cur_headers = dict(headers)
        errs = [e for e in self.gather(err) if e]
        if self.pipe and not errs:
            self.live_digest = tree_digest(self.include)  # after the gather: local rank 0 has written the files
        return errs[0] if errs else None

    # -- compile pipeline (rank side)
    def arm_envs(self, req, res=None):
        """Knob env per arm (errors recorded in res['arms']) and the unique (env, size) variants, in a fixed order."""
        arms = req['arms']
        base = req.get('base', os.environ.get('DEV_BASE_KNOBS', DEFAULT_KNOBS))
        trace = bool(req.get('trace', False))
        sizes = [int(x) for x in req.get('tokens', [128, 1024])]
        envs = {}
        for name, knobs in arms:
            try:
                env, notes = arm_env(base, knobs + (' DG_MEGA_MOE_IB_TRACE=1' if trace else ''), self.static)
                envs[name] = env
                if res is not None:
                    res['arms'][name] = {'knobs': knobs, 'env': env, 'notes': notes, 'sizes': {}}
            except ValueError as e:
                if res is not None:
                    res['arms'][name] = {'knobs': knobs, 'error': str(e), 'sizes': {}}
        tag = req['tag'] if 'tag' in req else header_tag(req.get('headers') or {})
        for env in envs.values():
            if tag:
                env['DG_MEGA_MOE_DEV_HDR'] = tag
        variants, vindex = [], {}
        for name, env in envs.items():
            for s in sizes:
                k = (tuple(sorted(env.items())), s)
                if k not in vindex:
                    vindex[k] = len(variants)
                    variants.append((env, s))
        return envs, sizes, tag, variants, vindex

    # The instantiation part of kernel.cu (everything from SRC_MARK on) depends only on the build, the allocation-time
    # knobs, the world size and (shape, tokens): the host's heuristics read no knob. Rank 0 remembers it per
    # (shape, tokens) in the store, so a later request for a token size whose case does not exist (and would exceed
    # DEV_MAX_CASES) can still be compiled ahead: knob #defines captured with any live case + the remembered
    # instantiation. A wrong prediction costs a wasted compile only: the request's own capture decides what runs.
    def tmpl_path(self, shape, tokens, env=None):
        host = sorted((k, v) for k, v in (env or {}).items() if k.startswith('DG_MEGA_MOE_HOST_') or k == 'DG_MEGA_MOE_DEV_BLOCK_M')  # hostcfg overrides, tile sweep
        ident = sha256_hex(self.build_id, json.dumps(sorted(self.static.items())), self.world,
                           json.dumps(sorted(shape.items())), tokens, *([json.dumps(host)] if host else []))[:32]
        return f'{STORE}/tmpl/{ident}'

    def learn_template(self, case, src, env=None):
        if SRC_MARK not in src:
            return
        p, inst = self.tmpl_path(case.shape, case.T, env), src[src.index(SRC_MARK):]
        try:
            with open(p) as f:
                if f.read() == inst:
                    return
        except OSError:
            pass
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(f'{p}.tmp{os.getpid()}', 'w') as f:
            f.write(inst)
        os.replace(f'{p}.tmp{os.getpid()}', p)

    def template(self, shape, tokens, env=None):
        try:
            with open(self.tmpl_path(shape, tokens, env)) as f:
                return f.read()
        except OSError:
            return None

    def phantom_case(self, shape, tokens):
        """Rank 0 only, no collective, no NCCL allocation: a Case good for capture() and nothing else, for a
        (shape, tokens) that has no live case. The host builds kernel.cu from the buffer's sizes (its own slicing of a
        buffer of the size it asks for), the peer pointer list, the weights' shapes and y's token count; it never
        needs the memory to be symmetric unless it launches. So: a plain local scratch tensor of the required size as
        the `base` of a SymmBuffer (same views as a real one), the handle of any live case, the weights of a live
        case of the same shape (or throw-away ones). Freed by the caller; never launched (compile_only captures only).
        The request's own capture on its real case still decides what runs: a wrong phantom costs a wasted compile."""
        import types
        import deep_gemm.mega as mega
        torch = self.torch
        live = next(iter(self.cases.values()))
        c = Case.__new__(Case)
        c.R, c.torch, c.dg = self, torch, self.dg
        c.H, c.I, c.E, c.K, c.S = shape['hidden'], shape['inter'], shape['experts'], shape['topk'], shape['shared']
        c.T, c.shape, c.burned, c.phantom = tokens, dict(shape), True, True
        c.epr = c.E // self.world
        ta = mega._C.get_token_alignment_for_mega_moe()
        num_bytes, _ = mega._C.get_symm_buffer_size_for_mega_moe(
            self.world, c.E, (tokens + ta - 1) // ta * ta, c.K, c.H, c.I, 'fp8xfp4', 'swiglu', c.S)
        scratch = types.SimpleNamespace(buffer=torch.empty(int(num_bytes), dtype=torch.int8, device='cuda'),
                                        handle=live.buffer.handle, group=self.group)
        c.buffer = mega.SymmBuffer(self.group, c.E, tokens, c.K, c.H, c.I, num_shared_experts=c.S, mma_type='fp8xfp4',
                                   base=scratch)
        same = next((x for x in self.cases.values() if x.shape == c.shape), None)
        if same is not None:
            c.tl1, c.tl2, c.tsl1, c.tsl2 = same.tl1, same.tl2, same.tsl1, same.tsl2
        else:
            cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            try:
                c.make_weights()   # shapes matter, values do not; the RNG streams of the real cases stay untouched
            finally:
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(gpu_rng)
            c.l1 = c.l2 = c.sl1 = c.sl2 = None
        stats = torch.zeros((c.epr,), dtype=torch.int, device='cuda')
        c.timing = {'stats': stats, 'stats0': stats}
        return c

    def capture_predicted(self, surrogate, env, inst):
        """capture() for a (shape, tokens) without a live case: the knob #defines from `surrogate`, then `inst`."""
        c = self.capture(surrogate, env, learn=False)
        try:
            if 'hit' in c:
                src, flags = entry_src_flags(c['hit'], self.include)
            elif 'src' in c:
                src, flags = c['src'], c['flags']
            else:
                return c
            return {'src': src[:src.index(SRC_MARK)] + inst, 'flags': flags, 'predicted': True}
        except Exception as e:  # noqa: BLE001
            return {'skip': f'{type(e).__name__}: {e}'}

    def capture(self, case, env, learn=True):
        """The exact JIT source + nvcc flags of one variant, without compiling: {'src', 'flags'}; {'hit': path} when
        the JIT cache already has it; {'err': text} when the host refuses the variant."""
        try:
            c = self.capture1(case, env)
            if learn and self.pipe:
                if 'src' in c:
                    self.learn_template(case, c['src'], env)
                elif 'hit' in c:
                    with open(f'{c["hit"]}/kernel.cu') as f:
                        self.learn_template(case, f.read(), env)
            return c
        except Exception as e:  # noqa: BLE001
            return {'err': f'capture failed: {type(e).__name__}: {e}'}

    def capture1(self, case, env):
        self.apply_env(env)
        out, cap = f'{self.gen_dir}/cap_out_r{self.grank}.txt', f'{self.gen_dir}/cap_r{self.grank}.json'
        for f in (out, cap):
            if os.path.exists(f):
                os.unlink(f)
        os.environ.update(DG_DEV_COMPILE_ONLY=out, DEV_NVCC_CAPTURE=cap, DEV_INC_DIGEST='capture', DEV_LIVE_INC=self.include)
        try:
            case.launch(case.timing, compile_only=True)
            with open(out) as f:
                return {'hit': f.read().strip()}
        except Exception as e:  # noqa: BLE001
            if os.path.exists(cap):
                with open(cap) as f:
                    return json.load(f)
            return {'err': str(e)[-6000:]}
        finally:
            for k in ('DG_DEV_COMPILE_ONLY', 'DEV_NVCC_CAPTURE', 'DEV_INC_DIGEST'):
                os.environ.pop(k, None)

    def run_prep(self, msg):
        """Compile-ahead step for a request still waiting its turn: rank 0 captures the sources of its variants (cases
        are created only while that stays within DEV_MAX_CASES) and leaves them for the supervisor's pool."""
        out = {'id': msg['id']}
        try:
            envs, sizes, tag, variants, _ = self.arm_envs(msg)
            shape = dict(SHAPES[msg.get('shape', 'flash')]) if isinstance(msg.get('shape', 'flash'), str) else dict(msg['shape'])
            new = sorted({s for s in sizes if (tuple(sorted(shape.items())), s) not in self.cases})
            # Rank 0 decides (its store holds the templates), every rank follows: 'create' the cases (collective),
            # 'predict' the sources of the sizes without a case, or 'nocase' (captured when the request is the head)
            mode = None
            if self.rank == 0:
                if not new or msg.get('force') or len(self.cases) + len(new) <= int(os.environ.get('DEV_MAX_CASES', 3)):
                    mode = 'create'
                elif self.cases and os.environ.get('DEV_PREDICT', '1') != '0':
                    mode = 'predict'
                else:
                    mode = 'nocase'
            mode = self.bcast(mode)
            out['ncases'], out['nnew'] = len(self.cases), len(new)   # for the supervisor's fresh-ranks-first decision
            if mode == 'nocase':
                out['nocase'] = True
            elif mode == 'create':
                cases = {s: self.case(shape, s) for s in sizes}
                if self.rank == 0:
                    out['variants'] = [self.capture(cases[s], env) for env, s in variants]
            elif self.rank == 0:
                # Sizes without a live case: capture on a phantom case (no collective: the other ranks do nothing);
                # if that is impossible, the remembered instantiation of that (shape, tokens); else wait for the head
                skey = tuple(sorted(shape.items()))
                have, why = {s: self.cases.get((skey, s)) for s in sizes}, {}
                for s in new:
                    try:
                        have[s] = self.phantom_case(shape, s)
                    except Exception as e:  # noqa: BLE001
                        why[s] = f'{type(e).__name__}: {e}'[-300:]
                        self.say(f'phantom case {s}: {traceback.format_exc()}')
                sur, vs = next(iter(self.cases.values())), []
                for env, s in variants:
                    if have[s] is not None:
                        c = self.capture(have[s], env, learn=not getattr(have[s], 'phantom', False))
                        if getattr(have[s], 'phantom', False) and 'src' in c:
                            c['predicted'] = True
                    elif self.template(shape, s, env):
                        c = self.capture_predicted(sur, env, self.template(shape, s, env))
                    else:
                        c = None
                    vs.append(c)
                for s in new:
                    have[s] = None
                if any(c is None for c in vs):
                    out['nocase'], out['why'] = True, why
                else:
                    out['variants'], out['predicted'] = vs, new
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            self.say(tb)
            if any(x in tb for x in ('CUDA', 'cuda', 'NCCL', 'illegal', 'trap', 'unspecified launch')):
                os._exit(17)
            out['error'] = f'{type(e).__name__}: {e}'[-2000:]
        self.apply_env(self.static_env())
        if self.rank == 0:
            atomic_write_json(f'{PREP_OUT}/{msg["id"]}.json', out)
            self.say(f'prep {msg["id"]}: {"no case" if out.get("nocase") else out.get("error") or str(len(out.get("variants", []))) + " variants"}')

    def quiet_on(self):
        """Local rank 0: stop this node's background compilers for the check + timing phases (pool SIGSTOPs them and
        acknowledges). Returns a note when the pool did not acknowledge."""
        if not self.pipe or self.local_rank != 0:
            return None
        self.quiet_nonce = f'{os.getpid()}-{time.time_ns()}'
        with open(f'{ROOT}/quiet.tmp', 'w') as f:
            f.write(self.quiet_nonce)
        os.replace(f'{ROOT}/quiet.tmp', f'{ROOT}/quiet')
        t = time.time()
        while time.time() - t < 30:
            try:
                with open(f'{ROOT}/quiet.ack') as f:
                    if f.read().strip() == self.quiet_nonce:
                        return None
            except OSError:
                pass
            time.sleep(0.02)
        return f'node {self.node}: the compile pool did not acknowledge the quiet window (background compiles may have run during timing)'

    def read_pin(self):
        """Local rank 0: the pool's gin_partition summary for this quiet window (None elsewhere / when absent)."""
        if not self.pipe or self.local_rank != 0 or self.quiet_nonce is None:
            return None
        try:
            with open(f'{ROOT}/pin.json') as f:
                p = json.load(f)
            return dict({k: v for k, v in p.items() if k != 'nonce'}, node=self.node) if p.get('nonce') == self.quiet_nonce else None
        except (OSError, ValueError):
            return None

    def quiet_off(self):
        if self.quiet_nonce is not None:
            self.quiet_nonce = None
            try:
                os.unlink(f'{ROOT}/quiet')
            except OSError:
                pass

    # -- cases
    def case(self, shape, tokens):
        key = (tuple(sorted(shape.items())), tokens)
        if key not in self.cases:
            t = time.time()
            before = set(os.listdir('/proc/self/task'))
            self.cases[key] = Case(self, shape, tokens)
            self.cases[key].tids = sorted(int(x) for x in set(os.listdir('/proc/self/task')) - before)
            # Every case allocates its own IB buffer = its own NCCL communicator with busy-polling CPU-proxy threads
            # (~2 per case per rank); with 6 cases a 64-CPU node ran ~96 spinning threads (load 120) and the GIN proxy,
            # time-sliced, turned into ms-scale launch stalls (b300_2n_a, 2026-11-06 04:2x). Restart (fresh processes, no
            # leftover proxies) after any request that leaves more than DEV_MAX_CASES cases alive.
            if len(self.cases) > int(os.environ.get('DEV_MAX_CASES', 3)):
                self.restart_after = True
            self.say(f'case {shape} x {tokens}: {time.time() - t:.1f} s')
        return self.cases[key]

    # -- request
    def handle(self, req):
        t_req = time.time()
        rid = req['id']
        res = dict(id=rid, ok=True, server_job=os.environ.get('JOB_ID'), gen=int(os.environ.get('DEV_GEN', 0)),
                   label=os.environ.get('DEV_LABEL', '').replace('_', ' '), cluster=os.environ.get('CLUSTER', ''),
                   notes=[], arms={}, times={})
        shape = dict(SHAPES[req.get('shape', 'flash')]) if isinstance(req.get('shape', 'flash'), str) else dict(req['shape'])
        sizes = [int(x) for x in req.get('tokens', [128, 1024])]
        arms = req['arms']  # [[name, knobs], ...]
        base = req.get('base', os.environ.get('DEV_BASE_KNOBS', DEFAULT_KNOBS))
        n_check = int(req.get('check', 2))
        launches, warmup, rounds = int(req.get('launches', 30)), int(req.get('warmup', 10)), int(req.get('rounds', 3))
        trace = bool(req.get('trace', False))
        compile_only = bool(req.get('compile_only', False))
        if compile_only:
            n_check, rounds = 0, 0
        res.update(shape=shape, tokens=sizes, base=base, launches=launches, warmup=warmup, rounds=rounds, check=n_check,
                   trace=trace, arm_order=[a[0] for a in arms], compile_only=compile_only, pipeline=self.pipe)

        # Arm envs
        headers = req.get('headers') or {}
        envs, sizes, tag, variants, vindex = self.arm_envs(req, res)
        oracle = {n for n, e in envs.items() if e.get('DG_MEGA_MOE_IB_ORACLE', '0') not in ('', '0')}
        if oracle and len(oracle) != len(envs):
            res.update(ok=False, error=f'ORACLE arms {sorted(oracle)} must be alone in a request (they leave the buffers '
                                       'un-re-armed; the server restarts its ranks after an ORACLE request)')
            return res
        self.restart_after = bool(oracle)
        if oracle:
            res['notes'].append('ORACLE request: the ranks restart afterwards for fresh buffers (~1 min); checks are '
                                'meaningless under ORACLE (identical-input replay), use --check 0')
        res['header_tag'], res['headers'] = tag, sorted(headers)
        for n, e in envs.items():
            if e.get(SCREEN_KNOB) in ('1', '2'):
                res['arms'][n]['notes'].append(
                    f'SCREENING arm: compiled with ptxas -O{e[SCREEN_KNOB]} (not the production -O3)' if self.pipe else
                    f'{SCREEN_KNOB} has no effect on this server (DEV_PIPELINE=0): compiled at -O3')
        if any(rel.startswith(HOST_INCLUDED) for rel in headers):
            res['notes'].append('a header under ' + '/'.join(HOST_INCLUDED) + ' changed: the host extension keeps its '
                                'build-time copy (buffer sizes / layout constants there need a new patch)')

        # 1. headers
        t = time.time()
        self.progress(rid, 'headers')
        err = self.apply_headers(headers)
        res['times']['headers'] = round(time.time() - t, 2)
        if err:
            res.update(ok=False, error=f'header install failed: {err}')
            return res

        # 2. cases (collective allocations; same order everywhere)
        t = time.time()
        self.progress(rid, 'cases')
        cases = {s: self.case(shape, s) for s in sizes}
        if len(self.cases) > int(os.environ.get('DEV_MAX_CASES', 3)):
            self.restart_after = True  # also when a compile-ahead 'prep' created this request's cases
        res['times']['cases'] = round(time.time() - t, 2)

        # 3. compile: unique (env, size) variants.
        #    Pipeline: global rank 0 captures every variant's exact source, the supervisors' pools compile each ONCE
        #    in the cluster (usually already done ahead, while earlier requests ran) and install the cubins into every
        #    node's store; the JIT compiles below are then store hits through the nvcc wrapper. Whatever the pool
        #    does not deliver, the wrapper compiles in place, as the old path did.
        #    Old path (DEV_PIPELINE=0): every node's local ranks run nvcc for the same set of variants.
        t = time.time()
        self.progress(rid, 'compile', f'{len(variants)} variants')
        plan = None
        if self.pipe:
            if self.rank == 0:
                caps = [self.capture(cases[s], env) for env, s in variants]
                self.apply_env(self.static_env())
                items = [c for c in caps if 'src' in c]
                plan = {'keys': [job_key(c['src'], c['flags'], self.live_digest) if 'src' in c else None for c in caps],
                        'src_sha': [sha256_hex(c['src']) if 'src' in c else None for c in caps], 'ack': None}
                if items:
                    atomic_write_json(f'{JOBS_IN}/{rid}.json', {'rid': rid, 'digest': self.live_digest, 'items': items})
                    ta = time.time()
                    while time.time() - ta < 30 and not os.path.exists(f'{JOBS_IN}/{rid}.ack'):
                        time.sleep(0.1)
                    try:
                        with open(f'{JOBS_IN}/{rid}.ack') as f:
                            plan['ack'] = json.load(f)
                        os.unlink(f'{JOBS_IN}/{rid}.ack')
                    except (OSError, ValueError):
                        plan['ack'] = {'ok': False, 'error': 'no answer from the supervisor'}
                    if not plan['ack'].get('ok'):
                        res['notes'].append(f'compile pool not used ({plan["ack"].get("error")}): compiled inside the ranks')
            plan = self.bcast(plan)
            res['times']['capture'] = round(time.time() - t, 2)
            # Sleep until this node's store has every object (or the pool no longer has the job)
            tw = time.time()
            keys = [k for k in plan['keys'] if k]
            self.progress(rid, 'compile', f'{len(variants)} variants, waiting for {sum(1 for k in keys if not obj_state(k))} of {len(keys)} pool jobs')
            while (plan['ack'] or {}).get('ok'):
                missing = [k for k in keys if not obj_state(k)]
                if not missing:
                    break
                if time.time() - tw > 45 and not (pool_alive(ROOT) and any(os.path.exists(f'{ROOT}/pend/{k}') for k in missing)):
                    break
                time.sleep(0.25)
            res['times']['compile_wait'] = round(time.time() - tw, 2)
            os.environ.update(DEV_INC_DIGEST=self.live_digest, DEV_LIVE_INC=self.include)
        ti = time.time()
        mine = {}
        for i in range(self.local_rank, len(variants), self.num_local):
            env, s = variants[i]
            self.apply_env(env)
            out = f'{self.gen_dir}/compile_r{self.grank}.txt'
            os.environ['DG_DEV_COMPILE_ONLY'] = out
            tc = time.time()
            try:
                cases[s].launch(cases[s].timing, compile_only=True)
                mine[i] = {'ok': True, 's': round(time.time() - tc, 1), 'path': open(out).read().strip(), 'node': self.node}
                if self.pipe:
                    path = mine[i]['path']
                    with open(f'{path}/kernel.cu') as f:
                        mine[i]['src_sha'] = sha256_hex(f.read())
                    mine[i]['cubin_sha'] = file_sha(f'{path}/kernel.cubin')
                    mine[i]['code_sha'] = cubin_code_sha(f'{path}/kernel.cubin')
                    mine[i]['prov'] = os.path.exists(prov_path(mine[i]['cubin_sha'], self.live_digest, mine[i]['src_sha']))
                    try:   # per-node provenance, reported only when the harness check fails (which node holds what, and how it got it)
                        _src, _flags = entry_src_flags(path, self.include)
                        _key = job_key(_src, _flags, self.live_digest)
                        _m = obj_meta(_key) or {}
                        mine[i]['where'] = (f"node {self.node}: tree {str(self.live_digest)[:8]} src {mine[i]['src_sha'][:8]} code {str(mine[i]['code_sha'])[:8]} "
                                            f"file {mine[i]['cubin_sha'][:8]} entry {os.path.basename(path)[-12:]} store {_key[:8]}:"
                                            f"{_m.get('how', 'absent')}/{_m.get('node', '-')}/{str(_m.get('cubin_sha'))[:8]} prov {int(mine[i]['prov'])}")
                    except Exception as _e:  # noqa: BLE001
                        mine[i]['where'] = f'node {self.node}: {type(_e).__name__}: {_e}'
            except Exception as e:  # noqa: BLE001
                mine[i] = {'ok': False, 's': round(time.time() - tc, 1), 'err': str(e)[-6000:], 'node': self.node}
            finally:
                os.environ.pop('DG_DEV_COMPILE_ONLY', None)
        # The JIT names a cache entry by the code AND by its own per-process hash of every included header, taken the
        # first time that process parsed the file and never refreshed. Ranks that first parsed under other headers (the
        # local rank that compiled the pristine warm-up, mostly) therefore look for a different entry name at their
        # first launch of an arm, miss, and call nvcc. With DEV_INC_DIGEST still set the wrapper serves that miss from
        # the store by content (seconds); without it every node recompiles inside the ranks while all others wait in
        # the first-launch rendezvous: the 8-15 min "stalls". DEV_KEEP_DIGEST=0 restores the old drop;
        # run_request clears the variable when the request ends.
        if os.environ.get('DEV_KEEP_DIGEST', '1') == '0':
            os.environ.pop('DEV_INC_DIGEST', None)
        # Wait for this node's other compiles with a sleeping file barrier BEFORE the collective: ranks idling inside
        # the all-gather busy-spin (~5 cores each), which starved the nvcc / ptxas of the ranks still compiling on a
        # 32-CPU node (b300_4n_a, 2026-11-06: 6 waiting ranks at 481% CPU each, 10-min collective timeout)
        open(f'{self.gen_dir}/cdone_{rid}_{self.local_rank}', 'w').close()
        while not all(os.path.exists(f'{self.gen_dir}/cdone_{rid}_{r}') for r in range(self.num_local)):
            time.sleep(0.5)
        compiled, per_node = {}, {}
        for part in self.gather(mine):
            for i, v in part.items():
                per_node.setdefault(i, []).append(v)
                if i not in compiled or not v['ok']:
                    compiled[i] = v
        if self.pipe:
            # The harness rule: a result comes from exactly the cubin the request's knobs + headers define. Every node
            # must have loaded the same code for a variant, built from the source rank 0 captured.
            res['variants'] = []
            for i, (env, s) in enumerate(variants):
                vs = per_node.get(i, [])
                if not vs or not all(v['ok'] for v in vs):
                    continue
                codes, srcs = {v.get('code_sha') for v in vs}, {v.get('src_sha') for v in vs}
                want = plan['src_sha'][i]
                bad = None
                if len(codes) != 1 or None in codes:
                    bad = f'the nodes hold different kernel code for this variant: {sorted(str(c) for c in codes)}'
                    bad += ' | ' + ' ; '.join(sorted(str(v.get('where', '')) for v in vs))
                elif len(srcs) != 1 or (want is not None and srcs != {want}):
                    bad = f'kernel.cu of the JIT cache entry differs from the source this request defines ({sorted(srcs)} vs {want})'
                    bad += ' | ' + ' ; '.join(sorted(str(v.get('where', '')) for v in vs))
                if bad:
                    compiled[i] = {'ok': False, 's': compiled[i]['s'], 'err': 'HARNESS CHECK FAILED: ' + bad}
                    continue
                key = plan['keys'][i]
                m = (obj_meta(key) or {}) if key else {}
                res['variants'].append(dict(
                    size=s, arms=[n for n, e in envs.items() if e == env], key=key, src_sha=vs[0]['src_sha'][:16],
                    code_sha=vs[0]['code_sha'], cubin_sha=sorted({v['cubin_sha'][:16] for v in vs}),
                    include_digest=self.live_digest, entry=os.path.basename(vs[0]['path']),
                    how=(m.get('how') or 'inline') if key else 'jit-cache', compile_s=m.get('secs') if key else None,
                    compile_node=m.get('node') if key else None, verified=all(v.get('prov') for v in vs)))
                # DEV_REQUIRE_VERIFIED (default 1): a variant no node can prove was built from THIS request's include tree
                # (no provenance record for (cubin, tree digest, source)) is refused. On the clean-ABI host the header
                # tag is not part of kernel.cu, so a header-only change with an unchanged plan is a JIT-cache hit on the
                # cubin of the OLD headers (gb200_32_a, 2026-11-13: a request ran the base cubin, verified = False).
                if not res['variants'][-1]['verified'] and os.environ.get('DEV_REQUIRE_VERIFIED', '1') != '0':
                    compiled[i] = {'ok': False, 's': compiled[i]['s'], 'err':
                                   'HARNESS CHECK FAILED: the cubin this variant would run was not built from this request\'s '
                                   'headers (JIT cache hit on an entry compiled under another include tree: verified = False). '
                                   'On the clean-ABI host give the arm a plan value that changes the source text, or restart '
                                   'the ranks. | ' + ' ; '.join(sorted(str(v.get('where', '')) for v in vs))}
            res['times']['install'] = round(time.time() - ti, 2)
        res['times']['compile'] = round(time.time() - t, 2)
        res['times']['compile_max_variant'] = max([v['s'] for v in compiled.values()], default=0)
        res['compiled_variants'] = len(variants)
        usable = {}
        for name, env in envs.items():
            usable[name] = True
            for s in sizes:
                v = compiled[vindex[(tuple(sorted(env.items())), s)]]
                res['arms'][name]['sizes'].setdefault(str(s), {})['compile_s'] = v['s']
                if not v['ok']:
                    usable[name] = False
                    res['arms'][name]['sizes'][str(s)]['compile_error'] = v['err']
                    res['arms'][name]['error'] = 'compile failed'
        # No background compiler runs on any node from here to the end of the timing (pool: SIGSTOP + acknowledge)
        no_quiet = compile_only or bool(req.get('no_quiet')) or os.environ.get('DEV_QUIET', '1') == '0'  # no_quiet: A/A tests only
        try:
            atomic_write_json(f'{ROOT}/gin_active.{os.getpid()}', sorted({t for c in cases.values() for t in getattr(c, 'tids', [])}))
        except OSError:
            pass
        qn = self.gather(self.quiet_on() if not no_quiet else None)
        res['notes'] += [n for n in qn if n]
        res['quiet'] = self.pipe and not no_quiet and not any(qn)
        pins = self.gather(self.read_pin() if not no_quiet else None)
        res['pin'] = [p for p in pins if p]
        # busy threads per node = ranks x (4 x live cases + 1) (two communicators per case, each a GIN progress thread +
        # a yield spinner; + the main thread): say so when this request is timed with more of them than CPUs
        n_busy = self.num_local * (4 * len(self.cases) + 1)
        if n_busy > len(os.sched_getaffinity(0)) + sum(p.get('reserved', 0) for p in res['pin'][:1]):
            res['notes'].append(f'{len(self.cases)} live cases = ~{n_busy} busy threads per node: oversubscribed; '
                                'stall-prone unless the GIN threads are pinned (see pin)')
        self.barrier()

        # 4. correctness vs the DeepEP reference (fresh activations / routing per test)
        t = time.time()
        traces = {}
        self.new_log_lines()
        if n_check > 0:
            local = {}
            for s in sizes:
                for name, env in envs.items():
                    if not usable[name]:
                        continue
                    self.progress(rid, 'check', f'{name} @{s}')
                    self.apply_env(env)
                    local[(name, s)] = cases[s].check(n_check, base_env=self.static_env())
                    if trace:
                        traces.setdefault((name, s), []).extend(self.new_log_lines())
            allc = self.gather(local)
            for name in envs:
                if not usable[name]:
                    continue
                for s in sizes:
                    per = [p[(name, s)] for p in allc]
                    res['arms'][name]['sizes'][str(s)]['check'] = {
                        'pass': all(p['pass'] for p in per), 'tests': n_check,
                        'max_calc_diff': max(p['max_calc_diff'] for p in per),
                        'max_rel_rms': max(p['max_rel_rms'] for p in per),
                        'nonfinite': sum(p['nonfinite'] for p in per),
                        'stats_mismatch': sum(p['stats_mismatch'] for p in per),
                        'failed_ranks': [p['rank'] for p in per if not p['pass']],
                        'ref_first_s': [min(p.get('ref_first_s', 0.0) for p in per), max(p.get('ref_first_s', 0.0) for p in per)],
                        'error': next((p['error'] for p in per if p.get('error')), None)}
                    fd = [dd for p in per for dd in p.get('detail', [])][:24]
                    if fd:  # only present when a test failed (bounded: <= 2 tests per failing rank, <= 8 rows each)
                        res['arms'][name]['sizes'][str(s)]['check']['fail_detail'] = fd
                    if not res['arms'][name]['sizes'][str(s)]['check']['pass']:
                        res['ok'] = False
            # A failing arm can leave the warm buffers dirty (e.g. un-re-armed partials, 2026-11-06 PCOMB_PARITY): every
            # later arm / request would then read stale data. Same decision on every rank (allc is all-gathered).
            if any(not p[k]['pass'] for p in allc for k in p):
                self.restart_after = True
                res['notes'].append('a check FAILED: the ranks restart after this request for fresh buffers; arms checked '
                                    'or timed after the first failure in this request may be contaminated')
        res['times']['check'] = round(time.time() - t, 2)

        # 5. timing, arms interleaved per round
        t = time.time()
        timed = {}
        burn = int(os.environ.get('DEV_BURNIN', 100)) if not compile_only else 0
        for s in sizes:
            first = next((n for n in envs if usable[n]), None)
            if burn and first is not None and not cases[s].burned:
                # The first launches on fresh buffers / communicators ran at ~5 ms on B300 (warmup, 23:15);
                # burn them in once per case before anything is timed
                self.progress(rid, 'timing', f'{s} tok, burn-in {burn} launches')
                self.apply_env(envs[first])
                tb = time.time()
                d, _ = cases[s].time_block(burn, 0, f'{self.gen_dir}/tr_r{self.grank}.json')
                cases[s].burned = True
                res['notes'].append(f'burn-in @{s}: {burn} launches in {time.time() - tb:.1f} s')
            if getattr(cases[s], 'skew', (0.0, 1.0))[0] and self.grank == 0:
                own = (cases[s].timing['topk_idx'] < cases[s].epr).float().mean().item() * self.world
                res['notes'].append(f'DEV_SKEW_R @{s}: score bias {cases[s].skew[0]:+.4f} on rank 0\'s experts; rank 0 rows / mean = '
                                    f'{cases[s].skew[1]:.3f} calibrated, {own:.3f} in rank 0\'s own timing tokens')
            for rd in range(rounds):
                for name, env in envs.items():
                    if not usable[name]:
                        continue
                    self.progress(rid, 'timing', f'{s} tok, round {rd + 1}/{rounds}, {name}')
                    self.apply_env(env)
                    durs, n_seen = cases[s].time_block(warmup, launches, f'{self.gen_dir}/tr_r{self.grank}.json')
                    timed.setdefault((name, s), {})[rd] = durs
                    if n_seen != warmup + launches:
                        res['notes'].append(f'rank {self.rank}: {name} @{s} round {rd}: {n_seen} kernel events, expected {warmup + launches}')
                    if trace:
                        traces.setdefault((name, s), []).extend(self.new_log_lines())
        res['times']['timing'] = round(time.time() - t, 2)
        if STALL_DIAG and res['times']['timing'] > 120 and os.environ.get('RESULTS_DIR'):
            # keep what this rank did during a long timing phase: its log does not otherwise leave the node
            try:
                dst = f"{os.environ['RESULTS_DIR']}/stall_" + re.sub(r'[^A-Za-z0-9_.-]', '_', str(rid))[:80]
                os.makedirs(dst, exist_ok=True)
                shutil.copyfile(self.log_path, f'{dst}/rank{self.grank}.log')
                res['notes'].append(f'DEV_STALL_DIAG: timing phase took {res["times"]["timing"]} s; rank logs copied to {dst}')
            except Exception as ex:  # noqa: BLE001
                self.say(f'STALL_DIAG: could not copy the rank log ({type(ex).__name__}: {ex})')
        allt = self.gather({f'{k[0]}|{k[1]}': v for k, v in timed.items()})
        alln = self.gather(res['notes'][-5:])
        if trace:
            n_ep = int(req.get('trace_epochs', 2))

            def last_epochs(lines):
                hits = [l for l in lines if TRACE_RE.search(l)]
                eps = sorted({int(m.group(1)) for l in hits for m in [re.search(r'epoch=(\d+)', l)] if m})[-n_ep:]
                return [l for l in hits if (m := re.search(r'epoch=(\d+)', l)) is None or int(m.group(1)) in eps][-400:]
            tr = self.gather({f'{k[0]}|{k[1]}': last_epochs(v) for k, v in traces.items()})
        alerts = self.gather([l for l in self.new_log_lines() if ALERT_RE.search(l)][-40:])
        self.quiet_off()
        self.apply_env(self.static_env())

        if self.rank != 0:
            return None
        res['notes'] = sorted(set(res['notes'] + [n for ns in alln for n in ns]))
        for name in envs:
            if not usable[name]:
                continue
            for s in sizes:
                per_rank = {r: allt[r].get(f'{name}|{s}', {}) for r in range(self.world)}
                res['arms'][name]['sizes'][str(s)]['timing'] = summarize_timing(per_rank)
        if trace:
            res['trace'] = {k: [l for r in range(self.world) for l in tr[r].get(k, [])] for k in
                            sorted({k for part in tr for k in part})}
        res['alerts'] = [f'rank {r}: {l}' for r in range(self.world) for l in alerts[r]][:400]
        res['times']['total_worker'] = round(time.time() - t_req, 2)
        res['summary'] = ' | '.join(
            f'{n}: ' + ' '.join(f'{s}={res["arms"][n]["sizes"].get(str(s), {}).get("timing", {}).get("median", "-")}'
                                for s in sizes) for n in res['arm_order'])
        return res

    def static_env(self):
        return {}

    def restore_all_headers(self):
        """Local rank 0: make the installed include tree identical to the pristine copy (a request that crashed the
        previous worker may have left its edits behind)."""
        n = 0
        if self.local_rank == 0 and self.include_orig:
            for d, _, fs in os.walk(self.include):
                for f in fs:
                    p = os.path.join(d, f)
                    o = os.path.join(self.include_orig, os.path.relpath(p, self.include))
                    if not os.path.exists(o):
                        os.unlink(p)
                        n += 1
            for d, _, fs in os.walk(self.include_orig):
                for f in fs:
                    o = os.path.join(d, f)
                    p = os.path.join(self.include, os.path.relpath(o, self.include_orig))
                    if not os.path.exists(p) or open(p, 'rb').read() != open(o, 'rb').read():
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        shutil.copyfile(o, p)
                        n += 1
            self.say(f'restored {n} include files to the pristine install')
        self.cur_headers = {}
        self.barrier()

    def serve(self):
        self.restore_all_headers()
        warm = os.environ.get('DEV_WARM', 'flash:128,1024')
        if warm:
            shape, toks = warm.split(':')
            wreq = dict(id=f'warmup-g{os.environ.get("DEV_GEN", 0)}-{os.environ.get("JOB_ID", "")}', shape=shape,
                        tokens=[int(x) for x in toks.split(',')], arms=[['default', '']], check=1,
                        launches=int(os.environ.get('DEV_WARM_LAUNCHES', 30)), rounds=1, warmup=10)
            self.run_request(wreq)
        self.barrier()
        if self.local_rank == 0:  # every node's supervisor waits for its own marker (was global rank 0 only: node 1 timed out)
            open(f'{self.gen_dir}/ready', 'w').close()
        self.say('ready', console=True)
        while True:
            msg = None
            if self.rank == 0:
                t = time.time()
                while time.time() - t < 0.5 and msg is None:
                    files = sorted(f for f in os.listdir(INBOX) if f.endswith('.json'))
                    if files:
                        with open(f'{INBOX}/{files[0]}') as f:
                            msg = json.load(f)
                        os.replace(f'{INBOX}/{files[0]}', f'{INBOX}/{files[0]}.taken')
                    else:
                        time.sleep(0.1)
            msg = self.bcast(msg)
            if msg is None:
                continue
            if msg.get('kind') == 'prep':
                self.run_prep(msg)
                continue
            self.run_request(msg)

    def run_request(self, req):
        self.say(f'request {req["id"]}', console=True)
        try:
            res = self.handle(req)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            self.say(tb)
            fatal = any(s in tb for s in ('CUDA', 'cuda', 'NCCL', 'illegal', 'trap', 'unspecified launch'))
            with open(f'{self.gen_dir}/crash_r{self.grank}.txt', 'w') as f:
                f.write(tb[-6000:])
            if fatal:
                # The CUDA context may be poisoned: exit so the supervisor restarts every rank
                os._exit(17)
            res = dict(id=req['id'], ok=False, error=f'{type(e).__name__}: {e}', traceback=tb[-4000:]) if self.rank == 0 else None
        self.quiet_off()
        os.environ.pop('DEV_INC_DIGEST', None)
        self.progress(req['id'], 'done')
        if self.rank == 0 and res is not None:
            if getattr(self, 'restart_after', False):
                res['restarting'] = True  # the supervisor hands nothing to these ranks any more
            atomic_write_json(f'{OUTBOX}/{req["id"]}.json', res)
            self.say(f'result {req["id"]}: {res.get("summary", res.get("error"))}', console=True)
        if getattr(self, 'restart_after', False):
            self.barrier()
            open(f'{self.gen_dir}/planned_r{self.grank}', 'w').close()
            time.sleep(3)  # let the supervisor pick up the result first
            os._exit(PLANNED_EXIT)


_SKEW_CACHE = {}


def skew_bias(torch, E, K, epr, world, say=None):
    """DEV_SKEW_R=R (server env; unset or 1 = no bias, inputs unchanged): a constant added to the routing scores of rank 0's
    experts (the first E / world) so that rank 0 receives ~R x the mean routed rows (load imbalance, max / mean = R). The bias
    is calibrated once per (E, K, world, R) by bisection on a fixed-seed CPU sample (its own generator: the global RNG stream
    and so every other input is untouched), so every rank computes the same value. Returns (bias, achieved max / mean)."""
    try:
        r = float(os.environ.get('DEV_SKEW_R') or 0)
    except ValueError:
        r = 0.0
    if r <= 0 or r == 1.0:
        return 0.0, 1.0
    key = (E, K, epr, world, r)
    if key not in _SKEW_CACHE:
        g = torch.Generator(device='cpu').manual_seed(20261113)
        sc = torch.randn((8192, E), generator=g, dtype=torch.float, device='cpu')

        def ratio(b):
            s2 = sc.clone()
            s2[:, :epr] += b
            idx = torch.topk(s2, K, dim=-1, largest=True, sorted=False).indices
            return (idx < epr).float().mean().item() * world

        lo, hi = (0.0, 4.0) if r > 1 else (-4.0, 0.0)
        for _ in range(26):
            mid = (lo + hi) / 2
            if ratio(mid) < r:
                lo = mid
            else:
                hi = mid
        b = (lo + hi) / 2
        _SKEW_CACHE[key] = (b, ratio(b))
        if say:
            say(f'DEV_SKEW_R={r}: E={E} K={K} world={world}: bias {b:+.4f} on experts [0, {epr}) -> rank 0 rows / mean = {_SKEW_CACHE[key][1]:.3f}')
    return _SKEW_CACHE[key]


class Case:
    """Buffers, weights and fixed timing inputs for one (shape, tokens); mirrors tests/test_mega_moe.py."""

    def __init__(self, R, shape, tokens):
        torch, dg = R.torch, R.dg
        self.R, self.torch, self.dg = R, torch, dg
        self.H, self.I, self.E, self.K, self.S = shape['hidden'], shape['inter'], shape['experts'], shape['topk'], shape['shared']
        self.T = tokens
        self.shape = dict(shape)
        self.burned = False
        self.epr = self.E // R.world
        self.skew = skew_bias(torch, self.E, self.K, self.epr, R.world, getattr(R, 'say', None))  # DEV_SKEW_R: (score bias on rank 0's experts, max / mean)
        self.buffer = dg.get_symm_buffer_for_mega_moe(R.group, self.E, tokens, self.K, self.H, self.I,
                                                      num_shared_experts=self.S, mma_type='fp8xfp4')
        self.nvl_ref_buffer = None
        if R.nvl_ref:
            saved = os.environ.pop('DG_MEGA_MOE_IB_VIRTUAL_LSA', None)
            try:
                self.nvl_ref_buffer = dg.get_symm_buffer_for_mega_moe(R.group, self.E, tokens, self.K, self.H, self.I,
                                                                      num_shared_experts=self.S, mma_type='fp8xfp4')
            finally:
                if saved is not None:
                    os.environ['DG_MEGA_MOE_IB_VIRTUAL_LSA'] = saved
            assert self.nvl_ref_buffer.handle.num_lsa_ranks == R.world
            self.nvl_stats = None
        self.ep_buffer = None
        if R.legacy:
            ep_type = getattr(R.deep_ep, 'EPBuffer', None) or R.deep_ep.ElasticBuffer
            # DEV_EP_GPU_TIMEOUT_S (default 100 = DeepEP's own default; was 10, copied from tests/test_mega_moe.py): the
            # reference's kernels are JIT-compiled per (shape, tokens) by every rank on its own, INSIDE dispatch() /
            # combine(), between that rank's previous kernel and its launch. A rank launches as soon as its own compile
            # ends and then waits this long in the kernel's opening barrier for its peers: with 10 s, a cold size whose
            # compiles spread over more than 10 s across ranks (many live cases, the quiet window's CPU confinement)
            # killed the worker in the REFERENCE (b300_4n_a, 2026-11-13).
            # The value is a template argument of DeepEP's kernels: changing it recompiles each of them once per node.
            self.ep_buffer = ep_type(R.group, num_max_tokens_per_rank=tokens, hidden=self.H, num_topk=self.K,
                                     use_fp8_dispatch=True, explicitly_destroy=True, allow_multiple_reduction=False,
                                     num_gpu_timeout_secs=int(os.environ.get('DEV_EP_GPU_TIMEOUT_S', 100)),
                                     num_cpu_timeout_secs=30)
            kw = {'wait_comm_stream': False} if hasattr(R.deep_ep, 'EPBuffer') else {'use_comm_stream': False}
            self.barrier_fn = lambda: self.ep_buffer.barrier(**kw)
        else:
            self.barrier_fn = lambda: R.dist.all_reduce(torch.empty(1, device='cuda'), group=R.group)
        torch.manual_seed(1000003 * R.rank + 7919 * tokens + self.E)
        self.make_weights()
        self.timing = self.make_inputs()

    def make_weights(self):
        torch, dg = self.torch, self.dg
        from deep_gemm.utils import per_token_cast_to_fp4
        epr, H, I = self.epr, self.H, self.I

        def cast_fp4(w):
            g, n, k = w.shape
            q = torch.empty((g, n, k // 2), device='cuda', dtype=torch.int8)
            sf = torch.empty((g, n, k // 32), device='cuda', dtype=torch.float)
            for i in range(g):
                q[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
            return q, dg.transform_sf_into_required_layout(sf, n, k, (1, 32), g)

        self.l1 = cast_fp4(torch.randn((epr, I * 2, H), dtype=torch.bfloat16, device='cuda'))
        self.l2 = cast_fp4(torch.randn((epr, H, I), dtype=torch.bfloat16, device='cuda'))
        SI = I * self.S
        self.sl1 = self.cast_fp8(torch.randn((SI * 2, H), dtype=torch.bfloat16, device='cuda'))[0::2]
        self.sl2 = self.cast_fp8(torch.randn((H, SI), dtype=torch.bfloat16, device='cuda'))[0::2]
        self.tl1, self.tl2 = dg.transform_weights_for_mega_moe(self.l1, self.l2)
        self.tsl1, self.tsl2 = dg.transform_weights_for_mega_moe(self.sl1, self.sl2)

    def cast_fp8(self, x):
        from deep_gemm.utils import align, per_token_cast_to_fp8
        x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        mn, k = x_sf.shape
        sf_tma = self.torch.empty_strided((mn, k), (1, align(mn, 4)), dtype=x_sf.dtype, device=x_sf.device)
        sf_tma.copy_(x_sf)
        return x_fp8, x_sf, sf_tma

    def shared_sf_layout(self, sf, block_m, num_max_sf_tokens):
        from deep_gemm.utils import align
        num_tokens, packed_sf_k = sf.shape
        result = self.torch.empty_strided((num_max_sf_tokens, packed_sf_k), (1, num_max_sf_tokens), dtype=sf.dtype, device=sf.device)
        result.zero_()
        for b in range((num_tokens + block_m - 1) // block_m):
            for m in range(min(block_m, num_tokens - b * block_m)):
                t = (m // 128) * 128 + (m % 32) * 4 + (m % 128) // 32
                result[b * align(block_m, 128) + t] = sf[b * block_m + m]
        return result

    def make_inputs(self):
        torch, dg, R = self.torch, self.dg, self.R
        T = self.T
        x = torch.randn((T, self.H), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((T, self.E), dtype=torch.float, device='cuda')
        if getattr(self, 'skew', (0.0, 1.0))[0]:
            scores[:, :self.epr] += self.skew[0]
        topk_weights, topk_idx = torch.topk(scores, self.K, dim=-1, largest=True, sorted=False)
        stats0 = torch.randint(0, 100, (self.epr,), dtype=torch.int, device='cuda')
        block_m = dg.get_block_m_for_mega_moe(R.world, self.E, self.buffer.num_max_tokens_per_rank, T, self.K, 'fp8xfp4')
        x_fp8, x_sf, x_sf_tma = self.cast_fp8(x)
        return dict(x=(x_fp8, x_sf), shared_x=(x_fp8, x_sf_tma), topk_idx=topk_idx, topk_weights=topk_weights,
                    shared_l1_x_sf=self.shared_sf_layout(x_sf, block_m, self.buffer.shared_l1_acts_sf.shape[0]),
                    stats0=stats0, stats=stats0.clone(), dev_bm=os.environ.get('DG_MEGA_MOE_DEV_BLOCK_M', ''))

    def copy_inputs(self, inp, buf):
        T = self.T
        buf.x[:T].copy_(inp['x'][0])
        buf.x_sf[:T].copy_(inp['x'][1])
        dst, src = buf.shared_l1_acts_sf, inp['shared_l1_x_sf']
        dev_bm = os.environ.get('DG_MEGA_MOE_DEV_BLOCK_M', '')
        if dev_bm != inp.get('dev_bm', ''):
            # Tile sweep: the host's block_m override differs from the one these inputs were laid out with (arm vs
            # reference / base): the shared expert's input scales are stored per block_m, so lay them out again (cached).
            bm = self.dg.get_block_m_for_mega_moe(self.R.world, self.E, self.buffer.num_max_tokens_per_rank, T, self.K, 'fp8xfp4')
            if f'shared_l1_x_sf@{bm}' not in inp:
                inp[f'shared_l1_x_sf@{bm}'] = self.shared_sf_layout(inp['x'][1], bm, dst.shape[0])
            src = inp[f'shared_l1_x_sf@{bm}']
        if dst.shape == src.shape:
            dst.copy_(src)
        else:
            dst[:T].copy_(src)
            if T < dst.shape[0]:
                dst[T:].copy_(src[-1:].expand(dst.shape[0] - T, -1))
        buf.topk_idx[:T].copy_(inp['topk_idx'])
        buf.topk_weights[:T].copy_(inp['topk_weights'])

    def launch(self, inp, buf=None, stats=None, compile_only=False):
        """The test's run_fused: reset stats, copy inputs into the buffer, launch."""
        torch = self.torch
        buf = self.buffer if buf is None else buf
        stats = inp['stats'] if stats is None else stats
        y = torch.empty((self.T, self.H), dtype=torch.bfloat16, device='cuda')
        if not compile_only:
            stats.copy_(inp['stats0'])
            self.copy_inputs(inp, buf)
        self.dg.fp8_fp4_mega_moe(y=y, l1_weights=self.tl1, l2_weights=self.tl2, sym_buffer=buf,
                                 cumulative_local_expert_recv_stats=stats, activation_clamp=10.0, fast_math=True,
                                 shared_l1_weights=self.tsl1, shared_l2_weights=self.tsl2)
        return y, stats

    def baseline(self, inp):
        """The test's run_baseline: DeepEP dispatch, grouped FP8xFP4 GEMMs + tilelang SwiGLU, DeepEP combine."""
        torch, dg, R = self.torch, self.dg, self.R
        ops = R.tilelang_ops
        stats = inp['stats0'].clone()
        recv_x, _, recv_topk_weights, handle, _ = self.ep_buffer.dispatch(
            inp['x'], topk_idx=inp['topk_idx'], topk_weights=inp['topk_weights'],
            cumulative_local_expert_recv_stats=stats, num_experts=self.E, expert_alignment=R.alignment,
            do_cpu_sync=False, do_handle_copy=False, do_expand=True, use_tma_aligned_col_major_sf=True)
        n = recv_x[0].size(0)
        gk = {'use_psum_layout': True, 'recipe': (1, 1, 32)}
        l2_y = torch.empty((n, self.H), dtype=torch.bfloat16, device='cuda')
        if n > 0:  # n == 0 only in the failure localiser's node-masked runs (a rank whose node is masked out)
            l1_y = torch.empty((n, self.I * 2), dtype=torch.bfloat16, device='cuda')
            dg.m_grouped_fp8_fp4_gemm_nt_contiguous(recv_x, self.l1, l1_y, handle.psum_num_recv_tokens_per_expert, **gk)
            l1_y = ops.swiglu_apply_weight_to_fp8(
                x=l1_y, topk_weights=recv_topk_weights, avail_tokens=handle.psum_num_recv_tokens_per_expert[-1],
                num_per_channels=32, use_col_major_scales=True, clamp_value=10.0, fast_math=True,
                round_scale=True, ue8m0_scale=True, output_bf16=False)
            dg.m_grouped_fp8_fp4_gemm_nt_contiguous(l1_y, self.l2, l2_y, handle.psum_num_recv_tokens_per_expert, **gk)
        # shared expert bias
        T, SI = self.T, self.I * self.S
        sy = torch.empty((T, self.H), dtype=torch.bfloat16, device='cuda')
        s1 = torch.empty((T, SI * 2), dtype=torch.bfloat16, device='cuda')
        dg.fp8_gemm_nt(inp['shared_x'], self.sl1, s1, recipe=(1, 1, 32), disable_ue8m0_cast=True)
        s2 = ops.swiglu_apply_weight_to_fp8(
            x=s1, topk_weights=None, avail_tokens=None, num_per_channels=32, use_col_major_scales=True,
            clamp_value=10.0, fast_math=True, round_scale=True, ue8m0_scale=True, output_bf16=False)
        dg.fp8_gemm_nt(s2, self.sl2, sy, recipe=(1, 1, 32), disable_ue8m0_cast=True)
        return self.ep_buffer.combine(l2_y, handle=handle, bias=sy)[0], stats

    def ref_run(self, inp, base_env):
        """The reference for one input set: (y, stats). Legacy = DeepEP + grouped GEMMs; else the NVLink-reference kernel."""
        torch, R = self.torch, self.R
        if R.legacy:
            return self.baseline(inp)
        env = dict(os.environ)
        R.apply_env(base_env)
        if self.nvl_stats is None:
            self.nvl_stats = torch.empty_like(inp['stats'])
        ref, ref_st = self.launch(inp, self.nvl_ref_buffer, self.nvl_stats)
        os.environ.clear()
        os.environ.update(env)
        return ref, ref_st

    def node_parts(self, inp, base_env, nodes, lsa):
        """COLLECTIVE (every rank runs it in the same order): this rank's output rows split by the node that owns the
        experts: runs the reference once per node with every other node's picks masked (-1). Returns (bias, [part_n]):
        ref_n = bias + part_n and ref = bias + sum(part_n), so bias = (sum(ref_n) - ref) / (nodes - 1)."""
        torch = self.torch
        refs = []
        for n in range(nodes):
            m = dict(inp)
            ti = inp['topk_idx'].clone()
            ti[(ti // (self.epr * lsa)) != n] = -1
            m['topk_idx'] = ti
            refs.append(self.ref_run(m, base_env)[0].float().clone())
        full = self.ref_run(inp, base_env)[0].float()
        bias = (sum(refs) - full) / max(1, nodes - 1)
        return bias, [r - bias for r in refs]

    def check(self, n, base_env):
        from deep_gemm.testing import calc_diff
        torch, R = self.torch, self.R
        out = dict(rank=R.rank, pass_=True, max_calc_diff=0.0, max_rel_rms=0.0, nonfinite=0, stats_mismatch=0)
        # Failure localiser. Local stage: always armed, runs only for a failing test (keeps references to the two
        # previous tests' outputs: no copies, no extra launches). Deep stage (DG_MEGA_MOE_DEV_FAILDEEP=1 in the arm): one
        # all_reduce per test so that every rank learns of a failure anywhere, then collective per-node reference runs
        # for the failing test and the two before it (exact 'minus node n' / 'stale record of node n' hypotheses).
        # DG_MEGA_MOE_DEV_INJECT=R_I_T_MODE (validation): on rank R, test I, token T: zeroqQ | scale | noise | stale.
        deep = os.environ.get('DG_MEGA_MOE_DEV_FAILDEEP', '0') == '1'
        inject = os.environ.get('DG_MEGA_MOE_DEV_INJECT', '')
        lsa = int(getattr(getattr(self.buffer, 'handle', None), 'num_lsa_ranks', 0) or os.environ.get('DEV_LSA', 8))
        nodes = max(1, R.world // max(1, lsa))
        hist = []  # newest first: (test index, y, ref, inp)
        details = []
        # DG_MEGA_MOE_DEV_ALT_TOKENS=a_b_c (arm knob, default off = unchanged; '_' because ',' separates knobs in an arm
        # string and arm_env only accepts [A-Za-z0-9_.+-] in a value): test i runs with the i-th token count of the
        # list (cyclic; each clamped to the case's own count) on the SAME warm buffer, like a server whose batch size changes
        # between launches. DG_MEGA_MOE_DEV_ALT_RANK=1 also shifts the position by the rank index, so ranks differ within a
        # launch. Keep every count in one tile class of the case (same block_m: see get_block_m_for_mega_moe), or the host
        # JIT-compiles another kernel in the middle of the check. The reference runs with the same inputs.
        alt = [int(v) for v in re.split(r'[^0-9]+', os.environ.get('DG_MEGA_MOE_DEV_ALT_TOKENS', '')) if v]
        alt_rank = os.environ.get('DG_MEGA_MOE_DEV_ALT_RANK', '0') == '1'
        case_T = self.T
        if alt:
            out['alt_tokens'] = alt
        try:
            for it in range(n):
                if alt:
                    self.T = min(case_T, alt[(it + (R.rank if alt_rank else 0)) % len(alt)])
                    hist = [h for h in hist if h[1].shape[0] == self.T]  # the localiser compares row by row
                inp = self.make_inputs()
                y, st = self.launch(inp)
                y, st = y.clone(), st.clone()
                t_ref = time.time()
                ref, ref_st = self.ref_run(inp, base_env)
                if it == 0:
                    # host time of this rank's first reference call of the check: mostly its JIT compiles / loads (dispatch
                    # runs with do_cpu_sync=False). The spread over ranks is what the reference's barriers have to absorb
                    out['ref_first_s'] = round(time.time() - t_ref, 2)
                torch.cuda.synchronize()
                if inject:
                    try:
                        ir, ii, itok, mode = inject.split('_', 3)
                        if int(ir) == R.rank and int(ii) == it:
                            itok = min(int(itok), self.T - 1)
                            if mode.startswith('zeroq'):
                                qq = int(mode[5:] or 0); q4 = self.H // 4
                                y[itok, qq * q4:(qq + 1) * q4] = 0
                            elif mode == 'scale':
                                y[itok] = (y[itok].float() * 0.5).to(y.dtype)
                            elif mode == 'noise':
                                y[itok] = torch.randn_like(y[itok].float()).to(y.dtype)
                            elif mode == 'stale' and hist:
                                y[itok] = hist[0][1][itok]
                    except Exception as e:  # noqa: BLE001
                        print(f'FAIL_DETAIL inject spec {inject!r} ignored: {e}', flush=True)
                diff = float(calc_diff(y, ref))
                r = ref.float()
                err = y.float() - r
                rel = (err.pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-30)).item()
                bad = int((~torch.isfinite(y.float())).sum().item())
                out['max_calc_diff'] = max(out['max_calc_diff'], diff)
                out['max_rel_rms'] = max(out['max_rel_rms'], rel)
                out['nonfinite'] += bad
                out['stats_mismatch'] += int(not torch.equal(st, ref_st))
                failed = not (diff < R.tol and bad == 0 and torch.equal(st, ref_st))
                det = None
                if failed:
                    out['pass_'] = False
                    if len(details) < 2:
                        try:
                            det = localise_failure(torch, y, ref, inp['topk_idx'], inp['topk_weights'], self.epr, lsa, R.rank,
                                                   prev=[(f'prev{i + 1}', h[1], h[2]) for i, h in enumerate(hist)])
                            det.update(test=it, tests=n, calc_diff=diff, rel_rms=rel, nonfinite=bad,
                                       stats_equal=bool(torch.equal(st, ref_st)), deep=False)
                            det['line'] = det['line'].replace('FAIL_DETAIL ', f'FAIL_DETAIL test={it}/{n} calc_diff={diff:.3g} ', 1)
                            print(det['line'], flush=True)  # in the rank log: survives even if the request dies later
                            details.append(det)
                        except Exception as e:  # noqa: BLE001
                            print(f'FAIL_DETAIL rank={R.rank} test={it} localiser error: {type(e).__name__}: {e}', flush=True)
                if deep:
                    flag = torch.tensor([1.0 if failed else 0.0], device='cuda')
                    R.dist.all_reduce(flag, group=R.group)
                    if flag.item() > 0 and not getattr(self, '_deep_done', False):
                        self._deep_done = True  # once per case: it costs (nodes + 1) x 3 reference runs on every rank
                        hyps = {}
                        try:
                            bias, parts = self.node_parts(inp, base_env, nodes, lsa)
                            rf = ref.float()
                            for nn in range(nodes):
                                hyps[f'minus_node{nn}'] = rf - parts[nn]
                            hyps['bias_only'] = bias
                            for i, h in enumerate(hist):
                                _, pparts = self.node_parts(h[3], base_env, nodes, lsa)
                                for nn in range(nodes):
                                    hyps[f'stale{i + 1}_node{nn}'] = rf - parts[nn] + pparts[nn]
                            torch.cuda.synchronize()
                            if det is not None:
                                d2 = localise_failure(torch, y, ref, inp['topk_idx'], inp['topk_weights'], self.epr, lsa, R.rank,
                                                      prev=[(f'prev{i + 1}', h[1], h[2]) for i, h in enumerate(hist)], hyps=hyps)
                                det['rows'], det['deep'] = d2['rows'], True
                                det['line'] = d2['line'].replace('FAIL_DETAIL ', f'FAIL_DETAIL DEEP test={it}/{n} calc_diff={diff:.3g} ', 1)
                                print(det['line'], flush=True)
                        except Exception as e:  # noqa: BLE001
                            if any(s in str(e) for s in ('CUDA', 'NCCL', 'illegal', 'trap')):
                                raise
                            print(f'FAIL_DETAIL rank={R.rank} test={it} deep stage error: {type(e).__name__}: {e}', flush=True)
                hist = ([(it, y, ref, inp)] + hist)[:2]
        except Exception as e:  # noqa: BLE001
            if any(s in str(e) for s in ('CUDA', 'NCCL', 'illegal', 'trap')):
                raise
            out['pass_'], out['error'] = False, f'{type(e).__name__}: {e}'[-2000:]
        finally:
            self.T = case_T
        out['pass'] = out.pop('pass_')
        if details:
            out['detail'] = details
        return out

    def alt_timing_inp(self, li, inp):
        """DG_MEGA_MOE_DEV_ALT_TIMING=1 with DG_MEGA_MOE_DEV_ALT_TOKENS (arm knobs, default off = unchanged): launch li of a
        timing block (burn-in included) runs with the li-th token count of the list (cyclic, clamped to the case's count,
        shifted by the rank index with DG_MEGA_MOE_DEV_ALT_RANK=1): launches with changing and per-rank different counts
        and NO reference run in between, which the check loop cannot give. The times of such a block are no benchmark."""
        if os.environ.get('DG_MEGA_MOE_DEV_ALT_TIMING', '0') != '1':
            return inp
        alt = [int(v) for v in re.split(r'[^0-9]+', os.environ.get('DG_MEGA_MOE_DEV_ALT_TOKENS', '')) if v]
        if not alt:
            return inp
        if not hasattr(self, 'alt_case_T'):
            self.alt_case_T, self.alt_inps = self.T, {}
        shift = self.R.rank if os.environ.get('DG_MEGA_MOE_DEV_ALT_RANK', '0') == '1' else 0
        self.T = min(self.alt_case_T, alt[(li + shift) % len(alt)])
        if self.T not in self.alt_inps:
            self.alt_inps[self.T] = self.make_inputs()
        return self.alt_inps[self.T]

    def time_block(self, warmup, launches, trace_path):
        """bench_kineto's per-launch protocol (L2 flush, 10 ms sleep, barrier, launch) under one profiler session;
        returns the CUPTI durations (us) of the timed launches and the number of mega_moe kernels seen."""
        if STALL_DIAG:
            return self.time_block_diag(warmup, launches, trace_path)
        torch = self.torch
        inp = self.timing
        # DG_MEGA_MOE_DEV_TIME_NVL=1 (arm knob, default off = unchanged; needs DG_MEGA_MOE_IB_NVL_REF=1): time the
        # NVLink-reference buffer (all ranks in one NVLink domain, the non-IB kernel) on the same GPUs and inputs
        nvl = os.environ.get('DG_MEGA_MOE_DEV_TIME_NVL', '0') == '1' and self.nvl_ref_buffer is not None
        if nvl:
            saved_env = dict(os.environ)
            self.R.apply_env({})
            if self.nvl_stats is None:
                self.nvl_stats = torch.empty_like(inp['stats'])
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
        with prof:
            for li in range(warmup + launches):
                torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda').zero_()
                torch.cuda._sleep(int(2e7))
                self.barrier_fn()
                if nvl:
                    self.launch(inp, self.nvl_ref_buffer, self.nvl_stats)
                else:
                    self.launch(self.alt_timing_inp(li, inp))
            torch.cuda.synchronize()
            self.T = getattr(self, 'alt_case_T', self.T)
        if nvl:
            os.environ.clear()
            os.environ.update(saved_env)
        prof.export_chrome_trace(trace_path)
        with open(trace_path) as f:
            ev = json.load(f)['traceEvents']
        os.unlink(trace_path)
        k = sorted((e['ts'], e['dur']) for e in ev if e.get('cat') == 'kernel' and 'mega_moe' in e.get('name', ''))
        durs = [d for _, d in k]
        return durs[warmup:], len(durs)


    def time_block_diag(self, warmup, launches, trace_path):
        """DEV_STALL_DIAG=1: time_block with host-side step timing per launch and a stack dump if this rank sits 30 s
        (stderr = the rank log). Same per-launch protocol and the same return value."""
        torch = self.torch
        inp = self.timing
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
        # same protocol, with host-side step timing and a stack dump if this rank sits 30 s (stderr = the rank log)
        import faulthandler
        slow, worst, tb0 = 0, (0.0, None), time.perf_counter()
        faulthandler.dump_traceback_later(30, repeat=True, file=sys.stderr)
        try:
            with prof:
                for li in range(warmup + launches):
                    t0 = time.perf_counter()
                    torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda').zero_()
                    t1 = time.perf_counter()
                    torch.cuda._sleep(int(2e7))
                    t2 = time.perf_counter()
                    self.barrier_fn()
                    t3 = time.perf_counter()
                    self.launch(self.alt_timing_inp(li, inp))
                    t4 = time.perf_counter()
                    if t4 - t0 > worst[0]:
                        worst = (t4 - t0, (li, t1 - t0, t2 - t1, t3 - t2, t4 - t3))
                    if t4 - t0 > 1.0:
                        slow += 1
                        if slow <= 5 or slow % 50 == 0:
                            self.R.say(f'STALL_DIAG slow launch {li} of {warmup + launches} @{self.T}: zero {t1 - t0:.3f} sleep '
                                       f'{t2 - t1:.3f} barrier {t3 - t2:.3f} launch {t4 - t3:.3f} s (slow launches so far: {slow})')
                ts = time.perf_counter()
                torch.cuda.synchronize()
                te = time.perf_counter()
                self.T = getattr(self, 'alt_case_T', self.T)
        finally:
            faulthandler.cancel_dump_traceback_later()
        if (slow or te - ts > 1.0) and worst[1] is not None:
            li, a, b, c, d = worst[1]
            self.R.say(f'STALL_DIAG block @{self.T}: {warmup + launches} launches in {te - tb0:.1f} s, {slow} over 1 s; worst launch '
                       f'{li}: zero {a:.3f} sleep {b:.3f} barrier {c:.3f} launch {d:.3f} s; final synchronize {te - ts:.3f} s')
        prof.export_chrome_trace(trace_path)
        with open(trace_path) as f:
            ev = json.load(f)['traceEvents']
        os.unlink(trace_path)
        k = sorted((e['ts'], e['dur']) for e in ev if e.get('cat') == 'kernel' and 'mega_moe' in e.get('name', ''))
        durs = [d for _, d in k]
        return durs[warmup:], len(durs)


def rank_main(local_rank):
    R = None
    os.environ.setdefault('NCCL_SET_THREAD_NAME', '1')   # 'NCCL GIN P<dev>-<t>' etc.: gin_partition finds them by name
    try:
        R = Rank(local_rank)
        R.serve()
    except BaseException:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        gd = os.environ.get('DEV_GEN_DIR', ROOT)
        with open(f'{gd}/crash_l{local_rank}_n{os.environ.get("RANK", 0)}.txt', 'w') as f:
            f.write(tb[-6000:])
        os._exit(18)


def worker():
    import torch.multiprocessing as mp
    mp.spawn(rank_main, nprocs=int(os.environ.get('GPUS_PER_NODE', 8)))


def fake_variants(msg):
    tag = msg['tag'] if 'tag' in msg else header_tag(msg.get('headers') or {})
    return [{'src': f'// fake kernel\n#define ARM "{knobs}"\n#define TAG {tag or "none"}\n#define TOKENS {s}\n',
             'flags': f'-O3 --include-path {INC_MARK}'} for _, knobs in msg['arms'] for s in msg.get('tokens', [1])]


def fakenvcc(argv):
    """Local tests: 'compiles' kernel.cu into sha(source, include tree); FAKE_NVCC_S seconds per compile."""
    if '--version' in argv:
        print('fake nvcc release 13.2')
        return 0
    src = open(argv[0]).read()
    inc = argv[argv.index('--include-path') + 1]
    time.sleep(float(os.environ.get('FAKE_NVCC_S', 1)))
    if 'COMPILE_ERROR' in src:
        print('kernel.cu(1): error: fake compile error')
        return 1
    with open(argv[argv.index('--output-file') + 1], 'w') as f:
        f.write(sha256_hex(src, tree_digest(inc)))
    return 0


def fakeworker():
    """Local protocol test: no GPUs. Reads the inbox like rank 0, sleeps, answers; 'crash' / 'hang' arms misbehave.
    With the pipeline it plays rank 0's part: prep (sources), job request, wait, JIT compile through the wrapper."""
    gd = os.environ['DEV_GEN_DIR']
    time.sleep(float(os.environ.get('DEV_FAKE_START', 0.5)))
    open(f'{gd}/ready', 'w').close()
    pipe = pipeline_on() and bool(os.environ.get('DG_JIT_NVCC_COMPILER'))
    inc, orig = os.environ.get('DEV_LIVE_INC', ''), os.environ.get('DEV_INCLUDE_ORIG', '')
    while True:
        files = sorted(f for f in os.listdir(INBOX) if f.endswith('.json'))
        if not files or int(os.environ.get('RANK', 0)) != 0:
            time.sleep(0.1)
            continue
        with open(f'{INBOX}/{files[0]}') as f:
            req = json.load(f)
        os.replace(f'{INBOX}/{files[0]}', f'{INBOX}/{files[0]}.taken')
        if req.get('kind') == 'prep':
            out = {'id': req['id'], 'variants': fake_variants(req)}
            if any('nocase' in a[0] for a in req['arms']) and not req.get('force'):
                out = {'id': req['id'], 'nocase': True}
            atomic_write_json(f'{PREP_OUT}/{req["id"]}.json', out)
            continue
        names = [a[0] for a in req['arms']]
        atomic_write_json(PROGRESS, {'id': req['id'], 'phase': 'timing', 'detail': 'fake', 'time': time.time()})
        if 'crash' in names:
            with open(f'{gd}/rank0.log', 'w') as f:
                f.write('DG_IB_HANG fake rank=0\nTraceback: boom\n')
            os._exit(3)
        if 'hang' in names:
            time.sleep(10 ** 6)
        res = {'id': req['id'], 'ok': True, 'arm_order': names, 'tokens': req['tokens'], 'times': {}, 'summary': 'fake', 'notes': []}
        if pipe:
            t = time.time()
            shutil.rmtree(f'{inc}/deep_gemm', ignore_errors=True)
            shutil.copytree(f'{orig}/deep_gemm', f'{inc}/deep_gemm')
            for rel, text in (req.get('headers') or {}).items():
                with open(f'{inc}/{rel}', 'w') as f:
                    f.write(text)
            digest = tree_digest(inc)
            items = fake_variants(req)
            keys = [job_key(c['src'], c['flags'], digest) for c in items]
            atomic_write_json(f'{JOBS_IN}/{req["id"]}.json', {'rid': req['id'], 'digest': digest, 'items': items})
            while not os.path.exists(f'{JOBS_IN}/{req["id"]}.ack'):
                time.sleep(0.05)
            ack = json.load(open(f'{JOBS_IN}/{req["id"]}.ack'))
            tw = time.time()
            while ack.get('ok') and not all(obj_state(k) for k in keys):
                if time.time() - tw > 10 and not any(os.path.exists(f'{ROOT}/pend/{k}') for k in keys if not obj_state(k)):
                    break
                time.sleep(0.05)
            res['times']['compile_wait'] = round(time.time() - tw, 2)
            res['variants'], res['ack'] = [], ack
            with open(f'{ROOT}/quiet', 'w') as f:
                f.write('fake')
            for c, k in zip(items, keys):
                d = f'{gd}/jit_{k}'
                os.makedirs(d, exist_ok=True)
                with open(f'{d}/kernel.cu', 'w') as f:
                    f.write(c['src'])
                p = subprocess.run([os.environ['DG_JIT_NVCC_COMPILER'], f'{d}/kernel.cu', '--cubin', '--output-file', f'{d}/kernel.cubin',
                                    '-O3', '--include-path', inc], env=dict(os.environ, DEV_INC_DIGEST=digest), capture_output=True, text=True)
                good = p.returncode == 0 and open(f'{d}/kernel.cubin').read() == sha256_hex(c['src'], digest)
                res['variants'].append({'key': k, 'rc': p.returncode, 'good': good, 'out': p.stdout[-300:], 'how': (obj_meta(k) or {}).get('how'),
                                        'node': (obj_meta(k) or {}).get('node')})
                if not good and 'COMPILE_ERROR' not in c['src']:
                    res['ok'] = False
            res['times']['compile'] = round(time.time() - t, 2)
        time.sleep(float(req.get('fake_s', 0.3)))
        try:
            os.unlink(f'{ROOT}/quiet')
        except OSError:
            pass
        res.update({'arms': {n: {'sizes': {str(s): {'timing': summarize_timing(
            {0: {0: [170.0 + i, 171.0 + i, 172.0 + i]}})} for s in req['tokens']}} for i, n in enumerate(names)}})
        res['times'].update({'check': 0.1, 'timing': 0.1})
        if 'planned' in names:
            res['restarting'] = bool(os.environ.get('FAKE_ANNOUNCE_RESTART', '1') == '1')
        atomic_write_json(f'{OUTBOX}/{req["id"]}.json', res)
        if 'planned' in names:
            open(f'{gd}/planned_r0', 'w').close()
            time.sleep(2)
            os._exit(PLANNED_EXIT)


# ----------------------------------------------------------------------------------------------------------------------

def selftest():
    static = {'DG_MEGA_MOE_IB': '1', 'DG_MEGA_MOE_IB_CONTEXTS': '8', 'DG_MEGA_MOE_IB_SKIP_CREDIT': '1'}
    assert parse_knobs('DG_MEGA_MOE_X=1, DG_MEGA_MOE_Y=2 -DG_MEGA_MOE_Z DG_MEGA_MOE_W') == [
        ('DG_MEGA_MOE_X', '1'), ('DG_MEGA_MOE_Y', '2'), ('DG_MEGA_MOE_Z', None), ('DG_MEGA_MOE_W', '1')]
    for bad in ('FOO=1', 'DG_MEGA_MOE_x=1', 'DG_MEGA_MOE_X=a;b'):
        try:
            parse_knobs(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    env, notes = arm_env(DEFAULT_KNOBS, '', static)
    assert 'DG_MEGA_MOE_IB_CONTEXTS' not in env and 'DG_MEGA_MOE_IB_SKIP_CREDIT' not in env and len(env) == 5, env
    env, notes = arm_env(DEFAULT_KNOBS, 'DG_MEGA_MOE_IB_HDR_CTX=0', static)
    assert 'DG_MEGA_MOE_IB_HDR_CTX' not in env and any('presence' in n for n in notes)
    env, _ = arm_env(DEFAULT_KNOBS, '-DG_MEGA_MOE_PULL_META_EARLY DG_MEGA_MOE_IB_NEWKNOB=3', static)
    assert 'DG_MEGA_MOE_PULL_META_EARLY' not in env and env['DG_MEGA_MOE_IB_NEWKNOB'] == '3'
    env, _ = arm_env(DEFAULT_KNOBS, 'DG_MEGA_MOE_IB_WAVE_SORT=0', static)
    assert env['DG_MEGA_MOE_IB_WAVE_SORT'] == '0'  # value-checked knob: "0" is passed through
    for bad in ('DG_MEGA_MOE_IB_CONTEXTS=16', '-DG_MEGA_MOE_IB_CONTEXTS', 'DG_MEGA_MOE_IB_SKIP_CREDIT=0', '-DG_MEGA_MOE_IB_SKIP_CREDIT'):
        try:
            arm_env(DEFAULT_KNOBS, bad, static)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert header_tag({}) == '' and header_tag({'a': 'x'}) == header_tag({'a': 'x'}) != header_tag({'a': 'y'})
    s = summarize_timing({0: {0: [10, 12], 1: [11, 11]}, 1: {0: [20, 20], 1: [10, 30]}})
    assert s['median'] == 15.5 and s['round_medians'] == [15.5, 15.5] and s['max'] == 30 and s['step_max'] == 30, s
    assert s['step_p50'] == 20 and s['n'] == 8, s
    assert norm_flags(['-O3', '--include-path', '/x/inc', '-I/y'], '/x/inc') == f'-O3 --include-path {INC_MARK} -I/y'
    assert job_key('a', 'f', 'd') == job_key('a', 'f', 'd') != job_key('a', 'f', 'e')
    print('selftest ok')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'supervise'
    if mode == 'supervise':
        sys.exit(Supervisor().run())
    elif mode == 'worker':
        worker()
    elif mode == 'fakeworker':
        fakeworker()
    elif mode == 'selftest':
        selftest()
    elif mode == 'pinplan':   # dry run of gin_partition on this node (changes nothing)
        print(json.dumps(gin_partition(sys.argv[2] if len(sys.argv) > 2 else None, apply=False)))
    elif mode == 'nvccwrap':
        sys.exit(nvccwrap(sys.argv[2:]))
    elif mode == 'fakenvcc':
        sys.exit(fakenvcc(sys.argv[2:]))
    else:
        sys.exit(f'unknown mode {mode}')
