"""Client for the Mega MoE dev server (jobs/devserver.sh + jobs/devserver.py).

    PY=python    # an interpreter with the `modal` client; MODAL_PROFILE / MODAL_ENVIRONMENT select the deployment
    $PY devclient.py --cluster b300_2n_b status
    # knob-only A/B (arms overlay the server's default knobs; "-NAME" unsets, "NAME=V" sets)
    $PY devclient.py --cluster b300_2n_b ab --arm A= --arm B=-DG_MEGA_MOE_IB_SKIP_CREDIT --tokens 128,1024
    # header change: every include file that differs from the server's base commit in a local checkout
    $PY devclient.py --cluster b300_2n_b ab --repo <checkout> --arm A= --arm B=DG_MEGA_MOE_IB_NEWKNOB=1
    # or explicit files (path inside deep_gemm/include is inferred, or give local:deep_gemm/impls/x.cuh)
    $PY devclient.py --cluster b300_2n_b ab --header <checkout>/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh ...
    $PY devclient.py --cluster b300_2n_b result <id> [--json]
    $PY devclient.py --cluster b300_2n_b stop

Results: per arm and token size, `median` = median over (rank, round) of the per-rank mean of the timed launches (the
slow path's headline number), p10/p50/p90/max over every launch of every rank, step = max over ranks per launch.
"""
import argparse
import json
import re
import os
import pathlib
import subprocess
import sys
import time
import uuid

ENV = os.environ.get('MODAL_ENVIRONMENT') or None


def state():
    if os.environ.get('DEV_DICT_DIR'):
        sys.path.insert(0, str(pathlib.Path(__file__).parent / 'jobs'))
        from devserver import FileDict
        return FileDict(os.environ['DEV_DICT_DIR'])
    # FUSED_IB_MODAL_PROFILE=<name> (optional guard): refuse to run under another Modal profile, which would silently
    # talk to a different workspace
    want = os.environ.get('FUSED_IB_MODAL_PROFILE')
    if want and os.environ.get('MODAL_PROFILE') != want:
        sys.exit(f'MODAL_PROFILE must be {want} (got {os.environ.get("MODAL_PROFILE")!r}): other profiles silently '
                 f'talk to a different workspace. export MODAL_PROFILE={want}')
    import modal
    return modal.Dict.from_name('fused-ib-state', environment_name=ENV)


def rel_include_path(p):
    p = str(p)
    if ':' in p:
        local, rel = p.split(':', 1)
        return pathlib.Path(local).expanduser(), rel
    marker = 'deep_gemm/include/'
    if marker not in p:
        sys.exit(f'{p}: cannot infer the path inside deep_gemm/include; use local_path:deep_gemm/...')
    return pathlib.Path(p).expanduser(), p.split(marker, 1)[1]


def repo_headers(repo, base):
    """Include files that differ between `base` and the working tree of `repo`; refuse non-header changes."""
    repo = str(pathlib.Path(repo).expanduser())
    names = subprocess.run(['git', '-C', repo, 'diff', '--name-only', base], capture_output=True, text=True, check=True).stdout.split()
    names += subprocess.run(['git', '-C', repo, 'ls-files', '--others', '--exclude-standard', 'deep_gemm/include'],
                            capture_output=True, text=True, check=True).stdout.split()
    hdr, other = {}, []
    for n in sorted(set(names)):
        if n.startswith('deep_gemm/include/'):
            path = pathlib.Path(repo) / n
            hdr[n[len('deep_gemm/include/'):]] = path.read_text() if path.exists() else None
        else:
            other.append(n)
    return hdr, other


def unknown_knobs(repo, arms):
    """Knob names used by the arms that appear nowhere in the repo's deep_gemm sources (include/, csrc/, *.py). An
    unknown DG_MEGA_MOE_* name is forwarded as a #define nobody reads: the arm silently measures another config."""
    names = set()
    for _, knobs in arms:
        for tok in re.split(r'[,\s]+', knobs.strip()):
            name = tok.lstrip('-').split('=', 1)[0]
            if name.startswith('DG_') and not name.startswith('DG_MEGA_MOE_DEV_'):
                names.add(name)
    if not names:
        return []
    repo = pathlib.Path(repo).expanduser()
    files = [f for d in ('deep_gemm/include', 'csrc') if (repo / d).is_dir() for f in (repo / d).rglob('*') if f.is_file()]
    files += list((repo / 'deep_gemm').rglob('*.py'))
    missing = set(names)
    for f in files:
        try:
            text = f.read_text(errors='replace')
        except OSError:
            continue
        missing -= {n for n in missing if re.search(rf'\b{re.escape(n)}\b', text)}
        if not missing:
            break
    return sorted(missing)


def fmt(v, w=7, p=1):
    if v is None or v == '-':
        return '-'.rjust(w)
    return f'{v:{w}.{p}f}' if isinstance(v, float) else str(v).rjust(w)


def print_result(res, client_s=None):
    if not res.get('ok') and res.get('error') and not res.get('arms'):
        print(f'REQUEST {res["id"]} FAILED: {res["error"]}')
        prog = res.get('progress') or {}
        if prog:
            print(f'  last phase: {prog.get("phase")} {prog.get("detail", "")}')
        stage, seen = [], set()
        for node, t in (res.get('logs') or {}).items():
            if not isinstance(t, dict):
                print(f'  {node}: {t}')
                continue
            stage += [f'{node} {l}' for l in t.get('stage', [])]
            shown = 0
            for l in t.get('alerts', []):
                body = l.split(': ', 1)[-1]
                if any(x in body for x in ('frame #', 'TCPStore', 'Search for `', 'CUDA_LAUNCH_BLOCKING', 'asynchronously reported')):
                    continue
                if 'STAGE' in body or 'grid sync' in body:
                    stage.append(f'{node} {l}')
                    continue
                key = re.sub(r'\[r\d+ [0-9:]+\] |rank=?\d+|\d{2}:\d{2}:\d{2}', '', body)
                if key in seen or shown >= 6:
                    continue
                seen.add(key)
                shown += 1
                print(f'  {node} alert: {l}')
            w = [l for l in t.get('worker', []) if l.strip()]
            if w:
                print(f'  {node} worker: {w[-1]}')
            crash = [k for k in t if k.startswith('crash')]
            if crash and node == 'node0':
                print(f'  {node} {crash[0]} (of {len(crash)}):\n    ' + '\n    '.join(t[crash[0]][-8:]))
        if stage:
            print(f'  hang forensics ({len(stage)} lines):')
            for l in stage[:int(os.environ.get('DEV_STAGE_LINES', 120))]:
                print('    ' + l)
        if res.get('traceback'):
            print(res['traceback'])
        return
    order = res.get('arm_order') or list(res.get('arms', {}))
    toks = res.get('tokens', [])
    first = order[0] if order else None
    if res.get('label'):
        print(f'*** {res.get("cluster", "")}: {res["label"]} ***')
    print(f'request {res["id"]}  ok={res.get("ok")}  gen={res.get("gen")}  headers={res.get("headers") or "pristine"} '
          f'{res.get("header_tag", "")}  launches={res.get("launches")}x{res.get("rounds")} rounds')
    print(f'{"arm":<10}{"tok":>6} {"check":>16} {"median":>8} {"vs " + str(first):>9} {"rounds":>22} {"p10":>6} {"p50":>6} '
          f'{"p90":>6} {"max":>6} {"step50":>7} {"n":>5} {">1.5x":>5} {"st/10k":>7} {"clean":>8}')
    for s in toks:
        ref = (res['arms'].get(first, {}).get('sizes', {}).get(str(s), {}).get('timing') or {}).get('median')
        for name in order:
            a = res['arms'].get(name, {})
            d = a.get('sizes', {}).get(str(s), {})
            if a.get('error') and not d.get('timing'):
                print(f'{name:<10}{s:>6}  ERROR {a["error"]}')
                if d.get('compile_error'):
                    print('    ' + '\n    '.join(d['compile_error'].splitlines()[-25:]))
                continue
            c = d.get('check')
            cs = '-' if not c else (f'PASS {c["max_calc_diff"]:.1e}' if c['pass'] else f'FAIL {c["max_calc_diff"]:.1e}')
            tm = d.get('timing') or {}
            med = tm.get('median')
            delta = '' if med is None or ref is None or name == first else f'{med - ref:+.1f}'
            rounds = '/'.join(f'{x:.1f}' for x in tm.get('round_medians', []))
            print(f'{name:<10}{s:>6} {cs:>16} {fmt(med, 8)} {delta:>9} {rounds:>22} {fmt(tm.get("p10"), 6, 0)} '
                  f'{fmt(tm.get("p50"), 6, 0)} {fmt(tm.get("p90"), 6, 0)} {fmt(tm.get("max"), 6, 0)} '
                  f'{fmt(tm.get("step_p50"), 7, 0)} {fmt(tm.get("n"), 5)} {fmt(tm.get("over_1_5x"), 5)} '
                  f'{fmt(tm.get("stalls_per_10k"), 7, 1)} {fmt(tm.get("median_clean"), 8)}')
            if c and not c['pass']:
                print(f'    check failed on ranks {c.get("failed_ranks")}: nonfinite={c.get("nonfinite")} '
                      f'stats_mismatch={c.get("stats_mismatch")} {c.get("error") or ""}')
                for fd in c.get('fail_detail') or []:  # one line per failing rank and test; full rows in the JSON
                    print('    ' + fd.get('line', str(fd)[:300]))
    for name in order:
        for n in res['arms'].get(name, {}).get('notes', []):
            print(f'  note [{name}]: {n}')
    for n in res.get('notes', []):
        print(f'  note: {n}')
    for l in res.get('alerts', [])[:20]:
        print(f'  alert: {l}')
    if res.get('trace'):
        for k, lines in res['trace'].items():
            print(f'--- trace {k} ({len(lines)} lines)')
            for l in lines:
                print('  ' + l)
    t = res.get('times', {})
    print(f'latency: server {res.get("elapsed_s")} s (worker {t.get("total_worker")}) = headers {t.get("headers")} + '
          f'cases {t.get("cases")} + compile {t.get("compile")} ({res.get("compiled_variants")} variants, slowest '
          f'{t.get("compile_max_variant")}) + check {t.get("check")} + timing {t.get("timing")} s'
          + (f'; client round trip {client_s:.0f} s' if client_s else ''))
    if res.get('pipeline'):
        # compile pipeline: where each variant's cubin came from (ahead / pool = compiled once in the cluster by the
        # supervisors' pool, shipped = by another node's pool, inline = by a rank as in the old path, jit-cache = the
        # JIT already had it) and how long the compiles took (they overlap earlier requests when 'ahead')
        vs, ah = res.get('variants') or [], res.get('ahead') or {}
        how = {}
        for v in vs:
            how[v.get('how')] = how.get(v.get('how'), 0) + 1
        cs = [v['compile_s'] for v in vs if v.get('compile_s')]
        print(f'pipeline: queued {res.get("queue_s")} s (held by the server {res.get("held_s")} s), '
              f'{"compiled ahead: " + str(ah.get("jobs")) + " job(s)" if ah.get("prepped") else "not compiled ahead"}; '
              f'in the request: capture {t.get("capture")} + wait {t.get("compile_wait")} + install {t.get("install")} s; '
              f'cubins {how}' + (f', compile {min(cs):.0f}-{max(cs):.0f} s each' if cs else '')
              + f'; quiet window {"yes" if res.get("quiet") else "NO"}'
              + ('' if all(v.get('verified') for v in vs) else f'; {sum(1 for v in vs if not v.get("verified"))} cubin(s) from the pre-pipeline JIT cache'))
        # code hash per arm and size (all ELF sections except symbol / string tables): the same hash in two requests
        # = byte-identical kernel code and data
        code = {(a, v.get('size')): (v.get('code_sha') or '?')[:8] for v in vs for a in v.get('arms', [])}
        for name in res.get('arm_order', []):
            hs = ' '.join(f'{sz}={code[(name, sz)]}' for sz in res.get('tokens', []) if (name, sz) in code)
            if hs:
                print(f'  code [{name}]: {hs}')
        if os.environ.get('DEV_SHOW_VARIANTS'):
            for v in vs:
                print(f'  variant {"/".join(v.get("arms", []))} @{v.get("size")}: {v.get("how")} node {v.get("compile_node")} '
                      f'{v.get("compile_s")} s  src {v.get("src_sha")} code {v.get("code_sha")} hdr {v.get("include_digest")} {v.get("entry")}')


def wait_result(st, K, rid, timeout, quiet=False):
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        res = st.get(f'{K}res/{rid}')
        if res is not None:
            return res, time.time() - t0
        s = st.get(K + 'status') or {}
        prog = s.get('progress') or {}
        line = (f'{s.get("state")} gen={s.get("gen")} req={s.get("request")} '
                f'{prog.get("phase", "") if prog.get("id") == rid else ""} {prog.get("detail", "") if prog.get("id") == rid else ""}')
        if line != last and not quiet:
            print(f'  [{time.time() - t0:5.0f} s] {line}', flush=True)
            last = line
        time.sleep(2)
    return None, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster', required=True)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    sp = sub.add_parser('stop')
    sp.add_argument('--drain', action='store_true', help='let the running request finish first (nothing new is started)')
    sub.add_parser('queue', help='list the requests waiting in the queue (and the ones the server already holds)')
    c = sub.add_parser('cancel', help='withdraw queued requests (not the one that is running)')
    c.add_argument('ids', nargs='+', help='request ids, or unique prefixes / name prefixes of ids')
    r = sub.add_parser('result')
    r.add_argument('id')
    r.add_argument('--json', action='store_true')
    r.add_argument('--wait', type=float, default=0, help='block up to this many seconds for the result')
    a = sub.add_parser('ab')
    a.add_argument('--arm', action='append', required=True, help='NAME=KNOBS (KNOBS: "K=V,K2=V2 -K3"; empty = default)')
    a.add_argument('--tokens', default='128,1024')
    a.add_argument('--shape', default='flash', help='flash | megaseek | JSON dict (hidden, inter, experts, topk, shared)')
    a.add_argument('--base', default=None, help='knob set the arms overlay (default: the server\'s DEV_BASE_KNOBS)')
    a.add_argument('--header', action='append', default=[], help='local header (path inside deep_gemm/include inferred)')
    a.add_argument('--repo', help='send every deep_gemm/include file that differs from the server\'s base commit')
    a.add_argument('--force', action='store_true', help='with --repo: ignore non-header changes')
    a.add_argument('--launches', type=int, default=30)
    a.add_argument('--warmup', type=int, default=10)
    a.add_argument('--rounds', type=int, default=3)
    a.add_argument('--check', type=int, default=2, help='DeepEP correctness tests per arm and size (0 = off)')
    a.add_argument('--compile-only', action='store_true', help='compile (and cache on every node) the variants; no check, no timing')
    a.add_argument('--allow-unknown-knobs', action='store_true',
                   help='send arms whose DG_* knob names appear nowhere in the --repo tree (they become unread #defines)')
    a.add_argument('--no-quiet', action='store_true', help='TEST ONLY: leave background compiles running during check + timing')
    a.add_argument('--no-ahead', action='store_true', help='do not compile this request ahead of its turn')
    a.add_argument('--trace', action='store_true', help='compile every arm with DG_MEGA_MOE_IB_TRACE=1, return the trace lines of the kernel (DG_IB_TRACE; OPUS_IB_TRACE on older trees)')
    a.add_argument('--trace-epochs', type=int, default=2, help='traced launches (every 16th epoch prints) kept per rank, arm and size')
    a.add_argument('--timeout', type=int, default=900, help='server-side request timeout, s')
    a.add_argument('--name', default='')
    a.add_argument('--no-wait', action='store_true')
    a.add_argument('--json', help='also write the raw result here')
    args = ap.parse_args()

    st = state()
    K = f'dev/{args.cluster}/'
    if args.cmd == 'status':
        srv, s = st.get(K + 'server') or {}, st.get(K + 'status') or {}
        if not s:
            print('no dev server has reported on', args.cluster)
            return
        age = time.time() - s.get('time', 0)
        if s.get('label'):
            print(f'*** {s["label"]} ***')
        print(f'{args.cluster}: {s.get("state")} (status {age:.0f} s old) queued={s.get("queued")} job={s.get("job")} gen={s.get("gen")} '
              f'restarts={s.get("restarts")} request={s.get("request")} progress={s.get("progress")}')
        if srv:
            print(f'  server: base_commit={srv.get("base_commit")} patch={srv.get("patch")} nodes={srv.get("nodes")} '
                  f'up {(time.time() - srv.get("started", time.time())) / 60:.0f} min, '
                  f'{(srv.get("deadline", 0) - time.time()) / 3600:.1f} h left; static={srv.get("static")}')
        w = st.get(K + 'warmup')
        if w:
            print(f'  last warmup {w.get("id")}: {w.get("summary") or w.get("error")}')
        if s.get('pipeline'):
            held = ', '.join(f'{h["id"].split("-")[0] or h["id"]}:{h["state"]} {h["jobs_done"]}/{h["jobs"]}' for h in s.get('pending') or [])
            print(f'  pipeline: pool {s.get("pool")}; held ahead: {held or "-"}')
        return
    if args.cmd == 'stop' and getattr(args, 'drain', False):
        st.put(K + 'drain', {'time': time.time()})
        print(f'drain requested for the {args.cluster} dev server: it finishes the running request, returns the requests '
              'it holds to the queue head and stops (servers older than 2026-11-12 20:30 ignore this)')
        return
    if args.cmd == 'stop':
        st.put(K + 'stop', {'time': time.time()})
        print(f'stop requested for the {args.cluster} dev server (the job ends; the runner takes the next job)')
        return
    if args.cmd in ('queue', 'cancel'):
        sys.path.insert(0, str(pathlib.Path(__file__).parent / 'jobs'))
        from devserver import ReqQueue
        rq = ReqQueue(args.cluster)
        s = st.get(K + 'status') or {}
        held = s.get('pending') or []   # requests a pipeline server already took from the queue (compile-ahead)
        if args.cmd == 'queue':
            print(f'{args.cluster}: running {s.get("request")}; {len(held)} held by the server, {rq.len()} in the queue')
            for h in held:
                print(f'  held    {h.get("id")}  {h.get("state", "")}  jobs {h.get("jobs_done", "-")}/{h.get("jobs", "-")}')
            for r in rq.peek():
                print(f'  queued  {r["id"]}  {len(r.get("arms", []))} arms x {r.get("tokens")}  {len(r.get("headers") or {})} header(s)  '
                      f'waiting {time.time() - r.get("submitted", time.time()):.0f} s')
            return
        want = list(args.ids)
        match = lambda rid: any(rid == w or rid.startswith(w) for w in want)
        removed = rq.remove(match)
        for r in removed:
            st.put(f'{K}res/{r["id"]}', dict(id=r['id'], ok=False, cancelled=True, error='cancelled by the client (was queued)'))
            print(f'cancelled {r["id"]} (removed from the queue)')
        for h in held:
            if match(h.get('id', '')):
                st.put(f'{K}cancel/{h["id"]}', {'time': time.time()})
                print(f'cancel requested for {h["id"]} (held by the server; it posts a "cancelled" result)')
        if s.get('request') and match(s['request']):
            print(f'{s["request"]} is running: not cancelled (it finishes or times out)')
        if not removed and not any(match(h.get('id', '')) for h in held):
            print('nothing matched in the queue')
        return
    if args.cmd == 'result':
        res = st.get(f'{K}res/{args.id}')
        if res is None and args.wait > 0:
            res, _ = wait_result(st, K, args.id, args.wait, quiet=args.json)
        if res is None:
            sys.exit('no such result (yet)')
        print(json.dumps(res, indent=1) if args.json else '', end='')
        if not args.json:
            print_result(res)
        return

    # ab
    srv = st.get(K + 'server') or {}
    s = st.get(K + 'status') or {}
    if not s or time.time() - s.get('time', 0) > 60 or s.get('state') in ('stopped', 'dead'):
        print(f'warning: dev server on {args.cluster} looks down (status: {s.get("state")}, '
              f'{time.time() - s.get("time", 0):.0f} s old); the request waits in the slot', file=sys.stderr)
    headers = {}
    if args.repo:
        base = srv.get('base_commit')
        if not base:
            sys.exit('server did not report a base commit (DEV_BASE_COMMIT); use --header')
        hdr, other = repo_headers(args.repo, base)
        if other and not args.force:
            sys.exit(f'non-header changes vs {base} need a new patch + server restart (or --force to ignore): {other}')
        for rel, text in hdr.items():
            if text is None:
                sys.exit(f'{rel} was deleted in the repo; the server cannot delete pristine headers')
            headers[rel] = text
    for h in args.header:
        local, rel = rel_include_path(h)
        headers[rel] = local.read_text()
    arms = []
    for spec in args.arm:
        name, _, knobs = spec.partition('=')
        arms.append([name, knobs])
    if len({a[0] for a in arms}) != len(arms):
        sys.exit('arm names must be unique')
    if args.repo:
        unk = unknown_knobs(args.repo, arms)
        if unk and not args.allow_unknown_knobs:
            sys.exit(f'REFUSED: knob(s) {unk} appear nowhere in {args.repo} (deep_gemm/include, csrc, deep_gemm/*.py): this '
                     'tree would ignore them and the arm would measure a different configuration. Wrong tree / typo? '
                     'Override with --allow-unknown-knobs.')
        if unk:
            print(f'WARNING: knob(s) {unk} are not read anywhere in {args.repo} (sent anyway)', file=sys.stderr)
    elif any('DG_' in k for _, k in arms):
        print('note: no --repo given: knob names are not checked against a source tree', file=sys.stderr)
    shape = args.shape if args.shape in ('flash', 'megaseek') else json.loads(args.shape)
    rid = f'{args.name + "-" if args.name else ""}{time.strftime("%m%d-%H%M%S")}-{uuid.uuid4().hex[:4]}'
    req = dict(id=rid, arms=arms, tokens=[int(x) for x in args.tokens.split(',')], shape=shape, headers=headers,
               launches=args.launches, warmup=args.warmup, rounds=args.rounds, check=args.check, trace=args.trace,
               trace_epochs=args.trace_epochs, timeout=args.timeout, submitted=time.time())
    if args.base is not None:
        req['base'] = args.base
    if args.compile_only:
        req['compile_only'] = True
    if args.no_ahead:
        req['no_ahead'] = True
    if args.no_quiet:
        req['no_quiet'] = True
    t0 = time.time()
    sys.path.insert(0, str(pathlib.Path(__file__).parent / 'jobs'))
    from devserver import ReqQueue
    rq = ReqQueue(args.cluster)
    ahead = rq.len()
    rq.put(req)
    print(f'queued behind {ahead} request(s); ' if ahead else '', end='')
    print(f'submitted {rid}: {len(arms)} arms x tokens {req["tokens"]}, {len(headers)} header(s) '
          f'{sorted(headers)[:4]}{"..." if len(headers) > 4 else ""}', flush=True)
    if args.no_wait:
        return
    res, waited = wait_result(st, K, rid, args.timeout + 3600)
    if res is None:
        sys.exit(f'{rid}: no result after {waited:.0f} s')
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res, indent=1))
    print_result(res, client_s=time.time() - t0)
    sys.exit(0 if res.get('ok') else 1)


if __name__ == '__main__':
    main()
