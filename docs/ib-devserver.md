# Fused-IB MegaMoE dev server

A dev server holds a multi-node B300 cluster warm and answers A/B requests for the fused-IB MegaMoE kernel on this
branch. A request names one or more *arms* (knob sets and/or edited kernel headers), a model shape and token counts.
The server compiles every variant, checks each against the DeepEP reference, times the arms interleaved, and returns
medians. Cached variants come back in 5–16 s, new variants in 75–180 s (almost all nvcc), against 10–20 min for a
fresh cluster job.

The harness is not in this repository. It lives with the Modal cluster runner:

| file | role |
|---|---|
| `cluster/runner.py` | Modal app that holds an N-node RDMA cluster for up to 23.5 h and runs submitted jobs on every node |
| `cluster/submit.py` | submit a job script to a held cluster; `--status`, `--start`, `--stop` |
| `cluster/jobs/devserver.sh` | the dev-server job: builds this branch as a patch, then starts `devserver.py` on every node |
| `cluster/jobs/devserver.py` | supervisor and worker processes (below) |
| `cluster/devclient.py` | client: `status`, `ab`, `result`, `stop` |
| `cluster/sweep_driver.py` | runs a token sweep as a series of small requests and saves JSON |

All commands below run from `cluster/` with the Modal profile that owns the runner:

```bash
export MODAL_PROFILE=modal-labs MODAL_ENVIRONMENT=jamesl-dev
PY=~/autoinference/.venv/bin/python
```

`devclient.py` refuses any other profile, because another profile silently talks to a different workspace.

## How it works

```
runner (Modal clustered function, one per node)
  └─ job: devserver.sh            builds DeepGEMM from the patch once, sets the server env
       └─ devserver.py supervise  per node, no CUDA; node 0 owns the request slot
            └─ devserver.py worker  one rank per GPU, NCCL initialised once
```

- **Supervisor** (per node). Starts the worker and restarts it when it dies or a request times out. Nodes stay in
  step through a shared generation counter: any node bumps it and every node restarts its worker. Node 0 pops
  requests from the queue, hands each to its worker through a local file, and writes the result (or a failure with
  the last log lines of every node).
- **Worker** (one rank per GPU). Keeps symmetric buffers, DeepEP reference buffers and weights per (shape, tokens)
  case. Per request it writes the request's headers over a private install, JIT-compiles every (arm, size) variant
  in parallel over the local ranks (compile-only hook `DG_DEV_COMPILE_ONLY`), runs the DeepEP correctness check per
  arm and size, then times the arms interleaved per round (A B C A B C ...).
- **Timing** reproduces `bench_kineto`: before every launch a 256 MB L2 flush, a GPU sleep and a barrier; the
  duration is the kernel's CUPTI duration. `median` is the median over (rank, round) of each rank's mean over the
  timed launches, the same headline number as the one-off test jobs.
- **State.** Requests go through the Modal Queue `fused-ib-devq` (one partition per cluster, FIFO, safe for several
  clients). Status and results live in the Dict `fused-ib-state` under `dev/<cluster>/`.

## Start a server

1. Hold a cluster. Clusters are named by provider and node count: `and2` / `and4` (Andromeda), `csc2` / `csc4`
   (Cirrascale), `any4` (any provider, released unless it has 800G ConnectX-8). A watchdog keeps one runner per
   group alive; `--start` lets it request one:

   ```bash
   $PY submit.py --cluster csc4 --start     # the watchdog requests it within 5 minutes
   $PY submit.py --cluster csc4 --status    # heartbeat, nodes, NICs, hours left
   ```

   Check the NIC list shows `800 Gb/sec` ConnectX-8 ports before using it. For a group of several clusters,
   release all but one once one is up (`--stop`), or you hold several clusters at once.

2. Make the patch: the diff of this branch against the upstream commit the job builds on.

   ```bash
   git -C ~/work/deepgemm-ib-integ diff 5081f85 <commit> > jobs/devserver_<name>.patch
   ```

3. Submit the server job. 4 × 8 B300 (32 GPUs), as started on 12 Nov:

   ```bash
   $PY submit.py --cluster csc4 --script jobs/devserver.sh \
     --file jobs/devserver.py --file jobs/setup_deps.sh --file jobs/nccl_topo.py --file jobs/devserver_<name>.patch \
     --env PATCH=devserver_<name>.patch --env DEV_BASE_COMMIT=<commit> \
     --env TEST_NODES=4 --env DG_MEGA_MOE_IB_GIN_TYPE=3 \
     --env DEV_WARM=megaseek:1024 --env DEV_HOURS=20 \
     --timeout 75600 --no-wait --name devserver_b300_ep32
   ```

   For 2 × 8 use `TEST_NODES=2`. Keep `DEV_HOURS` below the job `--timeout` and the runner's remaining hours.
   The first build takes 3–5 min, then the warm-up case compiles and checks.

| server env | meaning |
|---|---|
| `PATCH` | patch file shipped with `--file`, applied to upstream DeepGEMM and built into a private install |
| `DEV_BASE_COMMIT` | commit the patch was made from; `ab --repo` diffs headers against it |
| `TEST_NODES` | nodes the ranks span (2 = 16 GPUs, 4 = 32 GPUs) |
| `DG_MEGA_MOE_IB_GIN_TYPE` | Gin transport: 3 on B300 |
| `DEV_BASE_KNOBS` | knobs every arm overlays (default: `SKIP_CREDIT`, `HDR_CTX`, `CONTEXTS=8`, `FRONT`, `GPU_ACQ`, `STREAM_COMBINE`, `PULL_META_EARLY`) |
| `DEV_WARM` | cases built and checked at start, e.g. `flash:128,1024` or `megaseek:1024`; empty for none |
| `DEV_HOURS` | the supervisor exits cleanly after this long (default 5) |
| `NO_LEGACY=1` | skip DeepEP (no reference check; GB200) |

## Use it

```bash
$PY devclient.py --cluster csc4 status

# knob-only A/B: arms overlay the base knobs; "K=V" sets, "-K" unsets, commas within an arm, empty = base
$PY devclient.py --cluster csc4 ab --shape megaseek --tokens 128,1024 \
  --arm A= --arm B=DG_MEGA_MOE_IB_FWD_ROTATE=1,DG_MEGA_MOE_IB_SCAN_BATCH=16

# any shape as JSON (hidden, inter, experts, topk, shared); e.g. Kimi-K3's routed MoE sizes (see New shapes below)
$PY devclient.py --cluster csc4 ab --tokens 128,1024 --arm A= \
  --shape '{"hidden":3584,"inter":3072,"experts":896,"topk":16,"shared":1}'

# kernel header edits: every deep_gemm/include file in a checkout that differs from DEV_BASE_COMMIT
$PY devclient.py --cluster csc4 ab --repo ~/work/deepgemm-ib-ep32 --arm A= --arm B=DG_MEGA_MOE_IB_NEWKNOB=1
#   or one file: --header ~/work/x/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh

# traced: every arm built with DG_MEGA_MOE_IB_TRACE=1; returns the per-phase trace lines (timing not comparable)
$PY devclient.py --cluster csc4 ab --trace --arm A= --tokens 1024 --check 0

# zero-network ceiling: ORACLE arms must be alone in a request (the server restarts its ranks around it)
$PY devclient.py --cluster csc4 ab --arm O2=DG_MEGA_MOE_IB_ORACLE=2 --check 0 --tokens 1024

# don't block; fetch later
$PY devclient.py --cluster csc4 ab ... --no-wait
$PY devclient.py --cluster csc4 result <id> --wait 1800 [--json]

# a full token sweep as a series of 2-size requests, saved to results/report/<label>/
$PY sweep_driver.py --cluster csc4 --label megaseek_ib32 --shape megaseek \
  --tokens 32,64,128,256,512,1024 --arm TS8=<knobs> --timeout 3000

$PY devclient.py --cluster csc4 stop       # stop the server (the cluster stays held)
$PY submit.py --cluster csc4 --stop        # release the cluster
```

`ab` options: `--rounds 3`, `--launches 30`, `--warmup 10`, `--check 2` (DeepEP tests per arm and size, 0 = off),
`--timeout 900` (server side), `--base` (overlay a different knob set), `--json FILE`.

**Reading a result.** `median` is the headline; `rounds` are per-round medians (spread = noise); `p10/p50/p90/max`
cover every launch of every rank; `step50` is the median over launches of the slowest rank; `>1.5x` counts launches
above 1.5× the median. Arms are interleaved, so compare arms within one request; across clusters and days, medians
drift by 5–15 µs.

## What needs a restart

- **Per-server knobs** (set at allocation or import): `DG_MEGA_MOE_IB_CONTEXTS`, `SHARING`, `EXCLUSIVE`, `RAIL`,
  `GIN_TYPE`, `VIRTUAL_LSA`, `SKIP_CREDIT`. Mixing credit-check and skip-credit kernels on the same contexts hung.
  An arm may repeat the server's value but not change it.
- **Host or Python changes** (`csrc/`, `deep_gemm/*.py`) and `layout/` edits that change host-side sizes: make a
  new patch and restart the server. `--repo` refuses a checkout with non-header changes; `--force` sends the headers and ignores the rest.
- New knobs need no restart: any `DG_MEGA_MOE_*` variable the host does not read becomes `#define NAME VALUE` in the
  JIT build. Test it with `#if NAME`, not `#ifdef`. Presence-only knobs given `=0` are treated as unset.

## Pitfalls seen so far

- **At most 2 token sizes per request.** Each case keeps its own IB buffer, and each buffer creates its own NCCL
  communicator with busy-polling proxy threads. Six live cases ran each rank at about 12 busy cores, oversubscribed
  64-CPU nodes and turned launches into millisecond stalls. The server restarts its ranks when more than 3 cases are alive; `sweep_driver.py` sends 2
  sizes per request.
- **Planned-restart race.** A request picked up while the ranks exit for a planned restart can read as "worker
  exited", while the real result lands later. `sweep_driver.py` re-reads a failed result after 90 s before retrying.
- **First-launch compile skew.** Per-rank JIT compile time is bimodal (5–8 s or 40–55 s) against the kernel's 60 s
  timeout. The kernel skips its first launch per variant and rendezvouses; this holds only when every rank sees the
  same token count.
- **Runner job timeouts do not kill child processes.** A timed-out `torchrun` job leaves its ranks running and
  holding ports and GPUs. Kill them with a job that runs `pkill -9 -f <script>` on every node.
- **Size-gated knobs.** Several knobs compile in only above a size: `SCAN_SLICES` at ≥ 8192 top-k entries per rank,
  `X_ORDER` with one remote node, lean metadata and `FWD_SKIP_XPOLL` at top-k ≥ 8, `FWD_ROTATE` above 16 ranks and
  ≥ 512 max tokens. Code compiled in but switched off still costs 2–14 µs, so compare builds, not just runtime flags.
- **New shapes.** Shapes other than hidden 5120 have not been compiled with the IB path.
  - Hidden 3584 fails the forward-combine split assert ("IB forward piece": a row is 14 pieces of 512 bytes, which
    do not split 4 ways). It needs a split that divides the row (2 or 7).
  - The reference path always runs a shared expert, so `"shared": 0` is untested and likely fails in the check;
    use `"shared": 1` or extend the reference.
  - The kernel and the reference implement SwiGLU only (the host asserts `activation == "swiglu"`). A model with
    another expert activation, such as Kimi-K3's SiTU, can be timed at its sizes but not checked against itself.
