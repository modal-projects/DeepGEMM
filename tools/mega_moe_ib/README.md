# Benchmark harness for Mega MoE across NVLink domains

This is the harness the numbers in `docs/mega_moe_ib/REPRODUCTION.md` were measured with, with deployment
names removed. It runs on [Modal](https://modal.com) multi-node clusters with RDMA. Nothing in it is needed to
build or use the kernel; it exists so the tables can be reproduced and new changes can be A/B-timed.

| file | role |
|---|---|
| `runner.py` | Modal app. Holds an N-node cluster (`@modal.clustered(size=N, rdma=True)`) for up to 23.5 h and runs submitted jobs on every node. A `watchdog` cron (every 5 minutes) keeps enabled clusters alive and requeues an orphaned job once. |
| `submit.py` | Submit a job (a bash script plus files) to a held cluster; `--status`, `--stop`, `--start`. |
| `fetch.py` | Download a finished job's logs. |
| `jobs/devserver.sh`, `jobs/devserver.py` | The dev server job: builds one DeepGEMM tree, keeps the ranks and their buffers warm, and answers A/B requests. |
| `devclient.py` | Client of the dev server: `status`, `ab`, `result`, `queue`, `cancel`, `stop`. |
| `jobs/setup_deps.sh` | Sourced by the job: clones and builds DeepGEMM and DeepEP (the reference) once per container. |
| `jobs/inrepo_test.sh` | Runs `tests/test_mega_moe.py` across the nodes of a cluster in a plain environment (no dev server, no development switch). |
| `jobs/nccl_topo.py` | Writes an NCCL topology file for VM hosts whose GPUs and NICs share no PCIe path. |
| `mkab.py` | Builds one header tree that holds several kernels, each selected by an arm's define, for a same-request comparison of trees. |
| `gate1.sh` | Local check that the IB-off kernel compiles to the same PTX as upstream (front end only). |

## Deployment settings

All of them come from the environment. None has a deployment-specific default.

| variable | read by | meaning | default |
|---|---|---|---|
| `MODAL_PROFILE`, `MODAL_ENVIRONMENT` | the Modal client | workspace profile and environment | your Modal defaults |
| `FUSED_IB_APP` | `runner.py` | name of the Modal app | `fused-ib-runner` |
| `FUSED_IB_CLOUDS` | `runner.py` | JSON object `{cluster: cloud}`: pins a cluster function to one cloud | `{}` (not pinned) |
| `FUSED_IB_MODAL_PROFILE` | `devclient.py` | if set, the client refuses to run under another `MODAL_PROFILE` | unset (no check) |
| `DEV_MODAL_ENV` | `jobs/devserver.py` (in the container) | Modal environment of the state objects | the container's own |
| `DG_NCCL_INCLUDE_DIR` | the DeepGEMM JIT, in every rank process | include directory of the NCCL in use: the JIT compiles kernels that include `<nccl.h>`, and without it every launch across NVLink domains fails at its first compile | set by `jobs/devserver.sh` and `jobs/inrepo_test.sh` to the `nvidia-nccl-cu13` wheel's `include`; set it yourself in any other launcher |

State lives in three Modal objects of that environment: Queue `fused-ib-jobs` (one partition per cluster; the
dev server adds `fused-ib-devq`), Dict `fused-ib-state` (heartbeat, current job, exit codes, dev-server
requests and results), Volume `fused-ib-results` (logs).

Clusters are the functions in `runner.py`. Their names are labels; what matters is the hardware:

| name | hardware |
|---|---|
| `b300_2n_b`, `b300_2n_a` | 2 nodes x 8 B300, ConnectX-8 800G InfiniBand, bare metal |
| `b300_4n_b`, `b300_4n_a` | 4 nodes x 8 B300, same fabric |
| `gb200_16_a`, `gb200_32_a` | 4 / 8 GB200 trays x 4 GPUs in one rack, ConnectX-7 400G InfiniBand |
| `b300_2n_u`, `b300_4n_u`, `b300_4n_g` | unpinned B300 (`b300_4n_g` releases its nodes unless it finds an 800G ConnectX fabric) |
| `b200_2n_b`, `b200_2n_c`, `b200_2n_d` | 2 nodes x 8 B200 |

Image: `nvidia/cuda:13.2.0-devel-ubuntu24.04`, Python 3.12, torch 2.14.1 (cu130), `nvidia-nccl-cu13==2.32.3`.
The cluster functions run on `runc` and set `exclude_reserved_capacity`.

## Bring-up

```bash
modal deploy runner.py                         # the app and its watchdog
python submit.py --cluster b300_2n_b --start        # enable: the watchdog launches the runner within 5 minutes
python submit.py --cluster b300_2n_b --status       # heartbeat, NICs as the container sees them, hours left
```

A dev server builds upstream DeepGEMM at the commit pinned in `jobs/setup_deps.sh` (`DEEPGEMM_COMMIT`) plus
one patch, so the tree under test is shipped as a diff:

```bash
BASE=<commit of the tree under test>
git -C <checkout> diff 5081f85 $BASE > jobs/devserver_tree.patch
python submit.py --cluster b300_2n_b --script jobs/devserver.sh \
    --file jobs/devserver.py --file jobs/setup_deps.sh --file jobs/nccl_topo.py --file jobs/devserver_tree.patch \
    --env PATCH=devserver_tree.patch --env DEV_BASE_COMMIT=$BASE --env TEST_NODES=2 \
    --env DG_MEGA_MOE_IB_GIN_TYPE=3 --env DEV_HDR_FRESH=1 --env DEV_MAX_CASES=8 \
    --env DEV_HOURS=15 --env DEV_WARM=megaseek:1024 --timeout 57600 --no-wait --name devserver
python devclient.py --cluster b300_2n_b status
```

Server-side switches (all `--env` of the job; the full list is in the two `jobs/devserver.*` files):

| variable | meaning |
|---|---|
| `PATCH`, `DEV_BASE_COMMIT` | the tree; `devclient.py ab --repo` sends every header that differs from `DEV_BASE_COMMIT` |
| `DG_MEGA_MOE_DEV` | `jobs/devserver.sh` exports `DG_MEGA_MOE_DEV=1` by default: the host refuses plan overrides, trace, zero-network and compile-only switches without it, and the server uses the compile-only switch for every request. `--env DG_MEGA_MOE_DEV=0` overrides it (every such switch is then refused by name). It is fixed for the server's lifetime; an arm cannot set or unset it |
| `TEST_NODES` | nodes that take part (2 on a 4-node cluster gives a 16-GPU server) |
| `DG_MEGA_MOE_IB_GIN_TYPE` | NCCL Gin backend: 3 (GDAKI) on the InfiniBand clusters, 0 = any |
| `DG_MEGA_MOE_IB_VIRTUAL_LSA=n` | treat groups of n consecutive ranks as NVLink domains (GB200: 8 = two trays per logical node) |
| `NO_LEGACY=1`, `DG_MEGA_MOE_IB_NVL_REF=1` | no DeepEP reference; the check compares against the stock kernel on a buffer whose ranks form one NVLink domain (single-domain hardware only) |
| `DEV_HDR_FRESH=1` | one header set per rank process. Required for a tree whose JIT source does not name the header set (every in-repo tree) |
| `DEV_REQUIRE_VERIFIED` (default 1) | refuse a variant that no node can prove was built from this request's headers |
| `DEV_MAX_CASES` | live (shape, tokens) cases before the ranks restart (default 3) |
| `DEV_WARM` | cases built and checked at start |
| `DEV_BASE_KNOBS` | variables set on every arm. The default list holds names that only earlier, out-of-repo kernels read; the in-repo kernel reads none of them |
| `DEV_PIN` (default 1) | give every NCCL Gin progress thread a core of its own during measurements |
| `DEV_STALL_DIAG=1` | host-side timing of every launch of a timing block |

## Requests

```bash
P=DG_MEGA_MOE_IBP_                       # plan overrides read by the host (get_mega_moe_ib_plan)
python devclient.py --cluster b300_2n_b ab --repo <checkout> --shape megaseek --tokens 1024 \
    --arm BASE= --arm CAP7=${P}L1_LEAD_CAP=7 --check 0 --rounds 5 --name cap7_a
python devclient.py --cluster b300_2n_b result <id> [--json]
```

An arm is `NAME=KNOBS`: environment variables set while that arm's kernels are compiled and launched (empty =
the host's own plan). A request compiles every (arm, size) once in the cluster, runs `--check` correctness
tests per arm and size against the reference, then times the arms interleaved per round (A B C A B C ...).
Each timed launch follows `bench_kineto`: a 256 MB L2 flush, `torch.cuda._sleep(2e7)`, a barrier, the launch; the time is
the kernel's CUPTI duration. `--warmup` (10) launches precede `--launches` (30) timed ones in every round.
Reported per arm and size: `median` (median over (rank, round) of the per-rank mean), `clean` (the same without
launch steps above 1.5x the median launch), `st/10k` (those steps per 10,000), the round medians, a check
verdict with the largest `calc_diff`, the code hash of the cubin and whether it was verified. `times.case_parts` in the raw
result gives the seconds rank 0 spent creating each new case, split into the IB buffer and the harness's own parts.

Shapes: `--shape flash` (384 experts, top-6), `--shape megaseek` (1216 experts, top-19), or a JSON object,
e.g. `'{"hidden": 5120, "inter": 2304, "experts": 1600, "topk": 25, "shared": 1}'` (the "GigaSeek" shape).
The expert count is the group's total and does not change with the GPU count.

Test-only arm variables handled by the server (not by the kernel):

| variable | meaning |
|---|---|
| `DG_MEGA_MOE_DEV_ALT_TOKENS=a_b_c` | test i of the check runs with the i-th token count of the list (cyclic) on the same warm buffer; `--tokens` is the largest count |
| `DG_MEGA_MOE_DEV_ALT_RANK=1` | shift the list position by the rank index, so ranks differ within a launch |
| `DG_MEGA_MOE_DEV_ALT_TIMING=1` | the timing launches cycle through the list as well (no reference run in between; its times are no benchmark) |
| `DG_MEGA_MOE_DEV_TIME_NVL=1` | time the single-domain reference buffer instead (needs `DG_MEGA_MOE_IB_NVL_REF=1`). It did not take effect on the one server it was tried on: accept a result only if its code hash differs from the IB arm's |
| `DG_MEGA_MOE_IB_ORACLE=2` | zero-network build. Must be alone in its request, with `--check 0`; the ranks restart before and after |
| `--trace` | trace build: the kernel's trace lines come back in the result (`DG_IB_TRACE ...`; the server also matches the prefix `OPUS_` of reference trees older than the rename) |

Rules the harness enforces: an arm that sets an allocation-time variable to another value than the server's is
refused; a zero-network arm cannot share a request; a failed check restarts the ranks; background compiles are
stopped during check and timing. A failure result carries the last log lines of every node and, where the container
may read the kernel log, its `Xid` / `NVRM` lines (the fault address of a device fault).

`python jobs/devserver.py selftest` runs the harness's own unit tests without a GPU.

## Stopping

```bash
python devclient.py --cluster b300_2n_b stop        # ends the dev server job; the runner takes the next job
python submit.py --cluster b300_2n_b --stop         # releases the nodes and disables the cluster for the watchdog
```

A held cluster costs its full price while idle. Stop it when a session ends.
