# Mega MoE across NVLink domains: reproduction

How every kind of number quoted for the IB kernel is produced, on which hardware and software, and what a
tree must pass before its numbers count. The harness is in `tools/mega_moe_ib/` (its `README.md` has the
bring-up). Design: `ARCHITECTURE.md`. Open items: `TODO.md`.

The tree described is tag `fused-ib-base-v2.0` (commit `adac520`), called v2.0 below; v1.1 and v1.2 are the
tags `fused-ib-base-v1.1` (`7863474`) and `fused-ib-base-v1.2` (`6ebf8b4`).

All times are microseconds per kernel launch. "EP16" and "EP32" mean 16 and 32 GPUs (2 and 4 nodes of 8).

## 1. Hardware

| name used here | nodes | GPUs | NIC per GPU | where the kernel stands |
|---|---|---|---|---|
| B300 2 x 8 | 2 bare-metal nodes | 16 B300 | ConnectX-8, 800G InfiniBand | all gates run; fitted |
| B300 4 x 8 | 4 bare-metal nodes | 32 B300 | ConnectX-8, 800G InfiniBand | all gates run; fitted |
| GB200 4 x 8 | 8 trays of 4 GPUs in one rack, grouped as 4 logical nodes of 8 (`DG_MEGA_MOE_IB_VIRTUAL_LSA=8`) | 32 GB200 | ConnectX-7, 400G NDR InfiniBand, one rail per GPU | gated loads; gates run; not tuned (section 9) |

The B300 numbers are in section 8, the GB200 numbers in section 9. A B300 node here has 8 GPUs in one NVLink domain and about 115 host
CPUs, of which a container gets 64. The CPU count matters: with a Gin backend that uses host progress threads,
each of those threads needs a core of its own during a measurement (the harness pins them; `DEV_PIN`).

## 2. Software

| component | version | source |
|---|---|---|
| base image | `nvidia/cuda:13.2.0-devel-ubuntu24.04` | `tools/mega_moe_ib/runner.py` |
| Python | 3.12 | same |
| PyTorch | 2.14.1, cu130 wheels | same |
| NCCL | `nvidia-nccl-cu13==2.32.3` (the kernel uses the Gin device API; `comm/gin.cuh` depends on 2.32 headers) | `runner.py`, `jobs/devserver.sh` |
| host compiler | gcc / g++ 13 (C++20 `<format>`) | `jobs/setup_deps.sh` |
| DeepGEMM base | upstream `5081f85`; the tree under test is applied as a diff against it | `jobs/setup_deps.sh`, `jobs/devserver.sh` |
| reference | DeepEP `f6aa38f` (dispatch / combine) + grouped FP8xFP4 GEMMs + tilelang 0.1.15 SwiGLU | `jobs/setup_deps.sh`, `jobs/devserver.sh` |
| NCCL Gin backend | 3 (GDAKI) on the InfiniBand clusters | `DG_MEGA_MOE_IB_GIN_TYPE=3` |

Shapes (hidden 5120, intermediate 2304, one shared expert, FP8 x FP4):

| name | experts (whole group) | top-k |
|---|---|---|
| Flash | 384 | 6 |
| MegaSeek | 1,216 | 19 |
| GigaSeek | 1,600 | 25 |

The expert count is the group's total at every GPU count (76 MegaSeek experts per rank at EP16, 38 at EP32).

## 3. Build

The IB kernel is JIT-compiled like every DeepGEMM kernel; the host extension is built as usual:

```bash
git submodule update --init --recursive
DG_FORCE_BUILD=1 python setup.py bdist_wheel && pip install --no-deps --force-reinstall dist/*.whl
```

What an IB run needs beyond a stock build:

- Development switches (plan overrides, trace, zero network, compile-only) act only with `DG_MEGA_MOE_DEV=1`
  in the process and are refused by name without it. The dev server sets it: `jobs/devserver.sh` exports
  `DG_MEGA_MOE_DEV=1` by default (`--env DG_MEGA_MOE_DEV=0` overrides; it is an allocation-time variable
  there, so an arm cannot change it). A plain
  run of the kernel needs none of them.
- `DG_MEGA_MOE_IB=1` in the environment of every rank: Mega MoE buffers are then allocated as one NCCL
  window with the IB tail (`deep_gemm/mega/ib.py`).
- NCCL 2.32.3 headers and library from the `nvidia-nccl-cu13` wheel. `ib.py` finds them through the `nvidia`
  package, builds a small extension with `torch.utils.cpp_extension.load_inline` (needs the C++ toolchain at
  run time) and sets `DG_NCCL_INCLUDE_DIR` for the JIT. If another DeepGEMM kernel is compiled in the process
  before the first IB buffer is allocated, export `DG_NCCL_INCLUDE_DIR=<nccl>/include` at process start.
- `LD_LIBRARY_PATH` must put that NCCL first (`jobs/setup_deps.sh`).

## 4. The in-repo test

`tests/test_mega_moe.py` runs the IB kernel when `DG_MEGA_MOE_IB=1`. It needs no development switch. Three
things must be right, each found by a failed first attempt or by reading the test:

1. **`DG_NCCL_INCLUDE_DIR`** must point at the include directory of the NCCL in use, because the JIT compiles
   kernels that include `<nccl.h>`. Without it every launch across NVLink domains fails at its first compile
   with "nccl.h: No such file or directory".
2. **`DG_MEGA_MOE_IB_TOLERANCE`** (for example `1e-3`, a bound on `calc_diff`) must be set. Without it the test
   asserts bit equality with the reference, and a remote node's contribution crosses the wire as an fp8
   record. It is a variable of the test, not a host switch.
3. **A cold first run can fail in the reference, not in the kernel.** The test gives its DeepEP reference 10 s
   to meet its peers while every rank compiles the reference's kernels at first use. Run it again; the JIT
   caches are warm then.

The plain path pins no NCCL progress threads, so its timings are an upper bound of what section 8 reports.

Command lines, as run by `tools/mega_moe_ib/jobs/inrepo_test.sh` on every node of a cluster (`WORLD_SIZE` =
nodes, `RANK` = node index, `MASTER_ADDR` / `MASTER_PORT` as usual for `deep_gemm/utils/dist.py`):

```bash
export DG_NCCL_INCLUDE_DIR=<nccl>/include DG_MEGA_MOE_IB=1 DG_MEGA_MOE_IB_TOLERANCE=1e-3
export WORLD_SIZE=$NNODES RANK=$NODE_RANK
# Flash: --num-experts 384 --num-topk 6      MegaSeek: --num-experts 1216 --num-topk 19
python tests/test_mega_moe.py --num-processes 8 --hidden 5120 --intermediate-hidden 2304 \
    --num-experts 1216 --num-topk 19 --num-max-tokens-per-rank 1024 --num-tokens 1024 --num-shared-experts 1 \
    --mma-type fp8xfp4 --num-correctness-tests 20
# once, to see the plan and the link rate it was planned for:
DG_PRINT_CONFIGS=1 python tests/test_mega_moe.py ... --num-correctness-tests 1
```

Through the harness (builds DeepEP, tilelang and the tree as a patch on upstream; needs no dev server on the
cluster):

```bash
git -C <checkout> diff 5081f85 fused-ib-base-v2.0 > jobs/tree.patch
python submit.py --cluster <2-node cluster> --name inrepo --script jobs/inrepo_test.sh \
    --file jobs/setup_deps.sh --file jobs/nccl_topo.py --file jobs/tree.patch --env PATCH=tree.patch --timeout 3600
```

Result on v2.0, 2 nodes x 8 B300, plain environment (no `DG_MEGA_MOE_DEV`), 14 Nov 2026: **pass** for Flash
128 / 1024 and MegaSeek 128 / 1024, 20 correctness tests each, largest `calc_diff` 1.69e-4 to 1.77e-4. The
test's own timing read 152 / 292 / 312 / 697 us, in line with the 154 / 294 / 312 / 698 of the harness on the
same hardware. Every rank printed `MegaMoEIBPlan(planned for 800 Gb/s per GPU)`: the link-rate reader works in
the B300 container. The same job on 4 nodes x 8 B300 (32 ranks):
**pass** for the same four configurations, largest `calc_diff` 2.48e-4 to 2.55e-4, timing 153 / 397 / 240 / 681 us
(Flash 128 / 1024, MegaSeek 128 / 1024; 153 / 393 / 232 / 672 in section 8).

A single-node form exists for hardware where all ranks share one NVLink domain: `DG_MEGA_MOE_IB_VIRTUAL_LSA=4`
splits 8 GPUs into two virtual domains and `DG_MEGA_MOE_IB_NVL_REF=1` makes the reference the same kernel on a
second buffer whose ranks form one domain. The test file itself was not run in that form (not measured); the
harness's copy of the test's set-up was, on GB200.

## 5. How each kind of number is produced

All of it goes through `devclient.py ab` against a dev server that holds the ranks, their buffers and one tree
warm. A request names arms (environment variables per arm), one shape, token counts, a number of correctness
tests and a number of timing rounds.

**Definition of a time.** Per launch: 256 MB L2 flush, `torch.cuda._sleep(2e7)`, a group barrier, the launch;
the time is the kernel's CUPTI duration. Per round and arm: 10 warm-up launches, then 30 timed. Reported:
`median` = median over (rank, round) of the per-rank mean; `clean` = the same without launch steps (same
round, same index, all ranks) whose slowest rank is above 1.5x the median launch. Slow steps of +150 to +310 us
on 17 to 32 ranks occur at a low rate with any kernel on these servers; they are counted (`st/10k`) and the
clean median is the number quoted. A first launch on fresh buffers runs at about 5 ms; the server burns 100
launches in per case before it times anything.

**Rules.**

1. **Diagnostics check first and alone.** A new build is checked with diagnostics on (arm
   `DG_MEGA_MOE_IBP_DIAGNOSTICS=1`), `--check 300`, at a size and its odd neighbour, one arm per request,
   before any timing of it counts. A size and its odd neighbour run the same cubin; the odd count exercises
   the re-arm tail (invariant I5).
2. **Back-to-back diagnostics phase.** The check loop runs a reference between launches. The timing phase of
   that same request (`--rounds 1` or more) launches the diagnostics build back to back; a trap or a device
   message there fails the gate even if the check passed.
3. **Same-request arms.** A difference is quoted only between arms of one request: the level of one cubin
   moves by about 3 us (EP32) and up to 13 us (EP16 GigaSeek 1024) between requests. Timing requests carry
   several arms, `--check 0`, one size, 5 or 7 rounds.
4. **Both orders.** Every pair is run as two requests with the arm order reversed (`A B` and `B A`); both
   deltas are reported.
5. **Twins.** The base arm is given twice under two names. The twins' difference is the noise floor of that
   request; a delta inside it is zero. Deltas of 10 us or less at GigaSeek 1024 also need a size-matched
   control arm.
6. **Soak.** `--check 3000` at a size and its odd neighbour, per regime.
7. **Mixed-count lists.** Arm variables `DG_MEGA_MOE_DEV_ALT_TOKENS=a_b_c,DG_MEGA_MOE_DEV_ALT_RANK=1`: test i
   runs with the i-th count of the list, shifted by the rank index, on one warm buffer; `--tokens` is the
   largest count. This is what exercises counts that fall and rise between launches, zero-token sources and
   ranks in different regimes in one launch.
8. **Zero-network pair.** Arm `DG_MEGA_MOE_IB_ORACLE=2`, alone in its request, `--check 0`: after two
   launches over the real network the kernel puts nothing and re-arms nothing, so every poll finds its data.
   The server gives such a request fresh ranks before and after. Paired with a real-network request of the
   same tree and size, `real - zero network` is the network exposure. The inputs must be identical every
   launch; the outputs are not checked.
9. **Trace.** `--trace` compiles the trace build and returns the kernel's stage stamps
   (`impls/mega_moe_ib/trace.cuh`). The trace build is about 5 us slower; stamps are compared between traced
   arms only.
10. **Prediction before the run.** Not a harness rule, but the rule the numbers were collected under: the
    expected delta and its mechanism are written down before a timing is submitted, and a miss gets a traced
    cause before the item is closed.

**The host's plan.** Ordered iff `remote nodes x tokens x hidden / link rate >= 26.2 us`: from 512 tokens at
2 x 8 and 171 at 4 x 8 on 800G; from 256 and 86 on 400G, where the plan is also gated. `DG_PRINT_CONFIGS=1`
prints each plan with the rate it was planned for.

**Forcing a regime.** `DG_MEGA_MOE_IBP_ORDERED=0|1` sets only the regime. The tail policy is chosen before
the overrides, from the regime the host would have picked, so "Ordered with the host's policy" below the
boundary is spelled out: `DG_MEGA_MOE_IBP_ORDERED=1,DG_MEGA_MOE_IBP_L1_LEAD_CAP=9` and, with more than one
remote node, `DG_MEGA_MOE_IBP_RETIRE_PERCENT=50,DG_MEGA_MOE_IBP_RETIRE_AT_PERCENT=<90|92|71>,DG_MEGA_MOE_IBP_SHARED_L2_EARLY=1`
(top-19 / 25 / 6 on four nodes).

**Two trees.** `--repo` sends the headers of one tree; all arms of the request share them. Changes that are
plan fields need no second tree. For two trees there are two cases:

- *The host forwards arm variables into the JIT source as defines* (hosts built from the trees that preceded
  this one did; that is how v1.1 and v1.2 were timed against the reference kernel in section 8, through a
  compatibility layer those tags carried and v2.0 no longer has). Build one
  combined header tree with `tools/mega_moe_ib/mkab.py OUT exp=<tree>:DG_MEGA_MOE_EXP base=<tag>`: identical
  headers are copied once, differing ones become a dispatcher on the arm's define. The arms then run
  interleaved in one request. Check that an arm's code hash from the combined tree is stable across requests;
  it is not comparable with the hash from the arm's own tree, because the hash covers strings that hold
  include paths.
- *The in-repo host* prints only the plan into the JIT source, so a define cannot select a tree, and the JIT
  keys a kernel by its source text and by include hashes it caches per process. Two trees are therefore two
  requests, each on fresh ranks: start the server with `DEV_HDR_FRESH=1` (a request whose header set differs
  from the live ranks' gets new rank processes and an empty JIT cache first) and keep `DEV_REQUIRE_VERIFIED=1`
  (a variant that no node can prove was built from the request's headers is refused; without it a
  header-only change silently ran the base cubin once on 13 Nov). Quote such a pair with the between-request
  drift in mind: run each tree twice, alternating, and use twins.

**Harness switches these rules rely on** (all in `tools/mega_moe_ib/jobs/devserver.py`; added on 13 Nov 2026):

| switch | what it does | state |
|---|---|---|
| `DEV_HDR_FRESH=1` | one header set per rank process, own JIT cache per generation, no warm case | used for every in-repo-host number |
| `DEV_REQUIRE_VERIFIED` (default 1) | refuses unverified variants | on |
| `DEV_MAX_CASES` | live (shape, tokens) cases before the ranks restart; 8 was used on the in-repo-host servers, the default is 3 | each case holds its own communicator and progress threads |
| `DG_MEGA_MOE_DEV_ALT_TIMING=1` with `DG_MEGA_MOE_DEV_ALT_TOKENS` (and `_ALT_RANK=1`) | the timing launches cycle through the token list, shifted per rank: back-to-back launches with different counts per rank | written, selftest only; not run on a server |
| `DG_MEGA_MOE_DEV_TIME_NVL=1` | times the single-domain reference buffer instead of the IB buffer | did NOT take effect on the one server it was tried on (GB200, 32 GPUs: same code hashes and times as the IB arm). Treat a result as valid only if its code hash differs from the IB arm's |
| case timer | a result's `times.case_parts` gives rank 0's seconds per new case: the IB buffer alone, then the harness's reference buffers, weights and inputs | 32 ranks, top-19: 14.4 s for the first IB buffer of a process, 3.3 to 3.7 s for each later one |
| fault-address capture | a failure result carries the node's `Xid` / `NVRM` kernel-log lines (`logs.<node>.xid`), where the container may read the kernel log | installed; whether it has captured a fault address: not measured |

## 6. Gate list

A tree's numbers count only if it passes all of these. "diag" = diagnostics on; every request must report
`verified = True` for every variant.

| # | gate | what runs |
|---|---|---|
| 1 | IB off = stock PTX | front-end PTX of the IB-off kernel, the tree against upstream `5081f85`, two shapes; normalised md5 equal (below) |
| 2 | diag check, every tile class the plan can select | `--check 300`, each size with its odd neighbour, three shapes. EP32: Scan 128, 16, 40, 96, 240; Ordered 256, 512, 1024, 768, 384, 464 (Flash without 240 and 464). EP16: Scan 128, 256, 64, 16; Ordered 512, 1024, 768, 928 |
| 3 | changing-count lists, per-rank shift, diag, `--check 300` | Scan `128_120_144_127_136_121`; zero-token `20_0_17_1_0_13`; Ordered `512_490_640_511_600_483`; mixed-regime and crossing: `128_600`, `0_1024`, `8_600`, `0_0_0_600`, `200_300` (EP32), `600_128_1024_0_300_511` |
| 4 | soak | `--check 3000`: three shapes Ordered 512 / 511 and 1024 / 1023; Scan 128 / 127 |
| 5 | back-to-back diagnostics | timing phase of the diag builds: three shapes, Scan 128 / 127, Ordered 512 / 511, 1024 / 1023; no trap, no alert, no device message |
| 6 | timing against the previous base | same request, both orders, twins, at 128, 512, 1024 on three shapes at both GPU counts: equal or better, or the miss explained in writing |
| 7 | host | first IB launch of a buffer (cold JIT cache); an unknown `DG_MEGA_MOE_IBP_*` name is refused with its name; a development switch without `DG_MEGA_MOE_DEV=1` is refused with its name; a plan whose backend differs from the communicator's is refused; on a 400G node the host's plan is the gated one |
| 8 | experts per rank (Ordered, EP16) | diag check at 12, 24, 31, 32, 33, 38, 50, 76 experts per rank |

Gate 1 is local (front end only; needs nvcc, no GPU):

```bash
UPSTREAM=<upstream checkout at 5081f85> REPO=<tree under test> NVIDIA=<site-packages/nvidia> tools/mega_moe_ib/gate1.sh
```

It instantiates the IB-off kernel for two shapes at 8 ranks (304 experts top-19; 96 experts top-6; 1,024
tokens), compiles each to PTX against both include trees, normalises and compares md5. The normalisation
removes debug lines, string constants that hold include paths, and compiler-generated internal names; these
names carry the compiler's process id in a length-prefixed form, and without that step an unchanged tree
changes md5 when the process id gains a digit. With the CUDA 13.2 toolkit wheels both trees read
`f850b8e3b441` (top-19) and `d2fca7b8bc4e` (top-6) at this commit; the values depend on the compiler, their
equality does not. nvcc may exit non-zero after writing the PTX if its assembler rejects the PTX version; the
script judges by the PTX text.

### Gate results of v2.0

14 Nov 2026, in-repo host, host's own plan and tile rule throughout, every variant verified. 137 B300 result
files and the GB200 results (section 9) carried no kernel message: no landing, flag or header timeout, no
return-order line, no early-hint line; nothing trapped or hung. Gate 1 passes (`f850b8e3b441` /
`d2fca7b8bc4e`). The runs are spread over v2.0 itself and earlier states of the same commit stack, whose kernel
headers differ from v2.0's in comments only; the 2 x 8 idle lists, the 2 x 8 soaks and the in-repo test
ran on v2.0 itself.

B300, 4 x 8 (two servers: A for MegaSeek, B for GigaSeek and Flash) and 2 x 8 (one server):

| gate | 4 x 8 | 2 x 8 |
|---|---|---|
| diagnostics checks, every tile class of both regimes | MegaSeek 24 of 24 and GigaSeek 24 of 24 at check 300 (128 / 127, 16 / 15, 40 / 39, 96 / 95, 170 / 169, 171, 192 / 191, 224 / 223, 240, 256 / 255, 512 / 511, 1024 / 1023, 768 / 767). **Flash: 16 sizes at check 300 and 16 more at check 100**, all pass | every tile class of both regimes on three shapes: pass |
| changing-count, mixed-regime, crossing and idle lists, per-rank shift, check 300 | MegaSeek 7 of 7 (`170_171`, `160_171_200_150_192_128`, `0_171_170_0`, `600_128_1024_0_300_511`, `0_1024`, `512_490_640_511_600_483`, `128_120_144_127_136_121`); GigaSeek 2 of 2; Flash 2 of 2 | MegaSeek 7 of 7 (the same with `511_512`, `480_512_600_500_511_530`, `0_512_511_0` across the 2 x 8 boundary); GigaSeek 2 of 2; Flash 2 of 2 |
| idle-rank tile: lists `1024_0`, `512_0`, `0_0_0_600`, `600_128_1024_0_300_511` (and `8_1024` at 4 x 8), diagnostics, in the check and back to back | three shapes, check 100: pass 15 of 15; crossing lists `170_171` and `128_600` with the host choosing per rank: pass 6 of 6 | three shapes, check 300: pass 12 of 12 |
| back-to-back diagnostics, equal counts (5 rounds) | 128 / 127 / 192 / 191 / 512 / 511 / 1024 / 1023, three shapes: clean | clean |
| back-to-back diagnostics, different counts per rank | two lists per shape (one crossing 170 \| 171): clean | two lists per shape (one crossing 511 \| 512): clean |
| soak 3000 | MegaSeek 512 / 511, 1024 / 1023, 192 / 191, 224 / 223, 128 / 127: 10 of 10; GigaSeek and Flash 512 / 511, 1024 / 1023, 192 / 191, 224 / 223: 16 of 16 | MegaSeek 512 / 511, 1024 / 1023, 128 / 127; GigaSeek and Flash 512 / 511, 1024 / 1023: 14 of 14 |
| host | the plan by code hash: 192 tokens = Ordered with policy, 170 = Scan, not gated | plan line printed on every rank: planned for 800 Gb/s per GPU |

Slow launch steps in the back-to-back phases (0 to 15 of 150 per size) fell on whatever size was running when
a server had a stall episode; the same request on the other 4-node server moved them to other sizes.

Gate 6, timing against v1.2. On the in-repo host two trees cannot share a request, so each side is its own
request with twins (within 0.3), in the order v2.0, v1.2, v2.0, 5 rounds; the difference uses the mean of the
two v2.0 runs. v2.0 minus v1.2 at 128 / 512 / 1024 tokens:

| shape | 4 x 8 | 2 x 8 |
|---|---|---|
| MegaSeek | +0.3 / -1.4 / -1.7 | -0.7 / -1.3 / -2.1 |
| GigaSeek | +0.5 / -0.3 / +0.5 | -0.4 / -1.6 / -2.7 |
| Flash | +0.9 / +2.6 / -0.9 | -1.7 / -0.8 / -1.2 |

The Flash +2.6 at 512 is drift between requests: the first v2.0 run read 238.0, the second 232.8, v1.2 232.8.
A repeat of the MegaSeek pair at 2 x 8 on v2.0 itself read -1.3 / -0.9 / -0.6.

Where v2.0 changes the plan on 4 nodes (Ordered with policy instead of Scan at 171 to 255 tokens), host's plan
minus v1.2's Scan plan, same request, twins, at 176 / 192 / 224 tokens: MegaSeek -0.5 / -4.1 / -9.9; GigaSeek
-0.3 / -3.2 / -7.6; Flash -5.9 / -7.7 / -11.0.

Not run: back-to-back or soak runs with `gated` on at 800G; the in-repo test at 4 nodes (section 4).

## 7. Request lines for the headline points

`$T` is the checkout of the tree the server was started on (or a tree that differs from it in headers only).
The empty arm is the host's own plan. Each line stands for two requests, with the arm order reversed in the
second.

```bash
C32=<a B300 4 x 8 cluster>; C16=<a B300 2 x 8 cluster>
G='{"hidden": 5120, "inter": 2304, "experts": 1600, "topk": 25, "shared": 1}'
D=DG_MEGA_MOE_IBP_DIAGNOSTICS=1
AB="python devclient.py --cluster"

# 1. check first, alone (one request per shape and size pair)
$AB $C32 ab --repo $T --shape megaseek --arm D=$D --tokens 512,511   --check 300 --rounds 1 --name c_m512
$AB $C32 ab --repo $T --shape megaseek --arm D=$D --tokens 1024,1023 --check 300 --rounds 1 --name c_m1024
$AB $C32 ab --repo $T --shape megaseek --arm D=$D --tokens 128,127   --check 300 --rounds 1 --name c_m128

# 2. headline timings: one size per request, twins, 5 rounds
for s in 128 512 1024; do
  $AB $C32 ab --repo $T --shape megaseek --arm P= --arm P2= --tokens $s --check 0 --rounds 5 --name t_m$s
  $AB $C32 ab --repo $T --shape "$G"     --arm P= --arm P2= --tokens $s --check 0 --rounds 5 --name t_g$s
  $AB $C32 ab --repo $T --shape flash    --arm P= --arm P2= --tokens $s --check 0 --rounds 5 --name t_f$s
done
# the same nine on $C16

# 3. zero network (own request each; the ranks restart before and after)
$AB $C32 ab --repo $T --shape megaseek --arm Z=DG_MEGA_MOE_IB_ORACLE=2 --tokens 1024 --check 0 --rounds 5 --name z_m1024

# 4. soak
$AB $C32 ab --repo $T --shape megaseek --arm P= --tokens 512,511 --check 3000 --rounds 1 --name s_m512

# 5. lists
L=DG_MEGA_MOE_DEV_ALT_TOKENS=600_128_1024_0_300_511,DG_MEGA_MOE_DEV_ALT_RANK=1
$AB $C32 ab --repo $T --shape megaseek --arm L=$D,$L --tokens 1024 --check 300 --rounds 1 --name l_m

# 6. a plan-field A/B (example: the Ordered boundary at 192 tokens, EP32)
O=DG_MEGA_MOE_IBP_ORDERED=1,DG_MEGA_MOE_IBP_L1_LEAD_CAP=9,DG_MEGA_MOE_IBP_RETIRE_PERCENT=50,DG_MEGA_MOE_IBP_RETIRE_AT_PERCENT=90,DG_MEGA_MOE_IBP_SHARED_L2_EARLY=1
$AB $C32 ab --repo $T --shape megaseek --arm S= --arm O=$O --arm O2=$O --arm S2= --tokens 192 --check 0 --rounds 7 --name b_m192
```

A request with `--check` greater than 0 carries one arm. Start the server with `DEV_HDR_FRESH=1`
(`DG_MEGA_MOE_DEV=1` is the job script's default). An arm that moves a buffer to another kernel family inside a live process (for example a
tile override) needs fresh ranks: otherwise ranks meet the new kernels in different calls, the first-launch
rendezvous cannot line them up, and the request ends in a header timeout that is the harness's doing, not the
kernel's.

## 8. Expected numbers: B300

The data set is frozen at v1.2 (`6ebf8b4`; v1.1 is `7863474`, and v1.2 is v1.1 plus the regime bit and the
blank-header fix). v2.0 changes the plan, and with it the kernel a point runs, only at 171 to 255 tokens on 4
nodes at 800G, where it selects the Ordered regime; no row below is in that range. Its kernels differ from
v1.2's by the position of one barrier in shared memory, and timed against v1.2 at 128 / 512 / 1024 tokens it
reads within -2.7 to +0.9 us on all three shapes at both GPU counts, apart from one drifted Flash reading
(section 6), so the rows stand for v2.0.

How the rows were measured: 13 Nov 2026 (16 GPUs: into 14 Nov, 00:15 UTC). One request per point and arm
order, four arms in one request (reference | reference with the assert removed | v1.1 | v1.2) and reversed,
one token count per request, 5 rounds x 30 launches x all ranks, `--check 0`; each cell is order a / order b.
Each arm passed a 30-test check against the DeepEP reference in its own request. All four arms come from one
combined header tree on servers whose host forwards arm defines; v1.1 and v1.2 ran the plan their own host
prints for the point (32 GPUs: Scan at 128, Ordered with lead cap 9, retire 50 % and shared L2 early from 256;
16 GPUs: Scan at 128 and 256, Ordered with lead cap 9 from 512).

Columns:

- **reference kernel**: the kernel that preceded this work's rewrite, with its per-point switch strings.
- **reference, assert removed**: the same with one debug assert removed from its symmetric-pointer mapping
  (two run-time divisions and a trap path at every NVLink store and pull). This is the fair reference: the
  assert alone accounts for up to 28 us of the reference kernel's time.
- **GB200 NVLink**: the stock kernel on GB200 with all 16 or 32 GPUs in one NVLink domain, measured on 5 and 6
  Nov 2026. It is another machine. Taken at face value, with one caveat: with the network switched off, the
  top-19 kernel reads 198.0 / 359.6 / 590.1 us on B300 and 199.8 / 398.9 / 652.9 us on GB200 at 128 / 512 /
  1024 tokens, so GB200 has about 10 % less compute at 512 to 1024 tokens and none at 128. The same pair was
  not measured for the other two shapes. No scaled baseline is used anywhere. 768 tokens were not measured on
  NVLink.

### 32 GPUs: 4 nodes x 8 B300, 800G (4-node server A)

| shape | tokens | reference kernel | reference, assert removed | v1.1 | v1.2 | v1.2 - reference (assert removed) | GB200 NVLink | v1.2 - NVLink |
|---|---|---|---|---|---|---|---|---|
| MegaSeek | 128 | 246.2 / 246.2 | 245.1 / 245.3 | 231.7 / 232.0 | 232.1 / 232.1 | -13.1 | 206 | +26.1 |
| MegaSeek | 256 | 318.6 / 318.7 | 317.9 / 318.1 | 295.2 / 295.5 | 294.7 / 294.9 | -23.2 | 257 | +37.8 |
| MegaSeek | 512 | 460.4 / 460.2 | 453.9 / 454.0 | 427.1 / 427.1 | 426.8 / 427.1 | -27.0 | 407 | +19.9 |
| MegaSeek | 768 | 594.0 / 594.3 | 573.0 / 572.7 | 542.0 / 541.5 | 541.6 / 541.5 | -31.3 | - | - |
| MegaSeek | 1024 | 723.8 / 723.7 | 710.9 / 711.2 | 671.7 / 671.8 | 672.3 / 672.1 | -38.8 | 652 | +20.2 |
| GigaSeek | 128 | 289.6 / 289.7 | 287.9 / 288.1 | 273.7 / 273.5 | 273.6 / 273.7 | -14.4 | 258 | +15.6 |
| GigaSeek | 256 | 365.8 / 365.1 | 363.6 / 362.7 | 336.5 / 335.9 | 336.8 / 335.8 | -26.9 | 326 | +10.3 |
| GigaSeek | 512 | 552.6 / 552.7 | 548.0 / 548.0 | 515.9 / 515.9 | 515.8 / 515.8 | -32.2 | 528.5 | -12.7 |
| GigaSeek | 768 | 735.5 / 735.6 | 707.7 / 707.6 | 664.6 / 665.2 | 665.1 / 665.5 | -42.4 | - | - |
| GigaSeek | 1024 | 899.7 / 899.8 | 879.0 / 879.9 | 832.0 / 831.8 | 832.2 / 832.1 | -47.3 | 835 | -2.9 |
| Flash | 128 | 158.3 / 157.8 | 157.4 / 157.3 | 152.8 / 153.1 | 153.1 / 153.2 | -4.2 | 92 | +61.2 |
| Flash | 256 | 212.3 / 212.6 | 211.5 / 211.2 | 179.6 / 179.5 | 179.3 / 179.2 | -32.1 | 112 | +67.2 |
| Flash | 512 | 278.1 / 278.0 | 274.6 / 274.7 | 241.6 / 241.6 | 241.3 / 241.4 | -33.3 | 165 | +76.3 |
| Flash | 768 | 346.2 / 346.3 | 343.9 / 343.6 | 310.7 / 310.6 | 310.7 / 310.5 | -33.2 | - | - |
| Flash | 1024 | 430.5 / 430.3 | 426.4 / 426.2 | 393.6 / 393.6 | 393.2 / 393.0 | -33.1 | 238 | +155.1 |

### 16 GPUs: 2 nodes x 8 B300, 800G (2-node server)

| shape | tokens | reference kernel | reference, assert removed | v1.1 | v1.2 | v1.2 - reference (assert removed) | GB200 NVLink | v1.2 - NVLink |
|---|---|---|---|---|---|---|---|---|
| MegaSeek | 128 | 321.0 / 321.3 | 319.9 / 319.9 | 311.1 / 311.5 | 311.8 / 311.9 | -8.0 | 291.0 | +20.9 |
| MegaSeek | 256 | 361.8 / 361.5 | 359.1 / 358.9 | 347.9 / 348.0 | 348.8 / 348.9 | -10.2 | 337 | +11.8 |
| MegaSeek | 512 | 453.9 / 454.3 | 450.3 / 450.2 | 422.6 / 422.5 | 422.6 / 422.5 | -27.7 | 430.5 | -8.0 |
| MegaSeek | 768 | 562.7 / 562.6 | 552.3 / 552.7 | 516.6 / 517.2 | 516.7 / 516.5 | -35.9 | - | - |
| MegaSeek | 1024 | 743.8 / 743.5 | 740.7 / 740.0 | 697.2 / 697.6 | 697.3 / 697.6 | -42.9 | 721.5 | -24.1 |
| GigaSeek | 128 | 399.2 / 399.1 | 397.4 / 397.4 | 386.6 / 386.3 | 385.8 / 385.9 | -11.6 | 365.5 | +20.3 |
| GigaSeek | 256 | 451.3 / 453.9 | 448.2 / 449.7 | 434.1 / 433.4 | 433.1 / 433.0 | -15.9 | 425.5 (h) | +7.6 (h) |
| GigaSeek | 512 | 568.3 / 568.4 | 562.9 / 563.0 | 531.0 / 534.4 | 530.1 / 534.9 | -30.5 | 548 (h) | -15.5 (h) |
| GigaSeek | 768 | 720.0 / 721.4 | 713.1 / 713.3 | 657.3 / 657.5 | 656.9 / 656.4 | -56.6 | - | - |
| GigaSeek | 1024 | 951.5 / 951.2 | 949.1 / 948.8 | 903.0 / 903.2 | 902.9 / 904.0 | -45.5 | 925.0 | -21.5 |
| Flash | 128 | 158.3 / 160.9 | 156.8 / 159.9 | 152.6 / 155.0 | 153.2 / 155.5 | -4.0 | 126.0 | +28.4 |
| Flash | 256 | 181.4 / 181.7 | 180.1 / 180.0 | 169.2 / 169.5 | 169.5 / 169.6 | -10.5 | 142 | +27.5 |
| Flash | 512 | 228.9 / 229.1 | 228.4 / 228.5 | 206.1 / 206.1 | 205.6 / 205.5 | -22.9 | 173 | +32.5 |
| Flash | 768 | 282.7 / 285.4 | 276.1 / 278.5 | 241.6 / 243.8 | 241.9 / 243.7 | -34.6 | - | - |
| Flash | 1024 | 340.5 / 342.0 | 340.7 / 341.8 | 293.1 / 295.3 | 293.4 / 294.7 | -47.1 | 268.5 | +25.6 |

(h): the bar is from the same NVLink sweep but was not in the re-benchmark's own table; the difference is
computed here from the mean of the two orders. Flash 1024: the sweep log reads 269 where the table has 268.5.

Caveats of these rows:

- Levels differ by 3 to 9 us between servers and move by about 3 us between requests on one server (v1.1 at
  top-19, 32 GPUs, 1024 tokens read 678 in six earlier requests the same day and 671.7 here). Only differences
  inside a row count.
- GigaSeek at 16 GPUs, 512 tokens: orders a and b differ by 3 to 5 us in the two new arms and not in the
  reference arms; not explained. GigaSeek 256 and Flash 128 / 768 / 1024 at 16 GPUs: order b reads 2 to 3 us
  above order a in every arm.
- The Flash row at 32 GPUs, 512 tokens, replaces two earlier requests that read every arm 5 to 10 us high
  while other compiles ran on the server.
- 1,920 tokens: not measured.

Other reference values (top-19 and top-25, 32 GPUs, 13 Nov 2026, v1.1, 4-node server A, real and zero-network
from the same hour):

| point | real | zero network | exposure | stock kernel, one B300 node, equal per-GPU work (12 Nov) |
|---|---|---|---|---|
| MegaSeek 512 | 426.1 | 359.6 | 67 | 334.2 |
| MegaSeek 1024 | 678.0 | 590.1 | 88 | 549.0 |
| GigaSeek 512 | 515.1 | 457.7 | 57 | 429.5 |
| GigaSeek 1024 | 832.6 | 754.4 | 78 | 720.4 |
| MegaSeek 128 (4-node server B) | 232.5 to 233.8 | 198.0 | 35 | 184.5 |
| Flash 256 / 512 / 1024 | 179.1 / 241.1 / 393.4 | 120.4 / 167.1 / 245.0 | 58.7 / 74.0 / 148.4 | not measured / not measured / 202.3 |

Stage account of one point (MegaSeek, 32 GPUs, 128 tokens, traced pair, 13 Nov): of 38 us of network exposure
in the traced build, 17 are at the start (the counts wait for the header), 10 on the return wire after the
last put, 6 after the last stamp (mirror re-arm and exit), and under 5 in the pull and GEMM phases together.

Idle ranks. Lists cycle through the timing launches with a per-rank shift, so in every launch some ranks run
each count; timed twins, 5 rounds, 14 Nov 2026. "Before" is every rank's own tile (v1.2's kernels).

| list | 4 x 8: MegaSeek / GigaSeek / Flash | before | 2 x 8 on v2.0: MegaSeek / GigaSeek / Flash |
|---|---|---|---|
| `1024_0` | 560.4 / 683.2 / 412.5 | 1,321.8 / 1,707.0 / 486.4 | 484.1 / 611.2 / 246.8 |
| `512_0` | 341.9 / 406.0 / 230.6 | 718.0 / 920.5 / 295.6 | 403.2 / 507.4 / 195.5 |
| `0_0_0_600` | 347.4 / 421.6 / 234.0 | 494.2 (MegaSeek) | 415.0 / 523.1 / 200.4 |
| `600_128_1024_0_300_511` | 463.7 / 566.8 / 312.5 | 1,118.3 / 1,460.7 / 415.2 | 464.2 / 588.0 / 222.5 |
| `8_1024` (not covered by the rule) | 1,333.3 / 1,730.1 / 488.0 | 1,331.5 (MegaSeek) | not measured |

The 4 x 8 column was measured on an earlier state of v2.0's commit stack (same kernels and tile rule). At
2 x 8 the six-entry list read 1,147.0 / 1,489.6 / 414.8 before the rule (diagnostics builds, back to back).
Equal counts are unchanged: with and without the rule one cubin per size.

## 9. GB200

GB200 4 x 8 (32 GPUs, 8 trays, 400G NDR), 14 Nov 2026, v2.0's kernels (measured on earlier states of
v2.0's commit stack, whose kernel headers differ from it in comments only), host's own plan: gated by itself from the
link rate, Ordered from 86 tokens. Timed plain builds, 5 rounds x 30 launches x 32 ranks, every variant
verified, 0 slow launches of 150 at every point.

| shape | tokens | v2.0 | GB200 NVLink, 32 GPUs in one domain (5-6 Nov) | difference |
|---|---|---|---|---|
| MegaSeek | 32 | 196.5 | 177 | +19.5 |
| MegaSeek | 64 | 212.2 | 185 | +27.2 |
| MegaSeek | 128 | 262.8 | 206 | +56.8 |
| MegaSeek | 256 | 357.3 | 257 | +100.3 |
| MegaSeek | 512 | 554.4 | 407 | +147.4 |
| MegaSeek | 768 | 747.4 | not measured | - |
| MegaSeek | 1024 | 938.7 | 652 | +286.7 |
| GigaSeek | 128 | 308.5 | 258 | +50.5 |
| GigaSeek | 512 | 660.3 | 528.5 | +131.8 |
| GigaSeek | 1024 | 1,109.6 | 835 | +274.6 |
| Flash | 128 | 167.7 | 92 | +75.7 |
| Flash | 512 | 433.6 | 165 | +268.6 |
| Flash | 1024 | 808.4 | 238 | +570.4 |

Here both columns are the same GPUs, so the difference is the cost of 400G InfiniBand between trays against
NVLink, with this software. Zero network on the same hardware (top-19, 13 Nov): 199.8 / 398.9 / 652.9 at 128 /
512 / 1024. Without gating the same kernel reads 12,517 us at 128 tokens (forced `GATED=0`, same request).

Regime on 400G at 4 x 8: the host's plan (Ordered with policy, gated) minus the Scan plan (gated), same request,
twins within 0.8, 5 rounds:

| shape | 96 tokens | 128 | 192 |
|---|---|---|---|
| MegaSeek | +1.1 | -8.9 | -27.9 |
| GigaSeek | +6.9 | -1.6 | -24.9 |
| Flash | -3.2 | -11.8 | -18.7 |

Gates on GB200 4 x 8 (14 Nov 2026; 28 results, every variant verified, no device message of any kind, nothing
trapped or hung):

| gate | what ran | result |
|---|---|---|
| the host chooses gated | host plan, `GATED=1`, `GATED=0` at 32 and 128 tokens | code hash of the host plan = `GATED=1`, differs from `GATED=0` |
| diagnostics check 30 + back-to-back (3 rounds) | MegaSeek 32 / 31 / 128 / 127 / 512 / 511 / 1024 / 1023 and 64 / 85 / 86 / 96 / 192 / 256; Flash 32 / 128 / 512 / 1024; GigaSeek 128 / 512 / 1024; MegaSeek with the Scan plan forced at 96 / 128 / 127 / 192 | pass 25 of 25 |
| changing-count lists, per-rank shift, check 300 | MegaSeek `512_490_640_511_600_483`; Flash `1024_1023_600_1000_513_897` | pass 2 of 2 |
| mixed-regime and crossing lists, check 300 | MegaSeek `600_64_1024_0_300_511` and `40_200_86_85_0_128`; Flash `600_64_1024_0_300_511` | pass 4 of 4 |
| back-to-back with different counts per rank (diagnostics, no reference, 5 rounds) | MegaSeek `600_64_1024_0_300_511` | pass, 0 slow of 150 |
| soak 3000 | MegaSeek 512 / 511, 1024 / 1023; Flash 1024 / 1023 | pass 6 of 6 |
| idle-rank tile (on the kernels with that rule) | lists `1024_0` and `600_64_1024_0_300_511`, check 300; back-to-back | pass; mixed list 1,129.3 us before the rule, 662.1 with it (diagnostics builds); equal counts unchanged (262.0 / 556.1 / 940.0) |

Checks here used 30 tests where the B300 gate list uses 300. Server settings: `TEST_NODES=8`,
`DG_MEGA_MOE_IB_VIRTUAL_LSA=8`, `DG_MEGA_MOE_IB_GIN_TYPE=0`, `NO_LEGACY=1`, `DG_MEGA_MOE_IB_NVL_REF=1`,
`DG_MEGA_MOE_DEV=1`, `DEV_HDR_FRESH=1`, `DEV_MAX_CASES=8`. Harness note: with `DIAGNOSTICS=1`, requests whose
sizes include one tile class were refused by the harness check "the nodes hold different kernel code" (inline
compiles of one source differed between nodes); the same requests passed with another diagnostics value. Not a
kernel fault.

2 x 8 on GB200: no table for v2.0.

## 10. What cannot be reproduced from this repository alone

| item | why |
|---|---|
| The harness outside Modal | `runner.py` is a Modal app: clustered functions with RDMA, Modal Queue / Dict / Volume as the control plane. On another scheduler the dev server (`jobs/devserver.py supervise` on every node, with `DEV_DICT_DIR` pointing at a shared directory instead of a Modal Dict) is the part to keep; that mode is used by the selftest only and was not run on a cluster |
| The cluster hardware | 2 and 4 bare-metal B300 nodes with 800G InfiniBand and GPU-initiated RDMA enabled on the host (GDAKI) |
| The reference-kernel columns of section 8 | they were timed in one request through a combined header tree (`mkab.py`) on servers whose host extension was built from an earlier tree, with a compatibility layer that v1.1 and v1.2 carried and this tree does not. The reference kernel and its host are not in this repository, and the in-repo host cannot select a tree by an arm's define |
| The 32-GPU single-domain NVLink reference | needs a GB200 rack with 32 GPUs in one NVLink domain |
| Absolute levels | a cubin's level moves by 3 to 13 us between requests and by several us between servers; only same-request differences reproduce |
| Slow-launch rates | they depend on the host's CPU count and on pinning the Gin progress threads |
| First-launch times | they depend on the JIT disk cache and on the nodes' CPUs (about 67 s cold and 9 s cached for a 1,920-token buffer at 16 GPUs; about 2 minutes from submit to result for the first request on a fresh 32-GPU server) |
| Logs and result files of the campaign | not published |
