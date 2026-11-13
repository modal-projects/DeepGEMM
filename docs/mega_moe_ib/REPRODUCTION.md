# Mega MoE across NVLink domains: reproduction

How every kind of number quoted for the IB kernel is produced, on which hardware and software, and what a
tree must pass before its numbers count. The harness is in `tools/mega_moe_ib/` (its `README.md` has the
bring-up). Design: `ARCHITECTURE.md`. Open items: `TODO.md`.

All times are microseconds per kernel launch. "EP16" and "EP32" mean 16 and 32 GPUs (2 and 4 nodes of 8).

## 1. Hardware

| name used here | nodes | GPUs | NIC per GPU | where the kernel stands |
|---|---|---|---|---|
| B300 2 x 8 | 2 bare-metal nodes | 16 B300 | ConnectX-8, 800G InfiniBand | all gates run; fitted |
| B300 4 x 8 | 4 bare-metal nodes | 32 B300 | ConnectX-8, 800G InfiniBand | all gates run; fitted |
| GB200 4 x 8 | 8 trays of 4 GPUs in one rack, grouped as 4 logical nodes of 8 (`DG_MEGA_MOE_IB_VIRTUAL_LSA=8`) | 32 GB200 | ConnectX-7, 400G NDR InfiniBand, one rail per GPU | correct, not usable as is (section 9) |

Every number in section 8 is from B300. A B300 node here has 8 GPUs in one NVLink domain and about 115 host
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

- `DG_MEGA_MOE_IB=1` in the environment of every rank: Mega MoE buffers are then allocated as one NCCL
  window with the IB tail (`deep_gemm/mega/ib.py`).
- NCCL 2.32.3 headers and library from the `nvidia-nccl-cu13` wheel. `ib.py` finds them through the `nvidia`
  package, builds a small extension with `torch.utils.cpp_extension.load_inline` (needs the C++ toolchain at
  run time) and sets `DG_NCCL_INCLUDE_DIR` for the JIT. If another DeepGEMM kernel is compiled in the process
  before the first IB buffer is allocated, export `DG_NCCL_INCLUDE_DIR=<nccl>/include` at process start.
- `LD_LIBRARY_PATH` must put that NCCL first (`jobs/setup_deps.sh`).

## 4. The in-repo test

`tests/test_mega_moe.py` runs the IB kernel when `DG_MEGA_MOE_IB=1`. It has two references:

- with DeepEP and tilelang installed: the test's usual baseline (DeepEP dispatch, grouped GEMMs, combine);
- with `DG_MEGA_MOE_IB_NVL_REF=1`: the same kernel on a second buffer whose ranks form one NVLink domain,
  that is, the stock path on the same GPUs and inputs. This works only where all ranks really share one
  domain, so it is the single-node form of the test.

Because a remote node's contribution passes through an fp8 record, outputs are compared with a bound:
`DG_MEGA_MOE_IB_TOLERANCE=1e-3` (on `calc_diff`; passing runs read 1.7e-4 to 2.6e-4).

Single node, every IB path exercised by splitting 8 GPUs into two virtual domains of 4:

```bash
DG_MEGA_MOE_IB=1 DG_MEGA_MOE_IB_VIRTUAL_LSA=4 DG_MEGA_MOE_IB_NVL_REF=1 DG_MEGA_MOE_IB_TOLERANCE=1e-3 \
python tests/test_mega_moe.py --num-processes 8 --num-experts 1216 --num-topk 19 --hidden 5120 \
    --intermediate-hidden 2304 --num-max-tokens-per-rank 1024 --num-tokens 1024 --num-correctness-tests 300
```

Status: this command line was not run during the campaign as recorded (not measured). What was run is the
harness's own copy of the test's set-up (`Case` in `jobs/devserver.py`, which mirrors the test), including
with `VIRTUAL_LSA=8` and the single-domain reference on GB200. Multi-node runs of the test file need
`MASTER_ADDR`, `MASTER_PORT` and `WORLD_SIZE` (`deep_gemm/utils/dist.py`).

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

**Forcing a regime.** `DG_MEGA_MOE_IBP_ORDERED=0|1` sets only the regime. The tail policy is chosen before
the overrides, from the regime the host would have picked, so "Ordered with the host's policy" below the
boundary is spelled out: `DG_MEGA_MOE_IBP_ORDERED=1,DG_MEGA_MOE_IBP_L1_LEAD_CAP=9` and, with more than one
remote node, `DG_MEGA_MOE_IBP_RETIRE_PERCENT=50,DG_MEGA_MOE_IBP_RETIRE_AT_PERCENT=<90|92|71>,DG_MEGA_MOE_IBP_SHARED_L2_EARLY=1`
(top-19 / 25 / 6 on four nodes).

**Two trees in one request.** `--repo` sends the headers of one tree; all arms of the request share them. An
A/B between two trees in one request needs a combined header tree in which each differing header dispatches on
a macro one arm defines. The tool that builds such a tree is not part of this harness (section 10). Changes
that are plan fields need no second tree.

## 6. Gate list

A tree's numbers count only if it passes all of these. "diag" = diagnostics on; every request must report
`verified = True` for every variant.

| # | gate | what runs |
|---|---|---|
| 1 | IB off = stock PTX | front-end PTX of the IB-off kernel, this tree against upstream `5081f85`, two shapes; normalised md5 equal (below) |
| 2 | diag check, every tile class the plan can select | `--check 300`, each size with its odd neighbour, three shapes. EP32: Scan 128, 16, 40, 96, 240; Ordered 256, 512, 1024, 768, 384, 464 (Flash without 240 and 464). EP16: Scan 128, 256, 64, 16; Ordered 512, 1024, 768, 928 |
| 3 | changing-count lists, per-rank shift, diag, `--check 300` | Scan `128_120_144_127_136_121`; zero-token `20_0_17_1_0_13`; Ordered `512_490_640_511_600_483`; mixed-regime and crossing: `128_600`, `0_1024`, `8_600`, `0_0_0_600`, `200_300` (EP32), `600_128_1024_0_300_511` |
| 4 | soak | `--check 3000`: three shapes Ordered 512 / 511 and 1024 / 1023; Scan 128 / 127 |
| 5 | back-to-back diagnostics | timing phase of the diag builds: three shapes, Scan 128 / 127, Ordered 512 / 511, 1024 / 1023; no trap, no alert, no device message |
| 6 | timing against the previous base | same request, both orders, twins, at 128, 512, 1024 on three shapes at both GPU counts: equal or better, or the miss explained in writing |
| 7 | host | first IB launch of a buffer (cold JIT cache); an unknown `DG_MEGA_MOE_IBP_*` name is refused with its name; a plan whose backend differs from the communicator's is refused |
| 8 | experts per rank (Ordered, EP16) | diag check at 12, 24, 31, 32, 33, 38, 50, 76 experts per rank |

Gate 1, local (front end only; needs nvcc, not a GPU). For each of the two instantiations, compile the same
`kernel.cu` against the include tree of upstream and of the tree under test and compare:

```bash
cat > k.cu <<'EOT'
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        1024, 5120, 2304, 304, 1, 19,          // MegaSeek at 8 ranks; second shape: 96 experts, top-6
        240, 128, 128, 24, 8, 256, 128,        // second shape: 192, 128, 128, 32, 32, 256, 128
        15360, 16384, 4, 5120,                 // second shape: 15360, 20480, 4, 5120
        128, 128, 256,
        148, 8, 10.0f, true, cutlass::detail::float_e2m1_unpacksmem_t>);
}
EOT
ptx() {  # ptx <include dir> -> normalised md5
  nvcc -std=c++20 --gpu-architecture=compute_100a --ptx -o - k.cu -I"$1" -I<cutlass>/include -I<deep_jit>/include \
      -I<nccl>/include --diag-suppress=39,161,174,177,186,940 --expt-relaxed-constexpr -O1 \
  | grep -v '^//\|\.file\|\.loc' \
  | sed -E 's/_ZN[0-9]+_INTERNAL_[0-9a-f]+_[0-9a-zA-Z_]+/_ZN_I/g; s/\$str[0-9$_a-zA-Z]*//g; s/_INTERNAL_[0-9a-f]+_[0-9a-zA-Z_]+//g' \
  | grep -v '^.global .align 1 .b8 \[[0-9]*\] = {47, ' | md5sum
}
ptx <upstream>/deep_gemm/include; ptx <tree>/deep_gemm/include      # the two md5 must be equal
```

The first `sed` expression matters: internal-linkage names carry the compiler's process id in a
length-prefixed mangled name, and without it an unchanged tree changes md5 when the process id gains a digit.
The last `grep` drops string constants that hold include paths.

State of the gates for the tree at `6ebf8b4` (13 Nov 2026): gate 1 pass; gates 2 to 5 pass at EP32 on the
earlier host ABI and at EP16 on the in-repo host; the mixed-regime lists of gate 3 pass at EP32 and EP16 on the
in-repo host; gate 6 within +0.8 us of the previous base at nine EP32 points and not slower at EP16. Open at
that date: gates 2, 4 and 5 in full at EP32 on the in-repo host (`TODO.md` section (c)), and back-to-back
launches with different counts per rank (the harness mode exists, `DG_MEGA_MOE_DEV_ALT_TIMING=1`, and had not
been run on a server).

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

A request with `--check` greater than 0 carries one arm. On the in-repo host, send one request at a time to a
server unless it was started with the current `jobs/devserver.py` and `DEV_HDR_FRESH=1`.

## 8. Expected numbers

Final values come from the re-benchmark of the final tree. Until they are filled in, every cell below is a
placeholder; do not quote this table.

**PLACEHOLDER TABLE: to be filled from the re-benchmark (date, server, tree commit and host ABI per row).**

| shape | GPUs | tokens | regime (host plan) | time | twin | zero network | NVLink, 32 GPUs in one domain (GB200) | stock, one B300 node, equal per-GPU work | date | server | tree |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MegaSeek | 32 | 128 | Scan | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MegaSeek | 32 | 256 | Ordered | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MegaSeek | 32 | 512 | Ordered | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MegaSeek | 32 | 768 | Ordered | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MegaSeek | 32 | 1024 | Ordered | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GigaSeek | 32 | 128 | Scan | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GigaSeek | 32 | 256 | Ordered | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GigaSeek | 32 | 512 | Ordered | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GigaSeek | 32 | 768 | Ordered | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GigaSeek | 32 | 1024 | Ordered | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Flash | 32 | 128 | Scan | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Flash | 32 | 256 | Ordered | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Flash | 32 | 512 | Ordered | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Flash | 32 | 1024 | Ordered | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MegaSeek | 16 | 128 | Scan | TBD | TBD | TBD | - | TBD | TBD | TBD | TBD |
| MegaSeek | 16 | 256 | Scan | TBD | TBD | TBD | - | TBD | TBD | TBD | TBD |
| MegaSeek | 16 | 512 | Ordered | TBD | TBD | TBD | - | TBD | TBD | TBD | TBD |
| MegaSeek | 16 | 1024 | Ordered | TBD | TBD | TBD | - | TBD | TBD | TBD | TBD |
| GigaSeek | 16 | 128 | Scan | TBD | TBD | TBD | - | TBD | TBD | TBD | TBD |
| GigaSeek | 16 | 512 | Ordered | TBD | TBD | TBD | - | TBD | TBD | TBD | TBD |
| GigaSeek | 16 | 1024 | Ordered | TBD | TBD | TBD | - | TBD | TBD | TBD | TBD |
| Flash | 16 | 128 | Scan | TBD | TBD | TBD | - | TBD | TBD | TBD | TBD |
| Flash | 16 | 512 | Ordered | TBD | TBD | TBD | - | TBD | TBD | TBD | TBD |
| Flash | 16 | 1024 | Ordered | TBD | TBD | TBD | - | TBD | TBD | TBD | TBD |

How to read the two reference columns once filled:

- **NVLink, 32 GPUs in one domain** was measured on a GB200 rack (5 and 6 Nov 2026), not on B300. Comparing a
  B300 IB time with it compares two kinds of hardware.
- **Stock, one B300 node, equal per-GPU work** is the stock kernel on 8 B300 GPUs with the expert count
  scaled so that each GPU does the same work as in the 32-GPU run. It is the same-hardware reference for the
  zero-network column.

### Interim values (not final; for orientation only)

Measured 13 Nov 2026 on the tree one commit pair before this one (`7863474`; the two later commits read
within +0.8 us at nine EP32 points). EP32 rows: an earlier host ABI on the servers (the kernel's compat
layer), MegaSeek on one 4-node server and GigaSeek / Flash on another; EP16 on a 2-node server. Same-request
medians, order a / order b. Levels differ by several us between servers and requests.

| shape | GPUs | tokens | time (a / b) | zero network | NVLink, 32 GPUs, GB200 | stock, one B300 node |
|---|---|---|---|---|---|---|
| MegaSeek | 32 | 128 (Scan) | 232.5 / 232.5 | 198.0 | 206 | not measured |
| MegaSeek | 32 | 512 | 426.3 / 425.9 | 359.6 | 407 | 334.2 |
| MegaSeek | 32 | 1024 | 670.4 / 670.5 (675 to 678 in later requests) | 590.1 | 652 | 549.0 |
| GigaSeek | 32 | 128 (Scan) | 273.9 / 273.7 | not measured | 258 | not measured |
| GigaSeek | 32 | 512 | 510.1 / 511.5 | 457.7 | 528.5 | 429.5 |
| GigaSeek | 32 | 1024 | 831.7 / 832.8 | 754.4 | 835 | 720.4 |
| Flash | 32 | 512 | 236.0 / 233.9 | not measured here | not in the notes this table was built from | not measured |
| Flash | 32 | 1024 | 401.0 / 400.9 | not measured here | not in the notes this table was built from | not measured |
| MegaSeek | 16 | 128 (Scan) | 311.0 | not measured | - | not measured |
| MegaSeek | 16 | 1024 | 697.6 / 697.8 | not measured | - | not measured |
| GigaSeek | 16 | 1024 | 905.4 / 906.0 | not measured | - | not measured |
| Flash | 16 | 1024 | 294.1 / 294.4 | not measured | - | not measured |

Stage account of one point (MegaSeek, 32 GPUs, 128 tokens, traced pair, 13 Nov): of 38 us of network exposure
in the traced build, 17 are at the start (the counts wait for the header), 10 on the return wire after the
last put, 6 after the last stamp (mirror re-arm and exit), and under 5 in the pull and GEMM phases together.

## 9. GB200

The tree at `6ebf8b4` comes up on GB200 4 x 8 with the in-repo host and passes the single-domain reference
check (`calc_diff` 2.6e-4), but every launch takes milliseconds: about 10 ms at 128 tokens (MegaSeek, 4 x 8),
growing with the row bytes. With zero network the same kernel reads 201.7 us at that point. The cause and the
candidate fix are in `ARCHITECTURE.md` section 5 and `TODO.md` section (d). No GB200 number belongs in
section 8 until that item is closed.

Server settings used there: `TEST_NODES=8`, `DG_MEGA_MOE_IB_VIRTUAL_LSA=8`, `DG_MEGA_MOE_IB_GIN_TYPE=0`,
`NO_LEGACY=1`, `DG_MEGA_MOE_IB_NVL_REF=1`, `DEV_HDR_FRESH=1`, `DEV_MAX_CASES=8`.

## 10. What cannot be reproduced from this repository alone

| item | why |
|---|---|
| The harness outside Modal | `runner.py` is a Modal app: clustered functions with RDMA, Modal Queue / Dict / Volume as the control plane. On another scheduler the dev server (`jobs/devserver.py supervise` on every node, with `DEV_DICT_DIR` pointing at a shared directory instead of a Modal Dict) is the part to keep; that mode is used by the selftest only and was not run on a cluster |
| The cluster hardware | 2 and 4 bare-metal B300 nodes with 800G InfiniBand and GPU-initiated RDMA enabled on the host (GDAKI) |
| Comparisons with the kernel that preceded this tree | they were timed in one request through a combined header tree and the compat layer, on servers whose host extension was built from the earlier tree. The earlier kernel and the tool that builds combined trees are not in this repository |
| The 32-GPU single-domain NVLink reference | needs a GB200 rack with 32 GPUs in one NVLink domain |
| Absolute levels | a cubin's level moves by 3 to 13 us between requests and by several us between servers; only same-request differences reproduce |
| Slow-launch rates | they depend on the host's CPU count and on pinning the Gin progress threads |
| First-launch times | they depend on the JIT disk cache and on the nodes' CPUs (about 67 s cold and 9 s cached for a 1,920-token buffer at 16 GPUs; about 2 minutes from submit to result for the first request on a fresh 32-GPU server) |
| Logs and result files of the campaign | not published |
