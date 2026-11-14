# Mega MoE across NVLink domains: open work

State at commit `adac520` (14 Nov 2026). Design: `ARCHITECTURE.md`. Method, gates and tables: `REPRODUCTION.md`.

Conventions. Sizes are microseconds per launch, arm minus base in one request, on B300 unless GB200 is named.
"2 x 8" and "4 x 8" are 2 and 4 nodes of 8 GPUs; top-6 / top-19 / top-25 are the Flash / MegaSeek / GigaSeek
shapes. Every number was measured on 13 or 14 Nov 2026 with the method of `REPRODUCTION.md` (same request,
both arm orders, twins), on v1.1 or v1.2 unless a line says "this tree". Arms are plan-field overrides unless a
lead says it has code. "A candidate exists (not published)" means code was written and measured but is not in
this repository.

## (a) Autotuner for the host hyperparameters

### What is tuned today, by hand

Five values decide the plan beyond what the topology fixes. Four are in `get_mega_moe_ib_plan`
(`csrc/jit_kernels/heuristics/mega_moe.hpp`); the fifth is a device constant.

| # | hyperparameter | where | value today |
|---|---|---|---|
| 1 | Ordered boundary | the constant 26.2 us in `ordered iff remote nodes x tokens x hidden / link rate >= 26.2 us` | 512 / 171 tokens at 2 x 8 / 4 x 8 on 800G; 256 / 86 on 400G |
| 2 | L1 lead cap, in pool blocks | `plan.l1_lead_cap` | 9 in the Ordered regime, none in Scan |
| 3 | retire share and retire point | `plan.retire_percent`, `plan.retire_at_percent` | 50 % of the clusters at `round(100 x 0.6^(nodes / top-k))` % of the routed L2 with more than one remote node (90 / 92 / 71 for top-19 / 25 / 6 on four nodes); none with one |
| 4 | shared L2 early | `plan.shared_l2_early` | on wherever clusters retire |
| 5 | records per return put | `Geometry::kNumRecordsPerPut`, `impls/mega_moe_ib/geometry.cuh` | 8 with more than one remote node, 32 with one |

### Evidence that they are fitted, and to what

All were set on B300 with 800G links at 2 and 4 nodes of 8, mostly on top-19 and top-25. What is known about
how far each carries:

- **Boundary: one constant fits the node count and the link, not the top-k.** Measured crossovers in wire
  time, top-6 / top-19 / top-25: B300 2 x 8: 15 / 22 / 25 us; GB200 2 x 8: 16 / 24 / 30 us; B300 4 x 8, top-19:
  about 25 us. Points behind the 4 x 8 boundary on 800G (Ordered with policy minus Scan): top-19 at 128 / 160 /
  176 / 192 / 224 tokens: +5.2 / -1.1 / -1.9 / -5.8 / -11.4; top-25 at 192 / 224: -6.7 / -11.0; top-6: -8.2 /
  -11.7. On 400G at 4 x 8 (this tree, boundary 86): at 96 / 128 / 192 tokens top-19 +1.1 / -8.9 / -27.9, top-6
  -3.2 / -11.8 / -18.7, but **top-25 +6.9 / -1.6 / -24.9: at 96 tokens the rule loses 6.9 us** (its crossover is
  near 120 tokens, 37 us of wire time). Residuals elsewhere: top-6 at B300 2 x 8 is up to 5.5 us late between
  300 and 511 tokens (-3.1 / -5.5 at 384 / 448 for Ordered); top-25 at GB200 2 x 8, 256 tokens: +4.2. A top-k
  term of about 0.4 to 0.45 us per top-k in the threshold would cover these.
- **Lead cap.** B300 4 x 8, top-19, 512 tokens: 7 +1.2, 9 0, 11 +3.5, off +9.8. B300 2 x 8: 7 equals 9 within
  2.6 (top-19), 9 is better by 2 to 3 (top-25), 7 is better by 2.0 / 3.1 / 4.4 (top-6, 512 / 768 / 1024); top-6
  at 4 x 8, 512: 7 is -3.0. **On 400G, 2 x 8, 7 is better: -3.9 / -6.6 / -5 to -10 at top-19, 512 / 768 / 1024
  (cap 8: -2.5 / -3.0 / -2 to -7), smooth on the final gated form.** On an earlier form of gating the same
  sweep had cliffs of 30 to 60 us (cap 8 at 768 tokens, caps 5 and 6 at every size); they are gone on the final
  form and their cause was not traced. 4 x 8 on 400G: not measured.
- **Retire.** With one remote node any retire loses 9 to 25 on top-19, but top-6 at 2 x 8 gains: 50 % at 84 %
  minus no retire is -1.1 / -6.9 at 512 / 1024 on B300 and -0.6 / -2.2 on GB200. So "retire iff more than one
  remote node" is refuted for top-6; "retire iff 10 x remote nodes > top-k" covers both. **The retire point is
  not a function of nodes and top-k alone.** Top-6 at 4 x 8 (rule: 71 %): a later point is worse at 1,024
  tokens (84 %: +4.0; no retire: +7.6), an earlier one is better at 1,024 (55 %: -3.1; 40 %: -6.6) and worse
  at 512 (55 %: +4.0; 40 %: +15.0): the optimum moves with the token count, which the rule does not have. On
  top-19 earlier loses steeply (50 % at 87 %: +10; 66 % at 87 %: +23).
- **Shared L2 early.** Its measured gain (-3.3 at top-19, 4 x 8, 1024 tokens) comes from the scheduler's tail
  loop being compiled out, not from the schedule: three arms in one request, twins within 0.1, read 672.3 with
  the bit on, 673.3 off, 675.5 with the loop kept and the early schedule chosen at run time.
- **Records per put.** 8 against 32: -3 to -9 at 4 x 8; +1.1 / +4.3 / +4.3 at 2 x 8, 1024 tokens (top-19 /
  25 / 6). 16 at 2 x 8 reads -1.9 on top-6 on B300 and 0 to +4 on GB200.
- **Gating has a small-batch cost on 400G.** At 32 tokens (top-19, GB200 4 x 8, this tree) the gated plan reads
  196.4 and the ungated one 187.9: 8.5 us where the bulk is small. At 128 tokens ungated is 12,517. A token
  floor for gating is a sixth candidate parameter.
- **The top-6 shape is untuned.** Every top-6 number above was a spot check of values chosen on the other
  shapes. At 4 x 8 top-6 is wire-bound: 96 to 97 % of its distance to the single-domain time is network
  exposure at every size.

### Proposed design

1. **Starting point and fallback: the rule in the tree.** `ordered iff row_us >= 26.2`, with the link rate
   read from sysfs, plus the constants above. It reproduces every swept optimum on B300 at 2 and 4 nodes
   within the residuals listed, and 11 of 12 points on GB200 2 x 8.
2. **Search: a coordinate sweep by plan-field overrides.** Every hyperparameter except the fifth is already an
   override (`DG_MEGA_MOE_IBP_ORDERED`, `_L1_LEAD_CAP`, `_RETIRE_PERCENT`, `_RETIRE_AT_PERCENT`,
   `_SHARED_L2_EARLY`, `_GATED`; they need `DG_MEGA_MOE_DEV=1`); records per put needs a plan field first. One
   arm costs about 12 s of compile and under a minute of timing; a request with twins resolves 1 to 2 us in
   about 3 minutes per (shape, three sizes, four arms). The sweep starts at the rule's plan, moves one field at
   a time, and keeps a value only if it wins in both arm orders by more than the twins' spread.
3. **Output: a small per-deployment table** keyed by (link rate, nodes, ranks per node, top-k class, token
   class) that the host reads, with the rule as the fallback for a key that is absent. The token class is
   needed: the retire point and the gating floor depend on it.
4. **Acceptance of a tuned row**: the gate list of `REPRODUCTION.md` for every plan family the row changes
   (diagnostics checks in each tile class it newly reaches, soak, changing-count list), not just the timing.

### Cautions

- **Sweep the lead cap over several sizes together.** It showed cliffs on one tree and none on the next; 9 was
  never on a cliff in any sweep on either link.
- **Earlier retire loses steeply on top-19**; the floor near the rule is flat.
- **A boundary may differ per rank.** With the regime bit (`ARCHITECTURE.md` section 4) ranks on both sides of
  any boundary are safe in one launch, so the boundary can be a function of each rank's token count.
- **`gated` may not differ per rank.** It is chosen per rank from that rank's own link rate, and nothing
  checks that the ranks agree (section (c)). A tuner must set it for a whole group.
- **Each tuned boundary changes which builds exist**, and with them the first-launch compile set
  (`ARCHITECTURE.md` section 9). An arm that moves a buffer to another kernel family in a live process needs
  fresh ranks (`REPRODUCTION.md` section 7).
- **Levels drift between requests** by 3 to 13 us. A tuner that compares arms across requests will fit noise;
  on the in-repo host two trees cannot share a request, but plan fields can.
- **Put mode is a property of the communicator.** Nothing a tuner varies may change whether puts skip the
  credit check.

## (b) Recorded leads

Not built, or built and not adopted. Each with its measured size.

| lead | size | state |
|---|---|---|
| Top-k term in the regime threshold | -6.9 at top-25, 96 tokens, 400G 4 x 8; up to -5.5 at top-6, B300 2 x 8, 300 to 511 tokens; -4.2 at top-25, GB200 2 x 8, 256 | not built; needs the gates of the Ordered builds in the tile classes it newly reaches |
| Retire by remote nodes per top-k | -1.1 / -6.9 at top-6, 2 x 8, 512 / 1024 | needs 768 timed and the gates of that plan family |
| Retire point as a function of the token count | -6.6 at top-6, 4 x 8, 1024 with 40 %, which costs +15.0 at 512 | no rule yet |
| Lead cap 7 on 400G | -3.9 / -6.6 / -5 to -10 at top-19, GB200 2 x 8, 512 / 768 / 1024 | not adopted: 4 x 8 not measured, and the earlier cliffs are unexplained |
| No gating at small batch on 400G | -8.5 at 32 tokens, GB200 4 x 8 | not built; needs the size below which ungated loads stay safe |
| 16 records per put with one remote node | -1.9 at top-6, 1024, B300; 0 to +4 on GB200 | needs top-19 / 25 at 512 and 1024, Scan sizes, a soak |
| The tail schedule of "shared L2 early = off" without the code cost of the tail loop | about -2 at top-19, 4 x 8, 1024 | not built; see "Shared L2 early" in section (a) |
| Header as the first put of its Meta thread (today it follows the weights) | predicted -2 +- 1.5 at 4 x 8, 128 tokens on 800G; the start term on 400G at 1024 tokens is about 50 us | not built. Basis: at 128 tokens 17 of 35 us of exposure are at the start, where the counts wait for the header, and a thread's second put follows its first by about 5.7 us |
| Two row issuers per remote node in the Scan regime | -1.1 to -2.3 at 160 / 192 tokens, 4 x 8 | must stay one issuer in Ordered (+16 at 512); measured on an earlier tree of 13 Nov |
| This node's rows fill an expert's partial last pool block ("node rows last") | -5.1 at top-19 and -3.5 to -5.5 at top-25, 2 x 8, 1024; 0 to -1 at 4 x 8 | a candidate exists (not published; about 100 lines, default off); not adopted: it pays at one configuration only. Traced at 4 x 8: the freed CTAs wait on the next expert's full block instead |
| Proxy-sum batch re-poll (as the pull has) | predicted -2 +- 2; not measured | a candidate exists (not published) |
| Return order sent on the wire instead of sorted twice | not measured | would make invariant I10 hold by construction; a wire addition |
| Record flag inside the chunk's put (gated loads) | -10 / -16 / -23 at 512 / 768 / 1024, GB200 4 x 8 | not safe as tried: 7 launches of +1.2 ms in 900 (an in-bulk word polled blank by many warps), and the tail flag of a partial last chunk lands where a later launch's records go |
| First-need front-loading at 4 x 8 on slow links | model: 59 / 143 us of row wait at 512 / 1024 on 400G (8 / 28 on 800G) | none built. Model: a token has top-k / nodes experts on a node, so the share of a row stream needed through pool fraction x is 1 - (1 - x)^(top-k / nodes) |

Closed with a cause, recorded so they are not tried again:

| idea | result | cause |
|---|---|---|
| Local heads: an expert's partial block placed early in the pool, apart from its other blocks | +22 at top-19, 4 x 8, 512; +87 at 2 x 8, 1024 | separating an expert's blocks costs about 1.0 to 1.2 us per head. Rule for any pool layout: two blocks of one expert stay adjacent or at most 2 other pool blocks apart (step measured between 2 and 4 at 2 x 8, 1024 tokens: +4.8 at 2, +54.8 at 3, +66.7 at 4) |
| One wire format (sender tables) at every size | +10.4 at 4 x 8, 128; +8.5 at 2 x 8, 32 and 128 | the table chain sits in front of the counts. Scan cannot be dropped at small batch |
| Regime from the buffer's maximum token count | +12.2 at 2 x 8, 128 | same cause |
| A work-conserving L1 claim | no runnable unclaimed block in any sample | CTAs that wait, wait for rows (sampled at top-19 and top-25, 4 x 8, 512 and 1024) |
| Single-pass source combine | +5 to +8 at 1024 | `impls/mega_moe_ib/combine.cuh`, header comment |
| Earlier retire on 400G, to fill the idle rail | 50 % at 80 / 70 / 60 %: +1.6 / +27 / +62 (top-19, GB200 4 x 8, 1024) | records are not ready earlier: a token's record needs its last expert on the node |
| A rows-landed count as the landing proof (instead of a hint) | hang and fault on GB200; on B300 rows were read blank after their count | the count runs ahead of what remote readers see; hence "a hint is never a proof" (invariant I21) |
| One landing protocol for both NIC classes (a flag behind every bulk put, no read before it) | +5.6 / +19.7 / +26.2 at 128 / 512 / 1024 on B300 4 x 8 | the first rows of a put wait for its last byte; gating is selected by link rate instead |
| A launch fault, CUDA error 700 with no message, in some builds that added code (32 GPUs, Ordered regime) | three independent additions died within 10 s; the same trees pass 30 of 30 on the in-repo host | a defect of the compatibility layer for hosts built from earlier trees, which this tree no longer has. What that host lacked was not found. A device trap is error 719, not 700 |

Where the remaining distance is, for whoever picks the next lead (top-19, 4 x 8, zero-network arms):
the kernel without the wire is 25 to 41 us above the stock kernel on one B300 node at equal per-GPU work
(first routed claim +8 to +20, row wait 8 to 11 us per CTA even with no network because rows go through the
mirrors, tail +9 to +11); the tail policy costs nothing there. Network exposure on top is 35 us at 128 tokens
and 57 to 88 at 512 to 1024. (Zero network, top-19, 1024: plan 590.6, no retire 590.1, no cap 584.8, neither
599.4; stock on one node 549.0.)

## (c) Correctness and robustness

### Open

1. **Final gate table on B300.** This tree's kernels are new cubins everywhere (the start barrier moved in
   shared memory). Its full gate set on B300 at 16 and 32 GPUs was still running when this was written;
   `REPRODUCTION.md` section 6 holds the placeholder. The in-repo test command (`REPRODUCTION.md` section 4) has
   not been run.
2. **Ranks with a few tokens beside busy ranks.** A rank with zero tokens now takes the tile of a quarter of
   the buffer's maximum (`ARCHITECTURE.md` section 8): list `1024_0` at 4 x 8, top-19, 1,321.8 us before and
   564.6 after; top-25 1,707 and 676; top-6 486 and 424. A rank with 1 to 20 tokens still takes the smallest
   tile and slows the group the same way: **8 tokens next to 1,024: 1,331.5 us** (64 next to 1,024: 591).
   Raising every small count's tile costs equal small batches 8 to 10 us (measured at 32 and 64 tokens), and a
   tile rule must stay a function of the public block-size query's arguments, because callers lay the shared
   expert's scale factors out by it. The stock kernel has the same behaviour inside one NVLink domain.
3. **No cross-rank plan check.** Ranks build their plans independently. The regime is safe per source (regime
   bit). `gated` is not: it is chosen from each rank's own link rate, so nodes with different NICs, or one node
   whose sysfs is unreadable, would disagree, and a gated receiver would wait for record flags an ungated proxy
   never puts (a named timeout after about 8 s). Development overrides are per process too. Fix: a plan hash
   in the header word, or one comparison in the first-launch rendezvous.
4. **The new 4 x 8 boundary on 800G.** Diagnostics checks at 176 / 192 / 224 and soaks at 192 / 224 passed with
   the regime forced on v1.2; a changing-count list in which the host chooses per rank across 171 tokens was
   not run on B300 (on 400G the lists across 85 | 86 passed).
5. **Return records are fenced by lane 0 only** before a chunk put (`ib::reduce`, step 4), while staged rows
   are fenced by every writer. No failure in 3,000-launch soaks. Accept in writing, or time a fence in every
   lane.
6. **Do not put a printf plus trap in `ib::start`.** Count, and report at the next launch, as
   `report_return_gaps` does: a device trap there surfaced without its message.
7. **The reference kernel's hang with few experts per rank** was never root-caused; the in-repo table build
   is written not to depend on the expert count and passes at 12 to 76 experts per rank.

### Serving integration (from a code audit of v1.1, updated for this tree; none blocks benchmarking)

| item | state | what it takes |
|---|---|---|
| Error path | traps only. A timeout (8 s in named polls, 60 s in waits) ends in a device trap that kills the CUDA context of every waiting rank, sometimes without its message. Epochs are compared for equality across ranks, so a restarted rank cannot rejoin | design item: a bounded wait that sets a status word the host reads, and a host-side group reset. At minimum document "any timeout = rebuild the whole group" and have the serving layer do it |
| NaN aliases the data sentinel | open as a contract. A token row or scale-factor unit of all-ones bytes reads as "not landed" and spins. The record path does not produce the pattern (the fp8 conversion writes NaN as 0x7F) | state "inputs are finite" as a contract; add a debug-mode host check |
| Packaging | the kernel needs NCCL 2.32.3 headers (`comm/gin.cuh` uses 2.32 internals), a C++ toolchain at run time (`ib.py` builds an extension inline), `DG_NCCL_INCLUDE_DIR` set before any DeepGEMM kernel is compiled in the process, and `NCCL_GIN_GDAKI_QP_DEPTH=1024` honoured (set with `setdefault`; a smaller value silently wins while every put skips the credit check) | pin NCCL in the image; set the include directory at import; prebuild the extension; assert the queue depth after the communicator is created |
| Gin progress threads | with a backend that uses host progress threads, launches stall by milliseconds unless each thread has a core of its own. The harness pins them (`gin_partition` in `tools/mega_moe_ib/jobs/devserver.py`) | move the pinning into the serving launcher |
| Topology detection | ranks per node is guessed as `min(group size, torch.cuda.device_count())` (`ib.py`, `ranks_per_domain`); a process that sees one GPU gets 1. Node-contiguous rank numbering is assumed, not checked. The link rate is the mean of the fastest active InfiniBand ports, one per visible GPU; unreadable sysfs plans 800G and no gating | take ranks per node from NCCL's LSA team; assert the numbering; print the planned rate at start-up (`DG_PRINT_CONFIGS=1`) and treat "unknown" as a deployment error |
| Start-up cost: buffers | creating an IB buffer takes about 3.5 s at 32 ranks (3.3 to 3.7 s measured), and 14.4 s for the first one of a process (its communicator) | create buffers at start-up, not on the request path |
| Start-up cost: first launch | the first call of a buffer shape compiles every kernel the buffer can launch and returns after one rendezvous: about 67 s cold and 9 s from the JIT disk cache for a 1,920-token buffer at 16 GPUs; the first request on a fresh 32-GPU server took 2 min 05 s. There is no exported warm-up function | one collective dummy call per buffer shape at start-up (zero tokens works); a `warm_up(sym_buffer)` wrapper; persist `DG_JIT_CACHE_DIR` |
| Kernel families and the rendezvous | the set of launched kernels is per process, not per buffer; a call that moves a buffer to a kernel family some ranks have and others lack ends in a header timeout (seen in the harness with a tile override applied inside a live process; the same arm passes when its kernels are compiled first). The current kernel is marked before its family compiles; no rendezvous under CUDA graph capture | key the set by buffer; compile before marking; decide the capture case |
| Development switches | fixed for the host: plan overrides, trace, zero network, compile-only and no-rendezvous act only with `DG_MEGA_MOE_DEV=1` and are refused by name otherwise. Left outside the guard on purpose: `DG_MEGA_MOE_IB_VIRTUAL_LSA`, `DG_MEGA_MOE_IB_FORCE`, `DG_MEGA_MOE_IB_GIN_TYPE` in `ib.py` | decide whether a serving image should refuse those three |
| Limits surface late | the static asserts of `ARCHITECTURE.md` section 8 appear as a JIT error at the first launch; the start's scratch-size check is a device printf plus trap | mirror them as host asserts at buffer creation |
| `base=` buffer reuse | checks the stock byte count only; no IB tail, no sentinel fill | refuse it when IB is on unless the shape is identical |
| NCCL resources | `SymmBuffer.destroy()` drops references; the extension's `destroy` is never called | call it (about 5 lines) |

Fixed and kept here as a record: the regime chosen per rank (regime bit, v1.2); the epoch alias of the blank
header (v1.2); the first-launch deadlock with mixed tile classes (v1); the compatibility layer (removed in this
tree); development switches reachable in a serving process (this tree); zero-token ranks (this tree).

### Method gap

Two trees cannot be compared in one request on the in-repo host: it prints only the plan into the JIT source,
so an arm's define cannot select a tree in a combined header tree (`tools/mega_moe_ib/mkab.py`). Every
tree-against-tree number for this kernel was made on hosts that forward arm defines. On the in-repo host two
trees are two requests on fresh ranks, with the 3 to 13 us of between-request drift. A development-only
forwarded define (one macro, under `DG_MEGA_MOE_DEV=1`) would restore the same-request method.

## (d) 400G InfiniBand (GB200, ConnectX-7): a known software gap

Status: **usable with gated loads, which the host selects by itself from the link rate; gated on the GB200
4 x 8 gate set; not tuned.** Tables and gates: `REPRODUCTION.md` section 9.

- Without gating the same kernel takes 9.9 / 63.6 / 129.6 ms at 128 / 512 / 1024 tokens (top-19, 4 x 8). With
  it: 262.8 / 554.4 / 938.7 us, with 0 slow launches in 150 per point.
- **Measured distance to NVLink on the same GPUs** (32 GPUs in one NVLink domain, 5-6 Nov): top-19 +19.5 /
  +27.2 / +56.8 / +100.3 / +147.4 / +286.7 at 32 / 64 / 128 / 256 / 512 / 1024 tokens; top-25 +50.5 / +131.8 /
  +274.6 at 128 / 512 / 1024; top-6 +75.7 / +268.6 / +570.4. With the network switched off the kernel reads
  199.8 / 398.9 / 652.9 (top-19, 128 / 512 / 1024), which is the NVLink time: the whole distance is network
  exposure (about 63 / 155 / 286 us).
- How much of that is the link and how much is software is not settled. The link's share: at 1,024 tokens a
  rank moves 15.7 MB of rows out and 15.7 MB of records back per launch, 628 us of a 400G rail inside a 653 us
  kernel; the rail delivered 47.4 GB/s into each GPU (95 % of nominal), all rows were in at 383 us, and the
  return stream was offered between 610 and 800 us. A modelled floor on the current GEMM order is about 865 us
  at 1,024 tokens against about 940 measured. Software terms seen in a stage account of an earlier gated form at
  1,024 tokens: start +50 us (the header lands behind the weights and the bulk), the row stream's
  front-loading +110 to +140, a return tail of about +150 bound by the wire.
- What is recorded against the gap, none built (sizes in section (b)): the header as a first put; the record
  flag inside the chunk's put; no gating at small batch; lead cap 7; a top-k term in the regime threshold;
  records ready earlier, which needs a different expert order. Top-6 is far from NVLink at every size and has
  no stage account.
- Cause of the pathology, for the record: on this NIC class a bulk region that many GPU threads read over
  NVLink while it is still blank becomes visible only milliseconds later. Refuted as causes: a diagnostics
  build, a per-put timer, put size, context mapping, the Gin backend, rail bandwidth. One local reader of
  blank bulk memory is free; many local warps polling blank words inside a bulk region are not.
- Not measured on this tree: 2 x 8 on GB200 (plan-field sweeps exist on earlier gated forms: the 256-token
  boundary there was the swept optimum at 11 of 12 points); any gate with `gated` on at 800G (the option's
  early-hint counter fires on B300, so it stays off there).
