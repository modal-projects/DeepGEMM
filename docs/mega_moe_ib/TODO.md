# Mega MoE across NVLink domains: open work

State at commit `6ebf8b4` (13 Nov 2026). Design: `ARCHITECTURE.md`. Method and gates: `REPRODUCTION.md`.

Conventions. Sizes are microseconds per launch, arm minus base in one request, on B300 unless GB200 is named.
"2 x 8" and "4 x 8" are 2 and 4 nodes of 8 GPUs. Every number was measured on 13 Nov 2026 unless another
date is given, with the method of `REPRODUCTION.md` (same request, both arm orders, twins), on the tag before
this commit (`7863474`) unless stated. Arms are plan-field overrides unless a lead says it has code. "A
candidate exists (not published)" means code was written and measured but is not in this repository.

## (a) Autotuner for the host hyperparameters

### What is tuned today, by hand

Five values decide the plan beyond what the topology fixes. Four are in `get_mega_moe_ib_plan`
(`csrc/jit_kernels/heuristics/mega_moe.hpp`, struct `Fitted` and the lines below it); the fifth is a device
constant.

| # | hyperparameter | where | value today |
|---|---|---|---|
| 1 | Ordered boundary: tokens per rank from which a rank sends in the Ordered regime | `Fitted::min_ordered_tokens` | 512 with one remote node, 256 with more |
| 2 | L1 lead cap, in pool blocks | `Fitted::l1_lead_cap` | 9 in the Ordered regime, none in Scan |
| 3 | retire share and retire point | `Fitted::retire_percent`, `plan.retire_at_percent` | 50 % of the clusters at `round(100 x 0.6^(nodes / top-k))` % of the routed L2, with more than one remote node; none with one |
| 4 | shared L2 early | `plan.shared_l2_early` | on wherever clusters retire |
| 5 | records per return put | `Geometry::kNumRecordsPerPut`, `impls/mega_moe_ib/geometry.cuh` | 8 with more than one remote node, 32 with one |

### Evidence that they are fitted to one configuration

They were set on B300 with 800G links, at 2 and 4 nodes of 8, mostly on the top-19 and top-25 shapes.

- **Boundary, 4 x 8.** Ordered with the host's policy minus Scan, same request, twins within 0.2: top-19 at
  160 / 192 / 224 tokens: -1.1 / -5.8 / -11.4; top-25 at 192 / 224: -6.7 / -11.0; top-6 at 192 / 224:
  -8.2 / -11.7; at 128 Ordered still loses (+5.2). One constant, 256 to 192, gains 6 to 12 us at 192 to 224
  tokens on every shape (4 x 8, 7 rounds). A candidate exists (not published; host only, one constant). Not
  adopted yet: it needs an Ordered soak at 192 / 224 and a changing-count list across 192.
- **Boundary, 2 x 8.** Nothing was measured between 257 and 511 tokens when 512 was set. Later sweep,
  Ordered + cap minus Scan: top-6 -3.1 / -5.5 at 384 / 448. The best boundary differs per shape: about 350
  (top-19), 400 (top-25), 250 (top-6) by the model below.
- **Lead cap.** On B300 4 x 8, top-19, 512 tokens: 7 +1.2, 9 0, 11 +3.5, off +9.8. On GB200 2 x 8 (400G): 7
  minus 9 = -3.2 / -5.4 / -5.6 (top-19, 512 / 768 / 1024), -2.3 / -3.4 / -3 to -7 (top-25),
  -3.2 / -8.6 / -6.6 (top-6). On B300 2 x 8: 7 equals 9 within 2.6 (top-19), 9 is better by 2 to 3 (top-25),
  7 is better by 2 to 4.4 (top-6).
- **Retire.** With one remote node any retire loses 9 to 25 on top-19, but on top-6 at 2 x 8, 50 % at 84 %
  minus no retire is -1.1 / -6.9 at 512 / 1024. A rule "retire iff 10 x remote nodes > top-k" covers both.
- **Shared L2 early.** Its measured gain (-3.3 at top-19, 4 x 8, 1024 tokens) comes from the scheduler's tail
  loop being compiled out, not from the schedule: with the loop kept and the schedule chosen at run time the
  early schedule is +2.2 slower. (Three arms in one request, twins within 0.1:
  672.3 with the bit on, 673.3 off, 675.5 with the loop kept and the early schedule chosen at run time.)
- **Records per put.** 8 against 32: -3 to -9 at 4 x 8; +1.1 / +4.3 / +4.3 at 2 x 8, 1024 tokens (top-19 /
  25 / 6). 16 at 2 x 8 reads -1.9 on top-6 on B300 and 0 to +4 on GB200.
- **The top-6 shape is untuned.** Every top-6 number above was a spot check of values chosen on the other
  shapes. At 4 x 8 top-6 is wire-bound: 96 to 97 % of its distance to the single-domain time is network
  exposure at every size.

### Proposed design

1. **Starting point and fallback: a small physical model.** Only the regime depends on the link. With
   `S = remote nodes x tokens x hidden / link rate` (the time the row stream occupies the wire), the measured
   rule is: Ordered iff `S >= 6.55 us x (nodes + 2)`. At 800G this reproduces today's table at 2 and 4 nodes
   for every token count; at 400G it gives 256 (2 x 8) and 128 (4 x 8). On GB200 2 x 8 it was the swept optimum
   at 10 of 12 points, and today's table loses 14 to 17 us at 384 tokens there. The other four values are
   constants or topology rules in the model. The link rate can be read from sysfs
   (`/sys/class/infiniband/<dev>/ports/<n>/rate`, filtered by `link_layer`) when the communicator is created
   and carried in the buffer handle; unreadable means 800G. A candidate exists (not published;
   host +19 / -29 lines). Known misses of the model: a top-k term of
   about 0.45 us per top-k in the threshold; the lead cap (next point).
2. **Search: a coordinate sweep by plan-field overrides.** Every hyperparameter except the fifth is already an
   override (`DG_MEGA_MOE_IBP_ORDERED`, `_L1_LEAD_CAP`, `_RETIRE_PERCENT`, `_RETIRE_AT_PERCENT`,
   `_SHARED_L2_EARLY`); records per put needs a plan field first. One arm costs about 12 s of compile and under
   a minute of timing; a request with twins resolves 1 to 2 us in about 3 minutes per (shape, three sizes,
   four arms). The sweep starts at the model's prediction, moves one field at a time, and keeps a value only
   if it wins in both arm orders by more than the twins' spread.
3. **Output: a small per-deployment table** keyed by (link rate, nodes, ranks per node, top-k class) that the
   host reads in place of `Fitted`, with the model as the fallback for a key that is absent.
4. **Acceptance of a tuned row**: the gate list of `REPRODUCTION.md` for every plan family the row changes
   (diagnostics checks in each tile class it newly reaches, soak, changing-count list), not just the timing.

### Cautions

- **The lead cap is not smooth.** On GB200 2 x 8, top-19: cap 8 is -1.5 at 512 and 1024 and +61.5 at 768; caps
  5 and 6 are +24 to +59; 7 is a narrow valley. Top-25 has its cliff at 5. Nothing of this was traced. A tuner
  must sweep the cap over several sizes together and must not pick a value next to a cliff; 9 was never on a
  cliff in any sweep on either link.
- **Earlier retire loses steeply** (4 x 8, top-19, 1024: 50 % at 87 % is +10, 66 % at 87 % is +23); the floor
  near the rule is flat.
- **A boundary may differ per rank.** Since the regime bit (`ARCHITECTURE.md` section 4) ranks on both sides
  of any boundary are safe in one launch, so the boundary can be a function of each rank's token count.
- **Each tuned regime boundary changes which builds exist**, and with them the first-launch compile set
  (`ARCHITECTURE.md` section 9).
- **Levels drift between requests** by 3 to 13 us. A tuner that compares arms across requests will fit noise.
- **Put mode is a property of the communicator.** Nothing a tuner varies may change whether puts skip the
  credit check.
- Tuning on GB200 waits for item (d).

## (b) Recorded leads

Not built, or built and not adopted. Each with its measured size.

| lead | size | state |
|---|---|---|
| Ordered boundary 256 to 192 with more than one remote node | -6 to -12 at 192 to 224 tokens, three shapes | a candidate exists (not published); needs soak and crossing list; to be unified with the model's rule |
| Regime rule from the row stream's wire time | equal to the table at 800G; -14 to -17 at 384 tokens on GB200 2 x 8 | a candidate exists (not published); GB200 numbers on an earlier form of the item (d) candidate |
| Top-k term in the regime threshold | up to -5.5, top-6 only | crossovers at one remote node: 15 / 22 / 25 us of wire time on B300 for top-6 / 19 / 25; needs the sweep at three remote nodes |
| Retire by remote nodes per top-k | -1.1 / -6.9 at top-6, 2 x 8, 512 / 1024 | needs 768 timed and the gates of that plan family |
| Lead cap 7 on 400G | -3 to -8.6 on GB200 2 x 8 | blocked on the cause of the cliffs (section (a), cautions) |
| 16 records per put with one remote node | -1.9 at top-6, 1024, B300; 0 to +4 on GB200 | needs top-19 / 25 at 512 and 1024, Scan sizes, a soak |
| The tail schedule of "shared L2 early = off" without the code cost of the tail loop | about -2 at top-19, 4 x 8, 1024 | not built; see "Shared L2 early" in section (a) |
| Header as the first put of its Meta thread (today it follows the weights) | predicted -2 +- 1.5 at 4 x 8, 128 tokens; not measured in this form | not built; touches the same puts as item (d). Basis: at that point 17 of 35 us of exposure are at the start, where the counts wait for the header, and a thread's second put follows its first by about 5.7 us |
| Two row issuers per remote node in the Scan regime | -1.1 to -2.3 at 160 / 192 tokens, 4 x 8 | must stay one issuer in Ordered (+16 at 512); measured on an earlier tree of 13 Nov |
| This node's rows fill an expert's partial last pool block ("node rows last") | -5.1 at top-19 and -3.5 to -5.5 at top-25, 2 x 8, 1024; 0 to -1 at 4 x 8 | a candidate exists (not published; about 100 lines, default off); not adopted: it pays at one configuration only. Traced at 4 x 8: the freed CTAs wait on the next expert's full block instead |
| Proxy-sum batch re-poll (as the pull has) | predicted -2 +- 2; not measured | a candidate exists (not published) |
| Return order sent on the wire instead of sorted twice | not measured | would make invariant I10 hold by construction; a wire addition |
| First-need front-loading at 4 x 8 on slow links | model: 59 / 143 us of row wait at 512 / 1024 on 400G | none built. Model: a token has top-k / nodes experts on a node, so the share of a row stream needed through pool fraction x is 1 - (1 - x)^(top-k / nodes) |

Closed with a cause, recorded so they are not tried again:

| idea | result | cause |
|---|---|---|
| Local heads: an expert's partial block placed early in the pool, apart from its other blocks | +22 at top-19, 4 x 8, 512; +87 at 2 x 8, 1024 | separating an expert's blocks costs about 1.0 to 1.2 us per head. Rule for any pool layout: two blocks of one expert stay adjacent or at most 2 other pool blocks apart (step measured between 2 and 4 at 2 x 8, 1024 tokens: +4.8 at 2, +54.8 at 3, +66.7 at 4) |
| One wire format (sender tables) at every size | +10.4 at 4 x 8, 128; +8.5 at 2 x 8, 32 and 128 | the table chain sits in front of the counts. Scan cannot be dropped at small batch |
| Regime from the buffer's maximum token count | +12.2 at 2 x 8, 128 | same cause |
| A work-conserving L1 claim | no runnable unclaimed block in any sample | CTAs that wait, wait for rows (sampled at top-19 and top-25, 4 x 8, 512 and 1024) |
| Single-pass source combine | +5 to +8 at 1024 | `impls/mega_moe_ib/combine.cuh`, header comment |

Where the remaining distance is, for whoever picks the next lead (top-19, 4 x 8, zero-network arms):
the kernel without the wire is 25 to 41 us above the stock kernel on one B300 node at equal per-GPU work
(first routed claim +8 to +20, row wait 8 to 11 us per CTA even with no network because rows go through the
mirrors, tail +9 to +11); the tail policy costs nothing there. Network exposure on top is 35 us at 128 tokens
and 57 to 88 at 512 to 1024. (Zero network, top-19, 1024: plan 590.6, no retire 590.1, no cap 584.8, neither
599.4; stock on one node 549.0.)

## (c) Correctness and robustness

### Open

1. **Unexplained launch fault in some derived builds ("A2").** Several trees that add code to a passing
   build die with CUDA error 700 in their first launches, with no device message: always 32 GPUs, Ordered
   regime, and so far always under a host extension built from an earlier tree (the compat layer). Seen with
   three independent additions (a read-back loop with printf and trap in `ib::start`; a bound check in the
   table build; a row-watcher thread with a pull gate). The tree at `6ebf8b4` does not trigger it in any gate.
   Ruled out: a firing trap (that is error 719), a real return-order gap, stack size, code size, diagnostics,
   printf. Two hypotheses remain: the earlier host provides one word less than these builds touch, or a read
   in the base's own Ordered start is too early and only some timings expose it. The second would be a
   product bug. The discriminating run is one dying tree on the in-repo host at 32 GPUs, and the fault address
   of one death from the node's kernel log (the harness collects Xid lines into a failure result). **Until
   this is closed, do not put a printf plus trap in `ib::start`: count, and report at the next launch**, as
   `report_return_gaps` does.
2. **Gates on the in-repo host at 32 GPUs.** At this commit the 32-GPU diagnostics checks, soaks and
   back-to-back phase ran under the compat layer; on the in-repo host at 32 GPUs the mixed-regime lists and
   a first check ran. The full gate set there is open.
3. **Compat layer removal.** `layout/mega_moe_ib_compat.cuh` and every line marked `TEMPORARY`. One product
   change comes with it: the claim-after-start barrier becomes its own `SharedStorage` member and the host's
   shared-memory sum gains 8 bytes. That changes PTX, so the removal needs its own diagnostics checks and
   timing pairs. The kernel's trace prefix is renamed in the same commit (`tools/mega_moe_ib` already matches
   both prefixes).
4. **Back-to-back launches with different token counts per rank** were not run on a server (the harness mode
   is `DG_MEGA_MOE_DEV_ALT_TIMING=1`).
5. **Return records are fenced by lane 0 only** before a chunk put (`ib::reduce`, step 4), while staged rows
   are fenced by every writer. No failure in 3,000-launch soaks. Accept in writing, or time a fence in every
   lane.
6. **The old kernel's hang with few experts per rank** was never root-caused; the in-repo table build is
   written not to depend on the expert count and passes at 12 to 76 experts per rank. Recorded because the
   cause may be shared.

### Serving integration (from the audit of the previous tag; none blocks benchmarking)

| item | state | what it takes |
|---|---|---|
| Error path | traps only. A timeout (8 s in named polls, 60 s in waits) ends in a device trap that kills the CUDA context of every waiting rank, sometimes without its message. Epochs are compared for equality across ranks, so a restarted rank cannot rejoin | design item: a bounded wait that sets a status word the host reads, and a host-side group reset. At minimum document "any timeout = rebuild the whole group" and have the serving layer do it |
| NaN aliases the data sentinel | open as a contract. A token row or scale-factor unit of all-ones bytes reads as "not landed" and spins. The record path does not produce the pattern (the fp8 conversion writes NaN as 0x7F) | state "inputs are finite" as a contract; add a debug-mode host check |
| Packaging | the kernel needs NCCL 2.32.3 headers (`comm/gin.cuh` uses 2.32 internals), a C++ toolchain at run time (`ib.py` builds an extension inline), `DG_NCCL_INCLUDE_DIR` set before any DeepGEMM kernel is compiled in the process, and `NCCL_GIN_GDAKI_QP_DEPTH=1024` honoured (set with `setdefault`; a smaller value silently wins while every put skips the credit check) | pin NCCL in the image; set the include directory at import; prebuild the extension; assert the queue depth after the communicator is created |
| Gin progress threads | with a backend that uses host progress threads, launches stall by milliseconds unless each thread has a core of its own. The harness pins them (`gin_partition` in `tools/mega_moe_ib/jobs/devserver.py`) | move the pinning into the serving launcher |
| Topology detection | ranks per node is guessed as `min(group size, torch.cuda.device_count())` (`ib.py`, `ranks_per_domain`); a process that sees one GPU gets 1. Node-contiguous rank numbering is assumed, not checked | take it from NCCL's LSA team; assert the numbering |
| Warm-up call | the first call of a buffer shape compiles every kernel the buffer can launch and returns after one rendezvous (about 67 s cold and 9 s from the JIT disk cache for a 1,920-token buffer at 16 GPUs; 32 GPUs not measured). There is no exported warm-up function | one collective dummy call per buffer shape at start-up (zero tokens works); a `warm_up(sym_buffer)` wrapper; persist `DG_JIT_CACHE_DIR` |
| Development switches in serving builds | `DG_MEGA_MOE_IBP_*`, `DG_MEGA_MOE_IB_TRACE`, `DG_MEGA_MOE_IB_ORACLE` (a build that sends nothing after two launches: silently wrong output), `DG_DEV_COMPILE_ONLY` (the call becomes a no-op), `DG_MOE_NO_RENDEZVOUS`, `DG_MEGA_MOE_IB_VIRTUAL_LSA`, `DG_MEGA_MOE_IB_FORCE` are read in every process; an override with an empty value throws | gate them behind one development switch or compile them out |
| No cross-rank plan check | overrides are per process; ranks can disagree on everything but the regime | a plan hash in the header word, or compared once in the rendezvous |
| Limits surface late | the static asserts of `ARCHITECTURE.md` section 8 appear as a JIT error at the first launch; the start's scratch-size check is a device printf plus trap | mirror them as host asserts at buffer creation |
| `base=` buffer reuse | checks the stock byte count only; no IB tail, no sentinel fill | refuse it when IB is on unless the shape is identical |
| NCCL resources | `SymmBuffer.destroy()` drops references; the extension's `destroy` is never called | call it (about 5 lines) |
| Rendezvous residuals | launch state is an unsynchronised process global; the current kernel is marked before its family compiles; no rendezvous under CUDA graph capture | a few lines each; capture needs a decision |

Fixed since that audit and kept here as a record: the regime chosen per rank (regime bit); the epoch alias of
the blank header (a blank header no longer passes); the first-launch deadlock with mixed tile classes.

## (d) GB200 and ConnectX-7 class NICs

Status: **not usable from this tree; a candidate exists outside it.**

- The tree as is: correct on GB200 4 x 8 (single-domain reference check passes) and slow by orders of
  magnitude: 9.9 / 63.6 / 129.6 ms at 128 / 512 / 1024 tokens (top-19). Zero network on the same hardware:
  199.8 / 398.9 / 652.9 us. The fabric and image are healthy (the kernel that preceded this tree reads its
  old numbers there in the same session).
- Cause: "the load is the poll" (`comm/landing.cuh`). On this NIC class a bulk region that many GPU threads
  read over NVLink while it is still blank becomes visible only milliseconds later. Refuted as causes: a
  diagnostics build, a per-put timer, put size, context mapping, the Gin backend, rail bandwidth (47.4 GB/s
  into each GPU, 95 % of nominal). One local reader of blank bulk memory is free; many local warps polling
  blank words inside a bulk region are not safe either (7 launches of +1.2 ms in 900).
- A candidate exists (not published; on the previous tag; +80 / -4 lines):
  one plan field `gated`, default off; off is PTX-identical on 16 reference builds. On: a hint in front of
  each first read of bulk data (one watcher thread per source publishes rows landed, 16 at a time; one
  thread polls the last word of the index rows and of the tables; one flag word per return chunk), with
  every poll and the re-arm of this tree kept behind it, so correctness never rests on the hint. Proposed host
  rule: gated iff the link rate is known and at most 400G.
- Measured with the candidate, GB200 4 x 8, 13 Nov 2026, same session, 0 slow launches in 150 per point:

  | shape | tokens | candidate | kernel before this tree (6 Nov) | 32 GPUs in one NVLink domain (6 Nov) |
  |---|---|---|---|---|
  | top-19 | 32 / 64 / 128 | 194.9 / 213.1 / 271.0 | 238.8 / 267.6 / 333.2 | 177 / 185 / 206 |
  | top-19 | 256 / 512 / 768 / 1024 | 357.8 / 556.0 / 748.4 / 940.9 | 494.6 / 820.6 / - / 1,442.6 | 257 / 407 / - / 652 |
  | top-25 | 128 / 512 / 1024 | 313.1 / 662.3 / 1,112.5 | 388.0 / 939.6 / 1,645.0 | 258 / 528.5 / 835 |
  | top-6 | 128 / 512 / 1024 | 178.2 / 433.9 / 810.5 | 205.5 / 553.3 / 1,006.3 | 92 / 165 / 238 |

  Gates passed by the candidate on GB200: diagnostics checks at four sizes with the back-to-back phase,
  two changing-count lists, three soaks of 3,000. Not done: a rebase onto this commit (the header's regime bit
  must be masked where the candidate reads the token count), the host rule (needs the link rate in the handle,
  item (a) 1), and any gate on B300 with the field on.
- **On B300 the same idea in its first forms cost time and broke**: a flag behind every bulk put was +5.6 /
  +19.7 / +26.2 at 128 / 512 / 1024 (top-19, 4 x 8), and a rows-landed count used as proof was violated on
  B300 (rows read blank after their count). So this is one protocol selected by device kind, not one protocol
  everywhere, and on B300 the field stays off.
- What remains on GB200 after the candidate is link-shaped, not pathological: exposure 71 / 157 / 288 us at
  128 / 512 / 1024 (top-19): a start term of about 50 us, the row stream's front-loading, and a return tail
  bound by the wire. Leads recorded for it: the record flag inside the chunk's put (-10 / -16 / -23 at
  512 / 768 / 1024 by one table, with two hazards: the tail flag of a partial last chunk, and an in-bulk word
  polled by many warps); the regime boundary at 128 tokens for 4 x 8; lead cap 7; the header as a first put.
- 2 x 8 and 1 x 8 on GB200: only controls and plan-field sweeps exist (on an earlier form of the candidate);
  no table for this tree.
