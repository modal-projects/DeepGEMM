# Mega MoE across NVLink domains: architecture

This document describes the extension of the fused MoE kernel (`sm100_fp8_fp4_mega_moe`) to groups of ranks
that span more than one NVLink domain, with InfiniBand between the domains. It is written for a reader who
knows the stock kernel. Every statement about the code names the file and function it can be checked in; paths
are relative to `deep_gemm/include/deep_gemm/` unless they start with `csrc/` or `deep_gemm/`. The text
describes the tree at commit `6ebf8b4`.

Related documents: `REPRODUCTION.md` (how every number is produced), `TODO.md` (what is open).

## 1. Terms

| term | meaning |
|---|---|
| node | one NVLink domain: a set of ranks that reach each other by load / store through symmetric memory. `ranks_per_node` ranks each (`layout::IBPlan`). |
| remote node | a node other than the rank's own. `R = nodes - 1` (`ib::Geometry::kR`, `impls/mega_moe_ib/geometry.cuh`). |
| remote node slot `nr` | index in `[0, R)` that enumerates the other nodes in node order (`Geometry::node_of_slot`, `slot_of_node`). |
| counterpart | the rank with the same index on another node (`Geometry::counterpart`). All traffic between two nodes runs between counterparts. |
| source | a rank in its role as the owner of its tokens. |
| host | a rank in its role as the owner of its experts, as in the stock kernel. |
| proxy | a rank in its role as the stand-in, on its node, for its counterparts: it holds their inputs and returns their results. |
| Gin put | an RDMA write issued from device code through NCCL's GPU-initiated networking API (`comm/gin.cuh`). |
| mirror | the region on a proxy that a source's inputs are put into (`layout::IBBufferLayout`). |
| landing region | memory another party (a NIC or a peer GPU) fills while the kernel runs. |
| sentinel, blank | the value a landing region holds before it is written; data cannot equal it. |
| re-arm | to fill a landing region with its sentinel again. |
| epoch | a rank's launch counter for one buffer, starting at 1 (`ib::Launch::epoch`). Its lowest bit is the parity. |
| plan | `layout::IBPlan`: every build decision of an IB kernel that is not a template argument. |
| regime | Scan or Ordered: who builds the route tables and in which order token rows travel (section 4). |
| tail policy | how the scheduler orders the last routed tasks (`sched::SchedulePolicy`, section 7). |

An "IB build" is a kernel compiled with `DG_MEGA_MOE_IB`. A build without it is the stock kernel: every IB site
in a stock file is an added block under `#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB`, with the stock
statements kept in the `#else`.

## 2. Data path

In the stock kernel a host pulls each token row straight from its source over NVLink and writes each L2 output
row straight into the source's combine buffer. Across nodes there is no load / store path, and one put per
(token, expert) would scale wire traffic with top-k. The IB build routes everything between two nodes through
the counterpart pair instead:

```
 node A                                             node B
 ------------------------------                     ------------------------------------------------
 rank a (source)                                    rank b = counterpart of a (proxy for a)
   input region                                       mirror of a   [x | SF | top-k idx | weights |
     x rows, SF, top-k idx, weights                                  header | counts | tables]
        |                                                 ^   |
        |  (1) puts: idx, weights, header (Meta);         |   | (2) NVLink loads by every host of node B
        |      SF and x rows (Rows);                      |   v     that owns an expert of the token
        +------ Ordered: [counts | entries] (Tables) -----+  hosts b, b', ... : L1 GEMM, L2 GEMM
                                                              |
                                                              | (3) NVLink stores of L2 output rows
                                                              v     ("partials", bf16)
                                                          proxy_combine region on b
                                                              |
                                                              | (4) proxy sum on b: fp32 sum over the token's
                                                              v     experts on node B, quantised to fp8
   return landing  <------ (5) puts: one record per       fwd_stage on b
   (one record per           token (Return)
    token per remote node)
        |
        v (6) source combine on a: local experts + shared expert + one record per remote node -> y
```

1. **Rows out once per remote node.** A source puts its top-k index rows, top-k weight rows and a header, then
   its scale factors and token rows, into its mirror on each counterpart (`ib::start` in
   `impls/mega_moe_ib/start.cuh`, sender part). Nothing is sent per expert.
2. **Node-local pulls.** A host that needs a row of a remote source loads it from that source's counterpart on
   its own node, over NVLink (`ib::pull_row`, `impls/mega_moe_ib/pull.cuh`). Rows of node-local sources are
   loaded from the source as in the stock kernel.
3. **Partials to the proxy.** The L2 epilogue stores an output row of a remote source's token into the
   `proxy_combine` region on the source's counterpart (`ib::store_partial`, `impls/mega_moe_ib/proxy.cuh`),
   in the layout of the stock combine buffer: `[top-k slot][token][hidden]` bf16.
4. **Proxy sum.** The proxy sums a token's partials of its node in fp32, in top-k order, and quantises the sum
   to fp8 e4m3 with one f32 scale per 128 columns (`ib::reduce`, `proxy.cuh`).
5. **One return record per token per node.** Records are put back to the source, 8 or 32 per put
   (`Geometry::kNumRecordsPerPut`).
6. **Source combine.** The source adds its local experts' rows (stock loop), then one record per remote node
   (`ib::add_records`, `impls/mega_moe_ib/combine.cuh`).

Consequences:

- **Wire traffic does not depend on top-k.** Per launch a rank sends, to each remote node, `tokens x hidden`
  bytes of rows plus scale factors, indices and weights, and receives `tokens x record_bytes` back
  (`IBBufferLayout::record_bytes` = `hidden + hidden / 128 * 4`).
- **Outputs match the stock kernel to a tolerance, not bit for bit.** A remote node's contribution passes
  through an fp8 record. The in-repo test compares with a `calc_diff` bound (`tests/test_mega_moe.py`,
  `DG_MEGA_MOE_IB_TOLERANCE`).
- **A token with no expert on a node still costs that node a row and a record.** Every byte written is read
  (section 5), so the row is polled and an all-zero record is returned.

### Buffer

All IB state is appended to the stock symmetric buffer as a tail (`layout::IBBufferLayout`,
`layout/mega_moe_ib.cuh`); the stock layout is unchanged.

```
tail = [ MegaMoEIBSignals ] [ node slot 0 ] [ node slot 1 ] ...        one slot per node, same on every rank

node slot n on a rank of node m != n  (written by / returned to the counterpart on node n):
  mirror[parity 0], mirror[parity 1]    x rows | SF | top-k idx | weights | header | counts | entry tables
  ret[parity 0], ret[parity 1]          return records, return header
  proxy_combine                         partials of the tokens this rank proxies
  fwd_stage                             records staged for the return puts
```

The whole buffer is one NCCL window with the same layout on every rank, so a put's remote address is the
local address's offset (`comm::GinOf::offset`). The slot of a rank's own node is never a NIC target. It serves
as the source address of every forward put (`Slots::outbox_mirror`, `outbox_ret` in `geometry.cuh`: bytes at
offset `o` of the own slot go to offset `o` of the peer's slot for this node) and as scratch: `row_stage(nr)`
at the start of its `proxy_combine` region, `combine_scratch` at the end, with a static assert that the two do
not overlap (`ib::Slots`).

`MegaMoEIBSignals` holds GPU-local words only: the epoch, flags, work-queue counters and per-GPU tables. Other
nodes never write there.

Python allocates the buffer and fills the sentinels (`SymmBuffer.__init__` in `deep_gemm/mega/__init__.py`,
`allocate` in `deep_gemm/mega/ib.py`); it takes every offset from the same struct through
`get_ib_tail_for_mega_moe` (`csrc/apis/mega_moe.hpp`). The Gin handles reach the host as five extra entries
behind the per-rank pointers (`layout::kIBMagic`, `layout::IBHandles`) and the kernel as one extra argument.
No Python or C++ signature of the stock API changes.

### Streams

A put carries one of four streams (`comm::Stream` in `comm/gin.cuh`): Meta, Tables, Rows, Return. The stream
selects the Gin context (`GinOf::context_of`); Return spreads over the remaining contexts by chunk and node
(`ib::make_gin`). `comm/gin.cuh` states the three properties the kernel relies on: a put says nothing about
when its bytes land; puts issued by one thread on one stream to one peer land in issue order; a put may read
its source after it returns. Every put skips the send-queue credit check
(`ncclGinOptFlagsMaySkipCreditCheck`); this is a property of the communicator, not of a build (`ib.py` sets
`NCCL_GIN_GDAKI_QP_DEPTH=1024`).

## 3. Timeline of a launch

Warp roles are the stock ones: dispatch warps, MMA / loader warps, the scheduler warp, epilogue warps. Stock
steps are in plain text, IB additions are marked `+`.

```
dispatch warps                               epilogue warps                         scheduler
---------------------------------------      ----------------------------------     ----------------------
+ reset_counters (SM 1)                      + ib::start, role by SM index:         waits for start_barrier
  count own top-k, write slot entries          SM 0        post Meta, return order  (kClaimAfterStart)
  and counts to ranks of THIS node             SMs 1..R    post SF + rows
+ node_barrier (ranks of this node)            Ordered:    order SMs, stage SMs,
  fetch_expert_recv_count                                  8 table-build SMs
  (waits for every source's counts)            last SMs    receive one remote
+ pull loop on SMs [kFirstPullSM,                          source each
  kEndPullSM): slot_entry (await             + start_barrier.arrive ------------->  first claim
  tables_ready), pull_row (poll)               GEMM tasks: L1, L2                    L1 / L2 tasks under the
+ reduce: proxy sum, return puts             + store_partial for remote sources      tail policy
                                               L2 grid sync (join early: warp 0
                                               only)
                                             + reduce: proxy sum, return puts
  ---- dispatch / epilogue CTA barrier (both sides) ----
  clean workspace                              source combine, loop 1: local
+ node_barrier                                 experts, + park fp32 sums
+ rearm_mirrors (waits for all               + add_records: poll, add, re-arm,
  return chunks), epoch store                  cast, store y
```

Sites in `impls/sm100_fp8_fp4_mega_moe.cuh`, in kernel order: `ib::Launch` construction after `workspace`;
`reset_counters`, `trace::begin`, `report_return_gaps` at the start of the dispatch warps; skips of other
nodes' ranks in the slot-entry, grid-index and count pushes; `comm::node_barrier` in place of
`comm::nvlink_barrier` (twice); `__threadfence()` after `fetch_expert_recv_count`; the pull loop bounds;
`ib::slot_entry` and `ib::pull_row` in place of the stock per-row body; `ib::reduce` after the pull loop;
`ib::rearm_mirrors` and the epoch store after the final barrier; `scheduler.mainloop(..., &start_barrier)`;
`ib::start` and `start_barrier.arrive` before the epilogue's first task; `ib::store_partial` in the L2
epilogue; the join-early grid sync; `ib::reduce` after it; `ib::local_slot`, `ib::park`, `ib::add_records`,
`ib::await_empty_returns` in the combine.

## 4. The two regimes

A host cannot pull a remote row before it knows which pool slot of which expert the row belongs to. In the
stock kernel the source writes these slot entries and per-expert counts directly into every host. Across nodes
somebody else must produce them. The two regimes differ in who does, and in the order of the token rows.

| | Scan | Ordered |
|---|---|---|
| plan | `ordered = false` | `ordered = true` |
| who routes | the receivers: each host scans the source's top-k rows in the proxy's mirror (`ib::scan_route`, `impls/mega_moe_ib/route.cuh`) | the sender: 8 SMs build, per remote node, one message `[counts | entries]` and put it (`ib::build_tables`); each host copies its slice (`ib::receive_tables`) |
| earliest start | after the index rows have landed | before anything is on the wire |
| row order in the mirror | token order | first-need order |
| extra wire | none | counts and entries, 4 bytes per (token, expert) on remote nodes |
| extra start work | none | token sort per remote node, row staging, table build |

### Entry word

In both regimes a slot table entry of a remote source is one 32-bit word: `token * top-k + k` in the low 16
bits and the token's mirror row in the high 16 bits (`ib::slot_entry`, `pull.cuh`; written by `scan_route` with
row = token, and by `build_tables` with row = `need_pos`). The pull therefore has one decode for both regimes.
Only the x row is permuted; scale factors, weights and metadata are addressed by token (`ib::pull_row`).

### First-need order

Every rank runs its experts in index order. For a token and a remote node, the key is the smallest in-rank
expert index among the token's experts on that node (`experts per rank` if it has none): an estimate of when
that node first needs the row. The table build takes the keys in its count sweep (`build_tables`, step 1,
`atomicMin` into `need_keys`); one SM per remote node sorts the tokens by key (`sort_by_key`,
`impls/mega_moe_ib/sort.cuh`) into `need_pos` (token to mirror row) and `need_inv` (mirror row to token); idle
SMs copy the rows in that order into `row_stage(nr)`, 32 rows per chunk, and raise `rows_staged[nr][chunk]`
(`ib::start`, "First-need order" block); the row poster of the node waits for the chunks of each 128-row put
and puts from the stage. The rows of one node are issued by one thread on one stream, so they land in order.

### Return order

Both ends of a (source, node) pair need the source's tokens in the order in which the node will finish them:
the proxy returns records in that order and the source looks its records up by it. The key is the largest
in-rank expert index among the token's experts on the node (0 without one). The sort is a stable counting
sort, so the proxy, sorting the index rows in its mirror (`scan_route` takes the keys, `ib::start` sorts into
`return_perm`), and the source, sorting its own inputs (`build_return_keys`, `sort_by_key` into
`return_pos`), get the same order without exchanging it (`sort.cuh`, header comment).

### The regime bit

The host chooses the regime per rank from that rank's own token count (`get_mega_moe_ib_plan`,
`csrc/jit_kernels/heuristics/mega_moe.hpp`), so ranks of one launch can be in different regimes. A source
therefore announces its regime in its header word: `(epoch << 32) | regime bit | token count`, with the regime
bit at bit 31 (`Geometry::kHeaderOrdered`; put in `ib::start`, sender part). Every receiver reads the bit and
takes that source in the regime it announces (`is_src_ordered` in `ib::start`, receiver part): `receive_tables`
for an Ordered source, `scan_route` with routing for a Scan source. Both receive paths are compiled into every
IB build. The proxy stores the bit with the token count in `signals->src_num_tokens[node]`, and
`rearm_mirrors` re-arms the table and count regions only for a source that announced Ordered. A static assert
keeps token counts below the bit, so a header's low word is never all ones.

What the sender does (staging, table build, order SMs) is compiled only into an Ordered build
(`if constexpr (G::kOrdered)`).

## 5. Landing

File: `comm/landing.cuh`.

A put gives no completion signal to the receiving GPU. Instead every landing region is pre-filled with a
sentinel, and a reader loads a unit and looks at it: **the load is the poll**. Units and sentinels:

| data | unit | blank value |
|---|---|---|
| token rows, scale factors, partials, records | 16 bytes | all ones (an fp8 / bf16 NaN pattern) |
| top-k indices | 8 bytes | `INT64_MIN` (`kBlankIdx`) |
| weights, counts, table entries | 4 bytes | all ones (`kBlankWord`) |
| headers | 8 bytes | all ones |

Two rules make this a protocol (header comment of `landing.cuh`):

- **use**: a unit is polled against its sentinel before its value is used;
- **landing**: every unit written in a launch is polled by some reader before its region is re-armed and
  before the writer's source bytes may change. A reader that proves landing polls every unit of its region,
  whether or not it needs the value.

Forms: `landing::await` re-reads one unit until it differs (sites where the unit has normally landed);
`landing::await_all` re-reads all units of a batch that are still blank in every round, so a wait ends about one
round trip after the last unit lands. `pull_row` issues all loads of a row first and uses the batch form for
the row. `landing::spin<kDiag>` is the loop under both: in a diagnostics build (and at the sites that pass
`true`: the top-k scan, table counts and entries, the record poll) it prints the site and traps after about 8 s;
in a timed build the hot polls are a bare spin, bounded by the waits around them (`comm::wait_until`, 60 s, in
grid syncs, the node barrier, flag waits, the header wait and the re-arm wait).

Who proves what has landed:

| stream | units | proven by |
|---|---|---|
| Meta | every index entry and weight of a source | the proxy's pass in `scan_route` (`is_proxy`), in both regimes |
| Meta | header | `ib::start`, receiver part: one thread per receiver SM, 200 ns sleep between polls |
| Rows | every token row and scale factor | `ib::reduce`, step 1, for every token whether or not it has an expert on the node |
| Tables | every count; every entry | `receive_tables`: every receiver polls all counts of its node; each entry is polled by the rank that owns it |
| partials (NVLink) | every partial row | `ib::reduce`, step 2 (polled, then re-armed in place) |
| Return | every record of every remote node; the header of a zero-token source | `ib::add_records` (polled and re-armed in place, hit or not); `ib::await_empty_returns` |

Re-arm (`ib::rearm_mirrors`, `proxy.cuh`) runs after the node's final barrier. Per remote node it first waits
until `fwd_chunks_returned` reaches the chunk count of the source: then the proxy sum has polled every row and
the intake before it every index and weight. It re-arms exactly the rows that were sent (`num_tokens` of the
source), with `landing::rearm`, which also writes the bytes that do not fill a last 16-byte unit. A diagnostics
build traps on a unit that is still blank when it is re-armed (not for the table and count regions, where blank
units are legitimate).

Inside one GPU, "these stores are visible" is told by an epoch flag: every writer calls `__threadfence()`, the
CTA synchronises, one thread stores the launch's epoch into the flag word (`landing::raise`); a reader waits
until the word equals its epoch (`landing::await_raised`, with the 60 s timeout). Flags are never reset.
Flags: `tables_ready[source]`, `order_ready[node]`, `rows_staged[nr][chunk]`, `sort_shared` / `sort_done`
(`MegaMoEIBSignals`). A flag's writer and readers are always on the same GPU; what other GPUs need is proven
by data.

### Parity and epochs

Mirrors and return landings exist twice, selected by the parity of the epoch (`ib::Slots::mirror`, `ret`), and
the work-queue counters have one row per parity. A source's header carries its epoch; a receiver accepts a
header only if it is not blank and its high word equals the low 32 bits of the receiver's epoch (`ib::start`).
The epoch word is stored after the re-arm (kernel, after `rearm_mirrors`). Every source posts a header every
launch, also with zero tokens; a zero-token source's proxies return a header instead of records
(`ib::reduce`), which the source awaits and re-arms.

### Limit on ConnectX-7 class NICs

"The load is the poll" was developed and gated on B300 nodes with ConnectX-8 800G NICs. On GB200 trays with
ConnectX-7 400G NICs the same tree is correct but unusably slow: a bulk region that many GPU threads read
while it is still blank became visible only milliseconds later (about 10 ms per launch at 128 tokens on 32
GPUs; `REPRODUCTION.md`, "GB200"). Small puts were not affected, and a single local reader of blank bulk
memory was free. The tree at this commit contains no mitigation. `TODO.md` section (d) records the status of
the candidate (a plan-selected hint that delays the first read of bulk data until it is likely to have
landed, with the polls above kept behind it).

## 6. The pull, the proxy sum and the combine

**Pull (`pull.cuh`).** The stock pull loop stays (expert cursor, slot arithmetic, ring wait, bookkeeping).
`ib::slot_entry` waits once per (warp, remote source) for `tables_ready`. `ib::pull_row` replaces the stock
TMA copy for every row in an IB build: lane `l` of the warp owns the 16-byte units `l, 32 + l, ...`; all loads
of the row, its scale factors and its weight are issued first; remote units are polled; generic stores into
the L1 ring are followed by `fence.proxy.async` because the GEMM reads them by TMA. SMs whose epilogue warps
are busy at the start take no part in the pull (`kFirstPullSM`, `kEndPullSM`).

**Proxy sum (`proxy.cuh`, `ib::reduce`).** A work item is (return position, quarter of the hidden dimension),
claimed from one counter per source node with the scheduler's own claim primitive
(`MegaMoEScheduler::get_next_local_task_idx` on `fwd_item_claim`). A warp handles one item: landing proof of
the token row part; for each of the token's experts on this node, poll the partial, re-arm it, add in fp32;
quantise; store into `fwd_stage`; count the item in `fwd_chunk_done`. The warp that completes a chunk of
`kNumRecordsPerPut` records puts it and counts it in `fwd_chunks_returned`. The item loop waits for
`order_ready[node]` first and reads the source's token count only after it. Two call sites: the dispatch warps
after their pulls, and the epilogue warps after their last GEMM task. `first_slot` rotates the node order per
warp so that all return streams advance together. The counters of both parities are zeroed at kernel start
(`ib::reset_counters`).

**Source combine (`combine.cuh`).** Two passes. Pass 1 is the stock loop with two hooks: `ib::local_slot`
turns top-k slots whose expert is on another node into "no slot", and `ib::park` stores the fp32 sums of the
local part into `combine_scratch` instead of casting. Pass 2, `ib::add_records`, reloads the sums and, per
remote node, loads the token's record at `return_pos`, re-reads as a warp whatever has not landed, adds the
dequantised values if the token has an expert on that node, re-arms the record, then casts and stores as the
stock loop does. The header comment of `combine.cuh` gives the reason for two passes and its measured size.

## 7. Scheduler tail policy

File: `scheduler/mega_moe.cuh`. `MegaMoEScheduler` takes one more template argument, a policy type with five
members (`sched::SchedulePolicy`). Every member of the default selects the stock code at compile time. An IB
build passes `Geometry::TailPolicy`, filled from the plan.

| member | effect |
|---|---|
| `kL1LeadCap = C` | after an L2 task the next task is L1 only while L1 leads L2 by at most C pool blocks (claimed tasks of the locality domain); keeps L2 completions, which feed the proxy sum, steady to the end (`get_next_task`) |
| `kRetirePercent`, `kRetireAtPercent` | that share of the clusters, evenly spread, stops claiming once that share of the routed L2 tasks of its locality domain is handed out (`should_retire`). Their warps join the proxy sum |
| `kSharedL2Early` | the shared expert's L2 always runs before the routed tasks (`mainloop`) |
| `kClaimAfterStart` | the scheduler makes its first claim only after the cluster's epilogue threads have left `ib::start` (`mainloop` waits on `start_barrier`). Always true in an IB build |

A build that retires clusters also joins early (`Geometry::kJoinEarly`): after its last task only epilogue
warp 0 of an SM waits in the L2 grid sync; the other warps go to `ib::reduce`. The dispatch / epilogue CTA
barrier still holds every warp back from the combine until warp 0 is through.

In an IB build the claim-after-start barrier shares storage with `dispatch_barriers` in an anonymous union
(kernel, `SharedStorage`); an IB build has no pull barriers. The comment at the site says why and when it
becomes a member of its own.

## 8. The plan

`layout::IBPlan` (`layout/mega_moe_ib.cuh`) is filled by `get_mega_moe_ib_plan`
(`csrc/jit_kernels/heuristics/mega_moe.hpp`) and printed into the JIT source as the macro
`DG_MEGA_MOE_IB_PLAN` (`get_mega_moe_ib_defines`, `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`). The
kernel reads it in one place (`ib::BuildPlan`). No IB header reads another macro, except the temporary
`DG_MEGA_MOE_IB_COMPAT_ABI` (section 11).

| field | what it does | set by |
|---|---|---|
| `ranks_per_node` | node arithmetic and every per-node loop bound, at compile time | the buffer's handle (`IBHandles::ranks_per_node_of`) |
| `ordered` | the regime of this rank as a sender; SM roles | token count against the fitted boundary: 512 with one remote node, 256 with more |
| `gin_backend` | compiles one Gin backend's put instead of a run-time dispatch over all | the communicator (`ib.py`, `create`); the host asserts an override equals it |
| `l1_lead_cap` | `kL1LeadCap` | 9 in the Ordered regime, else 0 |
| `retire_percent` | `kRetirePercent`; also selects join-early | 50 in the Ordered regime with more than one remote node, else 0 |
| `retire_at_percent` | `kRetireAtPercent` | `round(100 x 0.6^(nodes / top-k))` when clusters retire |
| `shared_l2_early` | `kSharedL2Early` | on wherever clusters retire |
| `diagnostics` | timeouts with a site name in hot polls, re-arm trap, return-order count | 0; development override |
| `trace` | time stamps (`impls/mega_moe_ib/trace.cuh`) | false; development override |
| `zero_network` | from the third launch on: no puts, no re-arm, epochs not checked (`Geometry::is_network_live`, `kRearm`). A measurement of the kernel without the wire; valid only for identical inputs every launch | false; development override |

Fitted values are in one struct (`Fitted` in `get_mega_moe_ib_plan`), with the measurements that set them in
the comment beside it. They were fitted on B300 nodes at two and four nodes; `TODO.md` section (a) lists them
and the evidence of how far they carry.

One more fitted constant is in device code: `Geometry::kNumRecordsPerPut` = 8 with more than one remote node,
32 with one (`geometry.cuh`, with its measurement).

Development overrides: one environment variable per field, `DG_MEGA_MOE_IBP_<FIELD>`; a name that is no field
is refused. `DG_MEGA_MOE_IB_TRACE` and `DG_MEGA_MOE_IB_ORACLE >= 2` also set `trace` and `zero_network`.

Limits enforced by static asserts (they surface as a JIT compile error at the first launch): at most 16 nodes
and 64 remote source ranks (`MegaMoEIBSignals`); `tokens x top-k < 65536` (`slot_entry`, `build_tables`); at
most 256 return chunks per node, that is 2,048 tokens per rank with more than one remote node and 8,192 with
one (`Geometry::kNumMaxChunks`); at most 64 staged chunks of 32 rows in an Ordered build (2,048 tokens); at
most 32 ranks in a build that retires clusters (`kJoinEarly`); enough SMs for the start roles.

### SM roles at the start

`ib::start` dispatches by SM index only, because every start step uses 256-thread epilogue barriers
(`Geometry`, "Which SM does what"):

| SMs | role |
|---|---|
| 0 (`kMetaSM`) | per remote node two threads on Meta: one puts the index rows, the other the weight rows and then the header (which therefore lands after the weights); then the return order per remote node |
| `[1, 1 + R)` | one thread each: scale factors, then token rows in puts of 128 rows, for one remote node |
| `[kFirstPullSM, kFirstOrderSM)` (Ordered) | stage rows in first-need order, one 32-row chunk at a time |
| `[kFirstOrderSM, kFirstBuildSM)` (Ordered) | one SM per remote node: sort the tokens by first-need key |
| `[kFirstBuildSM, kFirstReceiverSM)` (Ordered) | 8 SMs: the table build |
| last `ranks - ranks_per_node` SMs | one remote source each (`Geometry::source_of_receiver`); the last `R` of them are the sources this rank is the proxy of and also do the proxy intake (token count, landing proof of indices and weights, return order, `order_ready`) |

## 9. First launch

Ranks JIT-compile on their own and can finish tens of seconds apart, while the kernel's waits on its peers
time out after 60 s. The host therefore does not launch the first time a multi-rank call meets a kernel object
this process has not launched: it sets a flag, and the Python wrapper barriers the group and calls again
(`mega_moe_launch_skipped` in `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`; `_call_with_rendezvous` in
`deep_gemm/mega/__init__.py`).

The barrier is a group collective, so every rank must enter it in the same call. A rank whose token count
crossed a tile class or the regime boundary alone would otherwise meet a new kernel object alone. So on that
first call the host compiles every kernel the buffer can launch: it builds the source for every token count
from 0 to the buffer's maximum, de-duplicates by source text, compiles each distinct one and marks it as
launched. After the one rendezvous no call of that buffer shape meets a new kernel object. The cost is
first-call time (one compile per tile class and plan; the JIT disk cache keeps them); `TODO.md` section (c)
lists the warm-up item. Under CUDA graph capture the launch is not skipped.

## 10. Invariants

The numbers are labels used in code review and in `TODO.md`; the text is the state at this commit. Each line names where the
invariant is enforced.

| # | invariant | where |
|---|---|---|
| I1 | Use: every NIC- or peer-written unit is polled against its sentinel before it is used. | `comm/landing.cuh` and the polls in `pull_row`, `reduce`, `scan_route`, `receive_tables`, `add_records` |
| I2 | Staged bytes are visible before the NIC reads them: every writer fences (device scope), the CTA synchronises, a flag is raised, then the put. | staging loop in `ib::start`; `build_tables` steps 2, 5, 6 |
| I3 | Nothing writes a staging region while a put may still read it: `combine_scratch` and `row_stage` do not overlap; across launches by I18 (b). | static assert in `ib::Slots`; I18 |
| I4 | Rows at or beyond a source's token count stay blank: senders put rows below the count only; receivers re-arm exactly what they were sent. | `ib::start` sender; `rearm_mirrors` |
| I5 | An odd number of index entries leaves 8 bytes that no 16-byte unit covers; the re-arm writes them. | `landing::rearm` |
| I6 | Valid data never equals a sentinel. Asserted for the packed entry and the header's low word; for fp8 / bf16 / f32 payloads it is a contract on the inputs (`TODO.md` section (c)). | static asserts in `slot_entry`, `build_tables`, `Geometry` |
| I7 | Epochs: the epoch word is stored after the re-arm; parity selects the mirror / return set and the counter rows; a header is `(epoch << 32) | regime bit | tokens`; a blank header never passes as this launch's; flags hold the epoch and are never reset; every source posts a header every launch, also with zero tokens, and a zero-token source's return header is awaited and re-armed. | `ib::Launch`; `ib::start`; kernel after `rearm_mirrors`; `reduce`; `await_empty_returns` |
| I8 | Count arrival: every remote source adds exactly `kNumSMs` of share per expert: store, fence, atomic add; the dispatch side fences after its wait. | `publish_counts`; kernel after `fetch_expert_recv_count` |
| I9 | The slot table of a remote source is complete before a pull reads it. Scan: table, fence, flag, then counts. Ordered: entries copied, fence, flag. | `tables_ready` in `ib::start`, `receive_tables`; `slot_entry` |
| I10 | The proxy's `return_perm` and the source's `return_pos` are the same order: one stable sort of the same rows. A diagnostics build counts empty positions and the next launch prints a non-zero count. | `sort_by_key`; `report_return_gaps` |
| I11 | First-need rows: the entry carries the mirror row; only x is permuted; metadata holds the true source rank; the rows of one node are issued by one thread on one stream. | `build_tables`; `pull_row`; `ib::start` row poster |
| I12 | A build serves a range of token counts; a count and its odd neighbour run the same cubin; what does not fit fails a static assert at compile time. | section 8, limits |
| I13 | The dispatch / epilogue CTA barrier before the combine holds back epilogue warps 1-7 until this GPU's L2 grid sync (join-early form) and ends this SM's proxy-sum items. | stock barrier; join-early block in the kernel |
| I14 | Proxy-sum counters are zero before the first claim. | `reset_counters`, before the dispatch grid sync every claimant passes |
| I15 | First-launch rendezvous: every rank enters it in the same call. | section 9 |
| I16 | Every start barrier is a 256-thread epilogue barrier: a role that contains one is entered under an SM-uniform condition. | `ib::start` dispatches by `sm_idx`; the per-source regime branch is uniform over an SM |
| I17 | `fence.proxy.async` after the pull's generic stores and before `l1_full_count` is raised. | `pull_row` |
| I18 | Landing: every put of a launch has landed before (a) its landing region is re-armed and (b) its source bytes may be rewritten. (a): section 5, proof table and the preconditions of `rearm_mirrors`. (b): a source's combine does not finish before the last record of every proxy; a proxy's last chunk is put after all its items; an item starts after `order_ready` and polls the token row; a record needs the partials of all the token's experts, hence their table entries. | `reduce`, `scan_route`, `receive_tables`, `add_records`, `rearm_mirrors` |
| I19 | A proxied source's token count and regime bit are published with its return order: stored before the fence, barrier and raise of `order_ready`, and read only after acquiring that flag. | `ib::start` proxy part; `reduce`. `rearm_mirrors` reads it after every dispatch warp has passed `reduce` |
| I20 | (added in v1.2) Regime per source: a receiver treats each source in the regime that source's header announces; a receiver of an Ordered source polls that source's tables only after its header. | `ib::start` receiver part |

Gate 1 is not an invariant of the IB path but of the change as a whole: with IB off, the kernel's PTX equals
the stock kernel's (`REPRODUCTION.md`, "Gate list").

## 11. File map

Line counts at `6ebf8b4`. "+n" is the number of lines added to a stock file (`git diff --stat 5081f85 6ebf8b4`).

| file | lines | content |
|---|---|---|
| `layout/mega_moe_ib.cuh` | 230 | `IBPlan`, `IBHandles`, `MegaMoEIBSignals`, `IBBufferLayout` |
| `comm/gin.cuh` | 105 | `Stream`, `GinOf` (put, put_value), single-backend helpers |
| `comm/landing.cuh` | 146 | sentinels, `load`, `spin`, `await`, `await_all`, flags, `rearm` |
| `comm/barrier.cuh` | 135 (+41) | `node_barrier` |
| `impls/mega_moe_ib/geometry.cuh` | 205 | `Geometry` (topology, roles, table offsets, `TailPolicy`), `Launch`, `make_gin`, `Slots` |
| `impls/mega_moe_ib/start.cuh` | 311 | `start` (all sender and receiver roles), `report_return_gaps` |
| `impls/mega_moe_ib/route.cuh` | 324 | `publish_counts`, `scan_route`, `build_tables`, `receive_tables` |
| `impls/mega_moe_ib/sort.cuh` | 116 | `build_return_keys`, `sort_by_key` |
| `impls/mega_moe_ib/pull.cuh` | 106 | `slot_entry`, `pull_row` |
| `impls/mega_moe_ib/proxy.cuh` | 236 | `store_partial`, `reset_counters`, `reduce`, `rearm_mirrors` |
| `impls/mega_moe_ib/combine.cuh` | 168 | `local_slot`, `park`, `add_records`, `await_empty_returns` |
| `impls/mega_moe_ib/trace.cuh` | 80 | trace stamps (compiled only with `plan.trace`) |
| `impls/sm100_fp8_fp4_mega_moe.cuh` | 1,717 (+189) | the hook sites of section 3 |
| `scheduler/mega_moe.cuh` | 569 (+98) | `SchedulePolicy` and its four mechanisms |
| `layout/mega_moe.cuh` | (+4) | compat padding only (temporary) |
| `layout/mega_moe_ib_compat.cuh` | 147 | temporary: the buffer and argument layout of hosts built before this tree |
| `csrc/jit_kernels/heuristics/mega_moe.hpp` | 365 (+100) | `get_mega_moe_ib_plan`, `to_string(IBPlan)` |
| `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` | 302 (+106) | plan define, handles argument, compile-only hook, first-launch rendezvous |
| `csrc/apis/mega_moe.hpp` | 425 (+27) | `get_ib_tail_for_mega_moe`, tail size check, three bindings |
| `csrc/runtime/jit.hpp` | (+4) | `DG_NCCL_INCLUDE_DIR` on the nvcc command line |
| `deep_gemm/mega/ib.py` | 198 | NCCL window, device communicator, handles |
| `deep_gemm/mega/__init__.py` | 256 (+44) | tail allocation, sentinel fill, rendezvous wrapper |
| `tests/test_mega_moe.py` | 509 (+75) | tolerance check, single-domain reference |

The compat layer (`DG_MEGA_MOE_IB_COMPAT_ABI`, every line marked `TEMPORARY`) lets this kernel run under a
host extension built from an earlier tree, which is how it was timed against the earlier kernel in one
request. The in-repo host never defines the macro. Its removal is a listed item (`TODO.md` section (c)).
