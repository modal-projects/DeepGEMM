#pragma once

// Mega MoE across NVLink domains ("IB"): everything the host, Python and the kernel must agree on.
//
// Ranks of one NVLink domain (a node) keep talking through load/store exactly as in the stock kernel. Ranks of
// other nodes are reached with NCCL Gin puts from inside the kernel. Nothing here changes the stock layout:
//   * the handles Gin needs travel behind the per-rank pointers in `sym_buffer_ptrs` (`IBHandles`), and reach the
//     kernel as one extra argument that only IB builds have; every build decision is one constant (`IBPlan`);
//   * all IB state lives in a tail appended to the symmetric buffer, after the bytes the stock kernel uses
//     (`IBBufferLayout`): first the IB signals, then one slot per node.
// A build without DG_MEGA_MOE_IB sees none of it.

#include <cstddef>
#include <cstdint>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe_ib_compat.cuh>  // TEMPORARY

namespace deep_gemm::layout {

// `sym_buffer_ptrs` from Python is [per-rank load/store pointers (0 for ranks of other nodes)], followed, for an IB
// group, by [kIBMagic, device pointer to ncclDevComm_t, ncclWindow_t, ranks per NVLink domain].
constexpr static int64_t kIBMagic = 0x1B1B1B1B1B1B1B1Bll;
constexpr static uint32_t kNumIBExtraPtrs = 4;

// Gin contexts the kernel spreads its puts over; Python requests as many when it creates the device communicator
constexpr static uint32_t kNumIBGinContexts = 8;

// Who builds the per-expert counts and slot tables of a remote source: the receivers, by scanning the top-k rows
// in the proxy's mirror, or the sender, which then puts one table message per node
enum class IBRouting : uint32_t { ReceiverScan = 0, SenderTables = 1 };

// Where a source's scale factors travel: on the Rows stream in front of the token rows, on the Rows stream behind
// them, or on the Meta stream (behind the top-k indices), where they neither wait for the rows nor delay them
enum class IBScaleFactors : uint32_t { BeforeRows = 0, AfterRows = 1, WithMeta = 2 };

// Every decision of an IB build that is not a template argument of the kernel. The host heuristics fill it
// (`get_mega_moe_ib_plan`) and print it into the JIT source as `DG_MEGA_MOE_IB_PLAN`; the kernel reads it as a
// constant. No header reads any other macro, so two builds differ exactly when their plans or shapes do.
struct IBPlan {
    uint32_t ranks_per_node = 1;          // ranks per NVLink domain; remote nodes R = ranks / ranks_per_node - 1
    IBRouting routing = IBRouting::ReceiverScan;
    bool first_need_order = false;        // rows are staged and posted in first-need order (needs SenderTables)
    uint32_t row_issuers = 1;             // threads (one SM each) that post a remote node's token rows; each keeps its puts in order
    uint32_t rows_per_put = 128;          // token rows per forward put
    bool contexts_per_node = false;       // Gin contexts by (stream, remote node) instead of by stream (comm/gin.cuh)
    IBScaleFactors scale_factors = IBScaleFactors::BeforeRows;
    // Tail policy, see sched::SchedulePolicy
    uint32_t l1_lead_cap = 0, retire_percent = 0, retire_at_percent = 100;
    bool shared_l2_early = false;
    // Measurement builds
    uint32_t diagnostics = 0;             // 1: hot polls time out with a site name, stage marks per SM
    bool trace = false;                   // per-stage time stamps
    bool zero_network = false;            // no puts, no re-arm, epochs not checked: the zero-network ceiling
};

// What Gin needs. Reaches the kernel as one trailing argument that only IB builds have.
struct IBHandles {
    int64_t gin_dev_comm = 0;    // device pointer to ncclDevComm_t
    int64_t gin_window = 0;      // ncclWindow_t of the whole symmetric buffer

    IBHandles() = default;

    CUTLASS_HOST_DEVICE IBHandles(const int64_t gin_dev_comm, const int64_t gin_window):
        gin_dev_comm(gin_dev_comm), gin_window(gin_window) {}

    template <typename Container>
    static bool present(const Container& c) {
        const auto size = static_cast<uint32_t>(c.size());
        return size > kNumIBExtraPtrs and c[size - kNumIBExtraPtrs] == kIBMagic;
    }

    // Number of ranks in a pointer list with or without the IB tail
    template <typename Container>
    static uint32_t num_ranks_of(const Container& c) {
        const auto size = static_cast<uint32_t>(c.size());
        return present(c) ? size - kNumIBExtraPtrs : size;
    }

    // Ranks per NVLink domain of an IB pointer list (a compile-time constant of the kernel, through the plan)
    template <typename Container>
    static uint32_t ranks_per_node_of(const Container& c) {
        DG_UNIFIED_ASSERT(present(c) and c[c.size() - 1] > 0);
        return static_cast<uint32_t>(c[c.size() - 1]);
    }

    template <typename Container>
    explicit IBHandles(const Container& c) {
        DG_UNIFIED_ASSERT(present(c));
        const auto size = static_cast<uint32_t>(c.size());
        gin_dev_comm = c[size - 3];
        gin_window = c[size - 2];
    }
};

// IB signals: GPU-local words only (other nodes never write here; they put into the node slots).
#if defined(DG_MEGA_MOE_IB_COMPAT_ABI) and DG_MEGA_MOE_IB_COMPAT_ABI
using MegaMoEIBSignals = ib_compat::OldIBSignals;  // TEMPORARY
#else
struct alignas(128) MegaMoEIBSignals {
    static constexpr uint32_t kMaxNodes = 16;
    static constexpr uint32_t kMaxFwdChunks = 64;    // forward-combine chunks per remote node (32 tokens each)
    static constexpr uint32_t kMaxSources = 64;      // remote source ranks
    static constexpr uint32_t kNumTableWords = 1u << 20;
    static constexpr uint32_t kNumSorts = 3, kMaxSortParts = 8;

    // Launch counter of this GPU. Its parity selects which copy of every NIC-written region this launch reads,
    // so a late write of the previous launch can never be taken for a write of this one.
    alignas(128) uint64_t epoch;

    // Proxy side, per source node: tokens of the source this GPU proxies (written by the proxy SM)
    alignas(128) uint32_t src_num_tokens[kMaxNodes];
    // Forward-combine bookkeeping, by parity and remote node: items claimed, records done per chunk, chunks returned
    alignas(128) uint32_t fwd_item_claim[2][kMaxNodes];
    alignas(128) uint32_t fwd_chunk_done[2][kMaxNodes][kMaxFwdChunks];
    alignas(128) uint32_t fwd_chunks_returned[2][kMaxNodes];
    // Receiver: epoch once the slot table of remote source k is complete on this GPU (read by its pull warps)
    alignas(128) uint64_t tables_ready[kMaxSources];
    // Proxy: epoch once the return order of remote source node n is written (read by the forward-combine)
    alignas(128) uint64_t order_ready[kMaxNodes];
    // Sender, first-need order: epoch once chunk c of the rows for remote node slot nr is staged
    alignas(128) uint64_t rows_staged[kMaxNodes][kMaxFwdChunks];
    // Multi-SM counting sorts (token sort, table builder, sliced scan): part p published its histogram row / is done
    alignas(128) uint64_t sort_shared[kNumSorts][kMaxSortParts];
    alignas(128) uint64_t sort_done[kNumSorts][kMaxSortParts];
    // Per-GPU tables (return order, first-need order, sort rows), sub-allocated by impls/mega_moe_ib/geometry.cuh
    alignas(128) uint32_t tables[kNumTableWords];

    // Diagnostics (hang forensics and the trace build)
    alignas(128) uint64_t sm_stage[256];  // per CTA: (epoch << 8) | stage, dumped by the grid-sync timeout
    uint32_t stage_dumped;
    alignas(128) uint64_t trace[48];
    alignas(128) uint32_t hist[4][64];
};
#endif

// Geometry of the IB tail of the symmetric buffer. One instance per (max tokens, hidden, top-k, experts); the
// kernel builds it from its template arguments and the host from the call's arguments, so the two cannot drift.
//
// tail = [MegaMoEIBSignals] [node slot 0] [node slot 1] ...   (one slot per node, the same on every rank)
// node slot n, seen from a GPU of node m != n, is where node n's counterpart GPU writes to / is written from:
//   mirror[2]       by parity: the source's inputs as the stock kernel reads them (x, SF, top-k idx, weights),
//                   a header, the per-expert counts and the sender-built entry tables
//   ret[2]          by parity: the return records (one per source token) and their header
//   proxy_combine   partial sums of the tokens this GPU proxies, [top-k slot][token][hidden] bf16
//   fwd_stage       records staged for the return puts
// The slot of a GPU's own node is never a NIC target; the kernel uses it as scratch (layout.cuh names the uses).
struct IBBufferLayout {
    int64_t num_max_tokens, hidden, num_topk, num_experts;

    CUTLASS_HOST_DEVICE constexpr IBBufferLayout(const int64_t& num_max_tokens, const int64_t& hidden,
                                                 const int64_t& num_topk, const int64_t& num_experts):
        num_max_tokens(num_max_tokens), hidden(hidden), num_topk(num_topk), num_experts(num_experts) {}

    // Mirror: the upstream input layout, then header, counts, entry tables
    CUTLASS_HOST_DEVICE constexpr int64_t token_bytes() const { return hidden; }
    CUTLASS_HOST_DEVICE constexpr int64_t sf_bytes() const { return hidden / 32; }
    CUTLASS_HOST_DEVICE constexpr int64_t off_sf() const { return num_max_tokens * token_bytes(); }
    CUTLASS_HOST_DEVICE constexpr int64_t off_idx() const { return off_sf() + num_max_tokens * sf_bytes(); }
    CUTLASS_HOST_DEVICE constexpr int64_t off_weights() const { return off_idx() + num_max_tokens * num_topk * 8; }
    CUTLASS_HOST_DEVICE constexpr int64_t off_header() const {
        return math::constexpr_align<int64_t>(off_weights() + num_max_tokens * num_topk * 4, 128);
    }
    CUTLASS_HOST_DEVICE constexpr int64_t off_counts() const { return off_header() + 128; }
    CUTLASS_HOST_DEVICE constexpr int64_t counts_bytes() const { return math::constexpr_align<int64_t>(num_experts * 4, 128); }
    CUTLASS_HOST_DEVICE constexpr int64_t off_tables() const { return off_counts() + counts_bytes(); }
    CUTLASS_HOST_DEVICE constexpr int64_t tables_bytes() const {
        return math::constexpr_align<int64_t>(num_max_tokens * num_topk * 4, 128);
    }
    CUTLASS_HOST_DEVICE constexpr int64_t mirror_stride() const { return off_tables() + tables_bytes(); }

    // Return records: fp8 e4m3 values + one f32 scale per 128 columns. The landing keeps room for 2 bytes per
    // column (the staging of the proxy combine shares the record geometry).
    CUTLASS_HOST_DEVICE constexpr int64_t record_bytes() const { return hidden + hidden / 128 * 4; }
    CUTLASS_HOST_DEVICE constexpr int64_t record_alloc() const { return hidden * 2; }
    CUTLASS_HOST_DEVICE constexpr int64_t off_ret_header() const { return num_max_tokens * record_alloc(); }
    CUTLASS_HOST_DEVICE constexpr int64_t ret_stride() const { return off_ret_header() + 128; }
    CUTLASS_HOST_DEVICE constexpr int64_t proxy_combine_stride() const { return num_topk * num_max_tokens * hidden * 2; }
    CUTLASS_HOST_DEVICE constexpr int64_t fwd_stage_stride() const { return num_max_tokens * record_alloc(); }

    // Offsets inside a node slot
    CUTLASS_HOST_DEVICE constexpr int64_t off_mirror(const int64_t& parity) const { return parity * mirror_stride(); }
    CUTLASS_HOST_DEVICE constexpr int64_t off_ret(const int64_t& parity) const { return 2 * mirror_stride() + parity * ret_stride(); }
    CUTLASS_HOST_DEVICE constexpr int64_t off_proxy_combine() const { return 2 * mirror_stride() + 2 * ret_stride(); }
    CUTLASS_HOST_DEVICE constexpr int64_t off_fwd_stage() const { return off_proxy_combine() + proxy_combine_stride(); }
    CUTLASS_HOST_DEVICE constexpr int64_t node_stride() const {
#if defined(DG_MEGA_MOE_IB_COMPAT_ABI) and DG_MEGA_MOE_IB_COMPAT_ABI
        return off_fwd_stage() + fwd_stage_stride() + ib_compat::num_old_node_slot_extra_bytes(num_max_tokens, num_experts);  // TEMPORARY
#else
        return math::constexpr_align<int64_t>(off_fwd_stage() + fwd_stage_stride(), 128);
#endif
    }

    // Offsets inside the tail, and its size. `signals()` is where the IB signals of a GPU live, given its stock
    // signals and the start of its tail.
    CUTLASS_HOST_DEVICE constexpr int64_t off_node(const int64_t& node) const {
#if defined(DG_MEGA_MOE_IB_COMPAT_ABI) and DG_MEGA_MOE_IB_COMPAT_ABI
        return node * node_stride();  // TEMPORARY
#else
        return math::constexpr_align<int64_t>(sizeof(MegaMoEIBSignals), 128) + node * node_stride();
#endif
    }
    CUTLASS_HOST_DEVICE static MegaMoEIBSignals* signals(void* stock_signals, void* tail) {
#if defined(DG_MEGA_MOE_IB_COMPAT_ABI) and DG_MEGA_MOE_IB_COMPAT_ABI
        return reinterpret_cast<MegaMoEIBSignals*>(static_cast<uint8_t*>(stock_signals) + ib_compat::kNumStockSignalBytes);  // TEMPORARY
#else
        return static_cast<MegaMoEIBSignals*>(tail);
#endif
    }
    CUTLASS_HOST_DEVICE constexpr int64_t num_bytes(const int64_t& num_nodes) const { return off_node(num_nodes); }

    // Where the tail starts, given the bytes of the stock buffer
    CUTLASS_HOST_DEVICE static constexpr int64_t tail_offset(const int64_t& num_stock_bytes) {
        return math::constexpr_align<int64_t>(num_stock_bytes, 128);
    }
};

} // namespace deep_gemm::layout
