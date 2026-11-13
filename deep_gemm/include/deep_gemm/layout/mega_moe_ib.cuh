#pragma once

// Mega MoE across NVLink domains ("IB"): everything the host, Python and the kernel must agree on.
//
// Ranks of one NVLink domain (a node) keep talking through load/store exactly as in the stock kernel. Ranks of
// other nodes are reached with NCCL Gin puts from inside the kernel. Nothing here changes the stock layout:
//   * the handles Gin needs travel behind the per-rank pointers in `sym_buffer_ptrs` (`IBHandles`), and reach the
//     kernel as one extra argument that only IB builds have;
//   * all IB state lives in a tail appended to the symmetric buffer, after the bytes the stock kernel uses
//     (`IBBufferLayout`): first the IB signals, then one slot per node.
// A build without DG_MEGA_MOE_IB sees none of it.

#include <cstddef>
#include <cstdint>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm::layout {

// `sym_buffer_ptrs` from Python is [per-rank load/store pointers (0 for ranks of other nodes)], followed, for an IB
// group, by [kIBMagic, device pointer to ncclDevComm_t, ncclWindow_t, ranks per NVLink domain].
constexpr static int64_t kIBMagic = 0x1B1B1B1B1B1B1B1Bll;
constexpr static uint32_t kNumIBExtraPtrs = 4;

struct IBHandles {
    int64_t gin_dev_comm = 0;    // device pointer to ncclDevComm_t
    int64_t gin_window = 0;      // ncclWindow_t of the whole symmetric buffer
    uint32_t num_lsa_ranks = 0;  // ranks per NVLink domain
    uint32_t rank_idx = 0;

    IBHandles() = default;

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

    template <typename Container>
    explicit IBHandles(const Container& c, const uint32_t& rank_idx): rank_idx(rank_idx) {
        DG_UNIFIED_ASSERT(present(c));
        const auto size = static_cast<uint32_t>(c.size());
        gin_dev_comm = c[size - 3];
        gin_window = c[size - 2];
        num_lsa_ranks = static_cast<uint32_t>(c[size - 1]);
        DG_UNIFIED_ASSERT(num_lsa_ranks > 0);
    }

    // Node (NVLink domain) of a rank, and whether a rank is reachable from this one by load/store
    CUTLASS_HOST_DEVICE uint32_t node_of(const uint32_t& r) const { return r / num_lsa_ranks; }
    CUTLASS_HOST_DEVICE uint32_t my_node() const { return rank_idx / num_lsa_ranks; }
    CUTLASS_HOST_DEVICE bool is_lsa(const uint32_t& r) const { return node_of(r) == my_node(); }
};

// IB signals: GPU-local words only (other nodes never write here; they put into the node slots).
struct alignas(128) MegaMoEIBSignals {
    static constexpr uint32_t kMaxNodes = 16;
    static constexpr uint32_t kMaxFwdChunks = 64;    // forward-combine chunks per remote node (32 tokens each)
    static constexpr uint32_t kMaxSources = 64;      // remote source ranks
    static constexpr uint32_t kNumTableWords = 1u << 20;

    // Launch counter of this GPU. Its parity selects which copy of every NIC-written region this launch reads,
    // so a late write of the previous launch can never be taken for a write of this one.
    alignas(128) uint64_t epoch;

    // Proxy side, per source node: tokens of the source this GPU proxies (written by the proxy SM)
    alignas(128) uint32_t src_num_tokens[kMaxNodes];
    // Forward-combine bookkeeping, by parity and remote node: items claimed, records done per chunk, chunks returned
    alignas(128) uint32_t fwd_item_claim[2][kMaxNodes];
    alignas(128) uint32_t fwd_chunk_done[2][kMaxNodes][kMaxFwdChunks];
    alignas(128) uint32_t fwd_chunks_returned[2][kMaxNodes];
    // Sender: epoch once this GPU posted its meta (other SMs may then post rows)
    alignas(128) uint64_t meta_posted;
    // Receiver: epoch once the slot tables / the order tables of remote source k are usable on this GPU
    alignas(128) uint64_t tables_ready[kMaxSources];
    alignas(128) uint64_t order_ready[kMaxSources];
    // Per-GPU tables and claim counters, sub-allocated by impls/mega_moe_ib/layout.cuh
    alignas(128) uint32_t tables[kNumTableWords];

    // Diagnostics (hang forensics and the trace build)
    alignas(128) uint64_t sm_stage[256];  // per CTA: (epoch << 8) | stage, dumped by the grid-sync timeout
    uint32_t stage_dumped;
    alignas(128) uint64_t trace[48];
    alignas(128) uint32_t hist[4][64];
};

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
        return math::constexpr_align<int64_t>(off_fwd_stage() + fwd_stage_stride(), 128);
    }

    // Offsets inside the tail, and its size
    CUTLASS_HOST_DEVICE constexpr int64_t off_signals() const { return 0; }
    CUTLASS_HOST_DEVICE constexpr int64_t off_node(const int64_t& node) const {
        return math::constexpr_align<int64_t>(sizeof(MegaMoEIBSignals), 128) + node * node_stride();
    }
    CUTLASS_HOST_DEVICE constexpr int64_t num_bytes(const int64_t& num_nodes) const { return off_node(num_nodes); }

    // Where the tail starts, given the bytes of the stock buffer
    CUTLASS_HOST_DEVICE static constexpr int64_t tail_offset(const int64_t& num_stock_bytes) {
        return math::constexpr_align<int64_t>(num_stock_bytes, 128);
    }
};

} // namespace deep_gemm::layout
