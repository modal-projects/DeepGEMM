#pragma once

// TEMPORARY (delete with the commit that added it): build mode DG_MEGA_MOE_IB_COMPAT_ABI=1.
//
// The development servers hold ranks warm behind a host extension built from an older tree. To time this
// kernel against that tree's kernel in ONE request, the kernel can be built against the old host's ABI:
//   1. the kernel argument is the old SymBuffer, which carries the Gin handles in three trailing fields;
//   2. the old signals struct had the IB fields inside it, so the stock data that follows starts later, and
//      the IB signals are the old fields at their old offsets;
//   3. the IB tail starts with node slot 0 (no signals in front) and every node slot ends with a table
//      region the kernel no longer uses.
// Nothing else differs: the same config, the same code, offsets as the only changed constants. The numbers
// below were read from the old tree (jamesl/fused-ib-ep32-trunk2 @ 181a88a, layout/mega_moe.cuh).

#if defined(DG_MEGA_MOE_IB_COMPAT_ABI) and DG_MEGA_MOE_IB_COMPAT_ABI

#include <cstddef>
#include <cstdint>

#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm::layout::ib_compat {

constexpr static uint64_t kNumStockSignalBytes = 21140864;  // sizeof(MegaMoESignals) on main
constexpr static uint64_t kNumOldSignalBytes = 26742784;    // sizeof(MegaMoESignals) in the old tree

template <uint32_t kNumRanks>
struct OldSymBuffer: SymBuffer<kNumRanks> {
    int64_t gin_dev_comm;
    int64_t gin_window;
    uint32_t num_lsa_ranks;
};
DG_STATIC_ASSERT(sizeof(OldSymBuffer<kNumMaxRanks>) == 616, "Old SymBuffer layout");

// The live IB signals under the names of layout/mega_moe_ib.cuh, at the old struct's offsets (relative to the
// first old IB field, which sits where the stock signals end)
struct OldIBSignals {
    static constexpr uint32_t kMaxNodes = 16;
    static constexpr uint32_t kMaxFwdChunks = 64;
    static constexpr uint32_t kMaxSources = 64;
    static constexpr uint32_t kNumTableWords = 1u << 20;
    static constexpr uint32_t kNumSorts = 3, kMaxSortParts = 8;

    uint64_t epoch;
    uint8_t pad0[3328 - 8];
    uint64_t trace[48];
    uint32_t hist[4][64];
    uint64_t sm_stage[256];
    uint32_t stage_dumped;
    // Flags the old kernel does not have live in old fields that no old kernel writes (epoch-tagged: stale or
    // foreign contents can never read as raised)
    uint8_t pad1a[8192 - 6784 - 4];
    uint64_t sort_shared[kNumSorts][kMaxSortParts];
    uint64_t sort_done[kNumSorts][kMaxSortParts];
    uint8_t pad1b[865152 - 8192 - 2 * 8 * kNumSorts * kMaxSortParts];
    uint32_t tables[kNumTableWords];
    uint8_t pad2[5583744 - 865152 - 4 * (1u << 20)];
    uint32_t src_num_tokens[kMaxNodes];
    uint8_t pad3[5583872 - 5583744 - 4 * 16];
    uint32_t fwd_chunk_done[2][kMaxNodes][kMaxFwdChunks];
    uint32_t fwd_chunks_returned[2][kMaxNodes];
    uint8_t pad4[5592576 - 5592192];                 // old meta-posted words (written by old kernels)
    uint32_t fwd_item_claim[2][kMaxNodes];
    uint64_t tables_ready[kMaxSources];
    uint64_t order_ready[kMaxSources];               // indexed by node here
    uint64_t rows_staged[kMaxNodes][kMaxFwdChunks];  // the old landed flags (dead in every frozen old build)
};
DG_STATIC_ASSERT(offsetof(OldIBSignals, trace) == 3328 and offsetof(OldIBSignals, hist) == 3712 and
                 offsetof(OldIBSignals, sm_stage) == 4736 and offsetof(OldIBSignals, stage_dumped) == 6784 and
                 offsetof(OldIBSignals, sort_shared) == 8192 and
                 offsetof(OldIBSignals, tables) == 865152 and offsetof(OldIBSignals, src_num_tokens) == 5583744 and
                 offsetof(OldIBSignals, fwd_chunk_done) == 5583872 and offsetof(OldIBSignals, fwd_chunks_returned) == 5592064 and
                 offsetof(OldIBSignals, fwd_item_claim) == 5592576 and
                 offsetof(OldIBSignals, tables_ready) == 5592704 and offsetof(OldIBSignals, order_ready) == 5593216 and
                 offsetof(OldIBSignals, rows_staged) == 5593728 and
                 sizeof(OldIBSignals) == kNumOldSignalBytes - kNumStockSignalBytes, "Old IB signal offsets");

// Bytes at the end of every old node slot that the kernel no longer uses ([rank][local expert][token] u32)
CUTLASS_HOST_DEVICE constexpr int64_t num_old_node_slot_extra_bytes(const int64_t& num_max_tokens, const int64_t& num_experts) {
    return num_experts * num_max_tokens * 4;
}

} // namespace deep_gemm::layout::ib_compat

// The old host cannot print a plan: it forwards DG_MEGA_MOE_* environment variables as integer defines. A compat
// build takes the plan's fields from DG_MEGA_MOE_IBP_<FIELD> (defaults: the scan regime on nodes of 8 ranks).
#ifndef DG_MEGA_MOE_IB_PLAN
#ifndef DG_MEGA_MOE_IBP_RANKS_PER_NODE
#define DG_MEGA_MOE_IBP_RANKS_PER_NODE 8
#endif
#ifndef DG_MEGA_MOE_IBP_ROUTING
#define DG_MEGA_MOE_IBP_ROUTING 0
#endif
#ifndef DG_MEGA_MOE_IBP_FIRST_NEED_ORDER
#define DG_MEGA_MOE_IBP_FIRST_NEED_ORDER 0
#endif
#ifndef DG_MEGA_MOE_IBP_ROW_ISSUERS
#define DG_MEGA_MOE_IBP_ROW_ISSUERS 1
#endif
#ifndef DG_MEGA_MOE_IBP_ROWS_PER_PUT
#define DG_MEGA_MOE_IBP_ROWS_PER_PUT 128
#endif
#ifndef DG_MEGA_MOE_IBP_CONTEXTS_PER_NODE
#define DG_MEGA_MOE_IBP_CONTEXTS_PER_NODE 0
#endif
#ifndef DG_MEGA_MOE_IBP_SCALE_FACTORS
#define DG_MEGA_MOE_IBP_SCALE_FACTORS 0
#endif
#ifndef DG_MEGA_MOE_IBP_L1_LEAD_CAP
#define DG_MEGA_MOE_IBP_L1_LEAD_CAP 0
#endif
#ifndef DG_MEGA_MOE_IBP_RETIRE_PERCENT
#define DG_MEGA_MOE_IBP_RETIRE_PERCENT 0
#endif
#ifndef DG_MEGA_MOE_IBP_RETIRE_AT_PERCENT
#define DG_MEGA_MOE_IBP_RETIRE_AT_PERCENT 100
#endif
#ifndef DG_MEGA_MOE_IBP_SHARED_L2_EARLY
#define DG_MEGA_MOE_IBP_SHARED_L2_EARLY 0
#endif
#ifndef DG_MEGA_MOE_IBP_DIAGNOSTICS
#define DG_MEGA_MOE_IBP_DIAGNOSTICS 0
#endif
// The old servers know two measurement switches by their old names (they isolate a zero-network request and
// collect trace lines by them): DG_MEGA_MOE_IB_TRACE=1 and DG_MEGA_MOE_IB_ORACLE=2
#ifndef DG_MEGA_MOE_IBP_TRACE
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
#define DG_MEGA_MOE_IBP_TRACE 1
#else
#define DG_MEGA_MOE_IBP_TRACE 0
#endif
#endif
#ifndef DG_MEGA_MOE_IBP_ZERO_NETWORK
#if defined(DG_MEGA_MOE_IB_ORACLE) and DG_MEGA_MOE_IB_ORACLE >= 2
#define DG_MEGA_MOE_IBP_ZERO_NETWORK 1
#else
#define DG_MEGA_MOE_IBP_ZERO_NETWORK 0
#endif
#endif
#ifndef DG_MEGA_MOE_IBP_GIN_BACKEND
#define DG_MEGA_MOE_IBP_GIN_BACKEND 0
#endif
#define DG_MEGA_MOE_IB_PLAN { \
    .ranks_per_node = DG_MEGA_MOE_IBP_RANKS_PER_NODE, .routing = deep_gemm::layout::IBRouting(DG_MEGA_MOE_IBP_ROUTING), \
    .first_need_order = DG_MEGA_MOE_IBP_FIRST_NEED_ORDER != 0, \
    .row_issuers = DG_MEGA_MOE_IBP_ROW_ISSUERS, \
    .rows_per_put = DG_MEGA_MOE_IBP_ROWS_PER_PUT, .contexts_per_node = DG_MEGA_MOE_IBP_CONTEXTS_PER_NODE != 0, \
    .scale_factors = deep_gemm::layout::IBScaleFactors(DG_MEGA_MOE_IBP_SCALE_FACTORS), \
    .gin_backend = DG_MEGA_MOE_IBP_GIN_BACKEND, .l1_lead_cap = DG_MEGA_MOE_IBP_L1_LEAD_CAP, \
    .retire_percent = DG_MEGA_MOE_IBP_RETIRE_PERCENT, .retire_at_percent = DG_MEGA_MOE_IBP_RETIRE_AT_PERCENT, \
    .shared_l2_early = DG_MEGA_MOE_IBP_SHARED_L2_EARLY != 0, .diagnostics = DG_MEGA_MOE_IBP_DIAGNOSTICS, \
    .trace = DG_MEGA_MOE_IBP_TRACE != 0, .zero_network = DG_MEGA_MOE_IBP_ZERO_NETWORK != 0}
#endif

#endif
