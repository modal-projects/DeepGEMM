#pragma once

// Mega MoE across NVLink domains: the row pull of the dispatch warps.
//
// The stock pull loop stays in the kernel (expert cursor, slot -> (source rank, index), ring wait, bookkeeping).
// Two things differ when sources can be on other nodes:
//   * a remote source's slot table is written by this GPU's receiver SMs, after its counts: wait for its flag;
//   * a remote source's row is read from its counterpart's mirror on this node, and the load is the poll.

#include <deep_gemm/impls/mega_moe_ib/geometry.cuh>
#include <deep_gemm/impls/mega_moe_ib/trace.cuh>

namespace deep_gemm::ib {

// The slot table entry of (local expert, source rank, index in rank). An entry is `token * top-k + k` in its low
// 16 bits; a remote source's entry carries the mirror row of the token in its high 16 bits.
// `tables_seen` has one bit per remote source this warp has already waited for.
template <typename G>
CUTLASS_DEVICE uint32_t slot_entry(const Launch& launch, const layout::Workspace& workspace,
                                   const uint32_t& expert_idx, const uint32_t& src_rank, const uint32_t& idx_in_rank,
                                   uint64_t& tables_seen) {
    DG_STATIC_ASSERT(static_cast<uint64_t>(G::kT) * G::kK < 65536, "Slot table entries pack token x top-k into 16 bits");
    if (G::node_of(src_rank) != launch.my_node) {
        const uint32_t k = G::source_of(src_rank, launch.my_node);
        if (not ((tables_seen >> k) & 1)) {
            comm::landing::await_raised(&launch.signals->tables_ready[k], launch.epoch, launch.site("tables_ready", src_rank));
            tables_seen |= 1ull << k;
        }
    }
    return *workspace.get_src_token_topk_idx_ptr(expert_idx, src_rank, idx_in_rank);
}

// Copy one token row, its scale factors and its top-k weight into the L1 buffers; returns the weight.
// The whole warp runs it; lane l owns the 16-byte units l, 32 + l, ... of the row. All loads of the row are
// issued first (scale factors, weight, row), so a row costs one NVLink round trip when it has landed.
// Rows of sources on other nodes come from the counterpart's mirror and are polled; node-local rows come from
// the source's input region as in the stock kernel and never wait.
template <typename G, uint32_t kNumRanks>
CUTLASS_DEVICE float pull_row(const Launch& launch, const layout::SymBuffer<kNumRanks>& sym_buffer,
                              const layout::MegaMoEBuffer& buffer,
                              const uint32_t& src_rank, const uint32_t& entry,
                              uint8_t* dst_row, uint32_t* dst_sf, const uint32_t& dst_sf_stride,
                              const uint32_t& lane_idx) {
    constexpr uint32_t kNumUnits = G::kH / (32 * 16);
    constexpr uint32_t kNumSFWords = G::kH / 128;
    constexpr uint32_t kNumSFLoads = math::constexpr_ceil_div(kNumSFWords, 32u);
    const uint32_t src_token_topk_idx = entry & 0xffff;
    const uint32_t src_token_idx = src_token_topk_idx / G::kK;
    const bool is_remote = G::node_of(src_rank) != launch.my_node;

    // Where the three pieces are: on the source itself, or in its mirror on its counterpart here
    const uint32_t holder = is_remote ? G::counterpart(src_rank, launch.my_node) : src_rank;
    const auto mirror = Slots<G>::mirror(launch, G::node_of(src_rank));
    const auto row = sym_buffer.map(is_remote ?
        mirror + static_cast<int64_t>(entry >> 16) * G::buf().token_bytes() :
        buffer.input_token_buffer.get_data_buffer(src_token_idx).template get_base_ptr<uint8_t>(), holder);
    const auto sf = sym_buffer.map(is_remote ?
        reinterpret_cast<uint32_t*>(mirror + G::buf().off_sf() + static_cast<int64_t>(src_token_idx) * G::buf().sf_bytes()) :
        buffer.input_sf_buffer.get_data_buffer(src_token_idx).template get_base_ptr<uint32_t>(), holder);
    const auto weight_ptr = sym_buffer.map(is_remote ?
        reinterpret_cast<uint32_t*>(mirror + G::buf().off_weights()) + src_token_topk_idx :
        buffer.input_topk_weights_buffer.template get_base_ptr<uint32_t>() + src_token_topk_idx, holder);

    // Issue everything
    uint32_t sf_words[kNumSFLoads];
    #pragma unroll
    for (uint32_t i = 0; i < kNumSFLoads; ++ i)
        sf_words[i] = i * 32 + lane_idx < kNumSFWords ? ptx::ld_volatile(sf + i * 32 + lane_idx) : 0u;
    uint32_t weight = ptx::ld_volatile(weight_ptr);
    uint4 units[kNumUnits];
    #pragma unroll
    for (uint32_t u = 0; u < kNumUnits; ++ u)
        units[u] = comm::landing::load(row + (u * 32 + lane_idx) * 16);

    // Weight (polled here in every build: no other reader is ahead of the pull for every source)
    if (is_remote)
        comm::landing::await<G::kDiag>(weight, weight_ptr, launch.site("pull weight", src_rank, src_token_topk_idx));

    // Row. All of its units that are still blank are re-read together: a row caught in flight then costs one
    // round trip after it lands, not one per unit (per-unit polling cost 15 us at 128 tokens on 32 GPUs)
    if (is_remote)
        comm::landing::await_all<G::kDiag, kNumUnits>(units, [&](const uint32_t& u) { return row + (u * 32 + lane_idx) * 16; },
                                                      launch.site("pull row", src_rank, src_token_idx));
    #pragma unroll
    for (uint32_t u = 0; u < kNumUnits; ++ u)
        *reinterpret_cast<uint4*>(dst_row + (u * 32 + lane_idx) * 16) = units[u];
    // Generic stores, read by the GEMM's TMA loads
    asm volatile("fence.proxy.async.global;" ::: "memory");

    // Scale factors
    #pragma unroll
    for (uint32_t i = 0; i < kNumSFLoads; ++ i) {
        const uint32_t j = i * 32 + lane_idx;
        if (j < kNumSFWords) {
            if (is_remote)
                comm::landing::await<G::kDiag>(sf_words[i], sf + j, launch.site("pull sf", src_rank, src_token_idx));
            dst_sf[j * dst_sf_stride] = sf_words[i];
        }
    }
    __syncwarp();
    trace::stamp<G>(launch, trace::kFirstRemoteRow, is_remote and lane_idx == 0);
    trace::stamp<G>(launch, trace::kLastRow, lane_idx == 0);
    return __uint_as_float(weight);
}

} // namespace deep_gemm::ib
