#pragma once

// Mega MoE across NVLink domains: the token sort.
//
// Both ends of a (source, node) pair need the source's tokens in the order in which that node will finish them:
// the proxy returns records in that order, the source looks its records up by it. Every GPU runs its experts in
// index order, so the key of a token is the largest in-rank index among its experts on the node (0 without one).
// The sort is a stable counting sort (ties in token order), so the proxy, which sorts the top-k rows in its
// mirror, and the source, which sorts its inputs, get the same order without exchanging it.

#include <deep_gemm/impls/mega_moe_ib/geometry.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::ib {

// Shared memory the sort borrows from an SM's epilogue warps before their first GEMM task
template <uint32_t kNumKeys, uint32_t kNumWarps>
struct SortScratch {
    uint32_t warp_counts[kNumWarps][kNumKeys];   // per warp: histogram, then exclusive prefix over the lower warps
    uint32_t key_starts[kNumKeys];
};

// keys[t] = max over token t's top-k entries on `node` of (expert % experts per rank), 0 without one.
// The whole CTA role (`kNumThreads` threads on barrier `kBarrierIdx`) runs it; `topk_idx` must have landed.
template <typename G, uint32_t kNumThreads, uint32_t kBarrierIdx>
CUTLASS_DEVICE void build_return_keys(uint32_t* keys, const uint64_t* topk_idx, const uint32_t& num_tokens,
                                      const uint32_t& node, const uint32_t& thread_idx) {
    constexpr uint32_t kNumLoads = 8;
    for (uint32_t t = thread_idx; t < num_tokens; t += kNumThreads)
        keys[t] = 0;
    ptx::sync_aligned(kNumThreads, kBarrierIdx);
    const uint32_t num_entries = num_tokens * G::kK;
    for (uint32_t i0 = thread_idx; i0 < num_entries; i0 += kNumThreads * kNumLoads) {
        uint64_t entries[kNumLoads];
        #pragma unroll
        for (uint32_t u = 0; u < kNumLoads; ++ u)
            entries[u] = i0 + u * kNumThreads < num_entries ? ptx::ld_volatile(topk_idx + i0 + u * kNumThreads) : ~0ull;
        #pragma unroll
        for (uint32_t u = 0; u < kNumLoads; ++ u) {
            const int expert = static_cast<int>(static_cast<int64_t>(entries[u]));
            if (expert >= 0 and G::node_of(static_cast<uint32_t>(expert) / G::kNumExpertsPerRank) == node)
                atomicMax(keys + (i0 + u * kNumThreads) / G::kK, static_cast<uint32_t>(expert) % G::kNumExpertsPerRank);
        }
    }
    ptx::sync_aligned(kNumThreads, kBarrierIdx);
}

// Stable counting sort of tokens [0, num_tokens) by keys[t] < kNumKeys; calls emit(token, position).
// Warp w takes the w-th contiguous span of tokens, so "lower warp, then lower lane" is token order.
template <uint32_t kNumKeys, uint32_t kNumThreads, uint32_t kBarrierIdx, typename emit_t>
CUTLASS_DEVICE void sort_by_key(const uint32_t* keys, const uint32_t& num_tokens,
                                SortScratch<kNumKeys, kNumThreads / 32>& scratch,
                                const uint32_t& thread_idx, const emit_t& emit) {
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    const auto volatile_keys = reinterpret_cast<const volatile uint32_t*>(keys);
    const uint32_t warp_idx = thread_idx / 32, lane_idx = thread_idx % 32;
    const uint32_t span = math::ceil_div(num_tokens, kNumThreads) * 32u;
    const uint32_t begin = cute::min(num_tokens, warp_idx * span), end = cute::min(num_tokens, begin + span);

    for (uint32_t i = thread_idx; i < kNumWarps * kNumKeys; i += kNumThreads)
        scratch.warp_counts[0][i] = 0;
    ptx::sync_aligned(kNumThreads, kBarrierIdx);

    // Histogram per warp
    for (uint32_t t = begin + lane_idx; t < end; t += 32)
        atomicAdd_block(scratch.warp_counts[warp_idx] + volatile_keys[t], 1u);
    ptx::sync_aligned(kNumThreads, kBarrierIdx);

    // Per key: exclusive prefix over the warps, and the key's total
    for (uint32_t key = thread_idx; key < kNumKeys; key += kNumThreads) {
        uint32_t run = 0;
        #pragma unroll
        for (uint32_t w = 0; w < kNumWarps; ++ w) {
            const uint32_t count = scratch.warp_counts[w][key];
            scratch.warp_counts[w][key] = run;
            run += count;
        }
        scratch.key_starts[key] = run;
    }
    ptx::sync_aligned(kNumThreads, kBarrierIdx);

    // Exclusive prefix over the keys (one warp)
    if (warp_idx == 0) {
        uint32_t carry = 0;
        for (uint32_t k0 = 0; k0 < kNumKeys; k0 += 32) {
            const uint32_t key = k0 + lane_idx;
            const uint32_t count = key < kNumKeys ? scratch.key_starts[key] : 0u;
            const uint32_t inclusive = math::warp_inclusive_sum(count, lane_idx);
            if (key < kNumKeys)
                scratch.key_starts[key] = carry + inclusive - count;
            carry += __shfl_sync(0xffffffffu, inclusive, 31);
        }
    }
    ptx::sync_aligned(kNumThreads, kBarrierIdx);

    // Place: start of the key + this key's tokens in lower warps + in earlier loads of this warp + in lower lanes
    for (uint32_t t0 = begin; t0 < end; t0 += 32) {
        const uint32_t token = t0 + lane_idx;
        const bool is_valid = token < end;
        const uint32_t key = is_valid ? volatile_keys[token] : 0xffffffffu;
        const uint32_t peers = __match_any_sync(0xffffffffu, key);
        const uint32_t rank = __popc(peers & ((1u << lane_idx) - 1u));
        const uint32_t base = is_valid ? scratch.key_starts[key] + scratch.warp_counts[warp_idx][key] : 0u;
        __syncwarp();
        if (is_valid) {
            emit(token, base + rank);
            if (rank == 0)
                scratch.warp_counts[warp_idx][key] += __popc(peers);
        }
        __syncwarp();
    }
    ptx::sync_aligned(kNumThreads, kBarrierIdx);
}

} // namespace deep_gemm::ib
