#pragma once

// Mega MoE across NVLink domains: the source combine.
//
// A token's output is the sum of its shared-expert row, the rows of its experts on this node (the stock combine
// buffer, the stock loop) and one record per remote node (the fp32 sum of that node's rows, quantized to fp8 by
// the proxy there). The stock loop does the local part and parks its fp32 sums; a second loop adds the records.
// Two passes on purpose: the records arrive last, and a warp that waited for one would still owe the local work
// of its later tokens (one pass measured +5 .. +8 us at 1024 tokens with the network in the loop).

#include <cuda_fp8.h>

#include <deep_gemm/impls/mega_moe_ib/geometry.cuh>
#include <deep_gemm/ptx/tma.cuh>

namespace deep_gemm::ib {

// What the stock combine loop sees in lane `lane_idx` for a token: the expert of that top-k slot, the shared
// marker, or -1; an expert on another node becomes -1 (its result is inside that node's record)
template <typename G>
CUTLASS_DEVICE int local_slot(const Launch& launch, const int& stored_topk_slot_idx, const uint32_t& lane_idx) {
    const bool is_remote = lane_idx < G::kK and stored_topk_slot_idx >= 0 and
        G::node_of(static_cast<uint32_t>(stored_topk_slot_idx) / G::kNumExpertsPerRank) != launch.my_node;
    return is_remote ? -1 : stored_topk_slot_idx;
}

// Park / reload a lane's fp32 sums of one token chunk (8 columns per 16-byte unit; lane-private, so no ordering)
template <uint32_t kNumUnits>
CUTLASS_DEVICE void park(const float2 (&reduced)[kNumUnits * 4], float* scratch, const uint32_t& lane_idx) {
    #pragma unroll
    for (uint32_t j = 0; j < kNumUnits; ++ j) {
        const auto dst = reinterpret_cast<float4*>(scratch + (j * 32 + lane_idx) * 8);
        dst[0] = make_float4(reduced[j * 4 + 0].x, reduced[j * 4 + 0].y, reduced[j * 4 + 1].x, reduced[j * 4 + 1].y);
        dst[1] = make_float4(reduced[j * 4 + 2].x, reduced[j * 4 + 2].y, reduced[j * 4 + 3].x, reduced[j * 4 + 3].y);
    }
}

// The second loop: for every token chunk of this warp, reload the parked sums, add each remote node's record,
// cast to bf16 and store to `y`. Every record of every remote node is polled and re-armed, also when the token
// has no expert there: the proxy writes all of them, and every byte written is read.
template <typename G, uint32_t kNumChunks, uint32_t kNumUnits>
CUTLASS_DEVICE void add_records(const Launch& launch, const int64_t* topk_idx, void* y, const uint32_t& num_tokens,
                                uint4* store_buffer, const uint32_t& first_token_chunk, const uint32_t& token_chunk_stride,
                                const uint32_t& lane_idx) {
    constexpr uint32_t kNumHiddenBytes = G::kH * sizeof(nv_bfloat16);
    constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
    DG_STATIC_ASSERT(kNumUnits * 32 * 16 == kNumChunkBytes, "Invalid combine chunk");
    const auto scratch = Slots<G>::combine_scratch(launch);

    for (uint32_t token_chunk_idx = first_token_chunk; token_chunk_idx < num_tokens * kNumChunks; token_chunk_idx += token_chunk_stride) {
        const uint32_t token_idx = token_chunk_idx / kNumChunks, chunk_idx = token_chunk_idx % kNumChunks;
        const uint32_t first_col = chunk_idx * (G::kH / kNumChunks);

        // Which remote nodes hold experts of this token
        const int expert = lane_idx < G::kK ? static_cast<int>(topk_idx[token_idx * G::kK + lane_idx]) : -1;
        const uint32_t node_bit = expert >= 0 ? 1u << G::node_of(static_cast<uint32_t>(expert) / G::kNumExpertsPerRank) : 0u;
        const uint32_t nodes_hit = __reduce_or_sync(0xffffffffu, node_bit);

        float2 reduced[kNumUnits * 4];
        #pragma unroll
        for (uint32_t j = 0; j < kNumUnits; ++ j) {
            const auto src = reinterpret_cast<const float4*>(scratch + static_cast<int64_t>(token_idx) * G::kH + first_col + (j * 32 + lane_idx) * 8);
            const float4 a = src[0], b = src[1];
            reduced[j * 4 + 0] = {a.x, a.y}, reduced[j * 4 + 1] = {a.z, a.w};
            reduced[j * 4 + 2] = {b.x, b.y}, reduced[j * 4 + 3] = {b.z, b.w};
        }

        #pragma unroll
        for (uint32_t nr = 0; nr < G::kR; ++ nr) {
            const uint32_t node = G::node_of_slot(nr, launch.my_node);
            const bool is_hit = (nodes_hit >> node) & 1u;
            const uint32_t position = launch.signals->tables[G::kOffReturnPos + nr * G::kTableT + token_idx];
            const auto record = Slots<G>::ret(launch, node) + static_cast<int64_t>(position) * G::buf().record_bytes();
            const auto value_ptr = [&](const uint32_t& j) { return reinterpret_cast<uint64_t*>(record + first_col + (j * 32 + lane_idx) * 8); };
            const auto scale_ptr = [&](const uint32_t& j) { return reinterpret_cast<uint32_t*>(record + G::kH + (first_col + (j * 32 + lane_idx) * 8) / 128 * 4); };

            // All loads of the record's part first; then, as a warp, re-read whatever has not landed
            uint64_t values[kNumUnits];
            uint32_t scales[kNumUnits];
            #pragma unroll
            for (uint32_t j = 0; j < kNumUnits; ++ j)
                values[j] = ptx::ld_volatile(value_ptr(j)), scales[j] = ptx::ld_volatile(scale_ptr(j));
            comm::landing::spin<true>([&]() {
                bool blank = false;
                #pragma unroll
                for (uint32_t j = 0; j < kNumUnits; ++ j)
                    blank |= values[j] == ~0ull or scales[j] == comm::landing::kBlankWord;
                return __any_sync(0xffffffffu, blank);
            }, [&]() {
                #pragma unroll
                for (uint32_t j = 0; j < kNumUnits; ++ j) {
                    if (values[j] == ~0ull) values[j] = ptx::ld_volatile(value_ptr(j));
                    if (scales[j] == comm::landing::kBlankWord) scales[j] = ptx::ld_volatile(scale_ptr(j));
                }
            }, launch.site("combine record", node, token_idx));

            #pragma unroll
            for (uint32_t j = 0; j < kNumUnits; ++ j) {
                if (is_hit) {
                    const float scale = __uint_as_float(scales[j]);
                    const auto pairs = reinterpret_cast<const __nv_fp8x2_storage_t*>(&values[j]);
                    #pragma unroll
                    for (uint32_t l = 0; l < 4; ++ l) {
                        const float2 f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(pairs[l], __NV_E4M3)));
                        reduced[j * 4 + l].x += f.x * scale;
                        reduced[j * 4 + l].y += f.y * scale;
                    }
                }
                if constexpr (G::kRearm)
                    *value_ptr(j) = ~0ull;
            }
            // A scale is read by 16 lanes and re-armed by one of them, after all have read it
            __syncwarp();
            if constexpr (G::kRearm) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumUnits; ++ j)
                    if ((first_col + (j * 32 + lane_idx) * 8) % 128 == 0)
                        *scale_ptr(j) = comm::landing::kBlankWord;
            }
        }

        // Cast and store, as the stock loop does
        #pragma unroll
        for (uint32_t j = 0; j < kNumUnits; ++ j) {
            uint4 casted;
            const auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
            #pragma unroll
            for (uint32_t l = 0; l < 4; ++ l)
                casted_bf16[l] = __float22bfloat162_rn(reduced[j * 4 + l]);
            if (j == 0) {
                ptx::tma_store_wait<0>();
                __syncwarp();
            }
            ptx::st_shared(store_buffer + j * 32 + lane_idx, casted.x, casted.y, casted.z, casted.w);
        }
        __syncwarp();
        if (cute::elect_one_sync()) {
            cute::tma_store_fence();
            ptx::tma_store_1d(math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_idx * kNumChunkBytes),
                              store_buffer, kNumChunkBytes);
            cute::tma_store_arrive();
        }
        __syncwarp();
    }
}

// A source without tokens gets no record, only a header per remote node: wait for it and re-arm it
template <typename G>
CUTLASS_DEVICE void await_empty_returns(const Launch& launch, const uint32_t& thread_idx) {
    if (thread_idx >= G::kR)
        return;
    const uint32_t node = G::node_of_slot(thread_idx, launch.my_node);
    const auto header_ptr = reinterpret_cast<uint64_t*>(Slots<G>::ret(launch, node) + G::buf().off_ret_header());
    comm::wait_until([&]() {
        const uint64_t header = ptx::ld_volatile(header_ptr);
        return header == launch.epoch or (not launch.network and header != ~0ull);
    }, [&]() { printf("DeepGEMM IB return header timeout: rank=%u, node=%u\n", launch.rank, node); });
    if constexpr (G::kRearm)
        *header_ptr = ~0ull;
}

} // namespace deep_gemm::ib
