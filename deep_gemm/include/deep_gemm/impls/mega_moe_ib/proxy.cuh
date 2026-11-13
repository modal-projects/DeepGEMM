#pragma once

// Mega MoE across NVLink domains: the proxy side of the return path.
//
// Hosts on this node write the L2 output rows of a remote source's tokens to this GPU, the source's counterpart
// (`store_partial`, same [top-k slot][token][hidden] bf16 layout as the stock combine buffer). The proxy sums a
// token's partials of this node in fp32, quantizes the sum to fp8 with one scale per 128 columns, and returns one
// record per token, 32 records per put (`reduce`). Summing here turns ~top-k / nodes bf16 rows per token into
// one fp8 record on the wire.

#include <cuda_fp8.h>

#include <deep_gemm/comm/gin.cuh>
#include <deep_gemm/impls/mega_moe_ib/geometry.cuh>
#include <deep_gemm/impls/mega_moe_ib/trace.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::ib {

namespace detail {
CUTLASS_DEVICE uint32_t add_acq_rel(uint32_t* ptr, const uint32_t& value) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(value) : "memory");
    return old;
}
} // namespace detail

// L2 epilogue: 16 bytes of an output row whose source is on another node go to the proxy-combine buffer of the
// source's node on its counterpart here. `offset` is the byte offset the stock kernel uses inside the source's
// combine buffer. No flag follows: the proxy polls the row itself.
template <typename G, uint32_t kNumRanks>
CUTLASS_DEVICE void store_partial(const Launch& launch, const layout::SymBuffer<kNumRanks>& sym_buffer,
                                  const uint32_t& src_rank, const int64_t& offset, const float4& packed) {
    const auto dst = reinterpret_cast<float4*>(Slots<G>::proxy_combine(launch, G::node_of(src_rank)) + offset);
    *sym_buffer.map(dst, G::counterpart(src_rank, launch.my_node)) = packed;
}

// Zero the work-queue counters. Run by one SM before the dispatch grid sync, which every claimant of this launch
// passes; the previous launch has ended, so nobody reads the other parity's rows either, and clearing both makes
// a launch independent of whatever ran before it.
template <typename G>
CUTLASS_DEVICE void reset_counters(const Launch& launch, const uint32_t& thread_idx, const uint32_t& num_threads) {
    constexpr uint32_t kNumNodes = layout::MegaMoEIBSignals::kMaxNodes;
    for (uint32_t i = thread_idx; i < 2 * kNumNodes * layout::MegaMoEIBSignals::kMaxFwdChunks; i += num_threads)
        launch.signals->fwd_chunk_done[0][0][i] = 0;
    if (thread_idx < 2 * kNumNodes) {
        launch.signals->fwd_item_claim[0][thread_idx] = 0;
        launch.signals->fwd_chunks_returned[0][thread_idx] = 0;
    }
}

// The proxy sum, run by every warp that has nothing else to do: dispatch warps after their pulls, epilogue warps
// after their last GEMM task. Work item = (return position, quarter of the hidden dimension), claimed from one
// counter per source node; a warp handles one item at a time, every lane 16 bytes of each row.
//   1. poll the item's part of the token's row (and its scale factors in quarter 0): the landing proof of the
//      forward path, for every token whether or not it has an expert on this node;
//   2. for each of the token's experts on this node, in top-k order: poll that partial row, re-arm it, add;
//   3. quantize, store the record into the staging buffer;
//   4. count the item; whoever completes a 32-record chunk puts it.
// `first_slot` rotates the node order per warp so that all return streams advance together.
template <typename G, typename sched_t>
CUTLASS_DEVICE void reduce(const Launch& launch, const uint32_t& first_slot, const bool& is_leader, const uint32_t& lane_idx) {
    constexpr uint32_t kNumQuarters = 4;
    constexpr uint32_t kNumRowUnits = G::kH * 2 / (32 * 16);          // 16-byte bf16 units per lane over a partial row
    constexpr uint32_t kNumItemUnits = kNumRowUnits / kNumQuarters;
    constexpr uint32_t kNumItemXUnits = G::kH / 16 / kNumQuarters;    // 16-byte fp8 units of the token row per item
    constexpr uint32_t kNumXLoads = math::constexpr_ceil_div(kNumItemXUnits, 32u);
    constexpr int64_t kNumRecordBytes = G::buf().record_bytes();
    DG_STATIC_ASSERT(kNumItemUnits * kNumQuarters * 32 * 16 == G::kH * 2 and kNumItemXUnits * 16 * kNumQuarters == G::kH, "Invalid hidden size");
    DG_STATIC_ASSERT(G::buf().sf_bytes() <= 32 * 16, "Scale factors of a token must fit one load per lane");
    const auto signals = launch.signals;
    trace::stamp<G>(launch, trace::kFirstReduce, lane_idx == 0);

    #pragma unroll
    for (uint32_t i = 0; i < G::kR; ++ i) {
        const uint32_t nr = first_slot + i < G::kR ? first_slot + i : first_slot + i - G::kR;
        const uint32_t node = G::node_of_slot(nr, launch.my_node);
        const uint32_t source = G::counterpart(launch.rank, node);
        const auto ret_dst = Slots<G>::outbox_ret(launch);

        // The intake of this source is done: its token count and return order are there, its top-k rows landed
        comm::landing::await_raised(&signals->order_ready[node], launch.epoch, launch.site("order_ready", node));
        const uint32_t num_tokens = signals->src_num_tokens[node];
        if (num_tokens == 0) {
            // Nothing to return; the source still waits for a header (every byte written is read)
            if (is_leader)
                make_gin<G>(launch, comm::Stream::Return, nr)
                    .put_value(reinterpret_cast<uint64_t*>(ret_dst + G::buf().off_ret_header()), launch.epoch, source);
            continue;
        }
        const uint32_t num_items = num_tokens * kNumQuarters;
        const auto mirror = Slots<G>::mirror(launch, node);
        const auto partials = Slots<G>::proxy_combine(launch, node);
        const auto stage = Slots<G>::fwd_stage(launch, node);
        const auto topk_idx = reinterpret_cast<const uint64_t*>(mirror + G::buf().off_idx());
        const auto return_perm = signals->tables + G::kOffReturnPerm + nr * G::kTableT;
        const auto claim_ptr = &signals->fwd_item_claim[launch.parity][node];

        for (uint32_t item = sched_t::get_next_local_task_idx(claim_ptr, 0); item < num_items;
             item = sched_t::get_next_local_task_idx(claim_ptr, 0)) {
            const uint32_t position = item / kNumQuarters, quarter = item % kNumQuarters;
            const uint32_t token = ptx::ld_volatile(return_perm + position);

            // The token's experts on this node, one per lane
            const int expert = lane_idx < G::kK ? static_cast<int>(static_cast<int64_t>(
                ptx::ld_volatile(topk_idx + token * G::kK + lane_idx))) : -1;
            const bool is_here = expert >= 0 and G::node_of(static_cast<uint32_t>(expert) / G::kNumExpertsPerRank) == launch.my_node;
            uint32_t slot_mask = __ballot_sync(0xffffffffu, is_here);

            // 1. Landing proof of the token row and its scale factors
            {
                const auto row = mirror + static_cast<int64_t>(token) * G::buf().token_bytes() + quarter * kNumItemXUnits * 16;
                uint4 units[kNumXLoads];
                #pragma unroll
                for (uint32_t u = 0; u < kNumXLoads; ++ u)
                    units[u] = u * 32 + lane_idx < kNumItemXUnits ? comm::landing::load(row + (u * 32 + lane_idx) * 16) : uint4{0, 0, 0, 0};
                #pragma unroll
                for (uint32_t u = 0; u < kNumXLoads; ++ u)
                    comm::landing::await<G::kDiag>(units[u], row + (u * 32 + lane_idx) * 16, launch.site("proxy row", source, token));
                if (quarter == 0 and lane_idx < G::buf().sf_bytes() / 16) {
                    const auto sf = mirror + G::buf().off_sf() + static_cast<int64_t>(token) * G::buf().sf_bytes() + lane_idx * 16;
                    auto unit = comm::landing::load(sf);
                    comm::landing::await<G::kDiag>(unit, sf, launch.site("proxy sf", source, token));
                }
            }

            // 2. Sum the partials in top-k order (fp32)
            float2 sums[kNumItemUnits][4];
            #pragma unroll
            for (uint32_t u = 0; u < kNumItemUnits; ++ u)
                #pragma unroll
                for (uint32_t l = 0; l < 4; ++ l)
                    sums[u][l] = {0.f, 0.f};
            while (slot_mask) {
                const uint32_t slot = static_cast<uint32_t>(__ffs(slot_mask) - 1);
                slot_mask &= slot_mask - 1;
                const auto row = partials + (static_cast<int64_t>(slot) * G::kT + token) * G::kH * 2
                                 + (quarter * kNumItemUnits * 32 + lane_idx) * 16;
                uint4 units[kNumItemUnits];
                #pragma unroll
                for (uint32_t u = 0; u < kNumItemUnits; ++ u)
                    units[u] = comm::landing::load(row + u * 32 * 16);
                #pragma unroll
                for (uint32_t u = 0; u < kNumItemUnits; ++ u) {
                    const auto ptr = row + u * 32 * 16;
                    comm::landing::await<G::kDiag>(units[u], ptr, launch.site("proxy partial", token, slot));
                    *reinterpret_cast<uint4*>(ptr) = comm::landing::blank_unit();
                    const auto values = reinterpret_cast<const nv_bfloat162*>(&units[u]);
                    #pragma unroll
                    for (uint32_t l = 0; l < 4; ++ l)
                        ptx::accumulate(sums[u][l], values[l]);
                }
            }

            // 3. Quantize: 8 columns per lane, one scale per 128 columns (a half-warp)
            const auto record = stage + static_cast<int64_t>(position) * kNumRecordBytes;
            #pragma unroll
            for (uint32_t u = 0; u < kNumItemUnits; ++ u) {
                float amax = 0.f;
                #pragma unroll
                for (uint32_t l = 0; l < 4; ++ l)
                    amax = fmaxf(amax, fmaxf(fabsf(sums[u][l].x), fabsf(sums[u][l].y)));
                #pragma unroll
                for (uint32_t d = 1; d < 16; d <<= 1)
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
                const float scale = amax > 0.f ? amax / 448.f : 1.f;
                const float inv_scale = 1.f / scale;
                uint2 quantized;
                const auto pairs = reinterpret_cast<__nv_fp8x2_storage_t*>(&quantized);
                #pragma unroll
                for (uint32_t l = 0; l < 4; ++ l)
                    pairs[l] = __nv_cvt_float2_to_fp8x2(make_float2(sums[u][l].x * inv_scale, sums[u][l].y * inv_scale), __NV_SATFINITE, __NV_E4M3);
                const uint32_t col = ((quarter * kNumItemUnits + u) * 32 + lane_idx) * 8;
                *reinterpret_cast<uint2*>(record + col) = quantized;
                if (lane_idx % 16 == 0)
                    *reinterpret_cast<float*>(record + G::kH + col / 128 * 4) = scale;
            }
            __syncwarp();

            // 4. Count the item; the warp that completes a chunk puts it
            if (lane_idx == 0) {
                const uint32_t chunk = position / G::kNumRecordsPerPut;
                const uint32_t num_records = cute::min(num_tokens - chunk * G::kNumRecordsPerPut, static_cast<uint32_t>(G::kNumRecordsPerPut));
                __threadfence();
                if (detail::add_acq_rel(&signals->fwd_chunk_done[launch.parity][node][chunk], 1u) + 1 == num_records * kNumQuarters) {
                    detail::add_acq_rel(&signals->fwd_chunks_returned[launch.parity][node], 1u);
                    const int64_t offset = static_cast<int64_t>(chunk) * G::kNumRecordsPerPut * kNumRecordBytes;
                    make_gin<G>(launch, comm::Stream::Return, nr, chunk)
                        .put(ret_dst + offset, stage + offset, static_cast<uint64_t>(num_records) * kNumRecordBytes, source);
                    trace::stamp<G>(launch, trace::kLastChunkPut);
                }
            }
            __syncwarp();
        }
    }
}

// Re-arm the mirrors of this launch's parity, after the node's last barrier (every pull of this node from them
// is done). Per remote node it first waits until every return chunk is put: then the proxy sum has polled every
// token row, and the intake before it every top-k row, so all the NIC wrote this launch has landed.
template <typename G, typename sync_scope_t>
CUTLASS_DEVICE void rearm_mirrors(const Launch& launch, const uint32_t& gtid, const uint32_t& num_grid_threads,
                                  const bool& is_waiter, const sync_scope_t& sync_scope) {
    using namespace comm::landing;
    #pragma unroll
    for (uint32_t nr = 0; nr < G::kR; ++ nr) {
        const uint32_t node = G::node_of_slot(nr, launch.my_node);
        const uint32_t num_tokens = ptx::ld_volatile(&launch.signals->src_num_tokens[node]);
        const uint32_t num_chunks = math::ceil_div(num_tokens, G::kNumRecordsPerPut);
        if (is_waiter) {
            const auto returned_ptr = &launch.signals->fwd_chunks_returned[launch.parity][node];
            comm::wait_until([&]() { return ptx::ld_acq(returned_ptr) >= num_chunks; }, [&]() {
                printf("DeepGEMM IB re-arm timeout: rank=%u node=%u returned=%u of %u\n", launch.rank, node, ptx::ld_acq(returned_ptr), num_chunks);
            });
        }
        sync_scope();
        if constexpr (not G::kRearm)
            continue;
        const auto mirror = Slots<G>::mirror(launch, node);
        const uint64_t num_entries = static_cast<uint64_t>(num_tokens) * G::kK;
        rearm<G::kDiag>(mirror, static_cast<uint64_t>(num_tokens) * G::buf().token_bytes(), blank_unit(), gtid, num_grid_threads, launch.site("rearm rows", node));
        rearm<G::kDiag>(mirror + G::buf().off_sf(), static_cast<uint64_t>(num_tokens) * G::buf().sf_bytes(), blank_unit(), gtid, num_grid_threads, launch.site("rearm sf", node));
        rearm<G::kDiag>(mirror + G::buf().off_idx(), num_entries * 8, uint4{0u, 0x80000000u, 0u, 0x80000000u}, gtid, num_grid_threads, launch.site("rearm idx", node));
        rearm<G::kDiag>(mirror + G::buf().off_weights(), num_entries * 4, blank_unit(), gtid, num_grid_threads, launch.site("rearm weights", node));
        if (gtid == 0)
            *reinterpret_cast<uint64_t*>(mirror + G::buf().off_header()) = ~0ull;
        if constexpr (G::kSenderTables) {
            rearm<false>(mirror + G::buf().off_tables(), num_entries * 4, blank_unit(), gtid, num_grid_threads, launch.site("rearm tables", node));
            rearm<false>(mirror + G::buf().off_counts(), G::buf().counts_bytes(), blank_unit(), gtid, num_grid_threads, launch.site("rearm counts", node));
        }
    }
}

} // namespace deep_gemm::ib
