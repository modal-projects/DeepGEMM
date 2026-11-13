#pragma once

// Mega MoE across NVLink domains: what the epilogue warps do before their first GEMM task.
//
// Sender (this GPU as a source), from the input region and with nothing staged:
//   SM 0             per remote node two threads on the Meta stream: one puts the top-k index rows, the other the
//                    weight rows and then the header (epoch, token count), which therefore lands after the weights;
//                    then the source's return order per remote node (sort.cuh);
//   SMs 1 .. kR      one thread per remote node on the Rows stream: scale factors, then the token rows in puts of
//                    kNumRowsPerPut rows, in order.
// Receiver (this GPU as a host and as a proxy), one remote source per SM of the last SMs:
//   every receiver   learns the source's token count from the header in the proxy's mirror, counts the source's
//                    top-k entries for this GPU's experts, writes the slot table, raises `tables_ready`, and hands
//                    the counts to the stock scheduler;
//   the proxy's SM   (the source's counterpart is this GPU) also publishes the token count, polls every weight
//                    (with the indices: the landing proof of the Meta stream), and writes the return order.

#include <deep_gemm/comm/gin.cuh>
#include <deep_gemm/impls/mega_moe_ib/geometry.cuh>
#include <deep_gemm/impls/mega_moe_ib/route.cuh>
#include <deep_gemm/impls/mega_moe_ib/trace.cuh>
#include <deep_gemm/impls/mega_moe_ib/sort.cuh>

namespace deep_gemm::ib {

template <typename G, uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kBarrierIdx>
__device__ __noinline__ void start(const Launch launch, const layout::Workspace workspace,
                                   const layout::SymBuffer<kNumRanks>* sym_buffer,
                                   uint8_t* const input, const uint32_t num_tokens,
                                   uint32_t* const smem, const uint32_t num_smem_bytes,
                                   const uint32_t sm_idx, const uint32_t thread_idx) {
    using namespace comm;
    constexpr uint32_t kNumKeys = G::kNumExpertsPerRank + 1;
    using scratch_t = SortScratch<kNumKeys, kNumThreads / 32>;
    constexpr uint32_t kNumCountWords = math::constexpr_align(G::kNumExpertsPerRank + 1, 16u);
    const uint32_t warp_idx = thread_idx / 32, lane_idx = thread_idx % 32;
    const auto sync = [&]() { ptx::sync_aligned(kNumThreads, kBarrierIdx); };
    const auto counts = smem;                                       // [experts per rank], then the token count
    auto& scratch = *reinterpret_cast<scratch_t*>(smem + kNumCountWords);
    const auto keys = launch.signals->tables + G::kOffSortKeys + sm_idx * G::kTableT;
    if (thread_idx == 0 and (kNumCountWords * 4 + sizeof(scratch_t) > num_smem_bytes)) {
        printf("DeepGEMM IB: the start does not fit this layout (smem %u bytes)\n", num_smem_bytes);
        DG_DEVICE_ASSERT(false);
    }

    // ---- Sender ----
    if (sm_idx == G::kMetaSM) {
        if (warp_idx < 2 and lane_idx < G::kR) {
            const uint32_t node = G::node_of_slot(lane_idx, launch.my_node), peer = G::counterpart(launch.rank, node);
            const auto dst = Slots<G>::outbox_mirror(launch);
            const Gin gin = make_gin<G>(launch, Stream::Meta, lane_idx);
            const uint64_t num_entries = static_cast<uint64_t>(num_tokens) * G::kK;
            if (warp_idx == 0) {
                if (num_tokens > 0)
                    gin.put(dst + G::buf().off_idx(), input + G::buf().off_idx(), num_entries * 8, peer);
                if (num_tokens > 0 and G::kPlan.scale_factors == layout::IBScaleFactors::WithMeta)
                    gin.put(dst + G::buf().off_sf(), input + G::buf().off_sf(), static_cast<uint64_t>(num_tokens) * G::buf().sf_bytes(), peer);
            } else {
                if (num_tokens > 0)
                    gin.put(dst + G::buf().off_weights(), input + G::buf().off_weights(), num_entries * 4, peer);
                gin.put_value(reinterpret_cast<uint64_t*>(dst + G::buf().off_header()), (launch.epoch << 32) | num_tokens, peer);
            }
        }
        __syncwarp();
        trace::stamp<G>(launch, trace::kMetaPosted, thread_idx == 0);
        // The return order of this source's tokens on each remote node (read by the source combine)
        for (uint32_t nr = 0; nr < G::kR; ++ nr) {
            build_return_keys<G, kNumThreads, kBarrierIdx>(
                keys, reinterpret_cast<const uint64_t*>(input + G::buf().off_idx()), num_tokens,
                G::node_of_slot(nr, launch.my_node), thread_idx);
            const auto return_pos = launch.signals->tables + G::kOffReturnPos + nr * G::kTableT;
            sort_by_key<kNumKeys, kNumThreads, kBarrierIdx>(keys, num_tokens, scratch, thread_idx,
                [&](const uint32_t& token, const uint32_t& position) { return_pos[token] = position; });
        }
        __threadfence();
        sync();
    } else if (sm_idx < G::kFirstRowSM + G::kR * G::kNumRowIssuers) {
        // Issuer j of a node takes the puts j, j + I, ...: each issuer's puts land in order
        if (thread_idx == 0 and num_tokens > 0) {
            const uint32_t nr = (sm_idx - G::kFirstRowSM) / G::kNumRowIssuers, issuer = (sm_idx - G::kFirstRowSM) % G::kNumRowIssuers;
            const uint32_t node = G::node_of_slot(nr, launch.my_node), peer = G::counterpart(launch.rank, node);
            const auto dst = Slots<G>::outbox_mirror(launch);
            const Gin gin = make_gin<G>(launch, Stream::Rows, nr, issuer);
            const auto put_scale_factors = [&](const layout::IBScaleFactors& when) {
                if (issuer == 0 and G::kPlan.scale_factors == when)
                    gin.put(dst + G::buf().off_sf(), input + G::buf().off_sf(), static_cast<uint64_t>(num_tokens) * G::buf().sf_bytes(), peer);
            };
            put_scale_factors(layout::IBScaleFactors::BeforeRows);
            for (uint32_t t = issuer * G::kNumRowsPerPut; t < num_tokens; t += G::kNumRowsPerPut * G::kNumRowIssuers) {
                const uint64_t offset = static_cast<uint64_t>(t) * G::buf().token_bytes();
                // First-need order: the rows of this put come from the stage, once their chunks are staged
                const uint8_t* rows = input;
                if constexpr (G::kPlan.first_need_order) {
                    rows = Slots<G>::row_stage(launch, nr);
                    for (uint32_t c = t / G::kNumRowsPerStage; c * G::kNumRowsPerStage < cute::min(num_tokens, t + G::kNumRowsPerPut); ++ c)
                        landing::await_raised(&launch.signals->rows_staged[nr][c], launch.epoch, launch.site("rows staged", nr, c));
                }
                gin.put(dst + offset, rows + offset,
                        static_cast<uint64_t>(cute::min(num_tokens - t, static_cast<uint32_t>(G::kNumRowsPerPut))) * G::buf().token_bytes(), peer);
            }
            put_scale_factors(layout::IBScaleFactors::AfterRows);
            trace::stamp<G>(launch, trace::kRowsPosted);
        }
        __syncwarp();
    }

    // First-need order: sort the tokens per remote node by when that node first needs them (one SM per node, on
    // the keys the table build takes in its count sweep); stage the rows in that order, 32 at a time on SMs that
    // are idle now (each raises a flag per chunk; the poster above waits for them)
    constexpr uint32_t kNeedSort = 0;
    const auto need_keys = launch.signals->tables + G::kOffSortKeys + G::kFirstOrderSM * G::kTableT;  // the order SMs' key rows
    if constexpr (G::kPlan.first_need_order) {
        const auto need_pos = launch.signals->tables + G::kOffNeedPos, need_inv = launch.signals->tables + G::kOffNeedInv;
        if (sm_idx >= G::kFirstOrderSM and sm_idx < G::kFirstBuildSM) {
            const uint32_t nr = sm_idx - G::kFirstOrderSM;
            if (thread_idx < kNumBuildParts)
                landing::await_raised(&launch.signals->sort_shared[kBuildSort][thread_idx], launch.epoch, launch.site("need keys", thread_idx));
            sync();
            sort_by_key<kNumKeys, kNumThreads, kBarrierIdx>(keys, num_tokens, scratch, thread_idx,
                [&](const uint32_t& token, const uint32_t& position) {
                    need_pos[nr * G::kTableT + token] = position;
                    need_inv[nr * G::kTableT + position] = token;
                });
            __threadfence();
            sync();
            if (thread_idx == 0)
                landing::raise(&launch.signals->sort_done[kNeedSort][nr], launch.epoch);
            trace::stamp<G>(launch, trace::kNeedOrderDone, thread_idx == 0);
        }
        if (sm_idx >= G::kFirstPullSM and sm_idx < G::kFirstOrderSM) {
            constexpr uint32_t kNumRowUnits = G::kH / 16, kNumLoads = 10;
            const uint32_t num_chunks = math::ceil_div(num_tokens, G::kNumRowsPerStage);
            const auto tokens = smem;  // [32] the tokens of this chunk's rows
            for (uint32_t q = sm_idx - G::kFirstPullSM; q < G::kR * num_chunks; q += G::kNumStageSMs) {
                const uint32_t nr = q / num_chunks, chunk = q % num_chunks;
                const uint32_t first_row = chunk * G::kNumRowsPerStage;
                const uint32_t num_rows = cute::min(num_tokens - first_row, static_cast<uint32_t>(G::kNumRowsPerStage));
                if (thread_idx == 0)
                    landing::await_raised(&launch.signals->sort_done[kNeedSort][nr], launch.epoch, launch.site("need order", nr));
                sync();
                if (thread_idx < num_rows)
                    tokens[thread_idx] = __ldcg(need_inv + nr * G::kTableT + first_row + thread_idx);
                sync();
                const auto stage = Slots<G>::row_stage(launch, nr) + static_cast<int64_t>(first_row) * G::buf().token_bytes();
                for (uint32_t u0 = thread_idx; u0 < num_rows * kNumRowUnits; u0 += kNumThreads * kNumLoads) {
                    uint4 units[kNumLoads];
                    #pragma unroll
                    for (uint32_t b = 0; b < kNumLoads; ++ b) {
                        const uint32_t u = u0 + b * kNumThreads;
                        if (u < num_rows * kNumRowUnits)
                            units[b] = __ldcg(reinterpret_cast<const uint4*>(input + static_cast<int64_t>(tokens[u / kNumRowUnits]) * G::buf().token_bytes()) + u % kNumRowUnits);
                    }
                    #pragma unroll
                    for (uint32_t b = 0; b < kNumLoads; ++ b) {
                        const uint32_t u = u0 + b * kNumThreads;
                        if (u < num_rows * kNumRowUnits)
                            reinterpret_cast<uint4*>(stage)[u] = units[b];
                    }
                }
                __threadfence();
                sync();
                if (thread_idx == 0)
                    landing::raise(&launch.signals->rows_staged[nr][chunk], launch.epoch);
            }
        }
    }

    // Sender tables: counts and slot entries for every remote node, built here and put as one message per node
    if constexpr (G::kSenderTables) {
        if (sm_idx >= G::kFirstBuildSM and sm_idx < G::kFirstReceiverSM) {
            const auto need_pos = launch.signals->tables + G::kOffNeedPos;
            build_tables<G, kNumThreads, kBarrierIdx>(launch, input, num_tokens, sm_idx - G::kFirstBuildSM, smem, num_smem_bytes, thread_idx, need_keys,
                [&](const uint32_t& nr, const uint32_t& token) {
                    return G::kPlan.first_need_order ? __ldcg(need_pos + nr * G::kTableT + token) : token;
                },
                [&]() {
                    // The entries carry the rows: the order of every remote node must be there before the scatter
                    if constexpr (G::kPlan.first_need_order) {
                        if (thread_idx < G::kR)
                            landing::await_raised(&launch.signals->sort_done[kNeedSort][thread_idx], launch.epoch, launch.site("need order", thread_idx));
                        sync();
                    }
                });
        }
    }

    // ---- Receiver ----
    if (sm_idx >= G::kFirstReceiverSM) {
        const uint32_t k = G::source_of_receiver(sm_idx, launch.rank);
        const uint32_t src_rank = G::rank_of_source(k, launch.my_node), src_node = G::node_of(src_rank);
        const uint32_t proxy = G::counterpart(src_rank, launch.my_node);
        const bool is_proxy = proxy == launch.rank;
        const auto mirror = sym_buffer->map(Slots<G>::mirror(launch, src_node), proxy);
        const auto topk_idx = reinterpret_cast<const uint64_t*>(mirror + G::buf().off_idx());
        const auto weights = reinterpret_cast<const uint32_t*>(mirror + G::buf().off_weights());

        // Sender tables: every receiver copies its slice of the source's message; only the proxy goes on to the
        // header and the top-k rows (token count, landing proof, return order)
        if constexpr (G::kSenderTables) {
            receive_tables<G, kNumThreads, kBarrierIdx>(launch, workspace, mirror, k, src_rank, counts, thread_idx,
                [&]() { publish_counts<kNumSMs>(workspace, src_rank, counts, thread_idx, kNumThreads); });
            trace::stamp<G>(launch, trace::kCountsPublished, thread_idx == 0);
            if (not is_proxy)
                return;
        }

        // The header: one thread polls it, slowly (many threads spinning on memory a NIC writes slow the NIC down)
        if (thread_idx == 0) {
            const auto header_ptr = reinterpret_cast<const uint64_t*>(mirror + G::buf().off_header());
            uint64_t header = 0;
            wait_until([&]() {
                header = ptx::ld_volatile(header_ptr);
                if (not launch.network and header != ~0ull)
                    return true;
                if ((header >> 32) != (launch.epoch & 0xffffffffull))
                    __nanosleep(200);
                return (header >> 32) == (launch.epoch & 0xffffffffull);
            }, [&]() {
                printf("DeepGEMM IB header timeout: rank=%u, source=%u, proxy=%u, header=%llx, epoch=%llu\n",
                       launch.rank, src_rank, proxy, static_cast<unsigned long long>(header), static_cast<unsigned long long>(launch.epoch));
            });
            counts[G::kNumExpertsPerRank] = static_cast<uint32_t>(header);
            trace::stamp<G>(launch, trace::kFirstHeaderSeen);
            trace::stamp<G>(launch, trace::kHeadersSeen);
        }
        for (uint32_t i = thread_idx; i < G::kNumExpertsPerRank; i += kNumThreads)
            counts[i] = 0;
        sync();
        const uint32_t num_src_tokens = counts[G::kNumExpertsPerRank];
        const uint32_t num_entries = num_src_tokens * G::kK;
        if (is_proxy) {
            for (uint32_t t = thread_idx; t < num_src_tokens; t += kNumThreads)
                keys[t] = 0;
            if (thread_idx == 0)
                launch.signals->src_num_tokens[src_node] = num_src_tokens;
            sync();
        }

        scan_route<G, kNumThreads>(launch, workspace, topk_idx, weights, src_rank, num_entries, is_proxy, counts, keys, thread_idx);
        if constexpr (not G::kSenderTables) {
            // Slot table before counts: a pull that sees the counts never waits for the table
            __threadfence();
            sync();
            if (thread_idx == 0)
                landing::raise(&launch.signals->tables_ready[k], launch.epoch);
            publish_counts<kNumSMs>(workspace, src_rank, counts, thread_idx, kNumThreads);
            sync();
            trace::stamp<G>(launch, trace::kCountsPublished, thread_idx == 0);
        } else {
            sync();
        }

        if (is_proxy) {
            const auto return_perm = launch.signals->tables + G::kOffReturnPerm + G::slot_of_node(src_node, launch.my_node) * G::kTableT;
            sort_by_key<kNumKeys, kNumThreads, kBarrierIdx>(keys, num_src_tokens, scratch, thread_idx,
                [&](const uint32_t& token, const uint32_t& position) { return_perm[position] = token; });
            __threadfence();
            sync();
            if (thread_idx == 0)
                landing::raise(&launch.signals->order_ready[src_node], launch.epoch);
            trace::stamp<G>(launch, trace::kIntakeDone, thread_idx == 0);
        }
    }
}

} // namespace deep_gemm::ib
