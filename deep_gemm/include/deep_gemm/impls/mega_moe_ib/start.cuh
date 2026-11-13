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
#include <deep_gemm/impls/mega_moe_ib/trace.cuh>
#include <deep_gemm/impls/mega_moe_ib/sort.cuh>

namespace deep_gemm::ib {

// Hand a remote source's per-expert counts to the stock scheduler, which waits until every source has added
// its share of `kNumSMs` to each expert's sum (a node-local source adds it from its own SM 0)
template <uint32_t kNumSMs>
CUTLASS_DEVICE void publish_counts(const layout::Workspace& workspace, const uint32_t& src_rank, const uint32_t* counts,
                                   const uint32_t& thread_idx, const uint32_t& num_threads) {
    for (uint32_t i = thread_idx; i < workspace.num_experts_per_rank; i += num_threads) {
        *workspace.get_expert_recv_count_ptr(src_rank, i) = counts[i];
        __threadfence();
        ptx::atomic_add(workspace.get_expert_recv_count_sum_ptr(i), (static_cast<uint64_t>(kNumSMs) << 32) | counts[i]);
    }
}

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
            if (issuer == 0)
                gin.put(dst + G::buf().off_sf(), input + G::buf().off_sf(), static_cast<uint64_t>(num_tokens) * G::buf().sf_bytes(), peer);
            for (uint32_t t = issuer * G::kNumRowsPerPut; t < num_tokens; t += G::kNumRowsPerPut * G::kNumRowIssuers) {
                const uint64_t offset = static_cast<uint64_t>(t) * G::buf().token_bytes();
                gin.put(dst + offset, input + offset,
                        static_cast<uint64_t>(cute::min(num_tokens - t, static_cast<uint32_t>(G::kNumRowsPerPut))) * G::buf().token_bytes(), peer);
            }
            trace::stamp<G>(launch, trace::kRowsPosted);
        }
        __syncwarp();
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

        // One pass over the source's top-k rows, a batch of loads in flight per thread, re-reading every entry of
        // the batch that has not landed yet. An entry for one of this GPU's experts takes the next slot of that
        // expert; the proxy also takes the return-order keys and proves the weights landed.
        constexpr uint32_t kNumLoads = 16;
        for (uint32_t i0 = thread_idx; i0 < num_entries; i0 += kNumThreads * kNumLoads) {
            uint64_t entries[kNumLoads];
            uint32_t entry_weights[kNumLoads];
            const auto reload = [&](const bool& all) {
                #pragma unroll
                for (uint32_t u = 0; u < kNumLoads; ++ u) {
                    const uint32_t i = i0 + u * kNumThreads;
                    if (i < num_entries and (all or entries[u] == landing::kBlankIdx))
                        entries[u] = ptx::ld_volatile(topk_idx + i);
                    if (is_proxy and i < num_entries and (all or entry_weights[u] == landing::kBlankWord))
                        entry_weights[u] = ptx::ld_volatile(weights + i);
                }
            };
            #pragma unroll
            for (uint32_t u = 0; u < kNumLoads; ++ u)
                entries[u] = 0, entry_weights[u] = 0;
            reload(true);
            landing::spin<true>([&]() {
                bool blank = false;
                #pragma unroll
                for (uint32_t u = 0; u < kNumLoads; ++ u)
                    blank |= entries[u] == landing::kBlankIdx or entry_weights[u] == landing::kBlankWord;
                return blank;
            }, [&]() { reload(false); }, launch.site("start top-k", src_rank, i0));
            #pragma unroll
            for (uint32_t u = 0; u < kNumLoads; ++ u) {
                const uint32_t i = i0 + u * kNumThreads;
                const int expert = static_cast<int>(static_cast<int64_t>(entries[u]));
                if (i >= num_entries or expert < 0)
                    continue;
                const uint32_t expert_rank = static_cast<uint32_t>(expert) / G::kNumExpertsPerRank;
                const uint32_t local_expert = static_cast<uint32_t>(expert) % G::kNumExpertsPerRank;
                if (is_proxy and G::node_of(expert_rank) == launch.my_node)
                    atomicMax(keys + i / G::kK, local_expert);
                if (expert_rank == launch.rank) {
                    const uint32_t slot = atomicAdd_block(counts + local_expert, 1u);
                    *workspace.get_src_token_topk_idx_ptr(local_expert, src_rank, slot) = i | ((i / G::kK) << 16);
                }
            }
        }
        // Slot table before counts: a pull that sees the counts never waits for the table
        __threadfence();
        sync();
        if (thread_idx == 0)
            landing::raise(&launch.signals->tables_ready[k], launch.epoch);
        publish_counts<kNumSMs>(workspace, src_rank, counts, thread_idx, kNumThreads);
        sync();
        trace::stamp<G>(launch, trace::kCountsPublished, thread_idx == 0);

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
