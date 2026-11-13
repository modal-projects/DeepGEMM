#pragma once

// Mega MoE across NVLink domains: route tables built by the sender (`IBRouting::SenderTables`).
//
// A receiver that scans a source's top-k rows cannot start before they land. Here the source itself sorts its
// top-k entries by expert before anything is on the wire and puts, per remote node, one message
//     [ counts u32[experts of the node] | entries u32[sum of the counts], expert-major ]
// whose counts end exactly where the mirror's table region starts. Every GPU of that node then only copies its
// slice. An entry is `token * top-k + k` in the low 16 bits and the token's mirror row in the high 16 bits.
//
// The build is a counting sort over kNumBuildParts SMs (contiguous slices of the entries). Each part counts its
// slice, publishes its histogram row, reads all rows (totals, and the share of the parts below it), and scatters
// its slice into the common staging in the outbox. Construction rules (they are what the old tree's hang with few
// experts per rank asks for): every expert has exactly one owning thread whatever the expert count, starts come
// from one prefix over all experts that every thread can read, and a part with nothing to do still publishes
// and raises its flags.

#include <deep_gemm/comm/gin.cuh>
#include <deep_gemm/impls/mega_moe_ib/geometry.cuh>
#include <deep_gemm/impls/mega_moe_ib/trace.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::ib {

constexpr uint32_t kNumBuildParts = layout::MegaMoEIBSignals::kMaxSortParts;
constexpr uint32_t kBuildSort = 1;  // row of `sort_shared` / `sort_done` the table build uses

// One part of the sender's table build. `row_of_token(nr, token)` is the mirror row an entry carries.
template <typename G, uint32_t kNumThreads, uint32_t kBarrierIdx, typename row_of_token_t>
CUTLASS_DEVICE void build_tables(const Launch& launch, uint8_t* const input, const uint32_t& num_tokens,
                                 const uint32_t& part, uint32_t* const smem, const uint32_t& num_smem_bytes,
                                 const uint32_t& thread_idx, const row_of_token_t& row_of_token) {
    using namespace comm;
    constexpr uint32_t kE = G::kNumExpertsPerRank * G::kNumNodes * G::kLsa;      // all experts
    constexpr uint32_t kNodeExperts = G::kNumExpertsPerRank * G::kLsa;
    constexpr uint32_t kNumWords = (kE + 1) / 2;                                // two 16-bit counters per word
    constexpr uint32_t kSpan = 2 * math::constexpr_ceil_div(kE, 2 * kNumThreads);  // experts per thread, whole words
    constexpr uint32_t kNumLoads = 8;
    DG_STATIC_ASSERT(static_cast<uint64_t>(G::kT) * G::kK < 65536, "Counters and cursors are 16 bits");
    DG_STATIC_ASSERT(kSpan * kNumThreads >= kE and kSpan % 2 == 0, "Every expert needs one owning thread");
    DG_STATIC_ASSERT(G::kOffSortRows + (kNumBuildParts * kNumWords + kNumBuildParts * (kE + 1)) <= layout::MegaMoEIBSignals::kNumTableWords, "Build rows do not fit");
    DG_STATIC_ASSERT(static_cast<int64_t>(G::kR) * kNodeExperts * 4 <= G::buf().counts_bytes(), "Staged counts do not fit the outbox");
    const uint32_t warp_idx = thread_idx / 32, lane_idx = thread_idx % 32;
    const auto sync = [&]() { ptx::sync_aligned(kNumThreads, kBarrierIdx); };
    const auto counters = smem;                           // packed 16-bit counts, then cursors
    const auto warp_sums = smem + kNumWords;              // [warps]
    const auto rows = launch.signals->tables + G::kOffSortRows;                       // [part][kNumWords], global
    const auto starts = rows + kNumBuildParts * kNumWords + part * (kE + 1);          // this part's prefix over all experts
    const auto topk_idx = reinterpret_cast<const uint64_t*>(input + G::buf().off_idx());
    const auto staging = reinterpret_cast<uint32_t*>(Slots<G>::outbox_mirror(launch) + G::buf().off_counts());
    const uint32_t num_entries = num_tokens * G::kK;
    const uint32_t begin = static_cast<uint32_t>(static_cast<uint64_t>(num_entries) * part / kNumBuildParts);
    const uint32_t end = static_cast<uint32_t>(static_cast<uint64_t>(num_entries) * (part + 1) / kNumBuildParts);
    if (thread_idx == 0 and (kNumWords + kNumThreads / 32) * 4 > num_smem_bytes) {
        printf("DeepGEMM IB: the table build does not fit this layout (smem %u bytes)\n", num_smem_bytes);
        DG_DEVICE_ASSERT(false);
    }
    const auto is_remote_expert = [&](const int& expert) {
        return expert >= 0 and G::node_of(static_cast<uint32_t>(expert) / G::kNumExpertsPerRank) != launch.my_node;
    };

    // 1. Count this part's slice (experts of this node are never counted, so their counts stay zero)
    for (uint32_t i = thread_idx; i < kNumWords; i += kNumThreads)
        counters[i] = 0;
    sync();
    for (uint32_t i0 = begin + thread_idx; i0 < end; i0 += kNumThreads * kNumLoads) {
        uint64_t entries[kNumLoads];
        #pragma unroll
        for (uint32_t u = 0; u < kNumLoads; ++ u)
            entries[u] = i0 + u * kNumThreads < end ? __ldcg(reinterpret_cast<const unsigned long long*>(topk_idx) + i0 + u * kNumThreads) : ~0ull;
        #pragma unroll
        for (uint32_t u = 0; u < kNumLoads; ++ u) {
            const int expert = static_cast<int>(static_cast<int64_t>(entries[u]));
            if (is_remote_expert(expert))
                atomicAdd_block(counters + expert / 2, 1u << (16 * (expert & 1)));
        }
    }
    sync();

    // 2. Publish the histogram row; wait for every part's row
    for (uint32_t i = thread_idx; i < kNumWords; i += kNumThreads)
        rows[part * kNumWords + i] = counters[i];
    __threadfence();
    sync();
    if (thread_idx == 0)
        landing::raise(&launch.signals->sort_shared[kBuildSort][part], launch.epoch);
    if (thread_idx < kNumBuildParts)
        landing::await_raised(&launch.signals->sort_shared[kBuildSort][thread_idx], launch.epoch, launch.site("build rows", thread_idx));
    sync();

    // 3. Per expert: total over the parts and the share of the parts below; one exclusive prefix over ALL experts
    //    (written where every thread of this SM can read it), and the cursor of this part
    uint32_t totals[kSpan], below[kSpan];
    uint32_t span_total = 0;
    #pragma unroll
    for (uint32_t u = 0; u < kSpan; ++ u) {
        const uint32_t expert = thread_idx * kSpan + u;
        totals[u] = below[u] = 0;
        if (expert < kE) {
            #pragma unroll
            for (uint32_t p = 0; p < kNumBuildParts; ++ p) {
                const uint32_t count = (__ldcg(rows + p * kNumWords + expert / 2) >> (16 * (expert & 1))) & 0xffffu;
                totals[u] += count;
                below[u] += p < part ? count : 0u;
            }
        }
        span_total += totals[u];
    }
    const uint32_t inclusive = math::warp_inclusive_sum(span_total, lane_idx);
    if (lane_idx == 31)
        warp_sums[warp_idx] = inclusive;
    sync();
    uint32_t prefix = inclusive - span_total;
    #pragma unroll
    for (uint32_t w = 0; w < kNumThreads / 32; ++ w)
        prefix += w < warp_idx ? warp_sums[w] : 0u;
    #pragma unroll
    for (uint32_t u = 0; u < kSpan; u += 2) {
        const uint32_t expert = thread_idx * kSpan + u;
        if (expert < kE) {
            const uint32_t prefix_hi = prefix + totals[u];
            starts[expert] = prefix;
            if (expert + 1 < kE)
                starts[expert + 1] = prefix_hi;
            counters[expert / 2] = (prefix + below[u]) | ((prefix_hi + below[u + 1]) << 16);
            if (expert + 2 >= kE)
                starts[kE] = prefix_hi + totals[u + 1];
            prefix = prefix_hi + totals[u + 1];
        }
    }
    __threadfence();
    sync();

    // 4. Part 0 stages the counts: node slot nr's counts sit right in front of its entries (see step 6)
    const uint32_t count_words = kNodeExperts;
    if (part == 0) {
        #pragma unroll
        for (uint32_t u = 0; u < kSpan; ++ u) {
            const uint32_t expert = thread_idx * kSpan + u;
            if (expert < kE and G::node_of(expert / G::kNumExpertsPerRank) != launch.my_node) {
                const uint32_t node = expert / kNodeExperts, nr = G::slot_of_node(node, launch.my_node);
                staging[nr * count_words + __ldcg(starts + node * kNodeExperts) + expert % kNodeExperts] = totals[u];
            }
        }
    }

    // 5. Scatter this part's slice: the entry with global position p goes to staging[(nr + 1) * count_words + p]
    for (uint32_t i0 = begin + thread_idx; i0 < end; i0 += kNumThreads * kNumLoads) {
        uint64_t entries[kNumLoads];
        #pragma unroll
        for (uint32_t u = 0; u < kNumLoads; ++ u)
            entries[u] = i0 + u * kNumThreads < end ? __ldcg(reinterpret_cast<const unsigned long long*>(topk_idx) + i0 + u * kNumThreads) : ~0ull;
        #pragma unroll
        for (uint32_t u = 0; u < kNumLoads; ++ u) {
            const int expert = static_cast<int>(static_cast<int64_t>(entries[u]));
            if (is_remote_expert(expert)) {
                const uint32_t i = i0 + u * kNumThreads, shift = 16 * (expert & 1);
                const uint32_t nr = G::slot_of_node(static_cast<uint32_t>(expert) / kNodeExperts, launch.my_node);
                const uint32_t position = (atomicAdd_block(counters + expert / 2, 1u << shift) >> shift) & 0xffffu;
                staging[(nr + 1) * count_words + position] = i | (row_of_token(nr, i / G::kK) << 16);
            }
        }
    }
    __threadfence();
    sync();
    if (thread_idx == 0)
        landing::raise(&launch.signals->sort_done[kBuildSort][part], launch.epoch);

    // 6. Part 0, once every part is done: one put per remote node, [counts | entries], contiguous in the staging
    //    because node n's entries have the global positions [starts[first expert of n], starts[first of n + 1])
    if (part == 0) {
        if (thread_idx < kNumBuildParts)
            landing::await_raised(&launch.signals->sort_done[kBuildSort][thread_idx], launch.epoch, launch.site("build done", thread_idx));
        sync();
        if (thread_idx < G::kR) {
            const uint32_t node = G::node_of_slot(thread_idx, launch.my_node);
            const uint32_t first = __ldcg(starts + node * kNodeExperts), last = __ldcg(starts + (node + 1) * kNodeExperts);
            make_gin<G>(launch, Stream::Tables, thread_idx).put(
                Slots<G>::outbox_mirror(launch) + G::buf().off_tables() - count_words * 4,
                staging + thread_idx * count_words + first, static_cast<uint64_t>(count_words + last - first) * 4,
                G::counterpart(launch.rank, node));
        }
        __syncwarp();
    }
}

// A receiver's side: poll the counts of the source's message (all ranks of this node: their sum below this rank
// is where this rank's entries start), hand the counts to the scheduler, copy this rank's entries into the slot
// table, raise `tables_ready`.
template <typename G, uint32_t kNumThreads, uint32_t kBarrierIdx, typename publish_t>
CUTLASS_DEVICE void receive_tables(const Launch& launch, const layout::Workspace& workspace, const uint8_t* mirror,
                                   const uint32_t& k, const uint32_t& src_rank, uint32_t* const counts,
                                   const uint32_t& thread_idx, const publish_t& publish) {
    using namespace comm;
    constexpr uint32_t kEr = G::kNumExpertsPerRank, kNodeExperts = kEr * G::kLsa;
    const auto sync = [&]() { ptx::sync_aligned(kNumThreads, kBarrierIdx); };
    const auto msg_counts = reinterpret_cast<const uint32_t*>(mirror + G::buf().off_tables()) - kNodeExperts;
    const auto msg_entries = reinterpret_cast<const uint32_t*>(mirror + G::buf().off_tables());
    const auto offsets = counts + kEr + 1;      // [experts per rank] exclusive prefix of this rank's counts
    const auto base = offsets + kEr;            // entries of the ranks below this one
    const uint32_t my_idx = launch.rank % G::kLsa;

    if (thread_idx == 0)
        *base = 0;
    sync();
    uint32_t below = 0;
    for (uint32_t q = thread_idx; q < kNodeExperts; q += kNumThreads) {
        uint32_t count = ptx::ld_volatile(msg_counts + q);
        landing::await<true>(count, msg_counts + q, launch.site("table counts", src_rank, q));
        if (q / kEr == my_idx)
            counts[q % kEr] = count;
        else if (q / kEr < my_idx)
            below += count;
    }
    below = __reduce_add_sync(0xffffffffu, below);
    if (thread_idx % 32 == 0 and below)
        atomicAdd_block(base, below);
    sync();
    if (thread_idx == 0) {
        uint32_t offset = 0;
        for (uint32_t e = 0; e < kEr; ++ e)
            offsets[e] = offset, offset += counts[e];
        counts[kEr] = offset;                   // this rank's entries in total (the slot of the token count is free here)
    }
    publish();
    sync();

    const uint32_t num_mine = counts[kEr];
    const auto entries = msg_entries + *base;
    for (uint32_t q = thread_idx; q < num_mine; q += kNumThreads) {
        uint32_t lo = 0, hi = kEr;
        while (hi - lo > 1) {
            const uint32_t mid = (lo + hi) / 2;
            if (offsets[mid] <= q) lo = mid; else hi = mid;
        }
        uint32_t entry = ptx::ld_volatile(entries + q);
        landing::await<true>(entry, entries + q, launch.site("table entry", src_rank, q));
        *workspace.get_src_token_topk_idx_ptr(lo, src_rank, q - offsets[lo]) = entry;
    }
    __threadfence();
    sync();
    if (thread_idx == 0)
        landing::raise(&launch.signals->tables_ready[k], launch.epoch);
}

} // namespace deep_gemm::ib
