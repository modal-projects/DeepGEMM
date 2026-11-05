#pragma once

#include <cutlass/arch/barrier.h>

#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/comm/gin.cuh>

namespace deep_gemm::comm {

// 60s timeout, at 2 GHz
constexpr int64_t kNumTimeoutCycles = 60ll * 2000000000ll;

// Spin until `pred()` holds; on timeout, `print_timeout()` runs before the assertion
template <typename pred_t, typename print_timeout_t>
CUTLASS_DEVICE void wait_until(const pred_t& pred, const print_timeout_t& print_timeout) {
    const auto start_clock = clock64();
    while (not pred()) {
        if (clock64() - start_clock >= kNumTimeoutCycles) {
            print_timeout();
            DG_DEVICE_ASSERT(false and "Timeout");
        }
    }
}

CUTLASS_DEVICE void cluster_sync_with_relaxed_arrive() {
    // Perform cluster_sync with `barrier.cluster.arrive.relaxed`
    // This is slightly faster than `cute::cluster_sync` but has weaker memory ordering guarantee
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}

template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0, typename sync_scope_t>
CUTLASS_DEVICE void grid_sync(const layout::Workspace& workspace,
                              const uint32_t& sm_idx, const uint32_t& thread_idx,
                              const sync_scope_t& sync_scope) {
    // NOTES: the implementation idea is from `cooperative_groups::this_grid().sync()`
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        const auto count_ptr = workspace.get_grid_sync_count_ptr<kGridSyncIndex>();
        const auto old_value = ptx::atomic_add_rel(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
        uint32_t new_value;
        wait_until([&]() { return ((new_value = ptx::ld_acq(count_ptr)) ^ old_value) & kFinishSumTag; }, [&]() {
            printf("DeepGEMM grid sync timeout: sm=%u, thread=%u, grid_sync_idx=%u, old=%u, current=%u, expected_tag=%u\n",
                   sm_idx, thread_idx, kGridSyncIndex, old_value, new_value, old_value ^ kFinishSumTag);
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            // Hang forensics: one SM per GPU dumps every CTA that is not at the grid sync yet for this launch
            if (atomicExch(&workspace.signals->ib_stage_dumped, 1u) == 0u) {
                const uint64_t ep = ptx::ld_volatile(workspace.get_ib_epoch_ptr()) + 1;
                for (uint32_t s = 0; s < kNumSMs; ++ s) {
                    const uint64_t v = ptx::ld_volatile(workspace.signals->ib_sm_stage + s);
                    if ((v >> 8) != ep or (v & 0xffu) < 7u)
                        printf("DeepGEMM STAGE sm=%u epoch=%llu (now %llu) stage=%llu\n", s, (unsigned long long) (v >> 8),
                               (unsigned long long) ep, (unsigned long long) (v & 0xffu));
                }
            }
#endif
        });
    }
    sync_scope();
}

template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kGridSyncIndex, uint32_t kTag, typename sync_scope_t>
CUTLASS_DEVICE void nvlink_barrier(const layout::Workspace& workspace,
                                   const layout::SymBuffer<kNumRanks>& sym_buffer,
                                   const uint32_t& sm_idx, const uint32_t& thread_idx,
                                   const sync_scope_t& sync_scope,
                                   const bool& sync_prologue = true,
                                   const bool& sync_epilogue = true,
                                   const bool& do_remote = true) {  // false: NVLink part only (remote readiness proven elsewhere)
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");

    // Grid sync before NVLink signaling
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);

    // NVLink cross-rank barrier, only SM 0 participates
    if (sm_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        // Send signals to remote ranks
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

        // Update status and wait arrival
        if (thread_idx == 0) {
            ptx::red_add(counter_ptr, 1);
            const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
            wait_until([&]() { return ptx::ld_acq_sys(signal_ptr) == target; }, [&]() {
                printf("DeepGEMM NVLink barrier timeout: rank=%d, counter=%d, signal=%d, target=%d, phase=%d, sign=%d, tag=%d\n",
                       sym_buffer.rank_idx, *counter_ptr, ptx::ld_acq_sys(signal_ptr), target, signal_phase, signal_sign, kTag);
            });
        }
    }

    // Grid sync after NVLink completion
    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
}

#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
// NVLink barrier among same-node ranks + epoch-tagged Gin flags across nodes (SM 0 only).
// Remote ranks never touch the GPU-atomic NVLink signal words.
template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kGridSyncIndex, uint32_t kTag,
          typename sync_scope_t, typename make_gin_t>
CUTLASS_DEVICE void hybrid_barrier(const layout::Workspace& workspace,
                                   const layout::SymBuffer<kNumRanks>& sym_buffer,
                                   const uint32_t& sm_idx, const uint32_t& thread_idx,
                                   const sync_scope_t& sync_scope,
                                   const make_gin_t& make_gin, const uint64_t& epoch,
                                   const bool& sync_prologue = true,
                                   const bool& sync_epilogue = true,
                                   const bool& do_remote = true) {  // false: NVLink part only (remote readiness proven elsewhere)
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);

    if (sm_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        if (thread_idx < kNumRanks and sym_buffer.is_lsa(thread_idx))
            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        // One strong signal per launch per tag per remote rank: the counter equals `epoch` once this rank
        // arrived. v2f: posted from one lane in a loop (contending lanes serialize on GDAKI queues)
        // v2h: one lane per remote rank (distinct QPs, no contention); Gin is built lazily so LSA-only runs
        // never construct it
        if (do_remote and thread_idx < kNumRanks and not sym_buffer.is_lsa(thread_idx))
            make_gin().signal_inc(workspace.get_ib_barrier_flag_ptr(kTag, sym_buffer.rank_idx), thread_idx);
        sync_scope();

        if (thread_idx == 0) {
            ptx::red_add(counter_ptr, 1);
            const int target = signal_sign ? 0 : static_cast<int>(sym_buffer.num_lsa_ranks);
            wait_until([&]() { return ptx::ld_acq_sys(signal_ptr) == target; }, [&]() {
                printf("DeepGEMM hybrid barrier (NVLink part) timeout: rank=%d, signal=%d, target=%d, tag=%d\n",
                       sym_buffer.rank_idx, ptx::ld_acq_sys(signal_ptr), target, kTag);
            });
        }
        if (do_remote and thread_idx < kNumRanks and not sym_buffer.is_lsa(thread_idx)) {
            const auto flag_ptr = workspace.get_ib_barrier_flag_ptr(kTag, thread_idx);
            wait_until([&]() { return ptx::ld_acq_sys(flag_ptr) == epoch; }, [&]() {
                printf("DeepGEMM hybrid barrier (IB part) timeout: rank=%d, peer=%u, flag=%lu, epoch=%lu, tag=%d\n",
                       sym_buffer.rank_idx, thread_idx, ptx::ld_acq_sys(flag_ptr), epoch, kTag);
            });
        }
        sync_scope();
    }

    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
}
#endif

} // namespace deep_gemm::comm
