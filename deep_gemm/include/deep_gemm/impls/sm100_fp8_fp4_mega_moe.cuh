#pragma once

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cuda_fp8.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// DG_MEGA_MOE_IB_RX_FN=1: the SST receive step as a real (non-inlined) function that takes scalars and pointers BY VALUE only
// (no closure, nothing of the caller is address-taken), so its temporaries do not take part in the kernel's register
// allocation. Same algorithm as the inline block in the D1 section (see there).
template <uint32_t kE, uint32_t kThreads, uint32_t kBarrier>
__device__ __noinline__ void ib_sst_receive_fn(
    const uint32_t* cnt_all, const uint32_t* stab, uint32_t* s_cnt, const uint32_t sst_n, const uint32_t myj,
    const uint32_t tid, const uint32_t lane,
    uint64_t* recv_cnt0, const int64_t recv_cnt_stride, const uint64_t* recv_sum0, const int64_t recv_sum_stride,
    uint32_t* tok0, const int64_t tok_stride, const uint64_t add_hi, const uint64_t* tab_ready, const uint64_t epoch,
    unsigned long long* tr_landed, unsigned long long* tr_published) {
    uint32_t* const s_off = s_cnt + kE + 1;
    uint32_t* const s_base = s_off + kE;
    if (tid == 0)
        *s_base = 0;
    ptx::sync_aligned(kThreads, kBarrier);
    uint32_t below = 0;
    for (uint32_t q = tid; q < sst_n; q += kThreads) {
        uint32_t v = ptx::ld_volatile(cnt_all + q);
        while (v == 0xffffffffu) v = ptx::ld_volatile(cnt_all + q);
        if (q / kE == myj)
            s_cnt[q % kE] = v;
        else if (q / kE < myj)
            below += v;
    }
    below = __reduce_add_sync(0xffffffff, below);
    if (lane == 0 and below)
        atomicAdd_block(s_base, below);
    ptx::sync_aligned(kThreads, kBarrier);
    if (tid == 0) {
        if (tr_landed) { uint64_t t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); atomicMax(tr_landed, static_cast<unsigned long long>(t_)); }
        uint32_t o = 0;
        for (uint32_t e = 0; e < kE; ++ e)
            s_off[e] = o, o += s_cnt[e];
    }
    for (uint32_t i = tid; i < kE; i += kThreads) {
        *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(recv_cnt0) + recv_cnt_stride * i) = static_cast<uint64_t>(s_cnt[i]);
        __threadfence();
        ptx::atomic_add(reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(recv_sum0) + recv_sum_stride * i), add_hi | static_cast<uint64_t>(s_cnt[i]));
    }
    ptx::sync_aligned(kThreads, kBarrier);
    if (tid == 0 and tr_published) { uint64_t t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); atomicMax(tr_published, static_cast<unsigned long long>(t_)); }
    const uint32_t* const st_tab = stab + *s_base;
    const uint32_t m = s_off[kE - 1] + s_cnt[kE - 1];
    for (uint32_t q = tid; q < m; q += kThreads) {
        uint32_t lo = 0, hi = kE;
        while (hi - lo > 1) { const uint32_t mid = (lo + hi) / 2; if (s_off[mid] <= q) lo = mid; else hi = mid; }
        uint32_t v = ptx::ld_volatile(st_tab + q);
        while (v == 0xffffffffu) v = ptx::ld_volatile(st_tab + q);
        *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(tok0) + tok_stride * lo + static_cast<int64_t>(q - s_off[lo]) * 4) = v;
    }
    __threadfence();
    ptx::sync_aligned(kThreads, kBarrier);
    if (tid == 0)
        ptx::st_rel_sys(tab_ready, epoch);
}

#if defined(DG_MEGA_MOE_IB_SPLIT) and DG_MEGA_MOE_IB_SPLIT
// DG_MEGA_MOE_IB_SPLIT=1: the kernel body is instantiated twice; the few SMs that run the heavy SST / ORD blocks (builders,
// poster, receivers) take the copy that contains them, every other SM runs a copy without them, so their register
// pressure cannot spill values of the hot loops (carrying the code cost +22..+28 us at 1024, start_d1)
template <
    bool kIBHeavy,
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M_L1, uint32_t STORE_BLOCK_M_L2,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumRingTokens,
    uint32_t kNumSFRingTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    typename weight_dtype_t,
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M,
    uint32_t kNumSharedSFTokens = layout::get_num_max_shared_sf_tokens(kNumMaxTokensPerRank),
    typename task_info_t = sched::TaskInfo<kHasShared>
>
CUTLASS_DEVICE void
sm100_fp8_fp4_mega_moe_body(void* y,
                            int* cumulative_local_expert_recv_stats,
                            const uint32_t num_tokens,
                            const layout::SymBuffer<kNumRanks>& sym_buffer,
                            const cute::TmaDescriptor& tensor_map_l1_acts,
                            const cute::TmaDescriptor& tensor_map_l1_acts_sf,
                            const cute::TmaDescriptor& tensor_map_l1_weights,
                            const cute::TmaDescriptor& tensor_map_l1_weights_sf,
                            const cute::TmaDescriptor& tensor_map_l1_output,
                            const cute::TmaDescriptor& tensor_map_l2_acts,
                            const cute::TmaDescriptor& tensor_map_l2_acts_sf,
                            const cute::TmaDescriptor& tensor_map_l2_weights,
                            const cute::TmaDescriptor& tensor_map_l2_weights_sf,
                            const cute::TmaDescriptor& tensor_map_shared_l1_acts,
                            const cute::TmaDescriptor& tensor_map_shared_l1_acts_sf,
                            const cute::TmaDescriptor& tensor_map_shared_l1_weights,
                            const cute::TmaDescriptor& tensor_map_shared_l1_weights_sf,
                            const cute::TmaDescriptor& tensor_map_shared_l1_output,
                            const cute::TmaDescriptor& tensor_map_shared_l2_acts,
                            const cute::TmaDescriptor& tensor_map_shared_l2_acts_sf,
                            const cute::TmaDescriptor& tensor_map_shared_l2_weights,
                            const cute::TmaDescriptor& tensor_map_shared_l2_weights_sf,
                            const uint8_t* sm_locality_domains) {
#else
template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M_L1, uint32_t STORE_BLOCK_M_L2,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumRingTokens,
    uint32_t kNumSFRingTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    typename weight_dtype_t,
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M,
    uint32_t kNumSharedSFTokens = layout::get_num_max_shared_sf_tokens(kNumMaxTokensPerRank),
    typename task_info_t = sched::TaskInfo<kHasShared>
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_impl(void* y,
                            int* cumulative_local_expert_recv_stats,
                            const uint32_t num_tokens,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights_sf,
                            const uint8_t* sm_locality_domains) {
    constexpr bool kIBHeavy = true;
#endif
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;

    // Use one domain when clusters cannot split evenly across device locality domains
    constexpr uint32_t kNumLocalityDomains =
        L1_SHAPE_N % (BLOCK_N * 2 * kNumDeviceLocalityDomains) == 0 and
        L2_SHAPE_N % (BLOCK_N * 2 * kNumDeviceLocalityDomains) == 0 ? kNumDeviceLocalityDomains : 1;

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

    // Thread indices
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights_sf);
    }

    // Workspaces and Buffer
    const auto buffer = layout::MegaMoEBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, kNumSFRingTokens,
        /*with_sf=*/ true,
        kNumSharedExperts
    );
    const auto workspace = buffer.workspace;
    // Tail-branch state carved from the end of ib_pool_combine_idx (the dev server's host build fixes the signals layout):
    // [0, 2048) u32 RET_GROUP counters [2][16][64], then 320 u64 of tail trace (ttrace 64, tchunk 128, tseen 128)
    uint32_t* const tl_fgroup = workspace.signals->ib_pool_combine_idx + layout::MegaMoESignals::kNumMaxIBPoolTokens - 4096;
    uint64_t* const tl_ttrace = reinterpret_cast<uint64_t*>(tl_fgroup + 2048);
    uint64_t* const tl_tchunk = tl_ttrace + 64;
    uint64_t* const tl_tseen = tl_tchunk + 128;
    // Tail knobs on the dev server (its JIT host passes only known defines): bits 8+ of DG_MEGA_MOE_IB_XSITES (whose low 6 bits are
    // a site mask used only with DG_MEGA_MOE_IB_SHARING >= 1): 1 FWD_JOIN, 2 FWD_BATCH=2, 4 C3_REPOLL, 8 C3_SPLIT, 16 C3_EARLY,
    // 32 FWD_DEFER_REARM. Pass XSITES = 63 + (bits << 8).
#define DG_TAIL_BITS ((DG_MEGA_MOE_IB_XSITES) >> 8)
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
#define IB_TRACE(i, cond) do { if (cond) { uint64_t t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); workspace.signals->ib_trace[i] = t_; } } while (0)
// max over every SM that reaches the point (absolute %globaltimer, so no per-launch reset is needed)
#define IB_TRACE_MAX(i, cond) do { if (cond) { uint64_t t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); atomicMax(reinterpret_cast<unsigned long long*>(workspace.signals->ib_trace + (i)), static_cast<unsigned long long>(t_)); } } while (0)
#define SL_TR(j, p, cond) do { if (cond) { uint64_t t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); atomicMax(reinterpret_cast<unsigned long long*>(ib_sl_tr + 5 * (j) + (p)), static_cast<unsigned long long>(t_)); } } while (0)
#define SL_ADD(i, cond) do { if (cond) atomicAdd(reinterpret_cast<unsigned long long*>(ib_sl_tr + 20 + (i)), 1ull); } while (0)
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
#define IB_HIST(h, cond) do { if ((cond) and ib_epoch % 16 == 0) { uint64_t t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); \
    const int64_t d_ = static_cast<int64_t>(t_) - static_cast<int64_t>(ptx::ld_volatile(workspace.signals->ib_trace + 16)); \
    const uint32_t b_ = d_ < 0 ? 0u : (d_ / 8000 > 63 ? 63u : static_cast<uint32_t>(d_ / 8000)); \
    atomicAdd(&workspace.signals->ib_hist[(h)][b_], 1u); } } while (0)
#else
#define IB_HIST(h, cond) do {} while (0)
#endif
// Tail trace slots (ib_ttrace): MAX = latest %globaltimer, MIN = earliest (reset per launch), ADD = counters / sums
#define TT_NOW(t_) uint64_t t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_))
#define TT_MAX(i, cond) do { if (cond) { TT_NOW(t_); atomicMax(reinterpret_cast<unsigned long long*>(tl_ttrace + (i)), static_cast<unsigned long long>(t_)); } } while (0)
#define TT_MIN(i, cond) do { if (cond) { TT_NOW(t_); atomicMin(reinterpret_cast<unsigned long long*>(tl_ttrace + (i)), static_cast<unsigned long long>(t_)); } } while (0)
#define TT_ADD(i, cond, v) do { if (cond) atomicAdd(reinterpret_cast<unsigned long long*>(tl_ttrace + (i)), static_cast<unsigned long long>(v)); } while (0)
#define TT_CLOCK(t_) uint64_t t_ = 0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_))
#else
#define IB_TRACE(i, cond) do {} while (0)
#define IB_TRACE_MAX(i, cond) do {} while (0)
#define SL_TR(j, p, cond) do {} while (0)
#define SL_ADD(i, cond) do {} while (0)
#define IB_HIST(h, cond) do {} while (0)
#define TT_MAX(i, cond) do {} while (0)
#define TT_MIN(i, cond) do {} while (0)
#define TT_ADD(i, cond, v) do {} while (0)
#define TT_CLOCK(t_) constexpr uint64_t t_ = 0
#endif
    // Tail slots: 0 routed L2 task done, 1 shared L2 task done, 2 fwd item: last partial seen, 3 fwd item done (chunk counted),
    // 4 return chunk put: before issue, 5 after issue, 6 forward-combine exit (dispatch warp), 7 epilogue reaches the
    // dispatch sync, 8 (min) source combine start, 9 source combine end, 10 return headers seen (SM 0), 11 dispatch cleanup
    // start, 12 dispatch after the cleanup barrier, 13 (min) epilogue reaches the dispatch sync; 16 fwd items, 17 fwd items
    // whose partials were not all there at the first load, 18 source-combine records not there at the first load,
    // 19 fwd items whose x part was not there, 20 sum of fwd item durations (ns) without partial misses, 21 source combine
    // tokens, 22 sum of source combine token durations (ns), 23 return chunk puts, 24 fwd items done by epilogue warps; 14 deferred re-arm done
    constexpr uint32_t kTTMinMask = (1u << 8) | (1u << 13);

    // Mega MoE over IB, opus design (REDESIGN_OPUS.md): counterpart forwarding. Every cross-node byte moves between
    // GPUs with the same local index. A source puts slices of its input region into a sentinel-filled mirror on its
    // counterpart (the proxy), which publishes counts / slot entries to its NVLink peers exactly like an NVLink
    // source would; peers pull rows from the proxy's mirror over NVLink; L2 rows for remote sources go to the
    // proxy's combine buffer; the proxy reduces them and returns one record per token. Readiness = data != sentinel.
#define IB_STAGE(st, cond) do {} while (0)  // NVLink build; the IB build redefines it below
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
    const auto& ib_dev_comm = *reinterpret_cast<const ncclDevComm_t*>(sym_buffer.gin_dev_comm);
    const auto ib_window = reinterpret_cast<ncclWindow_t>(sym_buffer.gin_window);
#undef IB_STAGE
#define IB_STAGE(st, cond) do { if (cond) atomicMax(reinterpret_cast<unsigned long long*>(workspace.signals->ib_sm_stage + sm_idx), \
    static_cast<unsigned long long>((ib_epoch << 8) | (st))); } while (0)
    const uint64_t ib_epoch = ptx::ld_volatile(workspace.get_ib_epoch_ptr()) + 1;
    IB_STAGE(1, thread_idx == 0);  // entered
    const uint32_t ib_parity = static_cast<uint32_t>(ib_epoch & 1);
    // DG_MEGA_MOE_IB_ORACLE=1 (ceiling measurement, results only valid for identical inputs every launch): nothing the
    // NIC writes is ever re-armed and, from epoch 3 on, the epoch tags are not checked, so every wait on remote data
    // passes at once with the identical previous launch's bytes (both parities filled by epochs 1-2); all puts still
    // run. Kernel time then = this design with zero network wait. NVLink-written partials keep their re-arm.
#ifndef DG_MEGA_MOE_IB_ORACLE
#define DG_MEGA_MOE_IB_ORACLE 0
#endif
    constexpr bool kIBOracle = DG_MEGA_MOE_IB_ORACLE >= 1;
    const bool ib_oracle_now = kIBOracle and ib_epoch > 2;
    // DG_MEGA_MOE_IB_FRONT=1 (front-end layout; the oracle showed D1 cannot finish before ~28 us @128 because the meta
    // SM and the x posters are D1 SMs): the meta is staged and sent by SM 0 (one contiguous idx..header put), the x/SF
    // pieces go out from SMs 1.. upward, so the last num_remote SMs (D1) start their header polls at once; the
    // meta-posted release is issued only when META_FIRST needs it
#if defined(DG_MEGA_MOE_IB_FRONT) and DG_MEGA_MOE_IB_FRONT
    constexpr bool kIBFront = true;
#else
    constexpr bool kIBFront = false;
#endif
#if defined(DG_MEGA_MOE_IB_META_FIRST_ALL) and DG_MEGA_MOE_IB_META_FIRST_ALL
    constexpr bool kIBMetaFirstAll = true;
#else
    constexpr bool kIBMetaFirstAll = false;
#endif
    constexpr uint32_t kIBMetaSM = kIBFront ? 0 : kNumSMs - 1;
    // DG_MEGA_MOE_IB_ORACLE=2: also no puts from epoch 3 on (no NIC traffic at all: the GB200 NIC-vs-NVLink-read
    // contention cannot show up; slightly optimistic, the put issue cost is gone too)
    const bool ib_gin_active = not (kIBOracle and DG_MEGA_MOE_IB_ORACLE >= 2 and ib_epoch > 2);
    const uint32_t ib_lsa = sym_buffer.num_lsa_ranks;
    const uint32_t ib_my_node = sym_buffer.rank_idx / ib_lsa;
    const uint32_t ib_num_nodes = kNumRanks / ib_lsa;
    const auto is_remote = [&](const uint32_t& r) { return not sym_buffer.is_lsa(r); };
    const auto ib_cp = [&](const uint32_t& r, const uint32_t& n) { return n * ib_lsa + r % ib_lsa; };  // counterpart of r on node n
    const auto ib_node_of_slot = [&](const uint32_t& nr) { return nr < ib_my_node ? nr : nr + 1; };     // nr-th remote node
    const auto make_gin = [&](const uint32_t& ctx) { auto g = comm::GinIB(ib_dev_comm, ib_window, sym_buffer.base, ctx); g.active = ib_gin_active; return g; };
    // Bisect knob for exclusive contexts: DG_MEGA_MOE_IB_XSITES = bitmask of posting sites that use their exclusive
    // index + DG_MEGA_MOE_IB_SHARING (1 meta, 2 meta header, 4 x/SF pieces, 8 combine chunks, 16 return header when a
    // node sent nothing, 32 return header after the last chunk); the other sites use GPU sharing on a separate pool of
    // contexts above the exclusive range (so they never touch an exclusive context).
#ifndef DG_MEGA_MOE_IB_XSITES
#define DG_MEGA_MOE_IB_XSITES 63
#endif
    const auto make_gin_site = [&](const uint32_t& site, const uint32_t& xctx, const uint32_t& orig, const uint32_t& xbase) {
        const auto build = [&]() {
            if (DG_MEGA_MOE_IB_SHARING >= 1 and (DG_MEGA_MOE_IB_XSITES & site))
                return comm::GinIB(ib_dev_comm, ib_window, sym_buffer.base, xctx);
            if (DG_MEGA_MOE_IB_SHARING >= 1)  // shared pool above the exclusive range
                return comm::GinIB(ib_dev_comm, ib_window, sym_buffer.base,
                                   xbase + orig % cute::max(1u, static_cast<uint32_t>(DG_MEGA_MOE_IB_CONTEXTS) - xbase), 0);
            return comm::GinIB(ib_dev_comm, ib_window, sym_buffer.base, orig);
        };
        auto g = build(); g.active = ib_gin_active; return g;
    };
    // Mirror layout = the upstream input region layout: x [T][H] fp8, SF [T][H/32], top-k idx [T][K] i64, weights [T][K] f32
    constexpr int64_t kIBT = kNumMaxTokensPerRank;
    constexpr int64_t kIBTokBytes = kHidden, kIBSFBytes = kHidden / 32;
    constexpr int64_t kIBOffSF = kIBT * kIBTokBytes;
    constexpr int64_t kIBOffIdx = kIBOffSF + kIBT * kIBSFBytes;
    constexpr int64_t kIBOffW = kIBOffIdx + kIBT * kNumTopk * 8;
    constexpr int64_t kIBOffHdr = math::constexpr_align<int64_t>(kIBOffW + kIBT * kNumTopk * 4, 128);
    // Early counts (DG_MEGA_MOE_IB_COUNTS_EARLY), after the header: [destination node-local rank][local expert] u32,
    // sentinel 0xffffffff. Always allocated (the host sizes the mirror the same way whatever the knobs)
    constexpr int64_t kIBOffCnt = kIBOffHdr + 128;
    constexpr int64_t kIBCntBytes = math::constexpr_align<int64_t>(static_cast<int64_t>(kNumExperts) * 4, 128);
    // Sender-built slot tables (DG_MEGA_MOE_IB_SENDER_TABLES): the source's entries for this node, [destination node-local
    // rank][local expert][slot] -> top-k entry index (token * K + k), compact (offsets from the counts), sentinel ~0u
    constexpr int64_t kIBOffSTab = kIBOffCnt + kIBCntBytes;
    constexpr int64_t kIBSTabBytes = math::constexpr_align<int64_t>(kIBT * kNumTopk * 4, 128);
    constexpr int64_t kIBMirrorStride = kIBOffSTab + kIBSTabBytes;
    // Return records (one per source token and remote node), forward staging, proxy combine buffer
    // P2: fp8 e4m3 (satfinite) + one f32 scale per 128 columns (5280 B at H = 5120); DG_MEGA_MOE_IB_BF16_COMBINE=1
    // keeps bf16 records. Allocation is sized for bf16 either way, so the layout does not depend on the mode.
#if defined(DG_MEGA_MOE_IB_BF16_COMBINE) and DG_MEGA_MOE_IB_BF16_COMBINE
    constexpr bool kIBFp8Combine = false;
#else
    constexpr bool kIBFp8Combine = true;
#endif
    constexpr int64_t kIBRecBytes = kIBFp8Combine ? kHidden + kHidden / 128 * 4 : kHidden * 2;
    constexpr int64_t kIBRecAlloc = kHidden * 2;
    constexpr int64_t kIBRetHdr = kIBT * kIBRecAlloc;
    constexpr int64_t kIBRetStride = kIBRetHdr + 128;
    constexpr int64_t kIBPCombStride = static_cast<int64_t>(kNumTopk) * kIBT * kHidden * 2;
    constexpr int64_t kIBFStageStride = kIBT * kIBRecAlloc;
    DG_STATIC_ASSERT(kIBRecBytes % 16 == 0, "IB record alignment");
    // DG_MEGA_MOE_IB_RETGROUPS=G (4 or 10; fp8 combine only): the return records are laid out group-major
    // ([group][token][kHidden / G fp8 | kHidden / G / 128 scales]) and the proxy puts each (32-token chunk, column group)
    // slice as soon as its columns are reduced, so with a column-major L2 (DG_MEGA_MOE_L2_COLMAJOR=1) most of the return
    // overlaps the L2 tail instead of following it (a token's whole record used to wait for every one of its experts' L2)
#if defined(DG_MEGA_MOE_IB_RETGROUPS) and DG_MEGA_MOE_IB_RETGROUPS
    constexpr uint32_t kIBRetGroups = DG_MEGA_MOE_IB_RETGROUPS;
#else
    constexpr uint32_t kIBRetGroups = 0;
#endif
    DG_STATIC_ASSERT(kIBRetGroups == 0, "IB return groups are not supported on the integration branch (the return puts are whole records)");
    constexpr uint32_t kIBGCols = kIBRetGroups > 0 ? kHidden / kIBRetGroups : kHidden;
    constexpr int64_t kIBGBytes = kIBGCols + kIBGCols / 128 * 4;  // one token's slice of one group
    DG_STATIC_ASSERT(kIBGBytes % 8 == 0, "IB return group alignment");
    // fp8 value / scale address of column `col` of token `t`'s record in a record buffer (stage or return)
    const auto ib_rec_q = [&](uint8_t* base, const uint32_t& t, const uint32_t& col) -> uint8_t* {
        if constexpr (kIBRetGroups > 0) {
            const uint32_t g = col / kIBGCols;
            return base + (static_cast<int64_t>(g) * kIBT + t) * kIBGBytes + col % kIBGCols;
        } else {
            return base + static_cast<int64_t>(t) * kIBRecBytes + col;
        }
    };
    const auto ib_rec_s = [&](uint8_t* base, const uint32_t& t, const uint32_t& col) -> uint8_t* {
        if constexpr (kIBRetGroups > 0) {
            const uint32_t g = col / kIBGCols;
            return base + (static_cast<int64_t>(g) * kIBT + t) * kIBGBytes + kIBGCols + (col % kIBGCols) / 128 * 4;
        } else {
            return base + static_cast<int64_t>(t) * kIBRecBytes + kHidden + col / 128 * 4;
        }
    };
    // Proxy-local slot table: entry (node-local rank, local expert, slot) -> pair index; the expert hosts read it over
    // NVLink in their pull (one load per row) instead of the proxy writing ~K x T scattered entries over NVLink
    constexpr int64_t kIBPTabStride = static_cast<int64_t>(kNumRanks) * kNumExpertsPerRank * kIBT * 4;
    constexpr int64_t kIBNodeStride = 2 * kIBMirrorStride + 2 * kIBRetStride + kIBPCombStride + kIBFStageStride + kIBPTabStride;
    constexpr uint32_t kIBChunk = 32;  // tokens per dispatch x put and per return put
    // DG_MEGA_MOE_IB_X_PIECE=N: tokens per dispatch x put (sender only; receivers poll rows), default kIBChunk.
    // DG_MEGA_MOE_IB_X_NCTX=C: x / SF pieces spread over at most C contexts (HDR_CTX). With X_ORDER, fewer and larger
    // pieces cut the serialized put-issue cost (~5-7 us per put through the CPU proxy, B300 trace) and keep in-order landing
#ifndef DG_MEGA_MOE_IB_X_PIECE
#define DG_MEGA_MOE_IB_X_PIECE 32
#endif
#ifndef DG_MEGA_MOE_IB_X_NCTX
#define DG_MEGA_MOE_IB_X_NCTX 64
#endif
    constexpr uint32_t kIBXPiece = DG_MEGA_MOE_IB_X_PIECE;
    DG_STATIC_ASSERT(kIBT <= kIBChunk * layout::MegaMoESignals::kIBOMaxChunks, "IB chunk table");
    // Gin context assignment. DG_MEGA_MOE_IB_SHARING == 2 (exclusive THREAD contexts): every concurrent issuer gets its
    // own context, which works because each piece / chunk is posted exactly once per launch by exactly one thread.
    // Layout over R = remote nodes, M = chunks per node: [meta R][meta header R][x/SF pieces R*(1+M)][combine chunks R*M]
    // [return header R] = R * (4 + 2M) contexts. Otherwise the original (shared, modulo) indices.
    constexpr bool kIBThreadCtx = DG_MEGA_MOE_IB_SHARING >= 1;
    constexpr uint32_t kIBM = (kIBT + kIBChunk - 1) / kIBChunk;
    const uint32_t ib_R = ib_num_nodes - 1;
    const auto ctx_meta = [&](const uint32_t& nr, const uint32_t& orig) { return kIBThreadCtx ? nr : orig; };
    const auto ctx_mhdr = [&](const uint32_t& nr, const uint32_t& orig) { return kIBThreadCtx ? ib_R + nr : orig; };
    const auto ctx_piece = [&](const uint32_t& q, const uint32_t& orig) { return kIBThreadCtx ? 2 * ib_R + q : orig; };
    const auto ctx_chunk = [&](const uint32_t& nr, const uint32_t& c, const uint32_t& orig) {
        return kIBThreadCtx ? 2 * ib_R + ib_R * (1 + kIBM) + nr * kIBM + c : orig; };
    const auto ctx_rhdr = [&](const uint32_t& nr, const uint32_t& orig) {
        return kIBThreadCtx ? 2 * ib_R + ib_R * (1 + kIBM) + ib_R * kIBM + nr : orig; };
    const uint32_t ib_xbase = ib_R * (4 + 2 * kIBM);
    // DG_MEGA_MOE_IB_HDR_CTX=1 (shared-context mode): the meta and the meta header get contexts of their own
    // ([0, R) and [R, 2R)) and the x/SF pieces use [2R, CONTEXTS), so the header never queues behind an x chunk on the same
    // QP (with the default indices the header (nr + 2) and x piece q = 1 (q + 1) share context 2 for any context count)
#if defined(DG_MEGA_MOE_IB_HDR_CTX) and DG_MEGA_MOE_IB_HDR_CTX
    constexpr bool kIBHdrCtx = true;
#else
    constexpr bool kIBHdrCtx = false;
#endif
    DG_STATIC_ASSERT(not kIBHdrCtx or DG_MEGA_MOE_IB_CONTEXTS >= 3, "DG_MEGA_MOE_IB_HDR_CTX needs >= 3 Gin contexts");
    // DG_MEGA_MOE_IB_COUNTS_EARLY=1 (probe): SM 0's dispatch warps put this rank's per-expert send counts for each remote
    // node (lsa x experts-per-rank u32, ~768 B) to the counterpart right after the dispatch grid sync, on the meta-header
    // context (posted ~10 us before the meta), plus an identical second put as a doorbell. The D1 scan polls them, records
    // when they land (trace slot 38) and checks them against its own counts (slot 39, sticky mismatch total). Nothing
    // uses them yet: the question is only when they arrive and whether sending them delays anything else.
#if defined(DG_MEGA_MOE_IB_COUNTS_EARLY) and DG_MEGA_MOE_IB_COUNTS_EARLY
    constexpr bool kIBCountsEarly = true;
#else
    constexpr bool kIBCountsEarly = false;
#endif
    DG_STATIC_ASSERT(not kIBCountsEarly or (kIBHdrCtx and not kIBThreadCtx), "DG_MEGA_MOE_IB_COUNTS_EARLY needs HDR_CTX and shared contexts");
    // DG_MEGA_MOE_IB_SPLIT_SCAN=1: the D1 scan first counts our experts' pairs per remote source (idx only) and publishes
    // the counts, then builds the slot tables (and the overlap orders) and raises ib_o_order_ready[k]; remote-row pulls
    // acquire that flag before reading the table. Traced at 1024 tok: header ~46 us, counts published ~80 (scan + publish);
    // the count-only pass should publish at ~header + a few us. Trace slot 32 = tables done (max over scanning SMs).
#if defined(DG_MEGA_MOE_IB_SPLIT_SCAN) and DG_MEGA_MOE_IB_SPLIT_SCAN
    constexpr bool kIBSplitScan = true;
#else
    constexpr bool kIBSplitScan = false;
#endif
    DG_STATIC_ASSERT(not (kIBSplitScan and kIBCountsEarly), "SPLIT_SCAN and COUNTS_EARLY both use trace slot 32");
    // DG_MEGA_MOE_IB_COUNTS_IN_META=1 (prior art: DeepEP notify_dispatch / pplx send counts with the first message): the meta
    // SM counts its top-k entries per remote-node expert while staging the meta, and puts each remote node's slice
    // ([destination node-local rank][local expert] u32, ~768 B Flash / 2.4 KB MegaSeek) right after the meta on the SAME
    // context (so it never sits on an idle one), before the header doorbell. Receivers publish those counts as soon as they
    // land, then the scan only builds the slot tables (SPLIT_SCAN protocol: ib_o_order_ready, remote pulls acquire it).
    // The scan's own per-expert totals are checked against them (trace slot 39 = sticky mismatch total).
    // Compile-time size gate (DG_MEGA_MOE_IB_ST_MIN against kNumMaxTokensPerRank * K): with SENDER_TABLES requested, both
    // SENDER_TABLES and COUNTS_IN_META are compiled out for small max-token kernels. The runtime gate alone still cost
    // +5..+7 us @128 (B300 dev server, gated-off code in the hot paths; SKIP_SCAN + LOCAL_FIRST alone cost nothing there)
#ifndef DG_MEGA_MOE_IB_ST_MIN
#define DG_MEGA_MOE_IB_ST_MIN 0
#endif
    constexpr bool kIBLargeT = static_cast<int64_t>(kNumMaxTokensPerRank) * kNumTopk >= static_cast<int64_t>(DG_MEGA_MOE_IB_ST_MIN);
#if defined(DG_MEGA_MOE_IB_SENDER_TABLES) and DG_MEGA_MOE_IB_SENDER_TABLES
    constexpr bool kIBSTRequested = true;
#else
    constexpr bool kIBSTRequested = false;
#endif
#if defined(DG_MEGA_MOE_IB_COUNTS_IN_META) and DG_MEGA_MOE_IB_COUNTS_IN_META
    constexpr bool kIBCountsInMeta = not kIBSTRequested or kIBLargeT;
#else
    constexpr bool kIBCountsInMeta = false;
#endif
    DG_STATIC_ASSERT(not kIBCountsInMeta or (kIBFront and kIBHdrCtx and not kIBCountsEarly), "COUNTS_IN_META needs FRONT + HDR_CTX, not COUNTS_EARLY");
    constexpr bool kIBEarlyPublish = kIBSplitScan or kIBCountsInMeta;
    // DG_MEGA_MOE_IB_CIM_DISPATCH=1 (with COUNTS_IN_META): the counts come from the dispatch warps' expert_send_count
    // (final after the dispatch grid sync) and SM 0's dispatch thread puts them on the META context, plus a header doorbell,
    // instead of the meta SM counting every top-k entry while staging (B300: CIM alone +5..+10 @128, +12 @MegaSeek 1024)
#if defined(DG_MEGA_MOE_IB_CIM_DISPATCH) and DG_MEGA_MOE_IB_CIM_DISPATCH
    constexpr bool kIBCimDispatch = true;
#else
    constexpr bool kIBCimDispatch = false;
#endif
    // DG_MEGA_MOE_IB_SENDER_TABLES=1 (with COUNTS_IN_META; prior art: DeepEP notify_dispatch / layout.cu): the meta SM also
    // builds the compact per-(destination rank, expert) entry lists for each remote node, in token order, and puts them
    // right after the counts on the same context. Receivers copy only their own slice (~T*K/16 entries per source instead
    // of scanning all T*K) and skip the scan, except the proxy's sort keys for its own counterpart (local memory). Remote
    // pulls poll the weight sentinel themselves (rule A, previously done by the scan).
    constexpr bool kIBSenderTables = kIBSTRequested and kIBLargeT;
    // DG_MEGA_MOE_IB_ST_MIN=N: use the sender tables only for a source with num_tokens * K >= N top-k entries (sender and
    // receivers agree through the header's num_tokens); below that the receiver's scan is cheaper (B300 dev server:
    // MegaSeek 128 tok +17 us with tables, 1024 tok -20 us)
    const auto ib_st_on = [&](const uint32_t& ntok) { return kIBSenderTables and ntok * kNumTopk >= static_cast<uint32_t>(DG_MEGA_MOE_IB_ST_MIN); };
    // With SENDER_TABLES, COUNTS_IN_META follows the same size gate (B300 traces: early counts alone do not start the
    // pulls earlier, they only pay off together with the tables + SKIP_SCAN, and cost +4..+16 us otherwise). Below the gate
    // a source takes the default path (counts published after the scan).
    const auto ib_cim_on = [&](const uint32_t& ntok) { return kIBCountsInMeta and (not kIBSenderTables or ib_st_on(ntok)); };
    const auto ib_early_on = [&](const uint32_t& ntok) { return kIBSplitScan or ib_cim_on(ntok); };
    // DG_MEGA_MOE_IB_SST=1 (start-side sender tables; EP32 late start): one SM's epilogue warps (the SM just below the scan
    // SMs, idle at kernel start) count this rank's top-k entries per remote expert, build each remote node's entry lists
    // (expert-major, roughly token order) and send [the node's counts | its entries] as ONE put per remote node on the header
    // context, before the meta is even staged. Receivers wait for that put only (not the header, not the idx / weight rows):
    // they publish the source's counts, copy their slice of the entries into the slot table and raise ib_o_tab_ready[k].
    // No receiver scans top-k lists on the count path; only the proxy still polls its counterpart's idx / weights (rule A
    // for its forward-combine) and builds its return order, on the last R SMs, which are the only scan SMs left out of the
    // pull rotation. Works with the lean meta (META_SPLIT / STAGE_NT) and replaces COUNTS_IN_META / SENDER_TABLES / SCAN_SLICES
    // at run time. Compiled in only for >= DG_MEGA_MOE_IB_SST_MIN_RANKS ranks (default 32), so EP16 builds are unchanged.
    // Wire layout in the receiver's mirror: counts u32 [node-local rank][local expert] end at kIBOffSTab, entries start there.
#if defined(DG_MEGA_MOE_IB_SST) and DG_MEGA_MOE_IB_SST
#ifndef DG_MEGA_MOE_IB_SST_MIN_RANKS
#define DG_MEGA_MOE_IB_SST_MIN_RANKS 32
#endif
    constexpr bool kIBSST = kNumRanks >= DG_MEGA_MOE_IB_SST_MIN_RANKS;
#else
    constexpr bool kIBSST = false;
#endif
    // DG_MEGA_MOE_IB_SST_FAST=1: sender passes by ROUNDS (a round of global loads costs 2-5 us at kernel start, and 64-bit
    // values in flight are limited by registers): the SST builder reads only the low 32 bits of each top-k index, 16 in
    // flight (half the rounds of 8 x 64-bit); the ORD order pass does the same with 19 x 32-bit per token and keeps the key
    // buckets per warp (match_any) instead of atomics on <= 39 hot shared-memory words
#if defined(DG_MEGA_MOE_IB_SST_FAST) and DG_MEGA_MOE_IB_SST_FAST
    constexpr bool kIBSSTFast = true;
#else
    constexpr bool kIBSSTFast = false;
#endif
    // DG_MEGA_MOE_IB_ORD_COAL=1: COALESCED order pass. The per-token order pass (one thread per token, its top-k loads in
    // flight) spent 28 of its 37 us at 1024 in the key pass: each warp-level load touched 32 different cache lines (lanes
    // 152 B apart). Here a warp reads ONE token per load (lane = top-k slot: 19 x 8 B contiguous), 8 tokens in flight,
    // reduces the key with a warp min, and keeps its buckets in its own shared-memory row (sequential in the warp: no
    // atomics, no match_any).
#if defined(DG_MEGA_MOE_IB_ORD_COAL) and DG_MEGA_MOE_IB_ORD_COAL
    constexpr bool kIBOrdCoal = true;
    DG_STATIC_ASSERT(kNumTopk <= 32, "ORD_COAL: one lane per top-k slot");
#else
    constexpr bool kIBOrdCoal = false;
#endif
    // DG_MEGA_MOE_IB_ORD_MSO=M (with SST_FAST): MULTI-SM order pass. M idle SMs (kNumSMs - remote - 5 - m) each take every M-th
    // token (one round of loads per SM up to 256 * M tokens), keep per-warp key buckets in their own shared memory, publish
    // their per-key totals as a row of plain stores + a GPU-scope flag, read all M rows once for the bucket starts (keys
    // below + the lower SMs' share), then write token -> row / row -> token and raise a second flag. No cross-SM atomics.
#if defined(DG_MEGA_MOE_IB_ORD_MSO) and DG_MEGA_MOE_IB_ORD_MSO
    constexpr uint32_t kIBMSO = kIBSSTFast ? DG_MEGA_MOE_IB_ORD_MSO : 0;
    DG_STATIC_ASSERT(kIBMSO <= 16, "ORD_MSO: at most 16 SMs (flag slots 200.. / 216..)");
#else
    constexpr uint32_t kIBMSO = 0;
#endif
    // DG_MEGA_MOE_IB_PUT_FN=1: the SST / ORD blocks issue their puts through ONE shared non-inlined function (comm::gin_put_fn,
    // scalars and pointers by value) instead of inlining the Gin put at each site
#if defined(DG_MEGA_MOE_IB_PUT_FN) and DG_MEGA_MOE_IB_PUT_FN
    constexpr bool kIBPutFn = DG_MEGA_MOE_IB_SHARING == 0;
#else
    constexpr bool kIBPutFn = false;
#endif
#define IB_PUT(g, ctx, dst, src, bytes, peer) do { if constexpr (kIBPutFn) { if (ib_gin_active) comm::gin_put_fn( \
        reinterpret_cast<const ncclDevComm_t*>(sym_buffer.gin_dev_comm), ib_window, sym_buffer.base, ctx, dst, src, bytes, peer); } \
    else { g.put(dst, src, bytes, peer); } } while (0)
#define IB_PUTV(g, ctx, ptr, val, peer) do { if constexpr (kIBPutFn) { if (ib_gin_active) comm::gin_put_value_fn( \
        reinterpret_cast<const ncclDevComm_t*>(sym_buffer.gin_dev_comm), ib_window, sym_buffer.base, ctx, ptr, val, peer); } \
    else { g.put_value(ptr, val, peer); } } while (0)
    // DG_MEGA_MOE_IB_CG_DROP=mask (codegen bisect, only meaningful with SST_DEAD): compile out 1 early-weight block, 2 SST
    // builder, 4 SST receive, 8 pull-loop waits / polls, 16 NB builder, 32 order builder + poster,
    // 512 ib_P select, 1024 pull-rotation select, 2048 k mapping, 4096 legacy-scan guards, 8192 header-only meta put, 16384 ib_sst = false (constant)
#ifndef DG_MEGA_MOE_IB_CG_DROP
#define DG_MEGA_MOE_IB_CG_DROP 0
#endif
    constexpr uint32_t kCGDrop = DG_MEGA_MOE_IB_CG_DROP;
    // DG_MEGA_MOE_IB_OUTLINE=1: the SST / ORD sender and receiver blocks become non-inlined lambdas, so the code they add does
    // not take part in the register allocation of the hot (MMA / epilogue / pull) loops of the one kernel function
#if defined(DG_MEGA_MOE_IB_OUTLINE) and DG_MEGA_MOE_IB_OUTLINE
#define DG_IB_OUTLINE_BEGIN [&]() __attribute__((noinline)) {
#define DG_IB_OUTLINE_END }();
#else
#define DG_IB_OUTLINE_BEGIN {
#define DG_IB_OUTLINE_END }
#endif
    // Runtime gate of every SST path (both paths are compiled): DG_MEGA_MOE_IB_SST_DEAD=1 is the CODEGEN CONTROL arm: the SST /
    // ORD code is compiled in but the kernel runs the legacy path (the condition is a memory load the compiler cannot fold)
#if defined(DG_MEGA_MOE_IB_SST_RELOAD) and DG_MEGA_MOE_IB_SST_RELOAD
    // DG_MEGA_MOE_IB_SST_RELOAD=1: the runtime flag is re-read from memory at each use (no kernel-wide live register)
#if defined(DG_MEGA_MOE_IB_SST_DEAD) and DG_MEGA_MOE_IB_SST_DEAD
#define ib_sst (kIBSST and not (DG_MEGA_MOE_IB_CG_DROP & 16384) and ptx::ld_volatile(workspace.get_ib_epoch_ptr()) == ~0ull)
#else
#define ib_sst (kIBSST and not (DG_MEGA_MOE_IB_CG_DROP & 16384) and ptx::ld_volatile(workspace.get_ib_epoch_ptr()) != ~0ull)
#endif
#else
#if defined(DG_MEGA_MOE_IB_SST_DEAD) and DG_MEGA_MOE_IB_SST_DEAD
    const bool ib_sst = kIBSST and not (DG_MEGA_MOE_IB_CG_DROP & 16384) and ptx::ld_volatile(workspace.get_ib_epoch_ptr()) == ~0ull;
#else
    const bool ib_sst = kIBSST and not (DG_MEGA_MOE_IB_CG_DROP & 16384) and ptx::ld_volatile(workspace.get_ib_epoch_ptr()) != ~0ull;
#endif
#endif
    DG_STATIC_ASSERT(not kIBSST or (kIBFront and kIBHdrCtx and not kIBThreadCtx and not kIBEarlyPublish and not kIBCountsEarly),
                     "DG_MEGA_MOE_IB_SST needs FRONT + HDR_CTX, shared contexts, and no COUNTS_IN_META / SPLIT_SCAN / COUNTS_EARLY");
    DG_STATIC_ASSERT(not kIBSST or static_cast<int64_t>(kNumMaxTokensPerRank) * kNumTopk < 65536, "SST cursors are 16-bit");
    // DG_MEGA_MOE_IB_SST_W=1 (with SST + META_SPLIT): with SST the pulls can start before the meta lands, and each remote row's
    // pull polls its top-k weight in the mirror. The weight rows < num_tokens go out at kernel start, straight from the input
    // region (no staging), from the idle SM below the SST builder, on the meta context; the meta SM then sends only the
    // header value instead of the whole kIBT-row weight region (146 KB MegaSeek, of which 136 KB are sentinels at 128
    // tokens). Mirror weight rows >= num_tokens are never written (they stay sentinels), rows < nt are re-armed as before.
#if defined(DG_MEGA_MOE_IB_SST_W) and DG_MEGA_MOE_IB_SST_W
    constexpr bool kIBSSTW = kIBSST;
#else
    constexpr bool kIBSSTW = false;
#endif
    DG_STATIC_ASSERT(2 * kNumExpertsPerRank + 2 <= 256, "D1 counters live in the first 1 KB of smem_d");
    const auto hdr_orig = [&](const uint32_t& nr) { return kIBHdrCtx ? ib_R + nr : nr + 2; };
    const auto ib_xnctx = [&]() { return cute::max(1u, cute::min(static_cast<uint32_t>(DG_MEGA_MOE_IB_X_NCTX), static_cast<uint32_t>(DG_MEGA_MOE_IB_CONTEXTS) - 2 * ib_R)); };
    const auto piece_orig = [&](const uint32_t& q) {
        return kIBHdrCtx ? 2 * ib_R + q % ib_xnctx() : q + 1; };
    if (kIBThreadCtx and ib_R * (4 + 2 * kIBM) > DG_MEGA_MOE_IB_CONTEXTS) {
        if (blockIdx.x == 0 and threadIdx.x == 0)
            printf("DeepGEMM IB: DG_MEGA_MOE_IB_SHARING>=1 needs %u Gin contexts (R=%u, M=%u), have %d\n",
                   ib_R * (4 + 2 * kIBM), ib_R, kIBM, DG_MEGA_MOE_IB_CONTEXTS);
        __trap();
    }
    const auto ib_tail = math::advance_ptr<uint8_t>(sym_buffer.get_base_ptr(), math::align<int64_t>(buffer.get_num_bytes(), 128));
    const auto ib_mirror = [&](const uint32_t& p, const uint32_t& n) { return ib_tail + n * kIBNodeStride + p * kIBMirrorStride; };
    const auto ib_ret = [&](const uint32_t& p, const uint32_t& n) { return ib_tail + n * kIBNodeStride + 2 * kIBMirrorStride + p * kIBRetStride; };
    const auto ib_pcomb = [&](const uint32_t& n) { return ib_tail + n * kIBNodeStride + 2 * kIBMirrorStride + 2 * kIBRetStride; };
    const auto ib_fstage = [&](const uint32_t& n) { return ib_pcomb(n) + kIBPCombStride; };
    const auto ib_ptab = [&](const uint32_t& n) { return reinterpret_cast<uint32_t*>(ib_fstage(n) + kIBFStageStride); };
    const auto ib_input_base = reinterpret_cast<uint8_t*>(buffer.input_token_buffer.get_base_ptr());
    const auto ib_atom_add_acq_rel = [](uint32_t* ptr, const uint32_t& v) {
        uint32_t old;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(v) : "memory");
        return old;
    };
    const auto ib_ldv4 = [](const void* ptr) {
        uint4 v;
        asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr) : "memory");
        return v;
    };
    const auto ib_is_sentinel4 = [](const uint4& v) { return (v.x & v.y & v.z & v.w) == 0xffffffffu; };
    constexpr uint64_t kIBIdxSentinel = 0x8000000000000000ull;
// Spin back-off on failed sentinel re-polls: measured neutral-to-negative (EP16 +5-7 us), off by default
#if defined(DG_MEGA_MOE_IB_BACKOFF) and DG_MEGA_MOE_IB_BACKOFF
// DG_MEGA_MOE_IB_BACKOFF_NS=N (default 256; GB200 stall study: fewer polls of NIC-written lines)
#ifndef DG_MEGA_MOE_IB_BACKOFF_NS
#define DG_MEGA_MOE_IB_BACKOFF_NS 256
#endif
#define IB_BACKOFF_NS DG_MEGA_MOE_IB_BACKOFF_NS
#define IB_BACKOFF(n) do { if ((n) > 8) __nanosleep(IB_BACKOFF_NS); } while (0)
#else
#define IB_BACKOFF(n) do {} while (0)
#endif
// DG_MEGA_MOE_IB_NO_DIAG=1: poll loops without the timeout diagnostics (no clock64, no printf argument set-up, no trap path):
// ~40 sites, several inside hot loops (codegen budget; the diagnostics stay the default)
#if defined(DG_MEGA_MOE_IB_NO_DIAG) and DG_MEGA_MOE_IB_NO_DIAG
#define IB_POLL(cond, reload, tag, a, b) do { uint32_t n_ = 0; while (cond) { IB_BACKOFF(++ n_); reload; } } while (0)
#else
#define IB_POLL(cond, reload, tag, a, b) do { const long long t0_ = clock64(); uint32_t n_ = 0; while (cond) { IB_BACKOFF(++ n_); reload; \
        if (clock64() - t0_ > (1ll << 34)) { printf("OPUS_HANG %s rank=%u sm=%u warp=%u lane=%u a=%u b=%u epoch=%llu\n", tag, \
            sym_buffer.rank_idx, sm_idx, warp_idx, lane_idx, (unsigned) (a), (unsigned) (b), (unsigned long long) ib_epoch); asm volatile("trap;"); } } } while (0)
#endif
#endif

    using L2KBlockDependency = sched::L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K>;

    // SF and its buffer configs
    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");

    // UTCCP 4x32 transpose index mapping within each 128-element group
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // Data types
    // NOTES: activations and shared weights are FP8 (e4m3); routed weights may be FP8 or FP4 (e2m1)
    using a_dtype_t = cutlass::float_e4m3_t;
    using shared_b_dtype_t = cutlass::float_e4m3_t;
    constexpr bool kIsWeightFP8 = cute::is_same_v<weight_dtype_t, cutlass::float_e4m3_t>;
    DG_STATIC_ASSERT(kIsWeightFP8 or cute::is_same_v<weight_dtype_t, cutlass::detail::float_e2m1_unpacksmem_t>, "Invalid routed weight type");

    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_BLOCK_K = 128;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    // Prefetch so the TMEM stage is released earlier, keeping the epilogue off the critical path
    constexpr uint32_t kNumEpiPrefetchStages = BLOCK_M == 240 ? 1 : 0;
    constexpr uint32_t ATOM_M = 8;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Scheduler configs
    constexpr uint32_t kNumScheduleStages = 2;
    constexpr uint32_t kNumScheduleConsumerThreads = 2 * kNumEpilogueThreads;

    // Shared memory sizes
    // NOTES: FP8 CD output for L1 (2 TMA stages, BLOCK_N/2 post-SwiGLU), BF16 output for L2 (no TMA, a single stage)
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t AMAX_REDUCTION_WARP_BUFFER_SIZE = STORE_BLOCK_M_L1 / 2; // float2

    struct SharedStorage {
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumDispatchWarps][kNumBytesPerPull];
        union {
            alignas(kSharedMemoryAlignment) cutlass::float_e4m3_t l1[kNumEpilogueWarpgroups][kNumTMAStoreStages][STORE_BLOCK_M_L1 * L1_OUT_BLOCK_N];
            alignas(kSharedMemoryAlignment) nv_bfloat16 l2[kNumEpilogueWarpgroups][STORE_BLOCK_M_L2 * BLOCK_N];
        } smem_d;
        alignas(kSharedMemoryAlignment) a_dtype_t smem_a[kNumStages][LOAD_BLOCK_M * BLOCK_K];
        alignas(kSharedMemoryAlignment) weight_dtype_t smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K];
        uint32_t smem_sfa[kNumStages][SF_BLOCK_M * (BLOCK_K / 128)];
        uint32_t smem_sfb[kNumStages][SF_BLOCK_N * (BLOCK_K / 128)];
        float2 amax_reduction[kNumEpilogueWarps][AMAX_REDUCTION_WARP_BUFFER_SIZE];
        task_info_t task_infos[kNumScheduleStages];
        Barrier dispatch_barriers[kNumDispatchWarps];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier tmem_full_barriers[kNumEpilogueStages];
        Barrier tmem_empty_barriers[kNumEpilogueStages];
        Barrier combine_barriers[kNumEpilogueWarps * 2];
        Barrier task_info_full_barriers[kNumScheduleStages];
        Barrier task_info_empty_barriers[kNumScheduleStages];
        uint32_t tmem_ptr_in_smem;
    };
    constexpr uint32_t kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers);
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);

    // Send buffers
    constexpr auto pull_layout = layout::Data(kNumBytesPerPull);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // A cluster sync is essential for 2CTA tensor memory allocation
    comm::cluster_sync_with_relaxed_arrive();

    // Initialization
    if (warp_idx == 0) {
        // Clean shared memory
        if (cute::elect_one_sync()) {
            // The bytes must be 8 bytes aligned
            ptx::st_shared_bulk(
                shared_storage.expert_token_count,
                math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment)
            );
        }
    } else if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            shared_storage.dispatch_barriers[i].init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Arrive at 2 CTAs, A + B
                shared_storage.full_barriers[i].init(2 * 2);
                shared_storage.empty_barriers[i].init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                // Arrive at all CTAs
                shared_storage.tmem_full_barriers[i].init(1);
                // Arrive only at the leader CTA
                shared_storage.tmem_empty_barriers[i].init(2 * kNumEpilogueThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                shared_storage.combine_barriers[i].init(1);
            #pragma unroll
            for (uint32_t i = 0; i < kNumScheduleStages; ++ i) {
                shared_storage.task_info_full_barriers[i].init(1);
                shared_storage.task_info_empty_barriers[i].init(kNumScheduleConsumerThreads);
            }
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate tensor memory
        IB_STAGE(2, lane_idx == 0);  // before the TMEM allocation
        Allocator().allocate(kNumTmemCols, &shared_storage.tmem_ptr_in_smem);
        IB_STAGE(3, lane_idx == 0);  // TMEM allocated
    }
    cute::cluster_sync();
    IB_STAGE(4, thread_idx == 0);  // prologue + cluster sync done

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();
    IB_STAGE(5, thread_idx == 0);  // PDL dependency done
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
    if (sm_idx == 0 and thread_idx == 0) {  // previous launch's ends relative to its start (slot 16 not yet overwritten)
        workspace.signals->ib_trace[35] = workspace.signals->ib_trace[33] - workspace.signals->ib_trace[16];
        workspace.signals->ib_trace[36] = workspace.signals->ib_trace[34] - workspace.signals->ib_trace[16];
    }
#endif
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
    if (sm_idx == 0) {  // tail slots: previous launch relative to its start, then reset; one entry per thread (the meta
        // poster is SM 0: a serial copy here delayed the first meta post)
        const uint64_t t16 = workspace.signals->ib_trace[16];
        const uint32_t i = threadIdx.x;
        if (i < 32) {
            auto* tt = tl_ttrace;
            const bool is_time = i < 16 or i >= 27;  // 27-31: end-of-kernel stamps (absolute max, reset to 0 per launch)
            tt[32 + i] = is_time ? static_cast<uint64_t>(static_cast<int64_t>(tt[i]) - static_cast<int64_t>(t16)) : tt[i];
            if (i >= 16) tt[i] = 0;
            if ((kTTMinMask >> i) & 1u) tt[i] = ~0ull;
        } else if (i >= 64 and i < 128) {
            auto* tc = tl_tchunk;
            auto* ts = tl_tseen;
            const uint32_t k = i - 64;
            tc[64 + k] = static_cast<uint64_t>(static_cast<int64_t>(tc[k]) - static_cast<int64_t>(t16));
            tc[k] = t16;
            ts[64 + k] = static_cast<uint64_t>(static_cast<int64_t>(ts[k]) - static_cast<int64_t>(t16));
            ts[k] = t16;
        }
        __syncthreads();  // every thread read t16 before thread 0 overwrites it below
    }
#endif
    IB_TRACE(16, sm_idx == 0 and thread_idx == 0);
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
    if (sm_idx == 0 and thread_idx == 0) { workspace.signals->ib_trace[10] = 0; workspace.signals->ib_trace[11] = 0; workspace.signals->ib_trace[37] = 0; }
    // min / max slots (atomics from many SMs; all of them happen after the dispatch grid sync)
    if (sm_idx == 0 and thread_idx == 0) {
        workspace.signals->ib_trace[40] = workspace.signals->ib_trace[41] = workspace.signals->ib_trace[44] = ~0ull;
        workspace.signals->ib_trace[42] = workspace.signals->ib_trace[45] = workspace.signals->ib_trace[43] = 0;
        workspace.signals->ib_trace[46] = ~0ull, workspace.signals->ib_trace[47] = 0;
        __threadfence();
    }
#endif

    // Task scheduler
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumSMs, kNumRanks,
        kNumRingBlocks, kNumLocalityDomains,
        kNumSharedExperts>(
            workspace,
            shared_storage.task_info_full_barriers,
            shared_storage.task_info_empty_barriers,
            shared_storage.task_infos
    );

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 2;

    // Adjust registers
    // NOTES: more experts per rank will cost more schedulers' registers
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
    constexpr bool kIsIB = true;
#else
    constexpr bool kIsIB = false;
#endif
    // IB: the dispatch warps hold Gin state and the row poll/copy; keep the 96-register split
    constexpr bool kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64 and not kIsIB;
    constexpr uint32_t kNumDispatchRegisters = kUseMoreEpilogueRegisters ? 48 : 96;
    constexpr uint32_t kNumNonEpilogueRegisters = kUseMoreEpilogueRegisters ? 40 : 72;
    constexpr uint32_t kNumEpilogueRegisters = kUseMoreEpilogueRegisters ? 208 : 168;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
    // C2 (forward-combine): for each source this GPU proxies, reduce every token's node-local partials (written into
    // our proxy combine buffer by the node's L2 epilogues), stage one record per token, and put 32-token chunks back
    // to the source; the warp completing the last chunk posts the return header. Every mirror row is polled here
    // (rule A: every byte the NIC wrote is awaited). Readiness of the partials: kIBReduceOnArrival (default) = the
    // rows themselves (bf16 0xFFFF sentinel, re-armed after reading; run by the idle dispatch warps during L2),
    // else the upstream per-rank L2 flags (run by the epilogue warps after the L2 grid sync).
#if defined(DG_MEGA_MOE_IB_FLAG_REDUCE) and DG_MEGA_MOE_IB_FLAG_REDUCE
    constexpr bool kIBReduceOnArrival = false;
#else
    constexpr bool kIBReduceOnArrival = true;
#endif
    // Overlap (both default off; the old path is unchanged when off).
    // WAVE_SORT: a source orders its x rows for node n by key1(t) = min local-expert index over t's picks on n (every GPU
    // runs its local expert 0 first), stages them in that order and posts contiguous 32-row chunks; the receivers'
    // host scan (D1) recomputes the same stable order from the top-k meta and stores token -> mirror row (slot table).
    // STREAM_COMBINE: the proxy reduces and returns its records in the order of key2(t) = max local-expert index over
    // t's picks on its node (the wave in which t's last partial lands), so early chunks complete during L2; the source
    // computes the same order and reads each token's record at its position.
#if defined(DG_MEGA_MOE_IB_WAVE_SORT) and DG_MEGA_MOE_IB_WAVE_SORT
    constexpr bool kIBWaveSort = true;
    // Debug modes: 2 = sort + staging copy in token order (identity slots), 3 = sort only, posts from the input
    constexpr uint32_t kIBWSMode = DG_MEGA_MOE_IB_WAVE_SORT;
#else
    constexpr bool kIBWaveSort = false;
    constexpr uint32_t kIBWSMode = 0;
#endif
    // DG_MEGA_MOE_IB_WS_POST=R (with WAVE_SORT=1; EP32 ordered x supply): the chunk SMs only sort + stage their 32 rows and
    // raise a flag; ONE thread per remote node (on an SM without scan work) posts the staged rows in wave order, R rows per
    // put, all on one context, so the wire carries each node's rows in order of first need (per-QP FIFO) instead of
    // ~33 racy pieces per node. R must be a multiple of 32.
#if defined(DG_MEGA_MOE_IB_WS_POST) and DG_MEGA_MOE_IB_WS_POST
    constexpr uint32_t kIBWSPost = DG_MEGA_MOE_IB_WS_POST;
    DG_STATIC_ASSERT(kIBWSMode == 1 and kIBWSPost % 32 == 0, "WS_POST needs WAVE_SORT=1 and a multiple of 32 rows");
#else
    constexpr uint32_t kIBWSPost = 0;
#endif
#if defined(DG_MEGA_MOE_IB_STREAM_COMBINE) and DG_MEGA_MOE_IB_STREAM_COMBINE
    constexpr bool kIBStreamCombine = true;
    // STREAM_COMBINE=2: the last dispatch warp of every SM does not pull; it runs the forward-combine from the start
    // (during the GEMMs), claiming items dynamically in wave order; the pull warps join once their pulls are done
    constexpr bool kIBCombineWarp = DG_MEGA_MOE_IB_STREAM_COMBINE >= 2 and kIBReduceOnArrival;
#else
    constexpr bool kIBStreamCombine = false;
    constexpr bool kIBCombineWarp = false;
#endif
    // DG_MEGA_MOE_IB_FWD_JOIN=1 (tail): forward-combine items are claimed dynamically (one atomic counter per node, in
    // record order), and the epilogue warps join right after their L2 work (after the epilogue grid sync / combine-ready
    // notify, before the dispatch sync they wait at anyway), so the items whose partials land at the end of L2 are reduced
    // by 12 warps per SM instead of 4. DG_MEGA_MOE_IB_FWD_BATCH=B (epilogue warps) / FWD_BATCH_D=B (dispatch warps): the
    // partial loads of B top-k slots are issued together (one HBM round trip per B slots instead of per slot; same order)
#if (defined(DG_MEGA_MOE_IB_FWD_JOIN) and DG_MEGA_MOE_IB_FWD_JOIN) or (DG_TAIL_BITS & 1)
    constexpr bool kIBFwdJoin = kIBReduceOnArrival;
#else
    constexpr bool kIBFwdJoin = false;
#endif
#ifndef DG_MEGA_MOE_IB_FWD_BATCH
#define DG_MEGA_MOE_IB_FWD_BATCH ((DG_TAIL_BITS & 2) ? 2 : 1)
#endif
#ifndef DG_MEGA_MOE_IB_FWD_BATCH_D
#define DG_MEGA_MOE_IB_FWD_BATCH_D 1
#endif
    constexpr bool kIBFwdDyn = kIBCombineWarp or kIBFwdJoin;
    // DG_MEGA_MOE_IB_FWD_SKIP_XPOLL=1 (tail, fewer forward-combine bytes): skip rule A's x / SF row poll for a token that has a node-local
    // expert. Every such row is pulled whole by the GPU hosting that expert, and its pull polls every byte before L1, while the mirror
    // re-arm (E) runs only after the node cleanup barrier (all pulls done). Rows of tokens without node-local experts are still polled
    // here. Saves ~1.3 KB of reads and one HBM round trip per item. Compiled in only for top-k >= 8 at EP16 or less (B300, TS7, 5 rounds:
    // MegaSeek 1024 -1.7, GigaSeek 1024 -6.6, 128 neutral; Flash 1024 -1.0 / +1.7 on two servers; EP32 MegaSeek 1024 +1.5: the forward tail
    // ends 5 us sooner, but the source combine then waits longer for the remote records, which arrive no sooner)
#if defined(DG_MEGA_MOE_IB_FWD_SKIP_XPOLL) and DG_MEGA_MOE_IB_FWD_SKIP_XPOLL
    constexpr bool kIBSkipXPoll = kNumTopk >= 8 and kNumRanks <= 16;
#else
    constexpr bool kIBSkipXPoll = false;
#endif
    // DG_MEGA_MOE_IB_FWD_INTERLEAVE=1 (with FWD_JOIN; tail, EP32): with >= 2 remote nodes the forward-combine items of all of them are
    // claimed from ONE counter, round-robin over the nodes (claim g -> node slot g % R, item g / R, each node's items still in its own
    // wave order). Without it every warp drains node slot 0's items before it claims any of slot 1's, so during L2 all warps wait on
    // slot 0 and the other nodes' items are all left for after L2. No effect with one remote node (EP16).
#if defined(DG_MEGA_MOE_IB_FWD_INTERLEAVE) and DG_MEGA_MOE_IB_FWD_INTERLEAVE
    // compiled in only for EP > 16 (>= 2 remote nodes with 8-GPU NVLink domains): at EP16 the extra code alone cost +13.8 us at MegaSeek
    // 1024 (dispatch-warp register pressure; the runtime path was off), so EP16 builds keep the old loop
    constexpr bool kIBFwdInterleave = kIBFwdDyn and kNumRanks > 16;
#else
    constexpr bool kIBFwdInterleave = false;
#endif
    // DG_MEGA_MOE_IB_FWD_ROTATE=1 (with FWD_JOIN, EP > 16; lighter alternative to FWD_INTERLEAVE): each warp starts its walk over the
    // remote nodes at slot (global warp % R), so the warps spread over the nodes from the start; each node keeps its own claim counter
    // and a warp moves on to the next node once its current one has no items left
#ifndef DG_MEGA_MOE_IB_FWD_ROTATE_MIN
#define DG_MEGA_MOE_IB_FWD_ROTATE_MIN 512
#endif
#if defined(DG_MEGA_MOE_IB_FWD_ROTATE) and DG_MEGA_MOE_IB_FWD_ROTATE
    // compiled in only for EP > 16 and a max-tokens-per-rank allocation >= FWD_ROTATE_MIN: with the code present but gated off it still
    // costs ~+2 us at EP32 128 (code generation; measured on Flash and MegaSeek), so small-batch builds keep the old loop
    constexpr bool kIBFwdRotate = kIBFwdDyn and kNumRanks > 16 and kNumMaxTokensPerRank >= DG_MEGA_MOE_IB_FWD_ROTATE_MIN;
#else
    constexpr bool kIBFwdRotate = false;
#endif
    // DG_MEGA_MOE_IB_FWD_PREFETCH=1 (tail): shorter dependency chain per forward-combine item: the next item is claimed and
    // its token looked up while the current one runs, and the item's x / SF sentinel loads are issued together with its
    // top-k load instead of after the on-node ballot (the tables are acquired via ib_o_order_ready, so plain L2 loads)
#if defined(DG_MEGA_MOE_IB_FWD_PREFETCH) and DG_MEGA_MOE_IB_FWD_PREFETCH
    constexpr bool kIBFwdPrefetch = true;
#else
    constexpr bool kIBFwdPrefetch = false;
#endif
    // DG_MEGA_MOE_IB_C3_EARLY=1 (tail): the epilogue warps only ARRIVE at the dispatch/epilogue barrier before the cleanup
    // (the dispatch warps still wait for them), so the source combine starts right after the L2 grid sync instead of after
    // this SM's dispatch warps finish their forward-combine items and the return puts (~5 us issue each). The combine's
    // shared-memory reuse is safe: the dispatch warps' pulls (the only smem users in that region) end before the last L1.
#if (defined(DG_MEGA_MOE_IB_C3_EARLY) and DG_MEGA_MOE_IB_C3_EARLY) or (DG_TAIL_BITS & 16)
    constexpr bool kIBC3Early = true;
#else
    constexpr bool kIBC3Early = false;
#endif
    // DG_MEGA_MOE_IB_FWD_DEFER_REARM=1 (tail; TRT-LLM Lamport off-path clear, PRIOR_ART 2b): the forward-combine does not
    // re-arm each partial row right after reading it (half the HBM traffic of the post-L2 reduce); the dispatch warps re-arm
    // every node-local (slot, token) row of this launch after the cleanup barrier with the epilogue, overlapping the
    // source combine / return wire. Safe: node peers write these rows again only after the next launch's pre-pull barrier.
    // DG_MEGA_MOE_IB_RET_GROUP=G (tail): the proxy puts G consecutive 32-record chunks as ONE return put once all G are reduced
    // (fewer, larger puts; tests whether the post-L2 return stream is bound by per-put CPU-proxy cost rather than the wire)
#ifndef DG_MEGA_MOE_IB_RET_GROUP
#define DG_MEGA_MOE_IB_RET_GROUP 1
#endif
    constexpr uint32_t kIBRetGroup = DG_MEGA_MOE_IB_RET_GROUP;
    DG_STATIC_ASSERT(kIBRetGroup >= 1, "IB return group");
    // DG_MEGA_MOE_IB_C3_REPOLL=1 (tail): when a return record has not landed at the source combine's first load, every
    // still-sentinel piece of it is re-loaded together each poll round, instead of the per-piece polls that, once the record
    // lands, re-load each of its ~10 fp8 pieces and ~10 scales one HBM round trip after another (~8 us per missed record)
#if (defined(DG_MEGA_MOE_IB_C3_REPOLL) and DG_MEGA_MOE_IB_C3_REPOLL) or (DG_TAIL_BITS & 4)
    constexpr bool kIBC3Repoll = true;
#else
    constexpr bool kIBC3Repoll = false;
#endif
    // DG_MEGA_MOE_IB_C3_SPLIT=1 (tail): the source combine first reduces the node-local slots of ALL its token-chunks (fp32
    // sums parked in our own node's unused proxy-combine region), then adds the records in a second pass, so a warp's second
    // token-chunk no longer starts its local loads only after the first one's record has landed
#if (defined(DG_MEGA_MOE_IB_C3_SPLIT) and DG_MEGA_MOE_IB_C3_SPLIT) or (DG_TAIL_BITS & 8)
    constexpr bool kIBC3Split = true;
#else
    constexpr bool kIBC3Split = false;
#endif
#if (defined(DG_MEGA_MOE_IB_FWD_DEFER_REARM) and DG_MEGA_MOE_IB_FWD_DEFER_REARM) or (DG_TAIL_BITS & 32)
    constexpr bool kIBDeferRearm = kIBReduceOnArrival;
#else
    constexpr bool kIBDeferRearm = false;
#endif
    // DG_MEGA_MOE_IB_EARLY_REARM=1 (tail 3, end of kernel): the dispatch warps re-arm the mirrors BEFORE the cleanup barrier, as
    // soon as every node-local rank has published its combine-ready flag (= its L2 grid sync, so all its L1 pulls from our
    // mirrors are done) and our forward-combine polled every row, instead of after the cleanup barrier at the very end
#if defined(DG_MEGA_MOE_IB_EARLY_REARM) and DG_MEGA_MOE_IB_EARLY_REARM
    constexpr bool kIBEarlyRearm = true;
#else
    constexpr bool kIBEarlyRearm = false;
#endif
    // DG_MEGA_MOE_IB_NO_RET_HDR=1 (tail 3): with tokens to return, the proxy posts no return header (the source polls every byte of
    // every record anyway, rule A) and the source skips the rule-H header wait; zero-token proxies still post it. Removes one
    // Gin put (and its serial issue on the warp that completes the last chunk) per remote node at the end of every launch
#if defined(DG_MEGA_MOE_IB_NO_RET_HDR) and DG_MEGA_MOE_IB_NO_RET_HDR
    constexpr bool kIBNoRetHdr = true;
#else
    constexpr bool kIBNoRetHdr = false;
#endif
    // DG_MEGA_MOE_IB_C3_HDR_GATE (GB200: the NIC's writes into the return buffer are throttled to tens of MB/s while the source
    // combine loads those lines; see CORRECTNESS_LOG 2026-11-06 05:03): 1 = no warp loads a return record before every proxy's
    // return header landed (forces the header put even with NO_RET_HDR); 2 = per-chunk gate without headers: SM 0's first
    // epilogue warp (which takes no combine work) polls only the last line of every return chunk and publishes a GPU-written
    // epoch flag per (node, chunk); the combine waits on the flag of a record's chunk before loading it.
#ifndef DG_MEGA_MOE_IB_C3_HDR_GATE
#define DG_MEGA_MOE_IB_C3_HDR_GATE 0
#endif
    constexpr uint32_t kIBC3Gate = DG_MEGA_MOE_IB_C3_HDR_GATE;
    DG_STATIC_ASSERT(kIBC3Gate <= 2, "C3_HDR_GATE is 0, 1 or 2");
    DG_STATIC_ASSERT(kIBC3Gate == 0 or kIBRetGroup == 1, "C3_HDR_GATE uses the RET_GROUP scratch (tl_fgroup)");
    constexpr bool kIBRetHdrPut = not kIBNoRetHdr or kIBC3Gate == 1;
    // DG_MEGA_MOE_IB_META_DIRECT=1 (Flash 128 latency; FRONT + HDR_CTX, no COUNTS_IN_META / SENDER_TABLES): the meta SM puts
    // the top-k idx rows and weight rows (rows < num_tokens) straight from the input region (no staging: the staging copy
    // was ~4-6 us of the meta's critical path) and the header, as three concurrent puts from three warps.
#ifndef DG_MEGA_MOE_IB_LEAN_MIN_TOPK
#define DG_MEGA_MOE_IB_LEAN_MIN_TOPK 8
#endif
    constexpr bool kIBLeanMeta = kNumTopk >= DG_MEGA_MOE_IB_LEAN_MIN_TOPK;
#if defined(DG_MEGA_MOE_IB_META_DIRECT) and DG_MEGA_MOE_IB_META_DIRECT
    constexpr bool kIBMetaDirect = true;
#else
    constexpr bool kIBMetaDirect = false;
#endif
    // Runtime size gate (per source: the sender uses its token count, the receivers / proxy the source's header count,
    // so both sides always agree): below DG_MEGA_MOE_IB_OVERLAP_MIN_TOKENS the old order is used
#ifndef DG_MEGA_MOE_IB_OVERLAP_MIN_TOKENS
#define DG_MEGA_MOE_IB_OVERLAP_MIN_TOKENS 0
#endif
    // PULL_FLAG_GATE: one warp per GPU (the last SM's last dispatch warp, which does not pull) polls the tails of the x
    // chunks landing in its OWN mirrors (local HBM) and publishes a per-(source node, chunk) epoch word; pull warps
    // busy-poll that word over NVLink before reading a row (then still check every sentinel). Removes the GB200 NIC
    // slowdown caused by hundreds of warps spinning on a proxy's mirror while its NIC writes there.
#if defined(DG_MEGA_MOE_IB_PULL_FLAG_GATE) and DG_MEGA_MOE_IB_PULL_FLAG_GATE
    constexpr bool kIBFlagGate = true;
#else
    constexpr bool kIBFlagGate = false;
#endif
    const auto ib_ws_on = [&](const uint32_t& n_tok) { return kIBWaveSort and n_tok >= DG_MEGA_MOE_IB_OVERLAP_MIN_TOKENS; };
    const auto ib_sc_on = [&](const uint32_t& n_tok) { return kIBStreamCombine and n_tok >= DG_MEGA_MOE_IB_OVERLAP_MIN_TOKENS; };
    // Tables in the (otherwise unused) ib_pool_combine_idx array, local to each GPU:
    //   slot [remote source k][T]: token -> mirror x row (receiver), perm2 [remote node][T]: return position -> token
    //   (proxy), pos2 [remote node][T]: token -> return position (source), keys [SM][2][T]: sort-key scratch
    constexpr uint32_t kIBTabT = static_cast<uint32_t>(kIBT);
    constexpr uint32_t kIBTabSlot = 0, kIBTabPerm2 = 64 * kIBTabT, kIBTabPos2 = 80 * kIBTabT, kIBTabKeys = 96 * kIBTabT;
    DG_STATIC_ASSERT(kIBTabKeys + kNumSMs * 2 * kIBTabT + 4096 <= layout::MegaMoESignals::kNumMaxIBPoolTokens, "IB overlap tables + tail carve-out");
    DG_STATIC_ASSERT(kNumExpertsPerRank < 255, "IB sort keys");
    uint32_t* const ib_tab = workspace.signals->ib_pool_combine_idx;
    // DG_MEGA_MOE_IB_SCAN_SLICES=P (literature: DeepEP channel prefix sums): each remote source's D1 scan is split over P
    // SMs by token range. Slice j counts its entries, publishes its partial counts (count-arrival share kNumSMs / P),
    // takes its slot base from the lower slices' counts (epoch-tagged flags), writes its slot entries and raises its own
    // tables flag; remote-row pulls acquire all P flags of the source. Slice 0 also runs the proxy's sort-key pass
    // (local memory, off the count path). Scratch below the tail carve-out at the end of ib_pool_combine_idx.
#ifndef DG_MEGA_MOE_IB_SCAN_SLICES
#define DG_MEGA_MOE_IB_SCAN_SLICES 1
#endif
    constexpr uint32_t kIBScanSlices = DG_MEGA_MOE_IB_SCAN_SLICES;
    // DG_MEGA_MOE_IB_SLICE_LIST=1: pass A keeps its matches in shared memory, so pass B reloads nothing over NVLink
#if defined(DG_MEGA_MOE_IB_SLICE_LIST)
    constexpr bool kIBSliceList = DG_MEGA_MOE_IB_SLICE_LIST != 0;
#else
    constexpr bool kIBSliceList = false;
#endif
    DG_STATIC_ASSERT(kIBScanSlices >= 1 and kIBScanSlices <= 4, "IB scan slices");
    // Runtime size gate (the buffers / JIT are sized for the max tokens, so it must be a runtime decision): slices pay off
    // only when there is a lot to scan (MegaSeek 1024: -34 us); below DG_MEGA_MOE_IB_SLICE_MIN top-k entries per rank the
    // single-SM scan and its SM layout are used unchanged (P4L cost +3 / +8 us at 128 and +3 at Flash 1024).
#ifndef DG_MEGA_MOE_IB_SLICE_MIN
#define DG_MEGA_MOE_IB_SLICE_MIN 8192
#endif
    // ... and the scan SMs (remote sources x P) stay within DG_MEGA_MOE_IB_SLICE_MAX_SMS (default 32): at EP32 (24 remote
    // sources) P=4 took 96 SMs out of the pull rotation and MegaSeek 1024 got +86 us (B300 and4, 2026-11-06)
#ifndef DG_MEGA_MOE_IB_SLICE_MAX_SMS
#define DG_MEGA_MOE_IB_SLICE_MAX_SMS 32
#endif
    const uint32_t ib_P = [&]() {
        if (ib_sst and not (kCGDrop & 512)) return 1u;  // no receiver scan on the count path
        if (kIBScanSlices <= 1 or num_tokens * kNumTopk < DG_MEGA_MOE_IB_SLICE_MIN) return 1u;
        const uint32_t nrem = cute::max(1u, kNumRanks - ib_lsa);
        uint32_t p = kIBScanSlices;
        while (p > 1 and nrem * p > DG_MEGA_MOE_IB_SLICE_MAX_SMS) -- p;
        return p;
    }();
    DG_STATIC_ASSERT(kIBScanSlices == 1 or not (kIBEarlyPublish or kIBCountsEarly or kIBWaveSort), "SCAN_SLICES excludes CIM / SPLIT_SCAN / COUNTS_EARLY / WAVE_SORT");
    DG_STATIC_ASSERT(kNumExpertsPerRank <= 256, "IB scan slice counts");
    constexpr uint32_t kIBSliceScratch = 64 * 4 * 256 + 2 * 64 * 4 * 2;  // counts u32 + two u64 flag arrays
    DG_STATIC_ASSERT(kIBTabKeys + kNumSMs * 2 * kIBTabT + 4096 + kIBSliceScratch + 64 + 512 <= layout::MegaMoESignals::kNumMaxIBPoolTokens, "IB scan slice scratch");
    uint32_t* const ib_sl_cnt = workspace.signals->ib_pool_combine_idx + layout::MegaMoESignals::kNumMaxIBPoolTokens - 4096 - kIBSliceScratch;  // [64 sources][4 slices][256 experts]
    uint64_t* const ib_sl_cflag = reinterpret_cast<uint64_t*>(ib_sl_cnt + 64 * 4 * 256);  // [64][4] epoch: slice counts written
    uint64_t* const ib_sl_tflag = ib_sl_cflag + 64 * 4;                                  // [64][4] epoch: slice slot entries written
    // Slice trace (DG_MEGA_MOE_IB_TRACE): [0, 20) = max %globaltimer per (slice j, phase p) at 5 * j + p
    // (p: 0 header seen, 1 pass A done, 2 base ready, 3 pass B loop done, 4 tflag raised); [20, 24) counters
    uint64_t* const ib_sl_tr = reinterpret_cast<uint64_t*>(ib_sl_cnt - 64);
    // X_ORDER: per-piece "put issued" epoch flags ([256] u64 below the slice trace)
    uint64_t* const ib_xo_flag = reinterpret_cast<uint64_t*>(ib_sl_cnt - 64 - 512);
    // DG_MEGA_MOE_IB_X_ORDER=1: the x / SF pieces of a Gin context are issued in piece (token) order: piece q waits until
    // piece q - (pieces' context count) was issued. Without it the ~33 pieces per node are issued by as many SMs at
    // once in racy order, the NIC shares bandwidth across the contexts, and every chunk lands near the end of the x
    // window (B300 Flash 1024 trace: pulls starting earlier found remote rows not landed), so no block can start early.
#if defined(DG_MEGA_MOE_IB_X_ORDER) and DG_MEGA_MOE_IB_X_ORDER
    constexpr bool kIBXOrder = true;
#else
    constexpr bool kIBXOrder = false;
#endif
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
    if (kIBScanSlices > 1 and sm_idx == 0 and thread_idx < 4) ib_sl_tr[20 + thread_idx] = 0;  // counters (before grid sync 0)
#endif
    // remote-source index k of rank r (the D1 scan order) and remote-node slot of node n
    const auto ib_nr_of_node = [&](const uint32_t& n) { return n < ib_my_node ? n : n - 1; };
    const auto ib_k_of_rank = [&](const uint32_t& r) { return ib_nr_of_node(r / ib_lsa) * ib_lsa + r % ib_lsa; };
    // Sort-key scratch: shared memory (the epilogue's store buffer, unused before the first task) when it fits
    constexpr uint32_t kIBSortSmemOff = 1024;  // after the D1 per-expert counts
    constexpr uint32_t kIBNumKeys = kNumExpertsPerRank + 1;
    // Sort warps: all epilogue warps when their per-warp key histograms fit in the epilogue store buffer, else fewer (GigaSeek at small
    // batches: 101 keys and BLOCK_M = 16 leave a 4 KB smem_d, too small for 9 x 101 counters; warps beyond kIBSortWarps get empty ranges)
    constexpr uint32_t kIBSortRoom = static_cast<uint32_t>(sizeof(SharedStorage::smem_d)) - kIBSortSmemOff - kIBChunk * 4;
    constexpr uint32_t kIBSortWarps =
        (kNumEpilogueWarps + 1) * kIBNumKeys * 4 <= kIBSortRoom ? kNumEpilogueWarps :
        (4u + 1u) * kIBNumKeys * 4 <= kIBSortRoom ? 4u : (2u + 1u) * kIBNumKeys * 4 <= kIBSortRoom ? 2u : 1u;
    DG_STATIC_ASSERT(kIBSortWarps <= kNumEpilogueWarps, "IB sort warps");
    // [kIBSortWarps][keys] per-warp counts / bases, [keys] bucket starts, then the chunk's inverse map
    constexpr uint32_t kIBSortSmemFixed = kIBSortSmemOff + (kIBSortWarps + 1) * kIBNumKeys * 4 + kIBChunk * 4;
    constexpr bool kIBKeysInSmem = kIBSortSmemFixed + 2 * kIBTabT * 4 <= sizeof(SharedStorage::smem_d);
    DG_STATIC_ASSERT(kIBSortSmemFixed <= sizeof(SharedStorage::smem_d), "IB sort shared memory");
    DG_STATIC_ASSERT(not kIBCountsInMeta or kNumExperts * 4 <= sizeof(SharedStorage::smem_d), "COUNTS_IN_META counters in smem_d");
    DG_STATIC_ASSERT(not kIBSenderTables or (kIBCountsInMeta and not kIBWaveSort), "SENDER_TABLES needs COUNTS_IN_META, no WAVE_SORT");
    // SST: 16-bit counters / cursors, two experts per u32, then per-warp sums and the node starts, in the epilogue store buffer
    constexpr uint32_t kSSTWords = (kNumExperts + 1) / 2;
    DG_STATIC_ASSERT(not kIBSST or (kSSTWords + kNumEpilogueWarps + layout::MegaMoESignals::kIBOMaxNodes + 1) * 4 <= sizeof(SharedStorage::smem_d),
                     "SST counters in smem_d");
    // SST + WAVE_SORT: no receiver but the proxy scans, so the sender packs each entry's mirror row (its wave position) into
    // the entry: bits [0, 16) = token * K + k, bits [16, 32) = x row in the mirror. The chunk-0 SM of each remote node
    // publishes its stable order (token -> position) in the x region of our own node's mirror slot (unused locally) and
    // raises ib_xo_flag[240 + nr]; the SST builder waits for the three flags before its scatter pass.
    constexpr bool kIBSSTWave = kIBSST and kIBWaveSort;
    DG_STATIC_ASSERT(not kIBSSTWave or kIBT < 65536, "SST + WAVE_SORT packs the mirror row in 16 bits");
    // DG_MEGA_MOE_IB_ORD=R (with SST; ordered x supply, EP32): the x rows of each remote node go on the wire in order of
    // FIRST NEED: key(t) = min local-expert index over t's picks on that node (every GPU runs its experts in index order).
    // One SM (kNumSMs - remote - 3) computes the keys of all remote nodes in one pass (one thread per token, top-k loads in
    // flight, no atomics on the keys), bucket-sorts them (any order within a bucket: nobody recomputes it), publishes
    // token -> row and row -> token per node and raises a flag; the chunk SMs then stage their 32 rows; ONE thread per
    // node (SM kNumSMs - remote - 4) posts the staged rows in order, R rows per put, on one context (per-QP FIFO). The SST
    // entries carry each entry's mirror row (as with SST + WAVE_SORT), so receivers need no order table; the proxy's
    // forward-combine polls row t for token t (rule A only needs every row polled once). Below DG_MEGA_MOE_IB_ORD_MIN
    // tokens (default 256) the source posts in token order as before (entries then carry row = token).
#if defined(DG_MEGA_MOE_IB_ORD) and DG_MEGA_MOE_IB_ORD
    constexpr uint32_t kIBOrd = kIBSST ? DG_MEGA_MOE_IB_ORD : 0;
    DG_STATIC_ASSERT(kIBOrd % 32 == 0 and not (kIBOrd > 0 and kIBWaveSort) and (kIBOrd == 0 or kIBXPiece == kIBChunk),
                     "ORD: a multiple of 32 rows per put, no WAVE_SORT, X_PIECE = 32");
    DG_STATIC_ASSERT(kIBOrd == 0 or 4 * (kNumExpertsPerRank + 1) * 4 <= sizeof(SharedStorage::smem_d), "ORD bucket counters in smem_d (<= 4 remote nodes, trapped at run time)");
#else
    constexpr uint32_t kIBOrd = 0;
#endif
#ifndef DG_MEGA_MOE_IB_ORD_MIN
#define DG_MEGA_MOE_IB_ORD_MIN 256
#endif
    const auto ib_ord_on = [&](const uint32_t& n_tok) { return kIBOrd > 0 and ib_sst and n_tok >= DG_MEGA_MOE_IB_ORD_MIN; };
    constexpr bool kIBPacked = kIBSSTWave or kIBOrd > 0;
    DG_STATIC_ASSERT(not kIBPacked or kIBT < 65536, "packed mirror row");
    // Own-node mirror slot, x region (unused locally): [nr][kIBT] token -> row, then [16 + nr][kIBT] row -> token
    const auto ib_ord_pos = [&](const uint32_t& nr) { return reinterpret_cast<uint32_t*>(ib_mirror(ib_parity, ib_my_node)) + static_cast<int64_t>(nr) * kIBT; };
    const auto ib_ord_inv = [&](const uint32_t& nr) { return reinterpret_cast<uint32_t*>(ib_mirror(ib_parity, ib_my_node)) + static_cast<int64_t>(16 + nr) * kIBT; };
    DG_STATIC_ASSERT(32 * 4 <= kIBTokBytes, "ORD tables fit in the x region");
    // DG_MEGA_MOE_IB_SF_FIRST=1: the SF piece is issued before its SM's STREAM_COMBINE order sort (every remote row's pull
    // polls its SF row; today the SF put waits ~25 us for the sort at 1024)
#if defined(DG_MEGA_MOE_IB_SF_FIRST) and DG_MEGA_MOE_IB_SF_FIRST
    constexpr bool kIBSFFirst = true;
#else
    constexpr bool kIBSFFirst = false;
#endif
    // DG_MEGA_MOE_IB_PREWARM=1: the ordered-x poster threads issue one 8-byte put on their context / peer at kernel start
    // (the first put a thread issues costs ~7-8.5 us, later ones ~2.5: probe 2026-11-12)
#if defined(DG_MEGA_MOE_IB_PREWARM) and DG_MEGA_MOE_IB_PREWARM
    constexpr bool kIBPrewarm = true;
#else
    constexpr bool kIBPrewarm = false;
#endif
    // DG_MEGA_MOE_IB_SST_RPUT=1 (with SST_W): the SST put is issued by the threads that issued the early weight put (the
    // SM below the builder), on the same (meta) context, so the issue is warm; the builder hands over the node starts
#if defined(DG_MEGA_MOE_IB_SST_RPUT) and DG_MEGA_MOE_IB_SST_RPUT
    constexpr bool kIBSSTRPut = kIBSSTW;
#else
    constexpr bool kIBSSTRPut = false;
#endif
    // DG_MEGA_MOE_IB_SST_NB=1 (with SST; "node builders"): ONE SM per remote node (SMs kNumSMs - remote - 5 - nr) builds that
    // node's whole start message from one pass over our top-k (one thread per token, its top-k loads in flight): per-expert
    // counts, the token's first-need key (min local-expert index on the node), then bucket order + expert-major entry
    // list (each entry with its mirror row) in a second pass over the now cached top-k. It replaces the single SST builder
    // SM (two passes over all entries with 8 loads in flight: staged at 14.7 / 39.6 us at 128 / 1024) and the ORD order SM.
    // Staging per node: [counts][entries] in the x region of our own node's mirror slot (after the ORD tables).
#if defined(DG_MEGA_MOE_IB_SST_NB) and DG_MEGA_MOE_IB_SST_NB
    constexpr bool kIBSSTNB = kIBSST;
    // SST_NB=2: counters in global memory (red / atomics with their results in flight) instead of shared-memory atomics
    // (which retry under contention: the one-SM passes measured ~10x slower than their load latency, start_o2t), and two
    // tokens per round
    constexpr bool kNBG = DG_MEGA_MOE_IB_SST_NB == 2;
    // SST_NB=3: no atomics on the (hot) key buckets: each warp keeps its own bucket counts in shared memory and resolves
    // equal keys inside the warp with match_any (shared-memory atomics are CAS spin loops: ATOMS.CAST.SPIN in the SASS)
    constexpr bool kNBW = DG_MEGA_MOE_IB_SST_NB == 3;
#else
    constexpr bool kIBSSTNB = false;
    constexpr bool kNBG = false;
    constexpr bool kNBW = false;
#endif
    // DG_MEGA_MOE_IB_ORD_GFENCE=1: the ORD chunk stagers fence at GPU scope (like the meta staging) instead of system scope
#if defined(DG_MEGA_MOE_IB_ORD_GFENCE) and DG_MEGA_MOE_IB_ORD_GFENCE
    constexpr bool kIBOrdGFence = true;
#else
    constexpr bool kIBOrdGFence = false;
#endif
    // DG_MEGA_MOE_IB_NB_XHOLD=1 (with SST_NB + SST_RPUT): x chunks posted in token order (below ORD_MIN) are issued only after
    // the node's start message is staged, so its (warm, 2.5 us) put reaches the NIC before any (cold, 7 us) x put
#if defined(DG_MEGA_MOE_IB_NB_XHOLD) and DG_MEGA_MOE_IB_NB_XHOLD
    constexpr bool kIBNBXHold = kIBSSTNB and kIBSSTRPut;
#else
    constexpr bool kIBNBXHold = false;
#endif
    // DG_MEGA_MOE_IB_WMARK=1 (with SST_NB): landing watermark. With early counts ~576 pull warps per GPU would spin on mirror
    // rows the NIC is still writing (NVLink loads of NIC-written lines). Instead ONE thread per mirror on the proxy (a
    // dispatch warp of SM 0, which never pulls) polls the last 16 bytes of each x row in row order (rows land in posting
    // order per QP) and publishes "rows landed up to" in ordinary memory (ib_o_landed[source node][0], epoch-tagged); a
    // pull waits for the watermark to pass its row (one lane, with a pause) and only then loads the row, still polling
    // every unit for the sentinel as before, so the watermark is a hint and correctness / rule A are unchanged. The start
    // message carries (epoch, ordered bit, num_tokens) 8 bytes ahead of the counts; a source that posts in token order
    // (below ORD_MIN) is published as fully available.
#if defined(DG_MEGA_MOE_IB_WMARK) and DG_MEGA_MOE_IB_WMARK
    constexpr bool kIBWmark = kIBSSTNB;
#else
    constexpr bool kIBWmark = false;
#endif
    constexpr int64_t kNBHdr = kIBWmark ? 8 : 0;
    constexpr uint32_t kNBCap = kNumExperts / 2;  // experts per node (>= 2 nodes)
    constexpr uint32_t kNBKeys = kNumExpertsPerRank + 1;
    DG_STATIC_ASSERT(not kIBSSTNB or (kNBCap + kNumEpilogueWarps + 1 + kNBKeys + (kNBW ? kNumEpilogueWarps * kNBKeys : 0u)) * 4 <= sizeof(SharedStorage::smem_d), "SST_NB counters in smem_d");
    DG_STATIC_ASSERT(not kIBSSTNB or kNumExpertsPerRank < 255, "SST_NB keys");
    constexpr int64_t kNBStgStride = static_cast<int64_t>(kNumExperts) * 4 + kIBT * kNumTopk * 4;
    DG_STATIC_ASSERT(not kIBSSTNB or 64 * kIBT * 4 + layout::MegaMoESignals::kIBOMaxNodes * kNBStgStride <= kIBT * kIBTokBytes, "SST_NB staging fits in the x region");
    const auto ib_nb_g = [&](const uint32_t& nr) {
        return reinterpret_cast<uint32_t*>(ib_mirror(ib_parity, ib_my_node) + 64 * kIBT * 4 + layout::MegaMoESignals::kIBOMaxNodes * kNBStgStride) + static_cast<int64_t>(nr) * (kNBCap + 64); };
    const auto ib_nb_stg = [&](const uint32_t& nr) { return ib_mirror(ib_parity, ib_my_node) + 64 * kIBT * 4 + static_cast<int64_t>(nr) * kNBStgStride; };
    DG_STATIC_ASSERT(kNumEpilogueWarps + 9 <= 28, "SENDER_TABLES scratch in the staging header slot");
    const auto ib_sort_smem = reinterpret_cast<uint8_t*>(&shared_storage.smem_d);
    const auto ib_whist = reinterpret_cast<uint32_t*>(ib_sort_smem + kIBSortSmemOff);
    const auto ib_bstart = ib_whist + kIBSortWarps * kIBNumKeys;
    const auto ib_inv = ib_bstart + kIBNumKeys;
    const auto ib_keys = [&](const uint32_t& which) {
        return kIBKeysInSmem ? reinterpret_cast<uint32_t*>(ib_sort_smem + kIBSortSmemFixed) + which * kIBTabT
                             : ib_tab + kIBTabKeys + (sm_idx * 2 + which) * kIBTabT;
    };
    // Stable counting sort, run by all epilogue threads of an SM: pos(t) = #{t': key(t') < key(t)} + #{t' < t:
    // key(t') == key(t)}. Warp w owns a contiguous token range: per-warp key counts, bases = bucket start + counts of
    // the lower warps, then each warp walks its range 32 tokens at a time (match_any for ties within a step).
    // Every key is read with volatile loads after a CTA barrier; `emit(t, pos)` runs on any warp.
    const auto ib_stable_sort = [&](const uint32_t* keys, const uint32_t& n, const uint32_t& ep_tid, const auto& emit) {
        const auto vkeys = reinterpret_cast<const volatile uint32_t*>(keys);
        const uint32_t w = ep_tid / 32;
        const uint32_t span = math::ceil_div(n, kIBSortWarps * 32u) * 32u;
        const uint32_t lo = cute::min(n, w * span), hi = cute::min(n, lo + span);
        for (uint32_t i = ep_tid; i < kIBSortWarps * kIBNumKeys; i += kNumEpilogueThreads)
            ib_whist[i] = 0;
        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
        for (uint32_t t = lo + lane_idx; t < hi; t += 32)
            atomicAdd_block(ib_whist + w * kIBNumKeys + vkeys[t], 1u);
        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
        for (uint32_t key = ep_tid; key < kIBNumKeys; key += kNumEpilogueThreads) {
            uint32_t run = 0;
            #pragma unroll
            for (uint32_t j = 0; j < kIBSortWarps; ++ j) {
                const uint32_t v = ib_whist[j * kIBNumKeys + key];
                ib_whist[j * kIBNumKeys + key] = run;
                run += v;
            }
            ib_bstart[key] = run;
        }
        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
        if (ep_tid < 32) {
            uint32_t carry = 0;
            for (uint32_t b = 0; b < kIBNumKeys; b += 32) {
                const uint32_t i = b + lane_idx;
                const uint32_t v = i < kIBNumKeys ? ib_bstart[i] : 0u;
                uint32_t incl = v;
                #pragma unroll
                for (uint32_t d = 1; d < 32; d <<= 1) {
                    const uint32_t x = __shfl_up_sync(0xffffffffu, incl, d);
                    if (lane_idx >= d) incl += x;
                }
                if (i < kIBNumKeys) ib_bstart[i] = carry + incl - v;
                carry += __shfl_sync(0xffffffffu, incl, 31);
            }
        }
        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
        const auto my_base = ib_whist + w * kIBNumKeys;
        for (uint32_t t0 = lo; t0 < hi; t0 += 32) {
            const uint32_t t = t0 + lane_idx;
            const bool valid = t < hi;
            const uint32_t key = valid ? vkeys[t] : 0xffffffffu;
            const uint32_t peers = __match_any_sync(0xffffffffu, key);
            const uint32_t rank = __popc(peers & ((1u << lane_idx) - 1u));
            const uint32_t base = valid ? ib_bstart[key] + my_base[key] : 0u;
            __syncwarp();
            if (valid) {
                emit(t, base + rank);
                if (rank == 0) my_base[key] += __popc(peers);
            }
            __syncwarp();
        }
        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    };
    // Sender keys from our own top-k (flat, coalesced, batched loads; shared/global atomics): key1 = min, key2 = max
    // local-expert index over t's picks on node n (key1 = kNumExpertsPerRank, key2 = 0 without picks there)
    const auto ib_build_keys = [&](uint32_t* keys, const uint32_t& n_tok, const uint32_t& n, const bool& is_max, const uint32_t& ep_tid) {
        for (uint32_t t = ep_tid; t < n_tok; t += kNumEpilogueThreads)
            keys[t] = is_max ? 0u : kNumExpertsPerRank;
        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
        const auto in_idx = buffer.input_topk_idx_buffer.get_base_ptr<int64_t>();
        constexpr uint32_t kB = 8;
        for (uint32_t i0 = 0; i0 < n_tok * kNumTopk; i0 += kNumEpilogueThreads * kB) {
            int64_t v[kB];
            #pragma unroll
            for (uint32_t u = 0; u < kB; ++ u) {
                const uint32_t i = i0 + u * kNumEpilogueThreads + ep_tid;
                v[u] = i < n_tok * kNumTopk ? __ldcg(reinterpret_cast<const long long*>(in_idx) + i) : -1ll;
            }
            #pragma unroll
            for (uint32_t u = 0; u < kB; ++ u) {
                const uint32_t i = i0 + u * kNumEpilogueThreads + ep_tid;
                const int e = static_cast<int>(v[u]);
                if (e >= 0 and static_cast<uint32_t>(e) / kNumExpertsPerRank / ib_lsa == n) {
                    const uint32_t le = static_cast<uint32_t>(e) % kNumExpertsPerRank;
                    if (is_max) atomicMax(keys + i / kNumTopk, le);
                    else atomicMin(keys + i / kNumTopk, le);
                }
            }
        }
        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    };
    // DG_MEGA_MOE_IB_NO_OVERLAP (measurement arm: the same kernel without IB / GEMM overlap; 1 = both gates, 2 = dispatch
    // gate only, 3 = return gate only). Dispatch gate: no routed L1 GEMM on any GPU of a node starts before every IB x / SF
    // byte destined to that node has landed (each GPU checks the mirrors it proxies, then pushes its epoch to every node
    // peer; the L1 loader waits for all node peers). Return gate: no forward-combine (the only IB return path) starts before
    // every GPU of the node has finished its L2 (each GPU pushes its epoch after the epilogue grid sync). NVLink overlap is
    // unchanged. Timing only: every pull and every reduce still polls each byte, so correctness does not depend on the flags.
    // Flags: ib_o_landed[own node][slot + node-local rank] (the flag gate only uses remote-node rows).
#if defined(DG_MEGA_MOE_IB_NO_OVERLAP) and DG_MEGA_MOE_IB_NO_OVERLAP
    constexpr uint32_t kIBNoOv = DG_MEGA_MOE_IB_NO_OVERLAP;
#else
    constexpr uint32_t kIBNoOv = 0;
#endif
    constexpr bool kIBNoOvDispatch = kIBNoOv == 1 or kIBNoOv == 2;
    constexpr bool kIBNoOvReturn = kIBNoOv == 1 or kIBNoOv == 3;
    constexpr uint32_t kIBNoOvLandedSlot = 16, kIBNoOvL2Slot = 40;
    const auto ib_noov_push = [&](const uint32_t& slot) {  // one thread
        for (uint32_t i = 0; i < ib_lsa; ++ i)
            ptx::st_rel_sys(sym_buffer.map(&workspace.signals->ib_o_landed[ib_my_node][slot + sym_buffer.rank_idx % ib_lsa],
                                           ib_my_node * ib_lsa + i), ib_epoch);
    };
    const auto ib_noov_wait = [&](const uint32_t& slot) {  // whole warp
        if (lane_idx < ib_lsa)
            while (ptx::ld_acq_sys(&workspace.signals->ib_o_landed[ib_my_node][slot + lane_idx]) < ib_epoch);
        __syncwarp();
    };
    const auto ib_noov_landed_check = [&]() {  // whole warp: every x row and the SF piece of every remote source we proxy
        for (uint32_t nr = 0; nr + 1 < ib_num_nodes; ++ nr) {
            const auto mir = ib_mirror(ib_parity, ib_node_of_slot(nr));
            uint64_t h;
            do { h = ptx::ld_volatile(reinterpret_cast<const uint64_t*>(mir + kIBOffHdr)); } while ((h >> 32) != (ib_epoch & 0xffffffffull));
            const uint32_t nt = static_cast<uint32_t>(h & 0xffffffffull);
            for (uint32_t r = lane_idx; r < nt; r += 32) {
                const auto row = mir + static_cast<int64_t>(r) * kIBTokBytes;
                while (ib_is_sentinel4(ib_ldv4(row)) or ib_is_sentinel4(ib_ldv4(row + kIBTokBytes - 16)));
            }
            if (nt > 0 and lane_idx == 0)
                while (ib_is_sentinel4(ib_ldv4(mir + kIBOffSF + static_cast<int64_t>(nt) * kIBSFBytes - 16)));
        }
        __syncwarp();
        if (lane_idx == 0)
            ib_noov_push(kIBNoOvLandedSlot);
        __syncwarp();
    };
    const auto ib_forward_combine = [&](const uint32_t& global_warp, const uint32_t& num_global_warps, const bool& is_leader,
                                        auto batch_tag, const bool& from_epilogue) {
        if constexpr (kIBNoOvReturn)
            ib_noov_wait(kIBNoOvL2Slot);
        constexpr uint32_t kBatch = static_cast<uint32_t>(decltype(batch_tag)::value);
        DG_STATIC_ASSERT(kBatch >= 1 and kBatch <= 8, "IB forward batch");
        const auto fwd_grid_idx = ptx::get_grid_idx() + 1;
        const uint32_t fi_R = ib_num_nodes - 1;
        const bool fi_on = kIBFwdInterleave and fi_R > 1;
        const auto fi_claim = [&]() {
            uint32_t v = 0;
            if (lane_idx == 0) v = atomicAdd(&workspace.signals->ib_o_fitem[ib_parity][ib_my_node], 1u);  // own-node slot: unused per node
            return __shfl_sync(0xffffffffu, v, 0);
        };
        uint32_t fi_max = 0, fi_g = 0;
        if (fi_on) {
            constexpr uint32_t kFiSplit = kIBRetGroups > 0 ? kIBRetGroups : 4;
            for (uint32_t q = 0; q < fi_R; ++ q) {
                const uint32_t qn = ib_node_of_slot(q);
                const uint32_t qnt = ptx::ld_volatile(&workspace.signals->ib_o_nt[qn]);
                fi_max = cute::max(fi_max, qnt * kFiSplit);
                if (qnt == 0 and is_leader)  // zero-token nodes: their header now, once
                    make_gin_site(16, ctx_rhdr(q, q), q, ib_xbase).put_value(reinterpret_cast<uint64_t*>(ib_ret(ib_parity, ib_my_node) + kIBRetHdr),
                                                                           ib_epoch, ib_cp(sym_buffer.rank_idx, qn));
            }
            fi_g = fi_claim();
        }
        // FWD_ROTATE: start slot, computed once (local decision, any value is correct); only from DG_MEGA_MOE_IB_FWD_ROTATE_MIN tokens on
        const uint32_t rot0 = (kIBFwdRotate and num_tokens >= DG_MEGA_MOE_IB_FWD_ROTATE_MIN) ? global_warp % fi_R : 0u;
        for (uint32_t nr_it = 0; fi_on ? fi_g < fi_R * fi_max : nr_it + 1 < ib_num_nodes; ++ nr_it) {
            uint32_t nr = fi_on ? fi_g % fi_R : nr_it + rot0;
            if (kIBFwdRotate and nr >= fi_R) nr -= fi_R;
            const uint32_t n = ib_node_of_slot(nr);
            const uint32_t s = ib_cp(sym_buffer.rank_idx, n);
            const uint32_t nt = ptx::ld_volatile(&workspace.signals->ib_o_nt[n]);
            const uint32_t nchunks = (nt + kIBChunk - 1) / kIBChunk;
            const auto ret_dst = ib_ret(ib_parity, ib_my_node);
            if (nt == 0 or (fi_on and fi_g / fi_R >= nt * (kIBRetGroups > 0 ? kIBRetGroups : 4))) {
                if (not fi_on and is_leader)
                    make_gin_site(16, ctx_rhdr(nr, nr), nr, ib_xbase).put_value(reinterpret_cast<uint64_t*>(ret_dst + kIBRetHdr), ib_epoch, s);
                if (fi_on) fi_g = fi_claim();
                continue;
            }
            const auto mir = ib_mirror(ib_parity, n);
            const auto fst = ib_fstage(n);
            const auto pc = ib_pcomb(n);
            const auto idx = reinterpret_cast<const uint64_t*>(mir + kIBOffIdx);
            // Work item = (token, quarter of the hidden dim): 4 warps per token, every load of an item issued back to back
            constexpr uint32_t kSplit = kIBRetGroups > 0 ? kIBRetGroups : 4;
            constexpr uint32_t kPieces = kHidden * 2 / (32 * 16);  // 16-byte bf16 units per lane over the row
            constexpr uint32_t kPPI = kPieces / kSplit;
            DG_STATIC_ASSERT(kPieces * 32 * 16 == kHidden * 2 and kPPI * kSplit == kPieces, "IB forward piece");
            constexpr uint32_t kXUnitsPerItem = kIBTokBytes / 16 / kSplit;
            DG_STATIC_ASSERT(kXUnitsPerItem * 16 * kSplit == kIBTokBytes, "IB forward x split");
            const uint32_t k_s = ib_k_of_rank(s);
            if (kIBWaveSort or kIBStreamCombine)
                IB_POLL(ptx::ld_acq_sys(&workspace.signals->ib_o_order_ready[k_s]) != ib_epoch, (void) 0, "fwd_order_ready", k_s, 0);
            const auto claim = [&](const uint32_t& prev) {
                if constexpr (not kIBFwdDyn) {
                    return prev + num_global_warps;
                } else {
                    uint32_t v = 0;
                    if (lane_idx == 0) v = atomicAdd(&workspace.signals->ib_o_fitem[ib_parity][n], 1u);
                    return __shfl_sync(0xffffffffu, v, 0);
                }
            };
            const uint32_t num_items = nt * kSplit;
            // p = record position (= token index unless STREAM_COMBINE), t = token
            const bool pf_on = kIBFwdPrefetch and ib_sc_on(nt) and not fi_on;
            const auto tok_of = [&](const uint32_t& it) -> uint32_t {
                const uint32_t p = it / kSplit;
                if (not ib_sc_on(nt)) return p;
                return pf_on ? __ldcg(ib_tab + kIBTabPerm2 + nr * kIBTabT + p) : ptx::ld_volatile(ib_tab + kIBTabPerm2 + nr * kIBTabT + p);
            };
            uint32_t item = fi_on ? fi_g / fi_R : (kIBFwdDyn ? claim(0) : global_warp);
            uint32_t t_next = item < num_items ? tok_of(item) : 0u;
            while (item < num_items) {
                const uint32_t p_rec = item / kSplit, qtr = item % kSplit;
                const uint32_t t = t_next;
                // next claim issued now, its result broadcast mid-item (the atomic's round trip overlaps this item's loads)
                uint32_t next_item = item + num_global_warps;
                if (kIBFwdDyn and pf_on and lane_idx == 0) next_item = atomicAdd(&workspace.signals->ib_o_fitem[ib_parity][n], 1u);
                const int e = lane_idx < kNumTopk ? static_cast<int>(static_cast<int64_t>(
                    pf_on ? __ldcg(reinterpret_cast<const unsigned long long*>(idx) + t * kNumTopk + lane_idx) : ptx::ld_volatile(idx + t * kNumTopk + lane_idx))) : -1;
                const uint32_t r = e >= 0 ? static_cast<uint32_t>(e) / kNumExpertsPerRank : 0u;
                const bool on = e >= 0 and r / ib_lsa == ib_my_node;
                uint32_t on_mask = 0;
                if (not pf_on) on_mask = __ballot_sync(0xffffffffu, on);
                if constexpr (not kIBReduceOnArrival) {
                    const auto flag_ptr = workspace.get_combine_ready_grid_idx_ptr(r);
                    comm::wait_until([&]() { return __all_sync(0xffffffffu, not on or ptx::ld_acq_sys(flag_ptr) == fwd_grid_idx); },
                                     [&]() { printf("DeepGEMM IB forward-combine flags timeout: rank=%u, source=%u, token=%u\n", sym_buffer.rank_idx, s, t); });
                }
                TT_CLOCK(tt_item0);
                bool tt_miss = false;
                // Rule A: this item's part of the token's x row (and, for quarter 0, its SF row) has landed
                if (not (kIBSkipXPoll and not pf_on and on_mask != 0)) {
                    constexpr uint32_t kU = (kXUnitsPerItem + 31) / 32;
                    const uint32_t xslot = ib_ws_on(nt) ? ptx::ld_volatile(ib_tab + kIBTabSlot + k_s * kIBTabT + t) : t;
                    const auto xrow = mir + static_cast<int64_t>(xslot) * kIBTokBytes + static_cast<int64_t>(qtr) * kXUnitsPerItem * 16;
                    uint4 v[kU];
                    #pragma unroll
                    for (uint32_t u = 0; u < kU; ++ u)
                        v[u] = u * 32 + lane_idx < kXUnitsPerItem ? ib_ldv4(xrow + (u * 32 + lane_idx) * 16) : uint4{0, 0, 0, 0};
                    const bool has_sf = qtr == 0 and lane_idx < kIBSFBytes / 16;
                    const auto sfp = mir + kIBOffSF + static_cast<int64_t>(t) * kIBSFBytes + lane_idx * 16;
                    uint4 sfv = {0, 0, 0, 0};
                    if (pf_on and has_sf) sfv = ib_ldv4(sfp);
                    if (pf_on) {
                        on_mask = __ballot_sync(0xffffffffu, on);
                        // the next item's token, off the critical path (its claim has returned by now)
                        if constexpr (kIBFwdDyn) next_item = __shfl_sync(0xffffffffu, next_item, 0);
                        if (next_item < num_items) t_next = tok_of(next_item);
                    }
                    TT_ADD(19, __any_sync(0xffffffffu, ib_is_sentinel4(v[0])) and lane_idx == 0, 1);
                    #pragma unroll
                    for (uint32_t u = 0; u < kU; ++ u)
                        IB_POLL(ib_is_sentinel4(v[u]), v[u] = ib_ldv4(xrow + (u * 32 + lane_idx) * 16), "c2_x", s, t);
                    if (has_sf) {
                        if (pf_on) IB_POLL(ib_is_sentinel4(sfv), sfv = ib_ldv4(sfp), "c2_sf", s, t);
                        else IB_POLL(ib_is_sentinel4(ib_ldv4(sfp)), (void) 0, "c2_sf", s, t);
                    }
                }
                // Reduce node-local slots in top-k order (fp32): per slot, the item's pieces are loaded together
                float2 acc[kPPI][4];
                #pragma unroll
                for (uint32_t i = 0; i < kPPI; ++ i)
                    #pragma unroll
                    for (uint32_t l = 0; l < 4; ++ l)
                        acc[i][l] = {0.f, 0.f};
                uint32_t m = on_mask;
                while (m) {
                    // kBatch slots at a time (top-k order kept): all their loads first, then the checks / accumulation
                    uint32_t ks[kBatch];
                    #pragma unroll
                    for (uint32_t bb = 0; bb < kBatch; ++ bb) {
                        ks[bb] = m ? static_cast<uint32_t>(__ffs(m) - 1) : 0xffffffffu;
                        m &= m - 1;
                    }
                    uint4 v[kBatch][kPPI];
                    #pragma unroll
                    for (uint32_t bb = 0; bb < kBatch; ++ bb) {
                        if (ks[bb] == 0xffffffffu) continue;
                        const auto row = pc + (static_cast<int64_t>(ks[bb]) * kIBT + t) * kHidden * 2;
                        #pragma unroll
                        for (uint32_t i = 0; i < kPPI; ++ i) {
                            const auto ptr = row + ((qtr * kPPI + i) * 32 + lane_idx) * 16;
                            if constexpr (kIBReduceOnArrival) v[bb][i] = ib_ldv4(ptr);
                            else v[bb][i] = __ldcg(reinterpret_cast<const uint4*>(ptr));
                        }
                    }
                    #pragma unroll
                    for (uint32_t bb = 0; bb < kBatch; ++ bb) {
                        if (ks[bb] == 0xffffffffu) continue;
                        const uint32_t k = ks[bb];
                        const auto row = pc + (static_cast<int64_t>(k) * kIBT + t) * kHidden * 2;
                        #pragma unroll
                        for (uint32_t i = 0; i < kPPI; ++ i) {
                            const auto ptr = row + ((qtr * kPPI + i) * 32 + lane_idx) * 16;
                            if constexpr (kIBReduceOnArrival) {
                                tt_miss |= ib_is_sentinel4(v[bb][i]);
                                IB_POLL(ib_is_sentinel4(v[bb][i]), v[bb][i] = ib_ldv4(ptr), "c2_pcomb", t, k);
                                if constexpr (not kIBDeferRearm)
                                    *reinterpret_cast<uint4*>(ptr) = uint4{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
                            }
                            const auto b = reinterpret_cast<const nv_bfloat162*>(&v[bb][i]);
                            #pragma unroll
                            for (uint32_t l = 0; l < 4; ++ l)
                                ptx::accumulate(acc[i][l], b[l]);
                        }
                    }
                }
                tt_miss = __any_sync(0xffffffffu, tt_miss);
                TT_MAX(2, lane_idx == 0);  // fwd item: all its partials seen
                const auto rec = fst + static_cast<int64_t>(p_rec) * kIBRecBytes;
                #pragma unroll
                for (uint32_t i = 0; i < kPPI; ++ i) {
                    const uint32_t p = qtr * kPPI + i;
                    if constexpr (kIBFp8Combine) {
                        // 8 columns per lane, 16 lanes per 128-column block: amax over the half-warp
                        float amax = 0.f;
                        #pragma unroll
                        for (uint32_t l = 0; l < 4; ++ l)
                            amax = fmaxf(amax, fmaxf(fabsf(acc[i][l].x), fabsf(acc[i][l].y)));
                        #pragma unroll
                        for (uint32_t d = 1; d < 16; d <<= 1)
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
                        const float scale = amax > 0.f ? amax / 448.f : 1.f;
                        const float inv = 1.f / scale;
                        uint2 q;
                        auto qs = reinterpret_cast<__nv_fp8x2_storage_t*>(&q);
                        #pragma unroll
                        for (uint32_t l = 0; l < 4; ++ l)
                            qs[l] = __nv_cvt_float2_to_fp8x2(make_float2(acc[i][l].x * inv, acc[i][l].y * inv), __NV_SATFINITE, __NV_E4M3);
                        const uint32_t col = (p * 32 + lane_idx) * 8;
                        *reinterpret_cast<uint2*>(ib_rec_q(fst, p_rec, col)) = q;
                        if (lane_idx % 16 == 0)
                            *reinterpret_cast<float*>(ib_rec_s(fst, p_rec, col)) = scale;
                    } else {
                        uint4 o;
                        auto ob = reinterpret_cast<nv_bfloat162*>(&o);
                        #pragma unroll
                        for (uint32_t l = 0; l < 4; ++ l)
                            ob[l] = __float22bfloat162_rn(acc[i][l]);
                        *reinterpret_cast<uint4*>(rec + (p * 32 + lane_idx) * 16) = o;
                    }
                }
                __syncwarp();
                // The warp completing a chunk posts it (lane 0); if it was the last chunk, lane 1 posts the return header
                // at the same time on another context (Gin ops cost ~10 us of issue each: never serialize two)
                const uint32_t c = p_rec / kIBChunk, size = cute::min(kIBChunk, nt - c * kIBChunk);
                uint32_t state = 0;
                // Grouped returns: the unit is (chunk, column group) = `size` items, each group slice is put on its own
                constexpr uint32_t kUnitsPerChunk = kIBRetGroups > 0 ? kIBRetGroups : 1;
                const uint32_t unit = kIBRetGroups > 0 ? c * kIBRetGroups + qtr : c;
                // RET_GROUP: chunks [g0, g1) go out as one put, issued by the warp completing the group's last chunk
                const uint32_t g0 = c / kIBRetGroup * kIBRetGroup, g1 = cute::min(nchunks, g0 + kIBRetGroup);
                bool do_put = false;
                if (lane_idx == 0) {
                    __threadfence();
                    if (ib_atom_add_acq_rel(&workspace.signals->ib_o_fchunk[ib_parity][n][unit], 1u) + 1 == size * (kSplit / kUnitsPerChunk)) {
                        do_put = kIBRetGroup == 1 or
                            ib_atom_add_acq_rel(tl_fgroup + (ib_parity * 16 + n) * 64 + c / kIBRetGroup, 1u) + 1 == g1 - g0;
                        state = ib_atom_add_acq_rel(&workspace.signals->ib_o_fdone[ib_parity][n], 1u) + 1 == nchunks * kUnitsPerChunk ? 2u : 1u;
                    }
                }
                state = __shfl_sync(0xffffffffu, state, 0);
                TT_MAX(3, lane_idx == 0);  // fwd item done
                TT_ADD(16, lane_idx == 0, 1);
                TT_ADD(24, lane_idx == 0 and from_epilogue, 1);
                TT_ADD(17, lane_idx == 0 and tt_miss, 1);
                {
                    TT_CLOCK(tt_item1);
                    TT_ADD(20, lane_idx == 0 and not tt_miss, tt_item1 - tt_item0);
                }
                if (state > 0 and lane_idx == 0 and do_put) {
                    const uint32_t pc0 = kIBRetGroup == 1 ? c : g0;
                    const uint32_t precs = kIBRetGroup == 1 ? size : cute::min(nt, g1 * kIBChunk) - g0 * kIBChunk;
                    const int64_t off = static_cast<int64_t>(pc0) * kIBChunk * kIBRecBytes;
                    TT_MAX(4, true);
                    TT_ADD(23, true, 1);
                    TT_ADD(26, workspace.signals->ib_trace[5] > workspace.signals->ib_trace[16], 1);  // put after our L2 grid sync
                    {
                        const auto gin_c = make_gin_site(8, ctx_chunk(nr, pc0, pc0 + nr), pc0 + nr, ib_xbase);
                        gin_c.put(ret_dst + off, fst + off, static_cast<uint64_t>(precs) * kIBRecBytes, s);
                        // DG_MEGA_MOE_IB_RET_CHUNK_DB=1: a trailing 8-byte put on the same context after the LAST chunk of
                        // this node (=2: after every chunk), the header-doorbell workaround applied to the return chunks
                        // (the last op on a CPU-proxy context can sit until the next one; with NO_RET_HDR nothing follows it)
#if defined(DG_MEGA_MOE_IB_RET_CHUNK_DB) and DG_MEGA_MOE_IB_RET_CHUNK_DB
                        if (DG_MEGA_MOE_IB_RET_CHUNK_DB >= 2 or state == 2)
                            gin_c.put_value(&workspace.signals->ib_o_scratch, ib_epoch, s);
#endif
                    }
                    TT_MAX(5, true);
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
                    if (nr == 0 and pc0 < 64) { TT_NOW(t_); tl_tchunk[pc0] = t_; }
#endif
                    IB_TRACE_MAX(14, true);  // proxy: return record chunk put issued, last one
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
                    {
                        uint64_t t_;
                        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_));
                        atomicMin(reinterpret_cast<unsigned long long*>(&workspace.signals->ib_trace[44]), static_cast<unsigned long long>(t_));
                        atomicMax(reinterpret_cast<unsigned long long*>(&workspace.signals->ib_trace[45]), static_cast<unsigned long long>(t_));
                    }
#endif
                }
                if (state == 2 and lane_idx == 1 and kIBRetHdrPut)
                    make_gin_site(32, ctx_rhdr(nr, c + nr + 1), c + nr + 1, ib_xbase).put_value(reinterpret_cast<uint64_t*>(ret_dst + kIBRetHdr), ib_epoch, s);
#if defined(DG_MEGA_MOE_IB_RET_DB) and DG_MEGA_MOE_IB_RET_DB
                if (state == 2 and lane_idx == 1 and kIBRetHdrPut)  // doorbell for the header itself
                    make_gin_site(32, ctx_rhdr(nr, c + nr + 1), c + nr + 1, ib_xbase).put_value(reinterpret_cast<uint64_t*>(ret_dst + kIBRetHdr), ib_epoch, s);
#endif
                __syncwarp();
                if (fi_on)
                    break;  // FWD_INTERLEAVE: one item per claim, the next claim may be another node's
                if (pf_on) {
                    item = next_item;
                } else {
                    item = claim(item);
                    t_next = item < num_items ? tok_of(item) : 0u;
                }
            }
            if (fi_on)
                fi_g = fi_claim();
        }
    };
#endif

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        // Dispatch warps
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            // TODO: figure out better unrolling
            // Now, `unroll` is better than `unroll 8`
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                // Allocate slots for each token-topk
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[i * kNumTopk + lane_idx]);
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        {
            // Next launch's forward-combine counters (the other parity was last used by the previous launch)
            if (sm_idx == 1) {
                for (uint32_t i = thread_idx; i < layout::MegaMoESignals::kIBOMaxNodes * layout::MegaMoESignals::kIBOMaxChunks; i += kNumDispatchThreads)
                    workspace.signals->ib_o_fchunk[ib_parity ^ 1][i / layout::MegaMoESignals::kIBOMaxChunks][i % layout::MegaMoESignals::kIBOMaxChunks] = 0;
                if (thread_idx < layout::MegaMoESignals::kIBOMaxNodes) {
                    workspace.signals->ib_o_fdone[ib_parity ^ 1][thread_idx] = 0;
                }
                if (kIBRetGroup > 1) {  // both parities (same history-independence as the claim counters below)
                    for (uint32_t i = thread_idx; i < 2 * layout::MegaMoESignals::kIBOMaxNodes * layout::MegaMoESignals::kIBOMaxChunks; i += kNumDispatchThreads)
                        tl_fgroup[i] = 0;
                }
                // Reset in EVERY variant, not only the FWD_JOIN / combine-warp ones (correctness, 2026-11-06): on a warm
                // buffer shared by several variants (dev server A/B), a non-JOIN launch left the counter alone, so a later
                // JOIN launch could start claiming at an earlier JOIN launch's final count and never reduce the items
                // below it (gb32 1106-024748-61af: e_fdone + c3_q hangs, no c2_* poll hang). RET_GROUP's tl_fgroup had
                // the same hazard (reset above, now both parities).
                // Both parities: this launch's own counter too, so a launch never depends on what earlier (possibly
                // older-header) variants left behind. Safe: every claim happens after the dispatch grid sync below.
#if defined(DG_MEGA_MOE_IB_LEGACY_FITEM_RESET) and DG_MEGA_MOE_IB_LEGACY_FITEM_RESET
                // A/B only: the old reset (other parity, FWD_JOIN variants only)
                if (kIBFwdDyn and thread_idx < layout::MegaMoESignals::kIBOMaxNodes)
                    workspace.signals->ib_o_fitem[ib_parity ^ 1][thread_idx] = 0;
#else
                if (thread_idx < 2 * layout::MegaMoESignals::kIBOMaxNodes)
                    workspace.signals->ib_o_fitem[thread_idx / layout::MegaMoESignals::kIBOMaxNodes][thread_idx % layout::MegaMoESignals::kIBOMaxNodes] = 0;
#endif
            }
        }
#endif
        IB_STAGE(6, thread_idx == 0);  // dispatch warps started
        // Count experts' tokens
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Get SM offset (~6.5 us)
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
            shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source indices (~2 us with 512 tokens)
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            if (is_remote(dst_rank_idx))
                return;  // the counterpart (proxy) on that node writes this entry
#endif
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        // Grid sync
        IB_STAGE(7, thread_idx == 0);  // at dispatch grid sync 0
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        // Write expert count
        if (sm_idx == 0) {
            // Push this launch's grid index to every peer for tagging combine readiness; +1 differs from a zeroed workspace
            DG_STATIC_ASSERT(kNumRanks <= kNumDispatchThreads, "Insufficient threads for the grid index push");
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            if (thread_idx < kNumRanks and not is_remote(thread_idx))
#else
            if (thread_idx < kNumRanks)
#endif
                *sym_buffer.map(workspace.get_peer_grid_idx_ptr(sym_buffer.rank_idx), thread_idx) = ptx::get_grid_idx() + 1;
            __syncwarp();

            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
                if (is_remote(dst_rank_idx))
                    continue;  // published by the proxy on that node
#endif
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            if (kIBCountsEarly or (kIBCimDispatch and ib_cim_on(num_tokens))) {
                // Stage the remote-destination counts ([destination rank][local expert] u32) in our own node's mirror slot
                // (unused locally) and put each remote node's slice to our counterpart there (same offset)
                const auto stg = ib_mirror(ib_parity, ib_my_node) + kIBOffCnt;
                for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                    if (is_remote(i / kNumExpertsPerRank))
                        reinterpret_cast<uint32_t*>(stg)[i] = static_cast<uint32_t>(*workspace.get_expert_send_count_ptr(i));
                }
                __threadfence();
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
                if (thread_idx + 1 < ib_num_nodes) {
                    const uint32_t nr = thread_idx, n = ib_node_of_slot(nr);
                    const auto gin = kIBCimDispatch ? make_gin_site(1, ctx_meta(nr, nr), nr, ib_xbase)
                                                    : make_gin_site(2, ctx_mhdr(nr, hdr_orig(nr)), hdr_orig(nr), ib_xbase);
                    const uint64_t bytes = static_cast<uint64_t>(ib_lsa) * kNumExpertsPerRank * 4;
                    const auto src = stg + static_cast<int64_t>(n) * static_cast<int64_t>(bytes);
                    gin.put(stg, src, bytes, ib_cp(sym_buffer.rank_idx, n));
                    if (kIBCimDispatch)  // doorbell on the meta context: the (identical) header value pushes the counts through
                        gin.put_value(reinterpret_cast<uint64_t*>(ib_mirror(ib_parity, ib_my_node) + kIBOffHdr), (ib_epoch << 32) | num_tokens,
                                      ib_cp(sym_buffer.rank_idx, n));
                    else
                        gin.put(stg, src, bytes, ib_cp(sym_buffer.rank_idx, n));  // doorbell: identical bytes push the first through
                }
                IB_TRACE(32, thread_idx == 0);
            }
#endif
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Barrier before pulling
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        // NVLink part only: proxied sources publish their entries through the count-arrival protocol (release)
        comm::hybrid_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            [&]() { return make_gin(0); }, ib_epoch, false, true, false);
        IB_TRACE(2, sm_idx == 0 and thread_idx == 0);
#else
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );
        IB_TRACE(2, sm_idx == 0 and thread_idx == 0);
#endif

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        if (kIBWmark and ib_sst) {
            // WMARK poller: SM 0's dispatch warp nr watches the mirror of our counterpart on remote node slot nr
            if (sm_idx == 0 and warp_idx < ib_R and lane_idx == 0) {
                const uint32_t nr = warp_idx, n = ib_node_of_slot(nr);
                const auto mirp = ib_mirror(ib_parity, n);
                const auto hw = reinterpret_cast<const uint64_t*>(mirp + kIBOffSTab - static_cast<int64_t>(kNumExpertsPerRank * ib_lsa) * 4 - 8);
                auto* const wm = reinterpret_cast<volatile uint64_t*>(&workspace.signals->ib_o_landed[n][0]);
                const uint64_t ep = ib_epoch & 0xffffffffull;
                uint64_t h = ptx::ld_volatile(hw);
                IB_POLL((h >> 32) != ep, h = ptx::ld_volatile(hw), "wm_hdr", nr, 0);
                const uint32_t nt = static_cast<uint32_t>(h & 0x7fffffffull);
                if (not ((h >> 31) & 1ull)) {
                    *wm = (ep << 32) | 0x7fffffffull;  // token-order source: no gating
                } else {
                    uint32_t p = 0;
                    constexpr uint32_t kWB = 16;  // row tails loaded per round
                    while (p < nt) {
                        uint4 v[kWB];
                        #pragma unroll
                        for (uint32_t b = 0; b < kWB; ++ b)
                            v[b] = ib_ldv4(mirp + static_cast<int64_t>(cute::min(p + b, nt - 1) + 1) * kIBTokBytes - 16);
                        uint32_t adv = 0;
                        #pragma unroll
                        for (uint32_t b = 0; b < kWB; ++ b)
                            if (adv == b and p + b < nt and not ib_is_sentinel4(v[b])) adv = b + 1;
                        if (adv) { p += adv; *wm = (ep << 32) | p; }
                        else __nanosleep(100);
                    }
                    *wm = (ep << 32) | 0x7fffffffull;
                }
            }
            __syncwarp();
        }
#endif

        // Pull token data and SF from remote ranks into local L1 buffer
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = &shared_storage.dispatch_barriers[warp_idx];

        // Per-rank counts for current expert (re-loaded when expert changes)
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        // Wait token data arrival
        IB_TRACE_MAX(12, thread_idx == 0);  // dispatch warps reach the count wait, slowest SM
        scheduler.fetch_expert_recv_count();
        IB_TRACE_MAX(13, thread_idx == 0);  // all counts seen (before the IB system fence), slowest SM
#if not (defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB)
        IB_TRACE_MAX(28, thread_idx == 0);  // NVLink build (DG_MEGA_MOE_NVL_TRACE): counts, slowest SM
#endif
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        // acquire: pairs with the D1 fence before its count add (entries, per-source counts). D1 runs on THIS GPU's SMs,
        // so GPU scope suffices; DG_MEGA_MOE_IB_GPU_ACQ=1 uses it (the system fence costs ~2.5 us on the slowest SM)
#if defined(DG_MEGA_MOE_IB_GPU_ACQ) and DG_MEGA_MOE_IB_GPU_ACQ
        __threadfence();
#else
        __threadfence_system();
#endif
        IB_TRACE(3, sm_idx == 0 and thread_idx == 0);
        IB_TRACE_MAX(28, thread_idx == 0);  // counts, slowest SM
#endif

#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        constexpr bool kSkipLastPullWarp = kIBCombineWarp;
        constexpr bool kFlagWarp = kIBFlagGate;
#else
        constexpr bool kSkipLastPullWarp = false;
        constexpr bool kFlagWarp = false;
#endif
        constexpr uint32_t kNumPullWarps = kSkipLastPullWarp ? kNumDispatchWarps - 1 : kNumDispatchWarps;
        // The flag warp (last SM, last dispatch warp) has the highest global pull index, so excluding it keeps the others
        const bool is_flag_warp = kFlagWarp and sm_idx == kNumSMs - 1 and warp_idx == kNumDispatchWarps - 1;
        constexpr uint32_t kNumGlobalWarpsAll = kNumSMs * kNumPullWarps - (kFlagWarp and not kSkipLastPullWarp ? 1u : 0u);
        // DG_MEGA_MOE_IB_PULL_SKIP_SCAN=1: the SMs running the D1 scan (the last kNumRanks - lsa) take no pull work. Their
        // dispatch warps only start pulling after their epilogue finishes D1 (the barrier above), so with them in the
        // rotation every block waited for the slowest scan (GB200 MegaSeek trace: pulls saw counts ~35 us after publish)
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB and defined(DG_MEGA_MOE_IB_PULL_SKIP_SCAN) and DG_MEGA_MOE_IB_PULL_SKIP_SCAN
        DG_STATIC_ASSERT(not kFlagWarp, "PULL_SKIP_SCAN excludes the flag gate (its warp is on a scan SM)");
        // The FRONT meta SM (SM 0) is excluded too: its epilogue stages the meta and, with SENDER_TABLES, builds the tables
        // before reaching the same barrier (B300 int15: +16 us at 128 with SENDER_TABLES when SM 0 still pulled)
        const uint32_t ib_pull_sm0 = kIBFront ? 1u : 0u;
        // SST: only the proxy's R counterpart-source SMs (the last R) still scan (rule A + return order, off the count path)
        // DG_MEGA_MOE_IB_SST_PULL_SMS=1 (mechanism arm): SST keeps the legacy rotation (all kNumRanks - lsa former scan SMs stay
        // out), DG_MEGA_MOE_IB_SST_PULL_SMS=N: the last N SMs stay out
#ifndef DG_MEGA_MOE_IB_SST_PULL_SMS
#define DG_MEGA_MOE_IB_SST_PULL_SMS 0
#endif
        const uint32_t ib_pull_sms = ib_sst and not (kCGDrop & 1024) ? kNumSMs - (DG_MEGA_MOE_IB_SST_PULL_SMS == 0 ? ib_R : DG_MEGA_MOE_IB_SST_PULL_SMS == 1 ? kNumRanks - sym_buffer.num_lsa_ranks : static_cast<uint32_t>(DG_MEGA_MOE_IB_SST_PULL_SMS))
                                            : kNumSMs - (kNumRanks - sym_buffer.num_lsa_ranks) * ib_P;
#else
        const uint32_t ib_pull_sm0 = 0;
        const uint32_t ib_pull_sms = kNumSMs;
#endif
        const uint32_t kNumGlobalWarps = ib_pull_sms == kNumSMs and ib_pull_sm0 == 0 ? kNumGlobalWarpsAll : (ib_pull_sms - ib_pull_sm0) * kNumPullWarps;
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        uint64_t ib_order_seen = 0;  // remote sources whose order tables our D1 has published (bit k)
        // DG_MEGA_MOE_IB_SST_LEAN=1: nothing of SST inside the pull row loop. Every pulling warp waits ONCE, before its
        // first row, until the slot tables of all remote sources are ready (plain spin, no printf / trap path), and the
        // rule-A weight re-poll is a two-instruction spin. (The per-row table-ready test and the IB_POLL bodies inside the
        // row loop cost +12 us at EP16 1024 even when they never execute: start c16e.)
#if defined(DG_MEGA_MOE_IB_SST_LEAN) and DG_MEGA_MOE_IB_SST_LEAN
        constexpr bool kIBSSTLean = true;
        if (ib_sst and sm_idx >= ib_pull_sm0 and sm_idx < ib_pull_sms) {
            if (lane_idx == 0) {
                for (uint32_t k_t = 0; k_t < kNumRanks - sym_buffer.num_lsa_ranks; ++ k_t)
                    while (ptx::ld_acq_sys(&workspace.signals->ib_o_tab_ready[k_t]) != ib_epoch);
            }
            __syncwarp();
        }
#else
        constexpr bool kIBSSTLean = false;
#endif
#endif
        for (uint32_t token_idx = (sm_idx - ib_pull_sm0) * kNumPullWarps + warp_idx;
             sm_idx >= ib_pull_sm0 and sm_idx < ib_pull_sms and not (kSkipLastPullWarp and warp_idx >= kNumPullWarps) and not is_flag_warp; token_idx += kNumGlobalWarps) {
            // Advance expert until within the range
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;

                // Update pool block offset for the new expert
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);

                // Move start and end to the next expert
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }

            // Finish all tokens
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            // Load per-rank counts when expert changes
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    // TODO: this is not coalesced
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // Round-robin rank selection via iterative min-peeling
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB and defined(DG_MEGA_MOE_IB_LOCAL_FIRST) and DG_MEGA_MOE_IB_LOCAL_FIRST
            // LOCAL_FIRST: the expert's node-local sources take the first slots (round-robin among them), then the
            // remote sources (round-robin among them, so each block still takes every remote source's earliest rows).
            // Block 0 then needs only local rows plus the front of the remote range. Only the pull uses this mapping
            // (the combine reads each pool row's TokenSrcMetadata).
            {
                uint32_t num_local_in_lane = 0;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    num_local_in_lane += (j < kNumRanks and not is_remote(j)) ? stored_rank_count[i] : 0u;
                }
                const uint32_t num_local = __reduce_add_sync(0xffffffff, num_local_in_lane);
                const bool in_local = slot_idx < num_local;
                if (not in_local)
                    slot_idx -= num_local;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    remaining[i] = (j < kNumRanks and is_remote(j) != in_local) ? stored_rank_count[i] : 0u;
                }
            }
#endif
            while (true) {
                // Compute active count and min across all ranks
                // NOTES: reduce within each lane first, then warp-reduce once
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                // Hit in the current round
                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                // Move into the next round
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            if (kIBEarlyPublish or (ib_sst and not kIBSSTLean and not (kCGDrop & 8))) {
                // SPLIT_SCAN: counts went out before the slot table of this remote source was written
                if (is_remote(current_rank_in_expert_idx)) {
                    const uint32_t k_t = ib_k_of_rank(current_rank_in_expert_idx);
                    if (not ((ib_order_seen >> k_t) & 1ull)) {
                        IB_POLL(ptx::ld_acq_sys(&workspace.signals->ib_o_tab_ready[k_t]) != ib_epoch,
                                (void) 0, "tab_ready", k_t, 0);
                        ib_order_seen |= 1ull << k_t;
                    }
                }
            }
#endif
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            if constexpr (kIBScanSlices > 1) {
                if (ib_P > 1 and is_remote(current_rank_in_expert_idx)) {
                    const uint32_t k_t = ib_k_of_rank(current_rank_in_expert_idx);
                    if (not ((ib_order_seen >> k_t) & 1ull)) {
                        if (lane_idx < ib_P)
                            IB_POLL(ptx::ld_acq_sys(ib_sl_tflag + k_t * 4 + lane_idx) != ib_epoch, (void) 0, "sl_tflag", k_t, lane_idx);
                        __syncwarp();
                        ib_order_seen |= 1ull << k_t;
                    }
                }
            }
#endif
            // Read source token-topk index (written by remote dispatch via NVLink)
            const uint32_t src_token_topk_raw = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            // SST + WAVE_SORT: remote entries carry their mirror row in the high half
            const bool ib_packed = kIBPacked and ib_sst and is_remote(current_rank_in_expert_idx);
            const uint32_t src_token_topk_idx = ib_packed ? (src_token_topk_raw & 0xffffu) : src_token_topk_raw;
#else
            const uint32_t src_token_topk_idx = src_token_topk_raw;
#endif
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

            // Hidden bytes are divided into chunks
            constexpr uint32_t kNumChunks = kHidden / kNumBytesPerPull;
            DG_STATIC_ASSERT(kNumChunks * kNumBytesPerPull == kHidden, "kNumBytesPerPull must divide hidden");

            // TMA load token from remote rank and store into local
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;

            // Wait for ring buffer slot to be available (previous consumer must have finished all N blocks)
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
            if (l1_empty_count_target > 0) {
                const auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB and defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
                uint64_t rw0_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(rw0_));
                while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
                uint64_t rw1_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(rw1_));
                // slot 29: max ring-slot wait in this launch (epoch-tagged so older launches lose the max); slot 30: last end
                if (lane_idx == 0) {
                    atomicMax(reinterpret_cast<unsigned long long*>(workspace.signals->ib_trace + 29),
                              static_cast<unsigned long long>((ib_epoch << 40) | ((rw1_ - rw0_) & ((1ull << 40) - 1))));
                    atomicMax(reinterpret_cast<unsigned long long*>(workspace.signals->ib_trace + 30), static_cast<unsigned long long>(rw1_));
                }
#else
                while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
#endif
            }

            const auto dst_base_ptr = buffer.l1_token_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).get_base_ptr();
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            // D2: a source on another node is pulled from its counterpart's mirror over NVLink; the load is the poll
            const bool ib_rem = is_remote(current_rank_in_expert_idx);
            const uint32_t ib_proxy = ib_rem ? ib_cp(current_rank_in_expert_idx, ib_my_node) : current_rank_in_expert_idx;
            const auto ib_mir = ib_mirror(ib_parity, current_rank_in_expert_idx / ib_lsa);
            // WAVE_SORT: the source's x rows sit in the mirror in wave order (slot table built by our D1 scan)
            if (kIBWaveSort and ib_rem and not (kIBSSTWave and ib_sst)) {
                const uint32_t k_r = ib_k_of_rank(current_rank_in_expert_idx);
                if (not ((ib_order_seen >> k_r) & 1ull)) {
                    IB_POLL(ptx::ld_acq_sys(&workspace.signals->ib_o_order_ready[k_r]) != ib_epoch, (void) 0, "order_ready", k_r, 0);
                    ib_order_seen |= 1ull << k_r;
                }
            }
            const uint32_t ib_xslot = ib_packed ? (src_token_topk_raw >> 16) : kIBWaveSort and ib_rem ?
                ib_tab[kIBTabSlot + ib_k_of_rank(current_rank_in_expert_idx) * kIBTabT + src_token_idx] : src_token_idx;
            if (kIBWmark and ib_sst) {
                if (ib_rem and lane_idx == 0) {
                    const auto wp = sym_buffer.map(reinterpret_cast<const uint64_t*>(&workspace.signals->ib_o_landed[current_rank_in_expert_idx / ib_lsa][0]), ib_proxy);
                    uint64_t wv = ptx::ld_volatile(wp);
                    IB_POLL((wv >> 32) != (ib_epoch & 0xffffffffull) or static_cast<uint32_t>(wv) <= ib_xslot,
                            (__nanosleep(200), wv = ptx::ld_volatile(wp)), "wmark", current_rank_in_expert_idx, ib_xslot);
                }
                __syncwarp();
            }
            const auto src_base_ptr = ib_rem ?
                sym_buffer.map(ib_mir + static_cast<int64_t>(ib_xslot) * kIBTokBytes, ib_proxy) :
                sym_buffer.map(reinterpret_cast<uint8_t*>(buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr()), current_rank_in_expert_idx);
            // DG_MEGA_MOE_IB_PULL_WHOLE=1 (bench agent: pulled 41.0 -> 36.1 us @128, 145 -> 123 @1024 on B300): issue every
            // 16-byte x load of the row plus its SF loads before polling any of them (one round of NVLink latency per row)
#if not defined(DG_MEGA_MOE_IB_PULL_WHOLE) or DG_MEGA_MOE_IB_PULL_WHOLE  // default ON (B300: -3 us @128, -7 @1024, DeepEP-checked)
            constexpr bool kIBPullWhole = true;
#else
            constexpr bool kIBPullWhole = false;
#endif
            constexpr uint32_t kIBSFIters = (kHidden / 128 + 31) / 32;
            uint32_t ib_sf_pre[kIBSFIters];
            if (kIBPullWhole and ib_rem) {
                const auto sf_src = sym_buffer.map(reinterpret_cast<uint32_t*>(ib_mir + kIBOffSF + static_cast<int64_t>(src_token_idx) * kIBSFBytes), ib_proxy);
                #pragma unroll
                for (uint32_t i = 0; i < kIBSFIters; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    ib_sf_pre[i] = j < kHidden / 128 ? ptx::ld_volatile(sf_src + j) : 0u;
                }
            }
            // DG_MEGA_MOE_PULL_META_EARLY=1: the row's top-k weight and (for NVLink-local rows) its SF are loaded here, before
            // the row data, so their NVLink round trips overlap the data copy (pull phase trace on B300: weight + SF ~4.2 us
            // of ~9.5 us per row, issued serially after the data). Remote weights were polled by the proxy before the counts.
#if defined(DG_MEGA_MOE_PULL_META_EARLY) and DG_MEGA_MOE_PULL_META_EARLY
            constexpr bool kMetaEarly = true;
#else
            constexpr bool kMetaEarly = false;
#endif
            float ib_w_pre = 0.f;
            if constexpr (kMetaEarly) {
                ib_w_pre = ib_rem ?
                    *sym_buffer.map(reinterpret_cast<const float*>(ib_mir + kIBOffW) + src_token_topk_idx, ib_proxy) :
                    *sym_buffer.map(buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx, current_rank_in_expert_idx);
                if constexpr (kIBSSTLean) {
                    if (ib_sst and ib_rem) {
                        const auto wp = sym_buffer.map(reinterpret_cast<const uint32_t*>(ib_mir + kIBOffW) + src_token_topk_idx, ib_proxy);
                        uint32_t wb = __float_as_uint(ib_w_pre);
                        while (wb == 0xffffffffu) wb = ptx::ld_volatile(wp);
                        ib_w_pre = __uint_as_float(wb);
                    }
                }
                if (kIBSenderTables or (ib_sst and not kIBSSTLean and not (kCGDrop & 8))) {  // rule A: no scan polled this weight
                    if (ib_rem) {
                        const auto wp = sym_buffer.map(reinterpret_cast<const uint32_t*>(ib_mir + kIBOffW) + src_token_topk_idx, ib_proxy);
                        uint32_t wb = __float_as_uint(ib_w_pre);
                        IB_POLL(wb == 0xffffffffu, wb = ptx::ld_volatile(wp), "pull_w", current_rank_in_expert_idx, src_token_topk_idx);
                        ib_w_pre = __uint_as_float(wb);
                    }
                }
                if (not ib_rem) {
                    const auto sf_src = sym_buffer.map(buffer.input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(), current_rank_in_expert_idx);
                    #pragma unroll
                    for (uint32_t i = 0; i < kIBSFIters; ++ i) {
                        const uint32_t j = i * 32 + lane_idx;
                        ib_sf_pre[i] = j < kHidden / 128 ? sf_src[j] : 0u;
                    }
                }
            }
            // DG_MEGA_MOE_IB_LOCAL_REGPULL=1 (Flash 128 latency): node-local rows take the register path too (all 16-byte
            // loads of the row in flight, posted stores) instead of TMA load -> mbarrier wait -> TMA store -> store wait
#if defined(DG_MEGA_MOE_IB_LOCAL_REGPULL) and DG_MEGA_MOE_IB_LOCAL_REGPULL
            constexpr bool kIBLocalReg = kIBLeanMeta;
#else
            constexpr bool kIBLocalReg = false;
#endif
            const bool ib_reg = ib_rem or kIBLocalReg;
            if (ib_reg) {
                // Every 16-byte unit is re-read until it is not the 0xFF sentinel, then stored into the L1 ring
                DG_STATIC_ASSERT(kIBTokBytes % (32 * 16) == 0, "IB row poll");
                constexpr uint32_t kUnits = kIBTokBytes / (32 * 16), kBatch = kIBPullWhole ? kUnits : 5;
                if (kIBFlagGate and ib_rem) {
                    // Probe first: one load of the row's first and last 16-byte units per lane; only a row that has not
                    // fully landed waits on the (NVLink) chunk flag, so landed rows pay nothing extra
                    const bool landed = __all_sync(0xffffffffu,
                        not ib_is_sentinel4(ib_ldv4(src_base_ptr + lane_idx * 16)) and
                        not ib_is_sentinel4(ib_ldv4(src_base_ptr + kIBTokBytes - 512 + lane_idx * 16)));
                    if (not landed and lane_idx == 0) {
                        const auto flag = sym_buffer.map(&workspace.signals->ib_o_landed[current_rank_in_expert_idx / ib_lsa][ib_xslot / kIBChunk], ib_proxy);
                        IB_POLL(ptx::ld_acq_sys(flag) != ib_epoch, (void) 0, "pull_flag", current_rank_in_expert_idx, ib_xslot);
                    }
                    __syncwarp();
                }
#if defined(DG_MEGA_MOE_IB_PULL_GATE) and DG_MEGA_MOE_IB_PULL_GATE
                // Gate: one lane polls the row's last 16 bytes with a pause before the warp reads the row (hundreds of
                // warps spinning on the proxy's HBM while its NIC writes there slows the NIC down, ms-scale on GB200)
                if (lane_idx == 0) {
                    const auto last = src_base_ptr + kIBTokBytes - 16;
                    IB_POLL(ib_is_sentinel4(ib_ldv4(last)), __nanosleep(DG_MEGA_MOE_IB_PULL_GATE), "pull_gate", current_rank_in_expert_idx, src_token_idx);
                }
                __syncwarp();
#endif
                #pragma unroll
                for (uint32_t u0 = 0; u0 < kUnits; u0 += kBatch) {
                    uint4 v[kBatch];
                    #pragma unroll
                    for (uint32_t b = 0; b < kBatch; ++ b)
                        if (u0 + b < kUnits) v[b] = ib_ldv4(src_base_ptr + ((u0 + b) * 32 + lane_idx) * 16);
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
                    // slot 10: remote rows whose FIRST load batch still saw a sentinel (row not fully landed when pulled);
                    // slot 11: remote rows pulled (both reset by SM 0 at kernel start; the pull follows a grid barrier)
                    if (u0 == 0 and ib_rem) {
                        bool miss = false;
                        #pragma unroll
                        for (uint32_t b = 0; b < kBatch; ++ b)
                            if (u0 + b < kUnits) miss |= ib_is_sentinel4(v[b]);
                        const bool any_miss = __any_sync(0xffffffffu, miss);
                        if (lane_idx == 0) {
                            atomicAdd(reinterpret_cast<unsigned long long*>(workspace.signals->ib_trace + 11), 1ull);
                            if (any_miss) atomicAdd(reinterpret_cast<unsigned long long*>(workspace.signals->ib_trace + 10), 1ull);
                        }
                    }
#endif
                    #pragma unroll
                    for (uint32_t b = 0; b < kBatch; ++ b) {
                        if (u0 + b < kUnits) {
                            const uint32_t o = ((u0 + b) * 32 + lane_idx) * 16;
                            IB_POLL(ib_rem and ib_is_sentinel4(v[b]), v[b] = ib_ldv4(src_base_ptr + o), "pull_x", current_rank_in_expert_idx, src_token_idx);
                            *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(dst_base_ptr) + o) = v[b];
                        }
                    }
                }
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
                if (lane_idx == 0 and ib_rem) {
                    uint64_t t_;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_));
                    atomicMin(reinterpret_cast<unsigned long long*>(&workspace.signals->ib_trace[41]), static_cast<unsigned long long>(t_));
                    atomicMax(reinterpret_cast<unsigned long long*>(&workspace.signals->ib_trace[42]), static_cast<unsigned long long>(t_));
                    if (current_expert_idx == 0)
                        atomicMax(reinterpret_cast<unsigned long long*>(&workspace.signals->ib_trace[43]), static_cast<unsigned long long>(t_));
                }
#endif
                asm volatile("fence.proxy.async.global;" ::: "memory");  // generic stores, read by the GEMM's TMA
                IB_TRACE_MAX(31, lane_idx == 0 and ib_rem);  // last remote row fully landed + copied, slowest warp
            }
#else
            constexpr bool ib_rem = false;
            constexpr bool ib_reg = false;
            const auto src_base_ptr = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
#endif
            const auto issue_and_wait_pull_store = [&](const uint32_t& i) {
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                ptx::tma_store_1d(
                    math::advance_ptr(dst_base_ptr, i * kNumBytesPerPull),
                    pull_buffer.get_base_ptr(), kNumBytesPerPull
                );
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
            };
            if (not ib_reg and cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumChunks; ++ i) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        math::advance_ptr(src_base_ptr, i * kNumBytesPerPull),
                        pull_mbarrier, kNumBytesPerPull
                    );
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                    i != (kNumChunks - 1) ? issue_and_wait_pull_store(i) : void();
                }
            }
            __syncwarp();

            // Load the weight first, so that its remote latency overlaps with the SF copy below
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            // (proxied weights were polled by the proxy before it published the counts)
            static_assert(not (kIBSenderTables or kIBSST) or kMetaEarly, "SENDER_TABLES / SST need PULL_META_EARLY (the pull polls the weight there)");
            const auto weight = kMetaEarly ? ib_w_pre : ib_rem ?
                *sym_buffer.map(reinterpret_cast<const float*>(ib_mir + kIBOffW) + src_token_topk_idx, ib_proxy) :
                *sym_buffer.map(buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx, current_rank_in_expert_idx);
#else
            const auto weight = *sym_buffer.map(
                buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                current_rank_in_expert_idx);
#endif

            // Load and store SF (overlaps with last chunk's TMA load from remote)
            constexpr uint32_t kNumSFUint32 = kHidden / 128;
            DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % 128 == 0, "Invalid SF");
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            const auto remote_sf_ptr = ib_rem ?
                sym_buffer.map(reinterpret_cast<uint32_t*>(ib_mir + kIBOffSF + static_cast<int64_t>(src_token_idx) * kIBSFBytes), ib_proxy) :
                sym_buffer.map(buffer.input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(), current_rank_in_expert_idx);
#else
            const auto remote_sf_ptr = sym_buffer.map(
                buffer.input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                current_rank_in_expert_idx);
#endif
            const auto local_sf_ptr = buffer.l1_sf_buffer.get_base_ptr<uint32_t>();
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t token_idx_in_block = token_idx_in_expert % BLOCK_M;
            const auto sf_ring_token_idx = ring_block_idx * SF_BLOCK_M +
                transform_sf_token_idx(token_idx_in_block);
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
                if (j < kNumSFUint32) {
                    uint32_t sf = ib_rem ? (kIBPullWhole ? ib_sf_pre[i] : ptx::ld_volatile(remote_sf_ptr + j)) : (kMetaEarly ? ib_sf_pre[i] : remote_sf_ptr[j]);
                    IB_POLL(ib_rem and sf == 0xffffffffu, sf = ptx::ld_volatile(remote_sf_ptr + j), "pull_sf", current_rank_in_expert_idx, src_token_idx);
                    local_sf_ptr[j * kNumSFRingTokens + sf_ring_token_idx] = sf;
                }
#else
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumSFRingTokens + sf_ring_token_idx] = remote_sf_ptr[j];
#endif
            }
            __syncwarp();

            // Store weights and metadata
            if (cute::elect_one_sync()) {
                *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = weight;

                // Write source metadata for combine write-back (logical pool token)
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                // Complete last chunk's store
                if (not ib_reg)
                    issue_and_wait_pull_store(kNumChunks - 1);
                const bool is_last_token = (token_idx == expert_end_idx - 1);
                IB_HIST(ib_rem ? 2 : 3, lane_idx == 0);
                ptx::red_add_rel(
                    workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks), 
                    is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u
                );
            }
            __syncwarp();
        }

#if not (defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB)
        IB_TRACE_MAX(27, thread_idx == 0);  // NVLink build: pull end, slowest SM
#endif
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        if (is_flag_warp) {
            // Lane l owns chunks l and l + 32 of every remote node; a chunk is published when its last row's last 16
            // bytes landed (the last row is known once the mirror header landed; before that only full chunks qualify)
            DG_STATIC_ASSERT(layout::MegaMoESignals::kIBOMaxChunks == 64, "IB flag gate chunks");
            uint32_t done = 0;  // bit nr * 2 + j
            const uint32_t num_remote_nodes = ib_num_nodes - 1;
            const uint32_t all_bits = num_remote_nodes >= 16 ? 0xffffffffu : (1u << (2 * num_remote_nodes)) - 1u;
            IB_POLL(__any_sync(0xffffffffu, done != all_bits), ([&]() {
                for (uint32_t nr = 0; nr < num_remote_nodes; ++ nr) {
                    const uint32_t n = ib_node_of_slot(nr);
                    const auto mir = ib_mirror(ib_parity, n);
                    const uint64_t h = ptx::ld_volatile(reinterpret_cast<const uint64_t*>(mir + kIBOffHdr));
                    const bool known = (h >> 32) == (ib_epoch & 0xffffffffull);
                    const uint32_t nt = static_cast<uint32_t>(h & 0xffffffffull);
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        const uint32_t bit = 1u << (nr * 2 + j), c = lane_idx + 32 * j;
                        if (done & bit) continue;
                        if (known and c * kIBChunk >= nt) { done |= bit; continue; }
                        const uint32_t last_row = known ? cute::min(c * kIBChunk + kIBChunk, nt) - 1 : c * kIBChunk + kIBChunk - 1;
                        if (last_row >= kIBT) { if (known) done |= bit; continue; }
                        if (not ib_is_sentinel4(ib_ldv4(mir + static_cast<int64_t>(last_row) * kIBTokBytes + kIBTokBytes - 16))) {
                            ptx::st_rel_sys(&workspace.signals->ib_o_landed[n][c], ib_epoch);
                            done |= bit;
                        }
                    }
                }
            })(), "flag_pub", done, all_bits);
        }
        IB_TRACE(4, sm_idx == 0 and thread_idx == 0);
        IB_TRACE_MAX(27, thread_idx == 0);  // pull end, slowest SM
        if constexpr (kIBReduceOnArrival) {
            // The dispatch warps are idle until the cleanup: reduce forwarded tokens as their partials land
            ib_forward_combine(warp_idx * kNumSMs + sm_idx, kNumSMs * kNumDispatchWarps, sm_idx == 0 and thread_idx == 0,
                               cute::Int<DG_MEGA_MOE_IB_FWD_BATCH_D>{}, false);
            IB_TRACE(6, sm_idx == 0 and thread_idx == 0);
            TT_MAX(6, lane_idx == 0);
        }
#endif
        // Clean workspace for the next usage, and also do cumulative stats
        // NOTES: it is overlapped with combine reduction epilogue
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        TT_MAX(11, thread_idx == 0);
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        const auto ib_rearm_mirrors = [&]() {
        // E: re-arm the mirrors of this parity. Every node-local pull from them is done (barrier above); the
        // forward-combine polled every row (fdone), so every byte the NIC wrote this launch has landed (rule A)
        for (uint32_t nr = 0; nr + 1 < ib_num_nodes; ++ nr) {
            const uint32_t n = ib_node_of_slot(nr);
            const uint32_t nt = ptx::ld_volatile(&workspace.signals->ib_o_nt[n]);
            const uint32_t nchunks = (nt + kIBChunk - 1) / kIBChunk;
            if (thread_idx == 0)
                IB_POLL(ptx::ld_acq(&workspace.signals->ib_o_fdone[ib_parity][n]) < nchunks * (kIBRetGroups > 0 ? kIBRetGroups : 1u), (void) 0, "e_fdone", n, nchunks);
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            const auto mir = ib_mirror(ib_parity, n);
            const uint32_t gtid = sm_idx * kNumDispatchThreads + thread_idx;
            constexpr uint32_t kGThreads = kNumSMs * kNumDispatchThreads;
            const uint4 ones = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
            const uint4 idx_sentinel = {0u, 0x80000000u, 0u, 0x80000000u};
            if (kIBOracle) continue;
            for (int64_t u = gtid; u < static_cast<int64_t>(nt) * kIBTokBytes / 16; u += kGThreads)
                reinterpret_cast<uint4*>(mir)[u] = ones;
            for (int64_t u = gtid; u < static_cast<int64_t>(nt) * kIBSFBytes / 16; u += kGThreads)
                reinterpret_cast<uint4*>(mir + kIBOffSF)[u] = ones;
            const int64_t meta_rows = static_cast<int64_t>(nt);
            for (int64_t u = gtid; u < meta_rows * kNumTopk * 8 / 16; u += kGThreads)
                reinterpret_cast<uint4*>(mir + kIBOffIdx)[u] = idx_sentinel;
            for (int64_t u = gtid; u < meta_rows * kNumTopk; u += kGThreads)
                reinterpret_cast<uint32_t*>(mir + kIBOffW)[u] = 0xffffffffu;
            if (gtid == 0)
                *reinterpret_cast<uint64_t*>(mir + kIBOffHdr) = ~0ull;
            if (kIBSenderTables or kIBSST) {
                for (int64_t u = gtid; u < static_cast<int64_t>(nt) * kNumTopk; u += kGThreads)
                    reinterpret_cast<uint32_t*>(mir + kIBOffSTab)[u] = 0xffffffffu;
            }
            if (kIBCountsEarly or kIBCountsInMeta or kIBSST) {
                for (int64_t u = gtid; u < kIBCntBytes / 4; u += kGThreads)
                    reinterpret_cast<uint32_t*>(mir + kIBOffCnt)[u] = 0xffffffffu;
            }
        }
        };
        if constexpr (kIBEarlyRearm) {
            // EARLY_REARM: every node-local rank (ourselves included) has passed its L2 grid sync, so its pulls from our mirrors
            // are done; the forward-combine's fdone wait inside the re-arm covers rule A
            const uint64_t er_grid_idx = ptx::get_grid_idx() + 1;
            if (thread_idx < kNumRanks and not is_remote(thread_idx)) {
                const auto fp = workspace.get_combine_ready_grid_idx_ptr(thread_idx);
                IB_POLL(ptx::ld_acq_sys(fp) != er_grid_idx, (void) 0, "early_rearm_ready", thread_idx, 0);
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            ib_rearm_mirrors();
        }
#endif
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        if constexpr (kIBDeferRearm) {
            // One warp per token: re-arm the node-local slots' partial rows (the forward-combine read every one of them)
            const uint32_t gw = warp_idx * kNumSMs + sm_idx;
            constexpr uint32_t kGW = kNumSMs * kNumDispatchWarps;
            constexpr uint32_t kRowUnits = kHidden * 2 / 16;
            const uint4 ones = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
            for (uint32_t nr = 0; nr + 1 < ib_num_nodes; ++ nr) {
                const uint32_t n = ib_node_of_slot(nr);
                const uint32_t nt = ptx::ld_volatile(&workspace.signals->ib_o_nt[n]);
                const auto idx = reinterpret_cast<const uint64_t*>(ib_mirror(ib_parity, n) + kIBOffIdx);
                const auto pc = ib_pcomb(n);
                // Every forward-combine item of this node on the whole GPU must have read its rows first (other SMs may
                // still be reducing): all chunks done
                if (thread_idx == 0)
                    IB_POLL(ptx::ld_acq(&workspace.signals->ib_o_fdone[ib_parity][n]) < (nt + kIBChunk - 1) / kIBChunk, (void) 0, "rearm_fdone", n, nt);
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
                for (uint32_t t = gw; t < nt; t += kGW) {
                    const int e = lane_idx < kNumTopk ? static_cast<int>(static_cast<int64_t>(ptx::ld_volatile(idx + t * kNumTopk + lane_idx))) : -1;
                    uint32_t m = __ballot_sync(0xffffffffu, e >= 0 and static_cast<uint32_t>(e) / kNumExpertsPerRank / ib_lsa == ib_my_node);
                    while (m) {
                        const uint32_t k = __ffs(m) - 1;
                        m &= m - 1;
                        const auto row = reinterpret_cast<uint4*>(pc + (static_cast<int64_t>(k) * kIBT + t) * kHidden * 2);
                        for (uint32_t u = lane_idx; u < kRowUnits; u += 32)
                            row[u] = ones;
                    }
                }
            }
            TT_MAX(14, lane_idx == 0);  // deferred re-arm done
        }
#endif

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            // SM 0: clear expert send count and schedule task counters
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if (warp_idx == 0 and cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumDeviceLocalityDomains; ++ i) {
                    workspace.get_l1_task_count_ptr()[i] = workspace.get_l2_task_count_ptr()[i] = 0;
                    workspace.get_shared_l1_task_count_ptr()[i] = workspace.get_shared_l2_task_count_ptr()[i] = 0;
                }
            }
            __syncwarp();
            for (uint32_t i = thread_idx; i < workspace.num_shared_l2_pool_blocks; i += kNumDispatchThreads)
                *workspace.get_shared_l2_full_count_ptr(i) = 0;
            __syncwarp();
        } else {
            // Other SMs: clean blocks
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                // Read expert token count before clearing
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                // Compute expert pool block offset
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                // Wait read count ready
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean expert token count, and add cumulative results
                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                // Clean per-rank token count
                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                // Clean L1 and L2 full stuffs and ring buffer counts
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l1_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_full_mask_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                }
                __syncwarp();
            }
        }

        // Wait for all ranks to finish cleaning
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        // NVLink part only: cross-node reuse is ordered by parity + the mandatory headers (REDESIGN_OPUS.md 6)
        comm::hybrid_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            [&]() { return make_gin(0); }, ib_epoch, true, false, false);
        TT_MAX(12, thread_idx == 0);
        if (not kIBEarlyRearm)
            ib_rearm_mirrors();
        IB_TRACE(17, sm_idx == 0 and thread_idx == 0);
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
        if (sm_idx == 0 and thread_idx == 0)
            workspace.signals->ib_trace[18] = workspace.signals->ib_trace[17] - workspace.signals->ib_trace[16];
#endif
        if (sm_idx == 0 and thread_idx == 0)
            *workspace.get_ib_epoch_ptr() = ib_epoch;
#else
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* Before the NVLink barrier, there is a grid sync */ true,
            /* At the end of kernel does not need to sync */ false
        );
#endif
        IB_TRACE_MAX(33, thread_idx == 0);  // dispatch warps done, slowest SM (reported by the next launch)
    } else if (warp_idx == kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for tokens with SFA
        task_info_t task_info;
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        bool ib_noov_published = false, ib_noov_ready = false;
#endif
        while (scheduler.get_next_task(task_info)) {
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            if constexpr (kIBNoOvDispatch) {
                const bool routed = task_info.block_phase == sched::BlockPhase::Linear1 or task_info.block_phase == sched::BlockPhase::Linear2;
                if (routed and sm_idx == 0 and not ib_noov_published)
                    ib_noov_landed_check(), ib_noov_published = true;
                if (task_info.block_phase == sched::BlockPhase::Linear1 and not ib_noov_ready)
                    ib_noov_wait(kIBNoOvLandedSlot), ib_noov_ready = true;
            }
#endif
            const auto tensor_map_a_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts;
            const auto tensor_map_sfa_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts_sf :
                                            task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts_sf :
                                            task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts_sf :
                                          /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts_sf;
            const auto num_k_blocks = math::ceil_div(task_info.shape_k, BLOCK_K);

            // Compute pool block offset for this expert
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;

            // Wait the entire token arrival
            if (task_info.block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_full_count_ptr(block_idx);
                const auto num_expected_tokens = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(ptr) != num_expected_tokens);
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
                if (cute::elect_one_sync()) {
                    uint64_t t_;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_));
                    atomicMin(reinterpret_cast<unsigned long long*>(&workspace.signals->ib_trace[40]), static_cast<unsigned long long>(t_));
                    IB_HIST(0, true);
                }
                __syncwarp();
#endif
            } else if (task_info.block_phase == sched::BlockPhase::SharedLinear2) {
                const auto ptr = workspace.get_shared_l2_full_count_ptr(block_idx);
                const auto num_expected_blocks = (SHARED_L2_SHAPE_K / BLOCK_N) * 2;
                while (ptx::ld_acq(ptr) != num_expected_blocks);
            }

            L2KBlockDependency l2_k_block_dependency(workspace.get_l2_full_mask_ptr(ring_block_idx), pool_block_idx / kNumRingBlocks);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                if (task_info.block_phase == sched::BlockPhase::Linear2)
                    l2_k_block_dependency.wait(k_block_idx);
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
                if (task_info.block_phase == sched::BlockPhase::Linear2 and k_block_idx == 0 and cute::elect_one_sync())
                    IB_HIST(1, true);
                __syncwarp();
#endif

                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute token offsets from block index
                uint32_t m_idx = block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t sfa_m_idx = block_idx * SF_BLOCK_M;
                uint32_t sfa_k_idx = k_block_idx * (BLOCK_K / 128);

                // Add 2 CTA offsets for non-leader CTA
                if (not is_leader_cta)
                    m_idx += task_info.get_umma_aligned_valid_m() / 2;

                // TMA copy tokens and SFA, then arrive at full barrier
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_a[stage_idx], k_idx, m_idx, 2);
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        tensor_map_sfa_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);
                    if (is_leader_cta) {
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_a[0]) * 2 + sizeof(SharedStorage::smem_sfa[0]) * 2);
                    } else {
                        shared_storage.full_barriers[stage_idx].arrive(0u);
                    }
                }
                __syncwarp();
            }
        }
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        if constexpr (kIBNoOvDispatch) {
            if (sm_idx == 0 and not ib_noov_published)  // no routed task here: still publish, the node's L1 loaders wait for it
                ib_noov_landed_check();
        }
#endif
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for weights with SF
#if defined(DG_MEGA_MOE_L2_PREFETCH) and DG_MEGA_MOE_L2_PREFETCH
        // DG_MEGA_MOE_L2_PREFETCH=E: before the first task (the routed L1 cannot start until every source's counts are in),
        // prefetch the L1 weight tiles of local experts [0, E) into L2 (the routed GEMMs are weight-stream bound at small
        // batch, experts run in index order). Tiles are split across SMs by (expert, n block); TMA L2 prefetch, no smem.
        {
            constexpr uint32_t kPrefetchExperts = cute::min(static_cast<uint32_t>(DG_MEGA_MOE_L2_PREFETCH), kNumExpertsPerRank);
            constexpr uint32_t kNBlocks = L1_SHAPE_N / BLOCK_N, kKBlocks = L1_SHAPE_K / BLOCK_K;
            constexpr uint32_t kPack = get_smem_pack_factor<weight_dtype_t>();
            constexpr uint32_t kAtomLogical = tma::get_inner_block_atom_size<BLOCK_K, kSwizzleBMode, kPack, weight_dtype_t>() * kPack;
            DG_STATIC_ASSERT(BLOCK_K % kAtomLogical == 0, "L2 prefetch atom");
            if (cute::elect_one_sync()) {
                const uint32_t half_n = L1_SHAPE_N / 2;
                for (uint32_t p = sm_idx; p < kPrefetchExperts * kNBlocks; p += kNumSMs) {
                    const uint32_t e = p / kNBlocks, n_idx = (p % kNBlocks) * BLOCK_N;
                    const uint32_t half_idx = n_idx >= half_n, half_n_idx = n_idx - half_idx * half_n;
                    for (uint32_t kb = 0; kb < kKBlocks; ++ kb)
                        #pragma unroll
                        for (uint32_t a = 0; a < BLOCK_K / kAtomLogical; ++ a)
                            asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];"
                                         :: "l"(reinterpret_cast<uint64_t>(&tensor_map_l1_weights)),
                                            "r"(kb * BLOCK_K + a * kAtomLogical), "r"(half_n_idx), "r"(half_idx), "r"(e) : "memory");
                }
            }
            __syncwarp();
        }
#endif
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const auto tensor_map_b_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights;
            const auto tensor_map_sfb_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights_sf :
                                            task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf :
                                            task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights_sf :
                                          /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights_sf;

            const auto shape_k = task_info.shape_k;
            const auto shape_n = task_info.shape_n;
            const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);
            const auto n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            const auto num_k_blocks = math::ceil_div(shape_k, BLOCK_K);

            const uint32_t half_n = shape_n / 2, n_idx = n_block_idx * BLOCK_N, half_idx = n_idx >= half_n;
            const uint32_t half_n_idx = n_idx - half_idx * half_n, expert_idx = task_info.is_shared() ? 0 : task_info.local_expert_idx;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute weight offset
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                uint32_t sfb_k_idx = task_info.is_shared() ? k_block_idx * (BLOCK_K / 128) : task_info.local_expert_idx * shape_sfb_k + k_block_idx * (BLOCK_K / 128);

                // TMA copy weights with SF
                if (cute::elect_one_sync()) {
                    if (task_info.is_shared()) {
                        tma::copy_nd<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, shared_b_dtype_t, 4>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], reinterpret_cast<shared_b_dtype_t*>(shared_storage.smem_b[stage_idx]), 2, k_idx, half_n_idx, half_idx, expert_idx);
                        tma::copy<BLOCK_N, 1, 0>(
                            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                        if (is_leader_cta) {
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_b[0]) * 2 + sizeof(SharedStorage::smem_sfb[0]) * 2);
                        } else {
                            shared_storage.full_barriers[stage_idx].arrive(0u);
                        }
                    } else {
                        tma::copy_nd<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, weight_dtype_t, 4>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], 2, k_idx, half_n_idx, half_idx, expert_idx);
                        tma::copy<BLOCK_N, 1, 0>(
                            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                        if (is_leader_cta) {
                            constexpr uint32_t kNumWeightBytes = sizeof(SharedStorage::smem_b[0]) * 2 /
                                (kIsWeightFP8 ? 1 : 2);
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(kNumWeightBytes + sizeof(SharedStorage::smem_sfb[0]) * 2);
                        } else {
                            shared_storage.full_barriers[stage_idx].arrive(0u);
                        }
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM MMA issue warp (only the leader CTA will run)
        if (is_leader_cta) {
            // Make instruction descriptor with block scaling
            // NOTES: always swap A/B
            auto routed_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                    weight_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                    UMMA_M, UMMA_N,
                    cute::UMMA::Major::K, cute::UMMA::Major::K
                >();
            auto shared_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                shared_b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);

            DG_STATIC_ASSERT(sizeof(weight_dtype_t) == sizeof(shared_b_dtype_t), "Weight SMEM descriptors must use identical addressing");
            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, UMMA_BLOCK_K, kSwizzleAMode>(shared_storage.smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, UMMA_BLOCK_K, kSwizzleBMode>(shared_storage.smem_b[0], 0, 0);
            const uint32_t a_desc_lo = a_desc.lo;
            const uint32_t b_desc_lo = b_desc.lo;

            // Checks for MMA instructions
            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            // Persistently schedule over blocks
            uint32_t current_iter_idx = 0;
            task_info_t task_info;
            while (scheduler.get_next_task(task_info)) {
                const auto num_k_blocks = task_info.shape_k / BLOCK_K;

                // Dynamic update of UMMA N based on effective M
                auto& instr_desc = task_info.is_shared() ? shared_instr_desc : routed_instr_desc;
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, task_info.get_umma_aligned_valid_m());

                // Wait tensor memory empty barrier arrival
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[accum_stage_idx].wait(accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                // Empty barrier arrival
                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.empty_barriers[stage_idx]));

                    // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                // Launch MMAs
                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait TMA load completion
                    shared_storage.full_barriers[stage_idx].wait(phase);
                    ptx::tcgen05_after_thread_sync();

                    const uint32_t a_desc_base_lo = a_desc_lo + stage_idx * sizeof(SharedStorage::smem_a[0]) / 16;
                    const uint32_t b_desc_base_lo = b_desc_lo + stage_idx * sizeof(SharedStorage::smem_b[0]) / 16;
                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t umma_k_block_idx = 0; umma_k_block_idx < BLOCK_K / UMMA_BLOCK_K; ++ umma_k_block_idx) {
                            // UTCCP copy SFA and SFB to TMEM
                            using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfa[stage_idx] + umma_k_block_idx * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfb[stage_idx] + umma_k_block_idx * SF_BLOCK_N + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                            }

                            // Issue UMMA
                            #pragma unroll
                            for (uint32_t k = 0; k < UMMA_BLOCK_K / UMMA_K; ++ k) {
                                const auto runtime_instr_desc =
                                    mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);
                                a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_M * sizeof(a_dtype_t), k * UMMA_K);
                                b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, weight_dtype_t>(b_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_N * sizeof(weight_dtype_t), k * UMMA_K);
                                ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                    b_desc, a_desc, accum_stage_idx * UMMA_N,
                                    k_block_idx > 0 or umma_k_block_idx > 0 or k > 0, runtime_instr_desc,
                                    kTmemStartColOfSFB, kTmemStartColOfSFA);
                            }
                        }
                    }
                    __syncwarp();

                    // Commit to the mbarrier object
                    // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
            }

            // To safely deconstruct barriers, we need another round of waits
            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages].wait(accum_phase_idx);
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // Do mainloop by the leader CTA
        if (is_leader_cta)
            scheduler.mainloop(num_tokens, sm_locality_domains);
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&shared_storage.tmem_ptr_in_smem) == 0);

        // GEMM epilogue warps
        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        // TODO: support effective block M
        // NOTES:
        //  - 2 warpgroups divide the whole BM into BM / 2
        //  - 4 warps divide the whole BN into BN / 4
        //  - BM / 2 is further divided into stored blocks, i.e. with `STORE_BLOCK_M_L1` / `STORE_BLOCK_M_L2` size
        //  - Store blocks are further divided into `ATOM_M`
        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M_L1 == 0 and WG_BLOCK_M % STORE_BLOCK_M_L2 == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M_L1 % ATOM_M == 0 and STORE_BLOCK_M_L2 % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        // D0 (on the epilogue warps, which are idle until the first task, so the dispatch grid sync never waits for
        // Gin posting). The meta (top-k idx, weights, header) is staged by the last SM into the mirror slot of our own
        // node (unused locally; rows >= num_tokens hold sentinels) and sent with ONE put per counterpart; then SF and
        // x chunks, one piece per SM from the end. Data-only puts; the input region is written before the launch.
        IB_TRACE(23, sm_idx == kIBMetaSM and epilogue_thread_idx == 0);
        if (kIBHeavy and kIBSSTW and ib_sst and not (kCGDrop & 1)) {
            DG_IB_OUTLINE_BEGIN
            if (sm_idx + 2 + (kNumRanks - ib_lsa) == kNumSMs and epilogue_warp_idx == 0 and lane_idx + 1 < ib_num_nodes and (num_tokens > 0 or kIBSSTRPut)) {
                const uint32_t nr = lane_idx, n = ib_node_of_slot(nr);
                const auto gin_w = make_gin_site(1, ctx_meta(nr, nr), nr, ib_xbase);
                if (num_tokens > 0)
                    IB_PUT(gin_w, nr, ib_mirror(ib_parity, ib_my_node) + kIBOffW, ib_input_base + kIBOffW,
                        static_cast<uint64_t>(num_tokens) * kNumTopk * 4, ib_cp(sym_buffer.rank_idx, n));
                SL_TR(0, 42, true);  // early weight put issued
                if constexpr (kIBSSTRPut) {
                    IB_POLL(ptx::ld_acq_gpu(ib_xo_flag + (kIBSSTNB ? 244 + nr : 250u)) != ib_epoch, (void) 0, "sst_rput", nr, 0);
                    const uint64_t ab = ptx::ld_volatile(ib_xo_flag + 251 + nr);
                    const uint32_t a = static_cast<uint32_t>(ab >> 32), b = static_cast<uint32_t>(ab);  // NB: a = 0, b = entries
                    const uint32_t cb = kNumExpertsPerRank * ib_lsa * 4;
                    IB_PUT(gin_w, nr, ib_mirror(ib_parity, ib_my_node) + kIBOffSTab - cb - kNBHdr,
                              (kIBSSTNB ? ib_nb_stg(nr) : ib_mirror(ib_parity, ib_my_node) + kIBOffCnt + static_cast<int64_t>(nr) * cb + static_cast<int64_t>(a) * 4),
                              static_cast<uint64_t>(kNBHdr) + static_cast<uint64_t>(cb) + static_cast<uint64_t>(b - a) * 4, ib_cp(sym_buffer.rank_idx, n));
                    SL_TR(0, 30, true);  // SST put issued (slowest lane)
                }
            }
            __syncwarp();
            DG_IB_OUTLINE_END
        }
        // ORD: the order builder SM (see the knob)
        if constexpr (kIBOrd > 0) {
            DG_IB_OUTLINE_BEGIN
            if (kIBHeavy and kIBPrewarm and sm_idx + 4 + (kNumRanks - ib_lsa) == kNumSMs and epilogue_warp_idx == 0 and lane_idx + 1 < ib_num_nodes and ib_ord_on(num_tokens)) {
                const uint32_t nr = lane_idx, orig = 2 * ib_R + nr % ib_xnctx();
                const auto gin_p = make_gin_site(4, ctx_piece(nr, orig), orig, ib_xbase);
                IB_PUTV(gin_p, orig, reinterpret_cast<uint64_t*>(ib_mirror(ib_parity, ib_my_node) + kIBOffHdr + 64), ib_epoch, ib_cp(sym_buffer.rank_idx, ib_node_of_slot(nr)));
            }
            __syncwarp();
            if (kIBOrdCoal and kIBHeavy and not kIBSSTNB and ib_ord_on(num_tokens) and sm_idx + 3 + (kNumRanks - ib_lsa) == kNumSMs) {
                constexpr uint32_t kOK = kNumExpertsPerRank + 1;
                constexpr uint32_t kOWStride = 3 * kOK;
                constexpr uint32_t kTB = 8;  // tokens in flight per warp
                DG_STATIC_ASSERT(not kIBOrdCoal or (kNumEpilogueWarps + 2) * kOWStride * 2 + 4 <= sizeof(SharedStorage::smem_d), "ORD_COAL buckets in smem_d");
                auto* const oh = reinterpret_cast<uint32_t*>(&shared_storage.smem_d);
                auto* const o_wh_all = reinterpret_cast<uint16_t*>(oh + kOWStride);
                auto* const o_wh = o_wh_all + epilogue_warp_idx * kOWStride;
                const auto idx32 = reinterpret_cast<const uint32_t*>(ib_input_base + kIBOffIdx);  // low halves
                const uint32_t node_experts = kNumExpertsPerRank * ib_lsa;
                if (ib_R > 3) __trap();
                SL_TR(0, 43, epilogue_thread_idx == 0);
                for (uint32_t i = epilogue_thread_idx; i < kOWStride + (kNumEpilogueWarps * kOWStride + 1) / 2; i += kNumEpilogueThreads)
                    oh[i] = 0;
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 44, epilogue_thread_idx == 0);
                // key of token t for remote-node slot nr: warp min over the lanes (= top-k slots) of the local expert index
                const auto ord_key = [&](const uint32_t& e, const uint32_t& nr) {
                    const bool valid = lane_idx < kNumTopk and static_cast<int>(e) >= 0;
                    const uint32_t n = ib_num_nodes <= 4 ? (e >= node_experts ? 1u : 0u) + (e >= 2 * node_experts ? 1u : 0u) + (e >= 3 * node_experts ? 1u : 0u) : e / node_experts;
                    const uint32_t slot = n - (n > ib_my_node ? 1u : 0u);
                    return __reduce_min_sync(0xffffffffu, (valid and n != ib_my_node and slot == nr) ? e % kNumExpertsPerRank : kNumExpertsPerRank);
                };
                for (uint32_t t0 = epilogue_warp_idx; t0 < num_tokens; t0 += kNumEpilogueWarps * kTB) {
                    uint32_t v[kTB];
                    #pragma unroll
                    for (uint32_t b = 0; b < kTB; ++ b) {
                        const uint32_t t = t0 + b * kNumEpilogueWarps;
                        v[b] = (t < num_tokens and lane_idx < kNumTopk) ? __ldcg(idx32 + 2 * (t * kNumTopk + lane_idx)) : 0xffffffffu;
                    }
                    #pragma unroll
                    for (uint32_t b = 0; b < kTB; ++ b) {
                        const uint32_t t = t0 + b * kNumEpilogueWarps;
                        for (uint32_t nr = 0; nr < ib_R; ++ nr) {
                            const uint32_t key = ord_key(v[b], nr);
                            if (t < num_tokens and lane_idx == 0)
                                o_wh[nr * kOK + key] += 1;
                        }
                    }
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 45, epilogue_thread_idx == 0);
                for (uint32_t q = epilogue_thread_idx; q < ib_R * kOK; q += kNumEpilogueThreads) {
                    uint32_t tot = 0;
                    #pragma unroll
                    for (uint32_t w = 0; w < kNumEpilogueWarps; ++ w) {
                        const uint32_t c = o_wh_all[w * kOWStride + q];
                        o_wh_all[w * kOWStride + q] = static_cast<uint16_t>(tot);
                        tot += c;
                    }
                    oh[q] = tot;
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_thread_idx < ib_R) {
                    uint32_t run = 0;
                    for (uint32_t key = 0; key < kOK; ++ key) {
                        const uint32_t c = oh[epilogue_thread_idx * kOK + key];
                        oh[epilogue_thread_idx * kOK + key] = run;
                        run += c;
                    }
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                for (uint32_t t0 = epilogue_warp_idx; t0 < num_tokens; t0 += kNumEpilogueWarps * kTB) {
                    uint32_t v[kTB];
                    #pragma unroll
                    for (uint32_t b = 0; b < kTB; ++ b) {
                        const uint32_t t = t0 + b * kNumEpilogueWarps;
                        v[b] = (t < num_tokens and lane_idx < kNumTopk) ? __ldcg(idx32 + 2 * (t * kNumTopk + lane_idx)) : 0xffffffffu;
                    }
                    #pragma unroll
                    for (uint32_t b = 0; b < kTB; ++ b) {
                        const uint32_t t = t0 + b * kNumEpilogueWarps;
                        for (uint32_t nr = 0; nr < ib_R; ++ nr) {
                            const uint32_t key = ord_key(v[b], nr);
                            if (t < num_tokens and lane_idx == 0) {
                                const uint32_t pos = oh[nr * kOK + key] + o_wh[nr * kOK + key];
                                o_wh[nr * kOK + key] += 1;
                                ib_ord_pos(nr)[t] = pos;
                                ib_ord_inv(nr)[pos] = t;
                            }
                        }
                    }
                }
                SL_TR(0, 46, epilogue_thread_idx == 0);
                __threadfence();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 47, epilogue_thread_idx == 0);
                if (epilogue_thread_idx < ib_R)
                    ptx::st_rel(ib_xo_flag + 240 + epilogue_thread_idx, ib_epoch);
                SL_TR(0, 40, epilogue_thread_idx == 0);  // order published
            }
            const uint32_t mso_m = kNumSMs - (kNumRanks - ib_lsa) - (kIBMSO > 0 ? 5u : 3u) - sm_idx;  // huge on the other SMs
            if (not kIBOrdCoal and kIBHeavy and not kIBSSTNB and ib_ord_on(num_tokens) and mso_m < (kIBMSO > 0 ? kIBMSO : 1u)) {
                constexpr uint32_t kOK = kNumExpertsPerRank + 1;  // buckets per node (the last: no pick on the node)
                constexpr uint32_t kMx = kIBMSO > 0 ? kIBMSO : 1u;
                constexpr uint32_t kOT = math::constexpr_ceil_div(static_cast<uint32_t>(kIBT), kNumEpilogueThreads * kMx);
                auto* const oh = reinterpret_cast<uint32_t*>(&shared_storage.smem_d);
                constexpr uint32_t kOWStride = 3 * kOK;  // FAST: per-warp buckets [warp][<= 3 remote nodes][key] after oh
                DG_STATIC_ASSERT(not kIBSSTFast or (kNumEpilogueWarps + 2 + (kIBMSO > 0 ? 2u : 0u)) * kOWStride * 2 + 4 <= sizeof(SharedStorage::smem_d), "ORD per-warp buckets in smem_d");
                auto* const o_wh_all = reinterpret_cast<uint16_t*>(oh + kOWStride);  // 16-bit: counts <= kIBT
                auto* const ob = oh + kOWStride + (kNumEpilogueWarps * kOWStride + 1) / 2;  // MSO: the lower SMs' share per (node, key)
                auto* const mso_g = reinterpret_cast<uint32_t*>(ib_mirror(ib_parity, ib_my_node)) + 48 * kIBT;  // MSO rows [m][3 * kOK]
                auto* const o_wh = o_wh_all + epilogue_warp_idx * kOWStride;
                const auto in_idx = reinterpret_cast<const long long*>(ib_input_base + kIBOffIdx);
                const uint32_t node_experts = kNumExpertsPerRank * ib_lsa;
                if (ib_R > (kIBSSTFast ? 3u : 4u)) __trap();  // keys are packed 8 bits per remote node
                for (uint32_t i = epilogue_thread_idx; i < (kIBSSTFast ? kOWStride + (kNumEpilogueWarps * kOWStride + 1) / 2 : ib_R * kOK); i += kNumEpilogueThreads)
                    oh[i] = 0;
                SL_TR(0, 43, epilogue_thread_idx == 0);  // order pass entered
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 44, epilogue_thread_idx == 0);  // zeroed
                uint32_t kk[kOT];
                for (uint32_t j = 0; j < kOT; ++ j) {
                    const uint32_t t = (j * kNumEpilogueThreads + epilogue_thread_idx) * kMx + (kIBMSO > 0 ? mso_m : 0u);
                    uint32_t packed = kNumExpertsPerRank * 0x01010101u;
                    if (t < num_tokens) {
                        uint32_t v[kNumTopk];
                        #pragma unroll
                        for (uint32_t k = 0; k < kNumTopk; ++ k)
                            v[k] = kIBSSTFast ? __ldcg(reinterpret_cast<const uint32_t*>(in_idx) + 2 * (t * kNumTopk + k)) : static_cast<uint32_t>(__ldcg(in_idx + t * kNumTopk + k));
                        #pragma unroll
                        for (uint32_t k = 0; k < kNumTopk; ++ k) {
                            const int e = static_cast<int>(v[k]);
                            const uint32_t eu = static_cast<uint32_t>(e);
                            const bool o_cmp = kIBSSTFast and ib_num_nodes <= 4;
                            const uint32_t o_n = o_cmp ? (eu >= node_experts ? 1u : 0u) + (eu >= 2 * node_experts ? 1u : 0u) + (eu >= 3 * node_experts ? 1u : 0u) : eu / node_experts;
                            if (e >= 0 and o_n != ib_my_node) {
                                const uint32_t sh = 8 * ib_nr_of_node(o_n);
                                const uint32_t le = static_cast<uint32_t>(e) % kNumExpertsPerRank;
                                if (le < ((packed >> sh) & 0xffu))
                                    packed = (packed & ~(0xffu << sh)) | (le << sh);
                            }
                        }
                        if constexpr (not kIBSSTFast) {
                            for (uint32_t nr = 0; nr < ib_R; ++ nr)
                                atomicAdd_block(oh + nr * kOK + ((packed >> (8 * nr)) & 0xffu), 1u);
                        }
                    }
                    if constexpr (kIBSSTFast) {  // per-warp buckets: equal keys inside the warp resolved with match_any, no atomics
                        for (uint32_t nr = 0; nr < ib_R; ++ nr) {
                            const uint32_t key = (packed >> (8 * nr)) & 0xffu;
                            const uint32_t peers = __match_any_sync(0xffffffffu, t < num_tokens ? key : 0xffffffffu);
                            if (t < num_tokens and (peers & ((1u << lane_idx) - 1u)) == 0)
                                o_wh[nr * kOK + key] += __popc(peers);
                            __syncwarp();
                        }
                    }
                    kk[j] = packed;
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 45, epilogue_thread_idx == 0);  // keys + histogram done
                if constexpr (kIBSSTFast) {  // per (node, key): per-warp counts -> per-warp bases, total into oh
                    for (uint32_t q = epilogue_thread_idx; q < ib_R * kOK; q += kNumEpilogueThreads) {
                        uint32_t tot = 0;
                        #pragma unroll
                        for (uint32_t w = 0; w < kNumEpilogueWarps; ++ w) {
                            const uint32_t c = o_wh_all[w * kOWStride + q];
                            o_wh_all[w * kOWStride + q] = static_cast<uint16_t>(tot);
                            tot += c;
                        }
                        oh[q] = tot;
                        if constexpr (kIBMSO > 0)
                            mso_g[mso_m * kOWStride + q] = tot;
                    }
                    if constexpr (kIBMSO > 0) {
                        __threadfence();
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_thread_idx == 0)
                            ptx::st_rel(ib_xo_flag + 200 + mso_m, ib_epoch);
                        if (epilogue_thread_idx < kIBMSO)
                            while (ptx::ld_acq_gpu(ib_xo_flag + 200 + epilogue_thread_idx) != ib_epoch);
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        for (uint32_t q = epilogue_thread_idx; q < ib_R * kOK; q += kNumEpilogueThreads) {
                            uint32_t all = 0, before = 0;
                            #pragma unroll
                            for (uint32_t m2 = 0; m2 < kMx; ++ m2) {
                                const uint32_t c = __ldcg(mso_g + m2 * kOWStride + q);
                                all += c;
                                if (m2 < mso_m) before += c;
                            }
                            oh[q] = all, ob[q] = before;
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
                if (epilogue_thread_idx < ib_R) {  // exclusive prefix per node
                    uint32_t run = 0;
                    for (uint32_t key = 0; key < kOK; ++ key) {
                        const uint32_t c = oh[epilogue_thread_idx * kOK + key];
                        oh[epilogue_thread_idx * kOK + key] = run;
                        run += c;
                    }
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                for (uint32_t j = 0; j < kOT; ++ j) {
                    const uint32_t t = (j * kNumEpilogueThreads + epilogue_thread_idx) * kMx + (kIBMSO > 0 ? mso_m : 0u);
                    if (t < num_tokens) {
                        if constexpr (not kIBSSTFast) {
                        for (uint32_t nr = 0; nr < ib_R; ++ nr) {
                            const uint32_t pos = atomicAdd_block(oh + nr * kOK + ((kk[j] >> (8 * nr)) & 0xffu), 1u);
                            ib_ord_pos(nr)[t] = pos;
                            ib_ord_inv(nr)[pos] = t;
                        }
                        }
                    }
                    if constexpr (kIBSSTFast) {
                        for (uint32_t nr = 0; nr < ib_R; ++ nr) {
                            const uint32_t key = (kk[j] >> (8 * nr)) & 0xffu;
                            const uint32_t peers = __match_any_sync(0xffffffffu, t < num_tokens ? key : 0xffffffffu);
                            const uint32_t rank = __popc(peers & ((1u << lane_idx) - 1u));
                            const uint32_t pos = t < num_tokens ? oh[nr * kOK + key] + (kIBMSO > 0 ? ob[nr * kOK + key] : 0u) + o_wh[nr * kOK + key] + rank : 0u;
                            __syncwarp();
                            if (t < num_tokens) {
                                ib_ord_pos(nr)[t] = pos;
                                ib_ord_inv(nr)[pos] = t;
                                if (rank == 0) o_wh[nr * kOK + key] += __popc(peers);
                            }
                            __syncwarp();
                        }
                    }
                }
                SL_TR(0, 46, epilogue_thread_idx == 0);  // positions written (before the fence)
                __threadfence();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 47, epilogue_thread_idx == 0);  // fenced
                if constexpr (kIBMSO > 0) {
                    if (epilogue_thread_idx == 0)
                        ptx::st_rel(ib_xo_flag + 216 + mso_m, ib_epoch);
                } else if (epilogue_thread_idx < ib_R)
                    ptx::st_rel(ib_xo_flag + 240 + epilogue_thread_idx, ib_epoch);
                SL_TR(0, 40, epilogue_thread_idx == 0);  // order published
            }
            DG_IB_OUTLINE_END
        }
        // SST_NB: one builder SM per remote node (see the knob)
        if (kIBHeavy and kIBSSTNB and ib_sst and not (kCGDrop & 16)) {
            DG_IB_OUTLINE_BEGIN
            const uint32_t nb_nr = kNumSMs - (kNumRanks - ib_lsa) - 5 - sm_idx;  // wraps to a huge value on the other SMs
            if (nb_nr < ib_R) {
                const uint32_t nr = nb_nr, n = ib_node_of_slot(nr);
                const uint32_t node_experts = kNumExpertsPerRank * ib_lsa, e_lo = n * node_experts;
                constexpr uint32_t kOT = math::constexpr_ceil_div(static_cast<uint32_t>(kIBT), kNumEpilogueThreads);
                constexpr uint32_t kPerN = math::constexpr_ceil_div(kNBCap, kNumEpilogueThreads);
                auto* const nb_smem = reinterpret_cast<uint32_t*>(&shared_storage.smem_d);
                auto* const cnt = kNBG ? ib_nb_g(nr) : nb_smem;
                auto* const nb_wsum = nb_smem + kNBCap;
                auto* const nb_tot = nb_wsum + kNumEpilogueWarps;
                auto* const oh = kNBG ? ib_nb_g(nr) + kNBCap : nb_tot + 1;
                auto* const nb_wh = nb_tot + 1 + kNBKeys + epilogue_warp_idx * kNBKeys;  // NB=3: this warp's bucket counts / bases
                const auto in_idx = reinterpret_cast<const long long*>(ib_input_base + kIBOffIdx);
                const bool ord = ib_ord_on(num_tokens);
                auto* const stg_cnt = reinterpret_cast<uint32_t*>(ib_nb_stg(nr) + kNBHdr);
                if (kIBWmark and epilogue_thread_idx == 0)
                    *reinterpret_cast<uint64_t*>(ib_nb_stg(nr)) = (ib_epoch << 32) | (ord ? 0x80000000ull : 0ull) | num_tokens;
                auto* const stg_ent = stg_cnt + node_experts;
                const auto nb_add = [&](uint32_t* const ptr) { if (kNBG) ptx::red_add(ptr, 1u); else atomicAdd_block(ptr, 1u); };
                const auto nb_take = [&](uint32_t* const ptr) { return kNBG ? ptx::atomic_add(ptr, 1u) : atomicAdd_block(ptr, 1u); };
                SL_TR(0, 33, epilogue_thread_idx == 0);
                for (uint32_t i = epilogue_thread_idx; i < kNBCap + kNumEpilogueWarps + 1 + kNBKeys + (kNBW ? kNumEpilogueWarps * kNBKeys : 0u); i += kNumEpilogueThreads)
                    nb_smem[i] = 0;
                if constexpr (kNBG) {
                    for (uint32_t i = epilogue_thread_idx; i < kNBCap + 64; i += kNumEpilogueThreads)
                        cnt[i] = 0;
                    __threadfence();
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 34, epilogue_thread_idx == 0);
                // Pass 1 (two tokens per round): counts per expert of the node, first-need key per token (registers) + histogram
                uint32_t kk[kOT + 1];
                for (uint32_t j = 0; j < kOT; j += 2) {
                    long long v[2][kNumTopk];
                    #pragma unroll
                    for (uint32_t h = 0; h < 2; ++ h) {
                        const uint32_t t = (j + h) * kNumEpilogueThreads + epilogue_thread_idx;
                        #pragma unroll
                        for (uint32_t k = 0; k < kNumTopk; ++ k)
                            v[h][k] = t < num_tokens ? __ldcg(in_idx + t * kNumTopk + k) : -1ll;
                    }
                    #pragma unroll
                    for (uint32_t h = 0; h < 2; ++ h) {
                        const uint32_t t = (j + h) * kNumEpilogueThreads + epilogue_thread_idx;
                        uint32_t key = kNumExpertsPerRank;
                        #pragma unroll
                        for (uint32_t k = 0; k < kNumTopk; ++ k) {
                            const int e = static_cast<int>(v[h][k]);
                            if (e >= 0 and static_cast<uint32_t>(e) / node_experts == n) {
                                nb_add(cnt + (static_cast<uint32_t>(e) - e_lo));
                                key = cute::min(key, static_cast<uint32_t>(e) % kNumExpertsPerRank);
                            }
                        }
                        if constexpr (kNBW) {
                            const uint32_t peers = __match_any_sync(0xffffffffu, t < num_tokens ? key : 0xffffffffu);
                            if (t < num_tokens and (peers & ((1u << lane_idx) - 1u)) == 0)
                                nb_wh[key] += __popc(peers);
                            __syncwarp();
                        } else if (t < num_tokens) {
                            nb_add(oh + key);
                        }
                        kk[j + h] = key;
                    }
                }
                if constexpr (kNBG)
                    __threadfence();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 35, epilogue_thread_idx == 0);
                // Exclusive prefix over the node's experts (cursors) + counts to the staging; bucket starts of the keys (warp 0)
                uint32_t nb_c[kPerN];
                uint32_t mine = 0;
                #pragma unroll
                for (uint32_t u = 0; u < kPerN; ++ u) {
                    const uint32_t g = epilogue_thread_idx * kPerN + u;
                    nb_c[u] = g < node_experts ? cnt[g] : 0u;
                }
                #pragma unroll
                for (uint32_t u = 0; u < kPerN; ++ u) {
                    const uint32_t g = epilogue_thread_idx * kPerN + u;
                    mine += nb_c[u];
                    if (g < node_experts)
                        stg_cnt[g] = nb_c[u];
                }
                const uint32_t inc = math::warp_inclusive_sum(mine, lane_idx);
                if (lane_idx == 31)
                    nb_wsum[epilogue_warp_idx] = inc;
                if (epilogue_warp_idx == 0) {  // exclusive prefix of the key buckets, 32 keys per step
                    auto* const wh0 = nb_tot + 1 + kNBKeys;
                    uint32_t carry = 0;
                    #pragma unroll
                    for (uint32_t c = 0; c < (kNBKeys + 31) / 32; ++ c) {
                        const uint32_t kq = lane_idx + 32 * c;
                        uint32_t v = kq < kNBKeys ? oh[kq] : 0u;
                        if constexpr (kNBW) {  // per-warp counts -> per-warp exclusive bases, total per key
                            v = 0;
                            #pragma unroll
                            for (uint32_t w = 0; w < kNumEpilogueWarps; ++ w) {
                                if (kq < kNBKeys) { const uint32_t cw = wh0[w * kNBKeys + kq]; wh0[w * kNBKeys + kq] = v; v += cw; }
                            }
                        }
                        const uint32_t incl = math::warp_inclusive_sum(v, lane_idx);
                        if (kq < kNBKeys) oh[kq] = carry + incl - v;
                        carry += __shfl_sync(0xffffffffu, incl, 31);
                    }
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                uint32_t run = inc - mine;
                for (uint32_t w = 0; w < epilogue_warp_idx; ++ w)
                    run += nb_wsum[w];
                #pragma unroll
                for (uint32_t u = 0; u < kPerN; ++ u) {
                    const uint32_t g = epilogue_thread_idx * kPerN + u;
                    if (g < node_experts)
                        cnt[g] = run;
                    run += nb_c[u];
                }
                if (epilogue_thread_idx == kNumEpilogueThreads - 1)
                    *nb_tot = run;  // the last thread's running sum = every entry of the node
                if constexpr (kNBG)
                    __threadfence();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 36, epilogue_thread_idx == 0);
                // Pass 2 (two tokens per round): the token's row (bucket order), then its entries for the node, expert-major
                for (uint32_t j = 0; j < kOT; j += 2) {
                    long long v[2][kNumTopk];
                    uint32_t row[2], sl[2][kNumTopk];
                    #pragma unroll
                    for (uint32_t h = 0; h < 2; ++ h) {
                        const uint32_t t = (j + h) * kNumEpilogueThreads + epilogue_thread_idx;
                        row[h] = t;
                        if constexpr (kNBW) {
                            const bool val = ord and t < num_tokens;
                            const uint32_t key = kk[j + h];
                            const uint32_t peers = __match_any_sync(0xffffffffu, val ? key : 0xffffffffu);
                            const uint32_t rank = __popc(peers & ((1u << lane_idx) - 1u));
                            if (val) row[h] = oh[key] + nb_wh[key] + rank;
                            __syncwarp();
                            if (val and rank == 0) nb_wh[key] += __popc(peers);
                            __syncwarp();
                        } else if (ord and t < num_tokens) {
                            row[h] = nb_take(oh + kk[j + h]);
                        }
                        #pragma unroll
                        for (uint32_t k = 0; k < kNumTopk; ++ k)
                            v[h][k] = t < num_tokens ? __ldcg(in_idx + t * kNumTopk + k) : -1ll;
                    }
                    #pragma unroll
                    for (uint32_t h = 0; h < 2; ++ h) {
                        #pragma unroll
                        for (uint32_t k = 0; k < kNumTopk; ++ k) {
                            const int e = static_cast<int>(v[h][k]);
                            sl[h][k] = 0;
                            if (e >= 0 and static_cast<uint32_t>(e) / node_experts == n)
                                sl[h][k] = nb_take(cnt + (static_cast<uint32_t>(e) - e_lo));
                        }
                    }
                    #pragma unroll
                    for (uint32_t h = 0; h < 2; ++ h) {
                        const uint32_t t = (j + h) * kNumEpilogueThreads + epilogue_thread_idx;
                        if (ord and t < num_tokens) {
                            ib_ord_pos(nr)[t] = row[h];
                            ib_ord_inv(nr)[row[h]] = t;
                        }
                        #pragma unroll
                        for (uint32_t k = 0; k < kNumTopk; ++ k) {
                            const int e = static_cast<int>(v[h][k]);
                            if (e >= 0 and static_cast<uint32_t>(e) / node_experts == n)
                                stg_ent[sl[h][k]] = kIBPacked ? ((t * kNumTopk + k) | (row[h] << 16)) : (t * kNumTopk + k);
                        }
                    }
                }
                SL_TR(0, 37, epilogue_thread_idx == 0);
                if (kIBSSTRPut and epilogue_thread_idx == 0)
                    ib_xo_flag[251 + nr] = static_cast<uint64_t>(*nb_tot);
                __threadfence();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 29, epilogue_thread_idx == 0);  // staged
                if (epilogue_thread_idx == 0) {
                    if constexpr (kIBSSTRPut) {
                        ptx::st_rel(ib_xo_flag + 244 + nr, ib_epoch);
                    } else {
#if defined(DG_MEGA_MOE_IB_SST_CTX) and DG_MEGA_MOE_IB_SST_CTX == 1
                        const auto gin_s = make_gin_site(1, ctx_meta(nr, nr), nr, ib_xbase);  // meta context
#else
                        const auto gin_s = make_gin_site(2, ctx_mhdr(nr, hdr_orig(nr)), hdr_orig(nr), ib_xbase);  // header context
#endif
#if defined(DG_MEGA_MOE_IB_SST_CTX) and DG_MEGA_MOE_IB_SST_CTX == 1
                        const uint32_t kNBCtx = nr;
#else
                        const uint32_t kNBCtx = hdr_orig(nr);
#endif
                        IB_PUT(gin_s, kNBCtx, ib_mirror(ib_parity, ib_my_node) + kIBOffSTab - static_cast<int64_t>(node_experts) * 4 - kNBHdr, ib_nb_stg(nr),
                                  static_cast<uint64_t>(kNBHdr) + static_cast<uint64_t>(node_experts) * 4 + static_cast<uint64_t>(*nb_tot) * 4, ib_cp(sym_buffer.rank_idx, n));
                        SL_TR(0, 30, true);
                    }
                    if (ord)
                        ptx::st_rel(ib_xo_flag + 240 + nr, ib_epoch);
                    SL_TR(0, 40, ord);
                }
                __syncwarp();
            }
            DG_IB_OUTLINE_END
        }
        if (kIBHeavy and kIBSST and not kIBSSTNB and ib_sst and not (kCGDrop & 2)) {
            DG_IB_OUTLINE_BEGIN
            if (sm_idx + 1 + (kNumRanks - ib_lsa) == kNumSMs) {
                // SST build (see the knob). Counters: 16 bits per global expert, two per word (counts <= kIBT, cursors < 65536)
                const auto in_idx = reinterpret_cast<const unsigned long long*>(ib_input_base + kIBOffIdx);
                auto* const c16 = reinterpret_cast<uint32_t*>(&shared_storage.smem_d);
                auto* const sst_wsum = c16 + kSSTWords;
                auto* const sst_nstart = sst_wsum + kNumEpilogueWarps;  // [node] first entry of the node's experts, [nodes] total
                const uint32_t sst_node_experts = kNumExpertsPerRank * ib_lsa;
                const uint32_t sst_ne = num_tokens * kNumTopk;
                // No runtime divisions per entry (ib_lsa is not a constant): own-node range test, node slot by compares
                const uint32_t sst_lo = ib_my_node * sst_node_experts, sst_hi = sst_lo + sst_node_experts;
                const bool sst_cmp = kIBSSTFast and ib_num_nodes <= 4;
                const auto sst_remote = [&](const uint32_t& e) { return sst_cmp ? (e < sst_lo or e >= sst_hi) : e / sst_node_experts != ib_my_node; };
                const auto sst_nr_of = [&](const uint32_t& e) {
                    if (not sst_cmp) return ib_nr_of_node(e / sst_node_experts);
                    const uint32_t n = (e >= sst_node_experts ? 1u : 0u) + (e >= 2 * sst_node_experts ? 1u : 0u) + (e >= 3 * sst_node_experts ? 1u : 0u);
                    return n - (n > ib_my_node ? 1u : 0u);
                };
                const auto stg = ib_mirror(ib_parity, ib_my_node) + kIBOffCnt;  // our own node's slot: unused locally
                constexpr uint32_t kSSTB = kIBSSTFast ? 16 : 8;  // loads in flight per thread
                const auto in_idx32 = reinterpret_cast<const uint32_t*>(ib_input_base + kIBOffIdx);  // low halves (little endian)
                SL_TR(0, 33, epilogue_thread_idx == 0);  // SST build entered
                for (uint32_t w = epilogue_thread_idx; w < kSSTWords; w += kNumEpilogueThreads)
                    c16[w] = 0;
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 34, epilogue_thread_idx == 0);  // counters zeroed
                for (uint32_t i0 = 0; i0 < sst_ne; i0 += kNumEpilogueThreads * kSSTB) {
                    uint32_t v[kSSTB];
                    #pragma unroll
                    for (uint32_t u = 0; u < kSSTB; ++ u) {
                        const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                        v[u] = i >= sst_ne ? 0xffffffffu : kIBSSTFast ? __ldcg(in_idx32 + 2 * i) : static_cast<uint32_t>(__ldcg(in_idx + i));
                    }
                    #pragma unroll
                    for (uint32_t u = 0; u < kSSTB; ++ u) {
                        const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                        const int e = static_cast<int>(static_cast<int64_t>(v[u]));
                        if (i < sst_ne and e >= 0 and sst_remote(static_cast<uint32_t>(e)))
                            atomicAdd_block(c16 + static_cast<uint32_t>(e) / 2, 1u << (16 * (static_cast<uint32_t>(e) & 1u)));
                    }
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 35, epilogue_thread_idx == 0);  // count pass done
                // Exclusive prefix over the experts in global order (our own node's count 0). Each thread owns kSSTPer
                // consecutive experts = whole counter words
                constexpr uint32_t kSSTPer = 2 * math::constexpr_ceil_div(kNumExperts, 2 * kNumEpilogueThreads);
                const uint32_t g0 = epilogue_thread_idx * kSSTPer;
                uint32_t sst_c[kSSTPer];
                uint32_t mine = 0;
                #pragma unroll
                for (uint32_t u = 0; u < kSSTPer; ++ u) {
                    const uint32_t g = g0 + u;
                    sst_c[u] = g < kNumExperts ? (c16[g / 2] >> (16 * (g & 1u))) & 0xffffu : 0u;
                    mine += sst_c[u];
                }
                const uint32_t inc = math::warp_inclusive_sum(mine, lane_idx);
                if (lane_idx == 31)
                    sst_wsum[epilogue_warp_idx] = inc;
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                uint32_t run = inc - mine;
                for (uint32_t w = 0; w < epilogue_warp_idx; ++ w)
                    run += sst_wsum[w];
                #pragma unroll
                for (uint32_t u = 0; u < kSSTPer; u += 2) {
                    const uint32_t g = g0 + u;
                    if (g < kNumExperts)
                        c16[g / 2] = run | ((run + sst_c[u]) << 16);  // cursors of g and g + 1
                    run += sst_c[u] + sst_c[u + 1];
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_thread_idx <= ib_num_nodes) {
                    uint32_t total = 0;
                    for (uint32_t w = 0; w < kNumEpilogueWarps; ++ w)
                        total += sst_wsum[w];
                    const uint32_t g = epilogue_thread_idx * sst_node_experts;
                    sst_nstart[epilogue_thread_idx] = g < kNumExperts ? (c16[g / 2] >> (16 * (g & 1u))) & 0xffffu : total;
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 36, epilogue_thread_idx == 0);  // prefix + node starts done
                // Staging per remote node slot nr: [counts of the node's experts][its entries], contiguous for one put:
                // counts at stg + nr * cb + 4 * start(node), entry with global position p at stg + (nr + 1) * cb + 4 * p
                const uint32_t sst_cb = sst_node_experts * 4;
                #pragma unroll
                for (uint32_t u = 0; u < kSSTPer; ++ u) {
                    const uint32_t g = g0 + u;
                    if (g < kNumExperts and g / sst_node_experts != ib_my_node) {
                        const uint32_t n = g / sst_node_experts, nr = ib_nr_of_node(n);
                        reinterpret_cast<uint32_t*>(stg + static_cast<int64_t>(nr) * sst_cb)[sst_nstart[n] + g % sst_node_experts] = sst_c[u];
                    }
                }
                // SST + WAVE_SORT: each entry also carries its x row in the mirror (the node's wave position of its token)
                const bool sst_wave = ((kIBSSTWave and ib_ws_on(num_tokens)) or ib_ord_on(num_tokens)) and num_tokens > 0;
                const auto sst_pos_g = reinterpret_cast<const uint32_t*>(ib_mirror(ib_parity, ib_my_node));
                if (kIBMSO > 0 and ib_ord_on(num_tokens) and num_tokens > 0) {
                    if (epilogue_thread_idx < kIBMSO)
                        while (ptx::ld_acq_gpu(ib_xo_flag + 216 + epilogue_thread_idx) != ib_epoch);
                } else if (sst_wave and epilogue_thread_idx + 1 < ib_num_nodes)  // ORD: one flag for every node
                    IB_POLL(ptx::ld_acq_gpu(ib_xo_flag + 240 + epilogue_thread_idx) != ib_epoch, (void) 0, "sst_wave", epilogue_thread_idx, 0);
                if constexpr (kIBPacked)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 41, epilogue_thread_idx == 0);  // wave order seen by the SST builder
                for (uint32_t i0 = 0; i0 < sst_ne; i0 += kNumEpilogueThreads * kSSTB) {
                    uint32_t v[kSSTB];
                    uint32_t xr[kSSTB];
                    #pragma unroll
                    for (uint32_t u = 0; u < kSSTB; ++ u) {
                        const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                        v[u] = i >= sst_ne ? 0xffffffffu : kIBSSTFast ? __ldcg(in_idx32 + 2 * i) : static_cast<uint32_t>(__ldcg(in_idx + i));
                    }
                    #pragma unroll
                    for (uint32_t u = 0; u < kSSTB; ++ u) {
                        const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                        const int e = static_cast<int>(static_cast<int64_t>(v[u]));
                        const bool ok = i < sst_ne and e >= 0 and sst_remote(static_cast<uint32_t>(e));
                        xr[u] = i / kNumTopk;
                        if (sst_wave and ok)
                            xr[u] = __ldcg(sst_pos_g + static_cast<int64_t>(sst_nr_of(static_cast<uint32_t>(e))) * kIBT + i / kNumTopk);
                    }
                    #pragma unroll
                    for (uint32_t u = 0; u < kSSTB; ++ u) {
                        const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                        const int e = static_cast<int>(static_cast<int64_t>(v[u]));
                        if (i < sst_ne and e >= 0 and sst_remote(static_cast<uint32_t>(e))) {
                            const uint32_t sh = 16 * (static_cast<uint32_t>(e) & 1u);
                            const uint32_t pos = (atomicAdd_block(c16 + static_cast<uint32_t>(e) / 2, 1u << sh) >> sh) & 0xffffu;
                            const uint32_t nr = sst_nr_of(static_cast<uint32_t>(e));
                            reinterpret_cast<uint32_t*>(stg + static_cast<int64_t>(nr + 1) * sst_cb)[pos] = kIBPacked ? (i | (xr[u] << 16)) : i;
                        }
                    }
                }
                SL_TR(0, 37, epilogue_thread_idx == 0);  // scatter pass done (before the fence)
                __threadfence();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                SL_TR(0, 29, epilogue_thread_idx == 0);  // SST staged
                if constexpr (kIBSSTRPut) {  // hand the node starts to the SM below (it issues the put)
                    if (epilogue_thread_idx + 1 < ib_num_nodes) {
                        const uint32_t n = ib_node_of_slot(epilogue_thread_idx);
                        ib_xo_flag[251 + epilogue_thread_idx] = (static_cast<uint64_t>(sst_nstart[n]) << 32) | sst_nstart[n + 1];
                    }
                    __threadfence();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_thread_idx == 0)
                        ptx::st_rel(ib_xo_flag + 250, ib_epoch);
                }
                if (not kIBSSTRPut and epilogue_warp_idx == 0 and lane_idx + 1 < ib_num_nodes) {
                    const uint32_t nr = lane_idx, n = ib_node_of_slot(nr);
                    const uint32_t a = sst_nstart[n], b = sst_nstart[n + 1];
#if defined(DG_MEGA_MOE_IB_SST_CTX) and DG_MEGA_MOE_IB_SST_CTX == 1
                    const auto gin_s = make_gin_site(1, ctx_meta(nr, nr), nr, ib_xbase);  // meta context
#else
                    const auto gin_s = make_gin_site(2, ctx_mhdr(nr, hdr_orig(nr)), hdr_orig(nr), ib_xbase);  // header context
#endif
#if defined(DG_MEGA_MOE_IB_SST_CTX) and DG_MEGA_MOE_IB_SST_CTX == 1
                    const uint32_t kNBCtx = nr;
#else
                    const uint32_t kNBCtx = hdr_orig(nr);
#endif
                    IB_PUT(gin_s, kNBCtx, ib_mirror(ib_parity, ib_my_node) + kIBOffSTab - sst_cb,
                              stg + static_cast<int64_t>(nr) * sst_cb + static_cast<int64_t>(a) * 4,
                              static_cast<uint64_t>(sst_cb) + static_cast<uint64_t>(b - a) * 4, ib_cp(sym_buffer.rank_idx, n));
                    SL_TR(0, 30, true);  // SST put issued (slowest lane)
                }
                __syncwarp();
#if defined(DG_MEGA_MOE_IB_SST_XHOLD) and DG_MEGA_MOE_IB_SST_XHOLD
                // DG_MEGA_MOE_IB_SST_XHOLD=1 (mechanism test): the x / SF pieces are issued only after the SST put, so its
                // issue never competes with the ~100 x put issues (flag = the unused last X_ORDER slot; GPU scope)
                if (epilogue_thread_idx == 0)
                    ptx::st_rel(ib_xo_flag + 255, ib_epoch);
#endif
            }
            DG_IB_OUTLINE_END
        }
#if defined(DG_MEGA_MOE_IB_DEBUG_D0) and DG_MEGA_MOE_IB_DEBUG_D0
        if (sm_idx == kNumSMs - 1 and epilogue_thread_idx == 0 and ib_epoch <= 2)
            printf("OPUS_D0 rank=%u epoch=%llu num_tokens=%u enter\n", sym_buffer.rank_idx, (unsigned long long) ib_epoch, num_tokens);
#endif
        if (sm_idx == kIBMetaSM and num_tokens == kIBT and not kIBFront) {
            // Full batch: every top-k / weight row is polled by the proxy, so the input slices go out as they are
            // (no staging): one put for top-k idx + weights (contiguous in the input region), one for the header
            if (epilogue_warp_idx == 0 and lane_idx % 16 + 1 < ib_num_nodes) {
                const uint32_t nr = lane_idx % 16, n = ib_node_of_slot(nr);
                const auto gin = lane_idx < 16 ? make_gin_site(1, ctx_meta(nr, nr), nr, ib_xbase) : make_gin_site(2, ctx_mhdr(nr, hdr_orig(nr)), hdr_orig(nr), ib_xbase);
                const auto dst = ib_mirror(ib_parity, ib_my_node);
                if (lane_idx < 16) {
                    gin.put(dst + kIBOffIdx, ib_input_base + kIBOffIdx, static_cast<uint64_t>(kIBT * kNumTopk * 12), ib_cp(sym_buffer.rank_idx, n));
                    // HDR_CTX: this context carries nothing else, and the last op on a context can sit unprocessed until
                    // the next one (late-tail); a trailing 8-byte put of the (identical) header value pushes it through
                    if (kIBHdrCtx)
                        gin.put_value(reinterpret_cast<uint64_t*>(dst + kIBOffHdr), (ib_epoch << 32) | num_tokens, ib_cp(sym_buffer.rank_idx, n));
                } else {
                    gin.put_value(reinterpret_cast<uint64_t*>(dst + kIBOffHdr), (ib_epoch << 32) | num_tokens, ib_cp(sym_buffer.rank_idx, n));
                    if (kIBHdrCtx)  // doorbell: a second identical header write pushes the first (same late-tail reason)
                        gin.put_value(reinterpret_cast<uint64_t*>(dst + kIBOffHdr), (ib_epoch << 32) | num_tokens, ib_cp(sym_buffer.rank_idx, n));
                }
            }
            __syncwarp();
            IB_TRACE(24, epilogue_thread_idx == 0);
            if (epilogue_thread_idx == 0)
                { uint64_t t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); workspace.signals->ib_o_meta_posted_t = t_; }
                ptx::st_rel(&workspace.signals->ib_o_meta_posted, ib_epoch);  // GPU scope: only this GPU's x posters read it (sys scope cost ~10 us)
        } else if (kIBMetaDirect and sm_idx == kIBMetaSM) {
            DG_STATIC_ASSERT(not kIBMetaDirect or (kIBFront and kIBHdrCtx and not kIBCountsInMeta and not kIBSenderTables and not kIBCountsEarly),
                             "META_DIRECT needs FRONT + HDR_CTX and no COUNTS_IN_META / SENDER_TABLES / COUNTS_EARLY");
            // Three puts from three warps, issued concurrently, straight from the input region (rows < num_tokens only, so no
            // stale row ever reaches a mirror): warp 0 the top-k idx rows (meta context), warp 1 the header, warp 2 the
            // weight rows (both on the header context). Receivers poll every idx / weight unit (rule A) and learn nt from
            // the header, so the three may land in any order.
            IB_TRACE(24, epilogue_thread_idx == 0);
            if (epilogue_warp_idx <= 2 and lane_idx + 1 < ib_num_nodes) {
                const uint32_t nr = lane_idx, n = ib_node_of_slot(nr);
                const auto dst = ib_mirror(ib_parity, ib_my_node);
                const uint32_t cp = ib_cp(sym_buffer.rank_idx, n);
                if (epilogue_warp_idx == 0) {
                    SL_TR(0, 24, true);
                    if (num_tokens > 0)
                        make_gin_site(1, ctx_meta(nr, nr), nr, ib_xbase).put(dst + kIBOffIdx, ib_input_base + kIBOffIdx,
                            static_cast<uint64_t>(num_tokens) * kNumTopk * 8, cp);
                    SL_TR(0, 25, true);
                } else if (epilogue_warp_idx == 1) {
                    make_gin_site(2, ctx_mhdr(nr, hdr_orig(nr)), hdr_orig(nr), ib_xbase).put_value(
                        reinterpret_cast<uint64_t*>(dst + kIBOffHdr), (ib_epoch << 32) | num_tokens, cp);
                    SL_TR(0, 26, true);
                } else if (num_tokens > 0) {
                    make_gin_site(2, ctx_mhdr(nr, hdr_orig(nr)), hdr_orig(nr), ib_xbase).put(dst + kIBOffW, ib_input_base + kIBOffW,
                        static_cast<uint64_t>(num_tokens) * kNumTopk * 4, cp);
                }
            }
            __syncwarp();
        } else if (sm_idx == kIBMetaSM) {
            const auto stg = ib_mirror(ib_parity, ib_my_node);
            const auto in_idx = reinterpret_cast<const uint64_t*>(ib_input_base + kIBOffIdx);
            const auto in_w = reinterpret_cast<const uint32_t*>(ib_input_base + kIBOffW);
            constexpr uint32_t kSB = 4;  // independent loads first, then the stores
            const auto cim_cnt = reinterpret_cast<uint32_t*>(&shared_storage.smem_d);  // COUNTS_IN_META: per global expert
            const uint32_t cim_node_experts = kNumExpertsPerRank * ib_lsa;
            // Two copies of the staging loop: the counting branch (COUNTS_IN_META) costs ~6 us of meta latency even when
            // gated off at runtime (B300 trace, MegaSeek 128), so the plain loop carries no counting code at all
            const bool cim_count_here = kIBCountsInMeta and not kIBCimDispatch and ib_cim_on(num_tokens);
            // STAGE_NT / META_SPLIT (and LOCAL_REGPULL) are compiled in only for kernels with top-k >= DG_MEGA_MOE_IB_LEAN_MIN_TOPK
            // (default 8: MegaSeek 19 yes, Flash 6 no). Measured (B300 EP16, 5 rounds): MegaSeek 128 -5.1, 1024 -16.3; Flash 128
            // -1.9 but Flash 1024 +2.8 (the x pieces bind there), and a runtime size gate measured ~10 us slower at MegaSeek
            // 1024 than the always-on form, so the decision is per kernel (compile time).
            constexpr bool meta_lean = kIBLeanMeta;
            if (cim_count_here) {
            {
                for (uint32_t g = epilogue_thread_idx; g < kNumExperts; g += kNumEpilogueThreads)
                    cim_cnt[g] = 0;
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
                for (uint32_t i0 = 0; i0 < kIBT * kNumTopk; i0 += kNumEpilogueThreads * kSB) {
                    uint64_t vi[kSB];
                    uint32_t vw[kSB];
                    #pragma unroll
                    for (uint32_t u = 0; u < kSB; ++ u) {
                        const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                        vi[u] = i < num_tokens * kNumTopk ? __ldcg(reinterpret_cast<const unsigned long long*>(in_idx) + i) : kIBIdxSentinel;
                        vw[u] = i < num_tokens * kNumTopk ? __ldcg(in_w + i) : 0xffffffffu;
                    }
                    #pragma unroll
                    for (uint32_t u = 0; u < kSB; ++ u) {
                        const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                        if (i < kIBT * kNumTopk) {
                            reinterpret_cast<uint64_t*>(stg + kIBOffIdx)[i] = vi[u];
                            reinterpret_cast<uint32_t*>(stg + kIBOffW)[i] = vw[u];
                        }
                        if constexpr (kIBCountsInMeta and not kIBCimDispatch) {
                            const int e = static_cast<int>(static_cast<int64_t>(vi[u]));
                            if (i < num_tokens * kNumTopk and e >= 0 and static_cast<uint32_t>(e) / cim_node_experts != ib_my_node)
                                atomicAdd_block(cim_cnt + e, 1u);
                        }
                    }
                }
            } else {
                // DG_MEGA_MOE_IB_STAGE_NT=1 (Flash 128 latency: staging all kIBT rows took ~5.4 us of the meta's critical path):
                // rows >= num_tokens of this parity's staging slot already hold sentinels unless the previous same-parity
                // launch had more tokens, so only rows < max(num_tokens, previous count) are (re)written. The previous
                // count lives in the staging header slot's last 8 bytes (never put). The put still carries every row, so
                // every mirror row >= nt stays a sentinel for later launches with more tokens (rule A).
#if defined(DG_MEGA_MOE_IB_STAGE_NT) and DG_MEGA_MOE_IB_STAGE_NT
                auto* const stg_prev = reinterpret_cast<uint32_t*>(stg + kIBOffHdr + 120);
                const uint32_t stage_rows = meta_lean ? cute::max(num_tokens, cute::min(ptx::ld_volatile(stg_prev), static_cast<uint32_t>(kIBT)))
                                                      : static_cast<uint32_t>(kIBT);  // meta_lean is constexpr
#else
                const uint32_t stage_rows = static_cast<uint32_t>(kIBT);
#endif
#if defined(DG_MEGA_MOE_IB_STAGE_NOUNROLL) and DG_MEGA_MOE_IB_STAGE_NOUNROLL
                #pragma unroll 1
#endif
                for (uint32_t i0 = 0; i0 < stage_rows * kNumTopk; i0 += kNumEpilogueThreads * kSB) {
                    uint64_t vi[kSB];
                    uint32_t vw[kSB];
                    #pragma unroll
                    for (uint32_t u = 0; u < kSB; ++ u) {
                        const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                        vi[u] = i < num_tokens * kNumTopk ? __ldcg(reinterpret_cast<const unsigned long long*>(in_idx) + i) : kIBIdxSentinel;
                        vw[u] = i < num_tokens * kNumTopk ? __ldcg(in_w + i) : 0xffffffffu;
                    }
                    #pragma unroll
                    for (uint32_t u = 0; u < kSB; ++ u) {
                        const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                        if (i < stage_rows * kNumTopk) {
                            reinterpret_cast<uint64_t*>(stg + kIBOffIdx)[i] = vi[u];
                            reinterpret_cast<uint32_t*>(stg + kIBOffW)[i] = vw[u];
                        }
                    }
                }
            }
            if constexpr (kIBCountsInMeta and not kIBCimDispatch) {
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                for (uint32_t g = epilogue_thread_idx; ib_cim_on(num_tokens) and g < kNumExperts; g += kNumEpilogueThreads)
                    if (g / cim_node_experts != ib_my_node)
                        reinterpret_cast<uint32_t*>(stg + kIBOffCnt)[g] = cim_cnt[g];
            }
            if (epilogue_thread_idx == 0)
                *reinterpret_cast<uint64_t*>(stg + kIBOffHdr) = (ib_epoch << 32) | num_tokens;
            __threadfence();
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
#if defined(DG_MEGA_MOE_IB_STAGE_NT) and DG_MEGA_MOE_IB_STAGE_NT
            // after the barrier: every thread has read the previous count; the meta put does not cover these bytes
            if (kIBLeanMeta and epilogue_thread_idx == 0 and not cim_count_here)
                *reinterpret_cast<uint32_t*>(stg + kIBOffHdr + 120) = num_tokens;
#endif
            IB_TRACE(24, epilogue_thread_idx == 0);
            // DG_MEGA_MOE_IB_META_SPLIT=1 (with STAGE_NT): the meta goes as two concurrent puts from two warps: the idx rows
            // < num_tokens (meta context; the receiver's idx rows >= nt are sentinels: it re-arms every row it was sent),
            // and the weight region + header (header context). Drops the (kIBT - nt) * K * 8 idx bytes of sentinels from the
            // critical put (kIBT = 1920 on the dev server: MegaSeek 1024 sends 292 KB of idx instead of 155 KB).
#if defined(DG_MEGA_MOE_IB_META_SPLIT) and DG_MEGA_MOE_IB_META_SPLIT
            constexpr bool kMetaSplit = true;
            DG_STATIC_ASSERT(not kMetaSplit or (kIBHdrCtx and not kIBCountsInMeta and not kIBSenderTables), "META_SPLIT needs HDR_CTX, no CIM / ST");
#else
            constexpr bool kMetaSplit = false;
#endif
            DG_STATIC_ASSERT(not kIBSSTW or (kMetaSplit and meta_lean), "SST_W needs META_SPLIT + STAGE_NT on a lean-meta kernel (top-k >= LEAN_MIN_TOPK)");
            if (kMetaSplit and meta_lean and epilogue_warp_idx <= 1 and lane_idx + 1 < ib_num_nodes) {
                const uint32_t nr = lane_idx, n = ib_node_of_slot(nr);
                const auto dst = ib_mirror(ib_parity, ib_my_node);
                const uint32_t cp = ib_cp(sym_buffer.rank_idx, n);
                if (epilogue_warp_idx == 0) {
                    SL_TR(0, 24, true);
                    if (num_tokens > 0)
                        make_gin_site(1, ctx_meta(nr, nr), nr, ib_xbase).put(dst + kIBOffIdx, stg + kIBOffIdx,
                            static_cast<uint64_t>(num_tokens) * kNumTopk * 8, cp);
                    SL_TR(0, 25, true);
                } else {
                    const auto gin_h = make_gin_site(2, ctx_mhdr(nr, hdr_orig(nr)), hdr_orig(nr), ib_xbase);
                    if (kIBSSTW and ib_sst and not (kCGDrop & 8192)) {  // the weight rows went out at kernel start: header only
                        gin_h.put_value(reinterpret_cast<uint64_t*>(dst + kIBOffHdr), (ib_epoch << 32) | num_tokens, cp);
                    } else {
                    gin_h.put(dst + kIBOffW, stg + kIBOffW, static_cast<uint64_t>(kIBOffHdr + 8 - kIBOffW), cp);
#if not (defined(DG_MEGA_MOE_IB_NO_META_DOORBELL) and DG_MEGA_MOE_IB_NO_META_DOORBELL)
                    gin_h.put_value(reinterpret_cast<uint64_t*>(dst + kIBOffHdr), (ib_epoch << 32) | num_tokens, cp);
#endif
                    }
                    SL_TR(0, 26, true);
                }
            }
            if (not (kMetaSplit and meta_lean) and epilogue_warp_idx == 0 and lane_idx + 1 < ib_num_nodes) {
                const uint32_t n = ib_node_of_slot(lane_idx);
                SL_TR(0, 24, true);  // meta put: before issue (slot 24), after the put (25), after the doorbell (26)
                const auto gin_m = make_gin_site(1, ctx_meta(lane_idx, lane_idx), lane_idx, ib_xbase);
                gin_m.put(ib_mirror(ib_parity, ib_my_node) + kIBOffIdx, stg + kIBOffIdx,
                          static_cast<uint64_t>(kIBOffHdr + 8 - kIBOffIdx), ib_cp(sym_buffer.rank_idx, n));
                SL_TR(0, 25, true);
                if (kIBCountsInMeta and not kIBCimDispatch and ib_cim_on(num_tokens)) {
                    const uint64_t cb = static_cast<uint64_t>(cim_node_experts) * 4;
                    gin_m.put(ib_mirror(ib_parity, ib_my_node) + kIBOffCnt, stg + kIBOffCnt + static_cast<int64_t>(n) * static_cast<int64_t>(cb),
                              cb, ib_cp(sym_buffer.rank_idx, n));
                }
                // DG_MEGA_MOE_IB_NO_META_DOORBELL=1 drops it: the "last op on an idle context lands late" sightings were
                // root-caused to per-rank JIT compile skew (CORRECTNESS_LOG), and the extra put costs ~4.5 us on SM 0
#if defined(DG_MEGA_MOE_IB_NO_META_DOORBELL) and DG_MEGA_MOE_IB_NO_META_DOORBELL
                constexpr bool kMetaDoorbell = false;
#else
                constexpr bool kMetaDoorbell = true;
#endif
                if (kIBHdrCtx and kMetaDoorbell)  // doorbell (see the full-batch branch): identical header value, pushes the meta put through
                    gin_m.put_value(reinterpret_cast<uint64_t*>(ib_mirror(ib_parity, ib_my_node) + kIBOffHdr), (ib_epoch << 32) | num_tokens,
                                    ib_cp(sym_buffer.rank_idx, n));
                SL_TR(0, 26, true);
            }
            __syncwarp();
            if (kIBCimDispatch and ib_st_on(num_tokens)) {
                // CIM_DISPATCH: the staging loop did not count; the tables need per-expert counts for their prefix
                for (uint32_t g = epilogue_thread_idx; g < kNumExperts; g += kNumEpilogueThreads)
                    cim_cnt[g] = 0;
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                for (uint32_t i = epilogue_thread_idx; i < num_tokens * kNumTopk; i += kNumEpilogueThreads) {
                    const int e = static_cast<int>(static_cast<int64_t>(__ldcg(reinterpret_cast<const unsigned long long*>(in_idx) + i)));
                    if (e >= 0 and static_cast<uint32_t>(e) / cim_node_experts != ib_my_node)
                        atomicAdd_block(cim_cnt + e, 1u);
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
            if (ib_st_on(num_tokens)) {
                // Built AFTER the meta + counts are posted (the build cost ~30 us of meta latency when it came first)
                // Exclusive prefix over remote experts in global order (our node's experts count 0): each thread owns a
                // contiguous run of experts, warp inclusive sums, then warp bases
                auto st_cur = cim_cnt;  // in place: each thread reads its counts before overwriting them with cursors
                auto st_wsum = reinterpret_cast<uint32_t*>(stg + kIBOffHdr + 16);  // header slot bytes 16..127 are never put
                auto st_nstart = st_wsum + kNumEpilogueWarps;  // [nodes + 1] slice starts
                constexpr uint32_t kPer = math::constexpr_ceil_div(kNumExperts, kNumEpilogueThreads);
                const uint32_t g0 = epilogue_thread_idx * kPer;
                uint32_t mine = 0;
                #pragma unroll
                for (uint32_t u = 0; u < kPer; ++ u) {
                    const uint32_t g = g0 + u;
                    if (g < kNumExperts and g / cim_node_experts != ib_my_node)
                        mine += cim_cnt[g];
                }
                const uint32_t inc = math::warp_inclusive_sum(mine, lane_idx);
                if (lane_idx == 31)
                    st_wsum[epilogue_warp_idx] = inc;
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                uint32_t run = inc - mine;
                for (uint32_t w = 0; w < epilogue_warp_idx; ++ w)
                    run += st_wsum[w];
                #pragma unroll
                for (uint32_t u = 0; u < kPer; ++ u) {
                    const uint32_t g = g0 + u;
                    if (g < kNumExperts) {
                        const uint32_t c = g / cim_node_experts != ib_my_node ? cim_cnt[g] : 0u;
                        st_cur[g] = run;
                        run += c;
                    }
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_thread_idx <= ib_num_nodes) {
                    uint32_t total = 0;
                    for (uint32_t w = 0; w < kNumEpilogueWarps; ++ w)
                        total += st_wsum[w];
                    const uint32_t g = epilogue_thread_idx * cim_node_experts;
                    st_nstart[epilogue_thread_idx] = g < kNumExperts ? st_cur[g] : total;
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                // Pass 2: entries in (roughly) token order; slots from shared-memory cursors
                const auto st_tab = reinterpret_cast<uint32_t*>(stg + kIBOffSTab);
                for (uint32_t i = epilogue_thread_idx; i < num_tokens * kNumTopk; i += kNumEpilogueThreads) {
                    const int e = static_cast<int>(static_cast<int64_t>(__ldcg(reinterpret_cast<const unsigned long long*>(in_idx) + i)));
                    if (e >= 0 and static_cast<uint32_t>(e) / cim_node_experts != ib_my_node)
                        st_tab[atomicAdd_block(st_cur + e, 1u)] = i;
                }
            }
            if (ib_st_on(num_tokens)) {
                __threadfence();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and lane_idx + 1 < ib_num_nodes) {
                    const uint32_t n = ib_node_of_slot(lane_idx);
                    const auto gin_m = make_gin_site(1, ctx_meta(lane_idx, lane_idx), lane_idx, ib_xbase);
                    const auto st_nstart = reinterpret_cast<const uint32_t*>(stg + kIBOffHdr + 16) + kNumEpilogueWarps;
                    const uint32_t a = st_nstart[n], b = st_nstart[n + 1];
                    if (b > a)
                        gin_m.put(ib_mirror(ib_parity, ib_my_node) + kIBOffSTab, stg + kIBOffSTab + static_cast<int64_t>(a) * 4,
                                  static_cast<uint64_t>(b - a) * 4, ib_cp(sym_buffer.rank_idx, n));
                    gin_m.put_value(reinterpret_cast<uint64_t*>(ib_mirror(ib_parity, ib_my_node) + kIBOffHdr), (ib_epoch << 32) | num_tokens,
                                    ib_cp(sym_buffer.rank_idx, n));  // doorbell for the table put (identical header value)
                }
                __syncwarp();
            }
            // The bulk x / SF puts are posted only after the meta, so the meta is not queued behind them on the wire
            if (epilogue_thread_idx == 0 and (kIBMetaFirstAll or not kIBFront))
                { uint64_t t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); workspace.signals->ib_o_meta_posted_t = t_; }
                ptx::st_rel(&workspace.signals->ib_o_meta_posted, ib_epoch);  // GPU scope: only this GPU's x posters read it (sys scope cost ~10 us)
        }
#if defined(DG_MEGA_MOE_IB_DEBUG_D0) and DG_MEGA_MOE_IB_DEBUG_D0
        if (sm_idx == kNumSMs - 1 and epilogue_thread_idx == 0 and ib_epoch <= 2)
            printf("OPUS_D0 rank=%u epoch=%llu meta posted\n", sym_buffer.rank_idx, (unsigned long long) ib_epoch);
#endif
        {
            DG_STATIC_ASSERT(kIBXPiece == kIBChunk or not kIBWaveSort, "X_PIECE != 32 excludes WAVE_SORT");
            const uint32_t num_x_chunks = (num_tokens + kIBXPiece - 1) / kIBXPiece;
            const uint32_t num_pieces = 1 + num_x_chunks;  // SF, x chunks
            const uint32_t q0 = kIBFront ? sm_idx - 1 : kNumSMs - 2 - sm_idx;
            const bool q_ok = kIBFront ? sm_idx >= 1 : sm_idx + 2 <= kNumSMs;
            for (uint32_t q = q0; q_ok and q < (ib_num_nodes - 1) * num_pieces; q += (kIBFront ? kNumSMs - 1 : kNumSMs)) {
                const uint32_t q_nr = q / num_pieces, q_piece = q % num_pieces, q_n = ib_node_of_slot(q_nr);
                // WAVE_SORT: x chunk = sorted positions [32c, 32c + 32): this SM computes the whole stable order (cheap,
                // every chunk's SM does it independently) and stages its chunk's rows contiguously
                const auto x_stage = ib_pcomb(ib_my_node) + static_cast<int64_t>(q_nr) * kIBT * kIBTokBytes;  // unused locally
                const bool ws_stage = ib_ws_on(num_tokens) and q_piece > 0 and num_tokens > 0;
                const bool ord_stage = ib_ord_on(num_tokens) and q_piece > 0 and num_tokens > 0;
                if (ws_stage or ord_stage) {
                    const uint32_t c = q_piece - 1, c0 = c * kIBChunk, c1 = cute::min(c0 + kIBChunk, num_tokens);
                    if (ord_stage) {
                        // ORD: wait for the published order, take this chunk's rows (row -> token)
                        if constexpr (kIBMSO > 0) {
                            if (epilogue_thread_idx < kIBMSO)
                                while (ptx::ld_acq_gpu(ib_xo_flag + 216 + epilogue_thread_idx) != ib_epoch);
                        } else if (epilogue_thread_idx == 0)
                            IB_POLL(ptx::ld_acq_gpu(ib_xo_flag + 240 + q_nr) != ib_epoch, (void) 0, "ord_flag", q, 0);
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_thread_idx < c1 - c0)
                            ib_inv[epilogue_thread_idx] = __ldcg(ib_ord_inv(q_nr) + c0 + epilogue_thread_idx);
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    } else {
                    const auto keys = ib_keys(0);
                    ib_build_keys(keys, num_tokens, q_n, false, epilogue_thread_idx);
                    uint32_t* const pos_g = reinterpret_cast<uint32_t*>(ib_mirror(ib_parity, ib_my_node)) + static_cast<int64_t>(q_nr) * kIBT;
                    ib_stable_sort(keys, num_tokens, epilogue_thread_idx, [&](const uint32_t& t, const uint32_t& pos) {
                        if (pos >= c0 and pos < c1) ib_inv[pos - c0] = kIBWSMode >= 2 ? pos : t;
                        if (kIBSSTWave and c == 0) pos_g[t] = kIBWSMode >= 2 ? t : pos;
                    });
                    if (kIBSSTWave and c == 0) {
                        __threadfence();
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_thread_idx == 0)
                            ptx::st_rel(ib_xo_flag + 240 + q_nr, ib_epoch);
                        SL_TR(0, 40, epilogue_thread_idx == 0);  // wave order published (slowest node)
                    }
                    }
                    // Row copy with kB independent 16-byte loads in flight per thread
                    constexpr uint32_t kRowUnits = kIBTokBytes / 16, kB = 10;
                    const uint32_t num_units = (c1 - c0) * kRowUnits;
                    for (uint32_t u0 = epilogue_thread_idx; u0 < num_units; u0 += kNumEpilogueThreads * kB) {
                        uint4 v[kB];
                        #pragma unroll
                        for (uint32_t b = 0; b < kB; ++ b) {
                            const uint32_t u = u0 + b * kNumEpilogueThreads;
                            if (u < num_units)
                                v[b] = __ldcg(reinterpret_cast<const uint4*>(ib_input_base + static_cast<int64_t>(ib_inv[u / kRowUnits]) * kIBTokBytes) + u % kRowUnits);
                        }
                        #pragma unroll
                        for (uint32_t b = 0; b < kB; ++ b) {
                            const uint32_t u = u0 + b * kNumEpilogueThreads;
                            if (u < num_units)
                                reinterpret_cast<uint4*>(x_stage + static_cast<int64_t>(c0 + u / kRowUnits) * kIBTokBytes)[u % kRowUnits] = v[b];
                        }
                    }
                    if (kIBOrdGFence and ord_stage) __threadfence();
                    else __threadfence_system();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
                // STREAM_COMBINE: the SM posting node n's SF piece also computes our records' return order from n
                const auto sc_sort = [&]() {
                    const auto keys = ib_keys(1);
                    ib_build_keys(keys, num_tokens, q_n, true, epilogue_thread_idx);
                    ib_stable_sort(keys, num_tokens, epilogue_thread_idx, [&](const uint32_t& t, const uint32_t& pos) {
                        ib_tab[kIBTabPos2 + q_nr * kIBTabT + t] = pos;
                    });
                };
                if (not kIBSFFirst and ib_sc_on(num_tokens) and q_piece == 0 and num_tokens > 0)
                    sc_sort();
                if (epilogue_warp_idx == 0 and lane_idx == 0 and num_tokens > 0) {
                    // Meta first (x / SF posted only after the meta put): opt-in only; on GB200 it turns into ms-scale
                    // stalls (EP32 with the proxy D1, EP16 with the host scan; mechanism not identified)
#if defined(DG_MEGA_MOE_IB_META_FIRST_ALL) and DG_MEGA_MOE_IB_META_FIRST_ALL
                    constexpr bool kMetaFirst = true;
#else
                    constexpr bool kMetaFirst = false;  // measured harmful with the host scan (GB200: ms-scale stalls)
#endif
                    while (kMetaFirst and ptx::ld_acq_gpu(&workspace.signals->ib_o_meta_posted) < ib_epoch);
#if defined(DG_MEGA_MOE_IB_SST_XHOLD) and DG_MEGA_MOE_IB_SST_XHOLD
                    if (ib_sst)
                        IB_POLL(ptx::ld_acq_gpu(ib_xo_flag + 255) != ib_epoch, (void) 0, "sst_xhold", q, 0);
#endif
                    if constexpr (kIBNBXHold) {
                        if (ib_sst and q_piece > 0 and not ib_ord_on(num_tokens))
                            IB_POLL(ptx::ld_acq_gpu(ib_xo_flag + 244 + q_nr) != ib_epoch, (void) 0, "nb_xhold", q, 0);
                    }
#if defined(DG_MEGA_MOE_IB_XDELAY_NS) and DG_MEGA_MOE_IB_XDELAY_NS
                    // DG_MEGA_MOE_IB_XDELAY_NS=N (with META_FIRST_ALL): x / SF pieces wait N ns after the meta was posted, so the
                    // meta is not interleaved with megabytes of x on the NIC (hypothesis test: counts @1024 land ~50 us late)
                    if constexpr (kMetaFirst) {
                        const uint64_t t_meta = ptx::ld_volatile(&workspace.signals->ib_o_meta_posted_t);
                        uint64_t t_; do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); } while (t_ < t_meta + DG_MEGA_MOE_IB_XDELAY_NS);
                    }
#endif
                    const uint32_t n = ib_node_of_slot(q / num_pieces), piece = q % num_pieces;
                    int64_t off = 0, bytes = 0;
                    if (piece == 0) {
                        off = kIBOffSF, bytes = static_cast<int64_t>(num_tokens) * kIBSFBytes;
                    } else {
                        const uint32_t t0 = (piece - 1) * kIBXPiece, t1 = cute::min(t0 + kIBXPiece, num_tokens);
                        off = static_cast<int64_t>(t0) * kIBTokBytes, bytes = static_cast<int64_t>(t1 - t0) * kIBTokBytes;
                    }
                    const auto src = ((ib_ws_on(num_tokens) and kIBWSMode != 3) or ib_ord_on(num_tokens)) and piece > 0 ? x_stage + off : ib_input_base + off;
                    constexpr uint32_t kXOCtx = kIBHdrCtx ? cute::max(1u, static_cast<uint32_t>(DG_MEGA_MOE_IB_CONTEXTS) - 2u) : 1u;
                    const uint32_t xo_nctx = kIBHdrCtx ? ib_xnctx() : kXOCtx;
                    // Size gate (DG_MEGA_MOE_IB_XO_MIN top-k entries per rank, default 8192): MegaSeek 1024 -5.3 / -5.5 on two
                    // servers, +0.5..+1.0 at 128 and ~0 on Flash 1024 (B300), so small batches keep the free-for-all issue
#ifndef DG_MEGA_MOE_IB_XO_MIN
#define DG_MEGA_MOE_IB_XO_MIN 8192
#endif
                    // and only with one remote node: the order chain spans every remote node's pieces (q), so at EP32 (3 remote
                    // nodes) it tripled the serialized put issue (MegaSeek 1024 +92 us on top of TS2, B300 and4)
                    const bool xo_on = kIBXOrder and num_tokens * kNumTopk >= DG_MEGA_MOE_IB_XO_MIN and ib_R == 1;
                    if (xo_on and q >= xo_nctx and q - xo_nctx < 256)  // previous piece on this context issued first
                        IB_POLL(ptx::ld_acq_gpu(ib_xo_flag + (q - xo_nctx)) != ib_epoch, (void) 0, "x_order", q, 0);
                    // WS_POST: this chunk is staged (system fence above); the node's poster thread issues it in wave order
                    const bool ws_posted = ((kIBWSPost > 0 and ib_ws_on(num_tokens)) or ib_ord_on(num_tokens)) and piece > 0;
                    if (not ws_posted)
                    make_gin_site(4, ctx_piece(q, piece_orig(q)), piece_orig(q), ib_xbase).put(ib_mirror(ib_parity, ib_my_node) + off, src, static_cast<uint64_t>(bytes),
                                        ib_cp(sym_buffer.rank_idx, n));
                    if ((xo_on or ws_posted) and q < 256)
                        ptx::st_rel(ib_xo_flag + q, ib_epoch);
                    SL_TR(0, 39, ws_posted);  // WS_POST: last chunk staged
                    SL_TR(0, 27, q == 0);
                    SL_TR(0, 28, not ws_posted);
                    IB_TRACE_MAX(7, not ws_posted);   // x/SF piece put issued, slowest poster
                    IB_TRACE_MAX(8, q == 0); // first piece (SF) issued
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
                    if (not ws_posted) {
                        uint64_t t_;
                        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_));
                        atomicMin(reinterpret_cast<unsigned long long*>(&workspace.signals->ib_trace[46]), static_cast<unsigned long long>(t_));
                        atomicMax(reinterpret_cast<unsigned long long*>(&workspace.signals->ib_trace[47]), static_cast<unsigned long long>(t_));
                    }
#endif
                }
                __syncwarp();
                if (kIBSFFirst and ib_sc_on(num_tokens) and q_piece == 0 and num_tokens > 0)
                    sc_sort();
            }
        }
        if constexpr (kIBWSPost > 0 or kIBOrd > 0) {
            DG_IB_OUTLINE_BEGIN
            if (kIBHeavy and ((kIBWSPost > 0 and ib_ws_on(num_tokens) and sm_idx + 3 + (kNumRanks - ib_lsa) == kNumSMs) or
                 (ib_ord_on(num_tokens) and sm_idx + 4 + (kNumRanks - ib_lsa) == kNumSMs)) and epilogue_warp_idx == 0 and
                lane_idx + 1 < ib_num_nodes and num_tokens > 0) {
                const uint32_t nr = lane_idx, n = ib_node_of_slot(nr);
                const uint32_t num_x_chunks = (num_tokens + kIBChunk - 1) / kIBChunk, num_pieces = 1 + num_x_chunks;
                const auto x_stage = ib_pcomb(ib_my_node) + static_cast<int64_t>(nr) * kIBT * kIBTokBytes;
                const uint32_t orig = 2 * ib_R + nr % ib_xnctx();
                const auto gin_x = make_gin_site(4, ctx_piece(nr, orig), orig, ib_xbase);
                constexpr uint32_t kPC = (kIBOrd > 0 ? kIBOrd : kIBWSPost) / kIBChunk;  // chunks per put
                for (uint32_t c0 = 0; c0 < num_x_chunks; c0 += kPC) {
                    const uint32_t c1 = cute::min(c0 + kPC, num_x_chunks);
                    for (uint32_t c = c0; c < c1; ++ c) {
                        const uint32_t q = nr * num_pieces + 1 + c;
                        if (q < 256)
                            IB_POLL(ptx::ld_acq_gpu(ib_xo_flag + q) != ib_epoch, (void) 0, "ws_post", nr, c);
                    }
                    const uint32_t t0 = c0 * kIBChunk, t1 = cute::min(c1 * kIBChunk, num_tokens);
                    IB_PUT(gin_x, orig, ib_mirror(ib_parity, ib_my_node) + static_cast<int64_t>(t0) * kIBTokBytes, x_stage + static_cast<int64_t>(t0) * kIBTokBytes,
                              static_cast<uint64_t>(t1 - t0) * kIBTokBytes, ib_cp(sym_buffer.rank_idx, n));
                    SL_TR(0, 28, true);
                    SL_TR(0, 38, c0 == 0);  // first wave piece issued (slowest node)
                    IB_TRACE_MAX(7, true);
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
                    {
                        uint64_t t_;
                        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_));
                        atomicMin(reinterpret_cast<unsigned long long*>(&workspace.signals->ib_trace[46]), static_cast<unsigned long long>(t_));
                        atomicMax(reinterpret_cast<unsigned long long*>(&workspace.signals->ib_trace[47]), static_cast<unsigned long long>(t_));
                    }
#endif
                }
            }
            __syncwarp();
            DG_IB_OUTLINE_END
        }
        IB_TRACE(0, sm_idx == kIBMetaSM and epilogue_thread_idx == 0);
        // D1 (host scan; v1 had the proxy do this and push tables over NVLink, measured ~20 us slower): every GPU scans,
        // for each remote source r, r's mirror on r's counterpart in this node (over NVLink, or locally when that is
        // us) and builds its OWN tables for its own experts exactly like the upstream receiver-side state: per-source
        // counts, slot entries (local stores), then the upstream count-arrival add (kNumSMs << 32 | count). One SM's
        // epilogue warps per remote source (the last SMs; idle until the first routed task, which needs these counts).
        {
            const uint32_t num_remote = kNumRanks - ib_lsa;
            if (sm_idx + num_remote * ib_P >= kNumSMs) {
                const uint32_t k = [&]() {
                    const uint32_t sm_rev = kNumSMs - 1 - sm_idx;
                    if (ib_sst and not (kCGDrop & 2048)) {
                        // SST: our own counterpart sources (the only ones still scanned) on the last R SMs, the others below
                        const uint32_t myj = sym_buffer.rank_idx % ib_lsa, lm1 = cute::max(1u, ib_lsa - 1);
                        if (sm_rev < ib_R)
                            return sm_rev * ib_lsa + myj;
                        const uint32_t q = sm_rev - ib_R, jj = q % lm1;
                        return (q / lm1) * ib_lsa + (jj < myj ? jj : jj + 1);
                    }
                    return sm_rev / ib_P;   // k-th remote source
                }();
                const uint32_t scan_j = (kNumSMs - 1 - sm_idx) % ib_P;  // token slice of it (SCAN_SLICES)
                const uint32_t r = ib_node_of_slot(k / ib_lsa) * ib_lsa + k % ib_lsa;
                const uint32_t proxy = ib_cp(r, ib_my_node);
                const auto mir = sym_buffer.map(ib_mirror(ib_parity, r / ib_lsa), proxy);
                const auto hdr = reinterpret_cast<const uint64_t*>(mir + kIBOffHdr);
                // COUNTS_EARLY probe: r's counts for our experts, in r's counterpart's mirror (our node-local rank's slice)
                const auto cnt_early = reinterpret_cast<const uint32_t*>(mir + kIBOffCnt) + (sym_buffer.rank_idx % ib_lsa) * kNumExpertsPerRank;
                if constexpr (kIBCountsEarly) {
                    if (epilogue_thread_idx == 0) {
                        comm::wait_until([&]() {
                            for (uint32_t e = 0; e < kNumExpertsPerRank; ++ e)
                                if (ptx::ld_volatile(cnt_early + e) == 0xffffffffu) { __nanosleep(100); return false; }
                            return true;
                        }, [&]() { printf("DeepGEMM IB early-count timeout: rank=%u, source=%u, proxy=%u, epoch=%llu\n",
                                          sym_buffer.rank_idx, r, proxy, (unsigned long long) ib_epoch); });
                        IB_TRACE_MAX(38, true);
                    }
                }
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE and defined(DG_MEGA_MOE_IB_PROBE) and DG_MEGA_MOE_IB_PROBE
                // DG_MEGA_MOE_IB_PROBE=S (trace builds only; start-side latency study): on the scan SM of our own counterpart
                // source (its mirror is local memory), one thread of epilogue warp 2 sends 4 put_value probes to that
                // counterpart at 0 / S / 2S / 3S ns after it got here (x4 for >= 512 tokens), each carrying the sender's
                // %globaltimer (48 bits) + an epoch tag; one thread of warp 1 polls the 4 words the counterpart sends us and
                // records t_seen - t_sent (one-way latency + clock offset; the reverse direction has the opposite offset).
                // DG_MEGA_MOE_IB_PROBE_CTX: 0 header context, 1 meta context, 2 first x context. The poller leaves when the
                // source's header is visible, so the scan is not delayed (later probes are then reported as missing).
                if (proxy == sym_buffer.rank_idx and scan_j == 0) {
                    const uint32_t p_nr = k / ib_lsa, p_n = r / ib_lsa;
                    constexpr uint64_t kPM = (1ull << 48) - 1;
                    uint64_t* const pr = tl_tseen + 128;  // free part of the tail carve-out
                    const uint64_t p_tag = (ib_epoch & 0xffffull) << 48;
                    if (epilogue_warp_idx == 2 and lane_idx == 0) {
#ifndef DG_MEGA_MOE_IB_PROBE_CTX
#define DG_MEGA_MOE_IB_PROBE_CTX 0
#endif
                        const uint32_t p_ctx = DG_MEGA_MOE_IB_PROBE_CTX == 0 ? hdr_orig(p_nr) : (DG_MEGA_MOE_IB_PROBE_CTX == 1 ? p_nr : piece_orig(0));
                        const auto p_gin = make_gin(p_ctx);
                        const uint64_t p_step = static_cast<uint64_t>(DG_MEGA_MOE_IB_PROBE) * (num_tokens >= 512 ? 4u : 1u);
                        uint64_t p_t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(p_t0));
                        auto* const p_dst = reinterpret_cast<uint64_t*>(ib_mirror(ib_parity, ib_my_node) + kIBOffHdr + 64);
                        for (uint32_t i = 0; i < 4; ++ i) {
                            uint64_t t_; bool p_hdr = false;
                            do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); p_hdr = (ptx::ld_volatile(hdr) >> 32) == (ib_epoch & 0xffffffffull); } while (t_ < p_t0 + i * p_step and not p_hdr);
                            if (p_hdr and i > 0) { pr[16 + p_nr * 8 + i] = 0, pr[16 + p_nr * 8 + 4 + i] = 0; continue; }
                            p_gin.put_value(p_dst + i, p_tag | (t_ & kPM), ib_cp(sym_buffer.rank_idx, p_n));
                            uint64_t t1_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1_));
                            pr[16 + p_nr * 8 + i] = t_, pr[16 + p_nr * 8 + 4 + i] = t1_;
                        }
                    }
                    if (epilogue_warp_idx == 1 and lane_idx == 0) {
                        auto* const p_src = reinterpret_cast<uint64_t*>(ib_mirror(ib_parity, p_n) + kIBOffHdr + 64);
                        uint64_t p_t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(p_t0));
                        for (uint32_t i = 0; i < 4; ++ i)
                            pr[p_nr * 4 + i] = 0x8000000000000000ull;
                        uint32_t seen = 0;
                        while (seen != 15u) {
                            for (uint32_t i = 0; i < 4; ++ i) {
                                if ((seen >> i) & 1u) continue;
                                const uint64_t v = ptx::ld_volatile(p_src + i);
                                if ((v & ~kPM) == p_tag and v != 0) {
                                    uint64_t t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_));
                                    pr[p_nr * 4 + i] = static_cast<uint64_t>(static_cast<int64_t>(t_ & kPM) - static_cast<int64_t>(v & kPM));
                                    seen |= 1u << i;
                                }
                            }
                            const uint64_t h = ptx::ld_volatile(hdr);
                            uint64_t t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_));
                            if ((h >> 32) == (ib_epoch & 0xffffffffull)) { pr[48 + p_nr] = t_; break; }
                            if (t_ > p_t0 + 2000000ull) break;
                        }
                    }
                    __syncwarp();
                }
#endif
#if defined(DG_MEGA_MOE_IB_RX_FN) and DG_MEGA_MOE_IB_RX_FN
                constexpr bool kIBRxFn = true;
#else
                constexpr bool kIBRxFn = false;
#endif
                if (kIBRxFn and kIBHeavy and ib_sst and not (kCGDrop & 4)) {
                    const auto rc0 = workspace.get_expert_recv_count_ptr(r, 0);
                    const auto rs0 = workspace.get_expert_recv_count_sum_ptr(0);
                    const auto tk0 = workspace.get_src_token_topk_idx_ptr(0, r, 0);
                    unsigned long long* tr_a = nullptr; unsigned long long* tr_b = nullptr;
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
                    tr_a = reinterpret_cast<unsigned long long*>(ib_sl_tr + 31);
                    tr_b = reinterpret_cast<unsigned long long*>(workspace.signals->ib_trace + 26);
#endif
                    ib_sst_receive_fn<kNumExpertsPerRank, kNumEpilogueThreads, kEpilogueFullBarrierIdx>(
                        reinterpret_cast<const uint32_t*>(mir + kIBOffSTab) - ib_lsa * kNumExpertsPerRank, reinterpret_cast<const uint32_t*>(mir + kIBOffSTab),
                        reinterpret_cast<uint32_t*>(&shared_storage.smem_d), ib_lsa * kNumExpertsPerRank, sym_buffer.rank_idx % ib_lsa,
                        epilogue_thread_idx, lane_idx,
                        rc0, reinterpret_cast<const uint8_t*>(workspace.get_expert_recv_count_ptr(r, 1)) - reinterpret_cast<const uint8_t*>(rc0),
                        reinterpret_cast<const uint64_t*>(rs0), reinterpret_cast<const uint8_t*>(workspace.get_expert_recv_count_sum_ptr(1)) - reinterpret_cast<const uint8_t*>(rs0),
                        tk0, reinterpret_cast<const uint8_t*>(workspace.get_src_token_topk_idx_ptr(1, r, 0)) - reinterpret_cast<const uint8_t*>(tk0),
                        static_cast<uint64_t>(kNumSMs) << 32, &workspace.signals->ib_o_tab_ready[k], ib_epoch, tr_a, tr_b);
                    IB_TRACE_MAX(32, epilogue_thread_idx == 0);  // tables done
                }
                if (not kIBRxFn and kIBHeavy and ib_sst and not (kCGDrop & 4)) {
                    DG_IB_OUTLINE_BEGIN
                    // SST: r's counts for this node ([node-local rank][local expert], ending at kIBOffSTab) and its entries
                    // (from kIBOffSTab) arrive in one early put. Publish our counts, copy our slice into the slot table
                    const uint32_t sst_n = ib_lsa * kNumExpertsPerRank;
                    const auto cnt_all = reinterpret_cast<const uint32_t*>(mir + kIBOffSTab) - sst_n;
                    const uint32_t myj = sym_buffer.rank_idx % ib_lsa;
                    auto s_cnt = reinterpret_cast<uint32_t*>(&shared_storage.smem_d);
                    auto s_off = s_cnt + kNumExpertsPerRank + 1;
                    auto s_base = s_off + kNumExpertsPerRank;
                    if (epilogue_thread_idx == 0)
                        *s_base = 0;
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    uint32_t below = 0;
                    for (uint32_t q = epilogue_thread_idx; not (kCGDrop & 64) and q < sst_n; q += kNumEpilogueThreads) {
                        uint32_t v = ptx::ld_volatile(cnt_all + q);
                        IB_POLL(v == 0xffffffffu, v = ptx::ld_volatile(cnt_all + q), "sst_cnt", r, q);
                        if (q / kNumExpertsPerRank == myj)
                            s_cnt[q % kNumExpertsPerRank] = v;
                        else if (q / kNumExpertsPerRank < myj)
                            below += v;
                    }
                    below = __reduce_add_sync(0xffffffff, below);
                    if (lane_idx == 0 and below)
                        atomicAdd_block(s_base, below);
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    SL_TR(0, 31, epilogue_thread_idx == 0);  // SST counts landed, slowest source
                    if (epilogue_thread_idx == 0) {  // per-expert offsets inside our slice (<= 256 experts per rank)
                        uint32_t o = 0;
                        for (uint32_t e = 0; e < kNumExpertsPerRank; ++ e)
                            s_off[e] = o, o += s_cnt[e];
                    }
                    for (uint32_t i = epilogue_thread_idx; not (kCGDrop & 128) and i < kNumExpertsPerRank; i += kNumEpilogueThreads) {
                        *workspace.get_expert_recv_count_ptr(r, i) = s_cnt[i];
                        __threadfence();
                        ptx::atomic_add(workspace.get_expert_recv_count_sum_ptr(i),
                                        (static_cast<uint64_t>(kNumSMs) << 32) | static_cast<uint64_t>(s_cnt[i]));
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    IB_TRACE_MAX(26, epilogue_thread_idx == 0);  // counts published, slowest source
                    const auto st_tab = reinterpret_cast<const uint32_t*>(mir + kIBOffSTab) + *s_base;
                    const uint32_t m = s_off[kNumExpertsPerRank - 1] + s_cnt[kNumExpertsPerRank - 1];
                    for (uint32_t q = epilogue_thread_idx; not (kCGDrop & 256) and q < m; q += kNumEpilogueThreads) {
                        uint32_t lo = 0, hi = kNumExpertsPerRank;  // last e with s_off[e] <= q
                        while (hi - lo > 1) { const uint32_t mid = (lo + hi) / 2; if (s_off[mid] <= q) lo = mid; else hi = mid; }
                        uint32_t v = ptx::ld_volatile(st_tab + q);
                        IB_POLL(v == 0xffffffffu, v = ptx::ld_volatile(st_tab + q), "sst_tab", r, q);
                        *workspace.get_src_token_topk_idx_ptr(lo, r, q - s_off[lo]) = v;
                    }
                    __threadfence();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_thread_idx == 0)
                        ptx::st_rel_sys(&workspace.signals->ib_o_tab_ready[k], ib_epoch);
                    IB_TRACE_MAX(32, epilogue_thread_idx == 0);  // tables done
                    DG_IB_OUTLINE_END
                }
                // SST: only r's proxy goes on (header, rule-A polls of idx / weights, return order); legacy path: everyone
                if (not (ib_sst and not (kCGDrop & 4096) and proxy != sym_buffer.rank_idx)) {
                // One thread polls the (NVLink-remote) header with a pause between polls: hundreds of threads spinning
                // on a proxy's HBM while its NIC writes there slows the NIC down drastically on GB200
                auto nt_smem = reinterpret_cast<uint32_t*>(&shared_storage.smem_d) + kNumExpertsPerRank;
                if (epilogue_thread_idx == 0) {
                    uint64_t h = 0;
                    comm::wait_until([&]() { h = ptx::ld_volatile(hdr); if (ib_oracle_now and h != ~0ull) return true; if ((h >> 32) != (ib_epoch & 0xffffffffull)) __nanosleep(200); return (h >> 32) == (ib_epoch & 0xffffffffull); }, [&]() {
                        printf("DeepGEMM IB header timeout: rank=%u, source=%u, proxy=%u, header=%llx, epoch=%llu\n",
                               sym_buffer.rank_idx, r, proxy, (unsigned long long) h, (unsigned long long) ib_epoch);
                    });
                    *nt_smem = static_cast<uint32_t>(h & 0xffffffffull);
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                const uint32_t nt = *nt_smem;
                IB_TRACE(20, epilogue_thread_idx == 0 and sm_idx == kNumSMs - 1);
                IB_TRACE_MAX(25, epilogue_thread_idx == 0);  // remote header seen, slowest scanning SM
                const auto idx = reinterpret_cast<const uint64_t*>(mir + kIBOffIdx);
                const auto wts = reinterpret_cast<const uint32_t*>(mir + kIBOffW);
                auto cnt = reinterpret_cast<uint32_t*>(&shared_storage.smem_d);
                auto cnt2 = cnt + kNumExpertsPerRank + 1;  // SPLIT_SCAN slot counters (after nt_smem)
                for (uint32_t i = epilogue_thread_idx; i < kNumExpertsPerRank; i += kNumEpilogueThreads)
                    cnt[i] = 0, cnt2[i] = 0;
                if (proxy == sym_buffer.rank_idx and epilogue_thread_idx == 0 and scan_j == 0)
                    workspace.signals->ib_o_nt[r / ib_lsa] = nt;  // the proxy's forward-combine and re-arm need it
                if (kIBScanSlices > 1 and ib_P > 1) {
                    const uint32_t t0 = static_cast<uint32_t>(static_cast<uint64_t>(nt) * scan_j / ib_P);
                    const uint32_t t1 = static_cast<uint32_t>(static_cast<uint64_t>(nt) * (scan_j + 1) / ib_P);
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);  // cnt / cnt2 zeroed above
                    SL_TR(scan_j, 0, epilogue_thread_idx == 0);
                    // SLICE_LIST: matches of pass A in smem_d after the first 1 KB (counters), capacity kSLCap
                    constexpr uint32_t kSLCap = (static_cast<uint32_t>(sizeof(SharedStorage::smem_d)) - 1024u) / 4u - 1u;
                    uint32_t* const sl_n = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&shared_storage.smem_d) + 1024);
                    uint32_t* const sl_list = sl_n + 1;
                    if constexpr (kIBSliceList) {
                        if (epilogue_thread_idx == 0) *sl_n = 0;
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    }
                    // Pass A: count this slice's entries for our experts. kSLB loads in flight per thread (NVLink latency-bound:
                    // one dependent load per iteration made the slices slower than the single-SM scan, B300 trace)
                    constexpr uint32_t kSLB = 8;
                    const uint32_t e0 = t0 * kNumTopk, e1 = t1 * kNumTopk;
                    for (uint32_t i0 = e0; i0 < e1; i0 += kNumEpilogueThreads * kSLB) {
                        uint64_t v[kSLB];
                        #pragma unroll
                        for (uint32_t u = 0; u < kSLB; ++ u) {
                            const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                            v[u] = i < e1 ? ptx::ld_volatile(idx + i) : 0ull;
                        }
                        #pragma unroll
                        for (uint32_t u = 0; u < kSLB; ++ u) {
                            const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                            if (i < e1) {
                                SL_ADD(0, v[u] == kIBIdxSentinel);
                                IB_POLL(v[u] == kIBIdxSentinel, v[u] = ptx::ld_volatile(idx + i), "sl_idx", r, i);
                                const int e = static_cast<int>(static_cast<int64_t>(v[u]));
                                if (e >= 0 and static_cast<uint32_t>(e) / kNumExpertsPerRank == sym_buffer.rank_idx) {
                                    atomicAdd_block(cnt + static_cast<uint32_t>(e) % kNumExpertsPerRank, 1u);
                                    if constexpr (kIBSliceList) {  // (entry, local expert) for pass B: no NVLink reload
                                        const uint32_t q = atomicAdd_block(sl_n, 1u);
                                        if (q < kSLCap)
                                            sl_list[q] = (i << 8) | (static_cast<uint32_t>(e) % kNumExpertsPerRank);
                                    }
                                }
                            }
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    uint32_t* const my_cnt = ib_sl_cnt + (k * 4 + scan_j) * 256;
                    for (uint32_t i = epilogue_thread_idx; i < kNumExpertsPerRank; i += kNumEpilogueThreads)
                        my_cnt[i] = cnt[i];
                    __threadfence();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_thread_idx == 0)
                        ptx::st_rel(ib_sl_cflag + k * 4 + scan_j, ib_epoch);
                    SL_TR(scan_j, 1, epilogue_thread_idx == 0);
                    // Partial count publish (count-arrival protocol: the shares of the P slices sum to kNumSMs)
                    const uint64_t share = scan_j == 0 ? kNumSMs - (ib_P - 1) * (kNumSMs / ib_P) : kNumSMs / ib_P;
                    for (uint32_t i = epilogue_thread_idx; i < kNumExpertsPerRank; i += kNumEpilogueThreads) {
                        atomicAdd(reinterpret_cast<unsigned long long*>(workspace.get_expert_recv_count_ptr(r, i)), static_cast<unsigned long long>(cnt[i]));
                        __threadfence();
                        ptx::atomic_add(workspace.get_expert_recv_count_sum_ptr(i), (share << 32) | static_cast<uint64_t>(cnt[i]));
                    }
                    IB_TRACE_MAX(26, epilogue_thread_idx == 0);  // counts published, slowest scanning SM
                    // Slot base from the lower slices
                    for (uint32_t i = epilogue_thread_idx; i < kNumExpertsPerRank; i += kNumEpilogueThreads)
                        cnt2[i] = 0;
                    if (epilogue_thread_idx < scan_j)
                        IB_POLL(ptx::ld_acq_sys(ib_sl_cflag + k * 4 + epilogue_thread_idx) != ib_epoch, (void) 0, "sl_cflag", k, epilogue_thread_idx);
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    for (uint32_t i = epilogue_thread_idx; i < kNumExpertsPerRank; i += kNumEpilogueThreads) {
                        uint32_t b = 0;
                        for (uint32_t jj = 0; jj < scan_j; ++ jj)
                            b += ptx::ld_volatile(ib_sl_cnt + (k * 4 + jj) * 256 + i);
                        cnt2[i] = b;
                        cnt[i] = 0;  // reused as the slot cursor
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    SL_TR(scan_j, 2, epilogue_thread_idx == 0);
                    // Pass B: slot entries (token order within the slice; slices are in token order); rule A for the weights
                    const bool sl_use_list = kIBSliceList and *sl_n <= kSLCap;
                    if (sl_use_list) {
                        // Only our matches: one independent weight load each (all in flight), then the local slot stores
                        const uint32_t n = *sl_n;
                        for (uint32_t q = epilogue_thread_idx; q < n; q += kNumEpilogueThreads) {
                            const uint32_t ent = sl_list[q], i = ent >> 8, le = ent & 0xffu;
                            uint32_t w = ptx::ld_volatile(wts + i);
                            SL_ADD(1, w == 0xffffffffu);
                            IB_POLL(w == 0xffffffffu, w = ptx::ld_volatile(wts + i), "sl_w", r, i);
                            *workspace.get_src_token_topk_idx_ptr(le, r, cnt2[le] + atomicAdd_block(cnt + le, 1u)) = i;
                        }
                    }
                    for (uint32_t i0 = e0; not sl_use_list and i0 < e1; i0 += kNumEpilogueThreads * kSLB) {
                        uint64_t v[kSLB];
                        uint32_t wv[kSLB];
                        #pragma unroll
                        for (uint32_t u = 0; u < kSLB; ++ u) {
                            const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                            v[u] = i < e1 ? ptx::ld_volatile(idx + i) : 0ull;
                            wv[u] = i < e1 ? ptx::ld_volatile(wts + i) : 0u;
                        }
                        #pragma unroll
                        for (uint32_t u = 0; u < kSLB; ++ u) {
                            const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                            if (i < e1) {
                                SL_ADD(1, wv[u] == 0xffffffffu);
                                IB_POLL(wv[u] == 0xffffffffu, wv[u] = ptx::ld_volatile(wts + i), "sl_w", r, i);
                                const int e = static_cast<int>(static_cast<int64_t>(v[u]));
                                if (e >= 0 and static_cast<uint32_t>(e) / kNumExpertsPerRank == sym_buffer.rank_idx) {
                                    const uint32_t le = static_cast<uint32_t>(e) % kNumExpertsPerRank;
                                    *workspace.get_src_token_topk_idx_ptr(le, r, cnt2[le] + atomicAdd_block(cnt + le, 1u)) = i;
                                }
                            }
                        }
                    }
                    SL_TR(scan_j, 3, true);
                    __threadfence();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_thread_idx == 0)
                        ptx::st_rel_sys(ib_sl_tflag + k * 4 + scan_j, ib_epoch);
                    SL_TR(scan_j, 4, epilogue_thread_idx == 0);
                    IB_TRACE_MAX(32, epilogue_thread_idx == 0);  // slice tables done
                }
                // Overlap orders: key1 (wave sort; every receiver) and key2 (stream combine; only r's proxy)
                const bool d1_key1 = ib_ws_on(nt) and scan_j == 0;
                const bool d1_key2 = ib_sc_on(nt) and proxy == sym_buffer.rank_idx and scan_j == 0;
                const auto keys1 = ib_keys(0), keys2 = ib_keys(1);
                if (d1_key1 or d1_key2) {
                    for (uint32_t t = epilogue_thread_idx; t < nt; t += kNumEpilogueThreads)
                        keys1[t] = kNumExpertsPerRank, keys2[t] = 0;
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                // DG_MEGA_MOE_IB_SCAN_BATCH=N: loads in flight per thread in the D1 scan (default 4). The scan reads each
                // remote source's T x K top-k entries over NVLink; MegaSeek @1024 (19456 entries / 256 threads) took ~68 us
#if defined(DG_MEGA_MOE_IB_SCAN_BATCH)
                constexpr uint32_t kB = DG_MEGA_MOE_IB_SCAN_BATCH;
#else
                constexpr uint32_t kB = 4;
#endif
                if (kIBEarlyPublish and ib_early_on(nt)) {
                    // Pass 1: counts only (idx), then publish them (same protocol as below); COUNTS_IN_META reads them
                    if (ib_st_on(nt)) {
                        // All of the node's counts ([rank][expert]): ours, and the sum of the lower ranks' (our slice base)
                        const auto cnt_all = reinterpret_cast<const uint32_t*>(mir + kIBOffCnt);
                        const uint32_t myj = sym_buffer.rank_idx % ib_lsa;
                        auto st_base = cnt2 + kNumExpertsPerRank;  // smem word after cnt2
                        if (epilogue_thread_idx == 0)
                            *st_base = 0;
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        uint32_t below = 0;
                        for (uint32_t q = epilogue_thread_idx; q < ib_lsa * kNumExpertsPerRank; q += kNumEpilogueThreads) {
                            uint32_t v = ptx::ld_volatile(cnt_all + q);
                            IB_POLL(v == 0xffffffffu, v = ptx::ld_volatile(cnt_all + q), "st_cnt", r, q);
                            if (q / kNumExpertsPerRank == myj)
                                cnt[q % kNumExpertsPerRank] = v;
                            else if (q / kNumExpertsPerRank < myj)
                                below += v;
                        }
                        below = __reduce_add_sync(0xffffffff, below);
                        if (lane_idx == 0 and below)
                            atomicAdd_block(st_base, below);
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_thread_idx == 0) {  // per-expert offsets inside our slice (<= 256 experts per rank)
                            uint32_t o = 0;
                            for (uint32_t e = 0; e < kNumExpertsPerRank; ++ e)
                                cnt2[e] = o, o += cnt[e];
                        }
                    } else if constexpr (kIBCountsInMeta) {
                        for (uint32_t i = epilogue_thread_idx; i < kNumExpertsPerRank; i += kNumEpilogueThreads) {
                            uint32_t v = ptx::ld_volatile(cnt_early + i);
                            IB_POLL(v == 0xffffffffu, v = ptx::ld_volatile(cnt_early + i), "cim", r, i);
                            cnt[i] = v;
                        }
                    } else
                    for (uint32_t i0 = 0; i0 < nt * kNumTopk; i0 += kNumEpilogueThreads * kB) {
                        uint64_t v[kB];
                        #pragma unroll
                        for (uint32_t u = 0; u < kB; ++ u) {
                            const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                            v[u] = i < nt * kNumTopk ? ptx::ld_volatile(idx + i) : 0ull;
                        }
                        #pragma unroll
                        for (uint32_t u = 0; u < kB; ++ u) {
                            const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                            if (i < nt * kNumTopk) {
                                IB_POLL(v[u] == kIBIdxSentinel, v[u] = ptx::ld_volatile(idx + i), "d1c_idx", r, i);
                                const int e = static_cast<int>(static_cast<int64_t>(v[u]));
                                if (e >= 0 and static_cast<uint32_t>(e) / kNumExpertsPerRank == sym_buffer.rank_idx)
                                    atomicAdd_block(cnt + static_cast<uint32_t>(e) % kNumExpertsPerRank, 1u);
                            }
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    for (uint32_t i = epilogue_thread_idx; i < kNumExpertsPerRank; i += kNumEpilogueThreads) {
                        *workspace.get_expert_recv_count_ptr(r, i) = cnt[i];
                        __threadfence();
                        ptx::atomic_add(workspace.get_expert_recv_count_sum_ptr(i),
                                        (static_cast<uint64_t>(kNumSMs) << 32) | static_cast<uint64_t>(cnt[i]));
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    IB_TRACE(22, epilogue_thread_idx == 0 and sm_idx == kNumSMs - 1);
                    IB_TRACE(19, epilogue_thread_idx == 0 and sm_idx == kNumSMs - 1);
                    IB_TRACE_MAX(26, epilogue_thread_idx == 0);  // counts published, slowest scanning SM
                    if (ib_st_on(nt)) {
                        // Copy our slice of r's sender-built table into the local slot table
                        const auto st_tab = reinterpret_cast<const uint32_t*>(mir + kIBOffSTab) + *(cnt2 + kNumExpertsPerRank);
                        const uint32_t m = cnt2[kNumExpertsPerRank - 1] + cnt[kNumExpertsPerRank - 1];
                        for (uint32_t q = epilogue_thread_idx; q < m; q += kNumEpilogueThreads) {
                            uint32_t lo = 0, hi = kNumExpertsPerRank;  // last e with cnt2[e] <= q
                            while (hi - lo > 1) { const uint32_t mid = (lo + hi) / 2; if (cnt2[mid] <= q) lo = mid; else hi = mid; }
                            uint32_t v = ptx::ld_volatile(st_tab + q);
                            IB_POLL(v == 0xffffffffu, v = ptx::ld_volatile(st_tab + q), "st_tab", r, q);
                            *workspace.get_src_token_topk_idx_ptr(lo, r, q - cnt2[lo]) = v;
                        }
                        __threadfence();
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_thread_idx == 0)  // pulls of r's rows need only this, not the proxy's sort below
                            ptx::st_rel_sys(&workspace.signals->ib_o_tab_ready[k], ib_epoch);
                    }
                }
                // Poll (top-k idx and weights; rule A for the weights the pull reads unchecked), then slots for the
                // pairs routed to OUR experts by shared-memory atomics; entries are local stores
                for (uint32_t i0 = 0; ((ib_P == 1 and (not ib_st_on(nt) or proxy == sym_buffer.rank_idx)) or d1_key1 or d1_key2) and i0 < nt * kNumTopk; i0 += kNumEpilogueThreads * kB) {
                    uint64_t v[kB];
                    uint32_t wv[kB];
                    #pragma unroll
                    for (uint32_t u = 0; u < kB; ++ u) {
                        const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                        v[u] = i < nt * kNumTopk ? ptx::ld_volatile(idx + i) : 0ull;
                        wv[u] = i < nt * kNumTopk ? ptx::ld_volatile(wts + i) : 0u;
                    }
                    #pragma unroll
                    for (uint32_t u = 0; u < kB; ++ u) {
                        const uint32_t i = i0 + u * kNumEpilogueThreads + epilogue_thread_idx;
                        if (i < nt * kNumTopk) {
                            IB_POLL(v[u] == kIBIdxSentinel, v[u] = ptx::ld_volatile(idx + i), "d1_idx", r, i);
                            IB_POLL(wv[u] == 0xffffffffu, wv[u] = ptx::ld_volatile(wts + i), "d1_w", r, i);
                            const int e = static_cast<int>(static_cast<int64_t>(v[u]));
                            if ((d1_key1 or d1_key2) and e >= 0 and static_cast<uint32_t>(e) / kNumExpertsPerRank / ib_lsa == ib_my_node) {
                                const uint32_t le = static_cast<uint32_t>(e) % kNumExpertsPerRank;
                                if (d1_key1) atomicMin(keys1 + i / kNumTopk, le);
                                if (d1_key2) atomicMax(keys2 + i / kNumTopk, le);
                            }
                            if (not (ib_sst and not (kCGDrop & 4096)) and ib_P == 1 and not ib_st_on(nt) and e >= 0 and static_cast<uint32_t>(e) / kNumExpertsPerRank == sym_buffer.rank_idx) {
                                const uint32_t le = static_cast<uint32_t>(e) % kNumExpertsPerRank;
                                const uint32_t slot = atomicAdd_block((ib_early_on(nt) ? cnt2 : cnt) + le, 1u);
                                *workspace.get_src_token_topk_idx_ptr(le, r, slot) = i;
                            }
                        }
                    }
                }
                if (kIBEarlyPublish and not ib_st_on(nt))
                    __threadfence();  // slot entries written by the scan above (also before the late publish below)
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                // EarlyPublish: remote-row pulls wait for this source's slot table; raise it as soon as the scan wrote it,
                // not after the overlap sort below (B300: +11 / +16 us @128 Flash / MegaSeek when it came after the sort)
                if (kIBEarlyPublish and not ib_st_on(nt) and epilogue_thread_idx == 0)
                    ptx::st_rel_sys(&workspace.signals->ib_o_tab_ready[k], ib_epoch);
                if (not ib_early_on(nt))
                    IB_TRACE(22, epilogue_thread_idx == 0 and sm_idx == kNumSMs - 1);
                if (not (ib_sst and not (kCGDrop & 4096)) and not ib_early_on(nt) and ib_P == 1) {
                    for (uint32_t i = epilogue_thread_idx; i < kNumExpertsPerRank; i += kNumEpilogueThreads) {
                        *workspace.get_expert_recv_count_ptr(r, i) = cnt[i];
                        if constexpr (kIBCountsEarly) {
                            if (ptx::ld_volatile(cnt_early + i) != cnt[i])
                                atomicAdd(reinterpret_cast<unsigned long long*>(workspace.signals->ib_trace + 39), 1ull);
                        }
                        __threadfence();
                        ptx::atomic_add(workspace.get_expert_recv_count_sum_ptr(i),
                                        (static_cast<uint64_t>(kNumSMs) << 32) | static_cast<uint64_t>(cnt[i]));
                    }
                } else if (not (ib_sst and not (kCGDrop & 4096))) {
                    if constexpr (kIBCountsInMeta) {
                        for (uint32_t i = epilogue_thread_idx; not ib_st_on(nt) and i < kNumExpertsPerRank; i += kNumEpilogueThreads)
                            if (cnt2[i] != cnt[i])
                                atomicAdd(reinterpret_cast<unsigned long long*>(workspace.signals->ib_trace + 39), 1ull);
                    }
                    IB_TRACE_MAX(32, epilogue_thread_idx == 0);  // tables done (the counts went out before pass 2)
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (not ib_early_on(nt)) {
                    IB_TRACE(19, epilogue_thread_idx == 0 and sm_idx == kNumSMs - 1);
                    IB_TRACE_MAX(26, epilogue_thread_idx == 0 and not ib_sst);  // scan done, slowest scanning SM
                }
                // Overlap orders, after the count publish (off the count critical path): slot table (token -> mirror
                // x row) and the proxy's return order, then an epoch flag the pulls / forward-combine acquire
                if constexpr (kIBWaveSort or kIBStreamCombine or kIBEarlyPublish) {
                    if (d1_key1) {
                        ib_stable_sort(keys1, nt, epilogue_thread_idx, [&](const uint32_t& t, const uint32_t& pos) {
                            ib_tab[kIBTabSlot + k * kIBTabT + t] = kIBWSMode >= 2 ? t : pos;
                        });
                    } else if (kIBWaveSort) {
                        for (uint32_t t = epilogue_thread_idx; t < nt; t += kNumEpilogueThreads)
                            ib_tab[kIBTabSlot + k * kIBTabT + t] = t;  // below the size gate: identity (the pull reads the table)
                    }
                    if (d1_key2) {
                        ib_stable_sort(keys2, nt, epilogue_thread_idx, [&](const uint32_t& t, const uint32_t& pos) {
                            ib_tab[kIBTabPerm2 + ib_nr_of_node(r / ib_lsa) * kIBTabT + pos] = t;
                        });
                    }
                    __threadfence();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_thread_idx == 0 and scan_j == 0)
                        ptx::st_rel_sys(&workspace.signals->ib_o_order_ready[k], ib_epoch);
                }
                }  // not (SST and not the proxy)
            }
        }
#endif

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Persistently schedule over blocks
        uint32_t current_iter_idx = 0;
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            // Wait UMMA arrival
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
            shared_storage.tmem_full_barriers[accum_stage_idx].wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            // Now we can release the task
            scheduler.release_task_info();

            // Compute offsets
            // NOTES: use shuffle here to let NVCC know warp divergence won't happen
            const uint32_t valid_m = ptx::exchange(task_info.valid_m, 0);
            const uint32_t wg_start_m = epilogue_wg_idx * WG_BLOCK_M;
            const uint32_t num_valid_wg_rows = valid_m <= wg_start_m ? 0u : cute::min(valid_m - wg_start_m, WG_BLOCK_M);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;  // Ring-buffer offset for reusable data buffers
            const uint32_t m_idx = block_idx * BLOCK_M;
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;  // Full-pool offset for non-ring metadata
            const uint32_t n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            uint32_t n_idx = n_block_idx * BLOCK_N;

            const auto release_tmem = [&]() {
                ptx::tcgen05_before_thread_sync();
                shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
            };
            // Store blocks are loaded from TMEM into a ring of registers ahead of use
            const auto load_epi_block = [&]<uint32_t kNumBuffers, uint32_t kNumAtomsPerStore>(
                    const uint32_t& s, uint32_t (&raw_values)[kNumBuffers][kNumAtomsPerStore][ATOM_M]) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                    auto& values = raw_values[s % kNumBuffers][i];
                    const uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + (s * kNumAtomsPerStore + i) * ATOM_M;
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                           values[0], values[1], values[2], values[3]);
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                           values[4], values[5], values[6], values[7]);
                }
            };
            // Issue the loads of store block `s` if it holds valid tokens; the last valid block is waited for right away and
            // the TMEM stage is released, as nothing else will be read from it
            const auto load_epi_block_if_valid = [&]<uint32_t kNumBuffers, uint32_t kNumAtomsPerStore>(
                    const uint32_t& s, const uint32_t& num_valid_store_blocks,
                    uint32_t (&raw_values)[kNumBuffers][kNumAtomsPerStore][ATOM_M]) {
                if (s < num_valid_store_blocks) {
                    load_epi_block(s, raw_values);
                    if (s + 1 == num_valid_store_blocks) {
                        cutlass::arch::fence_view_async_tmem_load();
                        release_tmem();
                    }
                }
            };
            // Prefetch the first `kNumPrefetchStages` store blocks (no wait); a warpgroup without valid tokens releases the stage right away
            const auto prefetch_prologue = [&]<uint32_t kNumBuffers, uint32_t kNumAtomsPerStore>(
                    const uint32_t& num_valid_store_blocks, uint32_t (&raw_values)[kNumBuffers][kNumAtomsPerStore][ATOM_M]) {
                #pragma unroll
                for (uint32_t s = 0; s + 1 < kNumBuffers; ++ s)
                    load_epi_block_if_valid(s, num_valid_store_blocks, raw_values);
                if (num_valid_store_blocks == 0)
                    release_tmem();
            };
            // Before computing store block `s`: wait for its loads (issued `kNumPrefetchStages` iterations earlier, or right
            // here without prefetch), then issue block `s + kNumPrefetchStages` so that it lands while block `s` is processed
            const auto wait_and_prefetch_next = [&]<uint32_t kNumBuffers, uint32_t kNumAtomsPerStore>(
                    const uint32_t& s, const uint32_t& num_valid_store_blocks,
                    uint32_t (&raw_values)[kNumBuffers][kNumAtomsPerStore][ATOM_M]) {
                constexpr uint32_t kNumPrefetchStages = kNumBuffers - 1;
                if constexpr (kNumPrefetchStages == 0)
                    load_epi_block_if_valid(s, num_valid_store_blocks, raw_values);
                cutlass::arch::fence_view_async_tmem_load();
                if constexpr (kNumPrefetchStages > 0)
                    load_epi_block_if_valid(s + kNumPrefetchStages, num_valid_store_blocks, raw_values);
            };

            if (task_info.block_phase == sched::BlockPhase::Linear1 or task_info.block_phase == sched::BlockPhase::SharedLinear1) {
                if (not task_info.is_shared()) {
                    // Wait L2 block empty
                    const auto l2_empty_ptr = workspace.get_l2_empty_count_ptr(ring_block_idx);
                    const auto num_expected_blocks = (L2_SHAPE_N / BLOCK_N) * (pool_block_idx / kNumRingBlocks);
                    while (ptx::ld_acq(l2_empty_ptr) != num_expected_blocks);
                }

                // Unified L1 epilogue: SwiGLU in-place using granularity 8 interleaved weights
                // With `SM100_TMEM_LOAD_16dp256b1x`, gate/up pairs are:
                float stored_cached_weight = 1.0f;

                constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M_L1 / ATOM_M;
                constexpr uint32_t kNumStoreBlocks = WG_BLOCK_M / STORE_BLOCK_M_L1;
                constexpr uint32_t kNumPrefetchStages = math::constexpr_min(kNumEpiPrefetchStages, kNumStoreBlocks - 1);
                uint32_t raw_values[kNumPrefetchStages + 1][kNumAtomsPerStore][ATOM_M];

                // Store blocks with at least one valid row (the last one may be partial); `<= kNumStoreBlocks` by construction
                const uint32_t num_valid_store_blocks = math::ceil_div(num_valid_wg_rows, STORE_BLOCK_M_L1);
                prefetch_prologue(num_valid_store_blocks, raw_values);

                #pragma unroll
                for (uint32_t s = 0; s < kNumStoreBlocks; ++ s) {
                    if (s >= num_valid_store_blocks)
                        break;
                    wait_and_prefetch_next(s, num_valid_store_blocks, raw_values);
                    auto& store_block_raw_values = raw_values[s % (kNumPrefetchStages + 1)];

                    // Iterate all atoms in the store block
                    float2 activation_values[kNumAtomsPerStore][2];
                    float2 amax_values[kNumAtomsPerStore];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        const uint32_t j = s * kNumAtomsPerStore + i;

                        // Load weights from global into register cache per 32 tokens
                        DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");
                        if (not task_info.is_shared() and (j * ATOM_M) % 32 == 0 and
                            (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                            stored_cached_weight = *buffer.l1_topk_weights_buffer
                                .get_data_buffer(ring_m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                .template get_base_ptr<float>();
                        }

                        // Load weights from register cache
                        const float2 weights = {
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                        };

                        // Apply SwiGLU: silu(gate) * up
                        auto fp32_values = reinterpret_cast<float2*>(store_block_raw_values[i]);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            auto bf16_gate = __float22bfloat162_rn(fp32_values[k * 2 + 0]);
                            auto bf16_up =   __float22bfloat162_rn(fp32_values[k * 2 + 1]);

                            // Clamp
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            // SwiGLU
                            auto gate = __bfloat1622float2(bf16_gate);
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                kFastMath ? __expf(-gate.y) : expf(-gate.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            if constexpr (kFastMath) {
                                gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                            } else {
                                gate = {gate.x / denom.x, gate.y / denom.y};
                            }
                            const auto up = __bfloat1622float2(bf16_up);
                            activation_values[i][k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                        }

                        // Amax reduction (thread-level)
                        float2 thread_local_amax = {0.f, 0.f};
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            thread_local_amax.x = fmaxf(thread_local_amax.x, fabsf(activation_values[i][k].x));
                            thread_local_amax.y = fmaxf(thread_local_amax.y, fabsf(activation_values[i][k].y));
                        }

                        // Amax reduction (warp-level)
                        amax_values[i].x = math::warp_reduce<4, true>(
                            thread_local_amax.x, math::ReduceMax<float>());
                        amax_values[i].y = math::warp_reduce<4, true>(
                            thread_local_amax.y, math::ReduceMax<float>());

                        // Reduce amax (warp-pair-level)
                        if (lane_idx < 4)
                            shared_storage.amax_reduction[epilogue_warp_idx][i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                        __syncwarp();
                    }

                    // Wait shared memory release from previous TMA store
                    // And fence `shared_storage.amax_reduction`
                    const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                    ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Cast to FP8 E4M3 and store into shared memory
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        // Reduce amax (warp-pair-level)
                        const float2 wp_amax =
                            shared_storage.amax_reduction[epilogue_warp_idx ^ 1][i * (ATOM_M / 2) + lane_idx % 4];
                        amax_values[i].x = fmaxf(amax_values[i].x, wp_amax.x);
                        amax_values[i].y = fmaxf(amax_values[i].y, wp_amax.y);

                        // Calculate SF
                        const uint2 sf_exp = {math::get_ue8m0_sf_exp(amax_values[i].x),
                                              math::get_ue8m0_sf_exp(amax_values[i].y)};
                        const float2 sf_inv = {math::get_ue8m0_sf_inv<float>(sf_exp.x),
                                               math::get_ue8m0_sf_inv<float>(sf_exp.y)};

                        // Cast
                        const float2 upper = __fmul2_rn(activation_values[i][0], sf_inv);
                        const float2 lower = __fmul2_rn(activation_values[i][1], sf_inv);
                        const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));

                        // STSM
                        uint32_t row = lane_idx;
                        uint32_t col = warp_idx_in_wg;
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx])
                            + i * ATOM_M * L1_OUT_BLOCK_N
                            + row * L1_OUT_BLOCK_N
                            // Use 64B swizzle for SwiGLU, so divided by 2
                            + (col ^ (row / 2)) * kNumBankGroupBytes;
                        ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);

                        // Store SF to `buffer.l2_sf_buffer` as UE8M0 (MN-major layout)
                        // Only one warp per pair writes (both hold the same SF after cross-warp reduce)
                        // Each lane < 4 holds SF for 2 rows (sf.x and sf.y)
                        if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                            const uint32_t k_idx = n_block_idx * 2 + warp_idx_in_wg / 2;
                            const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                            const uint32_t mn_stride = (task_info.is_shared() ? kNumSharedSFTokens : kNumSFRingTokens) * sizeof(uint32_t);
                            const auto sf_base_ptr = task_info.is_shared() ?
                                buffer.shared_l2_sf_buffer.get_base_ptr<uint8_t>() : buffer.l2_sf_buffer.get_base_ptr<uint8_t>();
                            // NOTES: consecutive tokens (t, t + 1) are in the same 32-group, so `sf_idx` differs by 4
                            // NOTES: originally there was:
                            //   - `const uint32_t token_idx_in_expert = task_info.m_block_idx * BLOCK_M + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L1 + i * ATOM_M + lane_idx * 2
                            //   - `task_info.pool_block_idx * SF_BLOCK_M + transform_sf_token_idx(token_idx_in_expert)`
                            // We find out that
                            //   1. `task_info.m_block_idx * BLOCK_M` mod `BLOCK_M` is 0, and `epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L1 + i * ATOM_M + lane_idx * 2` is always < `BLOCK_M`, so we can put `task_info.m_block_idx * BLOCK_M` outside
                            //   2. `lane_idx * 2` controls the lowest 3 bit of `token_idx_in_expert`, and `transform_sf_token_idx` is a bitwise-independent transformation if the input is less than `BLOCK_M`, so we can put `lane_idx * 2` outside
                            // This reduce the number of computation instructions.
                            const uint32_t token_base_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L1 + i * ATOM_M;
                            __builtin_assume(token_base_idx < BLOCK_M);
                            const auto sf_token_idx = block_idx * SF_BLOCK_M
                                + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                            const auto sf_addr = k_uint_idx * mn_stride + sf_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                            sf_base_ptr[sf_addr] = static_cast<uint8_t>(sf_exp.x);
                            sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] = static_cast<uint8_t>(sf_exp.y);
                        }
                        __syncwarp();
                    }
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Issue TMA store after all atoms in this store block
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        const auto tensor_map_l1_output_ptr = task_info.is_shared() ? &tensor_map_shared_l1_output : &tensor_map_l1_output;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            tensor_map_l1_output_ptr,
                            shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx],
                            out_n_idx,
                            m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L1);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }

                // Notify L2 and increment L1 empty count
                // TODO: less epilogue sync scope
                ptx::tma_store_wait<0>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    if (task_info.is_shared()) {
                        ptx::red_add_rel(
                            workspace.get_shared_l2_full_count_ptr(pool_block_idx), 1u);
                    } else {
                        L2KBlockDependency::arrive(workspace.get_l2_full_mask_ptr(ring_block_idx), n_block_idx);

                        // Increment L1 empty count for this physical slot (one per N block)
                        ptx::red_add(
                            workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                    }
                }
                __syncwarp();
            } else {
                // Increment L2 empty count for this physical slot (one per N block)
                if (not task_info.is_shared()) {
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        ptx::red_add(
                            workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
                    }
                    __syncwarp();
                }

                constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M_L2 / ATOM_M;
                constexpr uint32_t kNumStoreBlocks = WG_BLOCK_M / STORE_BLOCK_M_L2;
                constexpr uint32_t kNumPrefetchStages = math::constexpr_min(kNumEpiPrefetchStages, kNumStoreBlocks - 1);
                uint32_t raw_values[kNumPrefetchStages + 1][kNumAtomsPerStore][ATOM_M];

                // Store blocks with at least one valid row (the last one may be partial); `<= kNumStoreBlocks` by construction
                const uint32_t num_valid_store_blocks = math::ceil_div(num_valid_wg_rows, STORE_BLOCK_M_L2);
                prefetch_prologue(num_valid_store_blocks, raw_values);

                // L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink
                #pragma unroll
                for (uint32_t s = 0; s < kNumStoreBlocks; ++ s) {
                    if (s >= num_valid_store_blocks)
                        break;
                    wait_and_prefetch_next(s, num_valid_store_blocks, raw_values);
                    auto& store_block_raw_values = raw_values[s % (kNumPrefetchStages + 1)];

                    // Read the source metadata of this warp's rows (2 per atom) before the shared memory stores to overlap the latency
                    layout::TokenSrcMetadata cached_src_metadata[kNumAtomsPerStore];
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumAtomsPerStore; ++ j) {
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L2 + j * ATOM_M + warp_idx_in_wg * 2 + lane_idx / 16;
                        if (m_idx_in_block < valid_m)
                            cached_src_metadata[j] = task_info.is_shared() ?
                                layout::TokenSrcMetadata(sym_buffer.rank_idx, pool_m_idx + m_idx_in_block, kNumTopk) :
                                *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
                    }
                    __syncwarp();

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        // Wait shared memory release from previous NVLink store
                        // NOTES: skip for the first store block since the prior full barrier already ensures completion
                        if (i == 0 and s > 0)
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        // Store into shared memory
                        // NOTES: each lane provides its own address for stmatrix; 2 warps share a BF16 swizzle atom
                        auto& values = store_block_raw_values[i];
                        uint32_t row = lane_idx % 8;
                        uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                            (warp_idx_in_wg / 2) * STORE_BLOCK_M_L2 * kSwizzleCDMode +
                            i * ATOM_M * kSwizzleCDMode +
                            row * (kNumBankGroupBytes * 8) +
                            (col ^ row) * kNumBankGroupBytes;
                        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                            math::cast_into_bf16_and_pack(values[0], values[1]),
                            math::cast_into_bf16_and_pack(values[2], values[3]),
                            math::cast_into_bf16_and_pack(values[4], values[5]),
                            math::cast_into_bf16_and_pack(values[6], values[7]),
                            smem_ptr
                        );
                    }

                    // Wait shared memory ready
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Write into remote buffers
                    // Each warp writes 2 rows (lane_idx/16 splits the warp into two halves, one per row)
                    const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                    const uint32_t bank_group_idx = lane_idx % 8;

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumAtomsPerStore; ++ j) {
                        const uint32_t row_in_store = j * ATOM_M + warp_idx_in_wg * 2 + lane_idx / 16;
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L2 + row_in_store;

                        // Skip padding rows beyond the actual token count for this expert
                        if (m_idx_in_block >= valid_m)
                            break;

                        const auto& [dst_rank_idx, dst_token_idx, dst_topk_idx] = cached_src_metadata[j];

                        // Read from shared memory
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                            (lane_idx % 16 / 8) * STORE_BLOCK_M_L2 * kSwizzleCDMode +
                            row_in_store * kSwizzleCDMode +
                            (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                        const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));

                        // Write into remote
                        const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        const auto dst_ptr = math::advance_ptr<float4>(
                            dst_token.get_base_ptr(),
                            n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) + (lane_idx % 16) * static_cast<uint32_t>(sizeof(float4)));
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
                        // C1: rows of sources on other nodes go to the proxy's combine buffer (same layout), over NVLink
                        if (is_remote(dst_rank_idx)) {
                            const auto off = reinterpret_cast<const uint8_t*>(dst_ptr) -
                                reinterpret_cast<const uint8_t*>(buffer.combine_token_buffer.get_base_ptr());
                            *sym_buffer.map(reinterpret_cast<float4*>(ib_pcomb(dst_rank_idx / ib_lsa) + off),
                                            ib_cp(dst_rank_idx, ib_my_node)) = packed;
                            continue;
                        }
#endif
                        *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                    }
                }

                // Ensure the next epilogue safe to use shared memory
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                TT_MAX(task_info.is_shared() ? 1 : 0, epilogue_thread_idx == 0);
            }
        }

        // Deallocate tensor memory
        // NOTES: must be called by the same logical warp ID on both CTAs
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);

        // Combine: reduce top-k results and write back
        // NOTES: reuse shared memory from start up to the barriers
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        // 3 slots of chunk is needed: 2 load stages and 1 store
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;

        // NOTES: either 1 or 2 chunks for simplicity
        // NOTES: Restrict on both smem and register
        constexpr uint32_t kNumChunks =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= kNumReusableSmemBytes and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= kNumReusableSmemBytes, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements (one per lane)");
        DG_STATIC_ASSERT(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u, "Top-k + shared must fit in a single warp");

        // Verify combined shared memory budget at runtime
        DG_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= kNumReusableSmemBytes);

        // Per-warp buffer: 2 stage load buffers + 1 store buffer
        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer  = math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);
        // DG_MEGA_MOE_IB_C3_PAIR=1 (tail): each pipeline stage of the source combine loads up to TWO top-k slots (two TMA loads on
        // one barrier), so a token-chunk with ~10 local slots (MegaSeek) takes ~5 load round trips instead of ~10; the sum order
        // is unchanged (first slot, then second). Second-slot buffers at smem slots 3 and 4.
#if (defined(DG_MEGA_MOE_IB_C3_PAIR) and DG_MEGA_MOE_IB_C3_PAIR) or (DG_TAIL_BITS & 64)
        constexpr bool kC3Pair = true;
#else
        constexpr bool kC3Pair = false;
#endif
        DG_STATIC_ASSERT(not kC3Pair or 5 * kNumEpilogueWarps * kNumChunkBytes <= kNumReusableSmemBytes, "C3 pair buffers");
        const auto combine_load_buffer2 = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + (3 + i) * kNumEpilogueWarps) * kNumChunkBytes);
        });

        // Per-warp barriers
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return &shared_storage.combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;

        // Peers' grid indices for tagging their combine readiness; the load overlaps with the grid sync
        DG_STATIC_ASSERT(kNumRanks <= kNumEpilogueThreads, "Insufficient threads for combine readiness");
        uint64_t peer_grid_idx = 0;
        if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
            peer_grid_idx = *workspace.get_peer_grid_idx_ptr(epilogue_thread_idx);

        // All local L2 writes are done after this grid sync
        comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
            workspace, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );

        // Notify remote ranks; ordered before the cleanup barrier by the dispatch/epilogue sync below
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        if (sm_idx == 0 and epilogue_thread_idx < kNumRanks and not is_remote(epilogue_thread_idx))
#else
        if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
#endif
            ptx::st_rel_sys(sym_buffer.map(workspace.get_combine_ready_grid_idx_ptr(sym_buffer.rank_idx), epilogue_thread_idx), peer_grid_idx);
        IB_TRACE(5, sm_idx == 0 and epilogue_thread_idx == 0);
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        if constexpr (kIBNoOvReturn) {
            if (sm_idx == 0 and epilogue_thread_idx == 0)
                ib_noov_push(kIBNoOvL2Slot);
        }
#endif

#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        if constexpr (kIBFwdJoin) {
            // FWD_JOIN: help the dispatch warps with the forward-combine items still unclaimed
            ib_forward_combine(epilogue_warp_idx * kNumSMs + sm_idx, kNumSMs * kNumEpilogueWarps, false,
                               cute::Int<DG_MEGA_MOE_IB_FWD_BATCH>{}, true);
            TT_MAX(6, lane_idx == 0);
        }
#endif
        // Barrier with dispatch warps, so that they can do clean workspace
        TT_MAX(7, epilogue_thread_idx == 0);
        TT_MIN(13, epilogue_thread_idx == 0);
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        if constexpr (kIBC3Early) {
            asm volatile("barrier.arrive %0, %1;" : : "r"(kDispatchWithEpilogueBarrierIdx), "r"(kNumDispatchThreads + kNumEpilogueThreads) : "memory");
        } else {
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        }
#else
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
#endif

#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        if constexpr (not kIBReduceOnArrival) {
            ib_forward_combine(epilogue_warp_idx * kNumSMs + sm_idx, kNumSMs * kNumEpilogueWarps, sm_idx == 0 and epilogue_thread_idx == 0,
                               cute::Int<DG_MEGA_MOE_IB_FWD_BATCH>{}, true);
            IB_TRACE(6, sm_idx == 0 and epilogue_thread_idx == 0);
        }
#endif

        // Iterate over all token chunks, 1 token 1 topk latency: ~3 us
        IB_TRACE_MAX(15, epilogue_thread_idx == 0);  // source: final combine loop starts, slowest SM
        TT_MIN(8, epilogue_thread_idx == 0);
        const auto grid_idx = ptx::get_grid_idx() + 1;
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        constexpr bool kC3Split = kIBC3Split;
        DG_STATIC_ASSERT(not kIBC3Split or kNumTopk >= 2, "C3 split scratch");
        const auto c3_scratch = reinterpret_cast<float*>(ib_pcomb(ib_my_node));  // own node's slot: never used by the proxy
#else
        constexpr bool kC3Split = false;
        float* const c3_scratch = nullptr;
#endif
        // C3 gate (DG_MEGA_MOE_IB_C3_HDR_GATE, defined with NO_RET_HDR): 1 = all return headers, 2 = per-chunk tail flags
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        uint64_t* const c3_gate = reinterpret_cast<uint64_t*>(tl_fgroup);  // [16 nodes][64 chunks] epoch words (mode 1: word 0)
        // DG_MEGA_MOE_IB_C3_GATE_MIN=T: gate 2 only for a source with >= T tokens (source-side only, so a runtime decision
        // per source is consistent; GB200 MegaSeek 128: the gate costs ~+45 us there while the stall regime needs size)
#ifndef DG_MEGA_MOE_IB_C3_GATE_MIN
#define DG_MEGA_MOE_IB_C3_GATE_MIN 0
#endif
        const bool c3_g2 = kIBC3Gate == 2 and num_tokens >= DG_MEGA_MOE_IB_C3_GATE_MIN;
        const bool c3_poller = c3_g2 and sm_idx == 0 and epilogue_warp_idx == 0;
        const uint32_t c3_gw = epilogue_warp_idx * kNumSMs + sm_idx - (c3_g2 ? 1u : 0u);  // the poller takes no items
        const uint32_t c3_nw = kNumSMs * kNumEpilogueWarps - (c3_g2 ? 1u : 0u);
#else
        constexpr bool c3_g2 = false;
        constexpr bool c3_poller = false;
        const uint32_t c3_gw = epilogue_warp_idx * kNumSMs + sm_idx;
        constexpr uint32_t c3_nw = kNumSMs * kNumEpilogueWarps;
#endif
        const auto c3_gate_wait = [&]() {
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            if constexpr (kIBC3Gate == 1) {
                if (sm_idx == 0 and epilogue_warp_idx == 0) {
                    if (lane_idx + 1 < ib_num_nodes) {
                        const auto hdr = reinterpret_cast<const uint64_t*>(ib_ret(ib_parity, ib_node_of_slot(lane_idx)) + kIBRetHdr);
                        IB_POLL(ptx::ld_volatile(hdr) != ib_epoch, __nanosleep(500), "c3_gate_hdr", lane_idx, 0);
                    }
                    __syncwarp();
                    if (lane_idx == 0) ptx::st_rel(c3_gate, ib_epoch);
                } else if (lane_idx == 0) {
                    IB_POLL(ptx::ld_acq_gpu(c3_gate) != ib_epoch, __nanosleep(200), "c3_gate", 0, 0);
                }
                __syncwarp();
            }
#endif
        };
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        if (c3_poller) {
            // Mode 2 poller: lane-strided over (remote node, chunk); one 4-byte load of the chunk's last scale per probe
            const uint32_t nch = (num_tokens + kIBChunk - 1) / kIBChunk;
            const uint32_t items = (ib_num_nodes - 1) * nch;
            DG_STATIC_ASSERT(kIBFp8Combine, "C3_HDR_GATE=2 probes the fp8 record scales");
            uint32_t done_bits = 0;  // bit j: item lane_idx + 32 * j (<= 32 items per lane)
            const uint32_t my_items = items > lane_idx ? (items - lane_idx + 31) / 32 : 0;
            const auto all_mine = my_items >= 32 ? 0xffffffffu : (1u << my_items) - 1u;
            const long long t0 = clock64();
            while (__any_sync(0xffffffffu, done_bits != all_mine)) {
                for (uint32_t j = 0; j < my_items; ++ j) {
                    if ((done_bits >> j) & 1u) continue;
                    const uint32_t it = lane_idx + 32 * j, nr = it / nch, c = it % nch;
                    const uint32_t last = cute::min(c * kIBChunk + kIBChunk, num_tokens) - 1;
                    const auto sp = reinterpret_cast<const uint32_t*>(ib_rec_s(ib_ret(ib_parity, ib_node_of_slot(nr)), last, kHidden - 128));
                    if (ptx::ld_volatile(sp) != 0xffffffffu) {
                        ptx::st_rel(c3_gate + ib_node_of_slot(nr) * 64 + c, ib_epoch);
                        done_bits |= 1u << j;
                    }
                }
#ifndef DG_MEGA_MOE_IB_C3_GATE_SLEEP_NS
#define DG_MEGA_MOE_IB_C3_GATE_SLEEP_NS 500
#endif
                if (done_bits != all_mine) __nanosleep(DG_MEGA_MOE_IB_C3_GATE_SLEEP_NS);
                if (clock64() - t0 > (1ll << 34)) {
                    printf("OPUS_HANG c3_tailgate rank=%u lane=%u done=%x epoch=%llu\n", sym_buffer.rank_idx, lane_idx, done_bits, (unsigned long long) ib_epoch);
                    asm volatile("trap;");
                }
            }
        }
#endif
        if (not kC3Split) c3_gate_wait();
        for (uint32_t c3_pass = kC3Split ? 0u : 1u; c3_pass < 2; ++ c3_pass)
        for (uint32_t token_chunk_idx = (c3_pass == 1 and kC3Split ? (c3_gate_wait(), 0u) : 0u) + (c3_poller ? num_tokens * kNumChunks : c3_gw); token_chunk_idx < num_tokens * kNumChunks; token_chunk_idx += c3_nw) {
            TT_CLOCK(tt_c0);
            const bool c3_local = not kC3Split or c3_pass == 0;
            const uint32_t token_idx = token_chunk_idx / kNumChunks;
            const uint32_t chunk_idx = token_chunk_idx % kNumChunks;

            // Read top-k slot indices: each lane reads one slot, then broadcast via exchange
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token_idx * kNumTopk + lane_idx]) :
                (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            // C3: slots on other nodes arrive as one reduced record per node (below), not through the slot buffers
            const bool ib_slot_remote = lane_idx < kNumTopk and stored_topk_slot_idx >= 0 and
                is_remote(static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank);
            const uint32_t ib_node_bit = ib_slot_remote ? (1u << (static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank / ib_lsa)) : 0u;
            const uint32_t ib_nodes_hit = __reduce_or_sync(0xffffffffu, ib_node_bit);
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0 and not ib_slot_remote);
            const bool is_routed = lane_idx < kNumTopk and stored_topk_slot_idx >= 0 and not ib_slot_remote;
#else
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            // Wait for the ranks of the selected experts to finish their L2 writes
            const bool is_routed = lane_idx < kNumTopk and stored_topk_slot_idx >= 0;
#endif
            const uint32_t chunk_byte_offset = chunk_idx * kNumChunkBytes;
            float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
            const auto c3_scr = c3_scratch + static_cast<int64_t>(token_idx) * kHidden + chunk_idx * (kHidden / kNumChunks);
            if (c3_local) {
            const auto peer_ready_ptr = workspace.get_combine_ready_grid_idx_ptr(
                is_routed ? static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank : 0);
            comm::wait_until([&]() { return __all_sync(0xffffffff, not is_routed or ptx::ld_acq_sys(peer_ready_ptr) == grid_idx); },
                             [&]() { printf("DeepGEMM combine peers timeout: rank=%u, token=%u\n", sym_buffer.rank_idx, token_idx); });

            // Move mask and load
            uint32_t mask = total_mask;
            bool pair_two[2] = {false, false};
            const auto move_mask_and_load = [&](const uint32_t& i) {
                if (mask) {
                    // Shared slot goes first, then routed slots in top-k order
                    const uint32_t slot_idx = (mask >> kNumTopk) ? kNumTopk : __ffs(mask) - 1;
                    mask ^= 1 << slot_idx;
                    uint32_t slot2 = 0xffffffffu;
                    if (kC3Pair and mask) {
                        slot2 = (mask >> kNumTopk) ? kNumTopk : __ffs(mask) - 1;
                        mask ^= 1 << slot2;
                    }
                    pair_two[i] = slot2 != 0xffffffffu;

                    // Load
                    if (cute::elect_one_sync()) {
                        const auto src_ptr = math::advance_ptr<uint8_t>(
                            buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                                .get_data_buffer(token_idx).get_base_ptr(),
                            chunk_byte_offset);
                        ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                        if (slot2 != 0xffffffffu) {
                            const auto src2 = math::advance_ptr<uint8_t>(
                                buffer.combine_token_buffer.get_rank_buffer(slot2).get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer2[i], src2, combine_load_barriers[i], kNumChunkBytes);
                        }
                        ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes * (slot2 != 0xffffffffu ? 2u : 1u));
                    }
                    __syncwarp();
                    return true;
                }
                return false;
            };

            // Load the first selection
            bool do_reduce = move_mask_and_load(load_stage_idx);

            // Accumulate all top-k contributions for this chunk in float registers
            while (do_reduce) {
                // Prefetch next top-k into the buffer while current is being accumulated
                do_reduce = move_mask_and_load(load_stage_idx ^ 1);

                // Accumulate
                combine_load_barriers[load_stage_idx]->wait(combine_phase);
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                    const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                }
                if (kC3Pair and pair_two[load_stage_idx]) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer2[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                }
                cutlass::arch::fence_view_async_shared();
                combine_phase ^= load_stage_idx;
                load_stage_idx ^= 1;
            }

            TT_MAX(15, lane_idx == 0);  // source combine: a token-chunk's local part done, last one
            if (kC3Split) {
                // C3_SPLIT pass 0: park the fp32 local sum (lane-private layout, read back by the same lane in pass 1)
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    auto dst = reinterpret_cast<float4*>(c3_scr + (j * 32 + lane_idx) * 8);
                    dst[0] = make_float4(reduced[j * 4 + 0].x, reduced[j * 4 + 0].y, reduced[j * 4 + 1].x, reduced[j * 4 + 1].y);
                    dst[1] = make_float4(reduced[j * 4 + 2].x, reduced[j * 4 + 2].y, reduced[j * 4 + 3].x, reduced[j * 4 + 3].y);
                }
                continue;
            }
            } else {
                // C3_SPLIT pass 1: the local sum this lane parked in pass 0
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    const auto src = reinterpret_cast<const float4*>(c3_scr + (j * 32 + lane_idx) * 8);
                    const float4 a = src[0], b = src[1];
                    reduced[j * 4 + 0] = {a.x, a.y}; reduced[j * 4 + 1] = {a.z, a.w};
                    reduced[j * 4 + 2] = {b.x, b.y}; reduced[j * 4 + 3] = {b.z, b.w};
                }
            }
            TT_CLOCK(tt_c_loc);
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            // C3: one record per remote node; every record of every remote node is polled (rule A: the proxy writes
            // all of them), added if this token has experts there, then re-armed for the launch after next
            for (uint32_t nr = 0; nr + 1 < ib_num_nodes; ++ nr) {
                const uint32_t n = ib_node_of_slot(nr);
                const bool hit = (ib_nodes_hit >> n) & 1u;
                // STREAM_COMBINE: node n's proxy returned our records in its wave order (pos2, computed in D0)
                const uint32_t rec_pos = ib_sc_on(num_tokens) ? ib_tab[kIBTabPos2 + nr * kIBTabT + token_idx] : token_idx;
                if (kIBC3Gate == 2 and c3_g2) {
                    if (lane_idx == 0)
                        IB_POLL(ptx::ld_volatile(c3_gate + n * 64 + rec_pos / kIBChunk) != ib_epoch, __nanosleep(100), "c3_cflag", n, rec_pos);  // a hint only: every record is still sentinel-checked (rule A), so no acquire
                    __syncwarp();
                }
                if constexpr (kIBFp8Combine) {
                    const auto rbase = ib_ret(ib_parity, n);
                    // All loads of the record part first (independent), then the sentinel checks
                    uint64_t qv[kNumUint4PerLane];
                    uint32_t sv[kNumUint4PerLane];
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const uint32_t col = chunk_byte_offset / 2 + (j * 32 + lane_idx) * 8;
                        qv[j] = ptx::ld_volatile(reinterpret_cast<uint64_t*>(ib_rec_q(rbase, rec_pos, col)));
                        sv[j] = ptx::ld_volatile(reinterpret_cast<uint32_t*>(ib_rec_s(rbase, rec_pos, col)));
                    }
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
                    const bool tt_rmiss0 = __any_sync(0xffffffffu, qv[0] == ~0ull);
#endif
                    if constexpr (kIBC3Repoll) {
                        // batch re-poll: one round trip per round for all still-sentinel pieces
                        const long long rp_t0 = clock64();
                        while (true) {
                            bool miss = false;
                            #pragma unroll
                            for (uint32_t j = 0; j < kNumUint4PerLane; ++ j)
                                miss |= (qv[j] == ~0ull) or (sv[j] == 0xffffffffu);
                            if (not __any_sync(0xffffffffu, miss))
                                break;
#if defined(DG_MEGA_MOE_IB_BACKOFF) and DG_MEGA_MOE_IB_BACKOFF
                            __nanosleep(IB_BACKOFF_NS);
#endif
                            #pragma unroll
                            for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                                const uint32_t col = chunk_byte_offset / 2 + (j * 32 + lane_idx) * 8;
                                if (qv[j] == ~0ull) qv[j] = ptx::ld_volatile(reinterpret_cast<uint64_t*>(ib_rec_q(rbase, rec_pos, col)));
                                if (sv[j] == 0xffffffffu) sv[j] = ptx::ld_volatile(reinterpret_cast<uint32_t*>(ib_rec_s(rbase, rec_pos, col)));
                            }
                            if (clock64() - rp_t0 > (1ll << 34)) {
                                printf("OPUS_HANG c3_repoll rank=%u sm=%u warp=%u n=%u token=%u epoch=%llu\n", sym_buffer.rank_idx, sm_idx, warp_idx, n, token_idx, (unsigned long long) ib_epoch);
                                asm volatile("trap;");
                            }
                        }
                    }
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const uint32_t col = chunk_byte_offset / 2 + (j * 32 + lane_idx) * 8;
                        const auto qptr = reinterpret_cast<uint64_t*>(ib_rec_q(rbase, rec_pos, col));
                        const auto sptr = reinterpret_cast<uint32_t*>(ib_rec_s(rbase, rec_pos, col));
                        uint64_t q = qv[j];
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
                        const bool tt_rmiss = j == 0 and tt_rmiss0;
#endif
                        TT_ADD(18, tt_rmiss and lane_idx == 0, 1);
                        IB_POLL(q == ~0ull, q = ptx::ld_volatile(qptr), "c3_q", n, token_idx);
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
                        if (tt_rmiss and nr == 0 and rec_pos / kIBChunk < 64) {
                            __syncwarp();
                            if (lane_idx == 0) { TT_NOW(t_); atomicMax(reinterpret_cast<unsigned long long*>(tl_tseen + rec_pos / kIBChunk), static_cast<unsigned long long>(t_)); }
                        }
#endif
                        IB_TRACE_MAX(21, lane_idx == 0 and j == 0);  // source: a return record seen, last one
                        uint32_t sbits = sv[j];
                        IB_POLL(sbits == 0xffffffffu, sbits = ptx::ld_volatile(sptr), "c3_s", n, token_idx);
                        if (hit) {
                            const float scale = __uint_as_float(sbits);
                            const auto qs = reinterpret_cast<const __nv_fp8x2_storage_t*>(&q);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l) {
                                const float2 f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(qs[l], __NV_E4M3)));
                                reduced[j * kNumElemsPerUint4 + l].x += f.x * scale;
                                reduced[j * kNumElemsPerUint4 + l].y += f.y * scale;
                            }
                        }
                        if constexpr (not kIBOracle) *qptr = ~0ull;
                    }
                    __syncwarp();
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const uint32_t col = chunk_byte_offset / 2 + (j * 32 + lane_idx) * 8;
                        if (col % 128 == 0 and not kIBOracle)
                            *reinterpret_cast<uint32_t*>(ib_rec_s(rbase, rec_pos, col)) = 0xffffffffu;
                    }
                    continue;
                }
                const auto rec = ib_ret(ib_parity, n) + static_cast<int64_t>(rec_pos) * kIBRecBytes + chunk_byte_offset;
                uint4 vv[kNumUint4PerLane];
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j)
                    vv[j] = ib_ldv4(rec + (j * 32 + lane_idx) * 16);
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    const auto ptr = rec + (j * 32 + lane_idx) * 16;
                    uint4 v = vv[j];
                    IB_POLL(ib_is_sentinel4(v), v = ib_ldv4(ptr), "c3_bf16", n, token_idx);
                    if (hit) {
                        const auto b = reinterpret_cast<const nv_bfloat162*>(&v);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], b[l]);
                    }
                    if constexpr (not kIBOracle) *reinterpret_cast<uint4*>(ptr) = uint4{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
                }
            }
#endif

            // Cast
            #pragma unroll
            for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                uint4 casted;
                auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                #pragma unroll
                for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                    casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                // Wait share memory release and write
                if (j == 0) {
                    ptx::tma_store_wait<0>();
                    __syncwarp();
                }
                ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                               casted.x, casted.y, casted.z, casted.w);
            }
            __syncwarp();

            // TMA store the token chunk
            if (cute::elect_one_sync()) {
                cute::tma_store_fence();
                ptx::tma_store_1d(
                    math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                    combine_store_buffer, kNumChunkBytes);
                cute::tma_store_arrive();
            }
            __syncwarp();
            {
                TT_CLOCK(tt_c1);
                TT_ADD(21, lane_idx == 0, 1);
                TT_ADD(22, lane_idx == 0, tt_c1 - tt_c0);
                TT_ADD(25, lane_idx == 0, tt_c1 - tt_c_loc);  // sum: local part done -> token-chunk stored
            }
        }
        TT_MAX(9, epilogue_thread_idx == 0);
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        // Rule H: the return header of every proxy is awaited (even with zero tokens), then re-armed
        if (sm_idx == 0 and epilogue_thread_idx + 1 < ib_num_nodes and not (kIBNoRetHdr and num_tokens > 0)) {
            const uint32_t n = ib_node_of_slot(epilogue_thread_idx);
            const auto hdr = reinterpret_cast<uint64_t*>(ib_ret(ib_parity, n) + kIBRetHdr);
            comm::wait_until([&]() { return (ib_oracle_now and ptx::ld_volatile(hdr) != ~0ull) or ptx::ld_volatile(hdr) == ib_epoch; }, [&]() {
                printf("DeepGEMM IB return header timeout: rank=%u, node=%u, header=%llx, epoch=%llu\n",
                       sym_buffer.rank_idx, n, (unsigned long long) ptx::ld_volatile(hdr), (unsigned long long) ib_epoch);
            });
            if (not kIBOracle) *hdr = ~0ull;
        }
        TT_MAX(10, sm_idx == 0 and epilogue_thread_idx == 0);
#endif
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE and defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        IB_TRACE(9, sm_idx == 0 and epilogue_thread_idx == 0);
        if (sm_idx == 0 and epilogue_thread_idx == 0 and ib_epoch % 16 == 0) {
            const auto* t = workspace.signals->ib_trace;
            const auto us = [&](const uint32_t& i) { return ((int64_t) t[i] - (int64_t) t[16]) / 1e3; };
            printf("OPUS_TRACE rank=%u epoch=%llu us from kernel start: d0_start=%.1f staged=%.1f posted=%.1f pre_pull_barrier=%.1f d1_hdr=%.1f d1_slots=%.1f d1_done=%.1f counts=%.1f pull_end=%.1f l2_flags=%.1f fwd_done=%.1f combine_end=%.1f prev_cleanup_end=%.1f | max over SMs: hdr=%.1f scan=%.1f counts=%.1f pull_end=%.1f rem_row_last=%.1f ring_wait_max=%.1f ring_wait_last_end=%.1f xpost_last=%.1f sfpost=%.1f\n",
                   sym_buffer.rank_idx, (unsigned long long) ib_epoch, us(23), us(24), us(0), us(2), us(20), us(22), us(19), us(3), us(4), us(5), us(6), us(9), t[18] / 1e3,
                   us(25), us(26), us(28), us(27), us(31),
                   ((t[29] >> 40) == ib_epoch ? static_cast<double>(t[29] & ((1ull << 40) - 1)) / 1e3 : 0.0), us(30), us(7), us(8));
            // device printf takes at most 32 arguments: the rest goes on a second line
            printf("OPUS_TRACE2 rank=%u epoch=%llu | remote rows not landed at first pull: %llu of %llu | pre_fetch=%.1f fetched=%.1f | rec_put_last=%.1f comb_start=%.1f rec_seen_last=%.1f | prev_launch_end: dispatch=%.1f epilogue=%.1f | fwd_items_by_epilogue=%llu\n",
                   sym_buffer.rank_idx, (unsigned long long) ib_epoch, (unsigned long long) t[10], (unsigned long long) t[11], us(12), us(13), us(14), us(15), us(21),
                   static_cast<double>(static_cast<int64_t>(t[35])) / 1e3, static_cast<double>(static_cast<int64_t>(t[36])) / 1e3, (unsigned long long) t[37]);
            printf("OPUS_TRACE3 rank=%u epoch=%llu | overlap: xpost_first=%.1f xpost_last=%.1f l1_first=%.1f rem_row_first=%.1f rem_row_last=%.1f rem_row_last_e0=%.1f ret_chunk_first=%.1f ret_chunk_last=%.1f ws=%d sc=%d\n",
                   sym_buffer.rank_idx, (unsigned long long) ib_epoch, us(46), us(47), us(40), us(41), us(42), us(43), us(44), us(45), (int) kIBWaveSort, (int) kIBStreamCombine);
            printf("OPUS_TRACE4 rank=%u epoch=%llu | counts_early: posted=%.1f landed_max=%.1f mismatches=%llu ce=%d | split_scan tables_done=%.1f ss=%d cim=%d\n",
                   sym_buffer.rank_idx, (unsigned long long) ib_epoch, us(32), us(38), (unsigned long long) t[39], (int) kIBCountsEarly,
                   us(32), (int) kIBSplitScan, (int) kIBCountsInMeta);
            {
                const auto mu = [&](const uint32_t& i) { return ((int64_t) ib_sl_tr[i] - (int64_t) t[16]) / 1e3; };
                printf("OPUS_META rank=%u epoch=%llu | meta put: before=%.1f after_put=%.1f after_doorbell=%.1f | x pieces: first_issued=%.1f last_issued=%.1f | xorder=%d\n",
                       sym_buffer.rank_idx, (unsigned long long) ib_epoch, mu(24), mu(25), mu(26), mu(27), mu(28), (int) kIBXOrder);
            }
#if defined(DG_MEGA_MOE_IB_PROBE) and DG_MEGA_MOE_IB_PROBE
            {
                const uint64_t* const pr = tl_tseen + 128;
                const auto pd = [&](const uint32_t& i) { return static_cast<long long>(pr[i]) == static_cast<long long>(0x8000000000000000ull) ? -1e9 : static_cast<double>(static_cast<int64_t>(pr[i])) / 1e3; };
                const auto pu = [&](const uint32_t& i) { return ((int64_t) pr[i] - (int64_t) t[16]) / 1e3; };
                printf("OPUS_PROBE rank=%u epoch=%llu start48=%llu ctx=%d step=%d | seen-sent us n0: %.2f %.2f %.2f %.2f n1: %.2f %.2f %.2f %.2f n2: %.2f %.2f %.2f %.2f | hdr seen: %.1f %.1f %.1f\n",
                       sym_buffer.rank_idx, (unsigned long long) ib_epoch, (unsigned long long) (t[16] & ((1ull << 48) - 1)), (int) DG_MEGA_MOE_IB_PROBE_CTX, (int) DG_MEGA_MOE_IB_PROBE,
                       pd(0), pd(1), pd(2), pd(3), pd(4), pd(5), pd(6), pd(7), pd(8), pd(9), pd(10), pd(11), pu(48), pu(49), pu(50));
                printf("OPUS_PROBE2 rank=%u epoch=%llu | issue before/after us n0: %.2f/%.2f %.2f/%.2f %.2f/%.2f %.2f/%.2f n1: %.2f/%.2f %.2f/%.2f %.2f/%.2f %.2f/%.2f n2: %.2f/%.2f %.2f/%.2f %.2f/%.2f %.2f/%.2f\n",
                       sym_buffer.rank_idx, (unsigned long long) ib_epoch,
                       pu(16), pu(20), pu(17), pu(21), pu(18), pu(22), pu(19), pu(23), pu(24), pu(28), pu(25), pu(29), pu(26), pu(30), pu(27), pu(31),
                       pu(32), pu(36), pu(33), pu(37), pu(34), pu(38), pu(35), pu(39));
            }
#endif
            if constexpr (kIBWSPost > 0 or kIBOrd > 0) {
                const auto mu = [&](const uint32_t& i) { return ((int64_t) ib_sl_tr[i] - (int64_t) t[16]) / 1e3; };
                printf("OPUS_TRACEW rank=%u epoch=%llu | ws_post: first_piece_issued=%.1f last_chunk_staged=%.1f last_piece_issued=%.1f order_published=%.1f w_put=%.1f | order: enter=%.1f zeroed=%.1f keyed=%.1f placed=%.1f fenced=%.1f\n",
                       sym_buffer.rank_idx, (unsigned long long) ib_epoch, mu(38), mu(39), mu(28), mu(40), mu(42), mu(43), mu(44), mu(45), mu(46), mu(47));
            }
            if constexpr (kIBSST) {
                const auto mu = [&](const uint32_t& i) { return ((int64_t) ib_sl_tr[i] - (int64_t) t[16]) / 1e3; };
                printf("OPUS_SST rank=%u epoch=%llu | sender: staged=%.1f put_issued=%.1f | receiver (slowest source): counts_landed=%.1f counts_published=%.1f tables_done=%.1f | build: enter=%.1f zeroed=%.1f counted=%.1f prefix=%.1f scattered=%.1f | wave: published=%.1f seen=%.1f\n",
                       sym_buffer.rank_idx, (unsigned long long) ib_epoch, mu(29), mu(30), mu(31), us(26), us(32), mu(33), mu(34), mu(35), mu(36), mu(37), mu(40), mu(41));
            }
            if constexpr (kIBScanSlices > 1) {
                const auto su = [&](const uint32_t& i) { return ((int64_t) ib_sl_tr[i] - (int64_t) t[16]) / 1e3; };
                printf("OPUS_SLICES rank=%u epoch=%llu P=%u | j0 %.1f %.1f %.1f %.1f %.1f | j1 %.1f %.1f %.1f %.1f %.1f | j2 %.1f %.1f %.1f %.1f %.1f | j3 %.1f %.1f %.1f %.1f %.1f | idx_miss=%llu w_miss=%llu\n",
                       sym_buffer.rank_idx, (unsigned long long) ib_epoch, kIBScanSlices,
                       su(0), su(1), su(2), su(3), su(4), su(5), su(6), su(7), su(8), su(9),
                       su(10), su(11), su(12), su(13), su(14), su(15), su(16), su(17), su(18), su(19),
                       (unsigned long long) ib_sl_tr[20], (unsigned long long) ib_sl_tr[21]);
            }
            // 8-us histograms (bucket i = [8i, 8i+8) us from kernel start), then reset for the next printed launch
            for (uint32_t h = 0; h < 4; ++ h) {
                char buf[64 * 6 + 1]; uint32_t n = 0;
                for (uint32_t b = 0; b < 64; ++ b) {
                    uint32_t v = ptx::ld_volatile(&workspace.signals->ib_hist[h][b]);
                    workspace.signals->ib_hist[h][b] = 0;
                    char tmp[6]; uint32_t m = 0;
                    do { tmp[m ++] = static_cast<char>('0' + v % 10); v /= 10; } while (v and m < 5);
                    while (m) buf[n ++] = tmp[-- m];
                    buf[n ++] = b == 63 ? '\0' : ',';
                }
                printf("OPUS_HIST rank=%u epoch=%llu h=%u %s\n", sym_buffer.rank_idx, (unsigned long long) ib_epoch, h, buf);
            }
            const auto* tt = tl_ttrace + 32;  // previous launch
            const auto pu = [&](const uint32_t& i) { return static_cast<double>(static_cast<int64_t>(tt[i])) / 1e3; };
            printf("OPUS_TRACE5 rank=%u epoch=%llu | prev launch tail: l2r_last=%.1f l2s_last=%.1f fwd_part_last=%.1f fwd_item_last=%.1f put_pre_last=%.1f put_post_last=%.1f fwd_exit=%.1f epi_presync_min=%.1f epi_presync=%.1f c3_start_min=%.1f c3_end=%.1f hdr_done=%.1f clean_start=%.1f clean_bar=%.1f prev_disp_end=%.1f prev_epi_end=%.1f\n",
                   sym_buffer.rank_idx, (unsigned long long) ib_epoch, pu(0), pu(1), pu(2), pu(3), pu(4), pu(5), pu(6), pu(13), pu(7), pu(8), pu(9), pu(10), pu(11), pu(12),
                   static_cast<double>(static_cast<int64_t>(t[35])) / 1e3, static_cast<double>(static_cast<int64_t>(t[36])) / 1e3);
            {
                const auto* tc = tl_tchunk + 64;
                const auto cu = [&](const uint32_t& i) { return static_cast<double>(static_cast<int64_t>(tc[i])) / 1e3; };
                printf("OPUS_TRACE7 rank=%u epoch=%llu | prev launch chunk puts 0-15: c0=%.1f c1=%.1f c2=%.1f c3=%.1f c4=%.1f c5=%.1f c6=%.1f c7=%.1f c8=%.1f c9=%.1f c10=%.1f c11=%.1f c12=%.1f c13=%.1f c14=%.1f c15=%.1f\n",
                       sym_buffer.rank_idx, (unsigned long long) ib_epoch, cu(0), cu(1), cu(2), cu(3), cu(4), cu(5), cu(6), cu(7), cu(8), cu(9), cu(10), cu(11), cu(12), cu(13), cu(14), cu(15));
                const auto* tsn = tl_tseen + 64;
                const auto su = [&](const uint32_t& i) { return static_cast<double>(static_cast<int64_t>(tsn[i])) / 1e3; };
                printf("OPUS_TRACE9 rank=%u epoch=%llu | prev launch chunk landed (missed records) 0-15: s0=%.1f s1=%.1f s2=%.1f s3=%.1f s4=%.1f s5=%.1f s6=%.1f s7=%.1f s8=%.1f s9=%.1f s10=%.1f s11=%.1f s12=%.1f s13=%.1f s14=%.1f s15=%.1f\n",
                       sym_buffer.rank_idx, (unsigned long long) ib_epoch, su(0), su(1), su(2), su(3), su(4), su(5), su(6), su(7), su(8), su(9), su(10), su(11), su(12), su(13), su(14), su(15));
                printf("OPUS_TRACEA rank=%u epoch=%llu | prev launch chunk landed (missed records) 16-31: s16=%.1f s17=%.1f s18=%.1f s19=%.1f s20=%.1f s21=%.1f s22=%.1f s23=%.1f s24=%.1f s25=%.1f s26=%.1f s27=%.1f s28=%.1f s29=%.1f s30=%.1f s31=%.1f\n",
                       sym_buffer.rank_idx, (unsigned long long) ib_epoch, su(16), su(17), su(18), su(19), su(20), su(21), su(22), su(23), su(24), su(25), su(26), su(27), su(28), su(29), su(30), su(31));
                printf("OPUS_TRACE8 rank=%u epoch=%llu | prev launch chunk puts 16-31: c16=%.1f c17=%.1f c18=%.1f c19=%.1f c20=%.1f c21=%.1f c22=%.1f c23=%.1f c24=%.1f c25=%.1f c26=%.1f c27=%.1f c28=%.1f c29=%.1f c30=%.1f c31=%.1f\n",
                       sym_buffer.rank_idx, (unsigned long long) ib_epoch, cu(16), cu(17), cu(18), cu(19), cu(20), cu(21), cu(22), cu(23), cu(24), cu(25), cu(26), cu(27), cu(28), cu(29), cu(30), cu(31));
            }
            printf("OPUS_TRACEB rank=%u epoch=%llu | prev launch end: dispatch_last=%.1f epilogue_last=%.1f other_last=%.1f after_sysfence=%.1f epi_after_sysfence=%.1f\n",
                   sym_buffer.rank_idx, (unsigned long long) ib_epoch, pu(27), pu(28), pu(29), pu(30), pu(31));
            printf("OPUS_TRACE6 rank=%u epoch=%llu | prev launch tail counts: fwd_items=%llu fwd_part_miss=%llu fwd_x_miss=%llu fwd_item_ns=%.0f c3_rec_miss=%llu c3_tokens=%llu c3_token_ns=%.0f ret_puts=%llu fwd_items_epi=%llu rearm_done=%.1f c3_loc_last=%.1f c3_after_loc_ns=%.0f puts_after_l2=%llu | knobs join=%d batch=%d batch_d=%d pf=%d c3e=%d defer=%d repoll=%d retg=%d split=%d\n",
                   sym_buffer.rank_idx, (unsigned long long) ib_epoch, (unsigned long long) tt[16], (unsigned long long) tt[17], (unsigned long long) tt[19],
                   tt[16] > tt[17] ? static_cast<double>(tt[20]) / static_cast<double>(tt[16] - tt[17]) : 0.0,
                   (unsigned long long) tt[18], (unsigned long long) tt[21], tt[21] > 0 ? static_cast<double>(tt[22]) / static_cast<double>(tt[21]) : 0.0,
                   (unsigned long long) tt[23], (unsigned long long) tt[24], pu(14), pu(15),
                   tt[21] > 0 ? static_cast<double>(tt[25]) / static_cast<double>(tt[21]) : 0.0, (unsigned long long) tt[26],
                   (int) kIBFwdJoin, (int) DG_MEGA_MOE_IB_FWD_BATCH, (int) DG_MEGA_MOE_IB_FWD_BATCH_D, (int) kIBFwdPrefetch, (int) kIBC3Early, (int) kIBDeferRearm, (int) kIBC3Repoll, (int) kIBRetGroup, (int) kIBC3Split);
        }
#endif
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE and not (defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB)
        // NVLink-only build (DG_MEGA_MOE_NVL_TRACE=1): the same phase points as OPUS_TRACE, for like-for-like attribution
        if (sm_idx == 0 and epilogue_thread_idx == 0) {
            uint64_t t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_));
            auto* t = workspace.signals->ib_trace;
            const uint64_t launch = ++ t[1];
            const auto us = [&](const uint32_t& i) { return ((int64_t) t[i] - (int64_t) t[16]) / 1e3; };
            if (launch % 16 == 0)
                printf("NVL_TRACE rank=%u launch=%llu us from kernel start: barrier=%.1f l2_done=%.1f end=%.1f | max over SMs: pre_fetch=%.1f counts=%.1f pull_end=%.1f comb_start=%.1f\n",
                       sym_buffer.rank_idx, (unsigned long long) launch, us(2), us(5), ((int64_t) t_ - (int64_t) t[16]) / 1e3, us(12), us(28), us(27), us(15));
        }
#endif
        IB_TRACE_MAX(34, epilogue_thread_idx == 0);  // epilogue warps done, slowest SM (reported by the next launch)
    }
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
    {
        // End-of-kernel gap: each warp's last instruction by role (27 dispatch, 28 epilogue, 29 TMA / MMA / scheduler), then the
        // same after a system-scope fence (30 any warp, 31 epilogue warps): a long fence = outstanding writes still draining
        const uint32_t eslot = warp_idx < kNumDispatchWarps ? 27u : (warp_idx < kNumDispatchWarps + 4 ? 29u : 28u);
        const uint32_t emask = __activemask();
        __syncwarp(emask);
        TT_MAX(eslot, lane_idx == static_cast<uint32_t>(__ffs(emask) - 1));
        __threadfence_system();
        __syncwarp(emask);
        TT_MAX(30, lane_idx == static_cast<uint32_t>(__ffs(emask) - 1));
        TT_MAX(31, eslot == 28u and lane_idx == static_cast<uint32_t>(__ffs(emask) - 1));
    }
#endif
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

#if defined(DG_MEGA_MOE_IB_SPLIT) and DG_MEGA_MOE_IB_SPLIT
template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M_L1, uint32_t STORE_BLOCK_M_L2,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumRingTokens,
    uint32_t kNumSFRingTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    typename weight_dtype_t,
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M,
    uint32_t kNumSharedSFTokens = layout::get_num_max_shared_sf_tokens(kNumMaxTokensPerRank),
    typename task_info_t = sched::TaskInfo<kHasShared>
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_impl(void* y,
                            int* cumulative_local_expert_recv_stats,
                            const uint32_t num_tokens,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights_sf,
                            const uint8_t* sm_locality_domains) {
    // heavy copy: the former scan SMs (SST receive) and the 8 SMs below them (builders, early-weight put, poster)
    if (blockIdx.x + (kNumRanks - sym_buffer.num_lsa_ranks) + 8 >= kNumSMs)
        sm100_fp8_fp4_mega_moe_body<true, kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumSharedExperts, kNumTopk, BLOCK_M, BLOCK_N, BLOCK_K, STORE_BLOCK_M_L1, STORE_BLOCK_M_L2, SF_BLOCK_M, SF_BLOCK_N, kNumRingTokens, kNumSFRingTokens, kNumStages, kNumBytesPerPull, kNumDispatchThreads, kNumNonEpilogueThreads, kNumEpilogueThreads, kNumSMs, kNumRanks, kActivationClamp, kFastMath, weight_dtype_t>(
            y, cumulative_local_expert_recv_stats, num_tokens, sym_buffer, tensor_map_l1_acts, tensor_map_l1_acts_sf, tensor_map_l1_weights, tensor_map_l1_weights_sf, tensor_map_l1_output, tensor_map_l2_acts, tensor_map_l2_acts_sf, tensor_map_l2_weights, tensor_map_l2_weights_sf, tensor_map_shared_l1_acts, tensor_map_shared_l1_acts_sf, tensor_map_shared_l1_weights, tensor_map_shared_l1_weights_sf, tensor_map_shared_l1_output, tensor_map_shared_l2_acts, tensor_map_shared_l2_acts_sf, tensor_map_shared_l2_weights, tensor_map_shared_l2_weights_sf, sm_locality_domains);
    else
        sm100_fp8_fp4_mega_moe_body<false, kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumSharedExperts, kNumTopk, BLOCK_M, BLOCK_N, BLOCK_K, STORE_BLOCK_M_L1, STORE_BLOCK_M_L2, SF_BLOCK_M, SF_BLOCK_N, kNumRingTokens, kNumSFRingTokens, kNumStages, kNumBytesPerPull, kNumDispatchThreads, kNumNonEpilogueThreads, kNumEpilogueThreads, kNumSMs, kNumRanks, kActivationClamp, kFastMath, weight_dtype_t>(
            y, cumulative_local_expert_recv_stats, num_tokens, sym_buffer, tensor_map_l1_acts, tensor_map_l1_acts_sf, tensor_map_l1_weights, tensor_map_l1_weights_sf, tensor_map_l1_output, tensor_map_l2_acts, tensor_map_l2_acts_sf, tensor_map_l2_weights, tensor_map_l2_weights_sf, tensor_map_shared_l1_acts, tensor_map_shared_l1_acts_sf, tensor_map_shared_l1_weights, tensor_map_shared_l1_weights_sf, tensor_map_shared_l1_output, tensor_map_shared_l2_acts, tensor_map_shared_l2_acts_sf, tensor_map_shared_l2_weights, tensor_map_shared_l2_weights_sf, sm_locality_domains);
}
#endif

} // namespace deep_gemm
