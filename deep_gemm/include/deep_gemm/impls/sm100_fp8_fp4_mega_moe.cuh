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
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
#define IB_TRACE(i, cond) do { if (cond) { uint64_t t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); workspace.signals->ib_trace[i] = t_; } } while (0)
// max over every SM that reaches the point (absolute %globaltimer, so no per-launch reset is needed)
#define IB_TRACE_MAX(i, cond) do { if (cond) { uint64_t t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); atomicMax(reinterpret_cast<unsigned long long*>(workspace.signals->ib_trace + (i)), static_cast<unsigned long long>(t_)); } } while (0)
#else
#define IB_TRACE(i, cond) do {} while (0)
#define IB_TRACE_MAX(i, cond) do {} while (0)
#endif

    // Mega MoE over IB, opus design (REDESIGN_OPUS.md): counterpart forwarding. Every cross-node byte moves between
    // GPUs with the same local index. A source puts slices of its input region into a sentinel-filled mirror on its
    // counterpart (the proxy), which publishes counts / slot entries to its NVLink peers exactly like an NVLink
    // source would; peers pull rows from the proxy's mirror over NVLink; L2 rows for remote sources go to the
    // proxy's combine buffer; the proxy reduces them and returns one record per token. Readiness = data != sentinel.
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
    const auto& ib_dev_comm = *reinterpret_cast<const ncclDevComm_t*>(sym_buffer.gin_dev_comm);
    const auto ib_window = reinterpret_cast<ncclWindow_t>(sym_buffer.gin_window);
    const uint64_t ib_epoch = ptx::ld_volatile(workspace.get_ib_epoch_ptr()) + 1;
    const uint32_t ib_parity = static_cast<uint32_t>(ib_epoch & 1);
    const uint32_t ib_lsa = sym_buffer.num_lsa_ranks;
    const uint32_t ib_my_node = sym_buffer.rank_idx / ib_lsa;
    const uint32_t ib_num_nodes = kNumRanks / ib_lsa;
    const auto is_remote = [&](const uint32_t& r) { return not sym_buffer.is_lsa(r); };
    const auto ib_cp = [&](const uint32_t& r, const uint32_t& n) { return n * ib_lsa + r % ib_lsa; };  // counterpart of r on node n
    const auto ib_node_of_slot = [&](const uint32_t& nr) { return nr < ib_my_node ? nr : nr + 1; };     // nr-th remote node
    const auto make_gin = [&](const uint32_t& ctx) { return comm::GinIB(ib_dev_comm, ib_window, sym_buffer.base, ctx); };
    // Bisect knob for exclusive contexts: DG_MEGA_MOE_IB_XSITES = bitmask of posting sites that use their exclusive
    // index + DG_MEGA_MOE_IB_SHARING (1 meta, 2 meta header, 4 x/SF pieces, 8 combine chunks, 16 return header when a
    // node sent nothing, 32 return header after the last chunk); the other sites use GPU sharing on a separate pool of
    // contexts above the exclusive range (so they never touch an exclusive context).
#ifndef DG_MEGA_MOE_IB_XSITES
#define DG_MEGA_MOE_IB_XSITES 63
#endif
    const auto make_gin_site = [&](const uint32_t& site, const uint32_t& xctx, const uint32_t& orig, const uint32_t& xbase) {
        if (DG_MEGA_MOE_IB_SHARING >= 1 and (DG_MEGA_MOE_IB_XSITES & site))
            return comm::GinIB(ib_dev_comm, ib_window, sym_buffer.base, xctx);
        if (DG_MEGA_MOE_IB_SHARING >= 1)  // shared pool above the exclusive range
            return comm::GinIB(ib_dev_comm, ib_window, sym_buffer.base,
                               xbase + orig % cute::max(1u, static_cast<uint32_t>(DG_MEGA_MOE_IB_CONTEXTS) - xbase), 0);
        return comm::GinIB(ib_dev_comm, ib_window, sym_buffer.base, orig);
    };
    // Mirror layout = the upstream input region layout: x [T][H] fp8, SF [T][H/32], top-k idx [T][K] i64, weights [T][K] f32
    constexpr int64_t kIBT = kNumMaxTokensPerRank;
    constexpr int64_t kIBTokBytes = kHidden, kIBSFBytes = kHidden / 32;
    constexpr int64_t kIBOffSF = kIBT * kIBTokBytes;
    constexpr int64_t kIBOffIdx = kIBOffSF + kIBT * kIBSFBytes;
    constexpr int64_t kIBOffW = kIBOffIdx + kIBT * kNumTopk * 8;
    constexpr int64_t kIBOffHdr = math::constexpr_align<int64_t>(kIBOffW + kIBT * kNumTopk * 4, 128);
    constexpr int64_t kIBMirrorStride = kIBOffHdr + 128;
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
    // Proxy-local slot table: entry (node-local rank, local expert, slot) -> pair index; the expert hosts read it over
    // NVLink in their pull (one load per row) instead of the proxy writing ~K x T scattered entries over NVLink
    constexpr int64_t kIBPTabStride = static_cast<int64_t>(kNumRanks) * kNumExpertsPerRank * kIBT * 4;
    constexpr int64_t kIBNodeStride = 2 * kIBMirrorStride + 2 * kIBRetStride + kIBPCombStride + kIBFStageStride + kIBPTabStride;
    constexpr uint32_t kIBChunk = 32;  // tokens per dispatch x put and per return put
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
    const auto hdr_orig = [&](const uint32_t& nr) { return kIBHdrCtx ? ib_R + nr : nr + 2; };
    const auto piece_orig = [&](const uint32_t& q) {
        return kIBHdrCtx ? 2 * ib_R + q % cute::max(1u, static_cast<uint32_t>(DG_MEGA_MOE_IB_CONTEXTS) - 2 * ib_R) : q + 1; };
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
#define IB_BACKOFF(n) do { if ((n) > 8) __nanosleep(256); } while (0)
#else
#define IB_BACKOFF(n) do {} while (0)
#endif
#define IB_POLL(cond, reload, tag, a, b) do { const long long t0_ = clock64(); uint32_t n_ = 0; while (cond) { IB_BACKOFF(++ n_); reload; \
        if (clock64() - t0_ > (1ll << 34)) { printf("OPUS_HANG %s rank=%u sm=%u warp=%u lane=%u a=%u b=%u epoch=%llu\n", tag, \
            sym_buffer.rank_idx, sm_idx, warp_idx, lane_idx, (unsigned) (a), (unsigned) (b), (unsigned long long) ib_epoch); asm volatile("trap;"); } } } while (0)
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
        Allocator().allocate(kNumTmemCols, &shared_storage.tmem_ptr_in_smem);
    }
    cute::cluster_sync();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();
    IB_TRACE(16, sm_idx == 0 and thread_idx == 0);

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
    const auto ib_forward_combine = [&](const uint32_t& global_warp, const uint32_t& num_global_warps, const bool& is_leader) {
        const auto fwd_grid_idx = ptx::get_grid_idx() + 1;
        for (uint32_t nr = 0; nr + 1 < ib_num_nodes; ++ nr) {
            const uint32_t n = ib_node_of_slot(nr);
            const uint32_t s = ib_cp(sym_buffer.rank_idx, n);
            const uint32_t nt = ptx::ld_volatile(&workspace.signals->ib_o_nt[n]);
            const uint32_t nchunks = (nt + kIBChunk - 1) / kIBChunk;
            const auto ret_dst = ib_ret(ib_parity, ib_my_node);
            if (nt == 0) {
                if (is_leader)
                    make_gin_site(16, ctx_rhdr(nr, nr), nr, ib_xbase).put_value(reinterpret_cast<uint64_t*>(ret_dst + kIBRetHdr), ib_epoch, s);
                continue;
            }
            const auto mir = ib_mirror(ib_parity, n);
            const auto fst = ib_fstage(n);
            const auto pc = ib_pcomb(n);
            const auto idx = reinterpret_cast<const uint64_t*>(mir + kIBOffIdx);
            // Work item = (token, quarter of the hidden dim): 4 warps per token, every load of an item issued back to back
            constexpr uint32_t kSplit = 4;
            constexpr uint32_t kPieces = kHidden * 2 / (32 * 16);  // 16-byte bf16 units per lane over the row
            constexpr uint32_t kPPI = kPieces / kSplit;
            DG_STATIC_ASSERT(kPieces * 32 * 16 == kHidden * 2 and kPPI * kSplit == kPieces, "IB forward piece");
            constexpr uint32_t kXUnitsPerItem = kIBTokBytes / 16 / kSplit;
            DG_STATIC_ASSERT(kXUnitsPerItem * 16 * kSplit == kIBTokBytes, "IB forward x split");
            for (uint32_t item = global_warp; item < nt * kSplit; item += num_global_warps) {
                const uint32_t t = item / kSplit, qtr = item % kSplit;
                const int e = lane_idx < kNumTopk ? static_cast<int>(static_cast<int64_t>(ptx::ld_volatile(idx + t * kNumTopk + lane_idx))) : -1;
                const uint32_t r = e >= 0 ? static_cast<uint32_t>(e) / kNumExpertsPerRank : 0u;
                const bool on = e >= 0 and r / ib_lsa == ib_my_node;
                const uint32_t on_mask = __ballot_sync(0xffffffffu, on);
                if constexpr (not kIBReduceOnArrival) {
                    const auto flag_ptr = workspace.get_combine_ready_grid_idx_ptr(r);
                    comm::wait_until([&]() { return __all_sync(0xffffffffu, not on or ptx::ld_acq_sys(flag_ptr) == fwd_grid_idx); },
                                     [&]() { printf("DeepGEMM IB forward-combine flags timeout: rank=%u, source=%u, token=%u\n", sym_buffer.rank_idx, s, t); });
                }
                // Rule A: this item's part of the token's x row (and, for quarter 0, its SF row) has landed
                {
                    constexpr uint32_t kU = (kXUnitsPerItem + 31) / 32;
                    const auto xrow = mir + static_cast<int64_t>(t) * kIBTokBytes + static_cast<int64_t>(qtr) * kXUnitsPerItem * 16;
                    uint4 v[kU];
                    #pragma unroll
                    for (uint32_t u = 0; u < kU; ++ u)
                        v[u] = u * 32 + lane_idx < kXUnitsPerItem ? ib_ldv4(xrow + (u * 32 + lane_idx) * 16) : uint4{0, 0, 0, 0};
                    #pragma unroll
                    for (uint32_t u = 0; u < kU; ++ u)
                        IB_POLL(ib_is_sentinel4(v[u]), v[u] = ib_ldv4(xrow + (u * 32 + lane_idx) * 16), "c2_x", s, t);
                    if (qtr == 0 and lane_idx < kIBSFBytes / 16) {
                        const auto sfp = mir + kIBOffSF + static_cast<int64_t>(t) * kIBSFBytes + lane_idx * 16;
                        IB_POLL(ib_is_sentinel4(ib_ldv4(sfp)), (void) 0, "c2_sf", s, t);
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
                    const uint32_t k = __ffs(m) - 1;
                    m &= m - 1;
                    const auto row = pc + (static_cast<int64_t>(k) * kIBT + t) * kHidden * 2;
                    uint4 v[kPPI];
                    #pragma unroll
                    for (uint32_t i = 0; i < kPPI; ++ i) {
                        const auto ptr = row + ((qtr * kPPI + i) * 32 + lane_idx) * 16;
                        if constexpr (kIBReduceOnArrival) v[i] = ib_ldv4(ptr);
                        else v[i] = __ldcg(reinterpret_cast<const uint4*>(ptr));
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < kPPI; ++ i) {
                        const auto ptr = row + ((qtr * kPPI + i) * 32 + lane_idx) * 16;
                        if constexpr (kIBReduceOnArrival) {
                            IB_POLL(ib_is_sentinel4(v[i]), v[i] = ib_ldv4(ptr), "c2_pcomb", t, k);
                            *reinterpret_cast<uint4*>(ptr) = uint4{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
                        }
                        const auto b = reinterpret_cast<const nv_bfloat162*>(&v[i]);
                        #pragma unroll
                        for (uint32_t l = 0; l < 4; ++ l)
                            ptx::accumulate(acc[i][l], b[l]);
                    }
                }
                const auto rec = fst + static_cast<int64_t>(t) * kIBRecBytes;
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
                        *reinterpret_cast<uint2*>(rec + col) = q;
                        if (lane_idx % 16 == 0)
                            *reinterpret_cast<float*>(rec + kHidden + col / 128 * 4) = scale;
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
                const uint32_t c = t / kIBChunk, size = cute::min(kIBChunk, nt - c * kIBChunk);
                uint32_t state = 0;
                if (lane_idx == 0) {
                    __threadfence();
                    if (ib_atom_add_acq_rel(&workspace.signals->ib_o_fchunk[ib_parity][n][c], 1u) + 1 == size * kSplit)
                        state = ib_atom_add_acq_rel(&workspace.signals->ib_o_fdone[ib_parity][n], 1u) + 1 == nchunks ? 2u : 1u;
                }
                state = __shfl_sync(0xffffffffu, state, 0);
                if (state > 0 and lane_idx == 0) {
                    const int64_t off = static_cast<int64_t>(c) * kIBChunk * kIBRecBytes;
                    make_gin_site(8, ctx_chunk(nr, c, c + nr), c + nr, ib_xbase).put(ret_dst + off, fst + off, static_cast<uint64_t>(size) * kIBRecBytes, s);
                }
                if (state == 2 and lane_idx == 1)
                    make_gin_site(32, ctx_rhdr(nr, c + nr + 1), c + nr + 1, ib_xbase).put_value(reinterpret_cast<uint64_t*>(ret_dst + kIBRetHdr), ib_epoch, s);
                __syncwarp();
            }
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
                if (thread_idx < layout::MegaMoESignals::kIBOMaxNodes)
                    workspace.signals->ib_o_fdone[ib_parity ^ 1][thread_idx] = 0;
            }
        }
#endif
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
#endif

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

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
        scheduler.fetch_expert_recv_count();
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        __threadfence_system();  // acquire: pairs with the proxy's fence before its count add (entries, per-source counts)
        IB_TRACE(3, sm_idx == 0 and thread_idx == 0);
        IB_TRACE_MAX(28, thread_idx == 0);  // counts, slowest SM
#endif

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
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

            // Read source token-topk index (written by remote dispatch via NVLink)
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
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
            const auto src_base_ptr = ib_rem ?
                sym_buffer.map(ib_mir + static_cast<int64_t>(src_token_idx) * kIBTokBytes, ib_proxy) :
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
            if (ib_rem) {
                // Every 16-byte unit is re-read until it is not the 0xFF sentinel, then stored into the L1 ring
                DG_STATIC_ASSERT(kIBTokBytes % (32 * 16) == 0, "IB row poll");
                constexpr uint32_t kUnits = kIBTokBytes / (32 * 16), kBatch = kIBPullWhole ? kUnits : 5;
                #pragma unroll
                for (uint32_t u0 = 0; u0 < kUnits; u0 += kBatch) {
                    uint4 v[kBatch];
                    #pragma unroll
                    for (uint32_t b = 0; b < kBatch; ++ b)
                        if (u0 + b < kUnits) v[b] = ib_ldv4(src_base_ptr + ((u0 + b) * 32 + lane_idx) * 16);
                    #pragma unroll
                    for (uint32_t b = 0; b < kBatch; ++ b) {
                        if (u0 + b < kUnits) {
                            const uint32_t o = ((u0 + b) * 32 + lane_idx) * 16;
                            IB_POLL(ib_is_sentinel4(v[b]), v[b] = ib_ldv4(src_base_ptr + o), "pull_x", current_rank_in_expert_idx, src_token_idx);
                            *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(dst_base_ptr) + o) = v[b];
                        }
                    }
                }
                asm volatile("fence.proxy.async.global;" ::: "memory");  // generic stores, read by the GEMM's TMA
                IB_TRACE_MAX(31, lane_idx == 0);  // last remote row fully landed + copied, slowest warp
            }
#else
            constexpr bool ib_rem = false;
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
            if (not ib_rem and cute::elect_one_sync()) {
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
            const auto weight = ib_rem ?
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
                    uint32_t sf = ib_rem ? (kIBPullWhole ? ib_sf_pre[i] : ptx::ld_volatile(remote_sf_ptr + j)) : remote_sf_ptr[j];
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
                if (not ib_rem)
                    issue_and_wait_pull_store(kNumChunks - 1);
                const bool is_last_token = (token_idx == expert_end_idx - 1);
                ptx::red_add_rel(
                    workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks), 
                    is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u
                );
            }
            __syncwarp();
        }

#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        IB_TRACE(4, sm_idx == 0 and thread_idx == 0);
        IB_TRACE_MAX(27, thread_idx == 0);  // pull end, slowest SM
        if constexpr (kIBReduceOnArrival) {
            // The dispatch warps are idle until the cleanup: reduce forwarded tokens as their partials land
            ib_forward_combine(warp_idx * kNumSMs + sm_idx, kNumSMs * kNumDispatchWarps, sm_idx == 0 and thread_idx == 0);
            IB_TRACE(6, sm_idx == 0 and thread_idx == 0);
        }
#endif
        // Clean workspace for the next usage, and also do cumulative stats
        // NOTES: it is overlapped with combine reduction epilogue
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

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
        // E: re-arm the mirrors of this parity. Every node-local pull from them is done (barrier above); the
        // forward-combine polled every row (fdone), so every byte the NIC wrote this launch has landed (rule A)
        for (uint32_t nr = 0; nr + 1 < ib_num_nodes; ++ nr) {
            const uint32_t n = ib_node_of_slot(nr);
            const uint32_t nt = ptx::ld_volatile(&workspace.signals->ib_o_nt[n]);
            const uint32_t nchunks = (nt + kIBChunk - 1) / kIBChunk;
            if (thread_idx == 0)
                IB_POLL(ptx::ld_acq(&workspace.signals->ib_o_fdone[ib_parity][n]) < nchunks, (void) 0, "e_fdone", n, nchunks);
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            const auto mir = ib_mirror(ib_parity, n);
            const uint32_t gtid = sm_idx * kNumDispatchThreads + thread_idx;
            constexpr uint32_t kGThreads = kNumSMs * kNumDispatchThreads;
            const uint4 ones = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
            const uint4 idx_sentinel = {0u, 0x80000000u, 0u, 0x80000000u};
            for (int64_t u = gtid; u < static_cast<int64_t>(nt) * kIBTokBytes / 16; u += kGThreads)
                reinterpret_cast<uint4*>(mir)[u] = ones;
            for (int64_t u = gtid; u < static_cast<int64_t>(nt) * kIBSFBytes / 16; u += kGThreads)
                reinterpret_cast<uint4*>(mir + kIBOffSF)[u] = ones;
            for (int64_t u = gtid; u < static_cast<int64_t>(nt) * kNumTopk * 8 / 16; u += kGThreads)
                reinterpret_cast<uint4*>(mir + kIBOffIdx)[u] = idx_sentinel;
            for (int64_t u = gtid; u < static_cast<int64_t>(nt) * kNumTopk; u += kGThreads)
                reinterpret_cast<uint32_t*>(mir + kIBOffW)[u] = 0xffffffffu;
            if (gtid == 0)
                *reinterpret_cast<uint64_t*>(mir + kIBOffHdr) = ~0ull;
        }
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
    } else if (warp_idx == kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for tokens with SFA
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
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
            } else if (task_info.block_phase == sched::BlockPhase::SharedLinear2) {
                const auto ptr = workspace.get_shared_l2_full_count_ptr(block_idx);
                const auto num_expected_blocks = (SHARED_L2_SHAPE_K / BLOCK_N) * 2;
                while (ptx::ld_acq(ptr) != num_expected_blocks);
            }

            L2KBlockDependency l2_k_block_dependency(workspace.get_l2_full_mask_ptr(ring_block_idx), pool_block_idx / kNumRingBlocks);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                if (task_info.block_phase == sched::BlockPhase::Linear2)
                    l2_k_block_dependency.wait(k_block_idx);

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
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for weights with SF
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
        IB_TRACE(23, sm_idx == kNumSMs - 1 and epilogue_thread_idx == 0);
#if defined(DG_MEGA_MOE_IB_DEBUG_D0) and DG_MEGA_MOE_IB_DEBUG_D0
        if (sm_idx == kNumSMs - 1 and epilogue_thread_idx == 0 and ib_epoch <= 2)
            printf("OPUS_D0 rank=%u epoch=%llu num_tokens=%u enter\n", sym_buffer.rank_idx, (unsigned long long) ib_epoch, num_tokens);
#endif
        if (sm_idx == kNumSMs - 1 and num_tokens == kIBT) {
            // Full batch: every top-k / weight row is polled by the proxy, so the input slices go out as they are
            // (no staging): one put for top-k idx + weights (contiguous in the input region), one for the header
            if (epilogue_warp_idx == 0 and lane_idx % 16 + 1 < ib_num_nodes) {
                const uint32_t nr = lane_idx % 16, n = ib_node_of_slot(nr);
                const auto gin = lane_idx < 16 ? make_gin_site(1, ctx_meta(nr, nr), nr, ib_xbase) : make_gin_site(2, ctx_mhdr(nr, hdr_orig(nr)), hdr_orig(nr), ib_xbase);
                const auto dst = ib_mirror(ib_parity, ib_my_node);
                if (lane_idx < 16)
                    gin.put(dst + kIBOffIdx, ib_input_base + kIBOffIdx, static_cast<uint64_t>(kIBT * kNumTopk * 12), ib_cp(sym_buffer.rank_idx, n));
                else
                    gin.put_value(reinterpret_cast<uint64_t*>(dst + kIBOffHdr), (ib_epoch << 32) | num_tokens, ib_cp(sym_buffer.rank_idx, n));
            }
            __syncwarp();
            IB_TRACE(24, epilogue_thread_idx == 0);
            if (epilogue_thread_idx == 0)
                ptx::st_rel_sys(&workspace.signals->ib_o_meta_posted, ib_epoch);
        } else if (sm_idx == kNumSMs - 1) {
            const auto stg = ib_mirror(ib_parity, ib_my_node);
            const auto in_idx = reinterpret_cast<const uint64_t*>(ib_input_base + kIBOffIdx);
            const auto in_w = reinterpret_cast<const uint32_t*>(ib_input_base + kIBOffW);
            constexpr uint32_t kSB = 4;  // independent loads first, then the stores
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
                }
            }
            if (epilogue_thread_idx == 0)
                *reinterpret_cast<uint64_t*>(stg + kIBOffHdr) = (ib_epoch << 32) | num_tokens;
            __threadfence();
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            IB_TRACE(24, epilogue_thread_idx == 0);
            if (epilogue_warp_idx == 0 and lane_idx + 1 < ib_num_nodes) {
                const uint32_t n = ib_node_of_slot(lane_idx);
                make_gin_site(1, ctx_meta(lane_idx, lane_idx), lane_idx, ib_xbase).put(ib_mirror(ib_parity, ib_my_node) + kIBOffIdx, stg + kIBOffIdx,
                                       static_cast<uint64_t>(kIBOffHdr + 8 - kIBOffIdx), ib_cp(sym_buffer.rank_idx, n));
            }
            __syncwarp();
            // The bulk x / SF puts are posted only after the meta, so the meta is not queued behind them on the wire
            if (epilogue_thread_idx == 0)
                ptx::st_rel_sys(&workspace.signals->ib_o_meta_posted, ib_epoch);
        }
#if defined(DG_MEGA_MOE_IB_DEBUG_D0) and DG_MEGA_MOE_IB_DEBUG_D0
        if (sm_idx == kNumSMs - 1 and epilogue_thread_idx == 0 and ib_epoch <= 2)
            printf("OPUS_D0 rank=%u epoch=%llu meta posted\n", sym_buffer.rank_idx, (unsigned long long) ib_epoch);
#endif
        {
            const uint32_t num_x_chunks = (num_tokens + kIBChunk - 1) / kIBChunk;
            const uint32_t num_pieces = 1 + num_x_chunks;  // SF, x chunks
            for (uint32_t q = kNumSMs - 2 - sm_idx; sm_idx + 2 <= kNumSMs and q < (ib_num_nodes - 1) * num_pieces; q += kNumSMs) {
                if (epilogue_warp_idx == 0 and lane_idx == 0 and num_tokens > 0) {
                    // Meta first (x / SF posted only after the meta put): opt-in only; on GB200 it turns into ms-scale
                    // stalls (EP32 with the proxy D1, EP16 with the host scan; mechanism not identified)
#if defined(DG_MEGA_MOE_IB_META_FIRST_ALL) and DG_MEGA_MOE_IB_META_FIRST_ALL
                    constexpr bool kMetaFirst = true;
#else
                    constexpr bool kMetaFirst = false;  // measured harmful with the host scan (GB200: ms-scale stalls)
#endif
                    while (kMetaFirst and ptx::ld_acq_sys(&workspace.signals->ib_o_meta_posted) < ib_epoch);
                    const uint32_t n = ib_node_of_slot(q / num_pieces), piece = q % num_pieces;
                    int64_t off = 0, bytes = 0;
                    if (piece == 0) {
                        off = kIBOffSF, bytes = static_cast<int64_t>(num_tokens) * kIBSFBytes;
                    } else {
                        const uint32_t t0 = (piece - 1) * kIBChunk, t1 = cute::min(t0 + kIBChunk, num_tokens);
                        off = static_cast<int64_t>(t0) * kIBTokBytes, bytes = static_cast<int64_t>(t1 - t0) * kIBTokBytes;
                    }
                    make_gin_site(4, ctx_piece(q, piece_orig(q)), piece_orig(q), ib_xbase).put(ib_mirror(ib_parity, ib_my_node) + off, ib_input_base + off, static_cast<uint64_t>(bytes),
                                        ib_cp(sym_buffer.rank_idx, n));
                }
                __syncwarp();
            }
        }
        IB_TRACE(0, sm_idx == kNumSMs - 1 and epilogue_thread_idx == 0);
        // D1 (host scan; v1 had the proxy do this and push tables over NVLink, measured ~20 us slower): every GPU scans,
        // for each remote source r, r's mirror on r's counterpart in this node (over NVLink, or locally when that is
        // us) and builds its OWN tables for its own experts exactly like the upstream receiver-side state: per-source
        // counts, slot entries (local stores), then the upstream count-arrival add (kNumSMs << 32 | count). One SM's
        // epilogue warps per remote source (the last SMs; idle until the first routed task, which needs these counts).
        {
            const uint32_t num_remote = kNumRanks - ib_lsa;
            if (sm_idx + num_remote >= kNumSMs) {
                const uint32_t k = kNumSMs - 1 - sm_idx;                    // k-th remote source
                const uint32_t r = ib_node_of_slot(k / ib_lsa) * ib_lsa + k % ib_lsa;
                const uint32_t proxy = ib_cp(r, ib_my_node);
                const auto mir = sym_buffer.map(ib_mirror(ib_parity, r / ib_lsa), proxy);
                const auto hdr = reinterpret_cast<const uint64_t*>(mir + kIBOffHdr);
                // One thread polls the (NVLink-remote) header with a pause between polls: hundreds of threads spinning
                // on a proxy's HBM while its NIC writes there slows the NIC down drastically on GB200
                auto nt_smem = reinterpret_cast<uint32_t*>(&shared_storage.smem_d) + kNumExpertsPerRank;
                if (epilogue_thread_idx == 0) {
                    uint64_t h = 0;
                    comm::wait_until([&]() { h = ptx::ld_volatile(hdr); if ((h >> 32) != (ib_epoch & 0xffffffffull)) __nanosleep(200); return (h >> 32) == (ib_epoch & 0xffffffffull); }, [&]() {
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
                for (uint32_t i = epilogue_thread_idx; i < kNumExpertsPerRank; i += kNumEpilogueThreads)
                    cnt[i] = 0;
                if (proxy == sym_buffer.rank_idx and epilogue_thread_idx == 0)
                    workspace.signals->ib_o_nt[r / ib_lsa] = nt;  // the proxy's forward-combine and re-arm need it
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                // Poll (top-k idx and weights; rule A for the weights the pull reads unchecked), then slots for the
                // pairs routed to OUR experts by shared-memory atomics; entries are local stores
                constexpr uint32_t kB = 4;
                for (uint32_t i0 = 0; i0 < nt * kNumTopk; i0 += kNumEpilogueThreads * kB) {
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
                            if (e >= 0 and static_cast<uint32_t>(e) / kNumExpertsPerRank == sym_buffer.rank_idx) {
                                const uint32_t le = static_cast<uint32_t>(e) % kNumExpertsPerRank;
                                const uint32_t slot = atomicAdd_block(cnt + le, 1u);
                                *workspace.get_src_token_topk_idx_ptr(le, r, slot) = i;
                            }
                        }
                    }
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                IB_TRACE(22, epilogue_thread_idx == 0 and sm_idx == kNumSMs - 1);
                for (uint32_t i = epilogue_thread_idx; i < kNumExpertsPerRank; i += kNumEpilogueThreads) {
                    *workspace.get_expert_recv_count_ptr(r, i) = cnt[i];
                    __threadfence();
                    ptx::atomic_add(workspace.get_expert_recv_count_sum_ptr(i),
                                    (static_cast<uint64_t>(kNumSMs) << 32) | static_cast<uint64_t>(cnt[i]));
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                IB_TRACE(19, epilogue_thread_idx == 0 and sm_idx == kNumSMs - 1);
                IB_TRACE_MAX(26, epilogue_thread_idx == 0);  // scan done, slowest scanning SM
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

        // Barrier with dispatch warps, so that they can do clean workspace
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        if constexpr (not kIBReduceOnArrival) {
            ib_forward_combine(epilogue_warp_idx * kNumSMs + sm_idx, kNumSMs * kNumEpilogueWarps, sm_idx == 0 and epilogue_thread_idx == 0);
            IB_TRACE(6, sm_idx == 0 and epilogue_thread_idx == 0);
        }
#endif

        // Iterate over all token chunks, 1 token 1 topk latency: ~3 us
        const auto grid_idx = ptx::get_grid_idx() + 1;
        for (uint32_t token_chunk_idx = epilogue_warp_idx * kNumSMs + sm_idx; token_chunk_idx < num_tokens * kNumChunks; token_chunk_idx += kNumSMs * kNumEpilogueWarps) {
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
            const auto peer_ready_ptr = workspace.get_combine_ready_grid_idx_ptr(
                is_routed ? static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank : 0);
            comm::wait_until([&]() { return __all_sync(0xffffffff, not is_routed or ptx::ld_acq_sys(peer_ready_ptr) == grid_idx); },
                             [&]() { printf("DeepGEMM combine peers timeout: rank=%u, token=%u\n", sym_buffer.rank_idx, token_idx); });

            const uint32_t chunk_byte_offset = chunk_idx * kNumChunkBytes;

            // Move mask and load
            uint32_t mask = total_mask;
            const auto move_mask_and_load = [&](const uint32_t& i) {
                if (mask) {
                    // Shared slot goes first, then routed slots in top-k order
                    const uint32_t slot_idx = (mask >> kNumTopk) ? kNumTopk : __ffs(mask) - 1;
                    mask ^= 1 << slot_idx;

                    // Load
                    if (cute::elect_one_sync()) {
                        const auto src_ptr = math::advance_ptr<uint8_t>(
                            buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                                .get_data_buffer(token_idx).get_base_ptr(),
                            chunk_byte_offset);
                        ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                        ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                    }
                    __syncwarp();
                    return true;
                }
                return false;
            };

            // Load the first selection
            bool do_reduce = move_mask_and_load(load_stage_idx);

            // Accumulate all top-k contributions for this chunk in float registers
            float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
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
                cutlass::arch::fence_view_async_shared();
                combine_phase ^= load_stage_idx;
                load_stage_idx ^= 1;
            }

#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
            // C3: one record per remote node; every record of every remote node is polled (rule A: the proxy writes
            // all of them), added if this token has experts there, then re-armed for the launch after next
            for (uint32_t nr = 0; nr + 1 < ib_num_nodes; ++ nr) {
                const uint32_t n = ib_node_of_slot(nr);
                const bool hit = (ib_nodes_hit >> n) & 1u;
                if constexpr (kIBFp8Combine) {
                    const auto rec = ib_ret(ib_parity, n) + static_cast<int64_t>(token_idx) * kIBRecBytes;
                    // All loads of the record part first (independent), then the sentinel checks
                    uint64_t qv[kNumUint4PerLane];
                    uint32_t sv[kNumUint4PerLane];
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const uint32_t col = chunk_byte_offset / 2 + (j * 32 + lane_idx) * 8;
                        qv[j] = ptx::ld_volatile(reinterpret_cast<uint64_t*>(rec + col));
                        sv[j] = ptx::ld_volatile(reinterpret_cast<uint32_t*>(rec + kHidden + col / 128 * 4));
                    }
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const uint32_t col = chunk_byte_offset / 2 + (j * 32 + lane_idx) * 8;
                        const auto qptr = reinterpret_cast<uint64_t*>(rec + col);
                        const auto sptr = reinterpret_cast<uint32_t*>(rec + kHidden + col / 128 * 4);
                        uint64_t q = qv[j];
                        IB_POLL(q == ~0ull, q = ptx::ld_volatile(qptr), "c3_q", n, token_idx);
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
                        *qptr = ~0ull;
                    }
                    __syncwarp();
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const uint32_t col = chunk_byte_offset / 2 + (j * 32 + lane_idx) * 8;
                        if (col % 128 == 0)
                            *reinterpret_cast<uint32_t*>(rec + kHidden + col / 128 * 4) = 0xffffffffu;
                    }
                    continue;
                }
                const auto rec = ib_ret(ib_parity, n) + static_cast<int64_t>(token_idx) * kIBRecBytes + chunk_byte_offset;
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
                    *reinterpret_cast<uint4*>(ptr) = uint4{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
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
        }
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        // Rule H: the return header of every proxy is awaited (even with zero tokens), then re-armed
        if (sm_idx == 0 and epilogue_thread_idx + 1 < ib_num_nodes) {
            const uint32_t n = ib_node_of_slot(epilogue_thread_idx);
            const auto hdr = reinterpret_cast<uint64_t*>(ib_ret(ib_parity, n) + kIBRetHdr);
            comm::wait_until([&]() { return ptx::ld_volatile(hdr) == ib_epoch; }, [&]() {
                printf("DeepGEMM IB return header timeout: rank=%u, node=%u, header=%llx, epoch=%llu\n",
                       sym_buffer.rank_idx, n, (unsigned long long) ptx::ld_volatile(hdr), (unsigned long long) ib_epoch);
            });
            *hdr = ~0ull;
        }
#endif
#if defined(DG_MEGA_MOE_IB_TRACE) and DG_MEGA_MOE_IB_TRACE
        IB_TRACE(9, sm_idx == 0 and epilogue_thread_idx == 0);
        if (sm_idx == 0 and epilogue_thread_idx == 0 and ib_epoch % 16 == 0) {
            const auto* t = workspace.signals->ib_trace;
            const auto us = [&](const uint32_t& i) { return ((int64_t) t[i] - (int64_t) t[16]) / 1e3; };
            printf("OPUS_TRACE rank=%u epoch=%llu us from kernel start: d0_start=%.1f staged=%.1f posted=%.1f pre_pull_barrier=%.1f d1_hdr=%.1f d1_slots=%.1f d1_done=%.1f counts=%.1f pull_end=%.1f l2_flags=%.1f fwd_done=%.1f combine_end=%.1f prev_cleanup_end=%.1f | max over SMs: hdr=%.1f scan=%.1f counts=%.1f pull_end=%.1f rem_row_last=%.1f ring_wait_max=%.1f ring_wait_last_end=%.1f\n",
                   sym_buffer.rank_idx, (unsigned long long) ib_epoch, us(23), us(24), us(0), us(2), us(20), us(22), us(19), us(3), us(4), us(5), us(6), us(9), t[18] / 1e3,
                   us(25), us(26), us(28), us(27), us(31),
                   ((t[29] >> 40) == ib_epoch ? static_cast<double>(t[29] & ((1ull << 40) - 1)) / 1e3 : 0.0), us(30));
        }
#endif
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
