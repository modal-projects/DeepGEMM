#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_jit/utils/env.hpp>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "sm100.hpp"

namespace deep_gemm {

struct MegaMoEConfig {
    // Block tiling
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int store_block_m_l1, store_block_m_l2;

    // SF block sizes (UTCCP 128-aligned)
    int sf_block_m, sf_block_n;

    // Ring capacity and SF ring token count
    int num_ring_tokens;
    int num_sf_ring_tokens;

    // Swizzle modes for TMA descriptors
    int swizzle_acts_mode, swizzle_weights_mode;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    // Dispatch pull config
    int num_bytes_per_pull;

    // Host overrides (DG_MEGA_MOE_HOST_*): pull buffers (and mbarriers) per dispatch warp, extra reserved smem bytes
    int num_pull_bufs = 1, smem_extra = 0;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEConfig& config) {
        os << "MegaMoEConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m_l1=" << config.store_block_m_l1 << ", store_block_m_l2=" << config.store_block_m_l2
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", num_ring_tokens=" << config.num_ring_tokens
           << ", num_sf_ring_tokens=" << config.num_sf_ring_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", num_bytes_per_pull=" << config.num_bytes_per_pull
           << ", num_pull_bufs=" << config.num_pull_bufs << ", smem_extra=" << config.smem_extra << ")";
        return os;
    }
};

// Host-side layout overrides for the FP8xFP4 Mega MoE kernel (dev loop): DG_MEGA_MOE_HOST_<NAME>=<positive integer>.
// Unset, empty or "0" keeps the built-in value, so the default configuration is unchanged. These decide the launch
// (block dim, dynamic smem bytes) and the kernel's template arguments, hence they live on the host; the resolved values
// reach the kernel source as template arguments and DG_MEGA_MOE_HOSTCFG_* defines (part of the JIT cache key).
// A value that cannot be honoured throws: there is no silent fallback.
//   DG_MEGA_MOE_HOST_DISPATCH_THREADS  dispatch threads per CTA (128; multiple of 32, >= 64)
//   DG_MEGA_MOE_HOST_EPILOGUE_THREADS  epilogue threads per CTA (256; multiple of 128)
//   DG_MEGA_MOE_HOST_PULL_BUFS         smem pull buffers + mbarriers per dispatch warp (1; 1..4)
//   DG_MEGA_MOE_HOST_PULL_BYTES        bytes per pull buffer (hidden bytes halved until <= 8192; must divide them, % 16 == 0)
//   DG_MEGA_MOE_HOST_STAGES            TMA pipeline stages (the maximum that fits; 2 <= value <= that maximum)
//   DG_MEGA_MOE_HOST_SMEM_EXTRA        extra smem bytes reserved for the kernel header's own use (0; rounded up to 1024)
static int get_mega_moe_host_override(const char* name, const int& default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr or *value == '\0')
        return default_value;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (*end != '\0' or parsed < 0 or parsed > (1 << 20))
        DG_HOST_UNREACHABLE(std::format("{}={}: expected a non-negative integer", name, value));
    return parsed == 0 ? default_value : static_cast<int>(parsed);
}

#define DG_MEGA_MOE_HOST_CHECK(cond, ...) \
do { \
    if (not (cond)) \
        DG_HOST_UNREACHABLE(std::string("Mega MoE host override rejected (" #cond "): ") + std::format(__VA_ARGS__)); \
} while (0)

// We need to distribute work across locality domains, so we must use the whole GPU
static int get_num_sms_for_mega_moe() {
    const auto num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms == jit->device.get_num_sms());
    return num_sms;
}

static MmaKind parse_mma_kind(const std::string& mma_type_str) {
    if (mma_type_str == "bf16xbf16")
        return MmaKind::BF16;
    DG_HOST_ASSERT(mma_type_str == "fp8xfp4" or mma_type_str == "fp8xfp8");
    return MmaKind::MXFP8FP4;
}

static int get_num_mma_elem_bytes(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::BF16 ? 2 : 1;
}

static bool is_mma_with_sf(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::MXFP8FP4;
}

static std::tuple<int, int, int, int, int, int> get_block_config_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens,
    const MmaKind& mma_kind) {
    // Expected tokens per expert, plus a one-sigma routing margin for the tile choice
    const float num_expected_tokens = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const float num_covered_tokens = num_expected_tokens + std::sqrt(num_expected_tokens);

    // Every M block costs roughly 128 extra rows (its tasks reload the weight tiles), so use the fewest blocks
    // per expert and the smallest tile covering them: one block up to 192 rows, then blocks of up to 240 rows.
    // A single 240-row block per expert only pays off with many local experts (about 14 or more blocks per rank).
    int block_m = num_expected_tokens <= 10 ? 16 : 32;
    if (num_expected_tokens > 24) {
        int num_blocks = static_cast<int>(std::ceil(num_covered_tokens / 240));
        if (num_blocks == 1 and num_covered_tokens > 192 and num_experts / num_ranks < 14)
            num_blocks = 2;
        for (const int& candidate: {64, 128, 192, 240}) {
            block_m = candidate;
            if (num_blocks * candidate >= num_covered_tokens)
                break;
        }
    }

    // Use smaller, asymmetric store blocks at FP8xFP4 prefill to save shared memory for sufficient TMA pipeline depth
    const int store_block_m_l1 = block_m <= 16 ? 8 : block_m <= 64 ? 16 : block_m <= 192 ? 32 :
                                 mma_kind == MmaKind::MXFP8FP4 ? 24 : 40;
    const int store_block_m_l2 = (block_m == 240 and mma_kind == MmaKind::MXFP8FP4) ? 8 : store_block_m_l1;

    const int block_k = 128 / get_num_mma_elem_bytes(mma_kind);

    // Check whether our `block_m` lies in `kCandidateBlockM`
    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );

    // Return configs: 2-CTA clusters and 2 epilogue warpgroups
    return {2, block_m, store_block_m_l1, store_block_m_l2, block_k, 2 * 128};
}

static std::pair<int, int> get_pipeline_config_for_mega_moe(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k, 
    const int& num_bytes_per_pull, const int& store_block_m_l1, const int& store_block_m_l2,
    const int& sf_block_m, const int& sf_block_n, const int& gran_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const MmaKind& mma_kind,
    const int& num_pull_bufs = 1, const int& smem_extra = 0, const int& num_stages_override = 0) {
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumTMAStoreStages = 2;
    const int num_mma_elem_bytes = get_num_mma_elem_bytes(mma_kind);

    // Always multicast on A
    const int load_block_m = block_m / 2;

    // Dispatch region
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(num_bytes_per_pull), num_dispatch_warps * num_pull_bufs, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 output staging and L2 BF16 staging.
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m_l1 * (block_n / 2) * kNumTMAStoreStages * get_num_mma_elem_bytes(mma_kind);
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m_l2 * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

    // Schedule task payloads
    constexpr int kNumScheduleStages = 2;
    const int smem_task_info = kNumScheduleStages * static_cast<int>(sizeof(sched::TaskInfo<true>));

    // Barriers (stage-independent): dispatch + tensor memory full/empty + combine (2 per epilogue warp)
    // + schedule task publish full/empty barriers.
    const int smem_barriers = (num_dispatch_warps * num_pull_bufs + kNumEpilogueStages * 2 + num_epilogue_warps * 2 + kNumScheduleStages * 2) * 8;

    // Amax warp-pair reduction buffer for SwiGLU's cross-warp amax exchange.
    const int smem_amax_reduction = is_mma_with_sf(mma_kind) ?
        store_block_m_l1 * num_epilogue_warps * static_cast<int>(sizeof(float)) : 0;

    // Tensor memory pointer
    const int smem_tmem_ptr = 4;

    // SF is aligned to UTCCP 128-element granularity
    const int smem_sfa_per_stage = is_mma_with_sf(mma_kind) ? sf_block_m * (block_k / gran_k) : 0;
    const int smem_sfb_per_stage = is_mma_with_sf(mma_kind) ? sf_block_n * (block_k / gran_k) : 0;

    // Per-stage: A tile + B tile + optional SF tiles + full/empty barriers.
    const int smem_a_size_per_stage = load_block_m * block_k * num_mma_elem_bytes;
    const int smem_b_size_per_stage = block_n * block_k * num_mma_elem_bytes;
    DG_HOST_ASSERT(smem_a_size_per_stage % kSmemAlignment == 0);
    DG_HOST_ASSERT(smem_b_size_per_stage % kSmemAlignment == 0);
    const int smem_stage_barriers = 2 * 8;
    const int smem_size_per_stage = smem_a_size_per_stage + smem_b_size_per_stage + smem_sfa_per_stage + smem_sfb_per_stage + smem_stage_barriers;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers +
        smem_task_info + smem_tmem_ptr + align(smem_extra, kSmemAlignment);

    // Select maximum number of stages
    const int num_max_stages = (smem_capacity - smem_fixed) / smem_size_per_stage;
    DG_MEGA_MOE_HOST_CHECK(num_max_stages >= 2,
        "{} fixed smem bytes + 2 x {} per stage exceed the capacity {} ({} dispatch warps x {} pull buffers x {} bytes, {} extra)",
        smem_fixed, smem_size_per_stage, smem_capacity, num_dispatch_warps, num_pull_bufs, num_bytes_per_pull, smem_extra);
    DG_MEGA_MOE_HOST_CHECK(num_stages_override == 0 or (2 <= num_stages_override and num_stages_override <= num_max_stages),
        "DG_MEGA_MOE_HOST_STAGES={} outside [2, {}]", num_stages_override, num_max_stages);
    const int num_stages = num_stages_override == 0 ? num_max_stages : num_stages_override;

    return {num_stages, smem_fixed + num_stages * smem_size_per_stage};
}

static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_ring_tokens,
    const int& num_sf_ring_tokens,
    const MmaKind& mma_kind) {

    // Block config
    const auto [cluster_size, block_m, store_block_m_l1, store_block_m_l2, block_k, num_default_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens, mma_kind);
    DG_HOST_ASSERT(num_ring_tokens % block_m == 0);
    const int block_n = 128;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const auto [sf_block_m, sf_block_n] = is_mma_with_sf(mma_kind) ?
        SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, block_n, MmaKind::MXFP8FP4) : std::pair(0, 0);
    // NOTES: FP8 activations, FP8 weights, and FP4 weights unpacked to 8-bit in smem all use 128B swizzle
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;
    const int gran_k = 32;

    // Thread layout
    // NOTES: the DG_MEGA_MOE_HOST_* overrides apply to the FP8xFP4 kernel only (the BF16 kernel keeps the built-in layout)
    const bool allow_overrides = mma_kind == MmaKind::MXFP8FP4;
    const auto override_or = [&](const char* name, const int& default_value) {
        return allow_overrides ? get_mega_moe_host_override(name, default_value) : default_value;
    };
    const int num_dispatch_threads = override_or("DG_MEGA_MOE_HOST_DISPATCH_THREADS", 128);
    const int num_non_epilogue_threads = 128;
    const int num_epilogue_threads = override_or("DG_MEGA_MOE_HOST_EPILOGUE_THREADS", num_default_epilogue_threads);
    DG_MEGA_MOE_HOST_CHECK(num_dispatch_threads % 32 == 0 and num_dispatch_threads >= 64,
        "DG_MEGA_MOE_HOST_DISPATCH_THREADS={}: need a multiple of 32, at least 64", num_dispatch_threads);
    DG_MEGA_MOE_HOST_CHECK(num_epilogue_threads % 128 == 0,
        "DG_MEGA_MOE_HOST_EPILOGUE_THREADS={}: need a multiple of 128", num_epilogue_threads);
    DG_MEGA_MOE_HOST_CHECK(num_dispatch_threads + num_non_epilogue_threads + num_epilogue_threads <= 1024,
        "{} + {} + {} threads per CTA exceed 1024", num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads);

    // Pull: divide token bytes by 2 until <= num_max_pull_bytes
    const int num_max_pull_bytes = is_mma_with_sf(mma_kind) ? 8192 : 4096;
    const int num_token_bytes = hidden * get_num_mma_elem_bytes(mma_kind);
    int num_bytes_per_pull = num_token_bytes;
    while (num_bytes_per_pull > num_max_pull_bytes) {
        DG_HOST_ASSERT(num_bytes_per_pull % 2 == 0);
        num_bytes_per_pull /= 2;
    }
    num_bytes_per_pull = override_or("DG_MEGA_MOE_HOST_PULL_BYTES", num_bytes_per_pull);
    DG_MEGA_MOE_HOST_CHECK(num_bytes_per_pull % 16 == 0 and num_token_bytes % num_bytes_per_pull == 0,
        "DG_MEGA_MOE_HOST_PULL_BYTES={}: need a multiple of 16 that divides the {} token bytes", num_bytes_per_pull, num_token_bytes);
    const int num_pull_bufs = override_or("DG_MEGA_MOE_HOST_PULL_BUFS", 1);
    DG_MEGA_MOE_HOST_CHECK(num_pull_bufs <= 4, "DG_MEGA_MOE_HOST_PULL_BUFS={}: at most 4", num_pull_bufs);
    const int smem_extra = align(override_or("DG_MEGA_MOE_HOST_SMEM_EXTRA", 0), 1024);
    const int num_stages_override = override_or("DG_MEGA_MOE_HOST_STAGES", 0);

    // Pipeline
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k, num_bytes_per_pull, store_block_m_l1, store_block_m_l2,
        sf_block_m, sf_block_n, gran_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        mma_kind, num_pull_bufs, smem_extra, num_stages_override);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m_l1, store_block_m_l2,
        sf_block_m, sf_block_n,
        num_ring_tokens, is_mma_with_sf(mma_kind) ? num_sf_ring_tokens : 0,
        swizzle_acts_mode, swizzle_weights_mode,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        num_bytes_per_pull,
        num_pull_bufs, smem_extra
    };

    // Print configs for the first time
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = std::format(
            "MegaMoEConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
