#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <unistd.h>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/mega_moe_ib.cuh>
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
           << ", num_bytes_per_pull=" << config.num_bytes_per_pull << ")";
        return os;
    }
};

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
    const MmaKind& mma_kind) {
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
        static_cast<int>(layout::Buffer(layout::Data(num_bytes_per_pull), num_dispatch_warps, 1).get_num_bytes()),
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
    const int smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2 + kNumScheduleStages * 2) * 8;

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
        smem_task_info + smem_tmem_ptr;

    // Select maximum number of stages
    const int num_stages = (smem_capacity - smem_fixed) / smem_size_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

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
    const auto [cluster_size, block_m, store_block_m_l1, store_block_m_l2, block_k, num_epilogue_threads] =
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
    const int num_dispatch_threads = 128;
    const int num_non_epilogue_threads = 128;

    // Pull: divide token bytes by 2 until <= num_max_pull_bytes
    const int num_max_pull_bytes = is_mma_with_sf(mma_kind) ? 8192 : 4096;
    int num_bytes_per_pull = hidden * get_num_mma_elem_bytes(mma_kind);
    while (num_bytes_per_pull > num_max_pull_bytes) {
        DG_HOST_ASSERT(num_bytes_per_pull % 2 == 0);
        num_bytes_per_pull /= 2;
    }

    // Pipeline
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k, num_bytes_per_pull, store_block_m_l1, store_block_m_l2,
        sf_block_m, sf_block_n, gran_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        mma_kind);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m_l1, store_block_m_l2,
        sf_block_m, sf_block_n,
        num_ring_tokens, is_mma_with_sf(mma_kind) ? num_sf_ring_tokens : 0,
        swizzle_acts_mode, swizzle_weights_mode,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        num_bytes_per_pull
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

// The plan of an IB build (groups that span several NVLink domains), layout/mega_moe_ib.cuh.
// One physical quantity decides the regime: how long the token rows of a rank are on the wire,
//   row_us = remote nodes x tokens x hidden bytes / link rate per GPU.
// In the receiver scan (rows in token order) no expert's first block is complete before that stream ends; the
// ordered regime (the sender routes and posts rows in first-need order) hides it behind the GEMM and pays a
// fixed chain of sender tables in front of the counts. Ordered pays from 26.2 us of rows on the wire (512 tokens
// of 5,120-byte rows to one remote node at 800G), whatever the node count, link or top-k: 512 / 256 / 171 tokens
// on 2 / 3 / 4 nodes at 800G, 256 / 128 / 86 at 400G. Measured crossovers (same-request A/B, ordered + policy against the scan):
//   800G (B300), one remote node: 15 / 22 / 25 us (top-6 / 19 / 25; +5 / +2 / -1 us at 320 / 384 / 448 tokens,
//     -13 at 512, top-19); three remote nodes: ~25 us (+5 at 128 tokens, -1 / -6 / -11 at 160 / 192 / 224, -36 /
//     -87 at 512 / 1024, top-19).
//   400G (GB200), one remote node: 22 / 26 / 29 us (+6 / -1 / -17 at 192 / 256 / 384 tokens, top-19).
// `link_gbps` is the communicator's (deep_gemm/mega/ib.py); 0 = unknown, planned as 800G.
static layout::IBPlan get_mega_moe_ib_plan(
    const int& num_ranks, const int& num_ranks_per_node,
    const int& num_tokens, const int& num_topk,
    const int& gin_backend, const int& hidden, const int& link_gbps) {
    DG_HOST_ASSERT(num_ranks_per_node > 0 and num_ranks % num_ranks_per_node == 0 and num_ranks >= num_ranks_per_node);
    const int num_nodes = num_ranks / num_ranks_per_node;
    const double row_us = static_cast<double>(num_nodes - 1) * num_tokens * hidden / ((link_gbps > 0 ? link_gbps : 800) * 125.0);

    layout::IBPlan plan;
    plan.ranks_per_node = num_ranks_per_node;
    plan.gin_backend = static_cast<uint32_t>(gin_backend);  // the communicator's (0: unknown, generic dispatch)
    const bool is_ordered = num_nodes >= 2 and row_us >= 26.2;  // us: what the sender-table chain costs
    plan.ordered = is_ordered;
    if (is_ordered) {
        // L1 leads L2 by at most this many pool blocks: -4 .. -8 us with one remote node, -10 .. -18 with three
        // (7 = 9, 11 is 3 worse); a scheduler constant, no link term
        plan.l1_lead_cap = 9;
        // Half of the clusters stop claiming tasks near the end of the routed L2 and feed the proxy sum, which
        // serves nodes - 1 return streams: -3 .. -9 us with three remote nodes, +9 .. +25 with one
        plan.retire_percent = num_nodes > 2 ? 50 : 0;
        // The trigger: the share of the routed L2 after which a retiring cluster stops. The later, the more
        // nodes share a token's experts (fewer of its partial sums are still missing near the end):
        // 100 x 0.6^(nodes / top-k) = 90 at top-19, 92 at top-25 and 71 at top-6 on four nodes (measured: 92
        // against 90 is -9 us at top-25, 1024 tokens, and -3 at 768; 71 against 90 is -5 at top-6, 1024 tokens).
        if (plan.retire_percent > 0)
            plan.retire_at_percent = static_cast<uint32_t>(std::lround(100.0 * std::pow(0.6, static_cast<double>(num_nodes) / num_topk)));
        // The shared expert's L2 before the routed tasks, wherever clusters retire. On four nodes, on minus off:
        // top-19 -1.4 .. -3.2 us at 256 .. 768 tokens and 0 .. -2.7 at 1024; top-25 -4 at 512, 0 at 1024; top-6 0.
        // (With one remote node it cost 8 us at top-19, 1024 tokens: no retire there, so it is off.)
        plan.shared_l2_early = plan.retire_percent > 0;
    }

    // Development overrides, one variable per field: DG_MEGA_MOE_IBP_<FIELD>=value (what the development servers
    // pass per arm; a name that is no field is refused: a misspelt arm must not run the default plan). Their
    // switch names for a trace and a zero-network run also apply.
    for (char** entry = environ; *entry != nullptr; ++ entry) {
        constexpr std::string_view kPrefix = "DG_MEGA_MOE_IBP_";
        const std::string_view variable(*entry);
        if (not variable.starts_with(kPrefix) or variable.find('=') == std::string_view::npos)
            continue;
        const auto name = variable.substr(kPrefix.size(), variable.find('=') - kPrefix.size());
        const auto value = static_cast<uint32_t>(std::stoul(std::string(variable.substr(variable.find('=') + 1))));
        if (name == "ORDERED") plan.ordered = value;
        else if (name == "GIN_BACKEND") plan.gin_backend = value;
        else if (name == "L1_LEAD_CAP") plan.l1_lead_cap = value;
        else if (name == "RETIRE_PERCENT") plan.retire_percent = value;
        else if (name == "RETIRE_AT_PERCENT") plan.retire_at_percent = value;
        else if (name == "SHARED_L2_EARLY") plan.shared_l2_early = value;
        else if (name == "DIAGNOSTICS") plan.diagnostics = value;
        else if (name == "TRACE") plan.trace = value;
        else if (name == "ZERO_NETWORK") plan.zero_network = value;
        else DG_HOST_UNREACHABLE(std::format("Unknown Mega MoE IB plan override: {}", variable));
    }
    if (const char* value = std::getenv("DG_MEGA_MOE_IB_TRACE"); value != nullptr and std::stoul(value) != 0)
        plan.trace = true;
    if (const char* value = std::getenv("DG_MEGA_MOE_IB_ORACLE"); value != nullptr and std::stoul(value) >= 2)
        plan.zero_network = true;
    // The kernel compiles the plan's backend only and cannot check it before its puts run
    DG_HOST_ASSERT(plan.gin_backend == 0 or gin_backend == 0 or plan.gin_backend == static_cast<uint32_t>(gin_backend));
    return plan;
}

// The plan as the JIT source spells it (a designated initializer of layout::IBPlan, fields in declaration order)
static std::string to_string(const layout::IBPlan& plan) {
    return std::format(
        "{{.ranks_per_node = {}, .ordered = {}, "
        ".gin_backend = {}, "
        ".l1_lead_cap = {}, .retire_percent = {}, .retire_at_percent = {}, "
        ".shared_l2_early = {}, .diagnostics = {}, .trace = {}, .zero_network = {}}}",
        plan.ranks_per_node, plan.ordered,
        plan.gin_backend, plan.l1_lead_cap, plan.retire_percent, plan.retire_at_percent,
        plan.shared_l2_early, plan.diagnostics, plan.trace, plan.zero_network);
}

} // namespace deep_gemm
