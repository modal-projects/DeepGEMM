#pragma once

#include <format>
#include <unordered_set>
#include <torch/python.h>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/mega_moe.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// Set when the last sm100_fp8_fp4_mega_moe call skipped its launch for the first-launch rendezvous
static bool& mega_moe_launch_skipped() {
    static bool skipped = false;
    return skipped;
}

static void sm100_fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l1_acts_sf,
    const torch::Tensor& shared_l2_acts, const torch::Tensor& shared_l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const torch::Tensor& shared_l1_weights_sf, const torch::Tensor& shared_l2_weights_sf,
    const torch::Tensor& sm_locality_domains,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(layout::SymBuffer<>::num_ranks_of(sym_buffer_ptrs));
    const bool is_ib = num_ranks != static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        MmaKind::MXFP8FP4);

    // Make tensormap
    constexpr int kGranK = 32;
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_sf_ring_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim, static_cast<int>(l1_acts_sf.stride(-1)));
    const auto tensor_map_l1_weights = make_tma_weights_3d_desc(l1_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                           sf_smem_outer_dim, static_cast<int>(l1_weights_sf.stride(-1)));
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
    // so the swizzle mode is also halved (128 -> 64).
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_ring_tokens,
                                                       config.block_n / 2, config.store_block_m_l1,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode / 2);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_sf_ring_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim, static_cast<int>(l2_acts_sf.stride(-1)));
    const auto tensor_map_l2_weights = make_tma_weights_3d_desc(l2_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                           sf_smem_outer_dim, static_cast<int>(l2_weights_sf.stride(-1)));

    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_acts,
        hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_acts_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_acts_sf,
        static_cast<int>(shared_l1_acts_sf.size(0)), hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim, static_cast<int>(shared_l1_acts_sf.stride(-1))) : tensor_map_l1_acts_sf;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ?
        make_tma_weights_2d_desc(shared_l1_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_weights_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_weights_sf,
        shared_intermediate_hidden * 2, hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim, static_cast<int>(shared_l1_weights_sf.stride(-1))) : tensor_map_l1_weights_sf;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_n / 2, config.store_block_m_l1,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode / 2) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_acts_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_acts_sf,
        static_cast<int>(shared_l2_acts_sf.size(0)), shared_intermediate_hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim, static_cast<int>(shared_l2_acts_sf.stride(-1))) : tensor_map_l2_acts_sf;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ?
        make_tma_weights_2d_desc(shared_l2_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode) : tensor_map_l2_weights;
    const auto tensor_map_shared_l2_weights_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_weights_sf,
        hidden, shared_intermediate_hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim, static_cast<int>(shared_l2_weights_sf.stride(-1))) : tensor_map_l2_weights_sf;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = get_num_sms_for_mega_moe();

    // Compile
    const auto kernel = jit->compile("sm100_fp8_fp4_mega_moe", std::string(is_ib ? "#define DG_MEGA_MOE_IB 1\n" : "") +
        std::string(std::getenv("DG_MEGA_MOE_IB_KEEP_REGS") != nullptr ? "#define DG_MEGA_MOE_IB_KEEP_REGS 1\n" : "") +
        std::string(((is_ib and std::getenv("DG_MEGA_MOE_IB_TRACE") != nullptr) or (not is_ib and std::getenv("DG_MEGA_MOE_NVL_TRACE") != nullptr)) ? "#define DG_MEGA_MOE_IB_TRACE 1\n" : "") +
        std::string(std::getenv("DG_MEGA_MOE_IB_BF16_COMBINE") != nullptr ? "#define DG_MEGA_MOE_IB_BF16_COMBINE 1\n" : "") +
        std::string(std::getenv("DG_MEGA_MOE_IB_FLAG_REDUCE") != nullptr ? "#define DG_MEGA_MOE_IB_FLAG_REDUCE 1\n" : "") +
        std::string(std::getenv("DG_MEGA_MOE_IB_NO_META_FIRST") != nullptr ? "#define DG_MEGA_MOE_IB_NO_META_FIRST 1\n" : "") +
        std::string(std::getenv("DG_MEGA_MOE_IB_BACKOFF") != nullptr ? "#define DG_MEGA_MOE_IB_BACKOFF 1\n" : "") +
        (std::getenv("DG_MEGA_MOE_IB_PULL_WHOLE") != nullptr ? std::format("#define DG_MEGA_MOE_IB_PULL_WHOLE {}\n", std::getenv("DG_MEGA_MOE_IB_PULL_WHOLE")) : std::string()) +
        std::string(std::getenv("DG_MEGA_MOE_IB_HDR_CTX") != nullptr ? "#define DG_MEGA_MOE_IB_HDR_CTX 1\n" : "") +
        std::string(std::getenv("DG_MEGA_MOE_IB_SKIP_CREDIT") != nullptr ? "#define DG_MEGA_MOE_IB_SKIP_CREDIT 1\n" : "") +
        (std::getenv("DG_MEGA_MOE_IB_ORACLE") != nullptr ? std::format("#define DG_MEGA_MOE_IB_ORACLE {}\n", std::getenv("DG_MEGA_MOE_IB_ORACLE")) : std::string()) +
        std::string(std::getenv("DG_MEGA_MOE_IB_RET_DB") != nullptr ? "#define DG_MEGA_MOE_IB_RET_DB 1\n" : "") +
        std::string(std::getenv("DG_MEGA_MOE_IB_FRONT") != nullptr ? "#define DG_MEGA_MOE_IB_FRONT 1\n" : "") +
        std::string(std::getenv("DG_MEGA_MOE_IB_GPU_ACQ") != nullptr ? "#define DG_MEGA_MOE_IB_GPU_ACQ 1\n" : "") +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_PULL_META_EARLY") != nullptr ? "#define DG_MEGA_MOE_PULL_META_EARLY 1\n" : "") +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_IB_COUNTS_EARLY") != nullptr ? "#define DG_MEGA_MOE_IB_COUNTS_EARLY 1\n" : "") +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_IB_LOCAL_FIRST") != nullptr ? "#define DG_MEGA_MOE_IB_LOCAL_FIRST 1\n" : "") +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_IB_SPLIT_SCAN") != nullptr ? "#define DG_MEGA_MOE_IB_SPLIT_SCAN 1\n" : "") +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_IB_COUNTS_IN_META") != nullptr ? "#define DG_MEGA_MOE_IB_COUNTS_IN_META 1\n" : "") +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_IB_SENDER_TABLES") != nullptr ? "#define DG_MEGA_MOE_IB_SENDER_TABLES 1\n" : "") +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_IB_PULL_SKIP_SCAN") != nullptr ? "#define DG_MEGA_MOE_IB_PULL_SKIP_SCAN 1\n" : "") +
        (is_ib and std::getenv("DG_MEGA_MOE_IB_SCAN_BATCH") != nullptr ? std::format("#define DG_MEGA_MOE_IB_SCAN_BATCH {}\n", std::getenv("DG_MEGA_MOE_IB_SCAN_BATCH")) : std::string()) +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_IB_FWD_JOIN") != nullptr and std::string(std::getenv("DG_MEGA_MOE_IB_FWD_JOIN")) != "0" ? "#define DG_MEGA_MOE_IB_FWD_JOIN 1\n" : "") +
        (is_ib and std::getenv("DG_MEGA_MOE_IB_FWD_BATCH") != nullptr ? std::format("#define DG_MEGA_MOE_IB_FWD_BATCH {}\n", std::getenv("DG_MEGA_MOE_IB_FWD_BATCH")) : std::string()) +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_IB_FWD_PREFETCH") != nullptr and std::string(std::getenv("DG_MEGA_MOE_IB_FWD_PREFETCH")) != "0" ? "#define DG_MEGA_MOE_IB_FWD_PREFETCH 1\n" : "") +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_IB_C3_EARLY") != nullptr and std::string(std::getenv("DG_MEGA_MOE_IB_C3_EARLY")) != "0" ? "#define DG_MEGA_MOE_IB_C3_EARLY 1\n" : "") +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_IB_FWD_DEFER_REARM") != nullptr and std::string(std::getenv("DG_MEGA_MOE_IB_FWD_DEFER_REARM")) != "0" ? "#define DG_MEGA_MOE_IB_FWD_DEFER_REARM 1\n" : "") +
        (is_ib and std::getenv("DG_MEGA_MOE_IB_RET_GROUP") != nullptr ? std::format("#define DG_MEGA_MOE_IB_RET_GROUP {}\n", std::getenv("DG_MEGA_MOE_IB_RET_GROUP")) : std::string()) +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_IB_C3_REPOLL") != nullptr and std::string(std::getenv("DG_MEGA_MOE_IB_C3_REPOLL")) != "0" ? "#define DG_MEGA_MOE_IB_C3_REPOLL 1\n" : "") +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_IB_C3_SPLIT") != nullptr and std::string(std::getenv("DG_MEGA_MOE_IB_C3_SPLIT")) != "0" ? "#define DG_MEGA_MOE_IB_C3_SPLIT 1\n" : "") +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_IB_EARLY_REARM") != nullptr and std::string(std::getenv("DG_MEGA_MOE_IB_EARLY_REARM")) != "0" ? "#define DG_MEGA_MOE_IB_EARLY_REARM 1\n" : "") +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_IB_NO_RET_HDR") != nullptr and std::string(std::getenv("DG_MEGA_MOE_IB_NO_RET_HDR")) != "0" ? "#define DG_MEGA_MOE_IB_NO_RET_HDR 1\n" : "") +
        (is_ib and std::getenv("DG_MEGA_MOE_IB_FWD_BATCH_D") != nullptr ? std::format("#define DG_MEGA_MOE_IB_FWD_BATCH_D {}\n", std::getenv("DG_MEGA_MOE_IB_FWD_BATCH_D")) : std::string()) +
        (is_ib and std::getenv("DG_MEGA_MOE_IB_XDELAY_NS") != nullptr ? std::format("#define DG_MEGA_MOE_IB_XDELAY_NS {}\n", std::getenv("DG_MEGA_MOE_IB_XDELAY_NS")) : std::string()) +
        (std::getenv("DG_MEGA_MOE_L2_PREFETCH") != nullptr ? std::format("#define DG_MEGA_MOE_L2_PREFETCH {}\n", std::getenv("DG_MEGA_MOE_L2_PREFETCH")) : std::string()) +
        std::string(std::getenv("DG_MEGA_MOE_IB_FWD_DYN") != nullptr ? "#define DG_MEGA_MOE_IB_FWD_DYN 1\n" : "") +
        (std::getenv("DG_MEGA_MOE_IB_RETGROUPS") != nullptr ? std::format("#define DG_MEGA_MOE_IB_RETGROUPS {}\n", std::getenv("DG_MEGA_MOE_IB_RETGROUPS")) : std::string()) +
        std::string(is_ib and std::getenv("DG_MEGA_MOE_L2_COLMAJOR") != nullptr ? "#define DG_MEGA_MOE_L2_COLMAJOR 1\n" : "") +
        std::string(std::getenv("DG_MEGA_MOE_IB_DEBUG_D0") != nullptr ? "#define DG_MEGA_MOE_IB_DEBUG_D0 1\n" : "") +
        (std::getenv("DG_MEGA_MOE_IB_CONTEXTS") != nullptr ? std::format("#define DG_MEGA_MOE_IB_CONTEXTS {}\n", std::getenv("DG_MEGA_MOE_IB_CONTEXTS")) : std::string()) +
        (std::getenv("DG_MEGA_MOE_IB_SHARING") != nullptr ? std::format("#define DG_MEGA_MOE_IB_SHARING {}\n", std::getenv("DG_MEGA_MOE_IB_SHARING")) : std::string()) +
        (std::getenv("DG_MEGA_MOE_IB_XSITES") != nullptr ? std::format("#define DG_MEGA_MOE_IB_XSITES {}\n", std::getenv("DG_MEGA_MOE_IB_XSITES")) : std::string()) +
        std::string(std::getenv("DG_MEGA_MOE_IB_META_FIRST_ALL") != nullptr ? "#define DG_MEGA_MOE_IB_META_FIRST_ALL 1\n" : "") +
        (std::getenv("DG_MEGA_MOE_IB_OVERLAP_MIN_TOKENS") != nullptr ?
            std::format("#define DG_MEGA_MOE_IB_OVERLAP_MIN_TOKENS {}\n", std::getenv("DG_MEGA_MOE_IB_OVERLAP_MIN_TOKENS")) : std::string()) +
        std::string(std::getenv("DG_MEGA_MOE_IB_HDR_CTX") != nullptr ? "#define DG_MEGA_MOE_IB_HDR_CTX 1\n" : "") +
        std::string(std::getenv("DG_MEGA_MOE_IB_PULL_FLAG_GATE") != nullptr and std::string(std::getenv("DG_MEGA_MOE_IB_PULL_FLAG_GATE")) != "0" ? "#define DG_MEGA_MOE_IB_PULL_FLAG_GATE 1\n" : "") +
        (std::getenv("DG_MEGA_MOE_IB_PULL_GATE") != nullptr and std::string(std::getenv("DG_MEGA_MOE_IB_PULL_GATE")) != "0" ?
            std::format("#define DG_MEGA_MOE_IB_PULL_GATE {}\n", std::getenv("DG_MEGA_MOE_IB_PULL_GATE")) : std::string()) +
        (std::getenv("DG_MEGA_MOE_IB_WAVE_SORT") != nullptr and std::string(std::getenv("DG_MEGA_MOE_IB_WAVE_SORT")) != "0" ?
            std::format("#define DG_MEGA_MOE_IB_WAVE_SORT {}\n", std::getenv("DG_MEGA_MOE_IB_WAVE_SORT")) : std::string()) +
        (std::getenv("DG_MEGA_MOE_IB_STREAM_COMBINE") != nullptr and std::string(std::getenv("DG_MEGA_MOE_IB_STREAM_COMBINE")) != "0" ?
            std::format("#define DG_MEGA_MOE_IB_STREAM_COMBINE {}\n", std::getenv("DG_MEGA_MOE_IB_STREAM_COMBINE")) : std::string()) +
        (std::getenv("DG_MEGA_MOE_IB_DIAG") != nullptr ? std::format("#define DG_MEGA_MOE_IB_DIAG {}\n", std::getenv("DG_MEGA_MOE_IB_DIAG")) : std::string()) + std::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {}, {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {}
    >);
}};
)", num_max_tokens_per_rank,
        hidden, intermediate_hidden,
        num_experts, num_shared_experts,
        num_topk,
        config.block_m, config.block_n, config.block_k,
        config.store_block_m_l1, config.store_block_m_l2,
        config.sf_block_m, config.sf_block_n,
        config.num_ring_tokens,
        config.num_sf_ring_tokens,
        config.num_stages,
        config.num_bytes_per_pull,
        config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
        num_sms, num_ranks,
        to_string(activation_clamp),
        fast_math ? "true" : "false",
        to_string(l1_weights.scalar_type())));

    // First-launch rendezvous (correctness, 2026-11-06): ranks JIT-compile independently and finish 10s of seconds
    // apart, and the kernel's peer waits time out after 60 s (header / NVLink-barrier timeouts at epoch 1). The first
    // time a multi-rank call meets a kernel object this process has not launched yet (and the stream is not capturing),
    // nothing is launched and mega_moe_launch_skipped() is set: the Python wrapper barriers the group and calls again.
    // Steady state: one hash-set lookup. DG_MOE_NO_RENDEZVOUS=1 disables it (A/B and repro only).
    mega_moe_launch_skipped() = false;
    if (num_ranks > 1) {
        const char* no_rdv = std::getenv("DG_MOE_NO_RENDEZVOUS");
        static std::unordered_set<const void*> launched_kernels;
        if ((no_rdv == nullptr or std::string(no_rdv) == "0") and not launched_kernels.contains(kernel.get())) {
            cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
            DG_CUDA_RUNTIME_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream().stream(), &capture_status));
            launched_kernels.insert(kernel.get());
            if (capture_status == cudaStreamCaptureStatusNone) {
                mega_moe_launch_skipped() = true;
                return;
            }
        }
    }

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, 1, 1),
            .cluster_dim = dim3(2, 1, 1),
        },
        y.data_ptr(),
        cumulative_local_expert_recv_stats_ptr,
        num_tokens,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_l1_acts,
        tensor_map_l1_acts_sf,
        tensor_map_l1_weights,
        tensor_map_l1_weights_sf,
        tensor_map_l1_output,
        tensor_map_l2_acts,
        tensor_map_l2_acts_sf,
        tensor_map_l2_weights,
        tensor_map_l2_weights_sf,
        tensor_map_shared_l1_acts,
        tensor_map_shared_l1_acts_sf,
        tensor_map_shared_l1_weights,
        tensor_map_shared_l1_weights_sf,
        tensor_map_shared_l1_output,
        tensor_map_shared_l2_acts,
        tensor_map_shared_l2_acts_sf,
        tensor_map_shared_l2_weights,
        tensor_map_shared_l2_weights_sf,
        sm_locality_domains.data_ptr<uint8_t>());
}

} // namespace deep_gemm
