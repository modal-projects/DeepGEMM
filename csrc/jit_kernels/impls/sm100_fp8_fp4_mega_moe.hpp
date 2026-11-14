#pragma once

#include <cstdlib>
#include <format>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <torch/python.h>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/mega_moe_ib.cuh>
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

// What an IB build is compiled with: the switch and the plan (layout::IBPlan). Its headers read nothing else.
// DG_PRINT_CONFIGS=1 prints every plan once, with the link rate it was planned for.
static std::string get_mega_moe_ib_defines(const layout::IBPlan& plan, const int& link_gbps) {
    const auto rate = std::format("{} Gb/s per GPU{}", link_gbps > 0 ? link_gbps : 800, link_gbps > 0 ? "" : " (link rate unknown)");
    static std::unordered_set<std::string> printed;
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS") and printed.insert(to_string(plan)).second)
        std::cout << "MegaMoEIBPlan(planned for " << rate << "): " << to_string(plan) << std::endl;
    return std::format("#define DG_MEGA_MOE_IB 1\n#define DG_MEGA_MOE_IB_PLAN {}\n// planned for {}\n", to_string(plan), rate);
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
    // Ranks of other NVLink domains: the Gin handles ride behind the per-rank pointers (layout/mega_moe_ib.cuh)
    const bool is_ib = layout::IBHandles::present(sym_buffer_ptrs);
    const auto num_ranks = static_cast<int>(layout::IBHandles::num_ranks_of(sym_buffer_ptrs));
    const std::vector<int64_t> rank_ptrs(sym_buffer_ptrs.begin(), sym_buffer_ptrs.begin() + num_ranks);
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    const auto tile_tokens_of = [&](const int& tokens) {  // the count the tile is chosen for
        return is_ib ? get_mega_moe_ib_tile_tokens(tokens, num_max_tokens_per_rank) : tokens;
    };
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, tile_tokens_of(num_tokens), num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        MmaKind::MXFP8FP4, is_ib);

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

    // Compile. The source of the kernel for `tokens` tokens on this rank: the token count picks the tile (`c`) and,
    // across NVLink domains, the plan
    const auto make_code = [&](const MegaMoEConfig& c, const int& tokens) {
    const auto ib_defines = not is_ib ? std::string() : get_mega_moe_ib_defines(get_mega_moe_ib_plan(
        num_ranks, static_cast<int>(layout::IBHandles::ranks_per_node_of(sym_buffer_ptrs)), tokens, num_topk,
        static_cast<int>(layout::IBHandles::gin_backend_of(sym_buffer_ptrs)), hidden,
        static_cast<int>(layout::IBHandles::link_gbps_of(sym_buffer_ptrs))),
        static_cast<int>(layout::IBHandles::link_gbps_of(sym_buffer_ptrs)));
    return ib_defines + std::format(R"(
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
        c.block_m, c.block_n, c.block_k,
        c.store_block_m_l1, c.store_block_m_l2,
        c.sf_block_m, c.sf_block_n,
        c.num_ring_tokens,
        c.num_sf_ring_tokens,
        c.num_stages,
        c.num_bytes_per_pull,
        c.num_dispatch_threads, c.num_non_epilogue_threads, c.num_epilogue_threads,
        num_sms, num_ranks,
        to_string(activation_clamp),
        fast_math ? "true" : "false",
        to_string(l1_weights.scalar_type()));
    };
    const auto code = make_code(config, num_tokens);

    // DG_DEV_COMPILE_ONLY=<file> (development servers precompile variants in parallel): compile into the JIT
    // disk cache, write the cache entry's path to <file>, and return without loading or launching
    if (const char* compile_only_out = get_mega_moe_dev_env("DG_DEV_COMPILE_ONLY"); compile_only_out != nullptr) {
        std::ofstream(compile_only_out) << jit->compile_without_load("sm100_fp8_fp4_mega_moe", code).string();
        return;
    }
    const auto kernel = jit->compile("sm100_fp8_fp4_mega_moe", code);

    // First-launch rendezvous. Ranks JIT-compile on their own and finish tens of seconds apart, while the
    // kernel's waits on its peers time out after 60 s. So the first time a multi-rank call meets a kernel object
    // this process has not launched (and the stream is not capturing), nothing is launched and
    // mega_moe_launch_skipped() is set: the Python wrapper barriers the group and calls again.
    // Steady state: one hash-set lookup. DG_MOE_NO_RENDEZVOUS=1 disables it.
    // The rendezvous is a group barrier, so every rank must enter it in the SAME call. A rank whose token count
    // crosses a tile class (or a plan boundary) alone would meet a new kernel object alone and wait in the barrier
    // while its peers, already launched, wait for it in the kernel. So the first time, every kernel this buffer
    // can launch (any token count up to the buffer's maximum) is compiled and marked: one rendezvous per buffer
    // shape, and no rank meets a new kernel object afterwards. Costs first-call time only (one compile per tile
    // class and plan; the JIT disk cache keeps them).
    mega_moe_launch_skipped() = false;
    if (num_ranks > 1) {
        const char* no_rendezvous = get_mega_moe_dev_env("DG_MOE_NO_RENDEZVOUS");
        static std::unordered_set<const void*> launched_kernels;
        if ((no_rendezvous == nullptr or std::string(no_rendezvous) == "0") and not launched_kernels.contains(kernel.get())) {
            cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
            DG_CUDA_RUNTIME_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream().stream(), &capture_status));
            launched_kernels.insert(kernel.get());
            std::unordered_set<std::string> family{code};
            for (int tokens = 0; tokens <= num_max_tokens_per_rank; ++ tokens) {
                const auto other = make_code(get_mega_moe_config(
                    num_ranks, num_experts, num_experts_per_rank,
                    num_max_tokens_per_rank, tile_tokens_of(tokens), num_topk, hidden, intermediate_hidden,
                    num_ring_tokens, num_sf_ring_tokens,
                    MmaKind::MXFP8FP4, is_ib), tokens);
                if (family.insert(other).second)
                    launched_kernels.insert(jit->compile("sm100_fp8_fp4_mega_moe", other).get());
            }
            if (capture_status == cudaStreamCaptureStatusNone) {
                mega_moe_launch_skipped() = true;
                return;
            }
        }
    }

    // Launch. The IB kernel takes the Gin handles as one more argument, before the last one.
    const auto launch = [&](const auto&... ib_args) {
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
        layout::SymBuffer<>(rank_ptrs, rank_idx),
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
        ib_args...,
        sm_locality_domains.data_ptr<uint8_t>());
    };
    if (is_ib)
        launch(layout::IBHandles(sym_buffer_ptrs));
    else
        launch();
}

} // namespace deep_gemm
