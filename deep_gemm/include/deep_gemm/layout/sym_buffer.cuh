#pragma once

#include <deep_gemm/common/exception.cuh>

#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
#include <nccl.h>
#include <nccl_device.h>
#endif

namespace deep_gemm::layout {

constexpr static uint32_t kNumMaxRanks = 72;

// Mega MoE over IB: `sym_buffer_ptrs` from Python is [per-rank LSA pointers (0 for ranks on other
// nodes)] + [kIBMagic, device ptr to ncclDevComm_t, ncclWindow_t, num_lsa_ranks].
constexpr static int64_t kIBMagic = 0x1B1B1B1B1B1B1B1Bll;
constexpr static uint32_t kNumIBExtraPtrs = 4;

template <uint32_t kNumRanks = kNumMaxRanks>
struct SymBuffer {
    int64_t base;
    int64_t offsets[kNumMaxRanks];
    uint32_t rank_idx;
    // IB fields (zero when every rank is in this NVLink domain)
    int64_t gin_dev_comm = 0;   // device pointer to ncclDevComm_t
    int64_t gin_window = 0;     // ncclWindow_t
    uint32_t num_lsa_ranks = 0; // ranks per NVLink (LSA) domain; 0 = all ranks

    DG_STATIC_ASSERT(kNumRanks <= kNumMaxRanks, "Too many ranks");

    SymBuffer() = default;

    template <typename Container>
    explicit SymBuffer(const Container& c, const uint32_t& rank_idx): rank_idx(rank_idx) {
        auto size = static_cast<uint32_t>(c.size());
        if (size > kNumIBExtraPtrs and c[size - kNumIBExtraPtrs] == kIBMagic) {
            gin_dev_comm = c[size - 3];
            gin_window = c[size - 2];
            num_lsa_ranks = static_cast<uint32_t>(c[size - 1]);
            size -= kNumIBExtraPtrs;
        }
        base = c[rank_idx];
        for (uint32_t i = 0; i < kNumMaxRanks; ++ i)
            offsets[i] = (i < size and c[i] != 0) ? (c[i] - base) : 0;
    }

    template <typename Container>
    static uint32_t num_ranks_of(const Container& c) {
        const auto size = static_cast<uint32_t>(c.size());
        return (size > kNumIBExtraPtrs and c[size - kNumIBExtraPtrs] == kIBMagic) ? size - kNumIBExtraPtrs : size;
    }

#if defined(__CUDA_ARCH__) or defined(__CLION_IDE__)
    template <typename ptr_t = void*>
    CUTLASS_DEVICE ptr_t get_base_ptr() const {
        return reinterpret_cast<ptr_t>(base);
    }

    // Whether `dst_rank_idx` is reachable by load/store (same NVLink domain)
    CUTLASS_DEVICE bool is_lsa(const uint32_t& dst_rank_idx) const {
        return num_lsa_ranks == 0 or (dst_rank_idx / num_lsa_ranks == rank_idx / num_lsa_ranks);
    }

    template <typename ptr_t>
    CUTLASS_DEVICE ptr_t map(const ptr_t& ptr, const uint32_t& dst_rank_idx) const {
        if constexpr (kNumRanks == 1)
            return ptr;
#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
        // Any load/store to a rank on another node must go through Gin: catch un-ported paths
        DG_DEVICE_ASSERT(is_lsa(dst_rank_idx) and "Mega MoE IB: load/store map() to a remote-node rank");
#endif
        int64_t mapped_ptr = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
        return *reinterpret_cast<ptr_t*>(&mapped_ptr);
    }

    // Window offset of a local symmetric pointer (identical layout on every rank)
    template <typename ptr_t>
    CUTLASS_DEVICE uint64_t window_offset(const ptr_t& ptr) const {
        return static_cast<uint64_t>(reinterpret_cast<int64_t>(ptr) - base);
    }
#endif
};

} // namespace deep_gemm::layout
