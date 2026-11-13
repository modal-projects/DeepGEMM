#pragma once

// Mega MoE across NVLink domains: thin NCCL Gin (GPU-initiated RDMA) wrapper, compiled only with DG_MEGA_MOE_IB.
// Addresses are local symmetric pointers; the remote address is the same window offset on the peer (every rank
// has an identical buffer layout and the whole buffer is one NCCL window).
// Reference: DeepEP deep_ep/include/deep_ep/comm/handle.cuh.

#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
#include <nccl.h>
#include <nccl_device.h>

#include <deep_gemm/layout/mega_moe_ib.cuh>

namespace deep_gemm::comm {

// One Gin context (a set of queue pairs, one per peer). Contexts are shared by any thread of the GPU
// (NCCL_GIN_RESOURCE_SHARING_GPU); the kernel spreads its puts over `num_contexts` of them by index.
//
// Every put skips the send-queue credit check (it polls a completion queue in host memory: 1-2.5 us per post).
// That is safe only while the operations in flight per queue pair stay below its depth: deep_gemm/mega/ib.py sets
// NCCL_GIN_GDAKI_QP_DEPTH=1024, and one launch issues far fewer operations per context than that.
struct Gin {
    ncclGin gin;
    ncclTeam world;
    ncclWindow_t window;
    int64_t base;

    __device__ __forceinline__
    Gin(const layout::IBHandles& handles, const int64_t& base, const uint32_t& context_idx, const uint32_t& num_contexts):
        gin(*reinterpret_cast<const ncclDevComm_t*>(handles.gin_dev_comm), static_cast<int>(context_idx % num_contexts),
            NCCL_GIN_RESOURCE_SHARING_GPU),
        world(ncclTeamWorld(*reinterpret_cast<const ncclDevComm_t*>(handles.gin_dev_comm))),
        window(reinterpret_cast<ncclWindow_t>(handles.gin_window)), base(base) {}

    template <typename ptr_t>
    __device__ __forceinline__ int64_t offset(const ptr_t& ptr) const {
        return reinterpret_cast<int64_t>(ptr) - base;
    }

    // Write `value` into `sym_ptr` (local layout) on `dst_rank_idx`
    template <typename dtype_t>
    __device__ __forceinline__ void put_value(dtype_t* sym_ptr, const dtype_t& value, const uint32_t& dst_rank_idx) const {
        gin.putValue(world, static_cast<int>(dst_rank_idx), window, offset(sym_ptr), value,
                     ncclGin_None(), ncclCoopThread(), ncclGin_None(),
                     cuda::thread_scope_thread, cuda::thread_scope_device, ncclGinOptFlagsMaySkipCreditCheck);
    }

    // Copy `num_bytes` from local `src_ptr` (inside the window) to `dst_sym_ptr` (local layout) on `dst_rank_idx`.
    // Nothing is said about when the bytes land or in which order: readers find out by polling (poll.cuh).
    __device__ __forceinline__ void put(const void* dst_sym_ptr, const void* src_ptr, const uint64_t& num_bytes,
                                        const uint32_t& dst_rank_idx) const {
        gin.put(world, static_cast<int>(dst_rank_idx), window, offset(dst_sym_ptr), window, offset(src_ptr), num_bytes,
                ncclGin_None(), ncclGin_None(), ncclCoopThread(), ncclGin_None(),
                cuda::thread_scope_thread, cuda::thread_scope_device, ncclGinOptFlagsMaySkipCreditCheck);
    }
};

} // namespace deep_gemm::comm
#endif
