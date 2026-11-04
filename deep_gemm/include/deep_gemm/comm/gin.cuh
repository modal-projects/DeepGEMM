#pragma once

// Mega MoE over IB: thin NCCL Gin (GPU-initiated RDMA) wrapper, compiled only with DG_MEGA_MOE_IB.
// Addresses are local symmetric pointers; the remote address is the same window offset on the peer
// (every rank has an identical buffer layout and the whole buffer is one NCCL window).
// Reference: DeepEP deep_ep/include/deep_ep/comm/handle.cuh.

#if defined(DG_MEGA_MOE_IB) and DG_MEGA_MOE_IB
#include <nccl.h>
#include <nccl_device.h>

namespace deep_gemm::comm {

#ifndef DG_MEGA_MOE_IB_CONTEXTS
#define DG_MEGA_MOE_IB_CONTEXTS 4
#endif

struct GinIB {
    ncclGin gin;
    ncclTeam world;
    ncclWindow_t window;
    int64_t base;

    __device__ __forceinline__
    GinIB(const ncclDevComm_t& dev_comm, const ncclWindow_t& window, const int64_t& base, const uint32_t& context_idx):
        gin(dev_comm, static_cast<int>(context_idx % DG_MEGA_MOE_IB_CONTEXTS), NCCL_GIN_RESOURCE_SHARING_GPU),
        world(ncclTeamWorld(dev_comm)), window(window), base(base) {}

    template <typename ptr_t>
    __device__ __forceinline__ int64_t offset(const ptr_t& ptr) const {
        return reinterpret_cast<int64_t>(ptr) - base;
    }

    // Write `value` into `sym_ptr` (local layout) on `dst_rank_idx`
    template <typename dtype_t>
    __device__ __forceinline__ void put_value(dtype_t* sym_ptr, const dtype_t& value, const uint32_t& dst_rank_idx) const {
        gin.putValue(world, static_cast<int>(dst_rank_idx), window, offset(sym_ptr), value,
                     ncclGin_None(), ncclCoopThread(), ncclGin_None(),
                     cuda::thread_scope_thread, cuda::thread_scope_device, ncclGinOptFlagsDefault);
    }

    // Copy `num_bytes` from local `src_ptr` (inside the window) to `dst_sym_ptr` (local layout) on `dst_rank_idx`
    __device__ __forceinline__ void put(const void* dst_sym_ptr, const void* src_ptr, const uint64_t& num_bytes,
                                        const uint32_t& dst_rank_idx) const {
        gin.put(world, static_cast<int>(dst_rank_idx), window, offset(dst_sym_ptr), window, offset(src_ptr), num_bytes,
                ncclGin_None(), ncclGin_None(), ncclCoopThread(), ncclGin_None(),
                cuda::thread_scope_thread, cuda::thread_scope_device, ncclGinOptFlagsDefault);
    }

    // Copy `num_bytes` from `src_sym_ptr` (local layout) on `src_rank_idx` into local `dst_ptr` (inside the window)
    __device__ __forceinline__ void get(void* dst_ptr, const void* src_sym_ptr, const uint64_t& num_bytes,
                                        const uint32_t& src_rank_idx) const {
        gin.get(world, static_cast<int>(src_rank_idx), window, offset(src_sym_ptr), window, offset(dst_ptr), num_bytes,
                ncclCoopThread(), ncclGin_None(), ncclGinOptFlagsDefault, ncclGin_SegmentDevice());
    }

    // Wait until every operation this thread posted on its context has completed (puts placed remotely,
    // gets landed locally)
    __device__ __forceinline__ void flush() const {
        gin.flush(ncclCoopThread());
    }
};

} // namespace deep_gemm::comm
#endif
