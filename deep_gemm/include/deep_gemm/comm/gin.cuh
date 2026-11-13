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

// What a put carries. A Gin context is a set of queue pairs, one per peer, so two streams to one peer never share
// a queue and one stream keeps its order per peer: the context of a put is its stream (Return spreads its chunks
// over the remaining contexts by `lane`).
enum class Stream : uint32_t { Meta = 0, Tables = 1, Rows = 2, Return = 3 };

// One Gin context, shared by any thread of the GPU (NCCL_GIN_RESOURCE_SHARING_GPU).
//
// Guarantees, and nothing more:
//   * nothing is said about when the bytes land: readers find out by polling (comm/landing.cuh);
//   * puts issued by ONE thread on one stream to one peer land in issue order (queue-pair FIFO);
//   * a put may read its source after it returns: staged bytes are published before the put (landing::publish) and
//     not rewritten until the landing side has proven them landed.
//
// Every put skips the send-queue credit check (it polls a completion queue in host memory: 1-2.5 us per post).
// That is safe only while the operations in flight per queue pair stay below its depth: deep_gemm/mega/ib.py sets
// NCCL_GIN_GDAKI_QP_DEPTH=1024, and one launch issues far fewer operations per context than that.
struct Gin {
    ncclGin gin;
    ncclTeam world;
    ncclWindow_t window;
    int64_t base;
    bool active;  // false: every put is a no-op (zero-network measurement builds)

    __device__ __forceinline__ static uint32_t context_of(const Stream stream, const uint32_t lane) {
        constexpr uint32_t kNumReturn = layout::kNumIBGinContexts - static_cast<uint32_t>(Stream::Return);
        return stream == Stream::Return ? static_cast<uint32_t>(Stream::Return) + lane % kNumReturn
                                        : static_cast<uint32_t>(stream);
    }

    __device__ __forceinline__
    Gin(const layout::IBHandles& handles, const int64_t base, const Stream stream, const uint32_t lane = 0,
        const bool active = true):
        gin(*reinterpret_cast<const ncclDevComm_t*>(handles.gin_dev_comm), static_cast<int>(context_of(stream, lane)),
            NCCL_GIN_RESOURCE_SHARING_GPU),
        world(ncclTeamWorld(*reinterpret_cast<const ncclDevComm_t*>(handles.gin_dev_comm))),
        window(reinterpret_cast<ncclWindow_t>(handles.gin_window)), base(base), active(active) {}

    template <typename ptr_t>
    __device__ __forceinline__ int64_t offset(const ptr_t& ptr) const {
        return reinterpret_cast<int64_t>(ptr) - base;
    }

    // Write `value` into `sym_ptr` (local layout) on `dst_rank_idx`
    template <typename dtype_t>
    __device__ __forceinline__ void put_value(dtype_t* sym_ptr, const dtype_t& value, const uint32_t& dst_rank_idx) const {
        if (active) gin.putValue(world, static_cast<int>(dst_rank_idx), window, offset(sym_ptr), value,
                     ncclGin_None(), ncclCoopThread(), ncclGin_None(),
                     cuda::thread_scope_thread, cuda::thread_scope_device, ncclGinOptFlagsMaySkipCreditCheck);
    }

    // Copy `num_bytes` from local `src_ptr` (inside the window) to `dst_sym_ptr` (local layout) on `dst_rank_idx`.
    __device__ __forceinline__ void put(const void* dst_sym_ptr, const void* src_ptr, const uint64_t& num_bytes,
                                        const uint32_t& dst_rank_idx) const {
        if (active) gin.put(world, static_cast<int>(dst_rank_idx), window, offset(dst_sym_ptr), window, offset(src_ptr), num_bytes,
                ncclGin_None(), ncclGin_None(), ncclCoopThread(), ncclGin_None(),
                cuda::thread_scope_thread, cuda::thread_scope_device, ncclGinOptFlagsMaySkipCreditCheck);
    }
};

} // namespace deep_gemm::comm
#endif
