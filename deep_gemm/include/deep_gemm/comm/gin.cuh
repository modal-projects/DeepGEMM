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
// 0: NCCL_GIN_RESOURCE_SHARING_GPU (contexts shared by any thread: global atomic slot reservation, CAS publish, lock +
// atomic max per put, ~3-4 us of issue per put on CPU-proxy hosts). 2: NCCL_GIN_RESOURCE_SHARING_THREAD with exclusive
// contexts (one issuing thread per context at a time; ~0.9 us per put; the caller guarantees exclusivity and that the
// index is < DG_MEGA_MOE_IB_CONTEXTS, so no modulo). 1: NCCL_GIN_RESOURCE_SHARING_CTA with the same exclusive indices (~2.2 us).
#ifndef DG_MEGA_MOE_IB_SHARING
#define DG_MEGA_MOE_IB_SHARING 0
#endif
// DG_MEGA_MOE_IB_SKIP_CREDIT=1: every put/putValue passes ncclGinOptFlagsMaySkipCreditCheck (DOCA skips the send-queue
// availability check, which polls a CQ in host memory: -1..-2.5 us per post in the protocol bench). Safe only while the
// ops in flight per QP stay below the QP depth: ib.py raises NCCL_GIN_GDAKI_QP_DEPTH to 1024 with it, and each launch
// issues far fewer ops per context than that and flushes before the next launch.
#ifndef DG_MEGA_MOE_IB_SKIP_CREDIT
#define DG_MEGA_MOE_IB_SKIP_CREDIT 0
#endif
#define DG_IB_OPT_FLAGS (DG_MEGA_MOE_IB_SKIP_CREDIT ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault)

struct GinIB {
    ncclGin gin;
    ncclTeam world;
    ncclWindow_t window;
    int64_t base;
    bool active = true;  // false: every put/signal is a no-op (DG_MEGA_MOE_IB_ORACLE=2 ceiling runs only)

    __device__ __forceinline__
    GinIB(const ncclDevComm_t& dev_comm, const ncclWindow_t& window, const int64_t& base, const uint32_t& context_idx,
          const int& sharing = DG_MEGA_MOE_IB_SHARING):
        gin(dev_comm, static_cast<int>(sharing >= 1 ? context_idx : context_idx % DG_MEGA_MOE_IB_CONTEXTS),
            static_cast<ncclGinResourceSharingMode>(sharing)),
        world(ncclTeamWorld(dev_comm)), window(window), base(base) {}

    template <typename ptr_t>
    __device__ __forceinline__ int64_t offset(const ptr_t& ptr) const {
        return reinterpret_cast<int64_t>(ptr) - base;
    }

    // Write `value` into `sym_ptr` (local layout) on `dst_rank_idx`
    template <typename dtype_t>
    __device__ __forceinline__ void put_value(dtype_t* sym_ptr, const dtype_t& value, const uint32_t& dst_rank_idx) const {
        if (not active) return;
        gin.putValue(world, static_cast<int>(dst_rank_idx), window, offset(sym_ptr), value,
                     ncclGin_None(), ncclCoopThread(), ncclGin_None(),
                     cuda::thread_scope_thread, cuda::thread_scope_device, DG_IB_OPT_FLAGS);
    }

    // Copy `num_bytes` from local `src_ptr` (inside the window) to `dst_sym_ptr` (local layout) on `dst_rank_idx`
    __device__ __forceinline__ void put(const void* dst_sym_ptr, const void* src_ptr, const uint64_t& num_bytes,
                                        const uint32_t& dst_rank_idx) const {
        if (not active) return;
        gin.put(world, static_cast<int>(dst_rank_idx), window, offset(dst_sym_ptr), window, offset(src_ptr), num_bytes,
                ncclGin_None(), ncclGin_None(), ncclCoopThread(), ncclGin_None(),
                cuda::thread_scope_thread, cuda::thread_scope_device, DG_IB_OPT_FLAGS);
    }

    // Copy `num_bytes` from `src_sym_ptr` (local layout) on `src_rank_idx` into local `dst_ptr` (inside the window)
    __device__ __forceinline__ void get(void* dst_ptr, const void* src_sym_ptr, const uint64_t& num_bytes,
                                        const uint32_t& src_rank_idx) const {
        gin.get(world, static_cast<int>(src_rank_idx), window, offset(src_sym_ptr), window, offset(dst_ptr), num_bytes,
                ncclCoopThread(), ncclGin_None(), DG_IB_OPT_FLAGS, ncclGin_SegmentDevice());
    }

    // Put with a bundled WEAK signal: +1 on the 64-bit counter at `signal_sym_ptr` (local layout) on `dst_rank_idx`
    // once THIS put has settled (no ordering with respect to other puts, no fence). One NIC op.
    __device__ __forceinline__ void put_signal_weak(const void* dst_sym_ptr, const void* src_ptr, const uint64_t& num_bytes,
                                                    const uint32_t& dst_rank_idx, const uint64_t* signal_sym_ptr) const {
        if (not active) return;
        gin.put(world, static_cast<int>(dst_rank_idx), window, offset(dst_sym_ptr), window, offset(src_ptr), num_bytes,
                ncclGin_WeakVASignalInc{window, static_cast<size_t>(offset(signal_sym_ptr))}, ncclGin_None(), ncclCoopThread(), ncclGin_None(),
                cuda::thread_scope_thread, cuda::thread_scope_device, DG_IB_OPT_FLAGS);
    }

    // Strong signal: +1 on the 64-bit counter at `sym_ptr` (local layout) on `dst_rank_idx`. Its visibility
    // implies every preceding put on this context to that peer has settled. Only NIC signals may touch the
    // counter (never GPU atomics).
    __device__ __forceinline__ void signal_inc(const uint64_t* sym_ptr, const uint32_t& dst_rank_idx) const {
        if (not active) return;
        gin.signal(world, static_cast<int>(dst_rank_idx), ncclGin_StrongVASignalInc{window, static_cast<size_t>(offset(sym_ptr))});
    }

    // Wait until this context's outstanding gets have landed locally and put sources are reusable.
    // NOTES: flush does NOT mean puts have settled remotely; use `signal_inc` for that.
    __device__ __forceinline__ void flush() const {
        gin.flush(ncclCoopThread());
    }
};

} // namespace deep_gemm::comm
#endif
