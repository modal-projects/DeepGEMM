#pragma once

// Mega MoE across NVLink domains: time stamps of a trace build (`IBPlan::trace`). A normal build compiles none of
// this. Stamps are %globaltimer values kept in the IB signals: the latest (or earliest) thread to pass a point
// wins. The launch after a traced one prints them, relative to the traced launch's start, every 16th launch:
//   IB_TRACE rank epoch tokens | one value per stamp, in us
// (the line also starts with OPUS_ so that the development servers collect it).

#include <deep_gemm/impls/mega_moe_ib/geometry.cuh>

namespace deep_gemm::ib::trace {

enum Stamp : uint32_t {
    kBegin = 0,            // dispatch warps of SM 1 enter
    kMetaPosted,           // source: index / weight / header puts issued
    kRowsPosted,           // source: last row put issued (latest poster)
    kFirstHeaderSeen,      // receiver: earliest SM that saw a remote header
    kHeadersSeen,          // receiver: latest
    kCountsPublished,      // receiver: latest remote source handed to the scheduler
    kIntakeDone,           // proxy: latest return order written
    kFirstRemoteRow,       // pull: earliest remote row copied
    kLastRow,              // pull: latest row copied
    kFirstReduce,          // proxy sum: earliest warp to enter
    kLastTask,             // epilogue: latest SM to leave the GEMM task loop
    kLastChunkPut,         // proxy sum: latest return put issued
    kFirstRecords,         // source combine: earliest warp to start the record loop
    kEnd,                  // source combine: latest warp done
    kNumStamps
};
constexpr uint32_t kEarliestMask = (1u << kFirstHeaderSeen) | (1u << kFirstRemoteRow) | (1u << kFirstReduce) | (1u << kFirstRecords);

CUTLASS_DEVICE uint64_t now() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

template <typename G>
CUTLASS_DEVICE void stamp(const Launch& launch, const Stamp& slot, const bool& cond = true) {
    if constexpr (G::kPlan.trace) {
        if (cond) {
            const auto ptr = reinterpret_cast<unsigned long long*>(launch.signals->trace + slot);
            if ((kEarliestMask >> slot) & 1)
                atomicMin(ptr, static_cast<unsigned long long>(now()));
            else
                atomicMax(ptr, static_cast<unsigned long long>(now()));
        }
    }
}

// One thread, at kernel start, before any other stamp of this launch can matter: print the previous launch,
// then reset
template <typename G>
CUTLASS_DEVICE void begin(const Launch& launch, const uint32_t& num_tokens, const bool& is_printer) {
    if constexpr (G::kPlan.trace) {
        if (not is_printer)
            return;
        const auto t = launch.signals->trace;
        if (launch.epoch > 1 and (launch.epoch - 1) % 16 == 0) {
            const auto us = [&](const uint32_t& i) {
                return t[i] == 0 or t[i] == ~0ull ? -1.0 : static_cast<double>(static_cast<int64_t>(t[i] - t[kBegin])) / 1e3;
            };
            printf("OPUS_IB_TRACE rank=%u epoch=%llu tokens=%u | meta_posted=%.1f rows_posted=%.1f first_header=%.1f headers=%.1f "
                   "counts=%.1f intake=%.1f first_remote_row=%.1f last_row=%.1f first_reduce=%.1f last_task=%.1f "
                   "last_chunk_put=%.1f first_records=%.1f end=%.1f\n",
                   launch.rank, static_cast<unsigned long long>(launch.epoch - 1), num_tokens,
                   us(kMetaPosted), us(kRowsPosted), us(kFirstHeaderSeen), us(kHeadersSeen), us(kCountsPublished), us(kIntakeDone),
                   us(kFirstRemoteRow), us(kLastRow), us(kFirstReduce), us(kLastTask), us(kLastChunkPut), us(kFirstRecords), us(kEnd));
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumStamps; ++ i)
            t[i] = (kEarliestMask >> i) & 1 ? ~0ull : 0ull;
        t[kBegin] = now();
    }
}

} // namespace deep_gemm::ib::trace
