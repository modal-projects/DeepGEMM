#pragma once

// Landing: memory that another party fills while this kernel runs, and how a reader finds out.
//
// A landing region is pre-filled with a sentinel that data cannot equal. The writer is a NIC (Gin puts from
// another node) or a peer GPU (NVLink stores); it says nothing when it is done. A reader loads a unit and looks:
// "the load is the poll". Two rules make this a protocol:
//   use      a unit is polled against its sentinel before its value is used;
//   landing  every unit written in a launch is polled by some reader before the region is re-armed (filled with
//            the sentinel again) and before the writer's source bytes may change. Readers that prove landing
//            poll every unit of their region, whether or not they need its value.
// Units: 16 bytes of all-ones for token rows, scale factors, partial sums and records (an fp8 / bf16 NaN
// pattern); INT64_MIN for top-k indices; an all-ones word for weights, counts and table entries.
//
// Inside one GPU, "these stores are visible" is told with an epoch flag instead (publish / raised below): the
// writers fence, the CTA synchronizes, one thread releases the launch's epoch into the flag word. Flags are never
// reset: an old epoch is "not raised".

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm::comm::landing {

constexpr uint64_t kBlankIdx = 0x8000000000000000ull;
constexpr uint32_t kBlankWord = 0xffffffffu;

// Where a poll is, for the timeout message of a diagnostics build
struct Site {
    const char* name;
    uint32_t rank, a, b;
};

CUTLASS_DEVICE uint4 load(const void* unit) {
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(unit) : "memory");
    return v;
}

CUTLASS_DEVICE bool is_blank(const uint4& v) {
    return (v.x & v.y & v.z & v.w) == kBlankWord;
}

CUTLASS_DEVICE uint4 blank_unit() {
    return {kBlankWord, kBlankWord, kBlankWord, kBlankWord};
}

// Spin while `still_blank()` holds, calling `reload()` in between. A diagnostics build names the site and traps
// after ~8 s; a timed build is the bare spin (the timeout set-up at every hot poll cost 22 us at 1024 tokens),
// bounded instead by the waits around it: a unit that never lands stalls its warp before the dispatch / epilogue
// CTA barrier, so the peers' node barrier and grid syncs time out (comm::wait_until, 60 s).
template <bool kDiag, typename still_blank_t, typename reload_t>
CUTLASS_DEVICE void spin(const still_blank_t& still_blank, const reload_t& reload, const Site& site) {
    if constexpr (kDiag) {
        const auto start_clock = clock64();
        while (still_blank()) {
            reload();
            if (clock64() - start_clock > (1ll << 34)) {
                printf("DeepGEMM IB landing timeout: %s rank=%u sm=%u thread=%u a=%u b=%u\n",
                       site.name, site.rank, blockIdx.x, threadIdx.x, site.a, site.b);
                DG_DEVICE_ASSERT(false and "IB landing timeout");
            }
        }
    } else {
        while (still_blank())
            reload();
    }
}

// Serial form, for sites where a unit has normally landed: re-read this unit until it differs
template <bool kDiag>
CUTLASS_DEVICE void await(uint4& v, const void* unit, const Site& site) {
    spin<kDiag>([&]() { return is_blank(v); }, [&]() { v = load(unit); }, site);
}

template <bool kDiag>
CUTLASS_DEVICE void await(uint32_t& v, const uint32_t* word, const Site& site) {
    spin<kDiag>([&]() { return v == kBlankWord; }, [&]() { v = ptx::ld_volatile(word); }, site);
}

// Batch form, for sites that often arrive before the data: every round re-reads all units that are still blank,
// so the wait ends about one round trip after the last unit lands, however many were missing
template <bool kDiag, uint32_t kNum, typename unit_ptr_t>
CUTLASS_DEVICE void await_all(uint4 (&v)[kNum], const unit_ptr_t& unit_ptr, const Site& site) {
    const auto any_blank = [&]() {
        bool blank = false;
        #pragma unroll
        for (uint32_t i = 0; i < kNum; ++ i)
            blank |= is_blank(v[i]);
        return blank;
    };
    spin<kDiag>(any_blank, [&]() {
        #pragma unroll
        for (uint32_t i = 0; i < kNum; ++ i)
            if (is_blank(v[i])) v[i] = load(unit_ptr(i));
    }, site);
}

// Flags. `raise` is called by one thread after every writer fenced (`__threadfence()`, device scope) and the CTA
// synchronized; `await_raised` returns once the flag holds this launch's epoch. Writer and readers are always on
// the same GPU, so GPU scope is enough; what other GPUs need is proven by data, never by a flag.
CUTLASS_DEVICE void raise(uint64_t* flag, const uint64_t& epoch) {
    ptx::st_rel(flag, epoch);
}

CUTLASS_DEVICE bool is_raised(const uint64_t* flag, const uint64_t& epoch) {
    return ptx::ld_acq_gpu(flag) == epoch;
}

CUTLASS_DEVICE void await_raised(const uint64_t* flag, const uint64_t& epoch, const Site& site) {
    wait_until([&]() { return is_raised(flag, epoch); }, [&]() {
        printf("DeepGEMM IB flag timeout: %s rank=%u sm=%u thread=%u a=%u b=%u\n",
               site.name, site.rank, blockIdx.x, threadIdx.x, site.a, site.b);
    });
}

// Re-arm `num_bytes` (a multiple of 4) at `region` with 16-byte units of `pattern`, split over the whole grid:
// thread `gtid` of `num_threads` takes every `num_threads`-th unit. Whatever does not fill a last unit is written
// word by word by thread 0 (an odd number of top-k index entries leaves 8 bytes: missing them left a stale
// index for a later launch, about one launch in 1500).
// A diagnostics build traps on a unit that still holds `pattern`: something is re-armed that never landed.
template <bool kDiag>
CUTLASS_DEVICE void rearm(void* region, const uint64_t& num_bytes, const uint4& pattern,
                          const uint32_t& gtid, const uint32_t& num_threads, const Site& site) {
    const auto units = static_cast<uint4*>(region);
    const uint64_t num_units = num_bytes / 16;
    for (uint64_t i = gtid; i < num_units; i += num_threads) {
        if constexpr (kDiag) {
            const uint4 v = load(units + i);
            if (v.x == pattern.x and v.y == pattern.y and v.z == pattern.z and v.w == pattern.w) {
                printf("DeepGEMM IB re-arm of a unit that never landed: %s rank=%u unit=%llu a=%u\n",
                       site.name, site.rank, static_cast<unsigned long long>(i), site.a);
                DG_DEVICE_ASSERT(false and "IB re-arm before landing");
            }
        }
        units[i] = pattern;
    }
    if (gtid == 0) {
        const uint32_t words[4] = {pattern.x, pattern.y, pattern.z, pattern.w};
        for (uint64_t i = 0; i < (num_bytes % 16) / 4; ++ i)
            reinterpret_cast<uint32_t*>(units + num_units)[i] = words[i];
    }
}

} // namespace deep_gemm::comm::landing
