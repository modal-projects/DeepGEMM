#define DG_MEGA_MOE_IB 1
#define DG_MEGA_MOE_IB_TRACE 1
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        128, 5120, 2304, 384, 1, 6,
        32, 128, 128, 16, 16, 128, 128,
        4096, 16384, 6, 2560,
        128, 128, 256,
        148, 16, 10.0f, true, cutlass::detail::float_e2m1_unpacksmem_t>);
}
