#!/bin/bash
# Gate 1: with IB off, a tree compiles the Mega MoE kernel to the same PTX as upstream. Front end only: needs nvcc, no GPU.
#
#   UPSTREAM=<upstream checkout at 5081f85> REPO=<tree under test> NVIDIA=<site-packages/nvidia> ./gate1.sh
#
#   UPSTREAM, REPO   checkouts with submodules (cutlass and deep_jit are taken from REPO/third-party)
#   NVIDIA           the `nvidia` package directory of a Python environment that has the CUDA 13 toolkit wheels and
#                    nvidia-nccl-cu13 (nvcc at $NVIDIA/cu13/bin/nvcc); or set NVCC, CUDA_INC and NCCL_INC yourself
#   OUT              work directory (default: a new temporary one)
# Prints, per shape and tree: exit status, PTX lines, md5 of the normalised PTX. The gate passes when the two md5 of
# each shape are equal. Exit status 0 = pass.
# nvcc may exit non-zero after writing the PTX (its assembler can reject the PTX version): the gate reads the PTX text.
set -u
: "${UPSTREAM:?set UPSTREAM to a checkout of upstream DeepGEMM}" "${REPO:?set REPO to the tree under test}"
NVCC=${NVCC:-${NVIDIA:?set NVIDIA or NVCC}/cu13/bin/nvcc}
CUDA_INC=${CUDA_INC:-$NVIDIA/cu13/include}
NCCL_INC=${NCCL_INC:-$NVIDIA/nccl/include}
W=${OUT:-$(mktemp -d)}
mkdir -p "$W"

gen() { # name ranks experts topk tokens block_m store_m_l1 store_m_l2 sf_block_m ring_tokens sf_ring_tokens
cat > "$W/$1.cu" <<EOT
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        $5, 5120, 2304, $3, 1, $4,
        $6, 128, 128, $7, $8, $9, 128,
        ${10}, ${11}, 4, 5120,
        128, 128, 256,
        148, $2, 10.0f, true, cutlass::detail::float_e2m1_unpacksmem_t>);
}
EOT
}
# Normalisation: drop comments and debug lines; internal-linkage names carry the compiler's process id in a
# length-prefixed mangled name (the first expression: without it an unchanged tree changes md5 when the process id
# gains a digit); string constants that hold include paths are dropped
norm() {
  grep -v '^//\|\.file\|\.loc' "$1" \
    | sed -E 's/_ZN[0-9]+_INTERNAL_[0-9a-f]+_[0-9a-zA-Z_]+/_ZN_I/g; s/\$str[0-9$_a-zA-Z]*//g; s/_INTERNAL_[0-9a-f]+_[0-9a-zA-Z_]+//g' \
    | grep -v "^.global .align 1 .b8 \[[0-9]*\] = {47, "
}
cc() { # name include-dir tag
  "$NVCC" -std=c++20 --gpu-architecture=compute_100a --ptx -o "$W/$1_$3.ptx" "$W/$1.cu" -I"$2" \
    -I"$REPO/third-party/cutlass/include" -I"$REPO/third-party/deep_jit/include" \
    -I"$CUDA_INC" -I"$CUDA_INC/cccl" -I"$NCCL_INC" --diag-suppress=39,161,174,177,186,940 --expt-relaxed-constexpr -O1 \
    > "$W/$1_$3.log" 2>&1
  local rc=$?
  norm "$W/$1_$3.ptx" 2>/dev/null | md5sum | cut -c1-12 > "$W/$1_$3.md5"
  echo "$1 $3 exit $rc lines $(wc -l < "$W/$1_$3.ptx" 2>/dev/null) md5 $(cat "$W/$1_$3.md5")"
}
gen megaseek8 8 304 19 1024 240 24 8 256 15360 16384
gen flash8 8 96 6 1024 192 32 32 256 15360 20480
for n in megaseek8 flash8; do
  cc $n "$UPSTREAM/deep_gemm/include" upstream &
  cc $n "$REPO/deep_gemm/include" tree &
done
wait
status=0
for n in megaseek8 flash8; do
  if [ "$(wc -l < "$W/${n}_upstream.ptx" 2>/dev/null || echo 0)" -lt 1000 ]; then
    echo "$n: upstream did not compile, see $W/${n}_upstream.log"; status=1
  elif cmp -s "$W/${n}_upstream.md5" "$W/${n}_tree.md5"; then echo "$n: PASS"
  else echo "$n: FAIL (see $W/${n}_tree.log; diff the two .ptx after the normalisation above)"; status=1; fi
done
exit $status
