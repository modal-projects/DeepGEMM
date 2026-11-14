#!/usr/bin/env bash
# The IN-REPO test (tests/test_mega_moe.py) across the nodes of a cluster, the way a user of the repository runs it:
# DeepGEMM at upstream 5081f85 + PATCH (= git diff 5081f85 <tree under test>), DG_MEGA_MOE_IB=1, no DG_MEGA_MOE_DEV, no plan
# overrides, no dev server, no thread pinning. Reference = DeepEP + tilelang (correctness), as the test does by itself.
# Runs on every node of the cluster (submit.py); needs NO dev server on the cluster.
#   PATCH=<tree>.patch   CONFIGS="flash:128 flash:1024 megaseek:128 megaseek:1024"   NUM_CORRECTNESS=20
#   TOL=1e-3   the test's DG_MEGA_MOE_IB_TOLERANCE: without it the test asserts BIT equality with the reference, which the
#              IB path does not give across NVLink domains (fp8 return records); it is a test variable, not a host switch.
set -uo pipefail
JOBDIR=$PWD
PATCH=${PATCH:?PATCH=<file shipped with --file>}; CONFIGS=${CONFIGS:-"flash:128 flash:1024 megaseek:128 megaseek:1024"}
NUM_CORRECTNESS=${NUM_CORRECTNESS:-20}; TOL=${TOL:-1e-3}
for v in DG_MEGA_MOE_DEV DG_MEGA_MOE_IB_ORACLE DG_MEGA_MOE_IB_TRACE DG_DEV_COMPILE_ONLY DG_MOE_NO_RENDEZVOUS; do unset $v; done
env | grep -q "^DG_MEGA_MOE_IBP_" && { echo "PLAN_OVERRIDE_IN_ENV: refuse (this job is the plain path)"; exit 2; }
source ./setup_deps.sh
setup_deepep || { echo "DEEPEP_BUILD_FAILED"; exit 2; }

# DeepGEMM = upstream commit + the patch (rebuilt whenever the patch changes)
PATCH_SUM=$(sha256sum $PATCH | cut -c1-12)
if [ "$(cat $SRC/deepgemm_ib.sum 2>/dev/null)" != "$PATCH_SUM" ]; then
  rm -rf $SRC/deepgemm_ib
  clone DeepGEMM deepgemm_ib $DEEPGEMM_COMMIT || exit 2
  git -C $SRC/deepgemm_ib apply $PWD/$PATCH || { echo "PATCH_APPLY_FAILED"; exit 2; }
  (cd $SRC/deepgemm_ib && rm -rf build dist *.egg-info && DG_FORCE_BUILD=1 python setup.py bdist_wheel \
     && pip install --no-deps --force-reinstall dist/*.whl) > /tmp/deepgemm_ib_install.log 2>&1 \
     || { echo "DEEPGEMM_IB_BUILD_FAILED"; tail -60 /tmp/deepgemm_ib_install.log; exit 2; }
  echo $PATCH_SUM > $SRC/deepgemm_ib.sum
fi
python -c "import deep_gemm, deep_ep; print('deep_gemm', deep_gemm.__file__); print('deep_ep', deep_ep.__file__)" || exit 2

# The reference pipeline needs tilelang; keep torch pinned and NCCL at 2.32.3
if ! python -c "import tilelang" 2>/dev/null; then
  TV=$(python -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==$TV" > /tmp/torch_pin.txt
  pip install -q -c /tmp/torch_pin.txt tilelang==0.1.15 > /tmp/tilelang_install.log 2>&1 || {
    pip uninstall -y -q tilelang apache-tvm-ffi > /dev/null 2>&1; pip install -q --no-deps tilelang==0.1.15 > /dev/null 2>&1
    for i in 1 2 3 4 5 6 7 8; do
      miss=$(python -c "import tilelang" 2>&1 | grep -o "No module named '[^']*'" | head -1 | cut -d"'" -f2 | cut -d. -f1)
      [ -z "$miss" ] && break
      case $miss in tvm_ffi) pkg=apache-tvm-ffi;; *) pkg=$miss;; esac
      pip install -q -c /tmp/torch_pin.txt $pkg > /dev/null 2>&1
    done; }
fi
python -c "import tilelang; print('tilelang', tilelang.__version__)" 2>&1 | tail -1
pip install -q --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3 > /dev/null 2>&1
python -c "import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())"

# The test's own source on this tree: its arguments, and that it lays its inputs out by get_block_m_for_mega_moe
T=$SRC/deepgemm_ib/tests/test_mega_moe.py
for a in --num-processes --hidden --intermediate-hidden --num-experts --num-topk --num-max-tokens-per-rank --num-tokens \
         --num-shared-experts --mma-type --num-correctness-tests; do grep -q -- "'$a'" $T || { echo "TEST_ARG_MISSING $a"; exit 2; }; done
grep -q "get_block_m_for_mega_moe" $T && echo "test lays out inputs by get_block_m_for_mega_moe: yes" || { echo "TEST_DOES_NOT_USE_get_block_m"; exit 2; }

# The JIT compiles kernels that include <nccl.h>: the include directory of the NCCL in use must be given
export DG_NCCL_INCLUDE_DIR=$NCCL_ROOT/include
export DG_MEGA_MOE_IB=1 DG_MEGA_MOE_IB_TOLERANCE=$TOL NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export WORLD_SIZE=$NNODES RANK=$NODE_RANK
BASE_PORT=$((MASTER_PORT + 11)); N=0
cd $SRC/deepgemm_ib
run_test() {  # label, experts, top-k, tokens, correctness tests, [extra env as KEY=VAL]
  local label=$1 experts=$2 topk=$3 tokens=$4 nc=$5 extra=${6:-}
  N=$((N + 1)); export MASTER_PORT=$((BASE_PORT + 7 * N))
  echo "==== RUN $label: experts=$experts topk=$topk tokens=$tokens correctness=$nc $extra (node $NODE_RANK, $(date -u +%T))"
  local t0=$(date +%s)
  env $extra timeout ${TEST_TIMEOUT:-900} python tests/test_mega_moe.py --num-processes ${GPUS_PER_NODE:-8} --hidden 5120 --intermediate-hidden 2304 \
    --num-experts $experts --num-topk $topk --num-max-tokens-per-rank $tokens --num-tokens $tokens --num-shared-experts 1 \
    --mma-type fp8xfp4 --num-correctness-tests $nc > /tmp/inrepo_$N.log 2>&1
  local rc=$?
  grep -v "SyncActivityProfiler\|USDT\|ibv_query_port_speed\|DOCA failure\|GDRCopy\|rma_socket" /tmp/inrepo_$N.log \
    | grep -E "MegaMoEIBPlan|planned for|MegaMoEConfig|Config:| > |IB_ERR|Performance|EP +[0-9]+/|passed|Error|error|Assert|Traceback|timeout|Failed" | tail -${TAIL_LINES:-90}
  local worst=$(grep -o "calc_diff=[0-9.e+-]*" /tmp/inrepo_$N.log | cut -d= -f2 | sort -g | tail -1)
  echo "==== RESULT $label: exit $rc, $(( $(date +%s) - t0 )) s, max calc_diff printed ${worst:-none}, correctness lines $(grep -c 'Correctness test .* passed' /tmp/inrepo_$N.log), $(grep -m1 -o ' [0-9]* us, reduction' /tmp/inrepo_$N.log | head -1)"
  [ $rc -ne 0 ] && tail -25 /tmp/inrepo_$N.log
  cp /tmp/inrepo_$N.log $RESULTS_DIR/inrepo_${label}_node$NODE_RANK.log 2>/dev/null
  return $rc
}
FAIL=0
# One launch with DG_PRINT_CONFIGS=1: the "MegaMoEIBPlan(planned for <rate> Gb/s per GPU)" line
run_test print_configs_flash128 384 6 128 1 DG_PRINT_CONFIGS=1 || FAIL=$((FAIL + 1))
for c in $CONFIGS; do
  shape=${c%%:*}; tokens=${c##*:}
  case $shape in flash) E=384; K=6;; megaseek) E=1216; K=19;; gigaseek) E=1600; K=25;; *) echo "unknown shape $shape"; exit 2;; esac
  # The test's DeepEP reference waits only 10 s for its peers (num_gpu_timeout_secs=10 in the test) while every rank compiles the
  # reference's kernels on its own at their first use: a cold first run can die there. One retry (the JIT caches are warm then).
  run_test ${shape}_$tokens $E $K $tokens $NUM_CORRECTNESS || { echo "==== RETRY ${shape}_$tokens (first attempt failed: see above)"; run_test ${shape}_${tokens}_retry $E $K $tokens $NUM_CORRECTNESS || FAIL=$((FAIL + 1)); }
done
echo "INREPO_TEST_DONE failures=$FAIL patch=$PATCH ($PATCH_SUM) node=$NODE_RANK"
exit $FAIL
