#!/usr/bin/env bash
# Mega MoE dev server (devserver.py): build once, hold the ranks warm, answer A/B requests from ../devclient.py.
# Ship with: --file jobs/devserver.py --file jobs/setup_deps.sh --file jobs/nccl_topo.py --file jobs/<PATCH>
#   PATCH=dev*.patch       DeepGEMM patch (git diff 5081f85), built into /opt/devserver/py (site-packages untouched)
#   DEV_BASE_COMMIT=sha    commit the patch was made from (devclient --repo diffs include files against it)
#   DEV_HOURS=4            supervisor exits cleanly after this long (keep it below the job --timeout)
#   DEV_WARM=flash:128,1024   cases built + default arm compiled/checked at start ("" = none)
#   DEV_BASE_KNOBS=...     default knob set (arms overlay it); its allocation-time knobs are fixed for the server
#   TEST_NODES / SWEEP_NODES / NO_LEGACY / DG_MEGA_MOE_IB_* : see README.md (B300: TEST_NODES=2 GIN_TYPE=3)
# Stop: python devclient.py --cluster <c> stop   (or submit.py --stop, which also takes the runner down)
set -uo pipefail
JOBDIR=$PWD
TEST_NODES=${TEST_NODES:-2}; IB=${IB:-1}; PATCH=${PATCH:-devserver.patch}
if [ -n "${SWEEP_NODES:-}" ]; then
  if [ "$NODE_RANK" -ge "$SWEEP_NODES" ]; then echo "rank $NODE_RANK idle (SWEEP_NODES=$SWEEP_NODES)"; exit 0; fi
  export NNODES=$SWEEP_NODES
fi
[ "$TEST_NODES" -lt "$NNODES" ] && export NNODES=$TEST_NODES
[ "$NODE_RANK" -ge "$NNODES" ] && { echo "rank $NODE_RANK idle"; exit 0; }
echo "DEVSERVER node=$NODE_RANK nnodes=$NNODES host=$(hostname) $(date -u +%H:%M:%S) cpus=$(nproc)"
T0=$(date +%s)
source ./setup_deps.sh
if [ "${NO_LEGACY:-0}" != 1 ]; then setup_deepep || { echo "DEEPEP_BUILD_FAILED"; exit 2; }; fi

# Patched DeepGEMM in a private target dir; rebuilt only when the patch changes
DEVROOT=/opt/devserver; DGSRC=$SRC/deepgemm_dev; PYDIR=$DEVROOT/py
mkdir -p $DEVROOT
PATCH_SUM=$(sha256sum $PATCH | cut -c1-12)
if [ "$(cat $DEVROOT/py.sum 2>/dev/null)" != "$PATCH_SUM" ]; then
  rm -rf $DGSRC
  if [ -d $SRC/deepgemm_ib/.git ]; then  # reuse a checkout (submodules included) instead of cloning cutlass again
    cp -a $SRC/deepgemm_ib $DGSRC && git -C $DGSRC reset -q --hard $DEEPGEMM_COMMIT && git -C $DGSRC clean -qfdx
  else
    clone DeepGEMM deepgemm_dev $DEEPGEMM_COMMIT || exit 2
  fi
  git -C $DGSRC apply $JOBDIR/$PATCH || { echo "PATCH_APPLY_FAILED"; exit 2; }
  (cd $DGSRC && rm -rf build dist *.egg-info && DG_FORCE_BUILD=1 python setup.py bdist_wheel \
     && rm -rf $PYDIR.new && pip install -q --no-deps --target $PYDIR.new dist/*.whl) > /tmp/deepgemm_dev_install.log 2>&1 \
     || { echo "DEEPGEMM_DEV_BUILD_FAILED"; tail -60 /tmp/deepgemm_dev_install.log; exit 2; }
  rm -rf $PYDIR $DEVROOT/include_orig && mv $PYDIR.new $PYDIR && cp -a $PYDIR/deep_gemm/include $DEVROOT/include_orig
  echo $PATCH_SUM > $DEVROOT/py.sum
fi
export PYTHONPATH=$PYDIR:${PYTHONPATH:-}
export DEV_INCLUDE_ORIG=$DEVROOT/include_orig DEV_DG_SRC=$DGSRC
if [ "${NO_LEGACY:-0}" = 1 ]; then python -c "import deep_gemm; print('deep_gemm', deep_gemm.__file__)" || exit 2
else python -c "import deep_gemm, deep_ep; print('deep_gemm', deep_gemm.__file__); print('deep_ep', deep_ep.__file__)" || exit 2; fi

# Reference pipeline (DeepEP + tilelang SwiGLU) needs tilelang; keep our torch pinned
if [ "${NO_LEGACY:-0}" != 1 ] && ! python -c "import tilelang" 2>/dev/null; then
  TV=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  echo "torch==$TV" > /tmp/torch_pin.txt
  pip install -q -c /tmp/torch_pin.txt tilelang==0.1.15 > /tmp/tilelang_install.log 2>&1 || {
    pip uninstall -y -q tilelang apache-tvm-ffi > /dev/null 2>&1; pip install -q -c /tmp/torch_pin.txt tilelang==0.1.15 > /tmp/tilelang_install2.log 2>&1 || pip install -q --no-deps tilelang==0.1.15 > /dev/null 2>&1
    for i in 1 2 3 4 5 6 7 8; do
      miss=$(python -c "import tilelang" 2>&1 | grep -o "No module named '[^']*'" | head -1 | cut -d"'" -f2 | cut -d. -f1)
      [ -z "$miss" ] && break
      case $miss in tvm_ffi) pkg=apache-tvm-ffi;; *) pkg=$miss;; esac
      echo "tilelang: installing missing $pkg"; pip install -q -c /tmp/torch_pin.txt $pkg > /dev/null 2>&1
    done
  }
fi
pip install -q --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3 > /dev/null 2>&1
python -c "import torch; print('nccl', torch.cuda.nccl.version())"
if [ -n "${NCCL_IB_HCA:-}" ] && [ -f nccl_topo.py ] && [ -z "${NCCL_TOPO_FILE:-}" ] && \
   [ "${MODAL_CLOUD_PROVIDER:-}" = "CLOUD_PROVIDER_UNSPECIFIED" ]; then
  python nccl_topo.py /tmp/nccl_topo.xml > /dev/null && export NCCL_TOPO_FILE=/tmp/nccl_topo.xml
fi
python -c "import modal; print('modal', modal.__version__)" || { echo "NO_MODAL_CLIENT"; exit 2; }

# Server env: the test's, plus the base knobs at process start (ib.py reads some at import)
export DG_MEGA_MOE_IB=$IB DG_NCCL_INCLUDE_DIR=$NCCL_ROOT/include DG_JIT_DEBUG=0 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# JIT cache: private writable root; an existing ~/.dj is a read-only lookup (keys hash the code and every
# included header, so a hit there is content-identical)
export DG_JIT_CACHE_DIR=$DEVROOT/jit$([ -d $HOME/.dj ] && echo ":$HOME/.dj") DG_MEGA_MOE_IB_TOLERANCE=${DG_MEGA_MOE_IB_TOLERANCE:-1e-3}
export DEV_BASE_KNOBS=${DEV_BASE_KNOBS:-DG_MEGA_MOE_IB_SKIP_CREDIT=1,DG_MEGA_MOE_IB_HDR_CTX=1,DG_MEGA_MOE_IB_CONTEXTS=8,DG_MEGA_MOE_IB_FRONT=1,DG_MEGA_MOE_IB_GPU_ACQ=1,DG_MEGA_MOE_IB_STREAM_COMBINE=1,DG_MEGA_MOE_PULL_META_EARLY=1}
for kv in $(echo "$DEV_BASE_KNOBS" | tr ',' ' '); do export "$kv"; done
if [ "${NO_LEGACY:-0}" = 1 ]; then
  mkdir -p /tmp/nolegacy/deep_ep && echo 'raise ImportError("deep_ep hidden by NO_LEGACY")' > /tmp/nolegacy/deep_ep/__init__.py
  export PYTHONPATH=/tmp/nolegacy:$PYTHONPATH
fi
export DEV_ROOT=/tmp/devserver_$JOB_ID
# DEV_WIPE_STORE=1 (default off): remove this node's persisted compile store (cubins, remembered templates) and private JIT
# cache before the server starts, e.g. after a second supervisor published cubins under this node's index
if [ "${DEV_WIPE_STORE:-0}" = 1 ]; then
  rm -rf "$DEVROOT/store" "$DEVROOT/jit"
  echo "DEV_WIPE_STORE: removed $DEVROOT/store and $DEVROOT/jit on node ${NODE_RANK:-?}"
fi
echo "DEVSERVER setup done in $(( $(date +%s) - T0 )) s; starting supervisor $(date -u +%H:%M:%S)"
python devserver.py supervise
rc=$?
echo "DEVSERVER supervisor exit $rc $(date -u +%H:%M:%S)"
exit $rc
