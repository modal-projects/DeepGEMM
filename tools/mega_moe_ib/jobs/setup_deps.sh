# Sourced by jobs: clone and install DeepGEMM and DeepEP once per runner container (pinned commits).
# Installs persist for the runner's lifetime (same container across jobs).
DEEPGEMM_COMMIT=5081f85
DEEPEP_COMMIT=f6aa38f
SRC=/opt/src
mkdir -p $SRC
# the image python was built with clang; build extensions with the system gcc 13 (C++20 std::format)
export CC=gcc CXX=g++ LDSHARED="g++ -shared"
# DeepJIT needs elfutils headers (libdw); running runners predate the image fix
[ -f /usr/include/elfutils/libdwfl.h ] || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libdw-dev libelf-dev >/dev/null 2>&1)
NCCL_ROOT=$(python -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'nccl'))")
export EP_NCCL_ROOT_DIR=$NCCL_ROOT
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-}
# mlx5 hosts: a newer rdma-core, if one was built into /opt/rdma, for NCCL's mlx5dv/Gin paths;
# data-direct queries fail with EINVAL in some VM hosts, so skip them
if [ -f /opt/rdma/lib/libmlx5.so.1 ] && [ "${FI_PROVIDER:-}" != "efa" ]; then
  export LD_LIBRARY_PATH=/opt/rdma/lib:$LD_LIBRARY_PATH NCCL_IB_DATA_DIRECT=0
  # DeepEP sizes SMs from `ibstat` Rate, which fails here (exit 255); answer from sysfs in ibstat's format
  mkdir -p /tmp/ibstat_shim
  cat > /tmp/ibstat_shim/ibstat <<'SHIM'
#!/usr/bin/env bash
for d in /sys/class/infiniband/*; do
  n=$(basename $d); t=$(cat $d/hca_type 2>/dev/null || echo MT4129)
  r=$(cut -d' ' -f1 $d/ports/1/rate 2>/dev/null || echo 400)
  printf "CA '%s'\n\tCA type: %s\n\tNumber of ports: 1\n\tPort 1:\n\t\tState: Active\n\t\tRate: %s\n" "$n" "$t" "$r"
done
SHIM
  chmod +x /tmp/ibstat_shim/ibstat
  export PATH=/tmp/ibstat_shim:$PATH
fi

clone() {  # repo dir commit
  if [ ! -d $SRC/$2/.git ]; then
    git clone -q https://github.com/deepseek-ai/$1.git $SRC/$2 && git -C $SRC/$2 checkout -q $3 \
      && git -C $SRC/$2 submodule update --init --recursive -q \
      || { echo "clone of $1 failed"; return 1; }
  fi
}
setup_deepgemm() {
  python -c "import deep_gemm" 2>/dev/null && return 0
  clone DeepGEMM deepgemm $DEEPGEMM_COMMIT || return 1
  # upstream install.sh ignores build errors and force-reinstalls deps (would replace our torch)
  (cd $SRC/deepgemm && rm -rf build dist *.egg-info && DG_FORCE_BUILD=1 python setup.py bdist_wheel \
     && pip install --no-deps --force-reinstall dist/*.whl) > /tmp/deepgemm_install.log 2>&1
  python -c "import deep_gemm; print('deep_gemm ok', deep_gemm.__file__)" \
    || { echo "--- deepgemm install log"; tail -60 /tmp/deepgemm_install.log; return 1; }
}
setup_deepep() {
  # v2 marker: DeepEP patched to accept any NCCL Gin backend (proxy works without host PeerMappingOverride)
  python -c "import deep_ep" 2>/dev/null && [ -f $SRC/deepep.anygin ] && return 0
  clone DeepEP deepep $DEEPEP_COMMIT || return 1
  sed -i 's|props.ginSupport\[NCCL_GIN_TYPE_GDAKI\] and|true and|; s|reqs.ginType = NCCL_GIN_TYPE_GDAKI;|reqs.ginType = static_cast<ncclGinType_t>(get_env<int>("EP_GIN_TYPE", 0));|' \
    $SRC/deepep/csrc/kernels/comm/context.cpp
  grep -n "EP_GIN_TYPE\|true and" $SRC/deepep/csrc/kernels/comm/context.cpp | head -3
  (cd $SRC/deepep && rm -rf build dist && python setup.py bdist_wheel \
     && pip install --no-deps --force-reinstall dist/*.whl) > /tmp/deepep_install.log 2>&1
  python -c "import deep_ep; print('deep_ep ok', deep_ep.__file__)" \
    || { echo "--- deepep install log"; tail -60 /tmp/deepep_install.log; return 1; }
  touch $SRC/deepep.anygin
}
