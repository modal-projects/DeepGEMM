"""Mega MoE over IB: NCCL window + Gin device communicator for the symmetric buffer (DESIGN-v1 A1/A2).

The whole Mega MoE buffer is one NCCL window. Same-node peers are reached by load/store through NCCL's
LSA pointers (these replace torch symm_mem's peer pointers); ranks on other nodes are reached with NCCL
Gin from inside the kernel. The kernel receives the Gin handles appended to `buffer_ptrs` (see
`deep_gemm/include/deep_gemm/layout/sym_buffer.cuh`, kIBMagic).
"""
import os
import types

import torch
import torch.distributed as dist

_ext = None
IB_MAGIC = 0x1B1B1B1B1B1B1B1B
# DG_MEGA_MOE_IB_SHARING=2: exclusive THREAD-mode Gin contexts (one issuer per context; see comm/gin.cuh). The kernel
# needs R * (4 + 2 * ceil(max_tokens / 32)) contexts (R = remote nodes): 72 covers EP16 <= 1024 tok and EP32 <= 256 tok.
# A deep send queue keeps a put from ever polling the CQ for a free slot. Both are read by NCCL / the JIT later.
SHARING = int(os.environ.get('DG_MEGA_MOE_IB_SHARING', '0'))
if SHARING >= 1:
    os.environ.setdefault('DG_MEGA_MOE_IB_CONTEXTS', '72')
    os.environ.setdefault('NCCL_GIN_GDAKI_QP_DEPTH', '1024')
if os.environ.get('DG_MEGA_MOE_IB_SKIP_CREDIT'):  # skipping the availability check needs a deep send queue
    os.environ.setdefault('NCCL_GIN_GDAKI_QP_DEPTH', '1024')
NUM_GIN_CONTEXTS = int(os.environ.get('DG_MEGA_MOE_IB_CONTEXTS', '4'))

_CPP = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <vector>
#include <cstdlib>

#define CHECK_NCCL(x) do { ncclResult_t r = (x); TORCH_CHECK(r == ncclSuccess, "NCCL: ", ncclGetErrorString(r), " at ", #x); } while (0)
#define CHECK_CUDA(x) do { cudaError_t e = (x); TORCH_CHECK(e == cudaSuccess, "CUDA: ", cudaGetErrorString(e), " at ", #x); } while (0)

struct IBState {
    ncclComm_t comm = nullptr;
    ncclWindow_t window = nullptr;
    void* ptr = nullptr;
    void* dev_comm_host = nullptr;
    void* dev_comm_dev = nullptr;
};

std::vector<int64_t> nccl_unique_id() {
    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));
    std::vector<int64_t> out(sizeof(id));
    for (size_t i = 0; i < sizeof(id); ++ i) out[i] = static_cast<uint8_t>(id.internal[i]);
    return out;
}

// Returns (state handle, buffer ptr, per-rank LSA ptrs (0 for other nodes), dev_comm device ptr, window, lsa nranks)
std::tuple<int64_t, int64_t, std::vector<int64_t>, int64_t, int64_t, int64_t>
create(const std::vector<int64_t>& uid, int64_t rank, int64_t world, int64_t num_bytes, int64_t num_contexts, int64_t gin_type,
       int64_t virtual_lsa, int64_t exclusive) {
    auto* st = new IBState();
    ncclUniqueId id;
    for (size_t i = 0; i < sizeof(id); ++ i) id.internal[i] = static_cast<char>(uid[i]);
    CHECK_NCCL(ncclCommInitRank(&st->comm, static_cast<int>(world), id, static_cast<int>(rank)));

    CHECK_NCCL(ncclMemAlloc(&st->ptr, static_cast<size_t>(num_bytes)));
    CHECK_CUDA(cudaMemset(st->ptr, 0, static_cast<size_t>(num_bytes)));
    CHECK_NCCL(ncclCommWindowRegister(st->comm, st->ptr, static_cast<size_t>(num_bytes), &st->window, NCCL_WIN_STRICT_ORDERING));

    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
    CHECK_NCCL(ncclCommQueryProperties(st->comm, &props));
    ncclTeam_t lsa = ncclTeamLsa(st->comm);
    // virtual_lsa > 0: treat groups of `virtual_lsa` consecutive ranks as separate NVLink domains, so ranks in
    // other groups (even on this node) are reached with Gin; single-node bitwise checks of every IB path
    const int64_t num_lsa = virtual_lsa > 0 ? virtual_lsa : lsa.nRanks;
    const bool need_gin = num_lsa < world;
    TORCH_CHECK(not need_gin or props.ginType != NCCL_GIN_TYPE_NONE,
                "NCCL Gin is unavailable on this communicator (ginType=", props.ginType, ")");
    if (need_gin) {

    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    const bool rail = std::getenv("DG_MEGA_MOE_IB_RAIL") != nullptr;
    // 0 = any backend NCCL can set up (GDAKI needs host driver PeerMappingOverride; PROXY=2 works without)
    reqs.ginType = static_cast<ncclGinType_t>(gin_type);
    reqs.ginContextCount = static_cast<int>(num_contexts);
    reqs.ginExclusiveContexts = exclusive != 0;
    // DG_MEGA_MOE_IB_RAIL=1: rail-only Gin connections (DeepEP hybrid mode does this): the kernel only puts to its
    // counterparts (same local index), and the CPU proxy scans every QP (bench: one-way 5.23 / 7.07 us at 1 / 72 contexts)
    reqs.ginConnectionType = rail ? NCCL_GIN_CONNECTION_RAIL : NCCL_GIN_CONNECTION_FULL;
    reqs.ginSignalCount = 4;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)
    reqs.useRuntimeVersion = true;
    const size_t dev_comm_size = props.devCommRuntimeVersionSize;
#else
    const size_t dev_comm_size = sizeof(ncclDevComm_t);
#endif
    st->dev_comm_host = malloc(dev_comm_size);
    CHECK_NCCL(ncclDevCommCreate(st->comm, &reqs, static_cast<ncclDevComm_t*>(st->dev_comm_host)));
    {
        // ginContextCount is a hint: report what the devcomm actually holds. Device-side, context index i maps to
        // connection i % ginConnectionCount and context i / ginConnectionCount with no bound check, so an index beyond
        // ginContextCount * ginConnectionCount reads past the context arrays.
        const auto* dc = static_cast<const ncclDevComm_t*>(st->dev_comm_host);
        const int64_t have = static_cast<int64_t>(dc->ginContextCount) * static_cast<int64_t>(dc->ginConnectionCount);
        if (rank == 0)
            printf("DG_IB_GIN rail=%d requested_contexts=%lld exclusive=%d devcomm.ginContextCount=%u ginConnectionCount=%u usable_indices=%lld\n",
                   (int) rail, (long long) num_contexts, (int) exclusive, dc->ginContextCount, (unsigned) dc->ginConnectionCount, (long long) have);
        TORCH_CHECK(exclusive == 0 or have >= num_contexts, "NCCL Gin granted ", have, " context indices (ginContextCount ",
                    dc->ginContextCount, " x connections ", (int) dc->ginConnectionCount, "), fewer than the ", num_contexts,
                    " requested with exclusive contexts");
    }
    CHECK_CUDA(cudaMalloc(&st->dev_comm_dev, dev_comm_size));
    CHECK_CUDA(cudaMemcpy(st->dev_comm_dev, st->dev_comm_host, dev_comm_size, cudaMemcpyHostToDevice));
    }
    const int64_t node_first = rank - lsa.rank;
    const int64_t group_first = rank - rank % num_lsa;
    std::vector<int64_t> ptrs(static_cast<size_t>(world), 0);
    for (int i = 0; i < lsa.nRanks; ++ i) {
        const int64_t r = node_first + i;
        if (r < group_first or r >= group_first + num_lsa)
            continue;  // outside this (virtual) NVLink domain: Gin only
        void* p = nullptr;
        CHECK_NCCL(ncclGetLsaDevicePointer(st->window, 0, i, &p));
        ptrs[static_cast<size_t>(r)] = reinterpret_cast<int64_t>(p);
    }
    TORCH_CHECK(ptrs[static_cast<size_t>(rank)] == reinterpret_cast<int64_t>(st->ptr) or ptrs[static_cast<size_t>(rank)] != 0);
    return {reinterpret_cast<int64_t>(st), reinterpret_cast<int64_t>(st->ptr), ptrs,
            reinterpret_cast<int64_t>(st->dev_comm_dev), reinterpret_cast<int64_t>(st->window), num_lsa};
}

torch::Tensor as_tensor(int64_t ptr, int64_t num_bytes, int64_t device) {
    auto opts = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA, static_cast<int>(device));
    return torch::from_blob(reinterpret_cast<void*>(ptr), {num_bytes}, [](void*) {}, opts);
}

void destroy(int64_t handle) {
    auto* st = reinterpret_cast<IBState*>(handle);
    if (st->dev_comm_host != nullptr) {
        ncclDevCommDestroy(st->comm, static_cast<ncclDevComm_t*>(st->dev_comm_host));
        cudaFree(st->dev_comm_dev);
        free(st->dev_comm_host);
    }
    ncclCommWindowDeregister(st->comm, st->window);
    ncclMemFree(st->ptr);
    ncclCommDestroy(st->comm);
    delete st;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nccl_unique_id", &nccl_unique_id);
    m.def("create", &create);
    m.def("as_tensor", &as_tensor);
    m.def("destroy", &destroy);
}
'''


def nccl_root() -> str:
    import nvidia
    return os.path.join(list(nvidia.__path__)[0], 'nccl')


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline
        root = nccl_root()
        os.environ.setdefault('DG_NCCL_INCLUDE_DIR', os.path.join(root, 'include'))
        _ext = load_inline(
            name='dg_mega_moe_ib', cpp_sources=[_CPP], functions=None,
            extra_include_paths=[os.path.join(root, 'include')],
            extra_cflags=['-O2', '-std=c++20'],
            extra_ldflags=[f'-L{os.path.join(root, "lib")}', '-l:libnccl.so.2', f'-Wl,-rpath,{os.path.join(root, "lib")}'],
            with_cuda=True, verbose=False)
    return _ext


def enabled() -> bool:
    return os.environ.get('DG_MEGA_MOE_IB', '0') == '1'


def allocate(group: dist.ProcessGroup, num_bytes: int):
    """Allocate the Mega MoE symmetric buffer as one NCCL window. Returns (tensor, handle namespace)."""
    ext = _load()
    rank, world = group.rank(), group.size()
    uid = [ext.nccl_unique_id()] if rank == 0 else [None]
    dist.broadcast_object_list(uid, src=dist.get_global_rank(group, 0), group=group)
    gin_type = int(os.environ.get('DG_MEGA_MOE_IB_GIN_TYPE', '0'))  # 0 any, 2 proxy, 3 GDAKI, 5 EFA GDA
    virtual_lsa = int(os.environ.get('DG_MEGA_MOE_IB_VIRTUAL_LSA', '0'))
    state, ptr, lsa_ptrs, dev_comm, window, num_lsa = ext.create(uid[0], rank, world, num_bytes, NUM_GIN_CONTEXTS, gin_type,
                                                                 virtual_lsa, int(os.environ.get('DG_MEGA_MOE_IB_EXCLUSIVE', '1' if SHARING >= 1 else '0')))
    tensor = ext.as_tensor(ptr, num_bytes, torch.cuda.current_device())
    # DG_MEGA_MOE_IB_FORCE=1: compile the IB kernel even when all ranks share one NVLink domain (compile and
    # local-path checks on single-node clusters; Gin paths are never taken since every rank is load/store-reachable)
    force = os.environ.get('DG_MEGA_MOE_IB_FORCE', '0') == '1'
    buffer_ptrs = list(lsa_ptrs) + ([IB_MAGIC, dev_comm, window, num_lsa] if (num_lsa < world or force) else [])
    handle = types.SimpleNamespace(buffer_ptrs=buffer_ptrs, ib_state=state, num_lsa_ranks=num_lsa)
    return tensor, handle
