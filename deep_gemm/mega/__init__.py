import torch
import types
import atexit
import json
import os
import signal
import threading
import time
from typing import Tuple, Optional
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


MEGAMOE_KERNEL_TIMING_FIELD_NAMES = [
    "cta_samples",
    "cta_total_cycles",
    "recv_count_wait_cycles",
    "recv_count_wait_polls",
    "dispatch_nvlink_barrier_cycles",
    "cleanup_nvlink_barrier_cycles",
    "combine_nvlink_barrier_cycles",
    "l1_arrival_wait_cycles",
    "l1_arrival_wait_polls",
    "l2_arrival_wait_cycles",
    "l2_arrival_wait_polls",
    "pull_mbarrier_wait_cycles",
    "full_barrier_wait_cycles",
    "tmem_empty_wait_cycles",
    "tmem_full_wait_cycles",
    "combine_load_wait_cycles",
]

_MEGAMOE_TIMING_BUFFERS = []
_MEGAMOE_TIMING_LOCK = threading.Lock()
_MEGAMOE_TIMING_LAST_DUMP_TRIGGER_NS = None
_MEGAMOE_TIMING_LAST_RESET_TRIGGER_NS = None


def _timing_enabled_default() -> bool:
    return (
        os.getenv("SGLANG_ENABLE_MEGAMOE_KERNEL_TIMING", "0") == "1"
        or os.getenv("ENABLE_MEGAMOE_KERNEL_TIMING", "0") == "1"
    )


def _timing_path() -> str:
    return os.getenv(
        "SGLANG_MEGAMOE_KERNEL_TIMING_PATH",
        os.getenv("MEGAMOE_KERNEL_TIMING_PATH", "/tmp/megamoe_kernel_timing.jsonl"),
    )


def _timing_trigger_path() -> str:
    return os.getenv(
        "SGLANG_MEGAMOE_KERNEL_TIMING_TRIGGER_PATH",
        os.getenv(
            "MEGAMOE_KERNEL_TIMING_TRIGGER_PATH",
            os.path.join(os.path.dirname(_timing_path()), "dump.trigger"),
        ),
    )


def _timing_reset_trigger_path() -> str:
    return os.getenv(
        "SGLANG_MEGAMOE_KERNEL_TIMING_RESET_TRIGGER_PATH",
        os.getenv(
            "MEGAMOE_KERNEL_TIMING_RESET_TRIGGER_PATH",
            os.path.join(os.path.dirname(_timing_path()), "reset.trigger"),
        ),
    )


def _timing_fields() -> int:
    return int(
        os.getenv(
            "SGLANG_MEGAMOE_KERNEL_TIMING_FIELDS",
            os.getenv("MEGAMOE_KERNEL_TIMING_FIELDS", "32"),
        )
    )


def _timing_rank() -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.getenv("RANK", os.getenv("LOCAL_RANK", "-1")))


def dump_mega_moe_kernel_timing(
    path: Optional[str] = None,
    reason: str = "manual",
    include_empty: bool = False,
) -> None:
    if not _timing_enabled_default():
        return
    with _MEGAMOE_TIMING_LOCK:
        records = []
        buffers = list(_MEGAMOE_TIMING_BUFFERS)
        for sym_buffer in buffers:
            timing = getattr(sym_buffer, "mega_moe_timing", None)
            if timing is None:
                continue
            try:
                torch.cuda.synchronize(timing.device)
                values = [int(v) for v in timing.detach().cpu().tolist()]
            except Exception as exc:
                values = []
                error = repr(exc)
            else:
                error = None
            records.append(
                {
                    "ts": time.time(),
                    "event": "timing",
                    "reason": reason,
                    "pid": os.getpid(),
                    "rank": _timing_rank(),
                    "buffer_count": len(buffers),
                    "num_max_tokens_per_rank": sym_buffer.num_max_tokens_per_rank,
                    "num_experts": sym_buffer.num_experts,
                    "num_topk": sym_buffer.num_topk,
                    "field_names": MEGAMOE_KERNEL_TIMING_FIELD_NAMES,
                    "values": values,
                    "error": error,
                }
            )
        if not records and include_empty:
            records.append(
                {
                    "ts": time.time(),
                    "event": "no_buffers",
                    "reason": reason,
                    "pid": os.getpid(),
                    "rank": _timing_rank(),
                    "buffer_count": len(buffers),
                    "field_names": MEGAMOE_KERNEL_TIMING_FIELD_NAMES,
                    "values": [],
                    "error": None,
                }
            )
        if not records:
            return

        out_path = path or _timing_path()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "a") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")


def reset_mega_moe_kernel_timing(reason: str = "manual") -> None:
    if not _timing_enabled_default():
        return
    with _MEGAMOE_TIMING_LOCK:
        buffers = list(_MEGAMOE_TIMING_BUFFERS)
        for sym_buffer in buffers:
            timing = getattr(sym_buffer, "mega_moe_timing", None)
            if timing is not None:
                timing.zero_()
        try:
            out_path = _timing_path() + ".events"
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "event": "reset",
                            "reason": reason,
                            "pid": os.getpid(),
                            "rank": _timing_rank(),
                            "buffer_count": len(buffers),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        except Exception:
            pass


def _timing_trigger_loop() -> None:
    global _MEGAMOE_TIMING_LAST_DUMP_TRIGGER_NS
    global _MEGAMOE_TIMING_LAST_RESET_TRIGGER_NS
    while True:
        try:
            dump_path = _timing_trigger_path()
            if os.path.exists(dump_path):
                mtime_ns = os.stat(dump_path).st_mtime_ns
                if mtime_ns != _MEGAMOE_TIMING_LAST_DUMP_TRIGGER_NS:
                    _MEGAMOE_TIMING_LAST_DUMP_TRIGGER_NS = mtime_ns
                    dump_mega_moe_kernel_timing(
                        reason="trigger_file", include_empty=True
                    )
            reset_path = _timing_reset_trigger_path()
            if os.path.exists(reset_path):
                mtime_ns = os.stat(reset_path).st_mtime_ns
                if mtime_ns != _MEGAMOE_TIMING_LAST_RESET_TRIGGER_NS:
                    _MEGAMOE_TIMING_LAST_RESET_TRIGGER_NS = mtime_ns
                    reset_mega_moe_kernel_timing(reason="trigger_file")
        except Exception:
            pass
        time.sleep(1.0)


def _install_timing_hooks() -> None:
    if not _timing_enabled_default():
        return
    atexit.register(
        lambda: dump_mega_moe_kernel_timing(reason="atexit", include_empty=True)
    )
    try:
        signal.signal(
            signal.SIGUSR1,
            lambda *_: dump_mega_moe_kernel_timing(
                reason="signal", include_empty=True
            ),
        )
    except Exception:
        pass
    threading.Thread(target=_timing_trigger_loop, daemon=True).start()


_install_timing_hooks()


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 use_fp8_dispatch: bool = True,
                 activation: str = 'swiglu',
                 enable_kernel_timing: Optional[bool] = None):
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.enable_kernel_timing = (
            _timing_enabled_default()
            if enable_kernel_timing is None
            else enable_kernel_timing
        )

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            use_fp8_dispatch, activation
        )
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()
        if self.enable_kernel_timing:
            self.mega_moe_timing = torch.zeros(
                _timing_fields(), dtype=torch.int64, device='cuda'
            )
            _MEGAMOE_TIMING_BUFFERS.append(self)
        else:
            self.mega_moe_timing = None

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.mega_moe_timing = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 use_fp8_dispatch: bool = True,
                                 activation: str = 'swiglu',
                                 enable_kernel_timing: Optional[bool] = None) -> SymmBuffer:
    # Token count must be aligned to block sizes
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch, activation,
        enable_kernel_timing
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))


def _interleave_l1_weights(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _interleave_weights(l1_weights[0]), _interleave_weights(l1_weights[1])


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    # L1: interleave gate/up for weight and SF, then transpose SF for UTCCP.
    l1_w = _interleave_weights(l1_weights[0])
    l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
    l1_transformed = (l1_w, l1_sf)
    # L2: only transpose SF for UTCCP.
    l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    return l1_transformed, l2_transformed



def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True,
                     enable_kernel_timing: Optional[bool] = None):
    if enable_kernel_timing is None:
        enable_kernel_timing = getattr(sym_buffer, "enable_kernel_timing", False)
    kernel_timing = (
        getattr(sym_buffer, "mega_moe_timing", None)
        if enable_kernel_timing
        else None
    )
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        kernel_timing,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        enable_kernel_timing
    )
