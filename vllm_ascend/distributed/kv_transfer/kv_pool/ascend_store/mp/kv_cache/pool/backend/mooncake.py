"""Mooncake backend owned by one multiprocess Worker service."""

from typing import Any

import torch
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

from .....backend.mooncake_backend import MooncakeBackend


class MPMooncakeBackend(MooncakeBackend):
    """Register every Worker generation instead of using the process-wide one-shot path."""

    def __init__(self, parallel_config: Any, local_rank: int, lazy_init: bool = False):
        self.local_rank = local_rank
        self._mp_registered_ptrs: list[int] = []
        super().__init__(parallel_config, lazy_init=lazy_init)

    def set_device(self) -> None:
        torch.npu.set_device(self.local_rank)

    def register_buffer(self, ptrs: list[int], lengths: list[int]) -> None:
        if self._use_fabric_mem:
            return
        if self._mp_registered_ptrs:
            raise RuntimeError("Mooncake buffers are already registered for this Worker")

        self.set_device()
        transfer_engine = global_te.get_transfer_engine(get_ip(), device_name=None)
        registered: list[int] = []
        with global_te.register_buffer_lock:
            try:
                for ptr, length in zip(ptrs, lengths):
                    result = transfer_engine.register_memory(ptr, length)
                    if result != 0:
                        raise RuntimeError(f"Mooncake memory registration failed with code {result}")
                    registered.append(ptr)
            except BaseException:
                for ptr in reversed(registered):
                    transfer_engine.unregister_memory(ptr)
                raise
        self._mp_registered_ptrs = registered

    def unregister_buffer(self) -> None:
        if self._use_fabric_mem or not self._mp_registered_ptrs:
            return

        transfer_engine = global_te.get_transfer_engine(get_ip(), device_name=None)
        failed: list[tuple[int, object]] = []
        released: set[int] = set()
        with global_te.register_buffer_lock:
            for ptr in reversed(self._mp_registered_ptrs):
                result = transfer_engine.unregister_memory(ptr)
                if result != 0:
                    failed.append((ptr, result))
                else:
                    released.add(ptr)
        self._mp_registered_ptrs = [ptr for ptr in self._mp_registered_ptrs if ptr not in released]
        if failed:
            raise RuntimeError(f"Mooncake memory unregistration failed: {failed!r}")

    def close(self) -> None:
        try:
            self.unregister_buffer()
        finally:
            close = getattr(self.store, "close", None)
            if callable(close):
                close()
