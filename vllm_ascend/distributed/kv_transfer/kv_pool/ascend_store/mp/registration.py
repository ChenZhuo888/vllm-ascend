"""Registration models for AscendStore multiprocessing services.

Registration payloads use cloudpickle because VllmConfig and KVCacheConfig
contain framework-specific Python objects. The MP endpoint must therefore be
restricted to trusted processes.
"""

from dataclasses import dataclass
from typing import TypeVar

import cloudpickle
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

from .rpc import MPProtocolError

RegistrationT = TypeVar(
    "RegistrationT",
    bound="SchedulerRegistration | WorkerRegistration",
)


def _validate_engine_id(engine_id: str) -> None:
    if not isinstance(engine_id, str):
        raise TypeError(f"engine_id must be a string, got {type(engine_id).__name__}")
    if not engine_id:
        raise ValueError("engine_id must not be empty")


def _validate_rank(rank: int, field_name: str) -> None:
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise TypeError(f"{field_name} must be an integer, got {type(rank).__name__}")
    if rank < 0:
        raise ValueError(f"{field_name} must not be negative, got {rank}")


@dataclass(frozen=True)
class SchedulerIdentity:
    engine_id: str
    data_parallel_rank: int = 0

    def __post_init__(self) -> None:
        _validate_engine_id(self.engine_id)
        _validate_rank(self.data_parallel_rank, "data_parallel_rank")

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> "SchedulerIdentity":
        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set")
        return cls(
            engine_id=kv_transfer_config.engine_id,
            data_parallel_rank=vllm_config.parallel_config.data_parallel_rank,
        )


@dataclass(frozen=True)
class WorkerIdentity:
    engine_id: str
    rank: int
    data_parallel_rank: int = 0

    def __post_init__(self) -> None:
        _validate_engine_id(self.engine_id)
        _validate_rank(self.rank, "rank")
        _validate_rank(self.data_parallel_rank, "data_parallel_rank")

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> "WorkerIdentity":
        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set")
        return cls(
            engine_id=kv_transfer_config.engine_id,
            rank=vllm_config.parallel_config.rank,
            data_parallel_rank=vllm_config.parallel_config.data_parallel_rank,
        )


@dataclass(frozen=True)
class SchedulerRegistration:
    identity: SchedulerIdentity
    vllm_config: VllmConfig
    kv_cache_config: KVCacheConfig | None
    page_size_bytes: int

    @classmethod
    def create(
        cls, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None, page_size_bytes: int
    ) -> "SchedulerRegistration":
        _validate_rank(page_size_bytes, "page_size_bytes")
        return cls(
            identity=SchedulerIdentity.from_vllm_config(vllm_config),
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            page_size_bytes=page_size_bytes,
        )


@dataclass(frozen=True)
class WorkerRegistration:
    identity: WorkerIdentity
    vllm_config: VllmConfig
    kv_cache_config: KVCacheConfig | None

    @classmethod
    def create(cls, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None) -> "WorkerRegistration":
        return cls(
            identity=WorkerIdentity.from_vllm_config(vllm_config),
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
        )


def encode_registration(registration: SchedulerRegistration | WorkerRegistration) -> bytes:
    try:
        return cloudpickle.dumps(registration)
    except Exception as exc:
        raise MPProtocolError(f"Failed to encode {type(registration).__name__}") from exc


def decode_registration(payloads: tuple[bytes, ...], expected_type: type[RegistrationT]) -> RegistrationT:
    if len(payloads) != 1:
        raise MPProtocolError(f"{expected_type.__name__} expects 1 payload, got {len(payloads)}")

    try:
        registration = cloudpickle.loads(payloads[0])
    except Exception as exc:
        raise MPProtocolError(f"Failed to decode {expected_type.__name__}") from exc

    if not isinstance(registration, expected_type):
        raise MPProtocolError(f"Expected {expected_type.__name__}, got {type(registration).__name__}")
    return registration
