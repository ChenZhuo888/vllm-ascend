"""KV cache service identities and registration models."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheConfig

from .config import KVPoolConfigSpec

if TYPE_CHECKING:
    from ..pool_scheduler import KVPoolScheduler
    from ..pool_worker import KVPoolWorker

_LEGACY_SESSION_ID = "legacy"

WorkerLookupHandler = Callable[["SchedulerIdentity", int, Sequence[BlockHash], list[int] | None, bool, int], int]
SchedulerFactory = Callable[["SchedulerRegistration", WorkerLookupHandler], "KVPoolScheduler"]
WorkerFactory = Callable[["WorkerRegistration"], "KVPoolWorker"]


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


def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str):
        raise TypeError(f"session_id must be a string, got {type(session_id).__name__}")
    if not session_id:
        raise ValueError("session_id must not be empty")


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

    @classmethod
    def from_config_spec(cls, config: KVPoolConfigSpec) -> "SchedulerIdentity":
        return cls(
            engine_id=config.kv_transfer_config.engine_id,
            data_parallel_rank=config.parallel_config.data_parallel_rank,
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

    @classmethod
    def from_config_spec(cls, config: KVPoolConfigSpec) -> "WorkerIdentity":
        return cls(
            engine_id=config.kv_transfer_config.engine_id,
            rank=config.parallel_config.rank,
            data_parallel_rank=config.parallel_config.data_parallel_rank,
        )


@dataclass(frozen=True)
class SchedulerRegistration:
    identity: SchedulerIdentity
    config: KVPoolConfigSpec
    page_size_bytes: int
    session_id: str = _LEGACY_SESSION_ID

    def __post_init__(self) -> None:
        _validate_session_id(self.session_id)

    @classmethod
    def create(
        cls,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None,
        page_size_bytes: int,
        session_id: str = _LEGACY_SESSION_ID,
    ) -> "SchedulerRegistration":
        _validate_rank(page_size_bytes, "page_size_bytes")
        return cls(
            identity=SchedulerIdentity.from_vllm_config(vllm_config),
            config=KVPoolConfigSpec.from_vllm_config(vllm_config, kv_cache_config),
            page_size_bytes=page_size_bytes,
            session_id=session_id,
        )


@dataclass(frozen=True)
class WorkerRegistration:
    identity: WorkerIdentity
    config: KVPoolConfigSpec
    session_id: str = _LEGACY_SESSION_ID

    def __post_init__(self) -> None:
        _validate_session_id(self.session_id)

    @classmethod
    def create(
        cls,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None,
        session_id: str = _LEGACY_SESSION_ID,
    ) -> "WorkerRegistration":
        return cls(
            identity=WorkerIdentity.from_vllm_config(vllm_config),
            config=KVPoolConfigSpec.from_vllm_config(vllm_config, kv_cache_config),
            session_id=session_id,
        )
