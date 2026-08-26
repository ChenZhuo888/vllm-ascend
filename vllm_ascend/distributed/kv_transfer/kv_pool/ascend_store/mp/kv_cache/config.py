"""Pure-data configuration projected into KV cache service registrations."""

from __future__ import annotations

import enum
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from types import SimpleNamespace
from typing import Any

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheSpec

_DEFAULT_BLOCK_SIZE = 16
_DEFAULT_MODEL_LAYERS = 1
_DEFAULT_PARALLEL_SIZE = 1
_SUPPORTED_SPEC_MODULES = {
    "vllm.v1.kv_cache_interface",
    "vllm_ascend.core.kv_cache_interface",
}
_SUPPORTED_ENUM_MODULES = {
    "vllm.v1.attention.backends.registry",
    "vllm.v1.kv_cache_interface",
}

WireValue = (
    None | bool | int | float | str | bytes | list["WireValue"] | tuple["WireValue", ...] | dict[str, "WireValue"]
)


@dataclass(frozen=True)
class _DTypeSpec:
    name: str


@dataclass(frozen=True)
class _EnumSpec:
    module: str
    name: str
    value: WireValue


@dataclass(frozen=True)
class KVCacheSpecData:
    module: str
    name: str
    values: tuple[tuple[str, Any], ...]

    @classmethod
    def from_spec(cls, spec: KVCacheSpec) -> KVCacheSpecData:
        spec_type = type(spec)
        if spec_type.__module__ not in _SUPPORTED_SPEC_MODULES:
            raise TypeError(
                f"Unsupported KV cache spec type {spec_type.__module__}.{spec_type.__name__}; "
                "AscendStore MP registrations support vLLM and vLLM Ascend cache specs only"
            )
        if not is_dataclass(spec):
            raise TypeError(f"KV cache spec {spec_type.__name__} must be a dataclass")
        return cls(
            module=spec_type.__module__,
            name=spec_type.__name__,
            values=tuple((field.name, _project_value(getattr(spec, field.name))) for field in fields(spec)),
        )

    def build(self) -> KVCacheSpec:
        if self.module not in _SUPPORTED_SPEC_MODULES:
            raise TypeError(f"Unsupported KV cache spec module {self.module!r}")
        module = importlib.import_module(self.module)
        spec_type = getattr(module, self.name, None)
        if not isinstance(spec_type, type) or not issubclass(spec_type, KVCacheSpec):
            raise TypeError(f"Unsupported KV cache spec type {self.module}.{self.name}")
        return spec_type(**{name: _restore_value(value) for name, value in self.values})


@dataclass(frozen=True)
class KVCacheGroupData:
    layer_names: tuple[str, ...]
    kv_cache_spec: KVCacheSpecData
    is_eagle_group: bool

    @classmethod
    def from_group(cls, group: KVCacheGroupSpec) -> KVCacheGroupData:
        return cls(
            layer_names=tuple(group.layer_names),
            kv_cache_spec=KVCacheSpecData.from_spec(group.kv_cache_spec),
            is_eagle_group=bool(getattr(group, "is_eagle_group", False)),
        )

    def build(self) -> KVCacheGroupSpec:
        return KVCacheGroupSpec(
            layer_names=list(self.layer_names),
            kv_cache_spec=self.kv_cache_spec.build(),
            is_eagle_group=self.is_eagle_group,
        )


@dataclass(frozen=True)
class KVCacheConfigData:
    num_blocks: int
    groups: tuple[KVCacheGroupData, ...]

    @classmethod
    def from_config(cls, config: KVCacheConfig) -> KVCacheConfigData:
        return cls(
            num_blocks=_require_non_negative_int(config.num_blocks, "kv_cache_config.num_blocks"),
            groups=tuple(KVCacheGroupData.from_group(group) for group in config.kv_cache_groups),
        )

    def build(self) -> KVCacheConfig:
        # KVPoolScheduler and KVPoolWorker consume block/group metadata only.
        # Tensor allocation remains owned by the vLLM Worker process and is
        # registered separately through WorkerKVCacheSpec.
        return KVCacheConfig(
            num_blocks=self.num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[group.build() for group in self.groups],
        )


@dataclass(frozen=True)
class KVPoolModelConfigSpec:
    model: str
    max_model_len: int
    num_layers: int
    num_kv_heads: int
    num_hidden_layers: int
    use_mla: bool
    use_sparse: bool
    model_type: str | None
    compress_ratios: tuple[int, ...] | None

    @classmethod
    def from_vllm_config(cls, vllm_config: object) -> KVPoolModelConfigSpec:
        model_config = getattr(vllm_config, "model_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        hf_text_config = getattr(model_config, "hf_text_config", None)
        hf_config = getattr(model_config, "hf_config", None) or hf_text_config
        num_layers = _call_int(model_config, "get_num_layers", parallel_config, default=_DEFAULT_MODEL_LAYERS)
        compress_ratios = _optional_int_tuple(getattr(hf_text_config, "compress_ratios", None))
        if compress_ratios is None:
            compress_ratios = _optional_int_tuple(getattr(hf_config, "compress_ratios", None))
        return cls(
            model=_read_str(model_config, "model", ""),
            max_model_len=_read_int(model_config, "max_model_len", 0),
            num_layers=num_layers,
            num_kv_heads=_call_int(
                model_config,
                "get_total_num_kv_heads",
                default=_DEFAULT_PARALLEL_SIZE,
            ),
            num_hidden_layers=_read_int(hf_text_config, "num_hidden_layers", num_layers),
            use_mla=_read_bool(model_config, "use_mla", False),
            use_sparse=hf_text_config is not None and hasattr(hf_text_config, "index_topk"),
            model_type=_read_optional_str(hf_config, "model_type"),
            compress_ratios=compress_ratios,
        )

    @property
    def hf_text_config(self) -> SimpleNamespace:
        values: dict[str, object] = {
            "num_hidden_layers": self.num_hidden_layers,
        }
        if self.model_type is not None:
            values["model_type"] = self.model_type
        if self.compress_ratios is not None:
            values["compress_ratios"] = self.compress_ratios
        if self.use_sparse:
            values["index_topk"] = True
        return SimpleNamespace(**values)

    @property
    def hf_config(self) -> SimpleNamespace:
        return self.hf_text_config

    def get_num_layers(self, _parallel_config: object) -> int:
        return self.num_layers

    def get_total_num_kv_heads(self) -> int:
        return self.num_kv_heads


@dataclass(frozen=True)
class KVPoolParallelConfigSpec:
    rank: int
    world_size: int
    data_parallel_rank: int
    data_parallel_index: int
    data_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    prefill_context_parallel_size: int
    decode_context_parallel_size: int

    @classmethod
    def from_vllm_config(cls, vllm_config: object) -> KVPoolParallelConfigSpec:
        config = getattr(vllm_config, "parallel_config", None)
        tp_size = _read_positive_int(config, "tensor_parallel_size", _DEFAULT_PARALLEL_SIZE)
        pp_size = _read_positive_int(config, "pipeline_parallel_size", _DEFAULT_PARALLEL_SIZE)
        pcp_size = _read_positive_int(config, "prefill_context_parallel_size", _DEFAULT_PARALLEL_SIZE)
        return cls(
            rank=_read_int(config, "rank", 0),
            world_size=_read_positive_int(config, "world_size", tp_size * pp_size * pcp_size),
            data_parallel_rank=_read_int(config, "data_parallel_rank", 0),
            data_parallel_index=_read_int(config, "data_parallel_index", 0),
            data_parallel_size=_read_positive_int(config, "data_parallel_size", _DEFAULT_PARALLEL_SIZE),
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=_read_positive_int(
                config,
                "decode_context_parallel_size",
                _DEFAULT_PARALLEL_SIZE,
            ),
        )


@dataclass(frozen=True)
class KVPoolTransferConfigSpec:
    engine_id: str
    kv_role: str
    kv_connector: str | None
    kv_connector_extra_config: dict[str, Any]

    @classmethod
    def from_vllm_config(cls, vllm_config: object) -> KVPoolTransferConfigSpec:
        config = getattr(vllm_config, "kv_transfer_config", None)
        if config is None:
            raise ValueError("kv_transfer_config must be set")
        engine_id = _read_str(config, "engine_id", "")
        if not engine_id:
            raise ValueError("kv_transfer_config.engine_id must not be empty")
        extra_config = getattr(config, "kv_connector_extra_config", None)
        if not isinstance(extra_config, Mapping):
            extra_config = {}
        projected_extra_config = _project_extra_value(extra_config)
        assert isinstance(projected_extra_config, dict)
        return cls(
            engine_id=engine_id,
            kv_role=_read_str(config, "kv_role", "kv_both"),
            kv_connector=_read_optional_str(config, "kv_connector"),
            kv_connector_extra_config=projected_extra_config,
        )

    def get_from_extra_config(self, name: str, default: Any = None) -> Any:
        return self.kv_connector_extra_config.get(name, default)


@dataclass(frozen=True)
class KVPoolCacheConfigSpec:
    block_size: int
    prefix_match_unit: int | None

    @classmethod
    def from_vllm_config(cls, vllm_config: object) -> KVPoolCacheConfigSpec:
        config = getattr(vllm_config, "cache_config", None)
        return cls(
            block_size=_read_positive_int(config, "block_size", _DEFAULT_BLOCK_SIZE),
            prefix_match_unit=_read_optional_int(config, "prefix_match_unit"),
        )


@dataclass(frozen=True)
class KVPoolSchedulerConfigSpec:
    disable_hybrid_kv_cache_manager: bool

    @classmethod
    def from_vllm_config(cls, vllm_config: object) -> KVPoolSchedulerConfigSpec:
        config = getattr(vllm_config, "scheduler_config", None)
        return cls(
            disable_hybrid_kv_cache_manager=_read_bool(
                config,
                "disable_hybrid_kv_cache_manager",
                False,
            )
        )


@dataclass(frozen=True)
class KVPoolSpeculativeConfigSpec:
    num_speculative_tokens: int
    eagle_enabled: bool

    @classmethod
    def from_vllm_config(cls, vllm_config: object) -> KVPoolSpeculativeConfigSpec | None:
        config = getattr(vllm_config, "speculative_config", None)
        if config is None:
            return None
        use_eagle = getattr(config, "use_eagle", None)
        return cls(
            num_speculative_tokens=_read_int(config, "num_speculative_tokens", 0),
            eagle_enabled=use_eagle() is True if callable(use_eagle) else False,
        )

    def use_eagle(self) -> bool:
        return self.eagle_enabled


@dataclass(frozen=True)
class KVPoolConfigSpec:
    model_config: KVPoolModelConfigSpec
    parallel_config: KVPoolParallelConfigSpec
    kv_transfer_config: KVPoolTransferConfigSpec
    cache_config: KVPoolCacheConfigSpec
    scheduler_config: KVPoolSchedulerConfigSpec
    speculative_config: KVPoolSpeculativeConfigSpec | None
    kv_events_enabled: bool
    kv_cache_config: KVCacheConfigData | None

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: object,
        kv_cache_config: KVCacheConfig | None,
    ) -> KVPoolConfigSpec:
        kv_events_config = getattr(vllm_config, "kv_events_config", None)
        return cls(
            model_config=KVPoolModelConfigSpec.from_vllm_config(vllm_config),
            parallel_config=KVPoolParallelConfigSpec.from_vllm_config(vllm_config),
            kv_transfer_config=KVPoolTransferConfigSpec.from_vllm_config(vllm_config),
            cache_config=KVPoolCacheConfigSpec.from_vllm_config(vllm_config),
            scheduler_config=KVPoolSchedulerConfigSpec.from_vllm_config(vllm_config),
            speculative_config=KVPoolSpeculativeConfigSpec.from_vllm_config(vllm_config),
            kv_events_enabled=_read_bool(kv_events_config, "enable_kv_cache_events", False),
            kv_cache_config=KVCacheConfigData.from_config(kv_cache_config) if kv_cache_config is not None else None,
        )

    def build_runtime(self) -> tuple[SimpleNamespace, KVCacheConfig | None]:
        kv_events_config = SimpleNamespace(enable_kv_cache_events=True) if self.kv_events_enabled else None
        vllm_config = SimpleNamespace(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            kv_transfer_config=self.kv_transfer_config,
            cache_config=self.cache_config,
            scheduler_config=self.scheduler_config,
            speculative_config=self.speculative_config,
            kv_events_config=kv_events_config,
        )
        kv_cache_config = self.kv_cache_config.build() if self.kv_cache_config is not None else None
        return vllm_config, kv_cache_config


def _project_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, torch.dtype):
        return _DTypeSpec(str(value).removeprefix("torch."))
    if isinstance(value, KVCacheSpec):
        return KVCacheSpecData.from_spec(value)
    if isinstance(value, enum.Enum):
        enum_type = type(value)
        if enum_type.__module__ not in _SUPPORTED_ENUM_MODULES:
            raise TypeError(f"Unsupported configuration enum {enum_type.__module__}.{enum_type.__name__}")
        return _EnumSpec(enum_type.__module__, enum_type.__name__, _project_value(value.value))
    if isinstance(value, list):
        return [_project_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_project_value(item) for item in value)
    if isinstance(value, Mapping):
        projected = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Configuration mapping keys must be strings, got {type(key).__name__}")
            projected[key] = _project_value(item)
        return projected
    raise TypeError(f"Unsupported registration configuration value {type(value).__name__}")


def _project_extra_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, list):
        return [_project_extra_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_project_extra_value(item) for item in value)
    if isinstance(value, Mapping):
        projected = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Configuration mapping keys must be strings, got {type(key).__name__}")
            projected[key] = _project_extra_value(item)
        return projected
    raise TypeError(f"Unsupported registration configuration value {type(value).__name__}")


def _restore_value(value: Any) -> Any:
    if isinstance(value, _DTypeSpec):
        dtype = getattr(torch, value.name, None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Unsupported torch dtype {value.name!r}")
        return dtype
    if isinstance(value, _EnumSpec):
        if value.module not in _SUPPORTED_ENUM_MODULES:
            raise TypeError(f"Unsupported configuration enum module {value.module!r}")
        module = importlib.import_module(value.module)
        enum_type = getattr(module, value.name, None)
        if not isinstance(enum_type, type) or not issubclass(enum_type, enum.Enum):
            raise TypeError(f"Unsupported configuration enum {value.module}.{value.name}")
        return enum_type(_restore_value(value.value))
    if isinstance(value, KVCacheSpecData):
        return value.build()
    if isinstance(value, list):
        return [_restore_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _restore_value(item) for key, item in value.items()}
    return value


def _call_int(obj: object, name: str, *args: object, default: int) -> int:
    method = getattr(obj, name, None)
    if not callable(method):
        return default
    try:
        value = method(*args)
    except (AttributeError, TypeError, ValueError):
        return default
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _read_bool(obj: object, name: str, default: bool) -> bool:
    value = getattr(obj, name, default)
    return value if isinstance(value, bool) else default


def _read_int(obj: object, name: str, default: int) -> int:
    value = getattr(obj, name, default)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _read_positive_int(obj: object, name: str, default: int) -> int:
    value = _read_int(obj, name, default)
    return value if value > 0 else default


def _read_optional_int(obj: object, name: str) -> int | None:
    value = getattr(obj, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _read_str(obj: object, name: str, default: str) -> str:
    value = getattr(obj, name, default)
    return value if isinstance(value, str) else default


def _read_optional_str(obj: object, name: str) -> str | None:
    value = getattr(obj, name, None)
    return value if isinstance(value, str) else None


def _optional_int_tuple(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return None
    return tuple(value)


def _require_non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer, got {value!r}")
    return value
