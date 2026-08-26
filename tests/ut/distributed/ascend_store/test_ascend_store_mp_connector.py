from unittest.mock import MagicMock, patch

import pytest
import torch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_mp_connector import (
    AscendStoreMPConnector,
    AscendStoreMPConnectorMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.memory import (
    export_worker_kv_caches,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.view import KVCacheStorageSpec

# isort: on

CONNECTOR_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_mp_connector"
SERVER_URL = "ipc:///tmp/ascend_store_mp_test"


class _CPUMemoryAdapter:
    @staticmethod
    def export_storage(storage: torch.Tensor) -> KVCacheStorageSpec:
        return KVCacheStorageSpec(
            size_bytes=storage.untyped_storage().nbytes(),
            device_type="cpu",
            device_uuid="cpu",
            handle_type="test_cpu",
            handle_version=1,
            handle=b"handle",
        )


def _make_vllm_config(server_url: object = SERVER_URL, rank: int = 0) -> MagicMock:
    config = MagicMock()
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.rank = rank
    config.kv_transfer_config.kv_connector = "AscendStoreMPConnector"
    config.kv_transfer_config.engine_id = "engine-0"
    config.kv_transfer_config.kv_connector_extra_config = {}
    if server_url is not None:
        config.kv_transfer_config.kv_connector_extra_config["kv_cache_server_url"] = server_url
    return config


def _make_kv_cache_config() -> MagicMock:
    config = MagicMock()
    config.kv_cache_groups[0].kv_cache_spec.page_size_bytes = 1024
    return config


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_connector_registers_its_role(role: KVConnectorRole) -> None:
    config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        connector = AscendStoreMPConnector(config, role, kv_cache_config)

        client_class.assert_called_once_with(SERVER_URL)
        if role == KVConnectorRole.SCHEDULER:
            client_class.return_value.register_scheduler.assert_called_once_with(config, kv_cache_config, 1024)
            client_class.return_value.register_worker.assert_not_called()
        else:
            client_class.return_value.register_worker.assert_called_once_with(config, kv_cache_config)
            client_class.return_value.register_scheduler.assert_not_called()

        connector.shutdown()
        client_class.return_value.close.assert_called_once_with()


def test_scheduler_lookup_delegates_to_kv_cache_client() -> None:
    config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    request = MagicMock()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.lookup.return_value = (16, False)
        connector = AscendStoreMPConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)

        result = connector.get_num_new_matched_tokens(request, 32)

        assert result == (16, False)
        client_class.return_value.lookup.assert_called_once_with(request, 32)


def test_worker_cannot_call_scheduler_lookup() -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())

        with pytest.raises(RuntimeError, match="only available on the scheduler connector"):
            connector.get_num_new_matched_tokens(MagicMock(), 32)

        client_class.return_value.lookup.assert_not_called()


def test_worker_registers_process_neutral_kv_cache_layouts() -> None:
    config = _make_vllm_config()
    storage = torch.empty((4, 8), dtype=torch.float16)

    def export(caches, generation):
        return export_worker_kv_caches(caches, generation, _CPUMemoryAdapter())

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.export_worker_kv_caches", side_effect=export),
    ):
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.register_kv_caches({"layer.0": storage, "layer.1": storage[1:]})

    spec = client_class.return_value.register_kv_caches.call_args.args[0]
    first = spec.caches["layer.0"][0]
    second = spec.caches["layer.1"][0]
    assert first.storage_index == second.storage_index == 0
    assert first.storage_offset_bytes == 0
    assert second.storage_offset_bytes == storage.stride(0) * storage.element_size()
    assert first.shape == (4, 8)
    assert first.stride == storage.stride()
    assert first.dtype == "torch.float16"
    assert spec.generation == 1
    assert len(spec.storages) == 1
    assert spec.storages[0].size_bytes == storage.untyped_storage().nbytes()
    assert spec.storages[0].device_type == "cpu"


def test_worker_rejects_empty_kv_caches_before_rpc() -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        with pytest.raises(ValueError, match="must not be empty"):
            connector.register_kv_caches({})

        client_class.return_value.register_kv_caches.assert_not_called()


def test_worker_keeps_exported_cache_alive_until_shutdown() -> None:
    config = _make_vllm_config()
    exported = MagicMock()
    exported.spec.generation = 1

    def register(spec, on_registered):
        on_registered(spec)
        return True

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.export_worker_kv_caches", return_value=exported),
    ):
        client_class.return_value.register_kv_caches.side_effect = register
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.register_kv_caches({"layer.0": MagicMock()})

        exported.close.assert_not_called()
        connector.shutdown()

    client_class.return_value.register_kv_caches.assert_called_once_with(
        exported.spec,
        on_registered=connector._confirm_kv_cache_export,
    )
    exported.close.assert_called_once_with()


def test_worker_releases_exported_cache_when_registration_fails() -> None:
    config = _make_vllm_config()
    exported = MagicMock()

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.export_worker_kv_caches", return_value=exported),
    ):
        client_class.return_value.register_kv_caches.side_effect = RuntimeError("registration failed")
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())

        with pytest.raises(RuntimeError, match="registration failed"):
            connector.register_kv_caches({"layer.0": MagicMock()})

    exported.close.assert_called_once_with()


def test_worker_keeps_active_export_when_replacement_fails() -> None:
    config = _make_vllm_config()
    exports = []

    def export(_caches, generation):
        cache_export = MagicMock()
        cache_export.spec.generation = generation
        exports.append(cache_export)
        return cache_export

    def register(spec, on_registered):
        if spec.generation == 2:
            raise RuntimeError("registration failed")
        on_registered(spec)
        return True

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.export_worker_kv_caches", side_effect=export),
    ):
        client_class.return_value.register_kv_caches.side_effect = register
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.register_kv_caches({"layer.0": MagicMock()})

        with pytest.raises(RuntimeError, match="registration failed"):
            connector.register_kv_caches({"layer.0": MagicMock()})

        exports[0].close.assert_not_called()
        exports[1].close.assert_called_once_with()
        connector.shutdown()

    exports[0].close.assert_called_once_with()


def test_worker_keeps_active_export_while_replacement_is_unconfirmed() -> None:
    config = _make_vllm_config()
    exports = []
    callbacks = []

    def export(_caches, generation):
        cache_export = MagicMock()
        cache_export.spec.generation = generation
        exports.append(cache_export)
        return cache_export

    def register(spec, on_registered):
        callbacks.append(on_registered)
        if spec.generation == 1:
            on_registered(spec)
            return True
        return False

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.export_worker_kv_caches", side_effect=export),
    ):
        client_class.return_value.register_kv_caches.side_effect = register
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.register_kv_caches({"layer.0": MagicMock()})
        connector.register_kv_caches({"layer.0": MagicMock()})

        exports[0].close.assert_not_called()
        exports[1].close.assert_not_called()

        callbacks[1](exports[1].spec)

        exports[0].close.assert_called_once_with()
        exports[1].close.assert_not_called()
        connector.shutdown()

    exports[1].close.assert_called_once_with()


def test_connector_rejects_sleep_mode() -> None:
    config = _make_vllm_config()
    config.model_config.enable_sleep_mode = True

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        with pytest.raises(ValueError, match="does not support sleep mode"):
            AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())

        client_class.assert_not_called()


def test_build_connector_meta_returns_empty_metadata_when_degraded() -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.build_connector_meta.return_value = None
        connector = AscendStoreMPConnector(config, KVConnectorRole.SCHEDULER, _make_kv_cache_config())
        metadata = connector.build_connector_meta(MagicMock())

    assert isinstance(metadata, AscendStoreMPConnectorMetadata)


def test_connector_closes_client_when_registration_fails() -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.register_scheduler.side_effect = RuntimeError("registration failed")

        with pytest.raises(RuntimeError, match="registration failed"):
            AscendStoreMPConnector(config, KVConnectorRole.SCHEDULER, _make_kv_cache_config())

        client_class.return_value.close.assert_called_once_with()


@pytest.mark.parametrize("server_url", [None, "", 123])
def test_connector_rejects_invalid_server_url(server_url: object) -> None:
    config = _make_vllm_config(server_url)

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        with pytest.raises(ValueError, match="kv_cache_server_url"):
            AscendStoreMPConnector(config, KVConnectorRole.SCHEDULER, _make_kv_cache_config())

        client_class.assert_not_called()
