from unittest.mock import MagicMock, patch

import pytest

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_mp_connector import (
    AscendStoreMPConnector,
    AscendStoreMPConnectorMetadata,
)

# isort: on

CONNECTOR_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_mp_connector"
SERVER_URL = "ipc:///tmp/ascend_store_mp_test"


def _make_vllm_config(server_url: object = SERVER_URL) -> MagicMock:
    config = MagicMock()
    config.kv_transfer_config.kv_connector = "AscendStoreMPConnector"
    config.kv_transfer_config.kv_connector_extra_config = {}
    if server_url is not None:
        config.kv_transfer_config.kv_connector_extra_config["kv_cache_server_url"] = server_url
    return config


def _make_kv_cache_config() -> MagicMock:
    config = MagicMock()
    config.kv_cache_groups[0].kv_cache_spec.page_size_bytes = 1024
    return config


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_connector_creates_client_for_each_role(role) -> None:
    config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.KVPoolScheduler") as scheduler_class,
    ):
        connector = AscendStoreMPConnector(config, role, kv_cache_config)
        client_class.assert_called_once_with(SERVER_URL)

        if role == KVConnectorRole.SCHEDULER:
            scheduler_class.assert_called_once_with(
                config, use_layerwise=False, kv_cache_config=kv_cache_config, page_size_bytes=1024
            )
            assert scheduler_class.return_value.client is client_class.return_value
        else:
            scheduler_class.assert_not_called()

        connector.shutdown()
        client_class.return_value.close.assert_called_once_with()


def test_scheduler_lookup_delegates_to_pool_scheduler() -> None:
    config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    request = MagicMock()

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.KVPoolScheduler") as scheduler_class,
    ):
        scheduler = scheduler_class.return_value
        scheduler.get_num_new_matched_tokens.return_value = (16, False)
        connector = AscendStoreMPConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)

        with patch.object(AscendStoreMPConnector, "role", KVConnectorRole.SCHEDULER, create=True):
            result = connector.get_num_new_matched_tokens(request, 32)

        assert result == (16, False)
        assert scheduler.client is client_class.return_value
        scheduler.get_num_new_matched_tokens.assert_called_once_with(request, 32)


def test_worker_cannot_call_scheduler_lookup() -> None:
    config = _make_vllm_config()

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.KVPoolScheduler") as scheduler_class,
    ):
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())

        with patch.object(AscendStoreMPConnector, "role", KVConnectorRole.WORKER, create=True):
            with pytest.raises(RuntimeError, match="only available on the scheduler connector"):
                connector.get_num_new_matched_tokens(MagicMock(), 32)

        client_class.return_value.lookup.assert_not_called()
        scheduler_class.assert_not_called()


def test_build_connector_meta_returns_empty_metadata() -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient"), patch(f"{CONNECTOR_MODULE}.KVPoolScheduler"):
        connector = AscendStoreMPConnector(config, KVConnectorRole.SCHEDULER, _make_kv_cache_config())
        metadata = connector.build_connector_meta(MagicMock())

    assert isinstance(metadata, AscendStoreMPConnectorMetadata)


@pytest.mark.parametrize("server_url", [None, "", 123])
def test_connector_rejects_invalid_server_url(server_url: object) -> None:
    config = _make_vllm_config(server_url)

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.KVPoolScheduler") as scheduler_class,
    ):
        with pytest.raises(ValueError, match="kv_cache_server_url"):
            AscendStoreMPConnector(config, KVConnectorRole.SCHEDULER, _make_kv_cache_config())

        client_class.assert_not_called()
        scheduler_class.assert_not_called()
