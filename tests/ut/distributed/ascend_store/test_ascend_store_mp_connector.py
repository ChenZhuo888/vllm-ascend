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

CONNECTOR_MODULE = (
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store."
    "ascend_store_mp_connector"
)
SERVER_URL = "ipc:///tmp/ascend_store_mp_test"


def _make_vllm_config(server_url: object = SERVER_URL) -> MagicMock:
    config = MagicMock()
    config.kv_transfer_config.kv_connector = "AscendStoreMPConnector"
    config.kv_transfer_config.kv_connector_extra_config = {}

    if server_url is not None:
        config.kv_transfer_config.kv_connector_extra_config["kv_cache_server_url"] = server_url

    return config


@pytest.mark.parametrize(
    "role",
    [
        KVConnectorRole.SCHEDULER,
        KVConnectorRole.WORKER,
    ],
)
def test_connector_creates_client_for_each_role(role) -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        connector = AscendStoreMPConnector(
            vllm_config=config,
            role=role,
            kv_cache_config=MagicMock(),
        )
        client_class.assert_called_once_with(SERVER_URL)

        connector.shutdown()
        client_class.return_value.close.assert_called_once_with()


def test_scheduler_lookup_delegates_to_kv_cache_client() -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.lookup.return_value = 0
        connector = AscendStoreMPConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=MagicMock(),
        )

        with patch.object(
                AscendStoreMPConnector,
                "role",
                KVConnectorRole.SCHEDULER,
                create=True,
        ):
            result = connector.get_num_new_matched_tokens(
                request=MagicMock(),
                num_computed_tokens=32,
            )

        assert result == (0, False)
        client_class.return_value.lookup.assert_called_once_with(32)


def test_worker_cannot_call_scheduler_lookup() -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        connector = AscendStoreMPConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=MagicMock(),
        )

        with patch.object(
                AscendStoreMPConnector,
                "role",
                KVConnectorRole.WORKER,
                create=True,
        ):
            with pytest.raises(
                    RuntimeError,
                    match="only available on the scheduler connector",
            ):
                connector.get_num_new_matched_tokens(
                    request=MagicMock(),
                    num_computed_tokens=32,
                )

        client_class.return_value.lookup.assert_not_called()


def test_build_connector_meta_returns_empty_metadata() -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient"):
        connector = AscendStoreMPConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=MagicMock(),
        )

        metadata = connector.build_connector_meta(MagicMock())

    assert isinstance(metadata, AscendStoreMPConnectorMetadata)


@pytest.mark.parametrize("server_url", [None, "", 123])
def test_connector_rejects_invalid_server_url(server_url: object) -> None:
    config = _make_vllm_config(server_url)

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        with pytest.raises(ValueError, match="kv_cache_server_url"):
            AscendStoreMPConnector(
                vllm_config=config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=MagicMock(),
            )

        client_class.assert_not_called()
