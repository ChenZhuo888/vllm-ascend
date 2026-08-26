from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.distributed.kv_events import BlockStored

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import KVCacheClient
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.client import _RegistrationState
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.error import (
    SERVICE_NOT_REGISTERED_PREFIX,
    STALE_SESSION_PREFIX,
    ServiceSessionExpiredError,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.registration import (
    SchedulerRegistration,
    WorkerRegistration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.protocol import (
    encode_build_connector_worker_meta_response,
    encode_get_finished_response,
    encode_get_kv_events_response,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.synchronization import NPUEventSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.view import WorkerKVCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc import (
    MPRemoteError,
    MPRequestTimeoutError,
    MPServerBusyError,
    MPServerUnavailableError,
)

# isort: on

CLIENT_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.client"


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(engine_id="engine-0"),
        parallel_config=SimpleNamespace(rank=0, data_parallel_rank=0),
    )


def _configure_mock_client(client_class, request_side_effect) -> KVCacheClient:
    """Build a registered client whose RPC layer raises request_side_effect
    instead of talking to a server. Call inside a `with patch(...)` block."""
    rpc = client_class.return_value
    rpc.is_transport_connected = True
    rpc.request.side_effect = request_side_effect
    client = KVCacheClient("ipc:///tmp/ascend-store-test")
    registration = SchedulerRegistration.create(_make_config(), None, 0, session_id="sess")
    client._registration = (registration, (b"engine-0", b"0", b"payload"))
    client._registration_state = _RegistrationState.REGISTERED
    return client


def _configure_mock_worker_client(client_class, request_side_effect) -> KVCacheClient:
    rpc = client_class.return_value
    rpc.is_transport_connected = True
    rpc.request.side_effect = request_side_effect
    client = KVCacheClient("ipc:///tmp/ascend-store-test")
    registration = WorkerRegistration.create(_make_config(), None, session_id="sess")
    client._registration = (registration, (b"engine-0", b"0", b"0", b"sess", b"payload"))
    client._registration_state = _RegistrationState.REGISTERED
    return client


REQUEST = SimpleNamespace(request_id="r1", prompt_token_ids=[1], block_hashes=[b"h"], num_tokens=1)
BLOCKS = SimpleNamespace(get_block_ids=lambda: ([7],))
SCHEDULER_OUTPUT = SimpleNamespace(
    finished_req_ids=set(),
    preempted_req_ids=set(),
    num_scheduled_tokens={},
    scheduled_new_reqs=[],
    scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], num_computed_tokens=[]),
)
WORKER_KV_CACHE_SPEC = WorkerKVCacheSpec(generation=1, caches={"layer.0": ()}, storages=())


def test_worker_cache_registration_uses_worker_rpc() -> None:
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_worker_client(client_class, [[b"OK"]])
        confirmed = []

        assert client.register_kv_caches(WORKER_KV_CACHE_SPEC, on_registered=confirmed.append)
        request = client_class.return_value.request
        assert request.call_args.args[0].value == "REGISTER_KV_CACHES"
        assert confirmed == [WORKER_KV_CACHE_SPEC]


def test_worker_cache_registration_marks_client_unregistered_when_busy() -> None:
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_worker_client(client_class, MPServerBusyError("busy"))
        confirmed = []

        assert not client.register_kv_caches(WORKER_KV_CACHE_SPEC, on_registered=confirmed.append)
        assert not client.is_registered
        assert confirmed == []


def test_worker_wait_for_save_has_no_default_deadline() -> None:
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_worker_client(client_class, [[b"OK"]])
        metadata = AscendConnectorMetadata(set(), set())
        event = NPUEventSpec("host-0", b"event-handle")

        assert client.wait_for_save(metadata, event)

        request = client_class.return_value.request
        assert request.call_args.args[0].value == "WAIT_FOR_SAVE"
        assert request.call_args.kwargs["timeout_ms"] is None


def test_worker_get_finished_uses_bounded_worker_rpc() -> None:
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_worker_client(
            client_class,
            [list(encode_get_finished_response({"saving-0"}, {"loading-0"}))],
        )
        metadata = AscendConnectorMetadata(
            set(),
            set(),
            loading_req_ids={"loading-0"},
            delayed_free_req_ids={"saving-0"},
        )

        assert client.get_finished({"saving-0", "loading-0"}, metadata) == (
            {"saving-0"},
            {"loading-0"},
        )

        request = client_class.return_value.request
        assert request.call_args.args[0].value == "GET_FINISHED"
        assert request.call_args.kwargs["timeout_ms"] == 5000


def test_worker_get_finished_degrades_to_empty_sets_when_busy() -> None:
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_worker_client(client_class, MPServerBusyError("busy"))

        assert client.get_finished(set(), AscendConnectorMetadata(set(), set())) == (set(), set())


def test_worker_build_connector_meta_uses_worker_rpc() -> None:
    metadata = AscendStoreKVConnectorWorkerMetadata({7: 1})
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_worker_client(
            client_class,
            [list(encode_build_connector_worker_meta_response(metadata))],
        )

        assert client.build_connector_worker_meta() == metadata

        request = client_class.return_value.request
        assert request.call_args.args[0].value == "BUILD_CONNECTOR_WORKER_META"


def test_worker_build_connector_meta_degrades_to_none_when_busy() -> None:
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_worker_client(client_class, MPServerBusyError("busy"))

        assert client.build_connector_worker_meta() is None


def test_worker_get_kv_events_uses_worker_rpc() -> None:
    event = BlockStored(
        block_hashes=[b"hash-0"],
        parent_block_hash=None,
        token_ids=[1],
        block_size=1,
        lora_id=None,
        medium="CPU",
        lora_name=None,
    )
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_worker_client(
            client_class,
            [list(encode_get_kv_events_response([event]))],
        )

        assert client.get_kv_events() == [event]

        request = client_class.return_value.request
        assert request.call_args.args[0].value == "GET_KV_EVENTS"


def test_worker_get_kv_events_degrades_to_empty_list_when_busy() -> None:
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_worker_client(client_class, MPServerBusyError("busy"))

        assert client.get_kv_events() == []


def test_update_state_after_alloc_degrades_silently_on_timeout() -> None:
    # Regression: this path used to fall through to response validation and
    # raise UnboundLocalError after marking the client unregistered.
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_client(client_class, MPRequestTimeoutError("timeout"))
        client.update_state_after_alloc(REQUEST, BLOCKS, 0)
        assert not client.is_registered


def test_all_business_methods_return_their_degraded_values_when_busy() -> None:
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_client(client_class, MPServerBusyError("busy"))

        assert client.lookup(REQUEST, 0) == (0, False)
        client.update_state_after_alloc(REQUEST, BLOCKS, 0)
        assert client.build_connector_meta(SCHEDULER_OUTPUT, {}) is None
        assert client.request_finished("r1", [7]) == (False, None)
        assert client.update_connector_output({7: 1}) == []


@pytest.mark.parametrize("error", [MPRequestTimeoutError("t"), MPServerUnavailableError("u")])
def test_transport_errors_mark_client_unregistered(error) -> None:
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_client(client_class, error)

        assert client.lookup(REQUEST, 0) == (0, False)
        assert not client.is_registered


def test_stale_session_still_raises_after_marking_superseded() -> None:
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_client(client_class, MPRemoteError(f"{STALE_SESSION_PREFIX} superseded"))

        with pytest.raises(ServiceSessionExpiredError):
            client.lookup(REQUEST, 0)


def test_other_remote_errors_propagate() -> None:
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_client(client_class, MPRemoteError("boom"))

        with pytest.raises(MPRemoteError, match="boom"):
            client.lookup(REQUEST, 0)


def test_not_registered_remote_error_degrades_and_marks_unregistered() -> None:
    with patch(f"{CLIENT_MODULE}.MPClient") as client_class:
        client = _configure_mock_client(
            client_class, MPRemoteError(f"{SERVICE_NOT_REGISTERED_PREFIX} Scheduler(...) is not registered")
        )

        assert client.lookup(REQUEST, 0) == (0, False)
        assert not client.is_registered
