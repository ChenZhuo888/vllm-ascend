from unittest.mock import call, patch

import cloudpickle
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache import (
    KVCacheClient,
    KVCacheMethod,
    ServiceSessionExpiredError,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc import (
    MPRemoteError,
    MPRequestTimeoutError,
)

KV_CACHE_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache"


def _make_vllm_config():
    from types import SimpleNamespace

    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(engine_id="engine-0"),
        parallel_config=SimpleNamespace(rank=0, data_parallel_rank=0),
    )


def _make_request():
    from types import SimpleNamespace

    return SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=list(range(16)),
        block_hashes=[bytes.fromhex("01" * 32)],
        num_tokens=16,
    )


def test_recovery_reuses_the_same_session_after_initial_registration_failure() -> None:
    with patch(f"{KV_CACHE_MODULE}.MPClient") as rpc_client_class:
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.is_server_responsive = False
        rpc_client.ping.return_value = "OK"
        rpc_client.request.side_effect = [MPRequestTimeoutError("timeout"), [b"OK"]]

        client = KVCacheClient("tcp://127.0.0.1:12345")
        try:
            assert not client.register_scheduler(_make_vllm_config(), None, 0)

            recovery_callback = rpc_client.start_heartbeat.call_args.kwargs["recovery_callback"]
            assert recovery_callback()

            register_calls = [
                request_call
                for request_call in rpc_client.request.call_args_list
                if request_call.args[0] == KVCacheMethod.REGISTER_SCHEDULER
            ]
            first_registration = cloudpickle.loads(register_calls[0].args[1][0])
            second_registration = cloudpickle.loads(register_calls[1].args[1][0])

            assert first_registration.session_id == second_registration.session_id
            assert first_registration.identity == second_registration.identity
        finally:
            client.close()


def test_stale_session_during_recovery_becomes_terminal_for_kv_cache_client() -> None:
    with patch(f"{KV_CACHE_MODULE}.MPClient") as rpc_client_class:
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.is_server_responsive = True
        rpc_client.ping.return_value = "OK"
        rpc_client.request.side_effect = [[b"OK"], MPRemoteError("StaleSessionError: old session")]

        client = KVCacheClient("tcp://127.0.0.1:12345")
        try:
            assert client.register_scheduler(_make_vllm_config(), None, 0)

            rpc_client.is_server_responsive = False
            recovery_callback = rpc_client.start_heartbeat.call_args.kwargs["recovery_callback"]
            assert recovery_callback()
            assert not client.is_registered

            with pytest.raises(ServiceSessionExpiredError, match="superseded"):
                client.lookup(_make_request(), 0)
        finally:
            client.close()


def test_lookup_maps_remote_stale_session_to_terminal_client_state() -> None:
    with patch(f"{KV_CACHE_MODULE}.MPClient") as rpc_client_class:
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.is_server_responsive = True
        rpc_client.ping.return_value = "OK"
        rpc_client.request.side_effect = [[b"OK"], MPRemoteError("StaleSessionError: old session")]

        client = KVCacheClient("tcp://127.0.0.1:12345")
        try:
            assert client.register_scheduler(_make_vllm_config(), None, 0)

            with pytest.raises(ServiceSessionExpiredError, match="StaleSessionError"):
                client.lookup(_make_request(), 0)

            assert not client.is_registered
            with pytest.raises(ServiceSessionExpiredError, match="superseded"):
                client.lookup(_make_request(), 0)
        finally:
            client.close()


def test_close_stops_heartbeat_then_unregisters_the_same_session() -> None:
    with patch(f"{KV_CACHE_MODULE}.MPClient") as rpc_client_class:
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.is_server_responsive = True
        rpc_client.ping.return_value = "OK"
        rpc_client.request.return_value = [b"OK"]

        client = KVCacheClient("tcp://127.0.0.1:12345")
        assert client.register_scheduler(_make_vllm_config(), None, 0)

        register_call = rpc_client.request.call_args_list[0]
        registration = cloudpickle.loads(register_call.args[1][0])
        client.close()

        unregister_call = next(
            request_call
            for request_call in rpc_client.request.call_args_list
            if request_call.args[0] == KVCacheMethod.UNREGISTER_SCHEDULER
        )
        assert unregister_call.args[1][-1].decode() == registration.session_id

        stop_index = rpc_client.method_calls.index(call.stop_heartbeat())
        unregister_index = next(
            index
            for index, method_call in enumerate(rpc_client.method_calls)
            if method_call == call.request(
                KVCacheMethod.UNREGISTER_SCHEDULER,
                unregister_call.args[1],
                timeout_ms=500,
            )
        )
        assert stop_index < unregister_index
        rpc_client.close.assert_called_once_with()
