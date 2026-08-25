import contextlib
import multiprocessing
import threading
import time
import traceback
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.registration import WorkerRegistration
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.request_view import WorkerKVCacheSpec

_MESSAGE_TIMEOUT_S = 60.0
_PROCESS_EXIT_TIMEOUT_S = 10.0
_PRODUCER_RELEASE_TIMEOUT_S = 90.0
_CLIENT_CONNECT_TIMEOUT_S = 30.0
_SERVER_URL = "tcp://127.0.0.1:*"


class _ModelConfig:
    def __init__(self):
        self.model = "org/llama-ipc-test"
        self.max_model_len = 1024
        self.use_mla = False
        self.hf_text_config = SimpleNamespace()
        self.hf_config = self.hf_text_config

    @staticmethod
    def get_num_layers(_parallel_config) -> int:
        return 2

    @staticmethod
    def get_total_num_kv_heads() -> int:
        return 1


class _ObservedWorker:
    """Expose assertions from the real MPKVPoolWorker without adding test RPCs."""

    def __init__(self, registration: "WorkerRegistration", connection: Connection):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.mp_pool_worker import MPKVPoolWorker

        self._connection = connection
        self._worker = MPKVPoolWorker(
            registration.vllm_config,
            kv_cache_config=registration.kv_cache_config,
            rank=registration.identity.rank,
        )

    def configure_kv_caches(self, spec: "WorkerKVCacheSpec") -> None:
        previous_caches = getattr(self._worker, "kv_caches", None)
        self._worker.configure_kv_caches(spec)
        base = self._worker.kv_caches["base"][0]
        view = self._worker.kv_caches["view"][0]
        self._connection.send(
            (
                "configured",
                {
                    "generation": spec.generation,
                    "device_type": base.device.type,
                    "base_values": base.cpu().tolist(),
                    "view_values": view.cpu().tolist(),
                    "shared_storage": base.untyped_storage().data_ptr() == view.untyped_storage().data_ptr(),
                    "previous_released": None if previous_caches is None else previous_caches == {},
                },
            )
        )

    def close(self) -> None:
        generation = self._worker.kv_cache_spec.generation if self._worker.kv_cache_spec is not None else None
        current_caches = getattr(self._worker, "kv_caches", None)
        self._worker.close()
        self._connection.send(
            (
                "closed",
                {
                    "generation": generation,
                    "mapping_released": current_caches == {},
                    "worker_caches_empty": self._worker.kv_caches == {},
                },
            )
        )


def _make_worker_config():
    return SimpleNamespace(
        model_config=_ModelConfig(),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(
            engine_id="ascend-store-mp-ipc-test",
            kv_role="kv_producer",
            kv_connector_extra_config={"backend": "mooncake"},
        ),
        cache_config=SimpleNamespace(block_size=16, prefix_match_unit=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        kv_events_config=None,
    )


def _create_observed_worker(
    registration: "WorkerRegistration",
    observation_connection: Connection,
) -> _ObservedWorker:
    return _ObservedWorker(registration, observation_connection)


def _producer(connection: Connection) -> None:
    exported = None
    try:
        import torch
        import torch_npu  # noqa: F401

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache_memory import (
            export_worker_kv_caches,
        )

        if not torch.npu.is_available():
            raise RuntimeError("NPU is not available in the producer process")

        torch.npu.set_device(0)
        base = torch.arange(64, dtype=torch.float16, device="npu").reshape(8, 8)
        view = base[1:, ::2]
        torch.npu.synchronize()

        exported = export_worker_kv_caches({"base": base, "view": view}, generation=1)
        connection.send(("ready", exported.spec))

        if not connection.poll(_PRODUCER_RELEASE_TIMEOUT_S):
            raise TimeoutError("Timed out waiting for the consumer to release the IPC mapping")
        if connection.recv() != "release":
            raise RuntimeError("Producer received an unexpected control message")
    except BaseException:
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            connection.send(("error", traceback.format_exc()))
        raise
    finally:
        if exported is not None:
            exported.close()
        connection.close()


def _consumer(connection: Connection, spec) -> None:
    imported = None
    try:
        import torch
        import torch_npu  # noqa: F401

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache_memory import (
            import_worker_kv_caches,
        )

        if not torch.npu.is_available():
            raise RuntimeError("NPU is not available in the consumer process")

        imported = import_worker_kv_caches(spec)
        base = imported.tensors["base"][0]
        view = imported.tensors["view"][0]
        expected = torch.arange(64, dtype=torch.float16).reshape(8, 8)

        if base.device.type != "npu" or view.device.type != "npu":
            raise AssertionError("Imported KV cache tensors are not on NPU")
        if base.shape != (8, 8) or base.stride() != (8, 1):
            raise AssertionError(f"Unexpected base layout: shape={base.shape}, stride={base.stride()}")
        if view.shape != (7, 4) or view.stride() != (8, 2):
            raise AssertionError(f"Unexpected view layout: shape={view.shape}, stride={view.stride()}")
        if base.untyped_storage().data_ptr() != view.untyped_storage().data_ptr():
            raise AssertionError("Imported tensor views do not share one storage")
        if not torch.equal(base.cpu(), expected):
            raise AssertionError("Imported base tensor has incorrect values")
        if not torch.equal(view.cpu(), expected[1:, ::2]):
            raise AssertionError("Imported tensor view has incorrect values")

        connection.send(("ok", None))
    except BaseException:
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            connection.send(("error", traceback.format_exc()))
        raise
    finally:
        if imported is not None:
            imported.close()
        connection.close()


def _wait_until_connected(client) -> None:
    deadline = time.monotonic() + _CLIENT_CONNECT_TIMEOUT_S
    while not client.is_connected:
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the KV cache RPC server")
        time.sleep(0.05)


def _wait_until_registered(client) -> None:
    deadline = time.monotonic() + _MESSAGE_TIMEOUT_S
    while not client.is_registered:
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for Worker registration")
        time.sleep(0.05)


def _request_server_stop(server, connection: Connection) -> None:
    try:
        if connection.recv() == "stop":
            server.request_stop()
        else:
            server.abort()
    except (EOFError, OSError):
        server.abort()
    finally:
        connection.close()


def _run_kv_cache_server(
    endpoint_connection: Connection,
    observation_connection: Connection,
    control_connection: Connection,
) -> None:
    server = None
    control_thread = None
    try:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import KVCacheServer

        server = KVCacheServer(
            _SERVER_URL,
            max_workers=2,
            worker_factory=partial(
                _create_observed_worker,
                observation_connection=observation_connection,
            ),
        )
        control_thread = threading.Thread(
            target=_request_server_stop,
            args=(server, control_connection),
            daemon=True,
            name="ascend-store-mp-ipc-stop",
        )
        control_thread.start()
        endpoint_connection.send(("ready", server.endpoint))
        endpoint_connection.close()
        server.run()
    except BaseException:
        error = traceback.format_exc()
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            endpoint_connection.send(("error", error))
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            observation_connection.send(("server_error", error))
        raise
    finally:
        if server is not None and not server.close():
            server.abort()
        if control_thread is not None:
            control_thread.join(_PROCESS_EXIT_TIMEOUT_S)
        endpoint_connection.close()
        observation_connection.close()
        control_connection.close()


def _receive(connection: Connection, process_name: str):
    if not connection.poll(_MESSAGE_TIMEOUT_S):
        raise TimeoutError(f"Timed out waiting for the {process_name} process")
    try:
        return connection.recv()
    except EOFError as exc:
        raise RuntimeError(f"The {process_name} process exited without a result") from exc


def _stop_process(process: BaseProcess | None) -> tuple[int | None, bool]:
    if process is None:
        return None, False

    process.join(_PROCESS_EXIT_TIMEOUT_S)
    forced = process.is_alive()
    if forced:
        process.terminate()
        process.join(_PROCESS_EXIT_TIMEOUT_S)
    return process.exitcode, forced


def test_npu_kv_cache_storage_round_trip_across_processes() -> None:
    context = multiprocessing.get_context("spawn")
    producer_connection, producer_child_connection = context.Pipe()
    producer = context.Process(target=_producer, args=(producer_child_connection,), name="kv-cache-ipc-producer")
    consumer = None
    consumer_connection = None
    failure: BaseException | None = None
    producer_exitcode = None
    producer_forced = False
    consumer_exitcode = None
    consumer_forced = False

    producer.start()
    producer_child_connection.close()
    try:
        producer_status, producer_result = _receive(producer_connection, "producer")
        if producer_status != "ready":
            raise RuntimeError(f"NPU IPC export failed:\n{producer_result}")

        consumer_connection, consumer_child_connection = context.Pipe()
        consumer = context.Process(
            target=_consumer,
            args=(consumer_child_connection, producer_result),
            name="kv-cache-ipc-consumer",
        )
        consumer.start()
        consumer_child_connection.close()

        consumer_status, consumer_result = _receive(consumer_connection, "consumer")
        if consumer_status != "ok":
            raise RuntimeError(f"NPU IPC import failed:\n{consumer_result}")
    except BaseException as exc:
        failure = exc
    finally:
        if consumer_connection is not None:
            consumer_connection.close()
        consumer_exitcode, consumer_forced = _stop_process(consumer)
        # The producer owns the source allocation until the consumer has
        # released its imported mapping or has been terminated.
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            producer_connection.send("release")
        producer_connection.close()
        producer_exitcode, producer_forced = _stop_process(producer)

    if failure is not None:
        raise failure
    if consumer_forced or producer_forced:
        pytest.fail("NPU IPC child process did not exit after releasing its cache mapping")
    assert consumer_exitcode == 0
    assert producer_exitcode == 0


def test_worker_cache_registration_replaces_generation_and_releases_mapping() -> None:
    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import KVCacheClient
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache_memory import (
        export_worker_kv_caches,
    )

    context = multiprocessing.get_context("spawn")
    endpoint_connection, endpoint_child_connection = context.Pipe()
    observation_connection, observation_child_connection = context.Pipe()
    control_connection, control_child_connection = context.Pipe()
    server = context.Process(
        target=_run_kv_cache_server,
        args=(endpoint_child_connection, observation_child_connection, control_child_connection),
        name="kv-cache-ipc-server",
    )
    client = None
    exported_caches = []
    failure: BaseException | None = None
    server_exitcode = None
    server_forced = False

    server.start()
    endpoint_child_connection.close()
    observation_child_connection.close()
    control_child_connection.close()
    try:
        server_status, server_result = _receive(endpoint_connection, "KV cache server")
        if server_status != "ready":
            raise RuntimeError(f"KV cache server failed to start:\n{server_result}")

        if not torch.npu.is_available():
            raise RuntimeError("NPU is not available in the Worker process")
        torch.npu.set_device(0)
        client = KVCacheClient(server_result)
        _wait_until_connected(client)
        client.register_worker(_make_worker_config(), kv_cache_config=None)
        _wait_until_registered(client)

        for generation in (1, 2):
            base = torch.full((4, 4), generation, dtype=torch.float16, device="npu")
            view = base[1:, ::2]
            torch.npu.synchronize()
            exported = export_worker_kv_caches({"base": base, "view": view}, generation)
            exported_caches.append(exported)
            if not client.register_kv_caches(exported.spec, timeout_ms=int(_MESSAGE_TIMEOUT_S * 1000)):
                raise RuntimeError(f"KV cache generation {generation} was not registered")
            if len(exported_caches) > 1:
                exported_caches.pop(0).close()

        client.close()
        client = None
        exported_caches.pop().close()

        first_status, first_generation = _receive(observation_connection, "first cache generation")
        second_status, second_generation = _receive(observation_connection, "second cache generation")
        close_status, close_result = _receive(observation_connection, "Worker cache release")

        assert first_status == "configured"
        assert first_generation == {
            "generation": 1,
            "device_type": "npu",
            "base_values": [[1.0] * 4 for _ in range(4)],
            "view_values": [[1.0] * 2 for _ in range(3)],
            "shared_storage": True,
            "previous_released": None,
        }
        assert second_status == "configured"
        assert second_generation == {
            "generation": 2,
            "device_type": "npu",
            "base_values": [[2.0] * 4 for _ in range(4)],
            "view_values": [[2.0] * 2 for _ in range(3)],
            "shared_storage": True,
            "previous_released": True,
        }
        assert close_status == "closed"
        assert close_result == {
            "generation": 2,
            "mapping_released": True,
            "worker_caches_empty": True,
        }
    except BaseException as exc:
        failure = exc
    finally:
        endpoint_connection.close()
        if client is not None:
            client.close()
        for exported in exported_caches:
            exported.close()
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            control_connection.send("stop")
        control_connection.close()
        observation_connection.close()
        server_exitcode, server_forced = _stop_process(server)

    if failure is not None:
        raise failure
    if server_forced:
        pytest.fail("KV cache server did not stop after releasing its Worker mapping")
    assert server_exitcode == 0
