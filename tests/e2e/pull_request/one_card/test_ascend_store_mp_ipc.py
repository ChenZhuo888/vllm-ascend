import contextlib
import multiprocessing
import traceback
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

import pytest

_MESSAGE_TIMEOUT_S = 60.0
_PROCESS_EXIT_TIMEOUT_S = 10.0
_PRODUCER_RELEASE_TIMEOUT_S = 90.0


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
