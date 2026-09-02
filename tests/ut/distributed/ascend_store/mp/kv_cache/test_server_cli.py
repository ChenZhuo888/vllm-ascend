import signal
import threading
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache import cli


def test_main_runs_kv_cache_server(monkeypatch):
    run_server = MagicMock(return_value=0)
    monkeypatch.setattr(cli, "_run_kv_cache_server", run_server)

    assert (
        cli.main(
            [
                "kv-cache-server",
                "--bind-url",
                "tcp://127.0.0.1:6000",
                "--scheduler-threads",
                "2",
                "--worker-threads",
                "6",
            ]
        )
        == 0
    )

    run_server.assert_called_once_with("tcp://127.0.0.1:6000", 2, 6)


def test_main_uses_shared_server_defaults(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_STORE_SERVER_URL", "tcp://127.0.0.1:7000")
    run_server = MagicMock(return_value=0)
    monkeypatch.setattr(cli, "_run_kv_cache_server", run_server)

    assert cli.main(["kv-cache-server"]) == 0

    run_server.assert_called_once_with(
        "tcp://127.0.0.1:7000",
        cli.DEFAULT_SCHEDULER_THREADS,
        cli.DEFAULT_WORKER_THREADS,
    )


def test_main_rejects_empty_default_server_url(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_STORE_SERVER_URL", "")

    with pytest.raises(ValueError, match="VLLM_ASCEND_STORE_SERVER_URL"):
        cli.main(["kv-cache-server"])


@pytest.mark.parametrize("option", ["--scheduler-threads", "--worker-threads"])
@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_main_rejects_invalid_worker_count(option, value):
    with pytest.raises(SystemExit):
        cli.main(["kv-cache-server", option, value])


def test_repeated_sigterm_does_not_abort(monkeypatch):
    server, handlers, restored = _run_server_with_signals(monkeypatch, [signal.SIGTERM, signal.SIGTERM])

    server.request_stop.assert_called_once_with()
    server.abort.assert_not_called()
    assert restored == {signal.SIGINT, signal.SIGTERM}
    assert set(handlers) == {signal.SIGINT, signal.SIGTERM}


def test_second_sigint_aborts_once(monkeypatch):
    server, _, _ = _run_server_with_signals(monkeypatch, [signal.SIGINT, signal.SIGINT], wait_for_abort=True)

    server.request_stop.assert_called_once_with()
    server.abort.assert_called_once_with()


def test_sigint_aborts_after_sigterm(monkeypatch):
    server, _, _ = _run_server_with_signals(monkeypatch, [signal.SIGTERM, signal.SIGINT], wait_for_abort=True)

    server.request_stop.assert_called_once_with()
    server.abort.assert_called_once_with()


def test_failed_drain_request_aborts(monkeypatch):
    server, _, _ = _run_server_with_signals(
        monkeypatch,
        [signal.SIGTERM],
        request_stop_result=False,
        wait_for_abort=True,
    )

    server.request_stop.assert_called_once_with()
    server.abort.assert_called_once_with()


def _run_server_with_signals(
    monkeypatch,
    received_signals: list[int],
    *,
    request_stop_result: bool = True,
    wait_for_abort: bool = False,
) -> tuple[MagicMock, dict[int, object], set[int]]:
    handlers: dict[int, object] = {}
    restored: set[int] = set()

    def install_handler(signum, handler):
        if callable(handler):
            handlers[signum] = handler
        else:
            restored.add(signum)

    monkeypatch.setattr(cli.signal, "getsignal", lambda _signum: signal.SIG_DFL)
    monkeypatch.setattr(cli.signal, "signal", install_handler)

    stop_requested = threading.Event()
    abort_requested = threading.Event()
    server = MagicMock(endpoint="tcp://127.0.0.1:6000")

    def request_stop():
        stop_requested.set()
        return request_stop_result

    def abort():
        abort_requested.set()

    def run():
        for signum in received_signals:
            handler = handlers[signum]
            assert callable(handler)
            handler(signum, None)
        completed = abort_requested.wait(1) if wait_for_abort else stop_requested.wait(1)
        assert completed, "server shutdown was not requested"

    server.request_stop.side_effect = request_stop
    server.abort.side_effect = abort
    server.run.side_effect = run
    server_class = MagicMock(return_value=server)
    monkeypatch.setattr(cli, "KVCacheServer", server_class)

    assert cli._run_kv_cache_server("tcp://127.0.0.1:6000", 2, 6) == 0
    server_class.assert_called_once_with(
        "tcp://127.0.0.1:6000",
        scheduler_threads=2,
        worker_threads=6,
    )
    return server, handlers, restored
