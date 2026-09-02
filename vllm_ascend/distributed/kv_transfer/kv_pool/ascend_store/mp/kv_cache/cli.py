"""Process entry point for the AscendStore multiprocess KV cache server."""

import argparse
import logging
import signal
import threading
from collections.abc import Sequence
from types import FrameType

from vllm_ascend import envs

from .server import (
    DEFAULT_SCHEDULER_THREADS,
    DEFAULT_WORKER_THREADS,
    KVCacheServer,
)

logger = logging.getLogger(__name__)

# ==============================
# Command-line entry point
# ==============================

# The CLI turns process arguments into existing service configuration and keeps
# process supervision outside the RPC and KV cache layers. It intentionally
# exposes only settings that a server operator owns today.


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected vLLM Ascend command."""
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args(argv)
    if args.command == "kv-cache-server":
        return _run_kv_cache_server(
            args.bind_url,
            args.scheduler_threads,
            args.worker_threads,
        )
    raise RuntimeError(f"Unsupported command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-ascend")
    commands = parser.add_subparsers(dest="command", required=True)

    server_parser = commands.add_parser(
        "kv-cache-server",
        help="Run the AscendStore multiprocess KV cache server.",
    )
    default_bind_url = envs.VLLM_ASCEND_STORE_SERVER_URL
    if not default_bind_url:
        raise ValueError("VLLM_ASCEND_STORE_SERVER_URL must be a non-empty string")
    server_parser.add_argument(
        "--bind-url",
        type=_non_empty_url,
        default=default_bind_url,
        help=f"ZMQ URL to bind (default: {default_bind_url}).",
    )
    server_parser.add_argument(
        "--scheduler-threads",
        type=_positive_int,
        default=DEFAULT_SCHEDULER_THREADS,
        help=f"Scheduler execution threads (default: {DEFAULT_SCHEDULER_THREADS}).",
    )
    server_parser.add_argument(
        "--worker-threads",
        type=_positive_int,
        default=DEFAULT_WORKER_THREADS,
        help=f"Worker execution threads (default: {DEFAULT_WORKER_THREADS}).",
    )
    return parser


def _non_empty_url(value: str) -> str:
    if not value:
        raise argparse.ArgumentTypeError("expected a non-empty ZMQ URL")
    return value


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a value greater than 0, got {parsed}")
    return parsed


def _run_kv_cache_server(
    bind_url: str,
    scheduler_threads: int,
    worker_threads: int,
) -> int:
    shutdown_signals = _ShutdownSignals()
    shutdown_signals.install()
    server_stopped = threading.Event()
    control_thread: threading.Thread | None = None

    try:
        server = KVCacheServer(
            bind_url,
            scheduler_threads=scheduler_threads,
            worker_threads=worker_threads,
        )
        control_thread = threading.Thread(
            target=_coordinate_server_shutdown,
            args=(server, shutdown_signals, server_stopped),
            daemon=True,
            name="ascend-store-kv-shutdown",
        )
        control_thread.start()
        logger.info("AscendStore KV cache server listening on %s", server.endpoint)
        server.run()
    finally:
        server_stopped.set()
        shutdown_signals.wake_waiters()
        if control_thread is not None:
            control_thread.join()
        shutdown_signals.restore()
    return 0


# ==============================
# Server process shutdown
# ==============================

# Signal handlers only record intent because they run in the main thread and
# may interrupt code that already owns a server lock. A control thread performs
# the lifecycle calls: the first signal asks the run loop to drain, while a
# later Ctrl-C cancels that drain without starting either operation twice.


class _ShutdownSignals:
    """Record one graceful shutdown request and one later Ctrl-C abort."""

    def __init__(self) -> None:
        self.shutdown_requested = threading.Event()
        self.abort_requested = threading.Event()
        self._previous_handlers: dict[int, signal.Handlers] = {}

    def install(self) -> None:
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
        except BaseException:
            self.restore()
            raise

    def restore(self) -> None:
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()

    def wake_waiters(self) -> None:
        """Release the control thread after the server stops on its own."""
        self.shutdown_requested.set()
        self.abort_requested.set()

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        if not self.shutdown_requested.is_set():
            self.shutdown_requested.set()
        elif signum == signal.SIGINT:
            self.abort_requested.set()


def _coordinate_server_shutdown(
    server: KVCacheServer,
    shutdown_signals: _ShutdownSignals,
    server_stopped: threading.Event,
) -> None:
    shutdown_signals.shutdown_requested.wait()
    if server_stopped.is_set():
        return

    logger.info("Graceful shutdown started; press Ctrl-C again to abort outstanding requests")
    if not server.request_stop():
        logger.warning("Graceful shutdown is unavailable; aborting the KV cache server")
        server.abort()
        return

    shutdown_signals.abort_requested.wait()
    if server_stopped.is_set():
        return

    logger.warning("Forced KV cache server abort requested")
    server.abort()


if __name__ == "__main__":
    raise SystemExit(main())
