"""KV cache business facade for AscendStore multiprocessing mode."""

import enum
from collections.abc import Callable

from .rpc import MPClient, MPProtocolError, MPRequestTimeoutError, MPServer, MPServerUnavailableError

_DEFAULT_TIMEOUT_MS = 5000
_TOKEN_COUNT_BYTES = 8
_BYTE_ORDER = "big"

LookupHandler = Callable[[int], int]


class KVCacheMethod(str, enum.Enum):
    LOOKUP = "LOOKUP"


def _encode_token_count(token_count: int) -> bytes:
    if not isinstance(token_count, int):
        raise TypeError(f"token_count must be an integer, got {type(token_count).__name__}")
    if token_count < 0:
        raise ValueError(f"token_count must not be negative, got {token_count}")

    try:
        return token_count.to_bytes(_TOKEN_COUNT_BYTES, byteorder=_BYTE_ORDER)
    except OverflowError as exc:
        raise ValueError(f"token_count is too large: {token_count}") from exc


def _decode_token_count(payload: bytes) -> int:
    if not isinstance(payload, bytes):
        raise MPProtocolError(f"Token count payload must be bytes, got {type(payload).__name__}")
    if len(payload) != _TOKEN_COUNT_BYTES:
        raise MPProtocolError(f"Token count payload must contain {_TOKEN_COUNT_BYTES} bytes, got {len(payload)}")
    return int.from_bytes(payload, byteorder=_BYTE_ORDER)


def _default_lookup_handler(num_computed_tokens: int) -> int:
    return 0


class KVCacheClient:
    """KV cache RPC facade that treats server unavailability as a cache miss."""

    def __init__(self, server_url: str):
        self._rpc_client = MPClient(server_url)

    def __enter__(self) -> "KVCacheClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def is_connected(self) -> bool:
        return self._rpc_client.is_transport_connected

    def lookup(self, num_computed_tokens: int, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> int:
        payload = _encode_token_count(num_computed_tokens)

        try:
            responses = self._rpc_client.request(KVCacheMethod.LOOKUP, [payload], timeout_ms=timeout_ms)
        except (MPRequestTimeoutError, MPServerUnavailableError):
            return 0

        if len(responses) != 1:
            raise MPProtocolError(f"LOOKUP expects 1 response payload, got {len(responses)}")
        return _decode_token_count(responses[0])

    def close(self) -> None:
        self._rpc_client.close()


class KVCacheServer:
    """Adapts typed KV cache operations to MP RPC handlers."""

    def __init__(self, bind_url: str, max_workers: int = 4, lookup_handler: LookupHandler | None = None):
        self._lookup_handler = lookup_handler if lookup_handler is not None else _default_lookup_handler
        handlers = {KVCacheMethod.LOOKUP: self._handle_lookup}
        self._rpc_server = MPServer(bind_url, max_workers=max_workers, handlers=handlers)

    def __enter__(self) -> "KVCacheServer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def endpoint(self) -> str:
        return self._rpc_server.endpoint

    def _handle_lookup(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        if len(payloads) != 1:
            raise ValueError(f"LOOKUP expects 1 payload, got {len(payloads)}")

        num_computed_tokens = _decode_token_count(payloads[0])
        matched_tokens = self._lookup_handler(num_computed_tokens)
        return (_encode_token_count(matched_tokens),)

    def run(self) -> None:
        self._rpc_server.run()

    def close(self) -> None:
        self._rpc_server.close()
