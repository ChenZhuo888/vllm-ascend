"""KV cache business facade for AscendStore multiprocessing mode."""

import enum

from .rpc import MPClient, MPProtocolError, MPServer

_DEFAULT_TIMEOUT_MS = 5000
_TOKEN_COUNT_BYTES = 8
_BYTE_ORDER = "big"


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
        raise MPProtocolError(
            f"Token count payload must contain {_TOKEN_COUNT_BYTES} bytes, got {len(payload)}"
        )
    return int.from_bytes(payload, byteorder=_BYTE_ORDER)


class KVCacheClient:
    def __init__(self, server_url: str, connect_timeout_ms: int = _DEFAULT_TIMEOUT_MS):
        self._rpc_client = MPClient(server_url)

        try:
            self._rpc_client.wait_until_connected(connect_timeout_ms)
        except Exception:
            self._rpc_client.close()
            raise

    def __enter__(self) -> "KVCacheClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def is_connected(self) -> bool:
        return self._rpc_client.is_transport_connected

    def lookup(self, num_computed_tokens: int, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> int:
        responses = self._rpc_client.request(
            KVCacheMethod.LOOKUP,
            [_encode_token_count(num_computed_tokens)],
            timeout_ms=timeout_ms,
        )
        if len(responses) != 1:
            raise MPProtocolError(f"LOOKUP expects 1 response payload, got {len(responses)}")
        return _decode_token_count(responses[0])

    def close(self) -> None:
        self._rpc_client.close()


class KVCacheServer:
    def __init__(self, bind_url: str, max_workers: int = 4):
        self._rpc_server = MPServer(
            bind_url,
            max_workers=max_workers,
            handlers={KVCacheMethod.LOOKUP: self._handle_lookup},
        )

    def __enter__(self) -> "KVCacheServer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def endpoint(self) -> str:
        return self._rpc_server.endpoint

    @staticmethod
    def _handle_lookup(payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        if len(payloads) != 1:
            raise ValueError(f"LOOKUP expects 1 payload, got {len(payloads)}")

        _decode_token_count(payloads[0])
        return (_encode_token_count(0),)

    def run(self) -> None:
        self._rpc_server.run()

    def close(self) -> None:
        self._rpc_server.close()
