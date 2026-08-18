"""KV cache business facade for AscendStore multiprocessing mode."""

import enum
from collections.abc import Callable, Sequence

from vllm.v1.core.kv_cache_utils import BlockHash

from .rpc import MPClient, MPProtocolError, MPRequestTimeoutError, MPServer, MPServerUnavailableError

_DEFAULT_TIMEOUT_MS = 5000
_INTEGER_BYTES = 8
_BYTE_ORDER = "big"
_REQUEST_HEADER_PAYLOADS = 3

LookupHandler = Callable[[int, list[str], list[int] | None, bool, int], int]


class KVCacheMethod(str, enum.Enum):
    LOOKUP = "LOOKUP"


def _encode_non_negative_int(value: int, field_name: str) -> bytes:
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative, got {value}")

    try:
        return value.to_bytes(_INTEGER_BYTES, byteorder=_BYTE_ORDER)
    except OverflowError as exc:
        raise ValueError(f"{field_name} is too large: {value}") from exc


def _decode_non_negative_int(payload: bytes, field_name: str) -> int:
    if not isinstance(payload, bytes):
        raise MPProtocolError(f"{field_name} payload must be bytes, got {type(payload).__name__}")
    if len(payload) != _INTEGER_BYTES:
        raise MPProtocolError(f"{field_name} payload must contain {_INTEGER_BYTES} bytes, got {len(payload)}")
    return int.from_bytes(payload, byteorder=_BYTE_ORDER)


def _encode_block_hash(block_hash: BlockHash) -> bytes:
    if not isinstance(block_hash, bytes):
        raise TypeError(f"block_hash must be bytes, got {type(block_hash).__name__}")
    return block_hash.hex().encode("ascii")


def _decode_block_hash(payload: bytes) -> str:
    try:
        block_hash = payload.decode("ascii")
        decoded_hash = bytes.fromhex(block_hash)
    except (UnicodeDecodeError, ValueError) as exc:
        raise MPProtocolError("Block hash payload must contain a hexadecimal string") from exc

    if not decoded_hash:
        raise MPProtocolError("Block hash payload must not be empty")
    return block_hash


def _encode_lookup_request(
        token_len: int,
        block_hashes: Sequence[BlockHash],
        kv_cache_group_ids: Sequence[int] | None,
        hbm_hit_tokens: int,
) -> tuple[bytes, ...]:
    group_ids = list(kv_cache_group_ids or [0])
    payloads = [
        _encode_non_negative_int(token_len, "token_len"),
        _encode_non_negative_int(hbm_hit_tokens, "hbm_hit_tokens"),
        _encode_non_negative_int(len(group_ids), "kv_cache_group_count"),
    ]
    payloads.extend(_encode_non_negative_int(group_id, "kv_cache_group_id") for group_id in group_ids)
    payloads.extend(_encode_block_hash(block_hash) for block_hash in block_hashes)
    return tuple(payloads)


def _decode_lookup_request(payloads: tuple[bytes, ...]) -> tuple[int, list[str], list[int], int]:
    if len(payloads) < _REQUEST_HEADER_PAYLOADS:
        raise MPProtocolError(f"LOOKUP expects at least {_REQUEST_HEADER_PAYLOADS} payloads, got {len(payloads)}")

    token_len = _decode_non_negative_int(payloads[0], "token_len")
    hbm_hit_tokens = _decode_non_negative_int(payloads[1], "hbm_hit_tokens")
    group_count = _decode_non_negative_int(payloads[2], "kv_cache_group_count")
    if group_count == 0:
        raise MPProtocolError("LOOKUP requires at least one KV cache group")

    hash_start = _REQUEST_HEADER_PAYLOADS + group_count
    if len(payloads) < hash_start:
        actual_group_count = len(payloads) - _REQUEST_HEADER_PAYLOADS
        raise MPProtocolError(f"LOOKUP declares {group_count} KV cache groups, got {actual_group_count}")

    group_ids = [
        _decode_non_negative_int(payload, "kv_cache_group_id")
        for payload in payloads[_REQUEST_HEADER_PAYLOADS:hash_start]
    ]
    block_hashes = [_decode_block_hash(payload) for payload in payloads[hash_start:]]
    return token_len, block_hashes, group_ids, hbm_hit_tokens


def _default_lookup_handler(
        token_len: int,
        block_hashes: list[str],
        kv_cache_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
        hbm_hit_tokens: int = 0,
) -> int:
    return 0


class KVCacheClient:
    """Typed KV cache RPC client."""

    def __init__(self, server_url: str):
        self._rpc_client = MPClient(server_url)

    def __enter__(self) -> "KVCacheClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def is_connected(self) -> bool:
        return self._rpc_client.is_transport_connected

    def lookup(
            self,
            token_len: int,
            block_hashes: Sequence[BlockHash],
            kv_cache_group_ids: Sequence[int] | None = None,
            hbm_hit_tokens: int = 0,
            timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> int:
        payloads = _encode_lookup_request(token_len, block_hashes, kv_cache_group_ids, hbm_hit_tokens)

        try:
            responses = self._rpc_client.request(KVCacheMethod.LOOKUP, payloads, timeout_ms=timeout_ms)
        except (MPRequestTimeoutError, MPServerUnavailableError):
            return 0

        if len(responses) != 1:
            raise MPProtocolError(f"LOOKUP expects 1 response payload, got {len(responses)}")
        return _decode_non_negative_int(responses[0], "matched_tokens")

    def close(self) -> None:
        self._rpc_client.close()


class KVCacheServer:
    """Adapt typed KV cache operations to MP RPC handlers."""

    def __init__(self, bind_url: str, max_workers: int = 4, lookup_handler: LookupHandler | None = None):
        self._lookup_handler = lookup_handler or _default_lookup_handler
        self._rpc_server = MPServer(
            bind_url, max_workers=max_workers, handlers={KVCacheMethod.LOOKUP: self._handle_lookup}
        )

    def __enter__(self) -> "KVCacheServer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def endpoint(self) -> str:
        return self._rpc_server.endpoint

    def _handle_lookup(self, payloads: tuple[bytes, ...]) -> tuple[bytes, ...]:
        token_len, block_hashes, group_ids, hbm_hit_tokens = _decode_lookup_request(payloads)
        matched_tokens = self._lookup_handler(token_len, block_hashes, group_ids, False, hbm_hit_tokens)
        return (_encode_non_negative_int(matched_tokens, "matched_tokens"),)

    def run(self) -> None:
        self._rpc_server.run()

    def close(self) -> None:
        self._rpc_server.close()
