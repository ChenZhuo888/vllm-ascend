"""KV cache RPC methods and payload codecs."""

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import cloudpickle
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request

from .registration import (
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerIdentity,
    WorkerRegistration,
)
from .rpc import MPProtocolError

ACK_RESPONSE = b"OK"

_INTEGER_BYTES = 8
_BYTE_ORDER = "big"
_LOOKUP_HEADER_PAYLOADS = 6
_ASYNC_RESPONSE = b"\x01"
_SYNC_RESPONSE = b"\x00"

RegistrationT = TypeVar("RegistrationT", bound="SchedulerRegistration | WorkerRegistration")


class KVCacheMethod(str, enum.Enum):
    REGISTER_SCHEDULER = "REGISTER_SCHEDULER"
    REGISTER_WORKER = "REGISTER_WORKER"
    UNREGISTER_SCHEDULER = "UNREGISTER_SCHEDULER"
    UNREGISTER_WORKER = "UNREGISTER_WORKER"
    RENEW_SCHEDULER = "RENEW_SCHEDULER"
    RENEW_WORKER = "RENEW_WORKER"
    LOOKUP = "LOOKUP"


@dataclass
class LookupRequestView:
    """Request fields used by lookup; prompt_token_ids preserves length only."""

    request_id: str
    prompt_token_ids: range
    block_hashes: list[BlockHash]
    num_tokens: int


def encode_registration(registration: SchedulerRegistration | WorkerRegistration) -> bytes:
    try:
        return cloudpickle.dumps(registration)
    except Exception as exc:
        raise MPProtocolError(f"Failed to encode {type(registration).__name__}") from exc


def decode_registration(payloads: tuple[bytes, ...], expected_type: type[RegistrationT]) -> RegistrationT:
    if len(payloads) != 1:
        raise MPProtocolError(f"{expected_type.__name__} expects 1 payload, got {len(payloads)}")

    try:
        registration = cloudpickle.loads(payloads[0])
    except Exception as exc:
        raise MPProtocolError(f"Failed to decode {expected_type.__name__}") from exc

    if not isinstance(registration, expected_type):
        raise MPProtocolError(f"Expected {expected_type.__name__}, got {type(registration).__name__}")
    return registration


def encode_scheduler_session(identity: SchedulerIdentity, session_id: str) -> tuple[bytes, ...]:
    return (
        _encode_text(identity.engine_id, "engine_id"),
        _encode_non_negative_int(identity.data_parallel_rank, "data_parallel_rank"),
        _encode_text(session_id, "session_id"),
    )


def decode_scheduler_session(payloads: tuple[bytes, ...]) -> tuple[SchedulerIdentity, str]:
    if len(payloads) != 3:
        raise MPProtocolError(f"Scheduler session expects 3 payloads, got {len(payloads)}")
    identity = SchedulerIdentity(
        engine_id=_decode_text(payloads[0], "engine_id"),
        data_parallel_rank=_decode_non_negative_int(payloads[1], "data_parallel_rank"),
    )
    return identity, _decode_text(payloads[2], "session_id")


def encode_worker_session(identity: WorkerIdentity, session_id: str) -> tuple[bytes, ...]:
    return (
        _encode_text(identity.engine_id, "engine_id"),
        _encode_non_negative_int(identity.rank, "rank"),
        _encode_non_negative_int(identity.data_parallel_rank, "data_parallel_rank"),
        _encode_text(session_id, "session_id"),
    )


def decode_worker_session(payloads: tuple[bytes, ...]) -> tuple[WorkerIdentity, str]:
    if len(payloads) != 4:
        raise MPProtocolError(f"Worker session expects 4 payloads, got {len(payloads)}")
    identity = WorkerIdentity(
        engine_id=_decode_text(payloads[0], "engine_id"),
        rank=_decode_non_negative_int(payloads[1], "rank"),
        data_parallel_rank=_decode_non_negative_int(payloads[2], "data_parallel_rank"),
    )
    return identity, _decode_text(payloads[3], "session_id")


def encode_lookup_request(
    registration: SchedulerRegistration,
    request: Request,
    num_computed_tokens: int,
) -> tuple[bytes, ...]:
    identity = registration.identity
    payloads = [
        _encode_text(identity.engine_id, "engine_id"),
        _encode_non_negative_int(identity.data_parallel_rank, "data_parallel_rank"),
        _encode_text(request.request_id, "request_id"),
        _encode_non_negative_int(len(request.prompt_token_ids), "prompt_token_count"),
        _encode_non_negative_int(request.num_tokens, "num_tokens"),
        _encode_non_negative_int(num_computed_tokens, "num_computed_tokens"),
    ]
    payloads.extend(_encode_block_hash(block_hash) for block_hash in request.block_hashes)
    payloads.append(_encode_text(registration.session_id, "session_id"))
    return tuple(payloads)


def decode_lookup_request(payloads: tuple[bytes, ...]) -> tuple[SchedulerIdentity, str, LookupRequestView, int]:
    identity = _decode_lookup_identity(payloads)
    if len(payloads) == _LOOKUP_HEADER_PAYLOADS:
        raise MPProtocolError("LOOKUP expects a session_id payload")

    request = LookupRequestView(
        request_id=_decode_text(payloads[2], "request_id"),
        prompt_token_ids=range(_decode_non_negative_int(payloads[3], "prompt_token_count")),
        block_hashes=[_decode_block_hash(payload) for payload in payloads[_LOOKUP_HEADER_PAYLOADS:-1]],
        num_tokens=_decode_non_negative_int(payloads[4], "num_tokens"),
    )
    session_id = _decode_text(payloads[-1], "session_id")
    num_computed_tokens = _decode_non_negative_int(payloads[5], "num_computed_tokens")
    return identity, session_id, request, num_computed_tokens


def lookup_affinity_key(_client_identity: bytes, payloads: tuple[bytes, ...]) -> SchedulerIdentity:
    return _decode_lookup_identity(payloads)


def encode_lookup_response(matched_tokens: int, is_async: bool) -> tuple[bytes, ...]:
    return (
        _encode_non_negative_int(matched_tokens, "matched_tokens"),
        _ASYNC_RESPONSE if is_async else _SYNC_RESPONSE,
    )


def decode_lookup_response(payloads: Sequence[bytes]) -> tuple[int, bool]:
    if len(payloads) != 2:
        raise MPProtocolError(f"LOOKUP expects 2 response payloads, got {len(payloads)}")
    return _decode_non_negative_int(payloads[0], "matched_tokens"), _decode_async_response(payloads[1])


def _decode_lookup_identity(payloads: tuple[bytes, ...]) -> SchedulerIdentity:
    if len(payloads) < _LOOKUP_HEADER_PAYLOADS:
        raise MPProtocolError(f"LOOKUP expects at least {_LOOKUP_HEADER_PAYLOADS} payloads, got {len(payloads)}")
    return SchedulerIdentity(
        engine_id=_decode_text(payloads[0], "engine_id"),
        data_parallel_rank=_decode_non_negative_int(payloads[1], "data_parallel_rank"),
    )


def _encode_non_negative_int(value: int, field_name: str) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool):
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


def _encode_text(value: str, field_name: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value.encode()


def _decode_text(payload: bytes, field_name: str) -> str:
    try:
        value = payload.decode()
    except UnicodeDecodeError as exc:
        raise MPProtocolError(f"{field_name} payload must be valid UTF-8") from exc

    if not value:
        raise MPProtocolError(f"{field_name} payload must not be empty")
    return value


def _encode_block_hash(block_hash: BlockHash) -> bytes:
    if not isinstance(block_hash, bytes):
        raise TypeError(f"block_hash must be bytes, got {type(block_hash).__name__}")
    if not block_hash:
        raise ValueError("block_hash must not be empty")
    return block_hash


def _decode_block_hash(payload: bytes) -> BlockHash:
    if not isinstance(payload, bytes):
        raise MPProtocolError(f"block_hash payload must be bytes, got {type(payload).__name__}")
    if not payload:
        raise MPProtocolError("block_hash payload must not be empty")
    return payload


def _decode_async_response(payload: bytes) -> bool:
    if payload == _ASYNC_RESPONSE:
        return True
    if payload == _SYNC_RESPONSE:
        return False
    raise MPProtocolError(f"Invalid LOOKUP async response: {payload!r}")
