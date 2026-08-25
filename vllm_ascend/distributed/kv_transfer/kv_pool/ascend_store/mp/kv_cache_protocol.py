"""KV cache RPC methods and payload codecs."""

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import cloudpickle
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from .registration import (
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerIdentity,
    WorkerRegistration,
)
from .request_view import (
    BlocksView,
    CachedReqsView,
    RequestView,
    ScheduledNewReqPayload,
    SchedulerOutputView,
)
from .rpc import MPProtocolError

ACK_RESPONSE = b"OK"

_INTEGER_BYTES = 8
_BYTE_ORDER = "big"
_LOOKUP_HEADER_PAYLOADS = 6
_ASYNC_RESPONSE = b"\x01"
_SYNC_RESPONSE = b"\x00"

_Registration = SchedulerRegistration | WorkerRegistration
RegistrationT = TypeVar("RegistrationT", bound=_Registration)


class KVCacheMethod(str, enum.Enum):
    REGISTER_SCHEDULER = "REGISTER_SCHEDULER"
    REGISTER_WORKER = "REGISTER_WORKER"
    UNREGISTER_SCHEDULER = "UNREGISTER_SCHEDULER"
    UNREGISTER_WORKER = "UNREGISTER_WORKER"
    RENEW_SCHEDULER = "RENEW_SCHEDULER"
    RENEW_WORKER = "RENEW_WORKER"
    LOOKUP = "LOOKUP"
    UPDATE_STATE_AFTER_ALLOC = "UPDATE_STATE_AFTER_ALLOC"
    BUILD_CONNECTOR_META = "BUILD_CONNECTOR_META"
    REQUEST_FINISHED = "REQUEST_FINISHED"
    UPDATE_CONNECTOR_OUTPUT = "UPDATE_CONNECTOR_OUTPUT"


@dataclass
class LookupRequestView:
    """Request fields used by lookup; prompt_token_ids preserves length only."""

    request_id: str
    prompt_token_ids: range
    block_hashes: list[BlockHash]
    num_tokens: int


def encode_registration(registration: _Registration) -> bytes:
    try:
        return cloudpickle.dumps(registration)
    except Exception as exc:
        raise MPProtocolError(f"Failed to encode {type(registration).__name__}") from exc


def encode_registration_request(registration: _Registration) -> tuple[bytes, ...]:
    if isinstance(registration, SchedulerRegistration):
        identity_payloads = _encode_scheduler_identity(registration.identity)
    elif isinstance(registration, WorkerRegistration):
        identity_payloads = _encode_worker_identity(registration.identity)
    else:
        raise TypeError(f"Unsupported registration type: {type(registration).__name__}")
    return *identity_payloads, encode_registration(registration)


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


def _encode_body(value: dict, method: str) -> bytes:
    try:
        return cloudpickle.dumps(value)
    except Exception as exc:
        raise MPProtocolError(f"Failed to encode {method} body") from exc


def _decode_body(payload: bytes, method: str) -> dict:
    try:
        value = cloudpickle.loads(payload)
    except Exception as exc:
        raise MPProtocolError(f"Failed to decode {method} body") from exc

    if not isinstance(value, dict):
        raise MPProtocolError(f"{method} body must be a dict, got {type(value).__name__}")
    return value


def decode_registration_request(
    payloads: tuple[bytes, ...], expected_type: type[RegistrationT]
) -> tuple[RegistrationT, bytes]:
    if expected_type is SchedulerRegistration:
        expected_payloads = 3
        decode_identity = _decode_scheduler_identity
    elif expected_type is WorkerRegistration:
        expected_payloads = 4
        decode_identity = _decode_worker_identity
    else:
        raise TypeError(f"Unsupported registration type: {expected_type.__name__}")

    if len(payloads) != expected_payloads:
        raise MPProtocolError(f"{expected_type.__name__} expects {expected_payloads} payloads, got {len(payloads)}")

    identity = decode_identity(payloads)
    serialized_registration = payloads[-1]
    registration = decode_registration((serialized_registration,), expected_type)
    if registration.identity != identity:
        raise MPProtocolError(
            f"{expected_type.__name__} identity does not match request header: "
            f"{registration.identity!r} != {identity!r}"
        )
    return registration, serialized_registration


def encode_scheduler_session(identity: SchedulerIdentity, session_id: str) -> tuple[bytes, ...]:
    return *_encode_scheduler_identity(identity), _encode_text(session_id, "session_id")


def decode_scheduler_session(payloads: tuple[bytes, ...]) -> tuple[SchedulerIdentity, str]:
    if len(payloads) != 3:
        raise MPProtocolError(f"Scheduler session expects 3 payloads, got {len(payloads)}")
    return _decode_scheduler_identity(payloads), _decode_text(payloads[2], "session_id")


def encode_worker_session(identity: WorkerIdentity, session_id: str) -> tuple[bytes, ...]:
    return *_encode_worker_identity(identity), _encode_text(session_id, "session_id")


def decode_worker_session(payloads: tuple[bytes, ...]) -> tuple[WorkerIdentity, str]:
    if len(payloads) != 4:
        raise MPProtocolError(f"Worker session expects 4 payloads, got {len(payloads)}")
    return _decode_worker_identity(payloads), _decode_text(payloads[3], "session_id")


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


def scheduler_affinity_key(_client_identity: bytes, payloads: tuple[bytes, ...]) -> SchedulerIdentity:
    return _decode_scheduler_identity(payloads)


def lookup_affinity_key(_client_identity: bytes, payloads: tuple[bytes, ...]) -> SchedulerIdentity:
    return _decode_lookup_identity(payloads)


def worker_affinity_key(_client_identity: bytes, payloads: tuple[bytes, ...]) -> WorkerIdentity:
    return _decode_worker_identity(payloads)


def encode_lookup_response(matched_tokens: int, is_async: bool) -> tuple[bytes, ...]:
    return (
        _encode_non_negative_int(matched_tokens, "matched_tokens"),
        _ASYNC_RESPONSE if is_async else _SYNC_RESPONSE,
    )


def decode_lookup_response(payloads: Sequence[bytes]) -> tuple[int, bool]:
    if len(payloads) != 2:
        raise MPProtocolError(f"LOOKUP expects 2 response payloads, got {len(payloads)}")
    return _decode_non_negative_int(payloads[0], "matched_tokens"), _decode_async_response(payloads[1])


_UPDATE_STATE_AFTER_ALLOC_PAYLOADS = 6


def encode_update_state_after_alloc(
    registration: SchedulerRegistration,
    request: Request,
    blocks: KVCacheBlocks,
    num_external_tokens: int,
) -> tuple[bytes, ...]:
    """Register the request's static fields and report the allocation result.

    The identity header stays hand-encoded so affinity routing never depends
    on the pickled body; the heavy lists travel in one cloudpickle blob.
    """
    identity = registration.identity
    block_ids_by_group = [list(group) for group in blocks.get_block_ids()] if num_external_tokens > 0 else []
    body = {
        "prompt_token_ids": list(request.prompt_token_ids),
        "block_hashes": list(request.block_hashes),
        "num_tokens": request.num_tokens,
        "block_ids_by_group": block_ids_by_group,
    }
    return (
        _encode_text(identity.engine_id, "engine_id"),
        _encode_non_negative_int(identity.data_parallel_rank, "data_parallel_rank"),
        _encode_text(request.request_id, "request_id"),
        _encode_non_negative_int(num_external_tokens, "num_external_tokens"),
        _encode_text(registration.session_id, "session_id"),
        _encode_body(body, KVCacheMethod.UPDATE_STATE_AFTER_ALLOC.value),
    )


def decode_update_state_after_alloc(
    payloads: tuple[bytes, ...],
) -> tuple[SchedulerIdentity, str, RequestView, BlocksView, int]:
    _require_payload_count(payloads, _UPDATE_STATE_AFTER_ALLOC_PAYLOADS, KVCacheMethod.UPDATE_STATE_AFTER_ALLOC.value)
    identity = _decode_scheduler_identity(payloads)
    body = _decode_body(payloads[5], KVCacheMethod.UPDATE_STATE_AFTER_ALLOC.value)
    try:
        view = RequestView(
            request_id=_decode_text(payloads[2], "request_id"),
            prompt_token_ids=list(body["prompt_token_ids"]),
            block_hashes=list(body["block_hashes"]),
            num_prompt_tokens=len(body["prompt_token_ids"]),
            num_tokens=body["num_tokens"],
            all_token_ids=list(body["prompt_token_ids"]),
        )
        blocks = BlocksView(block_ids_by_group=[list(group) for group in body["block_ids_by_group"]])
        num_external_tokens = _decode_non_negative_int(payloads[3], "num_external_tokens")
    except KeyError as exc:
        raise MPProtocolError(f"UPDATE_STATE_AFTER_ALLOC body is missing key: {exc}") from exc
    session_id = _decode_text(payloads[4], "session_id")
    return identity, session_id, view, blocks, num_external_tokens


_BUILD_CONNECTOR_META_PAYLOADS = 4


def encode_build_connector_meta_request(
    registration: SchedulerRegistration,
    scheduler_output: SchedulerOutput,
    new_token_ids: dict[str, list[int]],
) -> tuple[bytes, ...]:
    """Project the SchedulerOutput fields the scheduler-side business needs.

    Identity stays hand-encoded for affinity routing; the projection body
    (per-step dynamic fields plus token increments) travels as one blob.
    new_token_ids carries the all_token_ids increments the connector collects
    from its local Request references.
    """
    identity = registration.identity
    cached = scheduler_output.scheduled_cached_reqs
    body = {
        "finished_req_ids": set(scheduler_output.finished_req_ids or ()),
        "preempted_req_ids": set(scheduler_output.preempted_req_ids or ()),
        "num_scheduled_tokens": dict(scheduler_output.num_scheduled_tokens),
        "new_reqs": [
            (req.req_id, req.num_computed_tokens, [list(group) for group in (req.block_ids or ())])
            for req in scheduler_output.scheduled_new_reqs
        ],
        "cached_req_ids": list(cached.req_ids),
        "cached_new_block_ids": [
            None if blocks is None else [list(group) for group in blocks] for blocks in cached.new_block_ids
        ],
        "cached_num_computed_tokens": list(cached.num_computed_tokens),
        "cached_new_token_ids": {req_id: list(tokens) for req_id, tokens in new_token_ids.items()},
    }
    return (
        *encode_scheduler_session(identity, registration.session_id),
        _encode_body(body, KVCacheMethod.BUILD_CONNECTOR_META.value),
    )


def decode_build_connector_meta_request(
    payloads: tuple[bytes, ...],
) -> tuple[SchedulerIdentity, str, SchedulerOutputView]:
    _require_payload_count(payloads, _BUILD_CONNECTOR_META_PAYLOADS, KVCacheMethod.BUILD_CONNECTOR_META.value)
    identity = _decode_scheduler_identity(payloads)
    session_id = _decode_text(payloads[2], "session_id")
    body = _decode_body(payloads[3], KVCacheMethod.BUILD_CONNECTOR_META.value)
    view = SchedulerOutputView(
        finished_req_ids=body["finished_req_ids"],
        preempted_req_ids=body["preempted_req_ids"],
        num_scheduled_tokens=body["num_scheduled_tokens"],
        scheduled_new_reqs=[
            ScheduledNewReqPayload(
                req_id=req_id,
                num_computed_tokens=num_computed_tokens,
                block_ids_by_group=block_ids_by_group,
            )
            for req_id, num_computed_tokens, block_ids_by_group in body["new_reqs"]
        ],
        scheduled_cached_reqs=CachedReqsView(
            req_ids=body["cached_req_ids"],
            new_block_ids=body["cached_new_block_ids"],
            num_computed_tokens=body["cached_num_computed_tokens"],
            new_token_ids=body["cached_new_token_ids"],
        ),
    )
    return identity, session_id, view


def encode_build_connector_meta_response(metadata, touch_block_ids: list[int]) -> bytes:
    return _encode_body(
        {"metadata": metadata, "touch_block_ids": list(touch_block_ids)},
        f"{KVCacheMethod.BUILD_CONNECTOR_META.value} response",
    )


def decode_build_connector_meta_response(payload: bytes) -> tuple:
    body = _decode_body(payload, f"{KVCacheMethod.BUILD_CONNECTOR_META.value} response")
    return body["metadata"], body["touch_block_ids"]


_REQUEST_FINISHED_PAYLOADS = 6
_SINGLE_GROUP_VARIANT = b"\x00"
_ALL_GROUPS_VARIANT = b"\x01"


def encode_request_finished(
    registration: SchedulerRegistration,
    request_id: str,
    block_ids,
    all_groups: bool,
) -> tuple[bytes, ...]:
    return (
        *_encode_scheduler_identity(registration.identity),
        _encode_text(request_id, "request_id"),
        _ALL_GROUPS_VARIANT if all_groups else _SINGLE_GROUP_VARIANT,
        _encode_text(registration.session_id, "session_id"),
        _encode_body({"block_ids": block_ids}, KVCacheMethod.REQUEST_FINISHED.value),
    )


def decode_request_finished(payloads: tuple[bytes, ...]) -> tuple[SchedulerIdentity, str, str, object, bool]:
    _require_payload_count(payloads, _REQUEST_FINISHED_PAYLOADS, KVCacheMethod.REQUEST_FINISHED.value)
    identity = _decode_scheduler_identity(payloads)
    request_id = _decode_text(payloads[2], "request_id")
    if payloads[3] not in (_SINGLE_GROUP_VARIANT, _ALL_GROUPS_VARIANT):
        raise MPProtocolError(f"Invalid REQUEST_FINISHED variant: {payloads[3]!r}")
    all_groups = payloads[3] == _ALL_GROUPS_VARIANT
    session_id = _decode_text(payloads[4], "session_id")
    block_ids = _decode_body(payloads[5], KVCacheMethod.REQUEST_FINISHED.value)["block_ids"]
    return identity, session_id, request_id, block_ids, all_groups


def encode_request_finished_response(delay_free: bool, extra: dict | None) -> bytes:
    return _encode_body(
        {"delay_free": delay_free, "extra": extra},
        f"{KVCacheMethod.REQUEST_FINISHED.value} response",
    )


def decode_request_finished_response(payload: bytes) -> tuple[bool, dict | None]:
    body = _decode_body(payload, f"{KVCacheMethod.REQUEST_FINISHED.value} response")
    return body["delay_free"], body["extra"]


_UPDATE_CONNECTOR_OUTPUT_PAYLOADS = 4


def encode_update_connector_output(
    registration: SchedulerRegistration,
    completed_events: dict[int, int],
) -> tuple[bytes, ...]:
    return (
        *encode_scheduler_session(registration.identity, registration.session_id),
        _encode_body(
            {"completed_events": dict(completed_events)},
            KVCacheMethod.UPDATE_CONNECTOR_OUTPUT.value,
        ),
    )


def decode_update_connector_output(payloads: tuple[bytes, ...]) -> tuple[SchedulerIdentity, str, dict[int, int]]:
    _require_payload_count(payloads, _UPDATE_CONNECTOR_OUTPUT_PAYLOADS, KVCacheMethod.UPDATE_CONNECTOR_OUTPUT.value)
    identity = _decode_scheduler_identity(payloads)
    session_id = _decode_text(payloads[2], "session_id")
    completed_events = _decode_body(payloads[3], KVCacheMethod.UPDATE_CONNECTOR_OUTPUT.value)["completed_events"]
    return identity, session_id, completed_events


def encode_update_connector_output_response(free_block_ids: list[int]) -> bytes:
    return _encode_body(
        {"free_block_ids": list(free_block_ids)},
        f"{KVCacheMethod.UPDATE_CONNECTOR_OUTPUT.value} response",
    )


def decode_update_connector_output_response(payload: bytes) -> list[int]:
    return _decode_body(payload, f"{KVCacheMethod.UPDATE_CONNECTOR_OUTPUT.value} response")["free_block_ids"]


def _decode_lookup_identity(payloads: tuple[bytes, ...]) -> SchedulerIdentity:
    if len(payloads) < _LOOKUP_HEADER_PAYLOADS:
        raise MPProtocolError(f"LOOKUP expects at least {_LOOKUP_HEADER_PAYLOADS} payloads, got {len(payloads)}")
    return _decode_scheduler_identity(payloads)


def _require_payload_count(payloads: tuple[bytes, ...], expected: int, method: str) -> None:
    if len(payloads) != expected:
        raise MPProtocolError(f"{method} expects {expected} payloads, got {len(payloads)}")


def _encode_scheduler_identity(identity: SchedulerIdentity) -> tuple[bytes, ...]:
    return (
        _encode_text(identity.engine_id, "engine_id"),
        _encode_non_negative_int(identity.data_parallel_rank, "data_parallel_rank"),
    )


def _decode_scheduler_identity(payloads: tuple[bytes, ...]) -> SchedulerIdentity:
    if len(payloads) < 2:
        raise MPProtocolError(f"Scheduler identity expects at least 2 payloads, got {len(payloads)}")
    return SchedulerIdentity(
        engine_id=_decode_text(payloads[0], "engine_id"),
        data_parallel_rank=_decode_non_negative_int(payloads[1], "data_parallel_rank"),
    )


def _encode_worker_identity(identity: WorkerIdentity) -> tuple[bytes, ...]:
    return (
        _encode_text(identity.engine_id, "engine_id"),
        _encode_non_negative_int(identity.rank, "rank"),
        _encode_non_negative_int(identity.data_parallel_rank, "data_parallel_rank"),
    )


def _decode_worker_identity(payloads: tuple[bytes, ...]) -> WorkerIdentity:
    if len(payloads) < 3:
        raise MPProtocolError(f"Worker identity expects at least 3 payloads, got {len(payloads)}")
    return WorkerIdentity(
        engine_id=_decode_text(payloads[0], "engine_id"),
        rank=_decode_non_negative_int(payloads[1], "rank"),
        data_parallel_rank=_decode_non_negative_int(payloads[2], "data_parallel_rank"),
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
