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

from ...metadata import AscendConnectorMetadata
from ..rpc import MPProtocolError
from .registration import (
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerIdentity,
    WorkerRegistration,
)
from .synchronization import NPUEventSpec
from .view import (
    BlocksView,
    CachedReqsView,
    RequestView,
    ScheduledNewReqPayload,
    SchedulerOutputView,
    WorkerKVCacheSpec,
)

ACK_RESPONSE = b"OK"

_INTEGER_BYTES = 8
_BYTE_ORDER = "big"
_SCHEDULER_REQUEST_PAYLOADS = 4
_WORKER_REQUEST_PAYLOADS = 5

_Registration = SchedulerRegistration | WorkerRegistration
RegistrationT = TypeVar("RegistrationT", bound=_Registration)


class KVCacheMethod(str, enum.Enum):
    REGISTER_SCHEDULER = "REGISTER_SCHEDULER"
    REGISTER_WORKER = "REGISTER_WORKER"
    REGISTER_KV_CACHES = "REGISTER_KV_CACHES"
    UNREGISTER_SCHEDULER = "UNREGISTER_SCHEDULER"
    UNREGISTER_WORKER = "UNREGISTER_WORKER"
    RENEW_SCHEDULER = "RENEW_SCHEDULER"
    RENEW_WORKER = "RENEW_WORKER"
    LOOKUP = "LOOKUP"
    UPDATE_STATE_AFTER_ALLOC = "UPDATE_STATE_AFTER_ALLOC"
    BUILD_CONNECTOR_META = "BUILD_CONNECTOR_META"
    REQUEST_FINISHED = "REQUEST_FINISHED"
    UPDATE_CONNECTOR_OUTPUT = "UPDATE_CONNECTOR_OUTPUT"
    WAIT_FOR_SAVE = "WAIT_FOR_SAVE"
    GET_FINISHED = "GET_FINISHED"


@dataclass
class LookupRequestView:
    """Request fields used by lookup; prompt_token_ids preserves length only."""

    request_id: str
    prompt_token_ids: range
    block_hashes: list[BlockHash]
    num_tokens: int


def decode_ack_response(responses: Sequence[bytes], method: KVCacheMethod) -> None:
    response = _single_response(responses, method.value)
    if response != ACK_RESPONSE:
        raise MPProtocolError(f"{method.value} expects an OK response, got {response!r}")


def encode_registration(registration: _Registration) -> bytes:
    try:
        return cloudpickle.dumps(registration)
    except Exception as exc:
        raise MPProtocolError(f"Failed to encode {type(registration).__name__}") from exc


def decode_registration(payloads: Sequence[bytes], expected_type: type[RegistrationT]) -> RegistrationT:
    payload = _single_response(payloads, expected_type.__name__)
    try:
        registration = cloudpickle.loads(payload)
    except Exception as exc:
        raise MPProtocolError(f"Failed to decode {expected_type.__name__}") from exc

    if not isinstance(registration, expected_type):
        raise MPProtocolError(f"Expected {expected_type.__name__}, got {type(registration).__name__}")
    return registration


def encode_registration_request(registration: _Registration) -> tuple[bytes, ...]:
    payload = encode_registration(registration)
    if isinstance(registration, SchedulerRegistration):
        return _encode_scheduler_envelope(registration.identity, registration.session_id, payload)
    if isinstance(registration, WorkerRegistration):
        return _encode_worker_envelope(registration.identity, registration.session_id, payload)
    raise TypeError(f"Unsupported registration type: {type(registration).__name__}")


def decode_registration_request(
    payloads: tuple[bytes, ...],
    expected_type: type[RegistrationT],
) -> tuple[RegistrationT, bytes]:
    if expected_type is SchedulerRegistration:
        identity, session_id, payload = _decode_scheduler_envelope(
            payloads,
            KVCacheMethod.REGISTER_SCHEDULER.value,
        )
    elif expected_type is WorkerRegistration:
        identity, session_id, payload = _decode_worker_envelope(
            payloads,
            KVCacheMethod.REGISTER_WORKER.value,
        )
    else:
        raise TypeError(f"Unsupported registration type: {expected_type.__name__}")

    registration = decode_registration((payload,), expected_type)
    if registration.identity != identity:
        raise MPProtocolError(
            f"{expected_type.__name__} identity does not match request header: "
            f"{registration.identity!r} != {identity!r}"
        )
    if registration.session_id != session_id:
        raise MPProtocolError(
            f"{expected_type.__name__} session does not match request header: "
            f"{registration.session_id!r} != {session_id!r}"
        )
    return registration, payload


def encode_scheduler_session(identity: SchedulerIdentity, session_id: str) -> tuple[bytes, ...]:
    return _encode_scheduler_envelope(identity, session_id, b"")


def decode_scheduler_session(payloads: tuple[bytes, ...]) -> tuple[SchedulerIdentity, str]:
    identity, session_id, body = _decode_scheduler_envelope(payloads, "Scheduler session")
    _require_empty_body(body, "Scheduler session")
    return identity, session_id


def encode_worker_session(identity: WorkerIdentity, session_id: str) -> tuple[bytes, ...]:
    return _encode_worker_envelope(identity, session_id, b"")


def decode_worker_session(payloads: tuple[bytes, ...]) -> tuple[WorkerIdentity, str]:
    identity, session_id, body = _decode_worker_envelope(payloads, "Worker session")
    _require_empty_body(body, "Worker session")
    return identity, session_id


def encode_register_kv_caches_request(
    registration: WorkerRegistration,
    spec: WorkerKVCacheSpec,
) -> tuple[bytes, ...]:
    if not isinstance(spec, WorkerKVCacheSpec):
        raise TypeError(f"spec must be WorkerKVCacheSpec, got {type(spec).__name__}")
    return _encode_worker_request(registration, {"spec": spec}, KVCacheMethod.REGISTER_KV_CACHES)


def decode_register_kv_caches_request(
    payloads: tuple[bytes, ...],
) -> tuple[WorkerIdentity, str, WorkerKVCacheSpec]:
    method = KVCacheMethod.REGISTER_KV_CACHES
    identity, session_id, body = _decode_worker_request(payloads, method)
    (spec,) = _body_fields(body, method.value, "spec")
    _require_type(spec, WorkerKVCacheSpec, "spec")
    return identity, session_id, spec


def encode_wait_for_save_request(
    registration: WorkerRegistration,
    metadata: AscendConnectorMetadata,
    event_spec: NPUEventSpec,
) -> tuple[bytes, ...]:
    if not isinstance(metadata, AscendConnectorMetadata):
        raise TypeError(f"metadata must be AscendConnectorMetadata, got {type(metadata).__name__}")
    if not isinstance(event_spec, NPUEventSpec):
        raise TypeError(f"event_spec must be NPUEventSpec, got {type(event_spec).__name__}")
    return _encode_worker_request(
        registration,
        {"metadata": metadata, "event_spec": event_spec},
        KVCacheMethod.WAIT_FOR_SAVE,
    )


def decode_wait_for_save_request(
    payloads: tuple[bytes, ...],
) -> tuple[WorkerIdentity, str, AscendConnectorMetadata, NPUEventSpec]:
    method = KVCacheMethod.WAIT_FOR_SAVE
    identity, session_id, body = _decode_worker_request(payloads, method)
    metadata, event_spec = _body_fields(body, method.value, "metadata", "event_spec")
    _require_type(metadata, AscendConnectorMetadata, "metadata")
    _require_type(event_spec, NPUEventSpec, "event_spec")
    return identity, session_id, metadata, event_spec


def encode_get_finished_request(
    registration: WorkerRegistration,
    finished_req_ids: set[str],
    metadata: AscendConnectorMetadata,
) -> tuple[bytes, ...]:
    if not isinstance(metadata, AscendConnectorMetadata):
        raise TypeError(f"metadata must be AscendConnectorMetadata, got {type(metadata).__name__}")
    return _encode_worker_request(
        registration,
        {"finished_req_ids": _validate_text_set(finished_req_ids, "finished_req_ids"), "metadata": metadata},
        KVCacheMethod.GET_FINISHED,
    )


def decode_get_finished_request(
    payloads: tuple[bytes, ...],
) -> tuple[WorkerIdentity, str, set[str], AscendConnectorMetadata]:
    method = KVCacheMethod.GET_FINISHED
    identity, session_id, body = _decode_worker_request(payloads, method)
    finished_req_ids, metadata = _body_fields(body, method.value, "finished_req_ids", "metadata")
    _require_type(metadata, AscendConnectorMetadata, "metadata")
    return identity, session_id, _decode_text_set(finished_req_ids, "finished_req_ids"), metadata


def encode_get_finished_response(
    done_sending: set[str],
    done_recving: set[str],
) -> tuple[bytes, ...]:
    return _encode_response(
        KVCacheMethod.GET_FINISHED,
        {
            "done_sending": _validate_text_set(done_sending, "done_sending"),
            "done_recving": _validate_text_set(done_recving, "done_recving"),
        },
    )


def decode_get_finished_response(responses: Sequence[bytes]) -> tuple[set[str], set[str]]:
    body = _decode_response(responses, KVCacheMethod.GET_FINISHED)
    done_sending, done_recving = _body_fields(
        body,
        "GET_FINISHED response",
        "done_sending",
        "done_recving",
    )
    return _decode_text_set(done_sending, "done_sending"), _decode_text_set(done_recving, "done_recving")


def scheduler_affinity_key(_client_identity: bytes, payloads: tuple[bytes, ...]) -> SchedulerIdentity:
    return _decode_scheduler_identity(payloads)


def worker_affinity_key(_client_identity: bytes, payloads: tuple[bytes, ...]) -> WorkerIdentity:
    return _decode_worker_identity(payloads)


def encode_lookup_request(
    registration: SchedulerRegistration,
    request: Request,
    num_computed_tokens: int,
) -> tuple[bytes, ...]:
    body = {
        "request": LookupRequestView(
            request_id=_validate_text(request.request_id, "request_id"),
            prompt_token_ids=range(len(request.prompt_token_ids)),
            block_hashes=[_validate_block_hash(value) for value in request.block_hashes],
            num_tokens=_validate_non_negative_int(request.num_tokens, "num_tokens"),
        ),
        "num_computed_tokens": _validate_non_negative_int(num_computed_tokens, "num_computed_tokens"),
    }
    return _encode_scheduler_request(registration, body, KVCacheMethod.LOOKUP)


def decode_lookup_request(payloads: tuple[bytes, ...]) -> tuple[SchedulerIdentity, str, LookupRequestView, int]:
    method = KVCacheMethod.LOOKUP
    identity, session_id, body = _decode_scheduler_request(payloads, method)
    request, num_computed_tokens = _body_fields(body, method.value, "request", "num_computed_tokens")
    _require_type(request, LookupRequestView, "request")
    return identity, session_id, request, _decode_non_negative_int_value(num_computed_tokens, "num_computed_tokens")


def encode_lookup_response(matched_tokens: int, is_async: bool) -> tuple[bytes, ...]:
    return _encode_response(
        KVCacheMethod.LOOKUP,
        {
            "matched_tokens": _validate_non_negative_int(matched_tokens, "matched_tokens"),
            "is_async": _validate_bool(is_async, "is_async"),
        },
    )


def decode_lookup_response(responses: Sequence[bytes]) -> tuple[int, bool]:
    body = _decode_response(responses, KVCacheMethod.LOOKUP)
    matched_tokens, is_async = _body_fields(body, "LOOKUP response", "matched_tokens", "is_async")
    return (
        _decode_non_negative_int_value(matched_tokens, "matched_tokens"),
        _decode_bool_value(is_async, "is_async"),
    )


def encode_update_state_after_alloc(
    registration: SchedulerRegistration,
    request: Request,
    blocks: KVCacheBlocks,
    num_external_tokens: int,
) -> tuple[bytes, ...]:
    prompt_token_ids = list(request.prompt_token_ids)
    num_external_tokens = _validate_non_negative_int(num_external_tokens, "num_external_tokens")
    body = {
        "request": RequestView(
            request_id=_validate_text(request.request_id, "request_id"),
            prompt_token_ids=prompt_token_ids,
            block_hashes=[_validate_block_hash(value) for value in request.block_hashes],
            num_prompt_tokens=len(prompt_token_ids),
            num_tokens=_validate_non_negative_int(request.num_tokens, "num_tokens"),
            all_token_ids=list(prompt_token_ids),
        ),
        "blocks": BlocksView([list(group) for group in blocks.get_block_ids()] if num_external_tokens > 0 else []),
        "num_external_tokens": num_external_tokens,
    }
    return _encode_scheduler_request(registration, body, KVCacheMethod.UPDATE_STATE_AFTER_ALLOC)


def decode_update_state_after_alloc(
    payloads: tuple[bytes, ...],
) -> tuple[SchedulerIdentity, str, RequestView, BlocksView, int]:
    method = KVCacheMethod.UPDATE_STATE_AFTER_ALLOC
    identity, session_id, body = _decode_scheduler_request(payloads, method)
    request, blocks, num_external_tokens = _body_fields(body, method.value, "request", "blocks", "num_external_tokens")
    _require_type(request, RequestView, "request")
    _require_type(blocks, BlocksView, "blocks")
    return (
        identity,
        session_id,
        request,
        blocks,
        _decode_non_negative_int_value(num_external_tokens, "num_external_tokens"),
    )


def encode_build_connector_meta_request(
    registration: SchedulerRegistration,
    scheduler_output: SchedulerOutput,
    new_token_ids: dict[str, list[int]],
) -> tuple[bytes, ...]:
    cached = scheduler_output.scheduled_cached_reqs
    output = SchedulerOutputView(
        finished_req_ids=set(scheduler_output.finished_req_ids or ()),
        preempted_req_ids=set(scheduler_output.preempted_req_ids or ()),
        num_scheduled_tokens=dict(scheduler_output.num_scheduled_tokens),
        scheduled_new_reqs=[
            ScheduledNewReqPayload(
                req.req_id,
                req.num_computed_tokens,
                [list(group) for group in (req.block_ids or ())],
            )
            for req in scheduler_output.scheduled_new_reqs
        ],
        scheduled_cached_reqs=CachedReqsView(
            req_ids=list(cached.req_ids),
            new_block_ids=[
                None if blocks is None else [list(group) for group in blocks] for blocks in cached.new_block_ids
            ],
            num_computed_tokens=list(cached.num_computed_tokens),
            new_token_ids={req_id: list(tokens) for req_id, tokens in new_token_ids.items()},
        ),
    )
    return _encode_scheduler_request(registration, {"output": output}, KVCacheMethod.BUILD_CONNECTOR_META)


def decode_build_connector_meta_request(
    payloads: tuple[bytes, ...],
) -> tuple[SchedulerIdentity, str, SchedulerOutputView]:
    method = KVCacheMethod.BUILD_CONNECTOR_META
    identity, session_id, body = _decode_scheduler_request(payloads, method)
    (output,) = _body_fields(body, method.value, "output")
    _require_type(output, SchedulerOutputView, "output")
    return identity, session_id, output


def encode_build_connector_meta_response(metadata, touch_block_ids: list[int]) -> tuple[bytes, ...]:
    return _encode_response(
        KVCacheMethod.BUILD_CONNECTOR_META,
        {"metadata": metadata, "touch_block_ids": list(touch_block_ids)},
    )


def decode_build_connector_meta_response(responses: Sequence[bytes]) -> tuple:
    body = _decode_response(responses, KVCacheMethod.BUILD_CONNECTOR_META)
    return _body_fields(body, "BUILD_CONNECTOR_META response", "metadata", "touch_block_ids")


def encode_request_finished(
    registration: SchedulerRegistration,
    request_id: str,
    block_ids,
    all_groups: bool,
) -> tuple[bytes, ...]:
    body = {
        "request_id": _validate_text(request_id, "request_id"),
        "block_ids": block_ids,
        "all_groups": _validate_bool(all_groups, "all_groups"),
    }
    return _encode_scheduler_request(registration, body, KVCacheMethod.REQUEST_FINISHED)


def decode_request_finished(payloads: tuple[bytes, ...]) -> tuple[SchedulerIdentity, str, str, object, bool]:
    method = KVCacheMethod.REQUEST_FINISHED
    identity, session_id, body = _decode_scheduler_request(payloads, method)
    request_id, block_ids, all_groups = _body_fields(
        body,
        method.value,
        "request_id",
        "block_ids",
        "all_groups",
    )
    return (
        identity,
        session_id,
        _decode_text_value(request_id, "request_id"),
        block_ids,
        _decode_bool_value(all_groups, "all_groups"),
    )


def encode_request_finished_response(delay_free: bool, extra: dict | None) -> tuple[bytes, ...]:
    return _encode_response(
        KVCacheMethod.REQUEST_FINISHED,
        {"delay_free": _validate_bool(delay_free, "delay_free"), "extra": extra},
    )


def decode_request_finished_response(responses: Sequence[bytes]) -> tuple[bool, dict | None]:
    body = _decode_response(responses, KVCacheMethod.REQUEST_FINISHED)
    delay_free, extra = _body_fields(body, "REQUEST_FINISHED response", "delay_free", "extra")
    return _decode_bool_value(delay_free, "delay_free"), extra


def encode_update_connector_output(
    registration: SchedulerRegistration,
    completed_events: dict[int, int],
) -> tuple[bytes, ...]:
    return _encode_scheduler_request(
        registration,
        {"completed_events": dict(completed_events)},
        KVCacheMethod.UPDATE_CONNECTOR_OUTPUT,
    )


def decode_update_connector_output(payloads: tuple[bytes, ...]) -> tuple[SchedulerIdentity, str, dict[int, int]]:
    method = KVCacheMethod.UPDATE_CONNECTOR_OUTPUT
    identity, session_id, body = _decode_scheduler_request(payloads, method)
    (completed_events,) = _body_fields(body, method.value, "completed_events")
    if not isinstance(completed_events, dict):
        raise MPProtocolError(f"completed_events must be a dict, got {type(completed_events).__name__}")
    return identity, session_id, completed_events


def encode_update_connector_output_response(free_block_ids: list[int]) -> tuple[bytes, ...]:
    return _encode_response(
        KVCacheMethod.UPDATE_CONNECTOR_OUTPUT,
        {"free_block_ids": list(free_block_ids)},
    )


def decode_update_connector_output_response(responses: Sequence[bytes]) -> list[int]:
    body = _decode_response(responses, KVCacheMethod.UPDATE_CONNECTOR_OUTPUT)
    (free_block_ids,) = _body_fields(body, "UPDATE_CONNECTOR_OUTPUT response", "free_block_ids")
    return _decode_list(free_block_ids, "free_block_ids")


def _encode_scheduler_request(
    registration: SchedulerRegistration,
    body: dict,
    method: KVCacheMethod,
) -> tuple[bytes, ...]:
    return _encode_scheduler_envelope(
        registration.identity,
        registration.session_id,
        _encode_body(body, method.value),
    )


def _decode_scheduler_request(
    payloads: tuple[bytes, ...],
    method: KVCacheMethod,
) -> tuple[SchedulerIdentity, str, dict]:
    identity, session_id, payload = _decode_scheduler_envelope(payloads, method.value)
    return identity, session_id, _decode_body(payload, method.value)


def _encode_worker_request(
    registration: WorkerRegistration,
    body: dict,
    method: KVCacheMethod,
) -> tuple[bytes, ...]:
    return _encode_worker_envelope(
        registration.identity,
        registration.session_id,
        _encode_body(body, method.value),
    )


def _decode_worker_request(
    payloads: tuple[bytes, ...],
    method: KVCacheMethod,
) -> tuple[WorkerIdentity, str, dict]:
    identity, session_id, payload = _decode_worker_envelope(payloads, method.value)
    return identity, session_id, _decode_body(payload, method.value)


def _encode_scheduler_envelope(
    identity: SchedulerIdentity,
    session_id: str,
    body: bytes,
) -> tuple[bytes, ...]:
    return *_encode_scheduler_identity(identity), _encode_text(session_id, "session_id"), body


def _decode_scheduler_envelope(
    payloads: tuple[bytes, ...],
    method: str,
) -> tuple[SchedulerIdentity, str, bytes]:
    _require_payload_count(payloads, _SCHEDULER_REQUEST_PAYLOADS, method)
    return _decode_scheduler_identity(payloads), _decode_text(payloads[2], "session_id"), payloads[3]


def _encode_worker_envelope(
    identity: WorkerIdentity,
    session_id: str,
    body: bytes,
) -> tuple[bytes, ...]:
    return *_encode_worker_identity(identity), _encode_text(session_id, "session_id"), body


def _decode_worker_envelope(
    payloads: tuple[bytes, ...],
    method: str,
) -> tuple[WorkerIdentity, str, bytes]:
    _require_payload_count(payloads, _WORKER_REQUEST_PAYLOADS, method)
    return _decode_worker_identity(payloads), _decode_text(payloads[3], "session_id"), payloads[4]


def _encode_response(method: KVCacheMethod, body: dict) -> tuple[bytes, ...]:
    return (_encode_body(body, f"{method.value} response"),)


def _decode_response(responses: Sequence[bytes], method: KVCacheMethod) -> dict:
    name = f"{method.value} response"
    return _decode_body(_single_response(responses, method.value), name)


def _single_response(responses: Sequence[bytes], method: str) -> bytes:
    if len(responses) != 1:
        raise MPProtocolError(f"{method} expects 1 response payload, got {len(responses)}")
    return responses[0]


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


def _body_fields(body: dict, method: str, *fields: str) -> tuple:
    missing = [field for field in fields if field not in body]
    if missing:
        raise MPProtocolError(f"{method} body is missing fields: {', '.join(missing)}")
    return tuple(body[field] for field in fields)


def _require_type(value, expected_type: type, field_name: str) -> None:
    if not isinstance(value, expected_type):
        raise MPProtocolError(f"{field_name} must be {expected_type.__name__}, got {type(value).__name__}")


def _require_empty_body(body: bytes, method: str) -> None:
    if body:
        raise MPProtocolError(f"{method} body must be empty")


def _require_payload_count(payloads: Sequence[bytes], expected: int, method: str) -> None:
    if len(payloads) != expected:
        raise MPProtocolError(f"{method} expects {expected} payloads, got {len(payloads)}")


def _encode_scheduler_identity(identity: SchedulerIdentity) -> tuple[bytes, ...]:
    return (
        _encode_text(identity.engine_id, "engine_id"),
        _encode_non_negative_int(identity.data_parallel_rank, "data_parallel_rank"),
    )


def _decode_scheduler_identity(payloads: Sequence[bytes]) -> SchedulerIdentity:
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


def _decode_worker_identity(payloads: Sequence[bytes]) -> WorkerIdentity:
    if len(payloads) < 3:
        raise MPProtocolError(f"Worker identity expects at least 3 payloads, got {len(payloads)}")
    return WorkerIdentity(
        engine_id=_decode_text(payloads[0], "engine_id"),
        rank=_decode_non_negative_int(payloads[1], "rank"),
        data_parallel_rank=_decode_non_negative_int(payloads[2], "data_parallel_rank"),
    )


def _encode_non_negative_int(value: int, field_name: str) -> bytes:
    value = _validate_non_negative_int(value, field_name)
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


def _validate_non_negative_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative, got {value}")
    return value


def _decode_non_negative_int_value(value, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MPProtocolError(f"{field_name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise MPProtocolError(f"{field_name} must not be negative, got {value}")
    return value


def _encode_text(value: str, field_name: str) -> bytes:
    return _validate_text(value, field_name).encode()


def _decode_text(payload: bytes, field_name: str) -> str:
    try:
        value = payload.decode()
    except (AttributeError, UnicodeDecodeError) as exc:
        raise MPProtocolError(f"{field_name} payload must be valid UTF-8 bytes") from exc
    return _decode_text_value(value, field_name)


def _validate_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _decode_text_value(value, field_name: str) -> str:
    if not isinstance(value, str):
        raise MPProtocolError(f"{field_name} must be a string, got {type(value).__name__}")
    if not value:
        raise MPProtocolError(f"{field_name} must not be empty")
    return value


def _validate_block_hash(value: BlockHash) -> BlockHash:
    if not isinstance(value, bytes):
        raise TypeError(f"block_hash must be bytes, got {type(value).__name__}")
    if not value:
        raise ValueError("block_hash must not be empty")
    return value


def _validate_text_set(value: set[str], field_name: str) -> set[str]:
    if not isinstance(value, set):
        raise TypeError(f"{field_name} must be a set, got {type(value).__name__}")
    return {_validate_text(item, f"{field_name} item") for item in value}


def _decode_text_set(value, field_name: str) -> set[str]:
    if not isinstance(value, set):
        raise MPProtocolError(f"{field_name} must be a set, got {type(value).__name__}")
    return {_decode_text_value(item, f"{field_name} item") for item in value}


def _validate_bool(value: bool, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean, got {type(value).__name__}")
    return value


def _decode_bool_value(value, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise MPProtocolError(f"{field_name} must be a boolean, got {type(value).__name__}")
    return value


def _decode_list(value, field_name: str) -> list:
    if not isinstance(value, (list, tuple)):
        raise MPProtocolError(f"{field_name} must be a list, got {type(value).__name__}")
    return list(value)
