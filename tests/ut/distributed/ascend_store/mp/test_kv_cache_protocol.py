from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache_protocol import (
    decode_lookup_request,
    decode_lookup_response,
    decode_registration,
    decode_scheduler_session,
    decode_worker_session,
    encode_lookup_request,
    encode_lookup_response,
    encode_registration,
    encode_scheduler_session,
    encode_worker_session,
    lookup_affinity_key,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.registration import (
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerIdentity,
    WorkerRegistration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc import MPProtocolError


def _make_vllm_config():
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(engine_id="engine-0"),
        parallel_config=SimpleNamespace(rank=2, data_parallel_rank=1),
    )


def test_registration_round_trip_and_type_validation() -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=4096,
        session_id="scheduler-session",
    )
    payload = encode_registration(registration)

    assert decode_registration((payload,), SchedulerRegistration) == registration
    with pytest.raises(MPProtocolError, match="Expected WorkerRegistration"):
        decode_registration((payload,), WorkerRegistration)


def test_service_session_round_trip() -> None:
    scheduler_identity = SchedulerIdentity("engine-0", data_parallel_rank=1)
    worker_identity = WorkerIdentity("engine-0", rank=2, data_parallel_rank=1)

    assert decode_scheduler_session(encode_scheduler_session(scheduler_identity, "scheduler-session")) == (
        scheduler_identity,
        "scheduler-session",
    )
    assert decode_worker_session(encode_worker_session(worker_identity, "worker-session")) == (
        worker_identity,
        "worker-session",
    )


def test_lookup_request_preserves_required_fields_and_response_round_trip() -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=4096,
        session_id="scheduler-session",
    )
    request = SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=[1, 2, 3],
        block_hashes=[b"hash-0", b"hash-1"],
        num_tokens=3,
    )

    payloads = encode_lookup_request(registration, request, num_computed_tokens=2)
    identity, session_id, decoded_request, num_computed_tokens = decode_lookup_request(payloads)

    assert lookup_affinity_key(b"client", payloads) == registration.identity
    assert identity == registration.identity
    assert session_id == registration.session_id
    assert decoded_request.request_id == request.request_id
    assert decoded_request.prompt_token_ids == range(len(request.prompt_token_ids))
    assert decoded_request.block_hashes == request.block_hashes
    assert decoded_request.num_tokens == request.num_tokens
    assert num_computed_tokens == 2
    assert decode_lookup_response(encode_lookup_response(16, True)) == (16, True)


def test_lookup_protocol_rejects_malformed_payloads() -> None:
    with pytest.raises(MPProtocolError, match="expects at least 6 payloads"):
        lookup_affinity_key(b"client", ())
    with pytest.raises(MPProtocolError, match="expects 2 response payloads"):
        decode_lookup_response([])
