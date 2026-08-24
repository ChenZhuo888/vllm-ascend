from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache_protocol import (
    decode_build_connector_meta_request,
    decode_build_connector_meta_response,
    decode_lookup_request,
    decode_lookup_response,
    decode_registration,
    decode_registration_request,
    decode_request_finished,
    decode_request_finished_response,
    decode_scheduler_session,
    decode_update_connector_output,
    decode_update_connector_output_response,
    decode_update_state_after_alloc,
    decode_worker_session,
    encode_build_connector_meta_request,
    encode_build_connector_meta_response,
    encode_lookup_request,
    encode_lookup_response,
    encode_registration,
    encode_registration_request,
    encode_request_finished,
    encode_request_finished_response,
    encode_scheduler_session,
    encode_update_connector_output,
    encode_update_connector_output_response,
    encode_update_state_after_alloc,
    encode_worker_session,
    lookup_affinity_key,
    scheduler_affinity_key,
    worker_affinity_key,
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
    scheduler_registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=4096,
        session_id="scheduler-session",
    )
    worker_registration = WorkerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        session_id="worker-session",
    )
    payload = encode_registration(scheduler_registration)

    assert decode_registration((payload,), SchedulerRegistration) == scheduler_registration
    with pytest.raises(MPProtocolError, match="Expected WorkerRegistration"):
        decode_registration((payload,), WorkerRegistration)

    scheduler_payloads = encode_registration_request(scheduler_registration)
    worker_payloads = encode_registration_request(worker_registration)
    assert decode_registration_request(scheduler_payloads, SchedulerRegistration) == (
        scheduler_registration,
        scheduler_payloads[-1],
    )
    assert decode_registration_request(worker_payloads, WorkerRegistration) == (
        worker_registration,
        worker_payloads[-1],
    )
    assert scheduler_affinity_key(b"client", scheduler_payloads) == scheduler_registration.identity
    assert worker_affinity_key(b"client", worker_payloads) == worker_registration.identity

    mismatched_payloads = (b"other-engine", *scheduler_payloads[1:])
    with pytest.raises(MPProtocolError, match="identity does not match request header"):
        decode_registration_request(mismatched_payloads, SchedulerRegistration)


def test_service_session_round_trip() -> None:
    scheduler_identity = SchedulerIdentity("engine-0", data_parallel_rank=1)
    worker_identity = WorkerIdentity("engine-0", rank=2, data_parallel_rank=1)
    scheduler_payloads = encode_scheduler_session(scheduler_identity, "scheduler-session")
    worker_payloads = encode_worker_session(worker_identity, "worker-session")

    assert decode_scheduler_session(scheduler_payloads) == (
        scheduler_identity,
        "scheduler-session",
    )
    assert decode_worker_session(worker_payloads) == (
        worker_identity,
        "worker-session",
    )
    assert scheduler_affinity_key(b"client", scheduler_payloads) == scheduler_identity
    assert worker_affinity_key(b"client", worker_payloads) == worker_identity


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
    with pytest.raises(MPProtocolError, match="expects at least 2 payloads"):
        scheduler_affinity_key(b"client", ())
    with pytest.raises(MPProtocolError, match="expects 2 response payloads"):
        decode_lookup_response([])


def test_update_state_after_alloc_round_trip() -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=0,
        session_id="scheduler-session",
    )
    request = SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=list(range(64)),
        block_hashes=[bytes([idx]) * 32 for idx in range(4)],
        num_tokens=64,
    )
    blocks = SimpleNamespace(get_block_ids=lambda: ([7, 8], [9]))

    payloads = encode_update_state_after_alloc(registration, request, blocks, num_external_tokens=48)
    identity, session_id, view, decoded_blocks, num_external_tokens = decode_update_state_after_alloc(payloads)

    assert scheduler_affinity_key(b"client", payloads) == registration.identity
    assert (identity, session_id) == (registration.identity, registration.session_id)
    assert num_external_tokens == 48
    assert view.request_id == request.request_id
    assert view.prompt_token_ids == request.prompt_token_ids
    assert view.block_hashes == request.block_hashes
    assert view.num_prompt_tokens == 64
    assert view.num_tokens == 64
    assert decoded_blocks.get_block_ids() == ([7, 8], [9])


def test_update_state_after_alloc_zero_external_carries_no_block_ids() -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=0,
        session_id="scheduler-session",
    )
    request = SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=[1, 2, 3],
        block_hashes=[b"hash-0", b"hash-1"],
        num_tokens=3,
    )
    blocks = SimpleNamespace(get_block_ids=lambda: ([7],))

    payloads = encode_update_state_after_alloc(registration, request, blocks, num_external_tokens=0)
    _, _, _, decoded_blocks, num_external_tokens = decode_update_state_after_alloc(payloads)

    assert num_external_tokens == 0
    assert decoded_blocks.get_block_ids() == ()

    with pytest.raises(MPProtocolError, match="expects 6 payloads"):
        decode_update_state_after_alloc(payloads[:5])


def test_build_connector_meta_request_round_trip() -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=0,
        session_id="scheduler-session",
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids={"done-0"},
        preempted_req_ids=set(),
        num_scheduled_tokens={"request-0": 48},
        scheduled_new_reqs=[SimpleNamespace(req_id="request-0", num_computed_tokens=16, block_ids=([7, 8], [9]))],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["request-1"],
            new_block_ids=[([10],)],
            num_computed_tokens=[64],
        ),
    )
    new_token_ids = {"request-1": [101, 102]}

    payloads = encode_build_connector_meta_request(registration, scheduler_output, new_token_ids)
    identity, session_id, view = decode_build_connector_meta_request(payloads)

    assert scheduler_affinity_key(b"client", payloads) == registration.identity
    assert (identity, session_id) == (registration.identity, registration.session_id)
    assert view.finished_req_ids == {"done-0"}
    assert view.preempted_req_ids == set()
    assert view.num_scheduled_tokens == {"request-0": 48}
    new_req = view.scheduled_new_reqs[0]
    assert (new_req.req_id, new_req.num_computed_tokens) == ("request-0", 16)
    assert new_req.block_ids_by_group == [[7, 8], [9]]
    cached = view.scheduled_cached_reqs
    assert cached.req_ids == ["request-1"]
    assert cached.new_block_ids == [[[10]]]
    assert cached.num_computed_tokens == [64]
    assert cached.new_token_ids == {"request-1": [101, 102]}

    with pytest.raises(MPProtocolError, match="expects 4 payloads"):
        decode_build_connector_meta_request(payloads[:3])


def test_build_connector_meta_response_round_trip() -> None:
    marker = SimpleNamespace(name="metadata-marker")
    payload = encode_build_connector_meta_response(marker, [5, 8])
    metadata, touch_block_ids = decode_build_connector_meta_response(payload)

    assert metadata == marker
    assert touch_block_ids == [5, 8]


def test_request_finished_round_trip() -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=0,
        session_id="scheduler-session",
    )
    for all_groups, block_ids in ((False, [7, 8]), (True, ([7], [8]))):
        payloads = encode_request_finished(registration, "request-0", block_ids, all_groups)
        identity, session_id, request_id, decoded_block_ids, decoded_all_groups = decode_request_finished(payloads)

        assert scheduler_affinity_key(b"client", payloads) == registration.identity
        assert identity == registration.identity
        assert session_id == registration.session_id
        assert request_id == "request-0"
        assert decoded_block_ids == block_ids
        assert decoded_all_groups is all_groups

    response = encode_request_finished_response(True, None)
    assert decode_request_finished_response(response) == (True, None)


def test_update_connector_output_round_trip() -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=0,
        session_id="scheduler-session",
    )

    payloads = encode_update_connector_output(registration, {7: 1, 9: 1})
    identity, session_id, completed_events = decode_update_connector_output(payloads)

    assert scheduler_affinity_key(b"client", payloads) == registration.identity
    assert (identity, session_id) == (registration.identity, registration.session_id)
    assert completed_events == {7: 1, 9: 1}

    assert decode_update_connector_output_response(encode_update_connector_output_response([5, 8])) == [5, 8]
