from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache_protocol import encode_registration
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache_service import KVCacheService
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.registration import (
    SchedulerRegistration,
    WorkerRegistration,
)

_BLOCK_HASHES = [bytes.fromhex("01" * 32), bytes.fromhex("02" * 32)]


class _FakeScheduler:
    def __init__(self, identity, lookup_handler):
        self._identity = identity
        self._lookup_handler = lookup_handler
        self.store_scheduler = object()

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> tuple[int, bool]:
        matched_tokens = self._lookup_handler(
            self._identity,
            len(request.prompt_token_ids),
            request.block_hashes,
            [0],
            False,
            num_computed_tokens,
        )
        return matched_tokens, False


class _FakeWorker:
    def __init__(self, matched_tokens: int = 0):
        self._matched_tokens = matched_tokens
        self.bound_store = None
        self.lookup_hashes = None

    def bind_lookup_store(self, store) -> None:
        self.bound_store = store

    def lookup_scheduler(
        self,
        token_len: int,
        block_hashes: list[str],
        kv_cache_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
        hbm_hit_tokens: int = 0,
    ) -> int:
        self.lookup_hashes = block_hashes
        return min(token_len, self._matched_tokens)


def _make_vllm_config(rank: int = 0, data_parallel_rank: int = 0):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(engine_id="engine-0"),
        parallel_config=SimpleNamespace(rank=rank, data_parallel_rank=data_parallel_rank),
    )


def _scheduler_registration(session_id: str, data_parallel_rank: int = 0) -> SchedulerRegistration:
    return SchedulerRegistration.create(
        _make_vllm_config(data_parallel_rank=data_parallel_rank),
        None,
        0,
        session_id=session_id,
    )


def _worker_registration(session_id: str, rank: int = 0, data_parallel_rank: int = 0) -> WorkerRegistration:
    return WorkerRegistration.create(
        _make_vllm_config(rank, data_parallel_rank),
        None,
        session_id=session_id,
    )


def _create_scheduler(registration, lookup_handler) -> _FakeScheduler:
    return _FakeScheduler(registration.identity, lookup_handler)


@pytest.mark.parametrize("worker_first", [True, False])
def test_lookup_store_is_bound_regardless_of_registration_order(worker_first: bool) -> None:
    scheduler = None
    worker = _FakeWorker()

    def scheduler_factory(registration, lookup_handler):
        nonlocal scheduler
        scheduler = _FakeScheduler(registration.identity, lookup_handler)
        return scheduler

    service = KVCacheService(scheduler_factory, lambda registration: worker)
    scheduler_registration = _scheduler_registration("scheduler-session")
    worker_registration = _worker_registration("worker-session")

    if worker_first:
        service.register_worker(worker_registration, encode_registration(worker_registration))
        service.register_scheduler(scheduler_registration, encode_registration(scheduler_registration))
    else:
        service.register_scheduler(scheduler_registration, encode_registration(scheduler_registration))
        service.register_worker(worker_registration, encode_registration(worker_registration))

    assert scheduler is not None
    assert worker.bound_store is scheduler.store_scheduler


def test_new_scheduler_session_rebinds_existing_worker_store() -> None:
    schedulers = []
    worker = _FakeWorker()

    def scheduler_factory(registration, lookup_handler):
        scheduler = _FakeScheduler(registration.identity, lookup_handler)
        schedulers.append(scheduler)
        return scheduler

    service = KVCacheService(scheduler_factory, lambda registration: worker)
    worker_registration = _worker_registration("worker-session")
    old_scheduler = _scheduler_registration("old-session")
    new_scheduler = _scheduler_registration("new-session")

    service.register_worker(worker_registration, encode_registration(worker_registration))
    service.register_scheduler(old_scheduler, encode_registration(old_scheduler))
    assert worker.bound_store is schedulers[0].store_scheduler

    service.register_scheduler(new_scheduler, encode_registration(new_scheduler))
    assert worker.bound_store is schedulers[1].store_scheduler


def test_lookup_routes_to_rank_zero_worker_in_the_same_dp_group() -> None:
    workers = {(0, 0): _FakeWorker(16), (0, 1): _FakeWorker(32), (1, 0): _FakeWorker(48)}

    def worker_factory(registration):
        identity = registration.identity
        return workers[(identity.data_parallel_rank, identity.rank)]

    service = KVCacheService(_create_scheduler, worker_factory)
    for data_parallel_rank, rank in workers:
        registration = _worker_registration(
            f"worker-{data_parallel_rank}-{rank}",
            rank,
            data_parallel_rank,
        )
        service.register_worker(registration, encode_registration(registration))

    scheduler_registration = _scheduler_registration("scheduler-session")
    service.register_scheduler(scheduler_registration, encode_registration(scheduler_registration))
    request = SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=list(range(64)),
        block_hashes=_BLOCK_HASHES,
        num_tokens=64,
    )

    assert service.lookup(scheduler_registration.identity, scheduler_registration.session_id, request, 0) == (16, False)
    assert workers[(0, 0)].lookup_hashes == [block_hash.hex() for block_hash in _BLOCK_HASHES]
    assert workers[(0, 1)].lookup_hashes is None
    assert workers[(1, 0)].lookup_hashes is None
    assert workers[(0, 0)].bound_store is not None
    assert workers[(0, 1)].bound_store is None
    assert workers[(1, 0)].bound_store is None


def test_lookup_does_not_fall_back_to_a_non_coordinator_worker() -> None:
    worker = _FakeWorker(32)
    service = KVCacheService(_create_scheduler, lambda registration: worker)
    worker_registration = _worker_registration("worker-session", rank=1)
    scheduler_registration = _scheduler_registration("scheduler-session")

    service.register_worker(worker_registration, encode_registration(worker_registration))
    service.register_scheduler(scheduler_registration, encode_registration(scheduler_registration))
    request = SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=list(range(32)),
        block_hashes=_BLOCK_HASHES,
        num_tokens=32,
    )

    assert service.lookup(scheduler_registration.identity, scheduler_registration.session_id, request, 0) == (0, False)
    assert worker.bound_store is None
    assert worker.lookup_hashes is None
