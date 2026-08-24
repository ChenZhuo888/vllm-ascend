"""Scheduler-side lookup logic reused inside the KVCacheServer process."""

from collections.abc import Sequence

from vllm.v1.core.kv_cache_utils import BlockHash

from ..pool_scheduler import KVPoolScheduler
from .registration import SchedulerIdentity, SchedulerRegistration, WorkerLookupHandler
from .request_view import BlockPoolProxy, SchedulerOutputView


class _WorkerLookupBridge:
    """Expose the worker's lookup through KVPoolScheduler's client interface.

    The original scheduler reaches the worker process over zmq via
    LookupKeyClient; inside the server both sides share one process, so the
    bridge invokes the worker's lookup entry directly. use_layerwise is always
    False here because the layerwise paths query the store scheduler directly
    and never go through the client.
    """

    def __init__(self, identity: SchedulerIdentity, lookup_handler: WorkerLookupHandler):
        self._identity = identity
        self._lookup_handler = lookup_handler

    def lookup(
        self,
        token_len: int,
        block_hashes: Sequence[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
        hbm_hit_tokens: int = 0,
    ) -> int:
        return self._lookup_handler(self._identity, token_len, block_hashes, kv_cache_group_ids, False, hbm_hit_tokens)


class MPKVPoolScheduler(KVPoolScheduler):
    """Run the original KVPoolScheduler inside the KVCacheServer process.

    Business methods are inherited unchanged. Only construction differs:
    use_layerwise comes from the engine's extra config, the lookup client is
    the in-process bridge instead of a zmq LookupKeyClient, and the block pool
    is a command-recording proxy whose instructions the connector replays on
    the real scheduler-process pool.
    """

    def __init__(self, registration: SchedulerRegistration, lookup_handler: WorkerLookupHandler):
        vllm_config = registration.vllm_config
        use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False)
        super().__init__(
            vllm_config,
            use_layerwise,
            kv_cache_config=registration.kv_cache_config,
            page_size_bytes=registration.page_size_bytes,
        )
        self.client = _WorkerLookupBridge(registration.identity, lookup_handler)  # type: ignore[assignment]
        self._block_pool = BlockPoolProxy()  # type: ignore[assignment]

    def build_connector_meta(self, scheduler_output: SchedulerOutputView):
        self._sync_request_views(scheduler_output)
        return super().build_connector_meta(scheduler_output)

    def take_block_pool_commands(self) -> list[int]:
        """Block ids the mamba bookkeeping touched; the connector replays them
        on the real BlockPool and clears the list."""
        return self._block_pool.take_touch_ids()

    def _sync_request_views(self, output: SchedulerOutputView) -> None:
        """Refresh the dynamic fields on registered views from the projection.

        The original code reads num_computed_tokens/block_ids/all_token_ids off
        live vLLM objects; here the same values arrive per step inside the
        SchedulerOutputView and are written back onto the stored views first,
        so the inherited methods see the same object state as inprocess mode.
        """
        refreshed_new_reqs = []
        for new_req in output.scheduled_new_reqs:
            entry = self._unfinished_requests.get(new_req.req_id)
            view = entry[0] if entry is not None else None
            if view is None:
                # Keep the payload object; the inherited new-request path
                # raises the same "not in _unfinished_requests" error.
                refreshed_new_reqs.append(new_req)
                continue
            view.num_computed_tokens = new_req.num_computed_tokens
            view.block_ids = tuple(list(group) for group in new_req.block_ids_by_group)
            refreshed_new_reqs.append(view)
        output.scheduled_new_reqs = refreshed_new_reqs

        cached = output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            entry = self._unfinished_requests.get(req_id)
            if entry is None:
                continue
            view = entry[0]
            view.num_computed_tokens = cached.num_computed_tokens[index]
            view.all_token_ids.extend(cached.new_token_ids.get(req_id, []))
