"""Serializable stand-ins for vLLM objects consumed inside the KVCacheServer process."""

from dataclasses import dataclass, field

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class RequestView:
    """Snapshot of a vLLM Request served to the server-side scheduler code.

    Static fields are registered once at update_state_after_alloc. Fields that
    vLLM mutates in place on the live Request (num_computed_tokens, block_ids,
    all_token_ids) are refreshed per scheduling step from the
    SchedulerOutputView projection before the inherited business methods run.
    """

    request_id: str
    prompt_token_ids: list[int]
    block_hashes: list[BlockHash]
    num_prompt_tokens: int
    num_tokens: int
    num_computed_tokens: int = 0
    block_ids: tuple[list[int], ...] = ()
    # Prompt tokens plus generated tokens, synced incrementally per step.
    all_token_ids: list[int] = field(default_factory=list)

    @property
    def req_id(self) -> str:
        return self.request_id


@dataclass
class BlocksView:
    """Grouped block ids in place of a KVCacheBlocks allocation result."""

    block_ids_by_group: list[list[int]]

    def get_block_ids(self) -> tuple[list[int], ...]:
        return tuple(self.block_ids_by_group)


@dataclass
class ScheduledNewReqPayload:
    """Wire projection of the dynamic fields of vLLM NewRequestData."""

    req_id: str
    num_computed_tokens: int
    block_ids_by_group: list[list[int]]


@dataclass
class CachedReqsView:
    """Server-side stand-in for ScheduledCachedReqs.

    Request ids and new blocks mirror the vLLM payload; num_computed_tokens
    and new_token_ids are the per-step dynamic fields refreshed onto the
    registered RequestViews.
    """

    req_ids: list[str]
    new_block_ids: list[list[list[int]] | None]
    num_computed_tokens: list[int]
    new_token_ids: dict[str, list[int]]


@dataclass
class SchedulerOutputView:
    """Server-side stand-in for SchedulerOutput.

    Elements of scheduled_new_reqs start as ScheduledNewReqPayload and are
    replaced with the refreshed RequestViews by _sync_request_views, so the
    inherited build_connector_meta reads a single object per new request.
    """

    finished_req_ids: set[str]
    preempted_req_ids: set[str]
    num_scheduled_tokens: dict[str, int]
    scheduled_new_reqs: list
    scheduled_cached_reqs: CachedReqsView


class _BlockIdIndex:
    """Stand-in for BlockPool.blocks (dict[int, Block]) inside the server.

    The inherited bookkeeping only uses ``blocks[block_id]`` to hand the
    result straight to ``touch``/``free_blocks``; it never reads any Block
    attribute. So every id can simply map to itself, which makes the
    recorded commands arrive as plain block-id lists.
    """

    def __getitem__(self, block_id: int) -> int:
        return block_id


class BlockPoolProxy:
    """Command-recording stand-in for the scheduler-process BlockPool.

    The inherited mamba bookkeeping reads blocks[id] and calls
    touch/free_blocks on whatever holds the _block_pool slot. The proxy turns
    those calls into block-id lists that the connector replays on the real
    pool after the RPC returns, so no object ever crosses the process border.
    """

    def __init__(self):
        self.blocks = _BlockIdIndex()
        self._touch_ids: list[int] = []
        self._free_ids: list[int] = []

    def touch(self, block_ids) -> None:
        self._touch_ids.extend(block_ids)

    def free_blocks(self, block_ids) -> None:
        self._free_ids.extend(block_ids)

    def take_touch_ids(self) -> list[int]:
        ids = self._touch_ids
        self._touch_ids = []
        return ids

    def take_free_ids(self) -> list[int]:
        ids = self._free_ids
        self._free_ids = []
        return ids
