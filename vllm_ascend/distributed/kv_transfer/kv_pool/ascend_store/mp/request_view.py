"""Serializable stand-ins for vLLM objects consumed inside the KVCacheServer process."""

from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class RequestView:
    """Static snapshot of a vLLm Request.

    The original scheduler code reads these fields off the live Request object
    across several callbacks; the server only ever sees snapshots, so this
    view is registered once (at update_state_after_alloc) and later business
    methods read it back from _unfinished_requests instead of the object.
    Fields that vLLM mutates in place (num_computed_tokens, all_token_ids)
    stay out until a later step syncs them per scheduling step.
    """

    request_id: str
    prompt_token_ids: list[int]
    block_hashes: list[BlockHash]
    num_prompt_tokens: int
    num_tokens: int


@dataclass
class BlocksView:
    """Grouped block ids in place of a KVCacheBlocks allocation result."""

    block_ids_by_group: list[list[int]]

    def get_block_ids(self) -> tuple[list[int], ...]:
        return tuple(self.block_ids_by_group)
