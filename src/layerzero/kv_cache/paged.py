"""Paged KV cache implementation.

This module provides:
- PagedKVCacheConfig for paged cache configuration
- PagedKVCache manager with block-based allocation
- FlashInfer-compatible data formatting
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import torch

from layerzero.kv_cache.strategy import KVCacheManager

logger = logging.getLogger(__name__)


def _is_power_of_two(n: int) -> bool:
    """Check if n is a power of two."""
    return n > 0 and (n & (n - 1)) == 0


@dataclass
class PagedKVCacheConfig:
    """Configuration for paged KV cache.

    Attributes:
        page_size: Number of tokens per page/block.
        num_blocks: Total number of blocks to allocate.
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension per head.
        dtype: Data type for cache tensors.
        device: Target device.
    """

    page_size: int = 16
    num_blocks: int = 1024
    num_layers: int = 1
    num_kv_heads: int = 8
    head_dim: int = 64
    dtype: str = "float16"
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not _is_power_of_two(self.page_size):
            raise ValueError(
                f"page_size must be a power of two, got {self.page_size}"
            )


@dataclass
class _SequenceInfo:
    """Internal sequence allocation info."""

    seq_id: int
    block_indices: list[int]
    current_len: int


class PagedKVCache(KVCacheManager):
    """Paged KV cache manager.

    Implements block-based KV cache allocation similar to vLLM's
    PagedAttention. Supports dynamic memory allocation with
    minimal fragmentation.

    Attributes:
        num_blocks: Total number of blocks.
        page_size: Tokens per block.
        num_free_blocks: Currently available blocks.
    """

    def __init__(self, config: PagedKVCacheConfig) -> None:
        """Initialize paged KV cache.

        Args:
            config: Cache configuration.
        """
        self._config = config
        self._lock = RLock()

        # Block management
        self._free_blocks: list[int] = list(range(config.num_blocks))
        self._sequences: dict[int, _SequenceInfo] = {}
        self._next_seq_id = 0

        # Allocate cache tensors
        self._k_cache, self._v_cache = self._allocate_cache_tensors()

        logger.debug(
            "PagedKVCache initialized: %d blocks, page_size=%d",
            config.num_blocks,
            config.page_size,
        )

    def _allocate_cache_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate K and V cache tensors.

        Returns:
            Tuple of (k_cache, v_cache) tensors.
        """
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "fp8_e4m3": torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16,
        }

        torch_dtype = dtype_map.get(self._config.dtype, torch.float16)

        # Shape: (num_blocks, num_layers, page_size, num_kv_heads, head_dim)
        shape = (
            self._config.num_blocks,
            self._config.num_layers,
            self._config.page_size,
            self._config.num_kv_heads,
            self._config.head_dim,
        )

        # Try GPU, fall back to CPU if not available
        try:
            device = self._config.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"

            k_cache = torch.zeros(shape, dtype=torch_dtype, device=device)
            v_cache = torch.zeros(shape, dtype=torch_dtype, device=device)
        except RuntimeError:
            # Fall back to CPU
            k_cache = torch.zeros(shape, dtype=torch_dtype, device="cpu")
            v_cache = torch.zeros(shape, dtype=torch_dtype, device="cpu")

        logger.debug(
            "Allocated cache tensors: shape=%s, dtype=%s",
            shape,
            torch_dtype,
        )

        return k_cache, v_cache

    @property
    def num_blocks(self) -> int:
        """Total number of blocks."""
        return self._config.num_blocks

    @property
    def page_size(self) -> int:
        """Tokens per block."""
        return self._config.page_size

    @property
    def num_free_blocks(self) -> int:
        """Currently available blocks."""
        with self._lock:
            return len(self._free_blocks)

    def allocate(self, seq_len: int) -> int:
        """Allocate blocks for a new sequence.

        Args:
            seq_len: Initial sequence length.

        Returns:
            Sequence ID.

        Raises:
            RuntimeError: If not enough free blocks.
        """
        with self._lock:
            # Calculate blocks needed
            num_blocks_needed = math.ceil(seq_len / self._config.page_size)

            if num_blocks_needed > len(self._free_blocks):
                raise RuntimeError(
                    f"Out of memory: need {num_blocks_needed} blocks, "
                    f"only {len(self._free_blocks)} available"
                )

            # Allocate blocks
            block_indices = []
            for _ in range(num_blocks_needed):
                block_idx = self._free_blocks.pop()
                block_indices.append(block_idx)

            # Create sequence info
            seq_id = self._next_seq_id
            self._next_seq_id += 1

            self._sequences[seq_id] = _SequenceInfo(
                seq_id=seq_id,
                block_indices=block_indices,
                current_len=seq_len,
            )

            logger.debug(
                "Allocated seq_id=%d: %d blocks for %d tokens",
                seq_id,
                num_blocks_needed,
                seq_len,
            )

            return seq_id

    def free(self, seq_id: int) -> None:
        """Free blocks for a sequence.

        Args:
            seq_id: Sequence ID to free.
        """
        with self._lock:
            if seq_id not in self._sequences:
                logger.warning("Attempted to free unknown seq_id=%d", seq_id)
                return

            seq_info = self._sequences.pop(seq_id)

            # Return blocks to free pool
            self._free_blocks.extend(seq_info.block_indices)

            logger.debug(
                "Freed seq_id=%d: %d blocks returned",
                seq_id,
                len(seq_info.block_indices),
            )

    def extend(self, seq_id: int, additional_len: int) -> None:
        """Extend an existing sequence.

        Args:
            seq_id: Sequence ID to extend.
            additional_len: Additional tokens to add.

        Raises:
            RuntimeError: If not enough free blocks.
        """
        with self._lock:
            if seq_id not in self._sequences:
                raise ValueError(f"Unknown seq_id={seq_id}")

            seq_info = self._sequences[seq_id]
            new_len = seq_info.current_len + additional_len
            new_blocks_needed = math.ceil(new_len / self._config.page_size)
            current_blocks = len(seq_info.block_indices)

            if new_blocks_needed > current_blocks:
                # Need more blocks
                blocks_to_add = new_blocks_needed - current_blocks

                if blocks_to_add > len(self._free_blocks):
                    raise RuntimeError(
                        f"Out of memory: need {blocks_to_add} more blocks, "
                        f"only {len(self._free_blocks)} available"
                    )

                for _ in range(blocks_to_add):
                    block_idx = self._free_blocks.pop()
                    seq_info.block_indices.append(block_idx)

            seq_info.current_len = new_len

            logger.debug(
                "Extended seq_id=%d to %d tokens (%d blocks)",
                seq_id,
                new_len,
                len(seq_info.block_indices),
            )

    def get_num_blocks_for_seq(self, seq_id: int) -> int:
        """Get number of blocks allocated for a sequence.

        Args:
            seq_id: Sequence ID.

        Returns:
            Number of blocks.
        """
        with self._lock:
            if seq_id not in self._sequences:
                return 0
            return len(self._sequences[seq_id].block_indices)

    def get_block_table(self, seq_id: int) -> torch.Tensor | None:
        """Get block table for a sequence.

        Args:
            seq_id: Sequence ID.

        Returns:
            Block table tensor or None if not found.
        """
        with self._lock:
            if seq_id not in self._sequences:
                return None

            block_indices = self._sequences[seq_id].block_indices
            return torch.tensor(block_indices, dtype=torch.int32)

    def get_cache_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get K and V cache tensors.

        Returns:
            Tuple of (k_cache, v_cache) tensors.
        """
        return self._k_cache, self._v_cache

    def get_metadata(self) -> dict[str, Any]:
        """Get cache metadata.

        Returns:
            Metadata dictionary.
        """
        with self._lock:
            return {
                "strategy": "paged",
                "page_size": self._config.page_size,
                "num_blocks": self._config.num_blocks,
                "num_free_blocks": len(self._free_blocks),
                "num_sequences": len(self._sequences),
                "dtype": self._config.dtype,
                "device": self._config.device,
            }

    def get_sequence_metadata(self, seq_id: int) -> dict[str, Any]:
        """Get metadata for a specific sequence.

        Args:
            seq_id: Sequence ID.

        Returns:
            Sequence metadata.
        """
        with self._lock:
            if seq_id not in self._sequences:
                raise ValueError(f"Unknown seq_id={seq_id}")

            seq_info = self._sequences[seq_id]
            return {
                "seq_id": seq_id,
                "num_blocks": len(seq_info.block_indices),
                "current_len": seq_info.current_len,
            }

    def get_flashinfer_data(
        self, seq_ids: list[int]
    ) -> dict[str, Any]:
        """Get FlashInfer-compatible batch data.

        Args:
            seq_ids: List of sequence IDs.

        Returns:
            Dictionary with block_tables, seq_lens, etc.
        """
        with self._lock:
            max_blocks = max(
                len(self._sequences[sid].block_indices)
                for sid in seq_ids
            )

            # Create padded block table
            block_tables = torch.zeros(
                (len(seq_ids), max_blocks),
                dtype=torch.int32,
            )

            seq_lens = []
            for i, seq_id in enumerate(seq_ids):
                seq_info = self._sequences[seq_id]
                block_count = len(seq_info.block_indices)
                block_tables[i, :block_count] = torch.tensor(
                    seq_info.block_indices, dtype=torch.int32
                )
                seq_lens.append(seq_info.current_len)

            return {
                "block_tables": block_tables,
                "seq_lens": seq_lens,
                "page_size": self._config.page_size,
            }

    def get_fragmentation_ratio(self) -> float:
        """Calculate memory fragmentation ratio.

        Returns:
            Fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented).
        """
        with self._lock:
            if not self._sequences:
                return 0.0

            # Calculate wasted space (allocated but unused slots in pages)
            total_allocated_slots = 0
            total_used_slots = 0

            for seq_info in self._sequences.values():
                num_blocks = len(seq_info.block_indices)
                total_allocated_slots += num_blocks * self._config.page_size
                total_used_slots += seq_info.current_len

            if total_allocated_slots == 0:
                return 0.0

            wasted = total_allocated_slots - total_used_slots
            return wasted / total_allocated_slots

    def defragment(self) -> None:
        """Defragment the cache by compacting blocks.

        Note: This is a no-op in the current implementation.
        Real defragmentation would require copying data between blocks.
        """
        with self._lock:
            # Sort free blocks to reduce fragmentation on future allocations
            self._free_blocks.sort(reverse=True)

            logger.debug("Defragmented cache: %d free blocks", len(self._free_blocks))
