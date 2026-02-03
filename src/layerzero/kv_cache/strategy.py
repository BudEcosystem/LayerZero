"""KV cache strategy definitions and selection.

This module provides:
- KVCacheStrategy enum
- KVCacheConfig for configuration
- KVCacheContext for kernel selection integration
- Strategy selection and memory estimation utilities
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique
from threading import RLock
from typing import Any

import torch

logger = logging.getLogger(__name__)


@unique
class KVCacheStrategy(str, Enum):
    """KV cache strategy types.

    CONTIGUOUS: Pre-allocated contiguous memory per sequence.
    PAGED: Block-based allocation with page tables (vLLM-style).
    CHUNKED: Fixed-size chunks with dynamic allocation.
    VIRTUAL: Virtual memory with OS-level paging (vAttention).
    """

    CONTIGUOUS = "contiguous"
    PAGED = "paged"
    CHUNKED = "chunked"
    VIRTUAL = "virtual"


@dataclass
class KVCacheConfig:
    """Configuration for KV cache.

    Attributes:
        strategy: Cache strategy type.
        page_size: Block/page size for paged/chunked strategies.
        num_blocks: Total number of blocks to allocate.
        dtype: Data type for cache tensors.
        use_cuda_graphs: Enable CUDA graph compatibility.
        device: Target device.
    """

    strategy: KVCacheStrategy = KVCacheStrategy.CONTIGUOUS
    page_size: int = 16
    num_blocks: int | None = None
    dtype: str = "float16"
    use_cuda_graphs: bool = False
    device: str = "cuda"


@dataclass
class KVCacheContext:
    """KV cache context for kernel selection.

    Attributes:
        layout: KV cache layout (nhd, hnd, bnhd).
        dtype: Cache data type.
        strategy: Cache strategy name.
        page_size: Page size if paged.
    """

    layout: str
    dtype: str
    strategy: str
    page_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        d = {
            "layout": self.layout,
            "dtype": self.dtype,
            "strategy": self.strategy,
        }

        if self.page_size is not None:
            d["page_size"] = self.page_size

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KVCacheContext":
        """Deserialize from dictionary.

        Args:
            d: Dictionary representation.

        Returns:
            KVCacheContext instance.
        """
        return cls(
            layout=d["layout"],
            dtype=d["dtype"],
            strategy=d["strategy"],
            page_size=d.get("page_size"),
        )


class KVCacheManager(ABC):
    """Abstract base class for KV cache managers.

    Provides interface for allocating, freeing, and managing
    KV cache memory across different strategies.
    """

    @abstractmethod
    def allocate(self, seq_len: int) -> int:
        """Allocate cache for a new sequence.

        Args:
            seq_len: Initial sequence length.

        Returns:
            Sequence ID for the allocation.

        Raises:
            RuntimeError: If allocation fails (out of memory).
        """
        ...

    @abstractmethod
    def free(self, seq_id: int) -> None:
        """Free cache for a sequence.

        Args:
            seq_id: Sequence ID to free.
        """
        ...

    @abstractmethod
    def get_cache_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get K and V cache tensors.

        Returns:
            Tuple of (k_cache, v_cache) tensors.
        """
        ...

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Get cache metadata.

        Returns:
            Dictionary with cache metadata.
        """
        ...


# ============================================================================
# Strategy Selection
# ============================================================================

# Dtype to bytes mapping
DTYPE_BYTES: dict[str, int] = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "int8": 1,
}


def estimate_cache_memory_bytes(
    batch_size: int,
    max_seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: str = "float16",
) -> int:
    """Estimate KV cache memory in bytes.

    Args:
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV heads.
        head_dim: Head dimension.
        dtype: Data type.

    Returns:
        Estimated memory in bytes.
    """
    bytes_per_element = DTYPE_BYTES.get(dtype, 2)

    # K + V per layer: 2 * seq_len * num_kv_heads * head_dim
    per_layer_per_seq = 2 * max_seq_len * num_kv_heads * head_dim * bytes_per_element

    total = batch_size * num_layers * per_layer_per_seq

    logger.debug(
        "Estimated KV cache: %d bytes (%.2f GB) for batch=%d, seq=%d, layers=%d",
        total,
        total / (1024**3),
        batch_size,
        max_seq_len,
        num_layers,
    )

    return total


def select_kv_cache_strategy(
    max_seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    memory_budget_gb: float,
    prefer_strategy: KVCacheStrategy | None = None,
) -> KVCacheStrategy:
    """Select optimal KV cache strategy.

    Args:
        max_seq_len: Maximum sequence length.
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV heads.
        head_dim: Head dimension.
        memory_budget_gb: Available memory in GB.
        prefer_strategy: Preferred strategy if compatible.

    Returns:
        Selected KV cache strategy.
    """
    # If explicit preference, use it
    if prefer_strategy is not None:
        logger.debug("Using preferred strategy: %s", prefer_strategy.value)
        return prefer_strategy

    # Estimate memory for single sequence
    seq_memory = estimate_cache_memory_bytes(
        batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    memory_budget_bytes = memory_budget_gb * (1024**3)

    # Decision logic
    if max_seq_len > 32000:
        # Large context: prefer paged or virtual
        if seq_memory * 2 > memory_budget_bytes:
            # Memory constrained: use virtual (OS-level paging)
            logger.debug(
                "Large context + memory constrained: selecting VIRTUAL"
            )
            return KVCacheStrategy.VIRTUAL
        else:
            # Paged for efficient memory management
            logger.debug("Large context: selecting PAGED")
            return KVCacheStrategy.PAGED
    elif max_seq_len > 8000:
        # Medium context: paged is generally good
        logger.debug("Medium context: selecting PAGED")
        return KVCacheStrategy.PAGED
    else:
        # Small context: contiguous is simpler and fast
        logger.debug("Small context: selecting CONTIGUOUS")
        return KVCacheStrategy.CONTIGUOUS


# ============================================================================
# Kernel Cache Compatibility
# ============================================================================


def is_kernel_cache_compatible(
    kernel_spec: dict[str, Any],
    strategy: str,
    layout: str,
) -> bool:
    """Check if kernel is compatible with cache configuration.

    Args:
        kernel_spec: Kernel specification dict.
        strategy: Cache strategy name.
        layout: Cache layout name.

    Returns:
        True if kernel supports the cache configuration.
    """
    # Check strategy support
    if strategy == "paged":
        if not kernel_spec.get("supports_paged_kv", False):
            return False

    # Check layout support
    supported_layouts = kernel_spec.get("supported_kv_layouts", [])
    if layout not in supported_layouts:
        return False

    return True


def filter_kernels_by_cache(
    kernels: list[dict[str, Any]],
    strategy: str,
    layout: str,
) -> list[dict[str, Any]]:
    """Filter kernels by cache compatibility.

    Args:
        kernels: List of kernel specs.
        strategy: Cache strategy name.
        layout: Cache layout name.

    Returns:
        Filtered list of compatible kernels.
    """
    filtered = [
        k for k in kernels
        if is_kernel_cache_compatible(k, strategy, layout)
    ]

    logger.debug(
        "Filtered %d kernels to %d for strategy=%s, layout=%s",
        len(kernels),
        len(filtered),
        strategy,
        layout,
    )

    return filtered
