"""KV Cache Strategy Abstraction.

This module provides:
- KVCacheStrategy enum (PAGED, CONTIGUOUS, CHUNKED, VIRTUAL)
- KVCacheLayout enum (NHD, HND, BNHD)
- Abstract KVCacheManager and concrete implementations
- Layout conversion utilities
- Cache selection based on context
"""
from __future__ import annotations

from layerzero.kv_cache.strategy import (
    KVCacheStrategy,
    KVCacheConfig,
    KVCacheContext,
    KVCacheManager,
    select_kv_cache_strategy,
    estimate_cache_memory_bytes,
    is_kernel_cache_compatible,
    filter_kernels_by_cache,
)
from layerzero.kv_cache.layouts import (
    KVCacheLayout,
    convert_layout,
    get_layout_dim_order,
    get_seq_dim,
    get_head_dim,
    validate_layout_shape,
    layout_from_string,
)
from layerzero.kv_cache.paged import (
    PagedKVCache,
    PagedKVCacheConfig,
)

__all__ = [
    # Strategy
    "KVCacheStrategy",
    "KVCacheConfig",
    "KVCacheContext",
    "KVCacheManager",
    "select_kv_cache_strategy",
    "estimate_cache_memory_bytes",
    "is_kernel_cache_compatible",
    "filter_kernels_by_cache",
    # Layouts
    "KVCacheLayout",
    "convert_layout",
    "get_layout_dim_order",
    "get_seq_dim",
    "get_head_dim",
    "validate_layout_shape",
    "layout_from_string",
    # Paged
    "PagedKVCache",
    "PagedKVCacheConfig",
]
