"""Tests for KV cache strategy abstraction."""
from __future__ import annotations

import pytest
from typing import Any

from layerzero.device import GPUGeneration


class TestKVCacheStrategy:
    """Tests for KV cache strategy enum and config."""

    def test_paged_cache_strategy_defined(self) -> None:
        """Paged cache strategy is defined."""
        from layerzero.kv_cache.strategy import KVCacheStrategy

        assert hasattr(KVCacheStrategy, "PAGED")
        assert KVCacheStrategy.PAGED.value == "paged"

    def test_contiguous_cache_strategy_defined(self) -> None:
        """Contiguous cache strategy is defined."""
        from layerzero.kv_cache.strategy import KVCacheStrategy

        assert hasattr(KVCacheStrategy, "CONTIGUOUS")
        assert KVCacheStrategy.CONTIGUOUS.value == "contiguous"

    def test_chunked_cache_strategy_defined(self) -> None:
        """Chunked cache strategy is defined."""
        from layerzero.kv_cache.strategy import KVCacheStrategy

        assert hasattr(KVCacheStrategy, "CHUNKED")
        assert KVCacheStrategy.CHUNKED.value == "chunked"

    def test_virtual_cache_strategy_defined(self) -> None:
        """Virtual (vAttention) cache strategy is defined."""
        from layerzero.kv_cache.strategy import KVCacheStrategy

        assert hasattr(KVCacheStrategy, "VIRTUAL")
        assert KVCacheStrategy.VIRTUAL.value == "virtual"

    def test_all_strategies_are_unique(self) -> None:
        """All strategy values are unique."""
        from layerzero.kv_cache.strategy import KVCacheStrategy

        values = [s.value for s in KVCacheStrategy]
        assert len(values) == len(set(values))


class TestKVCacheConfig:
    """Tests for KV cache configuration."""

    def test_config_default_values(self) -> None:
        """Config has sensible defaults."""
        from layerzero.kv_cache.strategy import KVCacheConfig, KVCacheStrategy

        config = KVCacheConfig()

        assert config.strategy == KVCacheStrategy.CONTIGUOUS
        assert config.page_size == 16
        assert config.num_blocks is None
        assert config.dtype == "float16"

    def test_config_paged_strategy(self) -> None:
        """Config with paged strategy."""
        from layerzero.kv_cache.strategy import KVCacheConfig, KVCacheStrategy

        config = KVCacheConfig(
            strategy=KVCacheStrategy.PAGED,
            page_size=32,
            num_blocks=1024,
        )

        assert config.strategy == KVCacheStrategy.PAGED
        assert config.page_size == 32
        assert config.num_blocks == 1024

    def test_config_virtual_strategy(self) -> None:
        """Config with virtual (vAttention) strategy."""
        from layerzero.kv_cache.strategy import KVCacheConfig, KVCacheStrategy

        config = KVCacheConfig(
            strategy=KVCacheStrategy.VIRTUAL,
            use_cuda_graphs=True,
        )

        assert config.strategy == KVCacheStrategy.VIRTUAL
        assert config.use_cuda_graphs is True

    def test_config_dtype_options(self) -> None:
        """Config supports multiple dtypes."""
        from layerzero.kv_cache.strategy import KVCacheConfig

        for dtype in ["float16", "bfloat16", "float32", "fp8_e4m3"]:
            config = KVCacheConfig(dtype=dtype)
            assert config.dtype == dtype


class TestKVCacheManager:
    """Tests for abstract KV cache manager."""

    def test_manager_is_abstract(self) -> None:
        """KVCacheManager is abstract and cannot be instantiated."""
        from layerzero.kv_cache.strategy import KVCacheManager

        with pytest.raises(TypeError, match="abstract"):
            KVCacheManager()  # type: ignore

    def test_manager_has_required_methods(self) -> None:
        """Manager defines required abstract methods."""
        from layerzero.kv_cache.strategy import KVCacheManager
        import inspect

        # Check abstract methods exist
        assert hasattr(KVCacheManager, "allocate")
        assert hasattr(KVCacheManager, "free")
        assert hasattr(KVCacheManager, "get_cache_tensors")
        assert hasattr(KVCacheManager, "get_metadata")

        # Verify they're abstract
        assert inspect.isabstract(KVCacheManager)


class TestKVCacheSelection:
    """Tests for KV cache strategy selection."""

    def test_select_contiguous_for_small_context(self) -> None:
        """Select contiguous cache for small context windows."""
        from layerzero.kv_cache.strategy import select_kv_cache_strategy

        strategy = select_kv_cache_strategy(
            max_seq_len=512,
            num_layers=32,
            num_kv_heads=8,
            head_dim=64,
            memory_budget_gb=8.0,
        )

        assert strategy.value in ["contiguous", "paged"]

    def test_select_paged_for_large_context(self) -> None:
        """Select paged cache for large context windows."""
        from layerzero.kv_cache.strategy import (
            select_kv_cache_strategy,
            KVCacheStrategy,
        )

        strategy = select_kv_cache_strategy(
            max_seq_len=128000,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            memory_budget_gb=24.0,
        )

        # Large context should prefer paged for memory efficiency
        assert strategy in [KVCacheStrategy.PAGED, KVCacheStrategy.VIRTUAL]

    def test_select_virtual_when_requested(self) -> None:
        """Select virtual when explicitly preferred."""
        from layerzero.kv_cache.strategy import (
            select_kv_cache_strategy,
            KVCacheStrategy,
        )

        strategy = select_kv_cache_strategy(
            max_seq_len=4096,
            num_layers=32,
            num_kv_heads=8,
            head_dim=64,
            memory_budget_gb=16.0,
            prefer_strategy=KVCacheStrategy.VIRTUAL,
        )

        assert strategy == KVCacheStrategy.VIRTUAL

    def test_estimate_cache_memory_contiguous(self) -> None:
        """Estimate memory for contiguous cache."""
        from layerzero.kv_cache.strategy import estimate_cache_memory_bytes

        # batch_size * num_layers * 2 (k+v) * seq_len * num_kv_heads * head_dim * dtype_size
        memory = estimate_cache_memory_bytes(
            batch_size=1,
            max_seq_len=1024,
            num_layers=32,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        # Expected: 1 * 32 * 2 * 1024 * 8 * 64 * 2 = 67108864 bytes = 64 MB
        expected = 1 * 32 * 2 * 1024 * 8 * 64 * 2
        assert memory == expected


class TestCacheContextIntegration:
    """Tests for KV cache info in SelectionContext."""

    def test_cache_layout_in_context(self) -> None:
        """kv_cache_layout is included in SelectionContext."""
        from layerzero.kv_cache.strategy import KVCacheContext

        context = KVCacheContext(
            layout="nhd",
            dtype="float16",
            strategy="paged",
        )

        assert context.layout == "nhd"
        assert context.dtype == "float16"
        assert context.strategy == "paged"

    def test_context_to_dict(self) -> None:
        """Context can be serialized to dict."""
        from layerzero.kv_cache.strategy import KVCacheContext

        context = KVCacheContext(
            layout="hnd",
            dtype="bfloat16",
            strategy="contiguous",
            page_size=32,
        )

        d = context.to_dict()
        assert d["layout"] == "hnd"
        assert d["dtype"] == "bfloat16"
        assert d["strategy"] == "contiguous"
        assert d["page_size"] == 32

    def test_context_from_dict(self) -> None:
        """Context can be deserialized from dict."""
        from layerzero.kv_cache.strategy import KVCacheContext

        d = {
            "layout": "nhd",
            "dtype": "float16",
            "strategy": "paged",
            "page_size": 16,
        }

        context = KVCacheContext.from_dict(d)
        assert context.layout == "nhd"
        assert context.dtype == "float16"
        assert context.strategy == "paged"
        assert context.page_size == 16


class TestKernelCacheCompatibility:
    """Tests for kernel and cache compatibility checking."""

    def test_kernel_supports_paged_cache(self) -> None:
        """Check if kernel supports paged KV cache."""
        from layerzero.kv_cache.strategy import is_kernel_cache_compatible

        kernel_spec = {
            "kernel_id": "flash_attn_v2",
            "supports_paged_kv": True,
            "supported_kv_layouts": ["nhd", "hnd"],
        }

        assert is_kernel_cache_compatible(kernel_spec, strategy="paged", layout="nhd")
        assert is_kernel_cache_compatible(kernel_spec, strategy="paged", layout="hnd")

    def test_kernel_not_supports_paged_cache(self) -> None:
        """Kernel without paged support is filtered."""
        from layerzero.kv_cache.strategy import is_kernel_cache_compatible

        kernel_spec = {
            "kernel_id": "basic_attention",
            "supports_paged_kv": False,
            "supported_kv_layouts": ["nhd"],
        }

        assert not is_kernel_cache_compatible(
            kernel_spec, strategy="paged", layout="nhd"
        )

    def test_kernel_layout_mismatch(self) -> None:
        """Kernel filtered if layout not supported."""
        from layerzero.kv_cache.strategy import is_kernel_cache_compatible

        kernel_spec = {
            "kernel_id": "flash_attn_v2",
            "supports_paged_kv": True,
            "supported_kv_layouts": ["nhd"],
        }

        assert not is_kernel_cache_compatible(
            kernel_spec, strategy="paged", layout="hnd"
        )

    def test_filter_kernels_by_cache(self) -> None:
        """Filter kernel list by cache requirements."""
        from layerzero.kv_cache.strategy import filter_kernels_by_cache

        kernels = [
            {"kernel_id": "k1", "supports_paged_kv": True, "supported_kv_layouts": ["nhd"]},
            {"kernel_id": "k2", "supports_paged_kv": False, "supported_kv_layouts": ["nhd"]},
            {"kernel_id": "k3", "supports_paged_kv": True, "supported_kv_layouts": ["hnd"]},
        ]

        filtered = filter_kernels_by_cache(
            kernels, strategy="paged", layout="nhd"
        )

        assert len(filtered) == 1
        assert filtered[0]["kernel_id"] == "k1"
