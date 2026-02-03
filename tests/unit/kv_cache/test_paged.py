"""Tests for paged KV cache implementation."""
from __future__ import annotations

import pytest
import torch


class TestPagedKVCacheConfig:
    """Tests for paged KV cache configuration."""

    def test_config_default_page_size(self) -> None:
        """Default page size is 16."""
        from layerzero.kv_cache.paged import PagedKVCacheConfig

        config = PagedKVCacheConfig()
        assert config.page_size == 16

    def test_config_custom_page_size(self) -> None:
        """Custom page size is respected."""
        from layerzero.kv_cache.paged import PagedKVCacheConfig

        config = PagedKVCacheConfig(page_size=32)
        assert config.page_size == 32

    def test_config_page_size_power_of_two(self) -> None:
        """Page size must be power of two."""
        from layerzero.kv_cache.paged import PagedKVCacheConfig

        with pytest.raises(ValueError, match="power of two"):
            PagedKVCacheConfig(page_size=17)

    def test_config_block_count(self) -> None:
        """Block count configuration."""
        from layerzero.kv_cache.paged import PagedKVCacheConfig

        config = PagedKVCacheConfig(num_blocks=1024, page_size=16)
        assert config.num_blocks == 1024


class TestPagedKVCache:
    """Tests for paged KV cache manager."""

    def test_create_paged_cache(self) -> None:
        """Create paged KV cache."""
        from layerzero.kv_cache.paged import PagedKVCache, PagedKVCacheConfig

        config = PagedKVCacheConfig(
            num_blocks=64,
            page_size=16,
            num_layers=32,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        cache = PagedKVCache(config)

        assert cache.num_blocks == 64
        assert cache.page_size == 16

    def test_allocate_sequence(self) -> None:
        """Allocate blocks for a sequence."""
        from layerzero.kv_cache.paged import PagedKVCache, PagedKVCacheConfig

        config = PagedKVCacheConfig(
            num_blocks=64,
            page_size=16,
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        cache = PagedKVCache(config)

        # Allocate for sequence of 50 tokens
        seq_id = cache.allocate(seq_len=50)

        assert seq_id is not None
        # 50 tokens / 16 page_size = 4 blocks needed (ceil)
        assert cache.get_num_blocks_for_seq(seq_id) == 4

    def test_free_sequence(self) -> None:
        """Free allocated blocks."""
        from layerzero.kv_cache.paged import PagedKVCache, PagedKVCacheConfig

        config = PagedKVCacheConfig(
            num_blocks=64,
            page_size=16,
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        cache = PagedKVCache(config)

        seq_id = cache.allocate(seq_len=50)
        initial_free = cache.num_free_blocks

        cache.free(seq_id)

        # Blocks returned to free pool
        assert cache.num_free_blocks > initial_free

    def test_get_block_table(self, sample_block_table: torch.Tensor) -> None:
        """Get block table for sequence."""
        from layerzero.kv_cache.paged import PagedKVCache, PagedKVCacheConfig

        config = PagedKVCacheConfig(
            num_blocks=64,
            page_size=16,
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        cache = PagedKVCache(config)

        seq_id = cache.allocate(seq_len=50)
        block_table = cache.get_block_table(seq_id)

        assert block_table is not None
        assert block_table.dtype == torch.int32

    def test_get_cache_tensors(self) -> None:
        """Get K and V cache tensors."""
        from layerzero.kv_cache.paged import PagedKVCache, PagedKVCacheConfig

        config = PagedKVCacheConfig(
            num_blocks=64,
            page_size=16,
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        cache = PagedKVCache(config)

        k_cache, v_cache = cache.get_cache_tensors()

        # Shape: (num_blocks, num_layers, 2, page_size, num_kv_heads, head_dim)
        # or similar layout depending on implementation
        assert k_cache is not None
        assert v_cache is not None

    def test_out_of_memory_raises(self) -> None:
        """Raise when no free blocks available."""
        from layerzero.kv_cache.paged import PagedKVCache, PagedKVCacheConfig

        config = PagedKVCacheConfig(
            num_blocks=4,  # Very small
            page_size=16,
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        cache = PagedKVCache(config)

        # Allocate until exhausted
        cache.allocate(seq_len=50)  # Uses 4 blocks

        with pytest.raises(RuntimeError, match="[Oo]ut of.*memory|blocks"):
            cache.allocate(seq_len=50)


class TestPagedKVCacheMetadata:
    """Tests for paged cache metadata."""

    def test_get_metadata(self) -> None:
        """Get cache metadata."""
        from layerzero.kv_cache.paged import PagedKVCache, PagedKVCacheConfig

        config = PagedKVCacheConfig(
            num_blocks=64,
            page_size=16,
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        cache = PagedKVCache(config)

        metadata = cache.get_metadata()

        assert metadata["strategy"] == "paged"
        assert metadata["page_size"] == 16
        assert metadata["num_blocks"] == 64
        assert metadata["dtype"] == "float16"

    def test_sequence_metadata(self) -> None:
        """Get metadata for specific sequence."""
        from layerzero.kv_cache.paged import PagedKVCache, PagedKVCacheConfig

        config = PagedKVCacheConfig(
            num_blocks=64,
            page_size=16,
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        cache = PagedKVCache(config)
        seq_id = cache.allocate(seq_len=50)

        seq_meta = cache.get_sequence_metadata(seq_id)

        assert seq_meta["seq_id"] == seq_id
        assert seq_meta["num_blocks"] == 4
        assert seq_meta["current_len"] == 50


class TestPagedKVCacheExtend:
    """Tests for extending cached sequences."""

    def test_extend_sequence(self) -> None:
        """Extend existing sequence."""
        from layerzero.kv_cache.paged import PagedKVCache, PagedKVCacheConfig

        config = PagedKVCacheConfig(
            num_blocks=64,
            page_size=16,
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        cache = PagedKVCache(config)
        seq_id = cache.allocate(seq_len=16)  # 1 block

        initial_blocks = cache.get_num_blocks_for_seq(seq_id)
        assert initial_blocks == 1

        # Extend past page boundary
        cache.extend(seq_id, additional_len=20)

        new_blocks = cache.get_num_blocks_for_seq(seq_id)
        assert new_blocks == 3  # ceil(36/16) = 3

    def test_extend_within_page(self) -> None:
        """Extend without needing new blocks."""
        from layerzero.kv_cache.paged import PagedKVCache, PagedKVCacheConfig

        config = PagedKVCacheConfig(
            num_blocks=64,
            page_size=16,
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        cache = PagedKVCache(config)
        seq_id = cache.allocate(seq_len=10)  # 1 block, 6 slots free

        initial_blocks = cache.get_num_blocks_for_seq(seq_id)

        cache.extend(seq_id, additional_len=5)  # Still fits in block

        assert cache.get_num_blocks_for_seq(seq_id) == initial_blocks


class TestFlashInferIntegration:
    """Tests for FlashInfer paged cache integration."""

    def test_flashinfer_format_block_table(self) -> None:
        """Block table is in FlashInfer-compatible format."""
        from layerzero.kv_cache.paged import PagedKVCache, PagedKVCacheConfig

        config = PagedKVCacheConfig(
            num_blocks=64,
            page_size=16,
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        cache = PagedKVCache(config)
        seq_id = cache.allocate(seq_len=50)

        # Get FlashInfer-compatible data
        fi_data = cache.get_flashinfer_data([seq_id])

        assert "block_tables" in fi_data
        assert "seq_lens" in fi_data
        assert fi_data["seq_lens"][0] == 50

    def test_flashinfer_batch_format(self) -> None:
        """Batch block tables for FlashInfer."""
        from layerzero.kv_cache.paged import PagedKVCache, PagedKVCacheConfig

        config = PagedKVCacheConfig(
            num_blocks=64,
            page_size=16,
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        cache = PagedKVCache(config)

        seq_ids = [
            cache.allocate(seq_len=30),
            cache.allocate(seq_len=50),
            cache.allocate(seq_len=20),
        ]

        fi_data = cache.get_flashinfer_data(seq_ids)

        assert fi_data["block_tables"].shape[0] == 3
        assert len(fi_data["seq_lens"]) == 3


class TestPagedCacheDefragmentation:
    """Tests for cache defragmentation."""

    def test_defragment_cache(self) -> None:
        """Defragment fragmented cache."""
        from layerzero.kv_cache.paged import PagedKVCache, PagedKVCacheConfig

        config = PagedKVCacheConfig(
            num_blocks=64,
            page_size=16,
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        cache = PagedKVCache(config)

        # Create fragmentation
        seq1 = cache.allocate(seq_len=30)
        seq2 = cache.allocate(seq_len=30)
        seq3 = cache.allocate(seq_len=30)

        cache.free(seq2)  # Create hole

        initial_fragmentation = cache.get_fragmentation_ratio()

        cache.defragment()

        # Should reduce fragmentation
        assert cache.get_fragmentation_ratio() <= initial_fragmentation


class TestPagedCacheThreadSafety:
    """Tests for thread-safe paged cache operations."""

    def test_concurrent_allocations(self) -> None:
        """Handle concurrent allocations safely."""
        import threading
        from layerzero.kv_cache.paged import PagedKVCache, PagedKVCacheConfig

        config = PagedKVCacheConfig(
            num_blocks=256,
            page_size=16,
            num_layers=2,
            num_kv_heads=8,
            head_dim=64,
            dtype="float16",
        )

        cache = PagedKVCache(config)
        errors: list[Exception] = []
        seq_ids: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                for _ in range(10):
                    seq_id = cache.allocate(seq_len=16)
                    with lock:
                        seq_ids.append(seq_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All seq_ids should be unique
        assert len(seq_ids) == len(set(seq_ids))
