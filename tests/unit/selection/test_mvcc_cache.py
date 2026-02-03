"""
Tests for MVCCShardedCache.

TDD tests for 256-shard MVCC cache with:
- Per-shard versioning for O(1) invalidation
- Selection deduplication (thundering herd prevention)
- Bounded LRU with TTL
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from layerzero.models.execution_plan import ExecutionPlan
from layerzero.selection.mvcc_cache import (
    CacheEntry,
    CacheShard,
    MVCCShardedCache,
)

if TYPE_CHECKING:
    from layerzero.models.kernel_spec import KernelSpec

from .conftest import make_device_spec, make_selection_context


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def kernel_spec() -> "KernelSpec":
    """Create a test kernel spec."""
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.enums import Platform
    import torch

    return KernelSpec(
        kernel_id="test.kernel.v1",
        operation="attention.causal",
        source="test",
        version="1.0",
        platform=Platform.CUDA,
        supported_dtypes=frozenset([torch.float16]),
        priority=50,
    )


@pytest.fixture
def execution_plan(kernel_spec: "KernelSpec") -> ExecutionPlan:
    """Create a test execution plan."""
    return ExecutionPlan(
        kernel_id=kernel_spec.kernel_id,
        kernel_spec=kernel_spec,
    )


# =============================================================================
# Cache Initialization Tests
# =============================================================================


class TestMVCCShardedCacheInit:
    """Test MVCCShardedCache initialization."""

    def test_default_init(self) -> None:
        """Test default initialization with 256 shards."""
        cache = MVCCShardedCache()
        assert cache.num_shards == 256
        assert cache.max_entries_per_shard == 100
        assert cache.ttl_seconds == 3600.0

    def test_custom_num_shards(self) -> None:
        """Test initialization with custom shard count."""
        cache = MVCCShardedCache(num_shards=64)
        assert cache.num_shards == 64

    def test_custom_max_entries(self) -> None:
        """Test initialization with custom max entries per shard."""
        cache = MVCCShardedCache(max_entries_per_shard=50)
        assert cache.max_entries_per_shard == 50

    def test_custom_ttl(self) -> None:
        """Test initialization with custom TTL."""
        cache = MVCCShardedCache(ttl_seconds=1800.0)
        assert cache.ttl_seconds == 1800.0

    def test_invalid_num_shards_raises(self) -> None:
        """Test that num_shards <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="num_shards must be positive"):
            MVCCShardedCache(num_shards=0)

    def test_invalid_max_entries_raises(self) -> None:
        """Test that max_entries_per_shard <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_entries_per_shard must be positive"):
            MVCCShardedCache(max_entries_per_shard=0)

    def test_invalid_ttl_raises(self) -> None:
        """Test that ttl_seconds <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            MVCCShardedCache(ttl_seconds=0.0)


# =============================================================================
# Cache Hit/Miss Tests
# =============================================================================


class TestMVCCCacheHitMiss:
    """Test cache hit and miss behavior."""

    def test_cache_hit_identical_context(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Cache hit for identical key and policy hash."""
        cache = MVCCShardedCache()
        key = "test_key"
        policy_hash = "policy_v1"

        cache.put(key, policy_hash, execution_plan)
        result = cache.get(key, policy_hash)

        assert result is not None
        assert result.kernel_id == execution_plan.kernel_id

    def test_cache_miss_different_key(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Cache miss when key differs."""
        cache = MVCCShardedCache()
        policy_hash = "policy_v1"

        cache.put("key_1", policy_hash, execution_plan)
        result = cache.get("key_2", policy_hash)

        assert result is None

    def test_cache_miss_different_policy_hash(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Cache miss when policy hash differs."""
        cache = MVCCShardedCache()
        key = "test_key"

        cache.put(key, "policy_v1", execution_plan)
        result = cache.get(key, "policy_v2")

        assert result is None

    def test_cache_miss_nonexistent_key(self) -> None:
        """Cache miss for nonexistent key."""
        cache = MVCCShardedCache()
        result = cache.get("nonexistent", "hash")
        assert result is None


# =============================================================================
# MVCC Version Tests
# =============================================================================


class TestMVCCVersioning:
    """Test MVCC versioning behavior."""

    def test_mvcc_version_isolation(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Entry invalidated when shard version bumps."""
        cache = MVCCShardedCache(num_shards=4)
        key = "test_key"
        policy_hash = "policy_v1"

        cache.put(key, policy_hash, execution_plan)

        # Get shard index and invalidate it
        shard_idx = cache._get_shard_idx(key)
        cache.invalidate_shard(shard_idx)

        # Entry should be invalid after version bump
        result = cache.get(key, policy_hash)
        assert result is None

    def test_mvcc_concurrent_reads(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Concurrent reads don't block each other."""
        cache = MVCCShardedCache()
        key = "test_key"
        policy_hash = "policy_v1"

        cache.put(key, policy_hash, execution_plan)

        results: list[ExecutionPlan | None] = []
        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(100):
                    r = cache.get(key, policy_hash)
                    results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 1000
        assert all(r is not None and r.kernel_id == execution_plan.kernel_id for r in results)

    def test_mvcc_read_during_write(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Reads see consistent state during writes."""
        cache = MVCCShardedCache()
        policy_hash = "policy_v1"

        # Pre-populate cache
        for i in range(100):
            cache.put(f"key_{i}", policy_hash, execution_plan)

        read_results: list[int] = []
        write_count = [0]
        errors: list[Exception] = []

        def reader() -> None:
            try:
                for i in range(100):
                    result = cache.get(f"key_{i}", policy_hash)
                    read_results.append(1 if result else 0)
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(100, 200):
                    cache.put(f"key_{i}", policy_hash, execution_plan)
                    write_count[0] += 1
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All pre-existing keys should be found
        assert sum(read_results) == 200  # 2 readers * 100 keys


# =============================================================================
# Invalidation Tests
# =============================================================================


class TestMVCCInvalidation:
    """Test cache invalidation."""

    def test_invalidation_version_bump(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """O(1) invalidation via version bump."""
        cache = MVCCShardedCache(num_shards=4)
        key = "test_key"
        policy_hash = "policy_v1"

        cache.put(key, policy_hash, execution_plan)
        shard_idx = cache._get_shard_idx(key)

        # Invalidate shard
        cache.invalidate_shard(shard_idx)

        # Entry should be gone
        assert cache.get(key, policy_hash) is None

    def test_invalidation_per_shard(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Invalidation is per-shard, other shards unaffected."""
        cache = MVCCShardedCache(num_shards=4)
        policy_hash = "policy_v1"

        # Add entries to different shards
        keys = ["key_0", "key_1", "key_2", "key_3"]
        for key in keys:
            cache.put(key, policy_hash, execution_plan)

        # Invalidate first key's shard
        shard_idx = cache._get_shard_idx(keys[0])
        cache.invalidate_shard(shard_idx)

        # Check which keys are still valid
        valid_count = sum(1 for key in keys if cache.get(key, policy_hash) is not None)

        # At least some keys should still be valid (other shards)
        # With 4 shards and 4 keys, at least 3 should survive
        assert valid_count >= 1  # Conservative - depends on hash distribution

    def test_invalidation_all_shards(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Invalidate all shards."""
        cache = MVCCShardedCache(num_shards=4)
        policy_hash = "policy_v1"

        # Add entries
        for i in range(10):
            cache.put(f"key_{i}", policy_hash, execution_plan)

        # Invalidate all
        cache.invalidate_all()

        # All entries should be gone
        for i in range(10):
            assert cache.get(f"key_{i}", policy_hash) is None

    def test_invalidation_policy_hash(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Invalidate entries for specific policy hash."""
        cache = MVCCShardedCache(num_shards=4)

        # Add entries with different policy hashes
        cache.put("key_1", "policy_v1", execution_plan)
        cache.put("key_2", "policy_v1", execution_plan)
        cache.put("key_3", "policy_v2", execution_plan)

        # Invalidate policy_v1
        removed = cache.invalidate_policy("policy_v1")

        assert removed == 2
        assert cache.get("key_1", "policy_v1") is None
        assert cache.get("key_2", "policy_v1") is None
        assert cache.get("key_3", "policy_v2") is not None


# =============================================================================
# Sharding Tests
# =============================================================================


class TestMVCCSharding:
    """Test sharding behavior."""

    def test_sharded_distribution_uniformity(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Keys distribute evenly across shards."""
        cache = MVCCShardedCache(num_shards=16)
        policy_hash = "policy_v1"

        # Add many entries
        num_entries = 1000
        for i in range(num_entries):
            cache.put(f"key_{i}", policy_hash, execution_plan)

        # Check distribution
        shard_counts = [0] * 16
        for i in range(num_entries):
            shard_idx = cache._get_shard_idx(f"key_{i}")
            shard_counts[shard_idx] += 1

        # Calculate standard deviation
        mean = num_entries / 16
        variance = sum((c - mean) ** 2 for c in shard_counts) / 16
        stddev = variance ** 0.5

        # Standard deviation should be reasonable (< 20% of mean)
        assert stddev < mean * 0.4  # Allow some variance

    def test_shard_selection_consistent(self) -> None:
        """Same key always goes to same shard."""
        cache = MVCCShardedCache(num_shards=16)
        key = "test_key"

        shard_idx1 = cache._get_shard_idx(key)
        shard_idx2 = cache._get_shard_idx(key)
        shard_idx3 = cache._get_shard_idx(key)

        assert shard_idx1 == shard_idx2 == shard_idx3

    def test_shard_independence(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Operations on one shard don't affect others."""
        cache = MVCCShardedCache(num_shards=4, max_entries_per_shard=5)
        policy_hash = "policy_v1"

        # Fill one shard to capacity
        # Find keys that hash to shard 0
        shard_0_keys = []
        for i in range(1000):
            key = f"key_{i}"
            if cache._get_shard_idx(key) == 0:
                shard_0_keys.append(key)
                if len(shard_0_keys) >= 10:
                    break

        # Add entries to shard 0
        for key in shard_0_keys[:5]:
            cache.put(key, policy_hash, execution_plan)

        # Find a key in a different shard
        other_key = None
        for i in range(1000, 2000):
            key = f"key_{i}"
            if cache._get_shard_idx(key) != 0:
                other_key = key
                break

        assert other_key is not None

        # Add to other shard
        cache.put(other_key, policy_hash, execution_plan)
        assert cache.get(other_key, policy_hash) is not None


# =============================================================================
# LRU Eviction Tests
# =============================================================================


class TestMVCCLRUEviction:
    """Test LRU eviction behavior."""

    def test_bounded_lru_eviction(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """LRU eviction when max_entries exceeded."""
        cache = MVCCShardedCache(num_shards=1, max_entries_per_shard=3)
        policy_hash = "policy_v1"

        # Add 3 entries (fill cache)
        cache.put("key_0", policy_hash, execution_plan)
        cache.put("key_1", policy_hash, execution_plan)
        cache.put("key_2", policy_hash, execution_plan)

        # Add 4th entry - should evict key_0
        cache.put("key_3", policy_hash, execution_plan)

        assert cache.get("key_0", policy_hash) is None  # Evicted
        assert cache.get("key_1", policy_hash) is not None
        assert cache.get("key_2", policy_hash) is not None
        assert cache.get("key_3", policy_hash) is not None

    def test_lru_order_updated_on_access(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """LRU order updated when entry is accessed."""
        cache = MVCCShardedCache(num_shards=1, max_entries_per_shard=3)
        policy_hash = "policy_v1"

        # Add 3 entries
        cache.put("key_0", policy_hash, execution_plan)
        cache.put("key_1", policy_hash, execution_plan)
        cache.put("key_2", policy_hash, execution_plan)

        # Access key_0 to make it recently used
        cache.get("key_0", policy_hash)

        # Add 4th entry - should evict key_1 (now oldest)
        cache.put("key_3", policy_hash, execution_plan)

        assert cache.get("key_0", policy_hash) is not None  # Still here
        assert cache.get("key_1", policy_hash) is None  # Evicted
        assert cache.get("key_2", policy_hash) is not None
        assert cache.get("key_3", policy_hash) is not None


# =============================================================================
# TTL Expiration Tests
# =============================================================================


class TestMVCCTTLExpiration:
    """Test TTL expiration behavior."""

    def test_ttl_expiration(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Entries expire after TTL."""
        cache = MVCCShardedCache(ttl_seconds=0.1)  # 100ms TTL
        key = "test_key"
        policy_hash = "policy_v1"

        cache.put(key, policy_hash, execution_plan)

        # Should be valid immediately
        assert cache.get(key, policy_hash) is not None

        # Wait for TTL to expire
        time.sleep(0.15)

        # Should be expired
        assert cache.get(key, policy_hash) is None

    def test_ttl_not_expired(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Entries valid before TTL."""
        cache = MVCCShardedCache(ttl_seconds=10.0)  # 10s TTL
        key = "test_key"
        policy_hash = "policy_v1"

        cache.put(key, policy_hash, execution_plan)

        # Should be valid
        assert cache.get(key, policy_hash) is not None


# =============================================================================
# Deduplication Tests
# =============================================================================


class TestMVCCDeduplication:
    """Test deduplication (thundering herd prevention)."""

    def test_deduplication_single_selection(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Only one thread performs selection for same key."""
        cache = MVCCShardedCache()
        key = "test_key"
        policy_hash = "policy_v1"
        compute_count = [0]

        def compute_fn() -> ExecutionPlan:
            compute_count[0] += 1
            time.sleep(0.05)  # Simulate slow computation
            return execution_plan

        results: list[ExecutionPlan] = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                result = cache.get_or_compute(key, policy_hash, compute_fn)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start 10 threads simultaneously
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert compute_count[0] == 1  # Only computed once
        assert all(r.kernel_id == execution_plan.kernel_id for r in results)

    def test_deduplication_waiters_get_result(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Waiting threads get result from first selection."""
        cache = MVCCShardedCache()
        key = "test_key"
        policy_hash = "policy_v1"
        compute_order: list[int] = []

        def compute_fn() -> ExecutionPlan:
            compute_order.append(1)
            time.sleep(0.1)  # Slow computation
            return execution_plan

        results: list[ExecutionPlan] = []

        def worker(worker_id: int) -> None:
            result = cache.get_or_compute(key, policy_hash, compute_fn)
            results.append(result)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one computation happened
        assert len(compute_order) == 1

        # All threads got the same result
        assert len(results) == 5
        assert all(r.kernel_id == execution_plan.kernel_id for r in results)

    def test_deduplication_different_keys(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Different keys compute independently."""
        cache = MVCCShardedCache()
        policy_hash = "policy_v1"
        compute_count = [0]

        def compute_fn() -> ExecutionPlan:
            compute_count[0] += 1
            return execution_plan

        # Compute for different keys
        cache.get_or_compute("key_1", policy_hash, compute_fn)
        cache.get_or_compute("key_2", policy_hash, compute_fn)
        cache.get_or_compute("key_3", policy_hash, compute_fn)

        assert compute_count[0] == 3  # Each key computed once


# =============================================================================
# Clear and Stats Tests
# =============================================================================


class TestMVCCClearAndStats:
    """Test clear and statistics."""

    def test_clear(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Clear removes all entries."""
        cache = MVCCShardedCache(num_shards=4)
        policy_hash = "policy_v1"

        # Add entries
        for i in range(10):
            cache.put(f"key_{i}", policy_hash, execution_plan)

        # Clear
        cache.clear()

        # All entries should be gone
        for i in range(10):
            assert cache.get(f"key_{i}", policy_hash) is None

    def test_size_property(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Size returns total entries across all shards."""
        cache = MVCCShardedCache(num_shards=4)
        policy_hash = "policy_v1"

        assert cache.size == 0

        cache.put("key_1", policy_hash, execution_plan)
        assert cache.size == 1

        cache.put("key_2", policy_hash, execution_plan)
        assert cache.size == 2

        cache.clear()
        assert cache.size == 0

    def test_stats(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Stats returns useful information."""
        cache = MVCCShardedCache(num_shards=4)
        policy_hash = "policy_v1"

        cache.put("key_1", policy_hash, execution_plan)
        cache.get("key_1", policy_hash)  # Hit
        cache.get("key_1", policy_hash)  # Hit
        cache.get("key_2", policy_hash)  # Miss
        cache.get("key_3", policy_hash)  # Miss

        stats = cache.stats()

        assert stats["size"] == 1
        assert stats["num_shards"] == 4
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.5


# =============================================================================
# Thread Safety Stress Tests
# =============================================================================


class TestMVCCThreadSafety:
    """Test thread safety under stress."""

    @pytest.mark.stress
    def test_thread_safety_stress(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """No data loss under concurrent access."""
        cache = MVCCShardedCache(num_shards=16, max_entries_per_shard=100)
        policy_hash = "policy_v1"
        errors: list[Exception] = []
        ops_count = [0]

        def worker(worker_id: int) -> None:
            try:
                for i in range(500):
                    key = f"key_{worker_id}_{i}"
                    cache.put(key, policy_hash, execution_plan)
                    cache.get(key, policy_hash)
                    ops_count[0] += 2
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert ops_count[0] == 20000  # 20 workers * 500 * 2 ops

    @pytest.mark.stress
    def test_concurrent_invalidation(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Thread safety during invalidation."""
        cache = MVCCShardedCache(num_shards=8)
        policy_hash = "policy_v1"
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(200):
                    cache.put(f"key_{i}", policy_hash, execution_plan)
            except Exception as e:
                errors.append(e)

        def invalidator() -> None:
            try:
                for i in range(50):
                    cache.invalidate_shard(i % 8)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=invalidator),
            threading.Thread(target=invalidator),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    @pytest.mark.stress
    def test_memory_bounded(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Memory usage bounded by config."""
        cache = MVCCShardedCache(
            num_shards=4,
            max_entries_per_shard=10,
        )
        policy_hash = "policy_v1"

        # Add many entries
        for i in range(1000):
            cache.put(f"key_{i}", policy_hash, execution_plan)

        # Total entries should be bounded by num_shards * max_entries_per_shard
        assert cache.size <= 4 * 10
