"""
Tests for SelectionCache.

TDD tests for LRU cache with policy hash invalidation.
"""
from __future__ import annotations

import pytest
import torch

from layerzero.models.execution_plan import ExecutionPlan
from layerzero.models.kernel_spec import KernelSpec
from layerzero.selection.cache import SelectionCache


class TestSelectionCacheInit:
    """Test SelectionCache initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        cache = SelectionCache()
        assert cache.max_size == 1000
        assert cache.size == 0

    def test_custom_max_size(self) -> None:
        """Test initialization with custom max_size."""
        cache = SelectionCache(max_size=500)
        assert cache.max_size == 500

    def test_min_max_size(self) -> None:
        """Test minimum max_size of 1."""
        cache = SelectionCache(max_size=1)
        assert cache.max_size == 1

    def test_zero_max_size_raises(self) -> None:
        """Test that max_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            SelectionCache(max_size=0)

    def test_negative_max_size_raises(self) -> None:
        """Test that negative max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            SelectionCache(max_size=-1)


class TestSelectionCacheGetPut:
    """Test get and put operations."""

    def test_put_and_get(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Test basic put and get."""
        cache = SelectionCache()
        key = "test_key"
        policy_hash = "policy_hash_v1"

        cache.put(key, policy_hash, execution_plan)
        result = cache.get(key, policy_hash)

        assert result is not None
        assert result.kernel_id == execution_plan.kernel_id

    def test_get_nonexistent_key(self) -> None:
        """Test get with nonexistent key returns None."""
        cache = SelectionCache()
        result = cache.get("nonexistent", "hash")
        assert result is None

    def test_get_with_wrong_policy_hash(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Test get with wrong policy hash returns None."""
        cache = SelectionCache()
        key = "test_key"

        cache.put(key, "policy_v1", execution_plan)
        result = cache.get(key, "policy_v2")

        assert result is None

    def test_put_overwrites_same_key(
        self,
        kernel_spec: KernelSpec,
    ) -> None:
        """Test put overwrites existing entry for same key."""
        cache = SelectionCache()
        key = "test_key"
        policy_hash = "policy_v1"

        plan1 = ExecutionPlan(
            kernel_id="kernel.v1",
            kernel_spec=kernel_spec,
        )
        plan2 = ExecutionPlan(
            kernel_id="kernel.v2",
            kernel_spec=kernel_spec,
        )

        cache.put(key, policy_hash, plan1)
        cache.put(key, policy_hash, plan2)

        result = cache.get(key, policy_hash)
        assert result is not None
        assert result.kernel_id == "kernel.v2"

    def test_multiple_entries(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Test multiple entries with different keys."""
        cache = SelectionCache()
        policy_hash = "policy_v1"

        for i in range(5):
            cache.put(f"key_{i}", policy_hash, execution_plan)

        for i in range(5):
            result = cache.get(f"key_{i}", policy_hash)
            assert result is not None

        assert cache.size == 5


class TestSelectionCacheLRU:
    """Test LRU eviction behavior."""

    def test_eviction_when_full(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Test oldest entries are evicted when cache is full."""
        cache = SelectionCache(max_size=3)
        policy_hash = "policy_v1"

        # Fill cache
        cache.put("key_0", policy_hash, execution_plan)
        cache.put("key_1", policy_hash, execution_plan)
        cache.put("key_2", policy_hash, execution_plan)

        assert cache.size == 3

        # Add one more - should evict key_0
        cache.put("key_3", policy_hash, execution_plan)

        assert cache.size == 3
        assert cache.get("key_0", policy_hash) is None
        assert cache.get("key_1", policy_hash) is not None
        assert cache.get("key_2", policy_hash) is not None
        assert cache.get("key_3", policy_hash) is not None

    def test_access_updates_lru_order(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Test that get updates LRU order."""
        cache = SelectionCache(max_size=3)
        policy_hash = "policy_v1"

        # Fill cache
        cache.put("key_0", policy_hash, execution_plan)
        cache.put("key_1", policy_hash, execution_plan)
        cache.put("key_2", policy_hash, execution_plan)

        # Access key_0 to make it recently used
        cache.get("key_0", policy_hash)

        # Add new entry - should evict key_1 (oldest not accessed)
        cache.put("key_3", policy_hash, execution_plan)

        assert cache.get("key_0", policy_hash) is not None
        assert cache.get("key_1", policy_hash) is None  # Evicted
        assert cache.get("key_2", policy_hash) is not None
        assert cache.get("key_3", policy_hash) is not None


class TestSelectionCacheInvalidation:
    """Test cache invalidation."""

    def test_invalidate_by_policy_hash(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Test invalidation removes entries for specific policy hash."""
        cache = SelectionCache()
        policy_v1 = "policy_v1"
        policy_v2 = "policy_v2"

        cache.put("key_1", policy_v1, execution_plan)
        cache.put("key_2", policy_v1, execution_plan)
        cache.put("key_3", policy_v2, execution_plan)

        assert cache.size == 3

        removed = cache.invalidate(policy_v1)

        assert removed == 2
        assert cache.size == 1
        assert cache.get("key_1", policy_v1) is None
        assert cache.get("key_2", policy_v1) is None
        assert cache.get("key_3", policy_v2) is not None

    def test_invalidate_nonexistent_policy(self) -> None:
        """Test invalidation with no matching entries."""
        cache = SelectionCache()
        removed = cache.invalidate("nonexistent")
        assert removed == 0

    def test_clear(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Test clear removes all entries."""
        cache = SelectionCache()
        policy_hash = "policy_v1"

        for i in range(5):
            cache.put(f"key_{i}", policy_hash, execution_plan)

        assert cache.size == 5

        cache.clear()

        assert cache.size == 0
        for i in range(5):
            assert cache.get(f"key_{i}", policy_hash) is None


class TestSelectionCacheThreadSafety:
    """Test thread safety."""

    def test_concurrent_put_get(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Test concurrent put and get operations."""
        import threading
        import random

        cache = SelectionCache(max_size=100)
        policy_hash = "policy_v1"
        errors: list[Exception] = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(50):
                    key = f"key_{worker_id}_{i}"
                    cache.put(key, policy_hash, execution_plan)
                    cache.get(key, policy_hash)
                    # Random access to other keys
                    other_key = f"key_{random.randint(0, 3)}_{random.randint(0, 49)}"
                    cache.get(other_key, policy_hash)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.size <= 100

    def test_concurrent_invalidation(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Test concurrent invalidation operations."""
        import threading

        cache = SelectionCache()
        errors: list[Exception] = []

        def writer(policy: str) -> None:
            try:
                for i in range(50):
                    cache.put(f"key_{policy}_{i}", policy, execution_plan)
            except Exception as e:
                errors.append(e)

        def invalidator(policy: str) -> None:
            try:
                for _ in range(10):
                    cache.invalidate(policy)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("policy_a",)),
            threading.Thread(target=writer, args=("policy_b",)),
            threading.Thread(target=invalidator, args=("policy_a",)),
            threading.Thread(target=invalidator, args=("policy_b",)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestSelectionCacheStats:
    """Test cache statistics."""

    def test_size_property(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Test size property reflects current entries."""
        cache = SelectionCache()
        assert cache.size == 0

        cache.put("key_1", "hash", execution_plan)
        assert cache.size == 1

        cache.put("key_2", "hash", execution_plan)
        assert cache.size == 2

        cache.clear()
        assert cache.size == 0

    def test_hit_rate_initially_zero(self) -> None:
        """Test hit rate is 0 with no accesses."""
        cache = SelectionCache()
        assert cache.hit_rate == 0.0

    def test_hit_rate_tracking(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Test hit rate calculation."""
        cache = SelectionCache()
        policy_hash = "policy_v1"

        cache.put("key_1", policy_hash, execution_plan)

        # 2 hits
        cache.get("key_1", policy_hash)
        cache.get("key_1", policy_hash)

        # 2 misses
        cache.get("nonexistent", policy_hash)
        cache.get("also_nonexistent", policy_hash)

        # 2 hits / 4 total = 0.5
        assert cache.hit_rate == 0.5

    def test_reset_stats(
        self,
        execution_plan: ExecutionPlan,
    ) -> None:
        """Test resetting statistics."""
        cache = SelectionCache()
        policy_hash = "policy_v1"

        cache.put("key_1", policy_hash, execution_plan)
        cache.get("key_1", policy_hash)  # hit
        cache.get("nonexistent", policy_hash)  # miss

        assert cache.hit_rate == 0.5

        cache.reset_stats()

        assert cache.hit_rate == 0.0
