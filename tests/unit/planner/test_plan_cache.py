"""Tests for plan cache."""
from __future__ import annotations

import pytest
import time
from unittest.mock import MagicMock, patch
from typing import Any

from layerzero.planner.plan_cache import (
    PlanCache,
    CacheConfig,
    PlanCacheEntry,
)
from layerzero.planner.multi_op import MultiOpPlan, OpPlan


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config values."""
        config = CacheConfig()

        assert config.max_entries == 1000
        assert config.ttl_seconds == 3600.0
        assert config.enable_lru is True

    def test_custom_values(self) -> None:
        """Custom config values."""
        config = CacheConfig(
            max_entries=500,
            ttl_seconds=1800.0,
            enable_lru=False,
        )

        assert config.max_entries == 500
        assert config.ttl_seconds == 1800.0
        assert config.enable_lru is False

    def test_config_immutable(self) -> None:
        """Config is immutable."""
        config = CacheConfig()

        with pytest.raises(AttributeError):
            config.max_entries = 100


class TestPlanCacheEntry:
    """Tests for PlanCacheEntry."""

    def test_creation(self) -> None:
        """Entry stores plan."""
        plan = MultiOpPlan(ops=[])
        entry = PlanCacheEntry(plan=plan)

        assert entry.plan is plan
        assert entry.access_count == 0

    def test_touch(self) -> None:
        """Touch updates access time and count."""
        plan = MultiOpPlan(ops=[])
        entry = PlanCacheEntry(plan=plan)

        initial_accessed = entry.last_accessed
        entry.touch()

        assert entry.access_count == 1
        assert entry.last_accessed >= initial_accessed

    def test_is_expired_false(self) -> None:
        """Entry not expired when within TTL."""
        plan = MultiOpPlan(ops=[])
        entry = PlanCacheEntry(plan=plan)

        assert entry.is_expired(3600.0) is False

    def test_is_expired_true(self) -> None:
        """Entry expired when beyond TTL."""
        plan = MultiOpPlan(ops=[])
        entry = PlanCacheEntry(plan=plan)

        # Simulate time passing by setting created_at in the past
        entry.created_at = time.monotonic() - 100

        assert entry.is_expired(50.0) is True


class TestPlanCache:
    """Tests for PlanCache."""

    def test_get_miss(self) -> None:
        """Cache miss returns None."""
        cache = PlanCache()
        ops = [{"op_type": "attention", "input_layout": "BSHD"}]

        result = cache.get(ops)

        assert result is None

    def test_put_and_get_hit(self) -> None:
        """Put then get returns cached plan."""
        cache = PlanCache()
        ops = [{"op_type": "attention", "input_layout": "BSHD"}]
        plan = MultiOpPlan(ops=[OpPlan("attention", "flash", "BSHD", "BSHD", 1.0)])

        cache.put(ops, plan)
        result = cache.get(ops)

        assert result is not None
        assert len(result.ops) == 1
        assert result.ops[0].op_type == "attention"

    def test_get_expired(self) -> None:
        """Expired entry returns None."""
        config = CacheConfig(ttl_seconds=0.001)  # Very short TTL
        cache = PlanCache(config=config)
        ops = [{"op_type": "attention", "input_layout": "BSHD"}]
        plan = MultiOpPlan(ops=[])

        cache.put(ops, plan)
        time.sleep(0.01)  # Wait for expiry
        result = cache.get(ops)

        assert result is None

    def test_invalidate_existing(self) -> None:
        """Invalidate removes existing entry."""
        cache = PlanCache()
        ops = [{"op_type": "attention", "input_layout": "BSHD"}]
        plan = MultiOpPlan(ops=[])

        cache.put(ops, plan)
        removed = cache.invalidate(ops)

        assert removed is True
        assert cache.get(ops) is None

    def test_invalidate_nonexistent(self) -> None:
        """Invalidate returns False for nonexistent entry."""
        cache = PlanCache()
        ops = [{"op_type": "attention", "input_layout": "BSHD"}]

        removed = cache.invalidate(ops)

        assert removed is False

    def test_clear(self) -> None:
        """Clear removes all entries."""
        cache = PlanCache()
        ops1 = [{"op_type": "attention", "input_layout": "BSHD"}]
        ops2 = [{"op_type": "layernorm", "input_layout": "BSHD"}]
        plan = MultiOpPlan(ops=[])

        cache.put(ops1, plan)
        cache.put(ops2, plan)
        count = cache.clear()

        assert count == 2
        assert cache.size() == 0

    def test_size(self) -> None:
        """Size returns entry count."""
        cache = PlanCache()
        ops1 = [{"op_type": "attention", "input_layout": "BSHD"}]
        ops2 = [{"op_type": "layernorm", "input_layout": "BSHD"}]
        plan = MultiOpPlan(ops=[])

        assert cache.size() == 0
        cache.put(ops1, plan)
        assert cache.size() == 1
        cache.put(ops2, plan)
        assert cache.size() == 2

    def test_stats(self) -> None:
        """Stats returns cache statistics."""
        cache = PlanCache()
        ops = [{"op_type": "attention", "input_layout": "BSHD"}]
        plan = MultiOpPlan(ops=[])

        cache.put(ops, plan)
        cache.get(ops)
        cache.get(ops)
        stats = cache.stats()

        assert stats["size"] == 1
        assert stats["max_entries"] == 1000
        assert stats["total_accesses"] == 2
        assert stats["ttl_seconds"] == 3600.0

    def test_eviction_lru(self) -> None:
        """LRU eviction removes least recently used."""
        config = CacheConfig(max_entries=2, enable_lru=True)
        cache = PlanCache(config=config)
        plan = MultiOpPlan(ops=[])

        ops1 = [{"op_type": "op1", "input_layout": "BSHD"}]
        ops2 = [{"op_type": "op2", "input_layout": "BSHD"}]
        ops3 = [{"op_type": "op3", "input_layout": "BSHD"}]

        cache.put(ops1, plan)
        cache.put(ops2, plan)

        # Access ops1 to make it more recently used
        cache.get(ops1)

        # Add ops3, should evict ops2 (LRU)
        cache.put(ops3, plan)

        assert cache.get(ops1) is not None  # Should still be there
        assert cache.get(ops2) is None  # Should be evicted
        assert cache.get(ops3) is not None  # Should be there

    def test_eviction_fifo(self) -> None:
        """FIFO eviction removes oldest."""
        config = CacheConfig(max_entries=2, enable_lru=False)
        cache = PlanCache(config=config)
        plan = MultiOpPlan(ops=[])

        ops1 = [{"op_type": "op1", "input_layout": "BSHD"}]
        ops2 = [{"op_type": "op2", "input_layout": "BSHD"}]
        ops3 = [{"op_type": "op3", "input_layout": "BSHD"}]

        cache.put(ops1, plan)
        time.sleep(0.001)
        cache.put(ops2, plan)

        # Access ops1 (doesn't matter for FIFO)
        cache.get(ops1)

        # Add ops3, should evict ops1 (oldest)
        cache.put(ops3, plan)

        assert cache.get(ops1) is None  # Should be evicted (oldest)
        assert cache.get(ops2) is not None  # Should still be there
        assert cache.get(ops3) is not None  # Should be there

    def test_deterministic_key(self) -> None:
        """Same operations produce same key."""
        cache = PlanCache()

        ops1 = [
            {"op_type": "attention", "input_layout": "BSHD", "output_layout": "BSHD"}
        ]
        ops2 = [
            {"output_layout": "BSHD", "op_type": "attention", "input_layout": "BSHD"}
        ]

        # Keys should be the same regardless of dict ordering
        plan = MultiOpPlan(ops=[])
        cache.put(ops1, plan)

        result = cache.get(ops2)
        assert result is not None

    def test_extra_fields_ignored(self) -> None:
        """Extra fields don't affect key."""
        cache = PlanCache()

        ops1 = [
            {"op_type": "attention", "input_layout": "BSHD", "extra": "ignored"}
        ]
        ops2 = [
            {"op_type": "attention", "input_layout": "BSHD"}
        ]

        # Keys should be the same since extra fields are ignored
        plan = MultiOpPlan(ops=[])
        cache.put(ops1, plan)

        result = cache.get(ops2)
        assert result is not None


class TestPlanCacheThreadSafety:
    """Tests for plan cache thread safety."""

    def test_concurrent_access(self) -> None:
        """Cache handles concurrent access."""
        import threading

        cache = PlanCache()
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(100):
                    ops = [{"op_type": f"op_{thread_id}_{i}", "input_layout": "BSHD"}]
                    plan = MultiOpPlan(ops=[])
                    cache.put(ops, plan)
                    cache.get(ops)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
