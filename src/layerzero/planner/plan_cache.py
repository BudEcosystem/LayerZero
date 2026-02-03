"""Plan cache for multi-operation planner.

This module provides caching for execution plans to avoid
recomputing plans for frequently seen operation sequences.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from layerzero.planner.multi_op import MultiOpPlan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for the plan cache.

    Attributes:
        max_entries: Maximum number of cached plans.
        ttl_seconds: Time-to-live for cache entries in seconds.
        enable_lru: Use LRU eviction when cache is full.
    """

    max_entries: int = 1000
    ttl_seconds: float = 3600.0  # 1 hour
    enable_lru: bool = True


@dataclass
class PlanCacheEntry:
    """A cached plan entry.

    Attributes:
        plan: The cached multi-op plan.
        created_at: Timestamp when entry was created.
        last_accessed: Timestamp of last access.
        access_count: Number of times entry was accessed.
    """

    plan: MultiOpPlan
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    access_count: int = 0

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if entry has expired."""
        return time.monotonic() - self.created_at > ttl_seconds

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.monotonic()
        self.access_count += 1


class PlanCache:
    """Cache for multi-operation plans.

    This cache stores computed plans keyed by a hash of the
    operation sequence. It supports TTL-based expiration and
    LRU eviction when the cache is full.
    """

    def __init__(
        self,
        config: CacheConfig | None = None,
    ) -> None:
        """Initialize the plan cache.

        Args:
            config: Cache configuration. Uses defaults if None.
        """
        self._config = config or CacheConfig()
        self._cache: dict[str, PlanCacheEntry] = {}
        self._lock = RLock()

        logger.debug(
            "PlanCache initialized with max_entries=%d, ttl=%.1f",
            self._config.max_entries,
            self._config.ttl_seconds,
        )

    @property
    def config(self) -> CacheConfig:
        """Get cache configuration."""
        return self._config

    def get(self, operations: list[dict[str, Any]]) -> MultiOpPlan | None:
        """Get a cached plan for operations.

        Args:
            operations: Operation sequence to look up.

        Returns:
            Cached plan if found and not expired, None otherwise.
        """
        key = self._compute_key(operations)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                logger.debug("Cache miss for key: %s", key[:16])
                return None

            if entry.is_expired(self._config.ttl_seconds):
                logger.debug("Cache entry expired for key: %s", key[:16])
                del self._cache[key]
                return None

            entry.touch()
            logger.debug(
                "Cache hit for key: %s (access count: %d)",
                key[:16],
                entry.access_count,
            )
            return entry.plan

    def put(
        self,
        operations: list[dict[str, Any]],
        plan: MultiOpPlan,
    ) -> None:
        """Store a plan in the cache.

        Args:
            operations: Operation sequence as key.
            plan: Plan to cache.
        """
        key = self._compute_key(operations)

        with self._lock:
            # Evict if necessary
            if len(self._cache) >= self._config.max_entries:
                self._evict()

            self._cache[key] = PlanCacheEntry(plan=plan)
            logger.debug("Cached plan for key: %s", key[:16])

    def invalidate(self, operations: list[dict[str, Any]]) -> bool:
        """Invalidate a cached plan.

        Args:
            operations: Operation sequence to invalidate.

        Returns:
            True if entry was removed, False if not found.
        """
        key = self._compute_key(operations)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug("Invalidated cache for key: %s", key[:16])
                return True
            return False

    def clear(self) -> int:
        """Clear all cached plans.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.debug("Cleared %d cache entries", count)
            return count

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        with self._lock:
            total_accesses = sum(
                entry.access_count for entry in self._cache.values()
            )
            return {
                "size": len(self._cache),
                "max_entries": self._config.max_entries,
                "total_accesses": total_accesses,
                "ttl_seconds": self._config.ttl_seconds,
            }

    def _compute_key(self, operations: list[dict[str, Any]]) -> str:
        """Compute cache key for operations.

        Args:
            operations: Operation sequence.

        Returns:
            SHA256 hash of the operation sequence.
        """
        # Create a stable string representation
        # Sort keys to ensure deterministic ordering
        normalized = []
        for op in operations:
            normalized_op = {
                k: v
                for k, v in sorted(op.items())
                if k
                in (
                    "op_type",
                    "input_layout",
                    "output_layout",
                    "input_dtype",
                    "output_dtype",
                )
            }
            normalized.append(normalized_op)

        data = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def _evict(self) -> None:
        """Evict entries to make room for new ones."""
        if not self._cache:
            return

        if self._config.enable_lru:
            # Evict least recently used entry
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].last_accessed,
            )
        else:
            # Evict oldest entry
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].created_at,
            )

        del self._cache[oldest_key]
        logger.debug("Evicted cache entry: %s", oldest_key[:16])
