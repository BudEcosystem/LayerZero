"""
LayerZero Selection Cache

Thread-safe LRU cache for kernel selection results.
Cache key includes policy hash for automatic invalidation.
"""
from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layerzero.models.execution_plan import ExecutionPlan


class SelectionCache:
    """Thread-safe LRU cache for kernel selections.

    Uses OrderedDict for O(1) LRU operations.
    Each entry includes policy hash for invalidation when policy changes.

    All public methods are thread-safe via RLock.

    Attributes:
        max_size: Maximum number of entries in cache.
    """

    __slots__ = (
        "_lock",
        "_cache",
        "_policy_hashes",
        "_max_size",
        "_hits",
        "_misses",
    )

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize selection cache.

        Args:
            max_size: Maximum number of cached entries. Must be positive.

        Raises:
            ValueError: If max_size is not positive.
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self._lock = RLock()
        self._cache: OrderedDict[str, "ExecutionPlan"] = OrderedDict()
        self._policy_hashes: dict[str, str] = {}  # key -> policy_hash
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    @property
    def max_size(self) -> int:
        """Get maximum cache size."""
        return self._max_size

    @property
    def size(self) -> int:
        """Get current number of cached entries."""
        with self._lock:
            return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate.

        Returns:
            Hit rate as float between 0.0 and 1.0.
            Returns 0.0 if no accesses have been made.
        """
        with self._lock:
            total = self._hits + self._misses
            if total == 0:
                return 0.0
            return self._hits / total

    def get(
        self,
        key: str,
        policy_hash: str,
    ) -> "ExecutionPlan | None":
        """Get cached selection if valid.

        Entry is valid only if:
        1. Key exists in cache
        2. Entry's policy hash matches the provided policy hash

        On hit, entry is moved to end (most recently used).

        Args:
            key: Cache key (from SelectionContext.cache_key()).
            policy_hash: Current policy hash for validation.

        Returns:
            ExecutionPlan if cache hit and valid, None otherwise.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            # Check policy hash matches
            if self._policy_hashes.get(key) != policy_hash:
                # Policy changed, entry is stale
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

    def put(
        self,
        key: str,
        policy_hash: str,
        plan: "ExecutionPlan",
    ) -> None:
        """Cache a selection result.

        If key already exists, it's updated and moved to end.
        If cache is full, oldest entry is evicted.

        Args:
            key: Cache key (from SelectionContext.cache_key()).
            policy_hash: Current policy hash for later validation.
            plan: Execution plan to cache.
        """
        with self._lock:
            # If key exists, remove it first (will be re-added at end)
            if key in self._cache:
                del self._cache[key]
                del self._policy_hashes[key]

            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._policy_hashes[oldest_key]

            # Add new entry at end (most recently used)
            self._cache[key] = plan
            self._policy_hashes[key] = policy_hash

    def invalidate(self, policy_hash: str) -> int:
        """Invalidate all entries for a specific policy hash.

        Args:
            policy_hash: Policy hash to invalidate entries for.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            # Find all keys with matching policy hash
            keys_to_remove = [
                key for key, ph in self._policy_hashes.items()
                if ph == policy_hash
            ]

            # Remove matching entries
            for key in keys_to_remove:
                del self._cache[key]
                del self._policy_hashes[key]

            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cached entries.

        Does not reset statistics.
        """
        with self._lock:
            self._cache.clear()
            self._policy_hashes.clear()

    def reset_stats(self) -> None:
        """Reset hit/miss statistics."""
        with self._lock:
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, float | int]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, size, max_size, and hit_rate.
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "max_size": self._max_size,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }
