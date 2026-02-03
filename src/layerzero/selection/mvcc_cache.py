"""
LayerZero MVCC Sharded Selection Cache

High-performance cache with:
- 256 shards for minimal lock contention
- Per-shard MVCC versioning for O(1) invalidation
- Selection deduplication (thundering herd prevention)
- Bounded LRU with TTL per shard
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Event, RLock
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from layerzero.models.execution_plan import ExecutionPlan


@dataclass
class CacheEntry:
    """Single cache entry with version tracking and TTL.

    Attributes:
        plan: Cached execution plan.
        version: Shard version when entry was created.
        policy_hash: Policy hash for change detection.
        timestamp: Creation time (monotonic) for TTL.
    """

    plan: "ExecutionPlan"
    version: int
    policy_hash: str
    timestamp: float


@dataclass
class CacheShard:
    """Single shard of the sharded cache.

    Attributes:
        version: MVCC version counter (bump to invalidate all entries).
        entries: Key to entry mapping.
        lru_order: LRU tracking via OrderedDict.
        lock: Per-shard lock for thread safety.
        inflight: In-flight computations for deduplication.
    """

    version: int = 0
    entries: dict[str, CacheEntry] = field(default_factory=dict)
    lru_order: OrderedDict[str, None] = field(default_factory=OrderedDict)
    lock: RLock = field(default_factory=RLock)
    inflight: dict[str, Event] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0


class MVCCShardedCache:
    """High-performance MVCC sharded selection cache.

    Features:
    - Configurable number of shards (default 256) for minimal lock contention
    - Per-shard MVCC versioning enables O(1) invalidation
    - Selection deduplication prevents thundering herd problem
    - Bounded LRU with configurable TTL per shard

    Thread Safety:
    - Each shard has its own lock
    - Operations only lock the relevant shard
    - Lock-free reads are not implemented (would require more complexity)

    Usage:
        cache = MVCCShardedCache()

        # Simple get/put
        cache.put(key, policy_hash, plan)
        result = cache.get(key, policy_hash)

        # Deduplication (thundering herd prevention)
        result = cache.get_or_compute(key, policy_hash, compute_fn)

        # Invalidation
        cache.invalidate_shard(shard_idx)  # O(1)
        cache.invalidate_all()              # O(num_shards)
        cache.invalidate_policy(hash)       # O(entries)
    """

    __slots__ = (
        "_num_shards",
        "_max_entries_per_shard",
        "_ttl_seconds",
        "_shards",
    )

    def __init__(
        self,
        num_shards: int = 256,
        max_entries_per_shard: int = 100,
        ttl_seconds: float = 3600.0,
    ) -> None:
        """Initialize sharded cache.

        Args:
            num_shards: Number of shards (default 256). More shards = less contention.
            max_entries_per_shard: Max entries per shard before LRU eviction.
            ttl_seconds: Time-to-live for entries in seconds.

        Raises:
            ValueError: If any parameter is not positive.
        """
        if num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if max_entries_per_shard <= 0:
            raise ValueError("max_entries_per_shard must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        self._num_shards = num_shards
        self._max_entries_per_shard = max_entries_per_shard
        self._ttl_seconds = ttl_seconds
        self._shards = [CacheShard() for _ in range(num_shards)]

    @property
    def num_shards(self) -> int:
        """Get number of shards."""
        return self._num_shards

    @property
    def max_entries_per_shard(self) -> int:
        """Get max entries per shard."""
        return self._max_entries_per_shard

    @property
    def ttl_seconds(self) -> float:
        """Get TTL in seconds."""
        return self._ttl_seconds

    @property
    def size(self) -> int:
        """Get total entries across all shards."""
        total = 0
        for shard in self._shards:
            with shard.lock:
                total += len(shard.entries)
        return total

    def _get_shard_idx(self, key: str) -> int:
        """Get shard index for key.

        Uses MD5 hash truncated to int for fast, well-distributed sharding.
        MD5 is not cryptographically secure but is fast and has good distribution.

        Args:
            key: Cache key.

        Returns:
            Shard index (0 to num_shards - 1).
        """
        # Use MD5 for fast hash with good distribution
        h = hashlib.md5(key.encode(), usedforsecurity=False).digest()
        # Convert first 8 bytes to int
        hash_int = int.from_bytes(h[:8], byteorder="little")
        return hash_int % self._num_shards

    def _is_valid(
        self,
        entry: CacheEntry,
        shard: CacheShard,
        policy_hash: str,
    ) -> bool:
        """Check if entry is valid.

        Entry is valid if:
        1. Entry version >= shard version (not invalidated)
        2. Entry policy_hash matches current
        3. Entry not expired (TTL)

        Args:
            entry: Cache entry to check.
            shard: Shard containing entry.
            policy_hash: Current policy hash.

        Returns:
            True if entry is valid.
        """
        # Check version (MVCC invalidation)
        if entry.version < shard.version:
            return False

        # Check policy hash
        if entry.policy_hash != policy_hash:
            return False

        # Check TTL
        age = time.monotonic() - entry.timestamp
        if age > self._ttl_seconds:
            return False

        return True

    def get(
        self,
        key: str,
        policy_hash: str,
    ) -> "ExecutionPlan | None":
        """Get cached entry if valid.

        Entry is valid if:
        1. Key exists in shard
        2. Entry version >= shard version (not invalidated)
        3. Entry policy_hash matches current
        4. Entry not expired (TTL)

        On hit, entry is moved to end of LRU order.

        Args:
            key: Cache key.
            policy_hash: Current policy hash for validation.

        Returns:
            ExecutionPlan if valid cache hit, None otherwise.
        """
        shard_idx = self._get_shard_idx(key)
        shard = self._shards[shard_idx]

        with shard.lock:
            entry = shard.entries.get(key)

            if entry is None:
                shard.misses += 1
                return None

            if not self._is_valid(entry, shard, policy_hash):
                shard.misses += 1
                return None

            # Update LRU order
            shard.lru_order.move_to_end(key)
            shard.hits += 1
            return entry.plan

    def put(
        self,
        key: str,
        policy_hash: str,
        plan: "ExecutionPlan",
    ) -> None:
        """Cache an entry.

        If key already exists, it's updated and moved to end of LRU.
        If shard is full, oldest entry is evicted.

        Args:
            key: Cache key.
            policy_hash: Current policy hash.
            plan: Execution plan to cache.
        """
        shard_idx = self._get_shard_idx(key)
        shard = self._shards[shard_idx]

        with shard.lock:
            self._put_entry(shard, key, policy_hash, plan)

    def _put_entry(
        self,
        shard: CacheShard,
        key: str,
        policy_hash: str,
        plan: "ExecutionPlan",
    ) -> None:
        """Put entry with LRU eviction (caller must hold lock).

        Args:
            shard: Shard to put entry in.
            key: Cache key.
            policy_hash: Current policy hash.
            plan: Execution plan to cache.
        """
        # Remove existing entry (to update LRU order)
        if key in shard.entries:
            del shard.entries[key]
            del shard.lru_order[key]

        # Evict oldest if at capacity
        while len(shard.entries) >= self._max_entries_per_shard:
            oldest_key = next(iter(shard.lru_order))
            del shard.entries[oldest_key]
            del shard.lru_order[oldest_key]

        # Add new entry
        shard.entries[key] = CacheEntry(
            plan=plan,
            version=shard.version,
            policy_hash=policy_hash,
            timestamp=time.monotonic(),
        )
        shard.lru_order[key] = None

    def get_or_compute(
        self,
        key: str,
        policy_hash: str,
        compute_fn: Callable[[], "ExecutionPlan"],
    ) -> "ExecutionPlan":
        """Get cached or compute with deduplication.

        If key not in cache:
        - First caller computes
        - Other callers wait for result
        - Result is cached and returned to all

        This prevents the "thundering herd" problem where many threads
        simultaneously compute the same expensive result.

        Args:
            key: Cache key.
            policy_hash: Current policy hash.
            compute_fn: Function to compute result if not cached.

        Returns:
            Cached or computed ExecutionPlan.
        """
        shard_idx = self._get_shard_idx(key)
        shard = self._shards[shard_idx]

        # Fast path: check cache
        with shard.lock:
            entry = shard.entries.get(key)
            if entry is not None and self._is_valid(entry, shard, policy_hash):
                shard.lru_order.move_to_end(key)
                shard.hits += 1
                return entry.plan

            shard.misses += 1

            # Check if another thread is computing
            if key in shard.inflight:
                event = shard.inflight[key]
                is_waiter = True
            else:
                # First caller - create event
                event = Event()
                shard.inflight[key] = event
                is_waiter = False

        # If we're a waiter, wait for result
        if is_waiter:
            event.wait()
            # Try cache again after wait
            with shard.lock:
                entry = shard.entries.get(key)
                if entry is not None and self._is_valid(entry, shard, policy_hash):
                    shard.lru_order.move_to_end(key)
                    return entry.plan
            # If still not in cache, we need to compute ourselves
            # This can happen if the first caller failed
            return self.get_or_compute(key, policy_hash, compute_fn)

        # We're the first caller - compute
        try:
            plan = compute_fn()

            # Store result
            with shard.lock:
                self._put_entry(shard, key, policy_hash, plan)

            return plan
        finally:
            # Signal waiters
            with shard.lock:
                if key in shard.inflight:
                    shard.inflight[key].set()
                    del shard.inflight[key]

    def invalidate_shard(self, shard_idx: int) -> None:
        """Invalidate a shard via version bump (O(1)).

        All entries in the shard become invalid because their version
        is now less than the shard version.

        Args:
            shard_idx: Index of shard to invalidate (0 to num_shards - 1).
        """
        if not 0 <= shard_idx < self._num_shards:
            return

        shard = self._shards[shard_idx]
        with shard.lock:
            shard.version += 1

    def invalidate_all(self) -> None:
        """Invalidate all shards.

        Bumps version on all shards, making all entries invalid.
        """
        for shard in self._shards:
            with shard.lock:
                shard.version += 1

    def invalidate_policy(self, policy_hash: str) -> int:
        """Invalidate entries for specific policy hash.

        Unlike version bump, this actually removes entries.

        Args:
            policy_hash: Policy hash to invalidate.

        Returns:
            Number of entries removed.
        """
        removed = 0
        for shard in self._shards:
            with shard.lock:
                keys_to_remove = [
                    key for key, entry in shard.entries.items()
                    if entry.policy_hash == policy_hash
                ]
                for key in keys_to_remove:
                    del shard.entries[key]
                    del shard.lru_order[key]
                removed += len(keys_to_remove)
        return removed

    def clear(self) -> None:
        """Clear all entries from all shards."""
        for shard in self._shards:
            with shard.lock:
                shard.entries.clear()
                shard.lru_order.clear()
                shard.version += 1  # Bump version for safety

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with size, num_shards, hits, misses, hit_rate.
        """
        total_entries = 0
        total_hits = 0
        total_misses = 0

        for shard in self._shards:
            with shard.lock:
                total_entries += len(shard.entries)
                total_hits += shard.hits
                total_misses += shard.misses

        total = total_hits + total_misses
        hit_rate = total_hits / total if total > 0 else 0.0

        return {
            "size": total_entries,
            "num_shards": self._num_shards,
            "max_entries_per_shard": self._max_entries_per_shard,
            "ttl_seconds": self._ttl_seconds,
            "hits": total_hits,
            "misses": total_misses,
            "hit_rate": hit_rate,
        }

    def reset_stats(self) -> None:
        """Reset hit/miss statistics."""
        for shard in self._shards:
            with shard.lock:
                shard.hits = 0
                shard.misses = 0
