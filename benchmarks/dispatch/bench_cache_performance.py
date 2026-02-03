#!/usr/bin/env python3
"""
Cache Performance Benchmark

Measures cache performance characteristics:
- LRU cache hit/miss latency
- MVCC sharded cache hit/miss latency
- Cache invalidation overhead
- Shard distribution analysis
- Thundering herd prevention overhead

Target Performance:
- Cache lookup time: <1us for hits
- Cache miss time: <2us
- Invalidation: O(1) for version bump
- Shard contention: minimal with 256 shards

Methodology:
1. Create caches with various configurations
2. Pre-populate with realistic data
3. Warm up with 5 iterations (discarded)
4. Run 1000+ iterations per operation
5. Measure total time and compute statistics
6. Report p50, p95, p99 latencies
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import statistics
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    warmup_iterations: int = 5
    benchmark_iterations: int = 10000
    cache_size: int = 10000
    num_shards: int = 256
    num_threads: int = 4
    output_json: Path | None = None


@dataclass(slots=True)
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    iterations: int
    latencies_ns: list[int] = field(default_factory=list)
    hit_rate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_ns(self) -> float:
        return statistics.mean(self.latencies_ns) if self.latencies_ns else 0.0

    @property
    def std_ns(self) -> float:
        return statistics.stdev(self.latencies_ns) if len(self.latencies_ns) > 1 else 0.0

    @property
    def min_ns(self) -> int:
        return min(self.latencies_ns) if self.latencies_ns else 0

    @property
    def max_ns(self) -> int:
        return max(self.latencies_ns) if self.latencies_ns else 0

    @property
    def p50_ns(self) -> float:
        return statistics.median(self.latencies_ns) if self.latencies_ns else 0.0

    @property
    def p95_ns(self) -> float:
        if not self.latencies_ns:
            return 0.0
        sorted_latencies = sorted(self.latencies_ns)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p99_ns(self) -> float:
        if not self.latencies_ns:
            return 0.0
        sorted_latencies = sorted(self.latencies_ns)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "mean_ns": self.mean_ns,
            "std_ns": self.std_ns,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "p50_ns": self.p50_ns,
            "p95_ns": self.p95_ns,
            "p99_ns": self.p99_ns,
            "mean_us": self.mean_ns / 1000,
            "p50_us": self.p50_ns / 1000,
            "p95_us": self.p95_ns / 1000,
            "p99_us": self.p99_ns / 1000,
            "hit_rate": self.hit_rate,
            "metadata": self.metadata,
        }

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"Benchmark: {self.name}")
        print(f"{'='*60}")
        print(f"  Iterations:     {self.iterations:,}")
        print(f"  Mean:           {self.mean_ns:.1f} ns ({self.mean_ns/1000:.3f} us)")
        print(f"  Std Dev:        {self.std_ns:.1f} ns")
        print(f"  Min:            {self.min_ns} ns")
        print(f"  Max:            {self.max_ns} ns")
        print(f"  P50 (Median):   {self.p50_ns:.1f} ns")
        print(f"  P95:            {self.p95_ns:.1f} ns")
        print(f"  P99:            {self.p99_ns:.1f} ns")
        if self.hit_rate > 0:
            print(f"  Hit Rate:       {self.hit_rate*100:.1f}%")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")


def benchmark_selection_cache_hit(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark SelectionCache hit latency.

    Measures the time to lookup a cached entry that exists.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.selection.cache import SelectionCache
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec
    except ImportError:
        return _benchmark_mock_lru_cache_hit(config)

    # Setup
    cache = SelectionCache(max_size=config.cache_size)
    policy_hash = "test_policy_v1"

    # Create mock execution plan
    mock_spec = KernelSpec(
        kernel_id="flash_attn_v2",
        operation="attention.causal",
        source="flash_attn",
        version="2.0.0",
        priority=100,
        impl=lambda **kwargs: None,
    )

    mock_plan = ExecutionPlan(
        kernel_id=mock_spec.kernel_id,
        kernel_spec=mock_spec,
        cached=False,
    )

    # Pre-populate cache
    test_keys = [f"ctx_key_{i}" for i in range(min(1000, config.cache_size))]
    for key in test_keys:
        cache.put(key, policy_hash, mock_plan)

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = cache.get(test_keys[0], policy_hash)

    # Reset stats
    cache.reset_stats()

    # Benchmark - only hit existing keys
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        key = test_keys[i % len(test_keys)]
        start = time.perf_counter_ns()
        _ = cache.get(key, policy_hash)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    stats = cache.stats()

    return BenchmarkResult(
        name="SelectionCache Hit Latency",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        hit_rate=stats.get("hit_rate", 0.0),
        metadata={"cache_size": cache.size},
    )


def _benchmark_mock_lru_cache_hit(config: BenchmarkConfig) -> BenchmarkResult:
    """Mock LRU cache hit benchmark."""
    from threading import RLock

    cache: OrderedDict[str, str] = OrderedDict()
    policy_hashes: dict[str, str] = {}
    lock = RLock()
    hits = 0
    misses = 0
    policy_hash = "test_policy_v1"

    # Pre-populate
    test_keys = [f"ctx_key_{i}" for i in range(min(1000, config.cache_size))]
    for key in test_keys:
        cache[key] = f"plan_{key}"
        policy_hashes[key] = policy_hash

    def get(key: str, ph: str) -> str | None:
        nonlocal hits, misses
        with lock:
            if key not in cache:
                misses += 1
                return None
            if policy_hashes.get(key) != ph:
                misses += 1
                return None
            cache.move_to_end(key)
            hits += 1
            return cache[key]

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = get(test_keys[0], policy_hash)

    # Reset
    hits = 0
    misses = 0

    # Benchmark
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        key = test_keys[i % len(test_keys)]
        start = time.perf_counter_ns()
        _ = get(key, policy_hash)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    total = hits + misses
    hit_rate = hits / total if total > 0 else 0.0

    return BenchmarkResult(
        name="SelectionCache Hit Latency [Mock]",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        hit_rate=hit_rate,
    )


def benchmark_selection_cache_miss(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark SelectionCache miss latency.

    Measures the time to lookup a key that doesn't exist.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.selection.cache import SelectionCache
    except ImportError:
        return _benchmark_mock_lru_cache_miss(config)

    # Setup
    cache = SelectionCache(max_size=config.cache_size)
    policy_hash = "test_policy_v1"

    # Don't populate - we want misses
    test_keys = [f"missing_key_{i}" for i in range(1000)]

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = cache.get(test_keys[0], policy_hash)

    # Reset stats
    cache.reset_stats()

    # Benchmark - all misses
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        key = test_keys[i % len(test_keys)]
        start = time.perf_counter_ns()
        _ = cache.get(key, policy_hash)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="SelectionCache Miss Latency",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        hit_rate=0.0,
    )


def _benchmark_mock_lru_cache_miss(config: BenchmarkConfig) -> BenchmarkResult:
    """Mock LRU cache miss benchmark."""
    from threading import RLock

    cache: OrderedDict[str, str] = OrderedDict()
    lock = RLock()
    policy_hash = "test_policy_v1"

    test_keys = [f"missing_key_{i}" for i in range(1000)]

    def get(key: str, ph: str) -> str | None:
        with lock:
            if key not in cache:
                return None
            return cache[key]

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = get(test_keys[0], policy_hash)

    # Benchmark
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        key = test_keys[i % len(test_keys)]
        start = time.perf_counter_ns()
        _ = get(key, policy_hash)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="SelectionCache Miss Latency [Mock]",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        hit_rate=0.0,
    )


def benchmark_mvcc_cache_hit(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark MVCC sharded cache hit latency.

    Measures lookup time with MD5-based sharding.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.selection.mvcc_cache import MVCCShardedCache
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec
    except ImportError:
        return _benchmark_mock_mvcc_cache_hit(config)

    # Setup
    cache = MVCCShardedCache(
        num_shards=config.num_shards,
        max_entries_per_shard=config.cache_size // config.num_shards + 1,
        ttl_seconds=3600.0,
    )
    policy_hash = "test_policy_v1"

    # Create mock execution plan
    mock_spec = KernelSpec(
        kernel_id="flash_attn_v2",
        operation="attention.causal",
        source="flash_attn",
        version="2.0.0",
        priority=100,
        impl=lambda **kwargs: None,
    )

    mock_plan = ExecutionPlan(
        kernel_id=mock_spec.kernel_id,
        kernel_spec=mock_spec,
        cached=False,
    )

    # Pre-populate cache
    test_keys = [f"ctx_key_{i}" for i in range(min(10000, config.cache_size))]
    for key in test_keys:
        cache.put(key, policy_hash, mock_plan)

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = cache.get(test_keys[0], policy_hash)

    # Reset stats
    cache.reset_stats()

    # Benchmark
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        key = test_keys[i % len(test_keys)]
        start = time.perf_counter_ns()
        _ = cache.get(key, policy_hash)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    stats = cache.stats()

    return BenchmarkResult(
        name="MVCC Sharded Cache Hit Latency",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        hit_rate=stats.get("hit_rate", 0.0),
        metadata={
            "num_shards": config.num_shards,
            "cache_size": stats.get("size", 0),
        },
    )


def _benchmark_mock_mvcc_cache_hit(config: BenchmarkConfig) -> BenchmarkResult:
    """Mock MVCC sharded cache hit benchmark."""
    from threading import RLock

    num_shards = config.num_shards
    shards: list[dict[str, str]] = [{} for _ in range(num_shards)]
    shard_locks: list[RLock] = [RLock() for _ in range(num_shards)]
    policy_hash = "test_policy_v1"
    hits = 0
    misses = 0

    def get_shard_idx(key: str) -> int:
        h = hashlib.md5(key.encode(), usedforsecurity=False).digest()
        return int.from_bytes(h[:8], byteorder="little") % num_shards

    # Pre-populate
    test_keys = [f"ctx_key_{i}" for i in range(min(10000, config.cache_size))]
    for key in test_keys:
        idx = get_shard_idx(key)
        shards[idx][key] = f"plan_{key}"

    def get(key: str) -> str | None:
        nonlocal hits, misses
        idx = get_shard_idx(key)
        with shard_locks[idx]:
            result = shards[idx].get(key)
            if result:
                hits += 1
            else:
                misses += 1
            return result

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = get(test_keys[0])

    # Reset
    hits = 0
    misses = 0

    # Benchmark
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        key = test_keys[i % len(test_keys)]
        start = time.perf_counter_ns()
        _ = get(key)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    total = hits + misses
    hit_rate = hits / total if total > 0 else 0.0

    return BenchmarkResult(
        name="MVCC Sharded Cache Hit Latency [Mock]",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        hit_rate=hit_rate,
        metadata={"num_shards": num_shards},
    )


def benchmark_mvcc_cache_miss(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark MVCC sharded cache miss latency.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.selection.mvcc_cache import MVCCShardedCache
    except ImportError:
        return _benchmark_mock_mvcc_cache_miss(config)

    # Setup
    cache = MVCCShardedCache(
        num_shards=config.num_shards,
        max_entries_per_shard=100,
        ttl_seconds=3600.0,
    )
    policy_hash = "test_policy_v1"

    # Don't populate - we want misses
    test_keys = [f"missing_key_{i}" for i in range(1000)]

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = cache.get(test_keys[0], policy_hash)

    # Reset stats
    cache.reset_stats()

    # Benchmark
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        key = test_keys[i % len(test_keys)]
        start = time.perf_counter_ns()
        _ = cache.get(key, policy_hash)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="MVCC Sharded Cache Miss Latency",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        hit_rate=0.0,
    )


def _benchmark_mock_mvcc_cache_miss(config: BenchmarkConfig) -> BenchmarkResult:
    """Mock MVCC cache miss benchmark."""
    from threading import RLock

    num_shards = config.num_shards
    shards: list[dict[str, str]] = [{} for _ in range(num_shards)]
    shard_locks: list[RLock] = [RLock() for _ in range(num_shards)]

    def get_shard_idx(key: str) -> int:
        h = hashlib.md5(key.encode(), usedforsecurity=False).digest()
        return int.from_bytes(h[:8], byteorder="little") % num_shards

    test_keys = [f"missing_key_{i}" for i in range(1000)]

    def get(key: str) -> str | None:
        idx = get_shard_idx(key)
        with shard_locks[idx]:
            return shards[idx].get(key)

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = get(test_keys[0])

    # Benchmark
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        key = test_keys[i % len(test_keys)]
        start = time.perf_counter_ns()
        _ = get(key)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="MVCC Sharded Cache Miss Latency [Mock]",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        hit_rate=0.0,
    )


def benchmark_mvcc_invalidation(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark MVCC cache invalidation overhead.

    Measures the time for O(1) version bump invalidation.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.selection.mvcc_cache import MVCCShardedCache
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec
    except ImportError:
        return _benchmark_mock_mvcc_invalidation(config)

    # Setup
    cache = MVCCShardedCache(
        num_shards=config.num_shards,
        max_entries_per_shard=100,
        ttl_seconds=3600.0,
    )

    # Warmup
    for _ in range(config.warmup_iterations):
        cache.invalidate_shard(0)

    # Benchmark - single shard invalidation
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        shard_idx = i % config.num_shards
        start = time.perf_counter_ns()
        cache.invalidate_shard(shard_idx)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="MVCC Shard Invalidation (O(1))",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        metadata={"operation": "invalidate_shard"},
    )


def _benchmark_mock_mvcc_invalidation(config: BenchmarkConfig) -> BenchmarkResult:
    """Mock MVCC invalidation benchmark."""
    from threading import RLock

    num_shards = config.num_shards
    shard_versions: list[int] = [0] * num_shards
    shard_locks: list[RLock] = [RLock() for _ in range(num_shards)]

    def invalidate_shard(idx: int) -> None:
        with shard_locks[idx]:
            shard_versions[idx] += 1

    # Warmup
    for _ in range(config.warmup_iterations):
        invalidate_shard(0)

    # Benchmark
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        shard_idx = i % num_shards
        start = time.perf_counter_ns()
        invalidate_shard(shard_idx)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="MVCC Shard Invalidation (O(1)) [Mock]",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
    )


def benchmark_mvcc_full_invalidation(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark MVCC cache full invalidation.

    Measures the time to invalidate all shards (O(num_shards)).

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.selection.mvcc_cache import MVCCShardedCache
    except ImportError:
        return _benchmark_mock_mvcc_full_invalidation(config)

    # Setup
    cache = MVCCShardedCache(
        num_shards=config.num_shards,
        max_entries_per_shard=100,
        ttl_seconds=3600.0,
    )

    # Warmup
    for _ in range(config.warmup_iterations):
        cache.invalidate_all()

    # Benchmark - fewer iterations since this is more expensive
    latencies: list[int] = []
    num_iterations = min(config.benchmark_iterations, 1000)

    for _ in range(num_iterations):
        start = time.perf_counter_ns()
        cache.invalidate_all()
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name=f"MVCC Full Invalidation ({config.num_shards} shards)",
        iterations=num_iterations,
        latencies_ns=latencies,
        metadata={"num_shards": config.num_shards},
    )


def _benchmark_mock_mvcc_full_invalidation(config: BenchmarkConfig) -> BenchmarkResult:
    """Mock MVCC full invalidation benchmark."""
    from threading import RLock

    num_shards = config.num_shards
    shard_versions: list[int] = [0] * num_shards
    shard_locks: list[RLock] = [RLock() for _ in range(num_shards)]

    def invalidate_all() -> None:
        for i in range(num_shards):
            with shard_locks[i]:
                shard_versions[i] += 1

    # Warmup
    for _ in range(config.warmup_iterations):
        invalidate_all()

    # Benchmark
    latencies: list[int] = []
    num_iterations = min(config.benchmark_iterations, 1000)

    for _ in range(num_iterations):
        start = time.perf_counter_ns()
        invalidate_all()
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name=f"MVCC Full Invalidation ({num_shards} shards) [Mock]",
        iterations=num_iterations,
        latencies_ns=latencies,
    )


def benchmark_cache_concurrent_access(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark cache performance under concurrent access.

    Measures contention and throughput with multiple threads.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.selection.mvcc_cache import MVCCShardedCache
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec
    except ImportError:
        return _benchmark_mock_cache_concurrent(config)

    # Setup
    cache = MVCCShardedCache(
        num_shards=config.num_shards,
        max_entries_per_shard=100,
        ttl_seconds=3600.0,
    )
    policy_hash = "test_policy_v1"

    # Create mock execution plan
    mock_spec = KernelSpec(
        kernel_id="flash_attn_v2",
        operation="attention.causal",
        source="flash_attn",
        version="2.0.0",
        priority=100,
        impl=lambda **kwargs: None,
    )

    mock_plan = ExecutionPlan(
        kernel_id=mock_spec.kernel_id,
        kernel_spec=mock_spec,
        cached=False,
    )

    # Pre-populate
    test_keys = [f"ctx_key_{i}" for i in range(1000)]
    for key in test_keys:
        cache.put(key, policy_hash, mock_plan)

    all_latencies: list[int] = []
    latencies_lock = threading.Lock()
    iterations_per_thread = config.benchmark_iterations // config.num_threads

    def worker() -> None:
        thread_latencies: list[int] = []
        for i in range(iterations_per_thread):
            key = test_keys[i % len(test_keys)]
            start = time.perf_counter_ns()
            result = cache.get(key, policy_hash)
            if result is None and i % 10 == 0:
                cache.put(key, policy_hash, mock_plan)
            end = time.perf_counter_ns()
            thread_latencies.append(end - start)

        with latencies_lock:
            all_latencies.extend(thread_latencies)

    # Warmup
    cache.reset_stats()

    # Run concurrent benchmark
    threads = [
        threading.Thread(target=worker)
        for _ in range(config.num_threads)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = cache.stats()

    return BenchmarkResult(
        name=f"MVCC Cache Concurrent Access ({config.num_threads} threads)",
        iterations=len(all_latencies),
        latencies_ns=all_latencies,
        hit_rate=stats.get("hit_rate", 0.0),
        metadata={
            "num_threads": config.num_threads,
            "num_shards": config.num_shards,
        },
    )


def _benchmark_mock_cache_concurrent(config: BenchmarkConfig) -> BenchmarkResult:
    """Mock concurrent cache benchmark."""
    from threading import RLock

    num_shards = config.num_shards
    shards: list[dict[str, str]] = [{} for _ in range(num_shards)]
    shard_locks: list[RLock] = [RLock() for _ in range(num_shards)]
    hits = 0
    misses = 0
    stats_lock = threading.Lock()

    def get_shard_idx(key: str) -> int:
        h = hashlib.md5(key.encode(), usedforsecurity=False).digest()
        return int.from_bytes(h[:8], byteorder="little") % num_shards

    # Pre-populate
    test_keys = [f"ctx_key_{i}" for i in range(1000)]
    for key in test_keys:
        idx = get_shard_idx(key)
        shards[idx][key] = f"plan_{key}"

    def get(key: str) -> str | None:
        nonlocal hits, misses
        idx = get_shard_idx(key)
        with shard_locks[idx]:
            result = shards[idx].get(key)
        with stats_lock:
            if result:
                hits += 1
            else:
                misses += 1
        return result

    def put(key: str, value: str) -> None:
        idx = get_shard_idx(key)
        with shard_locks[idx]:
            shards[idx][key] = value

    all_latencies: list[int] = []
    latencies_lock = threading.Lock()
    iterations_per_thread = config.benchmark_iterations // config.num_threads

    def worker() -> None:
        thread_latencies: list[int] = []
        for i in range(iterations_per_thread):
            key = test_keys[i % len(test_keys)]
            start = time.perf_counter_ns()
            result = get(key)
            if result is None and i % 10 == 0:
                put(key, f"plan_{key}")
            end = time.perf_counter_ns()
            thread_latencies.append(end - start)

        with latencies_lock:
            all_latencies.extend(thread_latencies)

    # Run
    threads = [
        threading.Thread(target=worker)
        for _ in range(config.num_threads)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = hits + misses
    hit_rate = hits / total if total > 0 else 0.0

    return BenchmarkResult(
        name=f"MVCC Cache Concurrent Access ({config.num_threads} threads) [Mock]",
        iterations=len(all_latencies),
        latencies_ns=all_latencies,
        hit_rate=hit_rate,
        metadata={"num_threads": config.num_threads},
    )


def benchmark_shard_distribution(config: BenchmarkConfig) -> BenchmarkResult:
    """Analyze shard distribution quality.

    Measures how evenly keys are distributed across shards.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with distribution analysis.
    """
    num_shards = config.num_shards
    num_keys = 10000
    shard_counts: list[int] = [0] * num_shards

    def get_shard_idx(key: str) -> int:
        h = hashlib.md5(key.encode(), usedforsecurity=False).digest()
        return int.from_bytes(h[:8], byteorder="little") % num_shards

    # Measure shard assignment time
    latencies: list[int] = []
    test_keys = [f"ctx_key_{i}_{random.random()}" for i in range(num_keys)]

    for key in test_keys:
        start = time.perf_counter_ns()
        idx = get_shard_idx(key)
        end = time.perf_counter_ns()
        latencies.append(end - start)
        shard_counts[idx] += 1

    # Analyze distribution
    mean_count = num_keys / num_shards
    max_deviation = max(abs(c - mean_count) for c in shard_counts)
    std_deviation = statistics.stdev(shard_counts)

    return BenchmarkResult(
        name="Shard Distribution Analysis",
        iterations=num_keys,
        latencies_ns=latencies,
        metadata={
            "num_shards": num_shards,
            "num_keys": num_keys,
            "expected_per_shard": mean_count,
            "max_deviation": max_deviation,
            "std_deviation": std_deviation,
            "max_shard_count": max(shard_counts),
            "min_shard_count": min(shard_counts),
        },
    )


def run_all_cache_benchmarks(
    config: BenchmarkConfig | None = None,
) -> list[BenchmarkResult]:
    """Run all cache performance benchmarks.

    Args:
        config: Benchmark configuration.

    Returns:
        List of BenchmarkResult objects.
    """
    if config is None:
        config = BenchmarkConfig()

    print("\n" + "="*70)
    print("CACHE PERFORMANCE BENCHMARKS")
    print("="*70)
    print(f"Warmup iterations: {config.warmup_iterations}")
    print(f"Benchmark iterations: {config.benchmark_iterations:,}")
    print(f"Cache size: {config.cache_size:,}")
    print(f"Number of shards: {config.num_shards}")
    print(f"Concurrent threads: {config.num_threads}")
    print("="*70)

    results: list[BenchmarkResult] = []

    # SelectionCache benchmarks
    print("\n[1/9] Running SelectionCache hit benchmark...")
    result = benchmark_selection_cache_hit(config)
    results.append(result)
    result.print_summary()

    print("\n[2/9] Running SelectionCache miss benchmark...")
    result = benchmark_selection_cache_miss(config)
    results.append(result)
    result.print_summary()

    # MVCC cache benchmarks
    print("\n[3/9] Running MVCC cache hit benchmark...")
    result = benchmark_mvcc_cache_hit(config)
    results.append(result)
    result.print_summary()

    print("\n[4/9] Running MVCC cache miss benchmark...")
    result = benchmark_mvcc_cache_miss(config)
    results.append(result)
    result.print_summary()

    # Invalidation benchmarks
    print("\n[5/9] Running MVCC shard invalidation benchmark...")
    result = benchmark_mvcc_invalidation(config)
    results.append(result)
    result.print_summary()

    print("\n[6/9] Running MVCC full invalidation benchmark...")
    result = benchmark_mvcc_full_invalidation(config)
    results.append(result)
    result.print_summary()

    # Concurrent access
    print("\n[7/9] Running concurrent access benchmark...")
    result = benchmark_cache_concurrent_access(config)
    results.append(result)
    result.print_summary()

    # Shard distribution
    print("\n[8/9] Running shard distribution analysis...")
    result = benchmark_shard_distribution(config)
    results.append(result)
    result.print_summary()

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Benchmark':<50} {'P50 (us)':<10} {'P99 (us)':<10} {'Hit Rate':<10}")
    print("-"*70)
    for r in results:
        hit_rate = f"{r.hit_rate*100:.1f}%" if r.hit_rate > 0 else "N/A"
        print(f"{r.name:<50} {r.p50_ns/1000:<10.3f} {r.p99_ns/1000:<10.3f} {hit_rate:<10}")

    # Target validation
    print("\n" + "="*70)
    print("TARGET VALIDATION")
    print("="*70)

    targets = [
        ("SelectionCache Hit", 1000, "P99 < 1us"),
        ("MVCC Sharded Cache Hit", 1000, "P99 < 1us"),
        ("MVCC Shard Invalidation", 500, "P99 < 500ns"),
    ]

    for name, target_ns, description in targets:
        for r in results:
            if name in r.name:
                status = "PASS" if r.p99_ns < target_ns else "FAIL"
                actual = f"{r.p99_ns:.1f}ns"
                target = f"{target_ns}ns"
                print(f"  [{status}] {description}: actual={actual}, target<{target}")
                break

    # Save to JSON
    if config.output_json:
        output_data = {
            "benchmark": "cache_performance",
            "config": {
                "warmup_iterations": config.warmup_iterations,
                "benchmark_iterations": config.benchmark_iterations,
                "cache_size": config.cache_size,
                "num_shards": config.num_shards,
                "num_threads": config.num_threads,
            },
            "results": [r.to_dict() for r in results],
        }
        with open(config.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {config.output_json}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cache Performance Benchmark")
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of benchmark iterations (default: 10000)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=10000,
        help="Cache size (default: 10000)",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=256,
        help="Number of MVCC shards (default: 256)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for concurrent tests (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    benchmark_config = BenchmarkConfig(
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        cache_size=args.cache_size,
        num_shards=args.shards,
        num_threads=args.threads,
        output_json=Path(args.output) if args.output else None,
    )

    run_all_cache_benchmarks(benchmark_config)
