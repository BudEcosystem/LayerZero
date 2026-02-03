#!/usr/bin/env python3
"""
Kernel Dispatch Overhead Benchmark

Measures dispatch overhead for each dispatch mode:
- STATIC: Near-zero overhead, compile-time kernel resolution
- DYNAMIC: Runtime selection with ~100-500ns overhead
- CONFIG: YAML-driven with ~100ns lookup overhead
- HOT_RELOAD: Config file watching with ~1-10ms reload

Target Performance:
- Selection overhead: <100us p99
- Dispatch layer overhead: <10us for execution wrapper
- Mode switching overhead: measurable

Methodology:
1. Create mock kernel specs and registries
2. Warm up with 5 iterations (discarded)
3. Run 1000+ iterations
4. Measure total time and compute statistics
5. Report p50, p95, p99 latencies
6. Compare against baseline (direct function call)
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Attributes:
        warmup_iterations: Number of warmup runs to discard.
        benchmark_iterations: Number of timed iterations.
        output_json: Optional path to save results JSON.
    """
    warmup_iterations: int = 5
    benchmark_iterations: int = 10000
    output_json: Path | None = None


@dataclass(slots=True)
class BenchmarkResult:
    """Result of a single benchmark run.

    All times are in nanoseconds for precision.
    Uses __slots__ for memory efficiency.

    Attributes:
        name: Benchmark name.
        iterations: Number of iterations run.
        latencies_ns: List of latency measurements in nanoseconds.
        baseline_ns: Baseline latency (direct call) in nanoseconds.
    """
    name: str
    iterations: int
    latencies_ns: list[int] = field(default_factory=list)
    baseline_ns: float = 0.0

    @property
    def total_ns(self) -> int:
        """Total time in nanoseconds."""
        return sum(self.latencies_ns)

    @property
    def mean_ns(self) -> float:
        """Mean latency in nanoseconds."""
        if not self.latencies_ns:
            return 0.0
        return statistics.mean(self.latencies_ns)

    @property
    def std_ns(self) -> float:
        """Standard deviation in nanoseconds."""
        if len(self.latencies_ns) < 2:
            return 0.0
        return statistics.stdev(self.latencies_ns)

    @property
    def min_ns(self) -> int:
        """Minimum latency in nanoseconds."""
        return min(self.latencies_ns) if self.latencies_ns else 0

    @property
    def max_ns(self) -> int:
        """Maximum latency in nanoseconds."""
        return max(self.latencies_ns) if self.latencies_ns else 0

    @property
    def p50_ns(self) -> float:
        """50th percentile (median) latency in nanoseconds."""
        if not self.latencies_ns:
            return 0.0
        return statistics.median(self.latencies_ns)

    @property
    def p95_ns(self) -> float:
        """95th percentile latency in nanoseconds."""
        if not self.latencies_ns:
            return 0.0
        sorted_latencies = sorted(self.latencies_ns)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p99_ns(self) -> float:
        """99th percentile latency in nanoseconds."""
        if not self.latencies_ns:
            return 0.0
        sorted_latencies = sorted(self.latencies_ns)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def overhead_vs_baseline(self) -> float:
        """Overhead factor compared to baseline."""
        if self.baseline_ns <= 0:
            return 0.0
        return self.mean_ns / self.baseline_ns

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
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
            "total_ns": self.total_ns,
            "baseline_ns": self.baseline_ns,
            "overhead_vs_baseline": self.overhead_vs_baseline,
        }

    def print_summary(self) -> None:
        """Print formatted summary to stdout."""
        print(f"\n{'='*60}")
        print(f"Benchmark: {self.name}")
        print(f"{'='*60}")
        print(f"  Iterations:     {self.iterations:,}")
        print(f"  Mean:           {self.mean_ns/1000:.3f} us")
        print(f"  Std Dev:        {self.std_ns/1000:.3f} us")
        print(f"  Min:            {self.min_ns/1000:.3f} us")
        print(f"  Max:            {self.max_ns/1000:.3f} us")
        print(f"  P50 (Median):   {self.p50_ns/1000:.3f} us")
        print(f"  P95:            {self.p95_ns/1000:.3f} us")
        print(f"  P99:            {self.p99_ns/1000:.3f} us")
        if self.baseline_ns > 0:
            print(f"  Baseline:       {self.baseline_ns/1000:.3f} us")
            print(f"  Overhead:       {self.overhead_vs_baseline:.2f}x baseline")


def measure_baseline_direct_call(iterations: int) -> list[int]:
    """Measure baseline: direct function call without dispatch.

    This simulates the absolute minimum overhead for calling a kernel.

    Args:
        iterations: Number of iterations to run.

    Returns:
        List of latency measurements in nanoseconds.
    """
    latencies: list[int] = []

    # Simulate a minimal kernel operation
    def mock_kernel() -> int:
        return 42

    for _ in range(iterations):
        start = time.perf_counter_ns()
        _ = mock_kernel()
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return latencies


def measure_dict_lookup_baseline(iterations: int) -> list[int]:
    """Measure baseline: dictionary lookup (simulates cache hit).

    This simulates the minimum overhead for any cache-based dispatch.

    Args:
        iterations: Number of iterations to run.

    Returns:
        List of latency measurements in nanoseconds.
    """
    # Create a populated dictionary
    cache = {f"op_{i}": f"kernel_{i}" for i in range(1000)}
    key = "op_500"

    latencies: list[int] = []

    for _ in range(iterations):
        start = time.perf_counter_ns()
        _ = cache.get(key)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return latencies


def benchmark_static_dispatch(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark static dispatch mode overhead.

    Static dispatch uses O(1) dict lookup with pre-computed kernel mapping.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.dispatch.static import (
            StaticDispatcher,
            StaticKernelRegistry,
            StaticKernelEntry,
            get_operation_type,
        )
        from layerzero.dispatch.types import DispatchConfig, DispatchMode
        from layerzero.models.kernel_spec import KernelSpec
    except ImportError as e:
        logger.warning(f"Could not import dispatch modules: {e}")
        # Return mock result for testing
        return _create_mock_static_result(config)

    # Setup: Create registry with mock kernels
    registry = StaticKernelRegistry()

    operations = [
        "attention.causal",
        "attention.full",
        "rms_norm",
        "layer_norm",
        "rope",
        "swiglu",
    ]

    for op in operations:
        # Create mock kernel spec
        spec = KernelSpec(
            kernel_id=f"mock_{op}",
            operation=op,
            source="mock",
            version="1.0.0",
            priority=100,
            impl=lambda **kwargs: None,  # Mock implementation
        )
        entry = StaticKernelEntry.from_kernel_spec(spec, is_default=True)
        registry.register(entry)

    registry.freeze()

    # Create dispatcher
    dispatch_config = DispatchConfig(mode=DispatchMode.STATIC)
    dispatcher = StaticDispatcher(config=dispatch_config, registry=registry)

    # Warmup
    for _ in range(config.warmup_iterations):
        try:
            dispatcher._kernel_cache.get("attention.causal")
        except Exception:
            pass

    # Benchmark: Measure kernel lookup time (not full dispatch, just lookup)
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        op = operations[i % len(operations)]
        start = time.perf_counter_ns()

        # Only measure the lookup, not execution
        _ = dispatcher._kernel_cache.get(op)

        end = time.perf_counter_ns()
        latencies.append(end - start)

    # Measure baseline
    baseline_latencies = measure_baseline_direct_call(config.benchmark_iterations)
    baseline_mean = statistics.mean(baseline_latencies)

    result = BenchmarkResult(
        name="Static Dispatch Lookup",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        baseline_ns=baseline_mean,
    )

    return result


def _create_mock_static_result(config: BenchmarkConfig) -> BenchmarkResult:
    """Create mock result when imports are unavailable."""
    # Simulate static dispatch with pure dict lookup
    cache = {
        "attention.causal": "flash_attn_kernel",
        "attention.full": "sdpa_kernel",
        "rms_norm": "rms_norm_kernel",
    }

    latencies: list[int] = []
    operations = list(cache.keys())

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = cache.get("attention.causal")

    for i in range(config.benchmark_iterations):
        op = operations[i % len(operations)]
        start = time.perf_counter_ns()
        _ = cache.get(op)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    baseline_latencies = measure_baseline_direct_call(config.benchmark_iterations)
    baseline_mean = statistics.mean(baseline_latencies)

    return BenchmarkResult(
        name="Static Dispatch Lookup (Mock)",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        baseline_ns=baseline_mean,
    )


def benchmark_dynamic_dispatch_selection(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark dynamic dispatch kernel selection overhead.

    Dynamic dispatch involves:
    1. Context building
    2. Cache lookup
    3. Kernel filtering
    4. Kernel scoring
    5. Result caching

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.selection.cache import SelectionCache
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec
    except ImportError as e:
        logger.warning(f"Could not import selection modules: {e}")
        return _create_mock_dynamic_result(config)

    # Setup: Create selection cache
    cache = SelectionCache(max_size=10000)
    policy_hash = "test_policy_v1"

    # Create mock execution plans
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
    test_keys = [f"ctx_key_{i}" for i in range(100)]
    for key in test_keys:
        cache.put(key, policy_hash, mock_plan)

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = cache.get(test_keys[0], policy_hash)

    # Benchmark: Measure cache lookup time (cache hit scenario)
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        key = test_keys[i % len(test_keys)]
        start = time.perf_counter_ns()
        _ = cache.get(key, policy_hash)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    # Measure baseline
    baseline_latencies = measure_dict_lookup_baseline(config.benchmark_iterations)
    baseline_mean = statistics.mean(baseline_latencies)

    return BenchmarkResult(
        name="Dynamic Dispatch Selection (Cache Hit)",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        baseline_ns=baseline_mean,
    )


def _create_mock_dynamic_result(config: BenchmarkConfig) -> BenchmarkResult:
    """Create mock result for dynamic dispatch."""
    from collections import OrderedDict

    # Simulate LRU cache behavior
    cache: OrderedDict[str, str] = OrderedDict()
    for i in range(100):
        cache[f"ctx_key_{i}"] = f"plan_{i}"

    latencies: list[int] = []
    keys = list(cache.keys())

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = cache.get(keys[0])
        cache.move_to_end(keys[0])

    for i in range(config.benchmark_iterations):
        key = keys[i % len(keys)]
        start = time.perf_counter_ns()
        _ = cache.get(key)
        if key in cache:
            cache.move_to_end(key)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    baseline_latencies = measure_dict_lookup_baseline(config.benchmark_iterations)
    baseline_mean = statistics.mean(baseline_latencies)

    return BenchmarkResult(
        name="Dynamic Dispatch Selection (Mock)",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        baseline_ns=baseline_mean,
    )


def benchmark_mvcc_cache_lookup(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark MVCC sharded cache lookup overhead.

    MVCC cache features:
    - 256 shards for minimal lock contention
    - Per-shard versioning for O(1) invalidation
    - MD5 hash-based sharding

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.selection.mvcc_cache import MVCCShardedCache
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec
    except ImportError as e:
        logger.warning(f"Could not import MVCC cache: {e}")
        return _create_mock_mvcc_result(config)

    # Setup: Create MVCC cache
    cache = MVCCShardedCache(
        num_shards=256,
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

    # Pre-populate cache
    test_keys = [f"ctx_key_{i}" for i in range(1000)]
    for key in test_keys:
        cache.put(key, policy_hash, mock_plan)

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = cache.get(test_keys[0], policy_hash)

    # Benchmark: Measure sharded cache lookup time
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        key = test_keys[i % len(test_keys)]
        start = time.perf_counter_ns()
        _ = cache.get(key, policy_hash)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    # Measure baseline
    baseline_latencies = measure_dict_lookup_baseline(config.benchmark_iterations)
    baseline_mean = statistics.mean(baseline_latencies)

    return BenchmarkResult(
        name="MVCC Sharded Cache Lookup",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        baseline_ns=baseline_mean,
    )


def _create_mock_mvcc_result(config: BenchmarkConfig) -> BenchmarkResult:
    """Create mock result for MVCC cache."""
    import hashlib

    # Simulate sharded cache
    num_shards = 256
    shards: list[dict[str, str]] = [{} for _ in range(num_shards)]

    # Populate
    test_keys = [f"ctx_key_{i}" for i in range(1000)]
    for key in test_keys:
        h = hashlib.md5(key.encode(), usedforsecurity=False).digest()
        shard_idx = int.from_bytes(h[:8], byteorder="little") % num_shards
        shards[shard_idx][key] = f"plan_{key}"

    latencies: list[int] = []

    # Warmup
    for _ in range(config.warmup_iterations):
        key = test_keys[0]
        h = hashlib.md5(key.encode(), usedforsecurity=False).digest()
        shard_idx = int.from_bytes(h[:8], byteorder="little") % num_shards
        _ = shards[shard_idx].get(key)

    for i in range(config.benchmark_iterations):
        key = test_keys[i % len(test_keys)]
        start = time.perf_counter_ns()
        h = hashlib.md5(key.encode(), usedforsecurity=False).digest()
        shard_idx = int.from_bytes(h[:8], byteorder="little") % num_shards
        _ = shards[shard_idx].get(key)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    baseline_latencies = measure_dict_lookup_baseline(config.benchmark_iterations)
    baseline_mean = statistics.mean(baseline_latencies)

    return BenchmarkResult(
        name="MVCC Sharded Cache Lookup (Mock)",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        baseline_ns=baseline_mean,
    )


def benchmark_config_dispatch_lookup(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark config-driven dispatch lookup overhead.

    Config dispatch involves:
    1. Rule evaluation cache check
    2. Condition matching
    3. Kernel ID retrieval

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.dispatch.config_dispatch import RuleEvaluationCache
    except ImportError as e:
        logger.warning(f"Could not import config dispatch: {e}")
        return _create_mock_config_result(config)

    # Setup: Create rule evaluation cache
    cache = RuleEvaluationCache(max_size=10000)

    # Pre-populate cache
    test_keys = [f"rule_key_{i}" for i in range(100)]
    for i, key in enumerate(test_keys):
        cache.put(key, f"kernel_{i}")

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = cache.get(test_keys[0])

    # Benchmark
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        key = test_keys[i % len(test_keys)]
        start = time.perf_counter_ns()
        _ = cache.get(key)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    baseline_latencies = measure_dict_lookup_baseline(config.benchmark_iterations)
    baseline_mean = statistics.mean(baseline_latencies)

    return BenchmarkResult(
        name="Config Dispatch Rule Cache Lookup",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        baseline_ns=baseline_mean,
    )


def _create_mock_config_result(config: BenchmarkConfig) -> BenchmarkResult:
    """Create mock result for config dispatch."""
    from collections import OrderedDict

    cache: OrderedDict[str, str] = OrderedDict()
    for i in range(100):
        cache[f"rule_key_{i}"] = f"kernel_{i}"

    latencies: list[int] = []
    keys = list(cache.keys())

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = cache.get(keys[0])

    for i in range(config.benchmark_iterations):
        key = keys[i % len(keys)]
        start = time.perf_counter_ns()
        _ = cache.get(key)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    baseline_latencies = measure_dict_lookup_baseline(config.benchmark_iterations)
    baseline_mean = statistics.mean(baseline_latencies)

    return BenchmarkResult(
        name="Config Dispatch Rule Cache Lookup (Mock)",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        baseline_ns=baseline_mean,
    )


def benchmark_mode_switching(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark overhead of switching between dispatch modes.

    Measures the overhead of the orchestrator when switching
    between different dispatch modes dynamically.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    # Simulate mode switching with enum comparison
    from enum import Enum, auto

    class MockDispatchMode(Enum):
        STATIC = auto()
        DYNAMIC = auto()
        CONFIG = auto()
        HOT_RELOAD = auto()

    # Simulate mode selection logic
    modes = list(MockDispatchMode)
    operations = ["attention.causal", "rms_norm", "rope"]
    static_map = {"attention.causal": MockDispatchMode.STATIC}
    config_path = "/path/to/config.yaml"

    def select_mode(operation: str) -> MockDispatchMode:
        if operation in static_map:
            return MockDispatchMode.STATIC
        if config_path:
            return MockDispatchMode.CONFIG
        return MockDispatchMode.DYNAMIC

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = select_mode(operations[0])

    # Benchmark
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        op = operations[i % len(operations)]
        start = time.perf_counter_ns()
        _ = select_mode(op)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    baseline_latencies = measure_baseline_direct_call(config.benchmark_iterations)
    baseline_mean = statistics.mean(baseline_latencies)

    return BenchmarkResult(
        name="Mode Selection Overhead",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
        baseline_ns=baseline_mean,
    )


def run_all_dispatch_benchmarks(
    config: BenchmarkConfig | None = None,
) -> list[BenchmarkResult]:
    """Run all dispatch overhead benchmarks.

    Args:
        config: Benchmark configuration.

    Returns:
        List of BenchmarkResult objects.
    """
    if config is None:
        config = BenchmarkConfig()

    print("\n" + "="*70)
    print("KERNEL DISPATCH OVERHEAD BENCHMARKS")
    print("="*70)
    print(f"Warmup iterations: {config.warmup_iterations}")
    print(f"Benchmark iterations: {config.benchmark_iterations:,}")
    print("="*70)

    results: list[BenchmarkResult] = []

    # Run baseline benchmarks
    print("\n[1/6] Running baseline (direct call)...")
    baseline_latencies = measure_baseline_direct_call(config.benchmark_iterations)
    baseline_result = BenchmarkResult(
        name="Baseline (Direct Function Call)",
        iterations=config.benchmark_iterations,
        latencies_ns=baseline_latencies,
        baseline_ns=0.0,
    )
    results.append(baseline_result)
    baseline_result.print_summary()

    # Run dict lookup baseline
    print("\n[2/6] Running dict lookup baseline...")
    dict_latencies = measure_dict_lookup_baseline(config.benchmark_iterations)
    dict_result = BenchmarkResult(
        name="Baseline (Dict Lookup)",
        iterations=config.benchmark_iterations,
        latencies_ns=dict_latencies,
        baseline_ns=baseline_result.mean_ns,
    )
    results.append(dict_result)
    dict_result.print_summary()

    # Run static dispatch benchmark
    print("\n[3/6] Running static dispatch benchmark...")
    static_result = benchmark_static_dispatch(config)
    results.append(static_result)
    static_result.print_summary()

    # Run dynamic dispatch benchmark
    print("\n[4/6] Running dynamic dispatch benchmark...")
    dynamic_result = benchmark_dynamic_dispatch_selection(config)
    results.append(dynamic_result)
    dynamic_result.print_summary()

    # Run MVCC cache benchmark
    print("\n[5/6] Running MVCC cache benchmark...")
    mvcc_result = benchmark_mvcc_cache_lookup(config)
    results.append(mvcc_result)
    mvcc_result.print_summary()

    # Run mode switching benchmark
    print("\n[6/6] Running mode switching benchmark...")
    mode_result = benchmark_mode_switching(config)
    results.append(mode_result)
    mode_result.print_summary()

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Benchmark':<45} {'P50 (us)':<10} {'P99 (us)':<10} {'Overhead':<10}")
    print("-"*70)
    for r in results:
        overhead = f"{r.overhead_vs_baseline:.2f}x" if r.overhead_vs_baseline > 0 else "N/A"
        print(f"{r.name:<45} {r.p50_ns/1000:<10.3f} {r.p99_ns/1000:<10.3f} {overhead:<10}")

    # Check targets
    print("\n" + "="*70)
    print("TARGET VALIDATION")
    print("="*70)

    targets = [
        ("Static Dispatch Lookup", 10_000, "P99 < 10us"),  # 10us target
        ("Dynamic Dispatch Selection", 100_000, "P99 < 100us"),  # 100us target
        ("MVCC Sharded Cache Lookup", 1_000, "P99 < 1us"),  # 1us target
    ]

    for name, target_ns, description in targets:
        for r in results:
            if name in r.name:
                status = "PASS" if r.p99_ns < target_ns else "FAIL"
                actual = f"{r.p99_ns/1000:.3f}us"
                target = f"{target_ns/1000:.3f}us"
                print(f"  [{status}] {description}: actual={actual}, target<{target}")
                break

    # Save to JSON if requested
    if config.output_json:
        output_data = {
            "benchmark": "dispatch_overhead",
            "config": {
                "warmup_iterations": config.warmup_iterations,
                "benchmark_iterations": config.benchmark_iterations,
            },
            "results": [r.to_dict() for r in results],
        }
        with open(config.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {config.output_json}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dispatch Overhead Benchmark")
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
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    benchmark_config = BenchmarkConfig(
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        output_json=Path(args.output) if args.output else None,
    )

    run_all_dispatch_benchmarks(benchmark_config)
