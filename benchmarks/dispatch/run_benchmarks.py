#!/usr/bin/env python3
"""
LayerZero Kernel Dispatch Benchmark Runner

Executes all dispatch benchmarks and generates a comprehensive report.

Usage:
    # Run all benchmarks with defaults
    python -m benchmarks.dispatch.run_benchmarks

    # Run with custom iterations
    python -m benchmarks.dispatch.run_benchmarks --iterations 50000

    # Run specific benchmark
    python -m benchmarks.dispatch.run_benchmarks --only dispatch

    # Save results to JSON
    python -m benchmarks.dispatch.run_benchmarks --output results.json

    # Quick mode (fewer iterations)
    python -m benchmarks.dispatch.run_benchmarks --quick

Benchmarks included:
1. Dispatch Overhead - Measures overhead for each dispatch mode
2. Circuit Breaker - Measures circuit breaker state check/transition performance
3. Cache Performance - Measures cache hit/miss latency and invalidation

Target Performance:
- Selection overhead: <100us p99
- Execution overhead: <10us for dispatch layer
- Cache lookup time: <1us
- Circuit breaker check time: <100ns
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunnerConfig:
    """Configuration for the benchmark runner.

    Attributes:
        warmup_iterations: Number of warmup iterations to discard.
        benchmark_iterations: Number of timed iterations per benchmark.
        num_threads: Number of threads for concurrent benchmarks.
        cache_size: Cache size for cache benchmarks.
        num_shards: Number of MVCC cache shards.
        output_json: Path to save combined results JSON.
        only: Run only specific benchmark suite (dispatch, circuit, cache).
        quick: Quick mode with fewer iterations.
        verbose: Verbose output.
    """
    warmup_iterations: int = 5
    benchmark_iterations: int = 10000
    num_threads: int = 4
    cache_size: int = 10000
    num_shards: int = 256
    output_json: Path | None = None
    only: str | None = None
    quick: bool = False
    verbose: bool = False


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results from a suite.

    Attributes:
        name: Suite name.
        results: List of individual benchmark results.
        total_time_seconds: Total time to run the suite.
    """
    name: str
    results: list[dict[str, Any]] = field(default_factory=list)
    total_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "results": self.results,
            "total_time_seconds": self.total_time_seconds,
        }


def get_system_info() -> dict[str, Any]:
    """Collect system information for reproducibility.

    Returns:
        Dictionary with system information.
    """
    import os

    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "timestamp": datetime.now().isoformat(),
    }

    # Try to get CPU frequency if available
    try:
        import psutil
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            info["cpu_freq_mhz"] = {
                "current": cpu_freq.current,
                "min": cpu_freq.min,
                "max": cpu_freq.max,
            }
        info["memory_gb"] = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    return info


def run_dispatch_benchmarks(config: RunnerConfig) -> BenchmarkSuite:
    """Run dispatch overhead benchmarks.

    Args:
        config: Runner configuration.

    Returns:
        BenchmarkSuite with results.
    """
    from benchmarks.dispatch.bench_dispatch_overhead import (
        BenchmarkConfig,
        run_all_dispatch_benchmarks,
    )

    bench_config = BenchmarkConfig(
        warmup_iterations=config.warmup_iterations,
        benchmark_iterations=config.benchmark_iterations,
    )

    start_time = time.monotonic()
    results = run_all_dispatch_benchmarks(bench_config)
    end_time = time.monotonic()

    suite = BenchmarkSuite(
        name="dispatch_overhead",
        results=[r.to_dict() for r in results],
        total_time_seconds=end_time - start_time,
    )

    return suite


def run_circuit_breaker_benchmarks(config: RunnerConfig) -> BenchmarkSuite:
    """Run circuit breaker benchmarks.

    Args:
        config: Runner configuration.

    Returns:
        BenchmarkSuite with results.
    """
    from benchmarks.dispatch.bench_circuit_breaker import (
        BenchmarkConfig,
        run_all_circuit_breaker_benchmarks,
    )

    bench_config = BenchmarkConfig(
        warmup_iterations=config.warmup_iterations,
        benchmark_iterations=config.benchmark_iterations,
        num_threads=config.num_threads,
    )

    start_time = time.monotonic()
    results = run_all_circuit_breaker_benchmarks(bench_config)
    end_time = time.monotonic()

    suite = BenchmarkSuite(
        name="circuit_breaker",
        results=[r.to_dict() for r in results],
        total_time_seconds=end_time - start_time,
    )

    return suite


def run_cache_benchmarks(config: RunnerConfig) -> BenchmarkSuite:
    """Run cache performance benchmarks.

    Args:
        config: Runner configuration.

    Returns:
        BenchmarkSuite with results.
    """
    from benchmarks.dispatch.bench_cache_performance import (
        BenchmarkConfig,
        run_all_cache_benchmarks,
    )

    bench_config = BenchmarkConfig(
        warmup_iterations=config.warmup_iterations,
        benchmark_iterations=config.benchmark_iterations,
        cache_size=config.cache_size,
        num_shards=config.num_shards,
        num_threads=config.num_threads,
    )

    start_time = time.monotonic()
    results = run_all_cache_benchmarks(bench_config)
    end_time = time.monotonic()

    suite = BenchmarkSuite(
        name="cache_performance",
        results=[r.to_dict() for r in results],
        total_time_seconds=end_time - start_time,
    )

    return suite


def print_final_summary(suites: list[BenchmarkSuite]) -> None:
    """Print final summary of all benchmarks.

    Args:
        suites: List of completed benchmark suites.
    """
    print("\n")
    print("="*70)
    print("FINAL BENCHMARK SUMMARY")
    print("="*70)

    total_time = sum(s.total_time_seconds for s in suites)
    total_benchmarks = sum(len(s.results) for s in suites)

    print(f"\nTotal benchmarks run: {total_benchmarks}")
    print(f"Total time: {total_time:.2f} seconds")

    # Performance targets summary
    print("\n" + "-"*70)
    print("PERFORMANCE TARGETS")
    print("-"*70)

    targets = {
        "Static Dispatch Lookup": ("p99_us", 10.0, "<10us"),
        "Dynamic Dispatch Selection": ("p99_us", 100.0, "<100us"),
        "MVCC Sharded Cache Hit": ("p99_us", 1.0, "<1us"),
        "Circuit Breaker is_allowed": ("p99_ns", 100.0, "<100ns"),
    }

    for suite in suites:
        for result in suite.results:
            name = result.get("name", "")
            for target_name, (metric, threshold, description) in targets.items():
                if target_name in name:
                    value = result.get(metric, 0)
                    passed = value < threshold
                    status = "PASS" if passed else "FAIL"
                    print(f"  [{status}] {target_name}: {value:.3f} {metric.split('_')[1]} (target: {description})")
                    break

    # Per-suite summary
    print("\n" + "-"*70)
    print("SUITE SUMMARY")
    print("-"*70)

    for suite in suites:
        print(f"\n{suite.name.upper()}:")
        print(f"  Benchmarks: {len(suite.results)}")
        print(f"  Time: {suite.total_time_seconds:.2f}s")

        # Find best and worst performing
        if suite.results:
            by_p99 = sorted(suite.results, key=lambda r: r.get("p99_us", 0))
            best = by_p99[0]
            worst = by_p99[-1]
            print(f"  Best P99: {best['name']} ({best.get('p99_us', 0):.3f}us)")
            print(f"  Worst P99: {worst['name']} ({worst.get('p99_us', 0):.3f}us)")


def save_results(
    suites: list[BenchmarkSuite],
    config: RunnerConfig,
    output_path: Path,
) -> None:
    """Save combined results to JSON file.

    Args:
        suites: List of benchmark suites.
        config: Runner configuration.
        output_path: Path to save JSON.
    """
    results = {
        "benchmark_run": {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "warmup_iterations": config.warmup_iterations,
                "benchmark_iterations": config.benchmark_iterations,
                "num_threads": config.num_threads,
                "cache_size": config.cache_size,
                "num_shards": config.num_shards,
            },
            "system_info": get_system_info(),
        },
        "suites": [s.to_dict() for s in suites],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main() -> int:
    """Main entry point for benchmark runner.

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="LayerZero Kernel Dispatch Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all benchmarks with defaults
    python -m benchmarks.dispatch.run_benchmarks

    # Run with more iterations for accuracy
    python -m benchmarks.dispatch.run_benchmarks --iterations 50000

    # Run only dispatch overhead benchmarks
    python -m benchmarks.dispatch.run_benchmarks --only dispatch

    # Quick mode (fewer iterations, faster results)
    python -m benchmarks.dispatch.run_benchmarks --quick

    # Save results to JSON
    python -m benchmarks.dispatch.run_benchmarks --output results.json
""",
    )

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
        "--threads",
        type=int,
        default=4,
        help="Number of threads for concurrent tests (default: 4)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=10000,
        help="Cache size for cache benchmarks (default: 10000)",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=256,
        help="Number of MVCC shards (default: 256)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path for results",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=["dispatch", "circuit", "cache"],
        default=None,
        help="Run only specific benchmark suite",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer iterations (1000)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Build configuration
    config = RunnerConfig(
        warmup_iterations=args.warmup,
        benchmark_iterations=1000 if args.quick else args.iterations,
        num_threads=args.threads,
        cache_size=args.cache_size,
        num_shards=args.shards,
        output_json=Path(args.output) if args.output else None,
        only=args.only,
        quick=args.quick,
        verbose=args.verbose,
    )

    if config.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print header
    print("\n")
    print("="*70)
    print("LAYERZERO KERNEL DISPATCH BENCHMARKS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Warmup iterations:    {config.warmup_iterations}")
    print(f"  Benchmark iterations: {config.benchmark_iterations:,}")
    print(f"  Concurrent threads:   {config.num_threads}")
    print(f"  Cache size:           {config.cache_size:,}")
    print(f"  MVCC shards:          {config.num_shards}")

    # Print system info
    sys_info = get_system_info()
    print(f"\nSystem Information:")
    print(f"  Platform:     {sys_info['platform']}")
    print(f"  Python:       {sys_info['python_version']}")
    print(f"  CPU:          {sys_info['processor']}")
    print(f"  CPU Count:    {sys_info['cpu_count']}")
    if "memory_gb" in sys_info:
        print(f"  Memory:       {sys_info['memory_gb']:.1f} GB")

    # Run benchmarks
    suites: list[BenchmarkSuite] = []
    total_start = time.monotonic()

    try:
        if config.only is None or config.only == "dispatch":
            print("\n" + "-"*70)
            print("Running Dispatch Overhead Benchmarks...")
            print("-"*70)
            suite = run_dispatch_benchmarks(config)
            suites.append(suite)

        if config.only is None or config.only == "circuit":
            print("\n" + "-"*70)
            print("Running Circuit Breaker Benchmarks...")
            print("-"*70)
            suite = run_circuit_breaker_benchmarks(config)
            suites.append(suite)

        if config.only is None or config.only == "cache":
            print("\n" + "-"*70)
            print("Running Cache Performance Benchmarks...")
            print("-"*70)
            suite = run_cache_benchmarks(config)
            suites.append(suite)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        return 1

    total_end = time.monotonic()
    total_time = total_end - total_start

    # Print final summary
    print_final_summary(suites)

    print(f"\n{'='*70}")
    print(f"Total benchmark time: {total_time:.2f} seconds")
    print("="*70)

    # Save results if requested
    if config.output_json:
        save_results(suites, config, config.output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
