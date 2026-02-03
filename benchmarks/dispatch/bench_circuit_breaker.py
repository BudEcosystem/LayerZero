#!/usr/bin/env python3
"""
Circuit Breaker Performance Benchmark

Measures performance characteristics of the circuit breaker pattern:
- State check time (is_allowed)
- Success/failure recording time
- State transition overhead
- Concurrent access performance

Target Performance:
- Circuit breaker check time: <100ns for normal operation
- State transition: <1us
- Concurrent access: linear scaling with threads

Methodology:
1. Create circuit breakers in various states
2. Warm up with 5 iterations (discarded)
3. Run 1000+ iterations per operation
4. Measure total time and compute statistics
5. Report p50, p95, p99 latencies
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    warmup_iterations: int = 5
    benchmark_iterations: int = 10000
    num_threads: int = 4
    output_json: Path | None = None


@dataclass(slots=True)
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    iterations: int
    latencies_ns: list[int] = field(default_factory=list)
    baseline_ns: float = 0.0

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

    @property
    def overhead_vs_baseline(self) -> float:
        return self.mean_ns / self.baseline_ns if self.baseline_ns > 0 else 0.0

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


def benchmark_circuit_is_allowed_closed(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark is_allowed() when circuit is CLOSED (normal operation).

    This is the most common case and should be extremely fast.
    Target: <100ns

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.dispatch.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
        )
    except ImportError:
        return _benchmark_mock_circuit_is_allowed_closed(config)

    # Setup: Create circuit breaker in CLOSED state
    cb_config = CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        cooldown_seconds=30.0,
    )
    circuit = CircuitBreaker("test_kernel", cb_config)

    kernel_ids = [f"kernel_{i}" for i in range(10)]

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = circuit.can_execute()

    # Benchmark
    latencies: list[int] = []

    for _ in range(config.benchmark_iterations):
        start = time.perf_counter_ns()
        _ = circuit.can_execute()
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="Circuit Breaker is_allowed (CLOSED state)",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
    )


def _benchmark_mock_circuit_is_allowed_closed(config: BenchmarkConfig) -> BenchmarkResult:
    """Mock benchmark for when imports are unavailable."""
    from enum import Enum, auto
    from threading import RLock

    class MockCircuitState(Enum):
        CLOSED = auto()
        OPEN = auto()
        HALF_OPEN = auto()

    # Simulate minimal circuit breaker check
    state = MockCircuitState.CLOSED
    lock = RLock()

    def can_execute() -> bool:
        with lock:
            return state == MockCircuitState.CLOSED

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = can_execute()

    # Benchmark
    latencies: list[int] = []

    for _ in range(config.benchmark_iterations):
        start = time.perf_counter_ns()
        _ = can_execute()
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="Circuit Breaker is_allowed (CLOSED state) [Mock]",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
    )


def benchmark_circuit_is_allowed_open(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark is_allowed() when circuit is OPEN.

    Checks cooldown logic which involves time comparison.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.dispatch.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
        )
    except ImportError:
        return _benchmark_mock_circuit_is_allowed_open(config)

    # Setup: Create circuit breaker and force it OPEN
    cb_config = CircuitBreakerConfig(
        failure_threshold=1,
        cooldown_seconds=3600.0,  # Long cooldown to stay OPEN
    )
    circuit = CircuitBreaker("test_kernel", cb_config)

    # Force open by recording failure
    circuit.record_failure()

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = circuit.can_execute()

    # Benchmark
    latencies: list[int] = []

    for _ in range(config.benchmark_iterations):
        start = time.perf_counter_ns()
        _ = circuit.can_execute()
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="Circuit Breaker is_allowed (OPEN state)",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
    )


def _benchmark_mock_circuit_is_allowed_open(config: BenchmarkConfig) -> BenchmarkResult:
    """Mock benchmark for OPEN state."""
    from threading import RLock

    cooldown_until = time.monotonic() + 3600.0
    lock = RLock()

    def can_execute() -> bool:
        with lock:
            now = time.monotonic()
            return now >= cooldown_until

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = can_execute()

    # Benchmark
    latencies: list[int] = []

    for _ in range(config.benchmark_iterations):
        start = time.perf_counter_ns()
        _ = can_execute()
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="Circuit Breaker is_allowed (OPEN state) [Mock]",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
    )


def benchmark_circuit_record_success(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark record_success() operation.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.dispatch.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
        )
    except ImportError:
        return _benchmark_mock_record_success(config)

    # Setup
    cb_config = CircuitBreakerConfig(failure_threshold=5)
    circuit = CircuitBreaker("test_kernel", cb_config)

    # Warmup
    for _ in range(config.warmup_iterations):
        circuit.record_success()

    # Benchmark
    latencies: list[int] = []

    for _ in range(config.benchmark_iterations):
        start = time.perf_counter_ns()
        circuit.record_success()
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="Circuit Breaker record_success",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
    )


def _benchmark_mock_record_success(config: BenchmarkConfig) -> BenchmarkResult:
    """Mock benchmark for record_success."""
    from threading import RLock

    lock = RLock()
    consecutive_failures = 0
    total_successes = 0

    def record_success() -> None:
        nonlocal consecutive_failures, total_successes
        with lock:
            consecutive_failures = 0
            total_successes += 1

    # Warmup
    for _ in range(config.warmup_iterations):
        record_success()

    # Benchmark
    latencies: list[int] = []

    for _ in range(config.benchmark_iterations):
        start = time.perf_counter_ns()
        record_success()
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="Circuit Breaker record_success [Mock]",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
    )


def benchmark_circuit_record_failure(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark record_failure() operation.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.dispatch.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
        )
    except ImportError:
        return _benchmark_mock_record_failure(config)

    # Setup - high threshold so we don't trip the circuit
    cb_config = CircuitBreakerConfig(failure_threshold=1000000)
    circuit = CircuitBreaker("test_kernel", cb_config)

    # Warmup
    for _ in range(config.warmup_iterations):
        circuit.record_failure()

    # Reset for actual benchmark
    circuit.reset()

    # Benchmark
    latencies: list[int] = []

    for _ in range(config.benchmark_iterations):
        start = time.perf_counter_ns()
        circuit.record_failure()
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="Circuit Breaker record_failure",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
    )


def _benchmark_mock_record_failure(config: BenchmarkConfig) -> BenchmarkResult:
    """Mock benchmark for record_failure."""
    from threading import RLock

    lock = RLock()
    consecutive_failures = 0
    total_failures = 0
    failure_threshold = 1000000

    def record_failure() -> None:
        nonlocal consecutive_failures, total_failures
        with lock:
            consecutive_failures += 1
            total_failures += 1
            if consecutive_failures >= failure_threshold:
                pass  # Transition to OPEN

    # Warmup
    for _ in range(config.warmup_iterations):
        record_failure()

    # Reset
    consecutive_failures = 0
    total_failures = 0

    # Benchmark
    latencies: list[int] = []

    for _ in range(config.benchmark_iterations):
        start = time.perf_counter_ns()
        record_failure()
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="Circuit Breaker record_failure [Mock]",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
    )


def benchmark_circuit_state_transition(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark state transition overhead (CLOSED -> OPEN -> HALF_OPEN).

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.dispatch.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
        )
    except ImportError:
        return _benchmark_mock_state_transition(config)

    latencies: list[int] = []

    for _ in range(config.benchmark_iterations // 10):  # Fewer iterations for transitions
        # Setup fresh circuit
        cb_config = CircuitBreakerConfig(
            failure_threshold=1,
            cooldown_seconds=0.0001,  # Very short cooldown
        )
        circuit = CircuitBreaker("test_kernel", cb_config)

        # Measure CLOSED -> OPEN transition
        start = time.perf_counter_ns()
        circuit.record_failure()
        end = time.perf_counter_ns()
        latencies.append(end - start)

        # Wait for cooldown
        time.sleep(0.001)

        # Measure check that triggers OPEN -> HALF_OPEN
        start = time.perf_counter_ns()
        _ = circuit.can_execute()
        end = time.perf_counter_ns()
        latencies.append(end - start)

        # Measure HALF_OPEN -> CLOSED transition
        start = time.perf_counter_ns()
        circuit.record_success()
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="Circuit Breaker State Transitions",
        iterations=len(latencies),
        latencies_ns=latencies,
    )


def _benchmark_mock_state_transition(config: BenchmarkConfig) -> BenchmarkResult:
    """Mock benchmark for state transitions."""
    from enum import Enum, auto
    from threading import RLock

    class MockState(Enum):
        CLOSED = auto()
        OPEN = auto()
        HALF_OPEN = auto()

    latencies: list[int] = []
    lock = RLock()

    for _ in range(config.benchmark_iterations // 10):
        state = MockState.CLOSED

        # CLOSED -> OPEN
        start = time.perf_counter_ns()
        with lock:
            state = MockState.OPEN
        end = time.perf_counter_ns()
        latencies.append(end - start)

        # OPEN -> HALF_OPEN
        start = time.perf_counter_ns()
        with lock:
            state = MockState.HALF_OPEN
        end = time.perf_counter_ns()
        latencies.append(end - start)

        # HALF_OPEN -> CLOSED
        start = time.perf_counter_ns()
        with lock:
            state = MockState.CLOSED
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="Circuit Breaker State Transitions [Mock]",
        iterations=len(latencies),
        latencies_ns=latencies,
    )


def benchmark_circuit_concurrent_access(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark concurrent access to circuit breaker.

    Measures performance with multiple threads accessing the same circuit.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.dispatch.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
        )
    except ImportError:
        return _benchmark_mock_concurrent_access(config)

    # Setup
    cb_config = CircuitBreakerConfig(failure_threshold=1000000)
    circuit = CircuitBreaker("test_kernel", cb_config)

    all_latencies: list[int] = []
    latencies_lock = threading.Lock()
    iterations_per_thread = config.benchmark_iterations // config.num_threads

    def worker() -> None:
        thread_latencies: list[int] = []
        for i in range(iterations_per_thread):
            start = time.perf_counter_ns()
            _ = circuit.can_execute()
            if i % 2 == 0:
                circuit.record_success()
            end = time.perf_counter_ns()
            thread_latencies.append(end - start)

        with latencies_lock:
            all_latencies.extend(thread_latencies)

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = circuit.can_execute()

    # Run concurrent benchmark
    threads = [
        threading.Thread(target=worker)
        for _ in range(config.num_threads)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return BenchmarkResult(
        name=f"Circuit Breaker Concurrent Access ({config.num_threads} threads)",
        iterations=len(all_latencies),
        latencies_ns=all_latencies,
    )


def _benchmark_mock_concurrent_access(config: BenchmarkConfig) -> BenchmarkResult:
    """Mock benchmark for concurrent access."""
    from threading import RLock

    lock = RLock()
    state = "CLOSED"
    failures = 0
    successes = 0

    all_latencies: list[int] = []
    latencies_lock = threading.Lock()
    iterations_per_thread = config.benchmark_iterations // config.num_threads

    def can_execute() -> bool:
        with lock:
            return state == "CLOSED"

    def record_success() -> None:
        nonlocal successes
        with lock:
            successes += 1

    def worker() -> None:
        thread_latencies: list[int] = []
        for i in range(iterations_per_thread):
            start = time.perf_counter_ns()
            _ = can_execute()
            if i % 2 == 0:
                record_success()
            end = time.perf_counter_ns()
            thread_latencies.append(end - start)

        with latencies_lock:
            all_latencies.extend(thread_latencies)

    # Run concurrent benchmark
    threads = [
        threading.Thread(target=worker)
        for _ in range(config.num_threads)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return BenchmarkResult(
        name=f"Circuit Breaker Concurrent Access ({config.num_threads} threads) [Mock]",
        iterations=len(all_latencies),
        latencies_ns=all_latencies,
    )


def benchmark_registry_lookup(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark circuit breaker registry lookup.

    Measures the overhead of looking up or creating circuits by kernel ID.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    try:
        from layerzero.dispatch.circuit_breaker import CircuitBreakerRegistry
    except ImportError:
        return _benchmark_mock_registry_lookup(config)

    # Setup
    registry = CircuitBreakerRegistry()

    # Pre-populate with some circuits
    kernel_ids = [f"kernel_{i}" for i in range(100)]
    for kid in kernel_ids:
        registry.get_or_create(kid)

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = registry.get_or_create(kernel_ids[0])

    # Benchmark - lookup existing circuits
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        kid = kernel_ids[i % len(kernel_ids)]
        start = time.perf_counter_ns()
        _ = registry.get_or_create(kid)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="Circuit Breaker Registry Lookup",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
    )


def _benchmark_mock_registry_lookup(config: BenchmarkConfig) -> BenchmarkResult:
    """Mock benchmark for registry lookup."""
    from threading import RLock

    circuits: dict[str, dict] = {}
    lock = RLock()

    # Pre-populate
    kernel_ids = [f"kernel_{i}" for i in range(100)]
    for kid in kernel_ids:
        circuits[kid] = {"state": "CLOSED", "failures": 0}

    def get_or_create(name: str) -> dict:
        with lock:
            if name not in circuits:
                circuits[name] = {"state": "CLOSED", "failures": 0}
            return circuits[name]

    # Warmup
    for _ in range(config.warmup_iterations):
        _ = get_or_create(kernel_ids[0])

    # Benchmark
    latencies: list[int] = []

    for i in range(config.benchmark_iterations):
        kid = kernel_ids[i % len(kernel_ids)]
        start = time.perf_counter_ns()
        _ = get_or_create(kid)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    return BenchmarkResult(
        name="Circuit Breaker Registry Lookup [Mock]",
        iterations=config.benchmark_iterations,
        latencies_ns=latencies,
    )


def run_all_circuit_breaker_benchmarks(
    config: BenchmarkConfig | None = None,
) -> list[BenchmarkResult]:
    """Run all circuit breaker benchmarks.

    Args:
        config: Benchmark configuration.

    Returns:
        List of BenchmarkResult objects.
    """
    if config is None:
        config = BenchmarkConfig()

    print("\n" + "="*70)
    print("CIRCUIT BREAKER PERFORMANCE BENCHMARKS")
    print("="*70)
    print(f"Warmup iterations: {config.warmup_iterations}")
    print(f"Benchmark iterations: {config.benchmark_iterations:,}")
    print(f"Concurrent threads: {config.num_threads}")
    print("="*70)

    results: list[BenchmarkResult] = []

    # is_allowed benchmarks
    print("\n[1/7] Running is_allowed (CLOSED state) benchmark...")
    result = benchmark_circuit_is_allowed_closed(config)
    results.append(result)
    result.print_summary()

    print("\n[2/7] Running is_allowed (OPEN state) benchmark...")
    result = benchmark_circuit_is_allowed_open(config)
    results.append(result)
    result.print_summary()

    # Record operations
    print("\n[3/7] Running record_success benchmark...")
    result = benchmark_circuit_record_success(config)
    results.append(result)
    result.print_summary()

    print("\n[4/7] Running record_failure benchmark...")
    result = benchmark_circuit_record_failure(config)
    results.append(result)
    result.print_summary()

    # State transitions
    print("\n[5/7] Running state transition benchmark...")
    result = benchmark_circuit_state_transition(config)
    results.append(result)
    result.print_summary()

    # Concurrent access
    print("\n[6/7] Running concurrent access benchmark...")
    result = benchmark_circuit_concurrent_access(config)
    results.append(result)
    result.print_summary()

    # Registry lookup
    print("\n[7/7] Running registry lookup benchmark...")
    result = benchmark_registry_lookup(config)
    results.append(result)
    result.print_summary()

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Benchmark':<55} {'P50 (ns)':<12} {'P99 (ns)':<12}")
    print("-"*70)
    for r in results:
        print(f"{r.name:<55} {r.p50_ns:<12.1f} {r.p99_ns:<12.1f}")

    # Target validation
    print("\n" + "="*70)
    print("TARGET VALIDATION")
    print("="*70)

    targets = [
        ("is_allowed (CLOSED state)", 100, "P99 < 100ns"),
        ("record_success", 500, "P99 < 500ns"),
        ("record_failure", 500, "P99 < 500ns"),
        ("Registry Lookup", 1000, "P99 < 1us"),
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
            "benchmark": "circuit_breaker",
            "config": {
                "warmup_iterations": config.warmup_iterations,
                "benchmark_iterations": config.benchmark_iterations,
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

    parser = argparse.ArgumentParser(description="Circuit Breaker Benchmark")
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
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    benchmark_config = BenchmarkConfig(
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        num_threads=args.threads,
        output_json=Path(args.output) if args.output else None,
    )

    run_all_circuit_breaker_benchmarks(benchmark_config)
