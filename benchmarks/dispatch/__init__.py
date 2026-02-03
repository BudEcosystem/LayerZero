"""
LayerZero Kernel Dispatch Performance Benchmarks

This package provides comprehensive benchmarks for the kernel dispatch system:
- bench_dispatch_overhead.py: Measure dispatch overhead for each mode
- bench_circuit_breaker.py: Benchmark circuit breaker performance
- bench_cache_performance.py: Benchmark cache hit/miss latency
- run_benchmarks.py: Runner script that executes all benchmarks

Benchmark Requirements:
- Nanosecond precision timing with time.perf_counter_ns()
- Warmup iterations (3-5 runs discarded)
- Report p50, p95, p99 latencies
- Minimum 1000 iterations per benchmark
- Compare against baseline (direct function call)

Target Performance:
- Selection overhead: <100us p99
- Execution overhead: <10us for dispatch layer
- Cache lookup time: <1us
- Circuit breaker check time: <100ns
"""
from __future__ import annotations

__all__ = [
    "BenchmarkResult",
    "BenchmarkConfig",
    "BenchmarkRunner",
]
