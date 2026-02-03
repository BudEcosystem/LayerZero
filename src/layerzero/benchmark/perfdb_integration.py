"""
PerfDB Integration for Benchmarks

Provides persistence for benchmark results.
"""
from __future__ import annotations

import logging
from typing import Any

from layerzero.benchmark.harness import BenchmarkResult

logger = logging.getLogger(__name__)

# In-memory cache for benchmark results
# In production, this would integrate with the actual PerfDB
_benchmark_cache: dict[str, dict[str, Any]] = {}
_benchmarks_valid: bool = True


def save_benchmark_to_perfdb(
    benchmark_id: str,
    result: BenchmarkResult,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save benchmark result to PerfDB.

    Args:
        benchmark_id: Unique identifier for the benchmark.
        result: Benchmark result to save.
        metadata: Optional metadata to associate with result.
    """
    global _benchmark_cache

    entry = {
        "result": result.to_dict(),
        "metadata": metadata or {},
        "valid": True,
    }

    _benchmark_cache[benchmark_id] = entry
    logger.debug(f"Saved benchmark {benchmark_id} to PerfDB")


def load_benchmark_from_perfdb(
    benchmark_id: str,
) -> BenchmarkResult | None:
    """Load benchmark result from PerfDB.

    Args:
        benchmark_id: Unique identifier for the benchmark.

    Returns:
        BenchmarkResult if found and valid, None otherwise.
    """
    global _benchmark_cache, _benchmarks_valid

    if not _benchmarks_valid:
        return None

    entry = _benchmark_cache.get(benchmark_id)
    if entry is None:
        return None

    if not entry.get("valid", False):
        return None

    return BenchmarkResult.from_dict(entry["result"])


def invalidate_benchmarks() -> None:
    """Invalidate all cached benchmark results.

    Marks all benchmarks as stale, requiring re-run.
    """
    global _benchmarks_valid, _benchmark_cache

    _benchmarks_valid = False

    for entry in _benchmark_cache.values():
        entry["valid"] = False

    logger.debug("Invalidated all benchmark results")


def clear_benchmark_cache() -> None:
    """Clear the benchmark cache entirely."""
    global _benchmark_cache, _benchmarks_valid

    _benchmark_cache.clear()
    _benchmarks_valid = True
    logger.debug("Cleared benchmark cache")


def get_benchmark_metadata(
    benchmark_id: str,
) -> dict[str, Any] | None:
    """Get metadata for a benchmark.

    Args:
        benchmark_id: Unique identifier for the benchmark.

    Returns:
        Metadata dictionary or None if not found.
    """
    entry = _benchmark_cache.get(benchmark_id)
    if entry is None:
        return None

    return entry.get("metadata")


def list_benchmark_ids() -> list[str]:
    """List all benchmark IDs in cache.

    Returns:
        List of benchmark identifiers.
    """
    return list(_benchmark_cache.keys())


def is_benchmark_valid(benchmark_id: str) -> bool:
    """Check if a benchmark result is valid.

    Args:
        benchmark_id: Unique identifier for the benchmark.

    Returns:
        True if benchmark exists and is valid.
    """
    global _benchmarks_valid

    if not _benchmarks_valid:
        return False

    entry = _benchmark_cache.get(benchmark_id)
    if entry is None:
        return False

    return entry.get("valid", False)
