"""
Kernel Comparison

Provides head-to-head kernel comparison functionality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from layerzero.benchmark.harness import BenchmarkConfig, BenchmarkHarness, BenchmarkResult
from layerzero.benchmark.stats import is_statistically_significant


@dataclass
class ComparisonResult:
    """Result of comparing two kernels.

    Attributes:
        baseline: Benchmark result for baseline kernel.
        candidate: Benchmark result for candidate kernel.
        speedup: Speedup factor (baseline / candidate).
        winner: Which kernel won ("baseline", "candidate", or "tie").
        significant: Whether the difference is statistically significant.
        p_value: P-value from statistical test.
    """

    baseline: BenchmarkResult
    candidate: BenchmarkResult
    speedup: float
    winner: str
    significant: bool
    p_value: float = 1.0

    def summary(self) -> str:
        """Generate summary string.

        Returns:
            Human-readable summary.
        """
        lines = [
            f"Baseline:  {self.baseline.median_ns / 1e6:.3f} ms (median)",
            f"Candidate: {self.candidate.median_ns / 1e6:.3f} ms (median)",
            f"Speedup:   {self.speedup:.2f}x",
            f"Winner:    {self.winner}",
            f"Significant: {self.significant} (p={self.p_value:.4f})",
        ]
        return "\n".join(lines)


def compare_kernels(
    baseline_fn: Callable[[], Any],
    candidate_fn: Callable[[], Any],
    config: BenchmarkConfig | None = None,
    significance_threshold: float = 0.05,
    min_speedup_threshold: float = 1.05,
) -> ComparisonResult:
    """Compare two kernel implementations.

    Runs both kernels through the benchmark harness and
    compares their performance statistically.

    Args:
        baseline_fn: Baseline kernel function.
        candidate_fn: Candidate kernel function.
        config: Benchmark configuration (uses defaults if None).
        significance_threshold: P-value threshold for significance.
        min_speedup_threshold: Minimum speedup to declare a winner.

    Returns:
        ComparisonResult with detailed comparison.

    Example:
        ```python
        result = compare_kernels(
            baseline_fn=lambda: torch.softmax(x, dim=-1),
            candidate_fn=lambda: custom_softmax(x),
        )
        print(result.summary())
        ```
    """
    if config is None:
        config = BenchmarkConfig()

    harness = BenchmarkHarness(config)

    # Run benchmarks
    baseline = harness.run(baseline_fn)
    candidate = harness.run(candidate_fn)

    # Calculate speedup (baseline / candidate)
    # > 1 means candidate is faster
    if candidate.median_ns > 0:
        speedup = baseline.median_ns / candidate.median_ns
    else:
        speedup = float("inf")

    # Statistical significance test
    baseline_floats = [float(x) for x in baseline.latencies_ns]
    candidate_floats = [float(x) for x in candidate.latencies_ns]

    significant, p_value = is_statistically_significant(
        baseline_floats,
        candidate_floats,
        p_threshold=significance_threshold,
    )

    # Determine winner
    if not significant:
        winner = "tie"
    elif speedup >= min_speedup_threshold:
        winner = "candidate"
    elif speedup <= 1 / min_speedup_threshold:
        winner = "baseline"
    else:
        winner = "tie"

    return ComparisonResult(
        baseline=baseline,
        candidate=candidate,
        speedup=speedup,
        winner=winner,
        significant=significant,
        p_value=p_value,
    )


def compare_multiple_kernels(
    kernels: dict[str, Callable[[], Any]],
    config: BenchmarkConfig | None = None,
) -> dict[str, BenchmarkResult]:
    """Compare multiple kernel implementations.

    Args:
        kernels: Dictionary mapping kernel names to functions.
        config: Benchmark configuration.

    Returns:
        Dictionary mapping kernel names to results.
    """
    if config is None:
        config = BenchmarkConfig()

    harness = BenchmarkHarness(config)
    results = {}

    for name, fn in kernels.items():
        results[name] = harness.run(fn)

    return results


def find_fastest_kernel(
    kernels: dict[str, Callable[[], Any]],
    config: BenchmarkConfig | None = None,
) -> tuple[str, BenchmarkResult]:
    """Find the fastest kernel from a set of candidates.

    Args:
        kernels: Dictionary mapping kernel names to functions.
        config: Benchmark configuration.

    Returns:
        Tuple of (kernel_name, result) for the fastest kernel.
    """
    results = compare_multiple_kernels(kernels, config)

    fastest_name = min(results.keys(), key=lambda k: results[k].median_ns)
    return fastest_name, results[fastest_name]
