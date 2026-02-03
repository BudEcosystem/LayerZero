"""
Statistical Analysis for Benchmarks

Provides statistical functions for benchmark analysis.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def calculate_percentile(values: list[float], percentile: float) -> float:
    """Calculate percentile value from a list.

    Uses linear interpolation between data points.

    Args:
        values: List of numeric values.
        percentile: Percentile to calculate (0-100).

    Returns:
        The percentile value.

    Raises:
        ValueError: If values is empty or percentile is out of range.
    """
    if not values:
        raise ValueError("Cannot calculate percentile of empty list")

    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n == 1:
        return sorted_values[0]

    # Calculate index
    index = (percentile / 100) * (n - 1)
    lower = int(index)
    upper = lower + 1

    if upper >= n:
        return sorted_values[-1]

    # Linear interpolation
    fraction = index - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def calculate_variance(values: list[float]) -> float:
    """Calculate population variance.

    Args:
        values: List of numeric values.

    Returns:
        Population variance.

    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("Cannot calculate variance of empty list")

    n = len(values)
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / n


def calculate_std(values: list[float]) -> float:
    """Calculate population standard deviation.

    Args:
        values: List of numeric values.

    Returns:
        Population standard deviation.
    """
    return math.sqrt(calculate_variance(values))


def calculate_mean(values: list[float]) -> float:
    """Calculate mean.

    Args:
        values: List of numeric values.

    Returns:
        Mean value.

    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("Cannot calculate mean of empty list")

    return sum(values) / len(values)


def is_statistically_significant(
    a: list[float],
    b: list[float],
    p_threshold: float = 0.05,
) -> tuple[bool, float]:
    """Check if difference between distributions is statistically significant.

    Uses Welch's t-test for unequal variances.

    Args:
        a: First distribution samples.
        b: Second distribution samples.
        p_threshold: P-value threshold for significance.

    Returns:
        Tuple of (is_significant, p_value).
    """
    if len(a) < 2 or len(b) < 2:
        return False, 1.0

    # Calculate means and variances
    mean_a = calculate_mean(a)
    mean_b = calculate_mean(b)

    var_a = calculate_variance(a)
    var_b = calculate_variance(b)

    n_a = len(a)
    n_b = len(b)

    # Handle zero variance case (all values identical)
    if var_a == 0 and var_b == 0:
        # Both have zero variance - if means are equal, not significant
        # If means are different, definitely significant
        if mean_a == mean_b:
            return False, 1.0
        else:
            return True, 0.0

    # Welch's t-test
    se_a = var_a / n_a
    se_b = var_b / n_b
    se_total = se_a + se_b

    if se_total == 0:
        # Shouldn't happen after above check, but safety fallback
        return mean_a != mean_b, 0.0 if mean_a != mean_b else 1.0

    t_stat = abs(mean_a - mean_b) / math.sqrt(se_total)

    # Calculate degrees of freedom (Welch-Satterthwaite)
    if se_a == 0:
        df = n_b - 1
    elif se_b == 0:
        df = n_a - 1
    else:
        df = (se_total ** 2) / (
            (se_a ** 2) / (n_a - 1) + (se_b ** 2) / (n_b - 1)
        )

    # Approximate p-value using normal approximation for large df
    # For small df, this is less accurate but still gives reasonable results
    p_value = 2 * (1 - _normal_cdf(t_stat))

    return p_value < p_threshold, p_value


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF.

    Uses error function approximation.

    Args:
        x: Value to evaluate.

    Returns:
        Approximate CDF value.
    """
    # Approximation using tanh
    return 0.5 * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def calculate_confidence_interval(
    values: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate confidence interval for mean.

    Args:
        values: List of numeric values.
        confidence: Confidence level (e.g., 0.95 for 95%).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if len(values) < 2:
        return values[0] if values else 0.0, values[0] if values else 0.0

    mean = calculate_mean(values)
    std = calculate_std(values)
    n = len(values)

    # Z-score for confidence level (approximate)
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    z = z_scores.get(confidence, 1.96)

    margin = z * std / math.sqrt(n)

    return mean - margin, mean + margin


def calculate_coefficient_of_variation(values: list[float]) -> float:
    """Calculate coefficient of variation (CV).

    CV = std / mean, expressed as a ratio.

    Args:
        values: List of numeric values.

    Returns:
        Coefficient of variation.
    """
    mean = calculate_mean(values)
    if mean == 0:
        return 0.0

    std = calculate_std(values)
    return std / abs(mean)
