"""
OpenTelemetry Metrics Exporter

Exports metrics in OpenTelemetry Protocol (OTLP) JSON format.
See: https://opentelemetry.io/docs/specs/otlp/
"""
from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from layerzero.telemetry.metrics import MetricsCollector


def export_opentelemetry(collector: "MetricsCollector") -> dict[str, Any]:
    """Export metrics in OpenTelemetry JSON format.

    Exports all collected metrics in the OpenTelemetry Protocol
    (OTLP) JSON format, suitable for sending to an OTLP receiver.

    Args:
        collector: MetricsCollector with recorded metrics.

    Returns:
        OpenTelemetry-compatible dictionary.

    Example:
        ```python
        collector = MetricsCollector()
        collector.record_selection("flash_attn.fwd", 1000, True)

        output = export_opentelemetry(collector)
        # Returns OTLP-compatible dict that can be JSON-serialized
        ```
    """
    snapshot = collector.snapshot()
    timestamp_ns = int(time.time() * 1e9)

    # Build metrics list
    metrics: list[dict[str, Any]] = []

    # Total selections (Sum/Counter)
    metrics.append({
        "name": "layerzero.selections.total",
        "description": "Total number of kernel selections",
        "unit": "1",
        "sum": {
            "dataPoints": [{
                "asInt": snapshot["total_selections"],
                "startTimeUnixNano": timestamp_ns,
                "timeUnixNano": timestamp_ns,
            }],
            "aggregationTemporality": 2,  # CUMULATIVE
            "isMonotonic": True,
        },
    })

    # Cache hit rate (Gauge)
    metrics.append({
        "name": "layerzero.cache.hit_rate",
        "description": "Cache hit rate (0-1)",
        "unit": "1",
        "gauge": {
            "dataPoints": [{
                "asDouble": snapshot["cache_hit_rate"],
                "timeUnixNano": timestamp_ns,
            }],
        },
    })

    # Cache hits (Sum/Counter)
    metrics.append({
        "name": "layerzero.cache.hits",
        "description": "Total cache hits",
        "unit": "1",
        "sum": {
            "dataPoints": [{
                "asInt": snapshot["cache_hits"],
                "startTimeUnixNano": timestamp_ns,
                "timeUnixNano": timestamp_ns,
            }],
            "aggregationTemporality": 2,
            "isMonotonic": True,
        },
    })

    # Selection latency (Histogram)
    histogram = snapshot.get("latency_histogram", {})
    if histogram and snapshot.get("latency_count", 0) > 0:
        metrics.append({
            "name": "layerzero.selection.latency",
            "description": "Selection latency in nanoseconds",
            "unit": "ns",
            "histogram": {
                "dataPoints": [{
                    "startTimeUnixNano": timestamp_ns,
                    "timeUnixNano": timestamp_ns,
                    "count": snapshot.get("latency_count", 0),
                    "sum": histogram.get("mean", 0) * snapshot.get("latency_count", 0),
                    "min": histogram.get("min", 0),
                    "max": histogram.get("max", 0),
                    "quantiles": [
                        {"quantile": 0.5, "value": histogram.get("p50", 0)},
                        {"quantile": 0.95, "value": histogram.get("p95", 0)},
                        {"quantile": 0.99, "value": histogram.get("p99", 0)},
                    ],
                }],
                "aggregationTemporality": 2,
            },
        })

    # Per-kernel usage (Sum with attributes)
    kernel_counts = snapshot.get("kernel_usage_counts", {})
    if kernel_counts:
        data_points = []
        for kernel_id, count in kernel_counts.items():
            data_points.append({
                "asInt": count,
                "startTimeUnixNano": timestamp_ns,
                "timeUnixNano": timestamp_ns,
                "attributes": [{
                    "key": "kernel_id",
                    "value": {"stringValue": kernel_id},
                }],
            })

        metrics.append({
            "name": "layerzero.kernel.selections",
            "description": "Selections per kernel",
            "unit": "1",
            "sum": {
                "dataPoints": data_points,
                "aggregationTemporality": 2,
                "isMonotonic": True,
            },
        })

    # Build full OTLP structure
    return {
        "resourceMetrics": [{
            "resource": {
                "attributes": [
                    {
                        "key": "service.name",
                        "value": {"stringValue": "layerzero"},
                    },
                    {
                        "key": "service.version",
                        "value": {"stringValue": "0.1.0"},
                    },
                ],
            },
            "scopeMetrics": [{
                "scope": {
                    "name": "layerzero.telemetry",
                    "version": "0.1.0",
                },
                "metrics": metrics,
            }],
        }],
    }
