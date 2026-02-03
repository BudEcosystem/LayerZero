"""
Prometheus Metrics Exporter

Exports metrics in Prometheus text exposition format.
See: https://prometheus.io/docs/instrumenting/exposition_formats/
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layerzero.telemetry.metrics import MetricsCollector


def export_prometheus(collector: "MetricsCollector") -> str:
    """Export metrics in Prometheus text format.

    Exports all collected metrics in the Prometheus text exposition
    format, suitable for scraping by a Prometheus server.

    Args:
        collector: MetricsCollector with recorded metrics.

    Returns:
        Prometheus-compatible text format string.

    Example:
        ```python
        collector = MetricsCollector()
        collector.record_selection("flash_attn.fwd", 1000, True)

        output = export_prometheus(collector)
        # Returns:
        # # HELP layerzero_selections_total Total kernel selections
        # # TYPE layerzero_selections_total counter
        # layerzero_selections_total 1
        # ...
        ```
    """
    lines: list[str] = []
    snapshot = collector.snapshot()

    # Total selections counter
    lines.append("# HELP layerzero_selections_total Total number of kernel selections")
    lines.append("# TYPE layerzero_selections_total counter")
    lines.append(f"layerzero_selections_total {snapshot['total_selections']}")
    lines.append("")

    # Cache hit rate gauge
    lines.append("# HELP layerzero_cache_hit_rate Cache hit rate (0-1)")
    lines.append("# TYPE layerzero_cache_hit_rate gauge")
    lines.append(f"layerzero_cache_hit_rate {snapshot['cache_hit_rate']:.6f}")
    lines.append("")

    # Cache hits counter
    lines.append("# HELP layerzero_cache_hits_total Total cache hits")
    lines.append("# TYPE layerzero_cache_hits_total counter")
    lines.append(f"layerzero_cache_hits_total {snapshot['cache_hits']}")
    lines.append("")

    # Selection latency histogram
    histogram = snapshot.get("latency_histogram", {})
    if histogram:
        lines.append("# HELP layerzero_selection_latency_ns Selection latency in nanoseconds")
        lines.append("# TYPE layerzero_selection_latency_ns summary")
        lines.append(f'layerzero_selection_latency_ns{{quantile="0.5"}} {histogram.get("p50", 0)}')
        lines.append(f'layerzero_selection_latency_ns{{quantile="0.95"}} {histogram.get("p95", 0)}')
        lines.append(f'layerzero_selection_latency_ns{{quantile="0.99"}} {histogram.get("p99", 0)}')
        lines.append(f"layerzero_selection_latency_ns_sum {histogram.get('mean', 0) * snapshot.get('latency_count', 0)}")
        lines.append(f"layerzero_selection_latency_ns_count {snapshot.get('latency_count', 0)}")
        lines.append("")

    # Per-kernel usage counts
    kernel_counts = snapshot.get("kernel_usage_counts", {})
    if kernel_counts:
        lines.append("# HELP layerzero_kernel_selections_total Selections per kernel")
        lines.append("# TYPE layerzero_kernel_selections_total counter")
        for kernel_id, count in sorted(kernel_counts.items()):
            # Escape kernel_id for Prometheus labels
            safe_id = kernel_id.replace('"', '\\"')
            lines.append(f'layerzero_kernel_selections_total{{kernel_id="{safe_id}"}} {count}')
        lines.append("")

    return "\n".join(lines)
