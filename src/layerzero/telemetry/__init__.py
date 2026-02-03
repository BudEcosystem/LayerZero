"""
LayerZero Telemetry and Explainability

Provides full visibility into kernel selection decisions:
- SelectionReport with detailed selection trace
- lz.explain() API for debugging
- Metrics collection and export (Prometheus, OpenTelemetry)
"""
from layerzero.telemetry.selection_report import (
    KernelCandidate,
    SelectionReport,
)
from layerzero.telemetry.metrics import MetricsCollector
from layerzero.telemetry.explain import explain
from layerzero.telemetry.exporters import (
    export_prometheus,
    export_opentelemetry,
)

__all__ = [
    "KernelCandidate",
    "SelectionReport",
    "MetricsCollector",
    "explain",
    "export_prometheus",
    "export_opentelemetry",
]
