"""Tests for metrics exporters."""
from __future__ import annotations

import json

import pytest

from layerzero.telemetry.metrics import MetricsCollector
from layerzero.telemetry.exporters.prometheus import export_prometheus
from layerzero.telemetry.exporters.opentelemetry import export_opentelemetry


class TestPrometheusExporter:
    """Test Prometheus metrics export."""

    def test_export_prometheus_returns_string(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """export_prometheus returns string."""
        output = export_prometheus(populated_collector)
        assert isinstance(output, str)

    def test_prometheus_format_valid(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Prometheus format is valid text exposition."""
        output = export_prometheus(populated_collector)
        # Prometheus text format has lines like:
        # metric_name{label="value"} 123.45
        lines = output.strip().split("\n")
        for line in lines:
            # Skip comments and empty lines
            if line.startswith("#") or not line.strip():
                continue
            # Should have metric name and value
            parts = line.split()
            assert len(parts) >= 2

    def test_prometheus_contains_selection_count(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Prometheus export contains selection count."""
        output = export_prometheus(populated_collector)
        assert "layerzero_selections_total" in output

    def test_prometheus_contains_cache_hit_rate(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Prometheus export contains cache hit rate."""
        output = export_prometheus(populated_collector)
        assert "layerzero_cache_hit_rate" in output

    def test_prometheus_contains_latency(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Prometheus export contains latency metrics."""
        output = export_prometheus(populated_collector)
        assert "layerzero_selection_latency" in output

    def test_prometheus_contains_kernel_counts(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Prometheus export contains per-kernel counts."""
        output = export_prometheus(populated_collector)
        assert "flash_attn" in output or "kernel_id" in output

    def test_prometheus_empty_collector(
        self,
        metrics_collector: MetricsCollector,
    ) -> None:
        """Prometheus export handles empty collector."""
        output = export_prometheus(metrics_collector)
        assert isinstance(output, str)
        # Should still have metric definitions
        assert "layerzero" in output


class TestOpenTelemetryExporter:
    """Test OpenTelemetry metrics export."""

    def test_export_opentelemetry_returns_dict(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """export_opentelemetry returns dict."""
        output = export_opentelemetry(populated_collector)
        assert isinstance(output, dict)

    def test_opentelemetry_json_serializable(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """OpenTelemetry export is JSON serializable."""
        output = export_opentelemetry(populated_collector)
        json_str = json.dumps(output)
        assert isinstance(json_str, str)

    def test_opentelemetry_contains_resource(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """OpenTelemetry export contains resource info."""
        output = export_opentelemetry(populated_collector)
        assert "resource" in output or "resourceMetrics" in output

    def test_opentelemetry_contains_metrics(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """OpenTelemetry export contains metrics."""
        output = export_opentelemetry(populated_collector)
        # Should have metrics array or scopeMetrics
        has_metrics = (
            "metrics" in output or
            "scopeMetrics" in output or
            "resourceMetrics" in output
        )
        assert has_metrics

    def test_opentelemetry_contains_selection_count(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """OpenTelemetry export contains selection count metric."""
        output = export_opentelemetry(populated_collector)
        output_str = json.dumps(output)
        assert "selections" in output_str.lower()

    def test_opentelemetry_contains_cache_metric(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """OpenTelemetry export contains cache metric."""
        output = export_opentelemetry(populated_collector)
        output_str = json.dumps(output)
        assert "cache" in output_str.lower()

    def test_opentelemetry_contains_latency_metric(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """OpenTelemetry export contains latency metric."""
        output = export_opentelemetry(populated_collector)
        output_str = json.dumps(output)
        assert "latency" in output_str.lower()

    def test_opentelemetry_empty_collector(
        self,
        metrics_collector: MetricsCollector,
    ) -> None:
        """OpenTelemetry export handles empty collector."""
        output = export_opentelemetry(metrics_collector)
        assert isinstance(output, dict)


class TestExporterConsistency:
    """Test consistency between exporters."""

    def test_both_exporters_show_same_total(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Both exporters reflect same total count."""
        prom_output = export_prometheus(populated_collector)
        otel_output = export_opentelemetry(populated_collector)

        # Both should reflect 5 total selections
        assert "5" in prom_output or "5.0" in prom_output
        otel_str = json.dumps(otel_output)
        # Value should appear somewhere in OTel output
        assert "5" in otel_str

    def test_exporters_handle_same_data(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Both exporters handle the same data without error."""
        # Just ensure both work on same collector
        prom_output = export_prometheus(populated_collector)
        otel_output = export_opentelemetry(populated_collector)

        assert len(prom_output) > 0
        assert len(otel_output) > 0
