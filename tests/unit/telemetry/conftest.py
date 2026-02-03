"""Pytest fixtures for telemetry tests."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from layerzero.telemetry.selection_report import (
    KernelCandidate,
    SelectionReport,
)
from layerzero.telemetry.metrics import MetricsCollector

if TYPE_CHECKING:
    pass


@pytest.fixture
def sample_candidates() -> tuple[KernelCandidate, ...]:
    """Sample kernel candidates for testing."""
    return (
        KernelCandidate(
            kernel_id="flash_attn.fwd",
            score=0.95,
            rejected=False,
            rejection_reasons=(),
            metadata={"version": "2.6.0"},
        ),
        KernelCandidate(
            kernel_id="torch.sdpa",
            score=0.75,
            rejected=False,
            rejection_reasons=(),
            metadata={"version": "2.4.0"},
        ),
        KernelCandidate(
            kernel_id="xformers.cutlass",
            score=None,
            rejected=True,
            rejection_reasons=("DTYPE_NOT_SUPPORTED", "CUDA_NOT_AVAILABLE"),
            metadata={"version": "0.0.28"},
        ),
    )


@pytest.fixture
def sample_report(sample_candidates: tuple[KernelCandidate, ...]) -> SelectionReport:
    """Sample selection report for testing."""
    return SelectionReport(
        operation="attention",
        chosen_kernel_id="flash_attn.fwd",
        candidates=sample_candidates,
        selection_latency_ns=1500,
        cache_hit=False,
        timestamp=time.time(),
        context={
            "batch_size": 8,
            "seq_len": 512,
            "dtype": "float16",
        },
    )


@pytest.fixture
def metrics_collector() -> MetricsCollector:
    """Fresh metrics collector for testing."""
    return MetricsCollector()


@pytest.fixture
def populated_collector() -> MetricsCollector:
    """Metrics collector with sample data."""
    collector = MetricsCollector()
    # Simulate some selections
    collector.record_selection("flash_attn.fwd", latency_ns=1000, cache_hit=True)
    collector.record_selection("flash_attn.fwd", latency_ns=1200, cache_hit=True)
    collector.record_selection("torch.sdpa", latency_ns=2000, cache_hit=False)
    collector.record_selection("flash_attn.fwd", latency_ns=800, cache_hit=True)
    collector.record_selection("torch.sdpa", latency_ns=2500, cache_hit=False)
    return collector
