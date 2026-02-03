"""Tests for SelectionReport."""
from __future__ import annotations

import json
import time

import pytest

from layerzero.telemetry.selection_report import (
    KernelCandidate,
    SelectionReport,
)


class TestKernelCandidate:
    """Test KernelCandidate dataclass."""

    def test_candidate_creation(self) -> None:
        """KernelCandidate can be created."""
        candidate = KernelCandidate(
            kernel_id="flash_attn.fwd",
            score=0.95,
            rejected=False,
            rejection_reasons=(),
            metadata={"version": "2.6.0"},
        )
        assert candidate.kernel_id == "flash_attn.fwd"
        assert candidate.score == 0.95
        assert candidate.rejected is False
        assert candidate.rejection_reasons == ()

    def test_rejected_candidate_has_reasons(self) -> None:
        """Rejected candidate has rejection reasons."""
        candidate = KernelCandidate(
            kernel_id="xformers.cutlass",
            score=None,
            rejected=True,
            rejection_reasons=("DTYPE_NOT_SUPPORTED",),
            metadata={},
        )
        assert candidate.rejected is True
        assert "DTYPE_NOT_SUPPORTED" in candidate.rejection_reasons
        assert candidate.score is None

    def test_candidate_is_frozen(self) -> None:
        """KernelCandidate is immutable."""
        candidate = KernelCandidate(
            kernel_id="test",
            score=0.5,
            rejected=False,
            rejection_reasons=(),
            metadata={},
        )
        with pytest.raises(AttributeError):
            candidate.score = 0.9  # type: ignore


class TestSelectionReportCreation:
    """Test SelectionReport creation."""

    def test_selection_report_creation(
        self,
        sample_candidates: tuple[KernelCandidate, ...],
    ) -> None:
        """SelectionReport can be created."""
        report = SelectionReport(
            operation="attention",
            chosen_kernel_id="flash_attn.fwd",
            candidates=sample_candidates,
            selection_latency_ns=1500,
            cache_hit=False,
            timestamp=time.time(),
            context={"batch_size": 8},
        )
        assert report is not None
        assert report.operation == "attention"

    def test_selection_report_contains_chosen_kernel(
        self,
        sample_report: SelectionReport,
    ) -> None:
        """Report contains chosen kernel ID."""
        assert sample_report.chosen_kernel_id == "flash_attn.fwd"

    def test_selection_report_contains_all_candidates(
        self,
        sample_report: SelectionReport,
    ) -> None:
        """Report contains all candidate kernels."""
        assert len(sample_report.candidates) == 3
        kernel_ids = [c.kernel_id for c in sample_report.candidates]
        assert "flash_attn.fwd" in kernel_ids
        assert "torch.sdpa" in kernel_ids
        assert "xformers.cutlass" in kernel_ids

    def test_selection_report_contains_rejection_reasons(
        self,
        sample_report: SelectionReport,
    ) -> None:
        """Report contains rejection reasons per kernel."""
        rejected = [c for c in sample_report.candidates if c.rejected]
        assert len(rejected) == 1
        assert rejected[0].kernel_id == "xformers.cutlass"
        assert len(rejected[0].rejection_reasons) > 0

    def test_selection_report_contains_scores(
        self,
        sample_report: SelectionReport,
    ) -> None:
        """Report contains scores for valid kernels."""
        valid = [c for c in sample_report.candidates if not c.rejected]
        assert len(valid) == 2
        for candidate in valid:
            assert candidate.score is not None
            assert 0 <= candidate.score <= 1

    def test_selection_report_contains_latency(
        self,
        sample_report: SelectionReport,
    ) -> None:
        """Report contains selection latency in nanoseconds."""
        assert sample_report.selection_latency_ns > 0
        assert sample_report.selection_latency_ns < 1_000_000  # < 1ms

    def test_selection_report_contains_cache_hit(
        self,
        sample_report: SelectionReport,
    ) -> None:
        """Report contains cache hit status."""
        assert isinstance(sample_report.cache_hit, bool)

    def test_selection_report_is_frozen(
        self,
        sample_report: SelectionReport,
    ) -> None:
        """SelectionReport is immutable."""
        with pytest.raises(AttributeError):
            sample_report.chosen_kernel_id = "other"  # type: ignore


class TestSelectionReportSerialization:
    """Test SelectionReport JSON serialization."""

    def test_selection_report_to_json(
        self,
        sample_report: SelectionReport,
    ) -> None:
        """Report serializes to JSON."""
        json_str = sample_report.to_json()
        assert isinstance(json_str, str)
        # Should be valid JSON
        data = json.loads(json_str)
        assert "operation" in data
        assert "chosen_kernel_id" in data

    def test_selection_report_from_json(
        self,
        sample_report: SelectionReport,
    ) -> None:
        """Report deserializes from JSON."""
        json_str = sample_report.to_json()
        restored = SelectionReport.from_json(json_str)
        assert restored.operation == sample_report.operation
        assert restored.chosen_kernel_id == sample_report.chosen_kernel_id
        assert len(restored.candidates) == len(sample_report.candidates)

    def test_selection_report_roundtrip(
        self,
        sample_report: SelectionReport,
    ) -> None:
        """Report survives JSON roundtrip."""
        json_str = sample_report.to_json()
        restored = SelectionReport.from_json(json_str)

        # Check all fields
        assert restored.operation == sample_report.operation
        assert restored.chosen_kernel_id == sample_report.chosen_kernel_id
        assert restored.selection_latency_ns == sample_report.selection_latency_ns
        assert restored.cache_hit == sample_report.cache_hit
        assert restored.context == sample_report.context

        # Check candidates
        for orig, rest in zip(sample_report.candidates, restored.candidates):
            assert orig.kernel_id == rest.kernel_id
            assert orig.score == rest.score
            assert orig.rejected == rest.rejected
            assert orig.rejection_reasons == rest.rejection_reasons

    def test_selection_report_to_dict(
        self,
        sample_report: SelectionReport,
    ) -> None:
        """Report converts to dict."""
        data = sample_report.to_dict()
        assert isinstance(data, dict)
        assert data["operation"] == "attention"
        assert data["chosen_kernel_id"] == "flash_attn.fwd"
        assert "candidates" in data
        assert len(data["candidates"]) == 3


class TestSelectionReportNoKernel:
    """Test SelectionReport when no kernel is selected."""

    def test_no_kernel_chosen(
        self,
        sample_candidates: tuple[KernelCandidate, ...],
    ) -> None:
        """Report can have no chosen kernel."""
        # Mark all as rejected
        all_rejected = tuple(
            KernelCandidate(
                kernel_id=c.kernel_id,
                score=None,
                rejected=True,
                rejection_reasons=("NOT_AVAILABLE",),
                metadata=c.metadata,
            )
            for c in sample_candidates
        )
        report = SelectionReport(
            operation="attention",
            chosen_kernel_id=None,
            candidates=all_rejected,
            selection_latency_ns=500,
            cache_hit=False,
            timestamp=time.time(),
            context={},
        )
        assert report.chosen_kernel_id is None
        assert all(c.rejected for c in report.candidates)

    def test_no_kernel_serializes(
        self,
        sample_candidates: tuple[KernelCandidate, ...],
    ) -> None:
        """Report with no chosen kernel serializes correctly."""
        all_rejected = tuple(
            KernelCandidate(
                kernel_id=c.kernel_id,
                score=None,
                rejected=True,
                rejection_reasons=("NOT_AVAILABLE",),
                metadata=c.metadata,
            )
            for c in sample_candidates
        )
        report = SelectionReport(
            operation="matmul",
            chosen_kernel_id=None,
            candidates=all_rejected,
            selection_latency_ns=500,
            cache_hit=False,
            timestamp=time.time(),
            context={},
        )
        json_str = report.to_json()
        restored = SelectionReport.from_json(json_str)
        assert restored.chosen_kernel_id is None
