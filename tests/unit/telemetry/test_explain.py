"""Tests for lz.explain() API."""
from __future__ import annotations

import pytest
import torch

from layerzero.telemetry.explain import explain
from layerzero.telemetry.selection_report import SelectionReport


class TestExplainAPI:
    """Test lz.explain() API."""

    def test_explain_returns_report(self) -> None:
        """lz.explain() returns SelectionReport."""
        report = explain("attention")
        assert isinstance(report, SelectionReport)

    def test_explain_with_operation(self) -> None:
        """explain() works with operation name."""
        report = explain("attention")
        assert report.operation == "attention"

    def test_explain_with_context(self) -> None:
        """explain() accepts context kwargs."""
        report = explain(
            "attention",
            batch_size=8,
            seq_len=512,
            dtype="float16",
        )
        assert "batch_size" in report.context
        assert report.context["batch_size"] == 8

    def test_explain_has_candidates(self) -> None:
        """explain() report has candidates."""
        report = explain("attention")
        assert len(report.candidates) > 0

    def test_explain_has_latency(self) -> None:
        """explain() report has selection latency."""
        report = explain("attention")
        assert report.selection_latency_ns >= 0


class TestExplainWithTensors:
    """Test lz.explain() with tensor inputs."""

    @pytest.fixture
    def sample_tensors(self) -> dict[str, torch.Tensor]:
        """Sample QKV tensors."""
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64

        return {
            "query": torch.randn(batch_size, num_heads, seq_len, head_dim),
            "key": torch.randn(batch_size, num_heads, seq_len, head_dim),
            "value": torch.randn(batch_size, num_heads, seq_len, head_dim),
        }

    def test_explain_with_tensors(
        self,
        sample_tensors: dict[str, torch.Tensor],
    ) -> None:
        """lz.explain() works with tensor inputs."""
        report = explain(
            "attention",
            sample_tensors["query"],
            sample_tensors["key"],
            sample_tensors["value"],
        )
        assert isinstance(report, SelectionReport)

    def test_explain_infers_dtype_from_tensors(
        self,
        sample_tensors: dict[str, torch.Tensor],
    ) -> None:
        """explain() infers dtype from input tensors."""
        report = explain(
            "attention",
            sample_tensors["query"],
            sample_tensors["key"],
            sample_tensors["value"],
        )
        # Should have inferred dtype
        assert "dtype" in report.context or "inferred_dtype" in report.context

    def test_explain_infers_device_from_tensors(
        self,
        sample_tensors: dict[str, torch.Tensor],
    ) -> None:
        """explain() infers device from input tensors."""
        report = explain(
            "attention",
            sample_tensors["query"],
            sample_tensors["key"],
            sample_tensors["value"],
        )
        # Should have inferred device
        assert "device" in report.context or "inferred_device" in report.context


class TestExplainRejectionReasons:
    """Test explain() rejection reason reporting."""

    def test_explain_shows_rejection_reasons(self) -> None:
        """explain() shows why kernels were rejected."""
        report = explain("attention")
        rejected = [c for c in report.candidates if c.rejected]
        # May or may not have rejected candidates depending on environment
        for candidate in rejected:
            assert len(candidate.rejection_reasons) > 0

    def test_rejection_reasons_are_strings(self) -> None:
        """Rejection reasons are strings."""
        report = explain("attention")
        for candidate in report.candidates:
            for reason in candidate.rejection_reasons:
                assert isinstance(reason, str)


class TestExplainScores:
    """Test explain() score reporting."""

    def test_explain_shows_scores(self) -> None:
        """explain() shows kernel scores."""
        report = explain("attention")
        valid = [c for c in report.candidates if not c.rejected]
        for candidate in valid:
            assert candidate.score is not None

    def test_scores_are_normalized(self) -> None:
        """Scores are between 0 and 1."""
        report = explain("attention")
        valid = [c for c in report.candidates if not c.rejected]
        for candidate in valid:
            if candidate.score is not None:
                assert 0.0 <= candidate.score <= 1.0


class TestExplainPrettyPrint:
    """Test explain() pretty printing."""

    def test_explain_str(self) -> None:
        """explain() result has readable string representation."""
        report = explain("attention")
        report_str = str(report)
        assert "attention" in report_str.lower()

    def test_explain_pretty_print(self) -> None:
        """explain() has pretty_print method."""
        report = explain("attention")
        assert hasattr(report, "pretty_print") or hasattr(report, "format")

    def test_explain_summary(self) -> None:
        """explain() has summary method."""
        report = explain("attention")
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestExplainOperations:
    """Test explain() with different operations."""

    @pytest.mark.parametrize("operation", [
        "attention",
        "matmul",
        "layer_norm",
        "rope",
        "softmax",
    ])
    def test_explain_various_operations(self, operation: str) -> None:
        """explain() works with various operations."""
        report = explain(operation)
        assert report.operation == operation
        assert isinstance(report, SelectionReport)
