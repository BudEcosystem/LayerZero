"""Tests for Liger CrossEntropy adapter."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.liger.cross_entropy import LigerCrossEntropyAdapter
from layerzero.backends.liger.version import is_liger_available
from layerzero.backends.base import BaseKernel
from layerzero.models.kernel_spec import KernelSpec


class TestLigerCrossEntropyAdapter:
    """Test Liger CrossEntropy adapter."""

    def test_inherits_base_kernel(self) -> None:
        """Adapter inherits from BaseKernel."""
        adapter = LigerCrossEntropyAdapter()
        assert isinstance(adapter, BaseKernel)

    def test_get_kernel_spec_returns_kernel_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        adapter = LigerCrossEntropyAdapter()
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_id_contains_liger(self) -> None:
        """kernel_id contains liger."""
        adapter = LigerCrossEntropyAdapter()
        spec = adapter.get_kernel_spec()
        assert "liger" in spec.kernel_id.lower()

    def test_operation_is_loss(self) -> None:
        """Operation is loss-related."""
        adapter = LigerCrossEntropyAdapter()
        spec = adapter.get_kernel_spec()
        assert "cross_entropy" in spec.operation.lower() or "loss" in spec.operation.lower()

    def test_source_is_liger(self) -> None:
        """Source is liger."""
        adapter = LigerCrossEntropyAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.source == "liger"

    def test_supports_fp32(self) -> None:
        """Supports fp32 (standard for loss computation)."""
        adapter = LigerCrossEntropyAdapter()
        spec = adapter.get_kernel_spec()
        assert torch.float32 in spec.supported_dtypes

    def test_is_available_property(self) -> None:
        """is_available returns bool."""
        adapter = LigerCrossEntropyAdapter()
        assert isinstance(adapter.is_available, bool)

    def test_version_property(self) -> None:
        """version returns tuple or None."""
        adapter = LigerCrossEntropyAdapter()
        version = adapter.version
        assert version is None or isinstance(version, tuple)

    @pytest.mark.skipif(
        not is_liger_available(),
        reason="Liger not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_call_with_valid_input(
        self,
        sample_logits_labels: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Adapter callable with valid input."""
        adapter = LigerCrossEntropyAdapter()
        logits, labels = sample_logits_labels

        loss = adapter(logits=logits, labels=labels)

        # Loss is scalar or batch-reduced
        assert loss.numel() == 1 or loss.ndim == 0

    def test_call_without_liger_raises(self) -> None:
        """Calling without liger raises RuntimeError."""
        if is_liger_available():
            pytest.skip("Liger is installed")

        adapter = LigerCrossEntropyAdapter()
        logits = torch.randn(256, 1000, dtype=torch.float32)
        labels = torch.randint(0, 1000, (256,))

        with pytest.raises(RuntimeError, match="[Ll]iger"):
            adapter(logits=logits, labels=labels)


class TestLigerCrossEntropyCorrectness:
    """Correctness tests comparing Liger CrossEntropy to reference."""

    @pytest.mark.correctness
    @pytest.mark.skipif(
        not is_liger_available(),
        reason="Liger not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_cross_entropy_vs_pytorch_fp32(self, device: torch.device) -> None:
        """Liger CrossEntropy matches PyTorch reference within tolerance."""
        adapter = LigerCrossEntropyAdapter()

        batch_seq, vocab_size = 256, 1000
        torch.manual_seed(42)

        logits = torch.randn(batch_seq, vocab_size, device=device, dtype=torch.float32)
        labels = torch.randint(0, vocab_size, (batch_seq,), device=device)

        # Liger output
        liger_loss = adapter(logits=logits, labels=labels)

        # PyTorch reference
        ref_loss = torch.nn.functional.cross_entropy(logits, labels)

        # Compare within tolerance
        diff = (liger_loss - ref_loss).abs().item()
        assert diff < 0.001, f"Liger vs PyTorch CrossEntropy diff: {diff}"

    @pytest.mark.correctness
    @pytest.mark.skipif(
        not is_liger_available(),
        reason="Liger not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_cross_entropy_with_ignore_index(self, device: torch.device) -> None:
        """Liger CrossEntropy handles ignore_index correctly."""
        adapter = LigerCrossEntropyAdapter()

        batch_seq, vocab_size = 256, 1000
        ignore_index = -100
        torch.manual_seed(42)

        logits = torch.randn(batch_seq, vocab_size, device=device, dtype=torch.float32)
        labels = torch.randint(0, vocab_size, (batch_seq,), device=device)
        # Set some labels to ignore_index
        labels[::10] = ignore_index

        # Liger output
        liger_loss = adapter(logits=logits, labels=labels, ignore_index=ignore_index)

        # PyTorch reference
        ref_loss = torch.nn.functional.cross_entropy(
            logits, labels, ignore_index=ignore_index
        )

        # Compare within tolerance
        diff = (liger_loss - ref_loss).abs().item()
        assert diff < 0.001, f"Liger vs PyTorch CrossEntropy with ignore_index diff: {diff}"
