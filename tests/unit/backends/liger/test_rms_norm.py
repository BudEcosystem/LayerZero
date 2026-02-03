"""Tests for Liger RMSNorm adapter."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.liger.rms_norm import LigerRMSNormAdapter
from layerzero.backends.liger.version import is_liger_available
from layerzero.backends.base import BaseKernel
from layerzero.models.kernel_spec import KernelSpec


class TestLigerRMSNormAdapter:
    """Test Liger RMSNorm adapter."""

    def test_inherits_base_kernel(self) -> None:
        """Adapter inherits from BaseKernel."""
        adapter = LigerRMSNormAdapter()
        assert isinstance(adapter, BaseKernel)

    def test_get_kernel_spec_returns_kernel_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        adapter = LigerRMSNormAdapter()
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_id_contains_liger(self) -> None:
        """kernel_id contains liger."""
        adapter = LigerRMSNormAdapter()
        spec = adapter.get_kernel_spec()
        assert "liger" in spec.kernel_id.lower()

    def test_operation_is_norm(self) -> None:
        """Operation is norm-related."""
        adapter = LigerRMSNormAdapter()
        spec = adapter.get_kernel_spec()
        assert "norm" in spec.operation.lower() or "rms" in spec.operation.lower()

    def test_source_is_liger(self) -> None:
        """Source is liger."""
        adapter = LigerRMSNormAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.source == "liger"

    def test_supports_fp16_bf16_fp32(self) -> None:
        """Supports fp16, bf16, and fp32."""
        adapter = LigerRMSNormAdapter()
        spec = adapter.get_kernel_spec()
        assert torch.float16 in spec.supported_dtypes
        assert torch.bfloat16 in spec.supported_dtypes
        assert torch.float32 in spec.supported_dtypes

    def test_is_available_property(self) -> None:
        """is_available returns bool."""
        adapter = LigerRMSNormAdapter()
        assert isinstance(adapter.is_available, bool)

    def test_version_property(self) -> None:
        """version returns tuple or None."""
        adapter = LigerRMSNormAdapter()
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
        sample_hidden_states: torch.Tensor,
        sample_rms_norm_weight: torch.Tensor,
    ) -> None:
        """Adapter callable with valid input."""
        adapter = LigerRMSNormAdapter()
        output = adapter(
            hidden_states=sample_hidden_states,
            weight=sample_rms_norm_weight,
            eps=1e-5,
        )

        # Output same shape as input
        assert output.shape == sample_hidden_states.shape

    def test_call_without_liger_raises(self) -> None:
        """Calling without liger raises RuntimeError."""
        if is_liger_available():
            pytest.skip("Liger is installed")

        adapter = LigerRMSNormAdapter()
        hidden_states = torch.randn(2, 16, 768, dtype=torch.float16)
        weight = torch.ones(768, dtype=torch.float16)

        with pytest.raises(RuntimeError, match="[Ll]iger"):
            adapter(hidden_states=hidden_states, weight=weight, eps=1e-5)


class TestLigerRMSNormCorrectness:
    """Correctness tests comparing Liger RMSNorm to reference."""

    @pytest.mark.correctness
    @pytest.mark.skipif(
        not is_liger_available(),
        reason="Liger not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_rms_norm_vs_pytorch_fp16(self, device: torch.device) -> None:
        """Liger RMSNorm matches PyTorch reference within fp16 tolerance."""
        adapter = LigerRMSNormAdapter()

        batch, seq, hidden_dim = 2, 64, 256
        eps = 1e-5
        torch.manual_seed(42)

        hidden_states = torch.randn(batch, seq, hidden_dim, device=device, dtype=torch.float16)
        weight = torch.ones(hidden_dim, device=device, dtype=torch.float16)

        # Liger output
        liger_output = adapter(hidden_states=hidden_states, weight=weight, eps=eps)

        # PyTorch reference
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        ref_output = hidden_states * torch.rsqrt(variance + eps) * weight

        # Compare within tolerance
        max_diff = (liger_output - ref_output).abs().max().item()
        assert max_diff < 0.01, f"Liger vs PyTorch RMSNorm max diff: {max_diff}"

    @pytest.mark.correctness
    @pytest.mark.skipif(
        not is_liger_available(),
        reason="Liger not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_rms_norm_vs_pytorch_bf16(self, device: torch.device) -> None:
        """Liger RMSNorm matches PyTorch reference within bf16 tolerance."""
        adapter = LigerRMSNormAdapter()

        batch, seq, hidden_dim = 2, 64, 256
        eps = 1e-5
        torch.manual_seed(42)

        hidden_states = torch.randn(batch, seq, hidden_dim, device=device, dtype=torch.bfloat16)
        weight = torch.ones(hidden_dim, device=device, dtype=torch.bfloat16)

        # Liger output
        liger_output = adapter(hidden_states=hidden_states, weight=weight, eps=eps)

        # PyTorch reference
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        ref_output = hidden_states * torch.rsqrt(variance + eps) * weight

        # BF16 has lower precision
        max_diff = (liger_output - ref_output).abs().max().item()
        assert max_diff < 0.02, f"Liger vs PyTorch RMSNorm max diff: {max_diff}"
