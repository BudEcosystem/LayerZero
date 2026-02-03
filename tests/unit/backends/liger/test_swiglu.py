"""Tests for Liger SwiGLU adapter."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.liger.swiglu import LigerSwiGLUAdapter
from layerzero.backends.liger.version import is_liger_available
from layerzero.backends.base import BaseKernel
from layerzero.models.kernel_spec import KernelSpec


class TestLigerSwiGLUAdapter:
    """Test Liger SwiGLU adapter."""

    def test_inherits_base_kernel(self) -> None:
        """Adapter inherits from BaseKernel."""
        adapter = LigerSwiGLUAdapter()
        assert isinstance(adapter, BaseKernel)

    def test_get_kernel_spec_returns_kernel_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        adapter = LigerSwiGLUAdapter()
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_id_contains_liger(self) -> None:
        """kernel_id contains liger."""
        adapter = LigerSwiGLUAdapter()
        spec = adapter.get_kernel_spec()
        assert "liger" in spec.kernel_id.lower()

    def test_operation_is_activation(self) -> None:
        """Operation is activation-related."""
        adapter = LigerSwiGLUAdapter()
        spec = adapter.get_kernel_spec()
        assert "swiglu" in spec.operation.lower() or "activation" in spec.operation.lower()

    def test_source_is_liger(self) -> None:
        """Source is liger."""
        adapter = LigerSwiGLUAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.source == "liger"

    def test_supports_fp16_bf16_fp32(self) -> None:
        """Supports fp16, bf16, and fp32."""
        adapter = LigerSwiGLUAdapter()
        spec = adapter.get_kernel_spec()
        assert torch.float16 in spec.supported_dtypes
        assert torch.bfloat16 in spec.supported_dtypes
        assert torch.float32 in spec.supported_dtypes

    def test_is_available_property(self) -> None:
        """is_available returns bool."""
        adapter = LigerSwiGLUAdapter()
        assert isinstance(adapter.is_available, bool)

    def test_version_property(self) -> None:
        """version returns tuple or None."""
        adapter = LigerSwiGLUAdapter()
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
        sample_gate_up: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Adapter callable with valid input."""
        adapter = LigerSwiGLUAdapter()
        gate, up = sample_gate_up

        output = adapter(gate=gate, up=up)

        # Output same shape as input
        assert output.shape == gate.shape

    def test_call_without_liger_raises(self) -> None:
        """Calling without liger raises RuntimeError."""
        if is_liger_available():
            pytest.skip("Liger is installed")

        adapter = LigerSwiGLUAdapter()
        gate = torch.randn(2, 16, 768, dtype=torch.float16)
        up = torch.randn(2, 16, 768, dtype=torch.float16)

        with pytest.raises(RuntimeError, match="[Ll]iger"):
            adapter(gate=gate, up=up)


class TestLigerSwiGLUCorrectness:
    """Correctness tests comparing Liger SwiGLU to reference."""

    @pytest.mark.correctness
    @pytest.mark.skipif(
        not is_liger_available(),
        reason="Liger not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_swiglu_vs_pytorch_fp16(self, device: torch.device) -> None:
        """Liger SwiGLU matches PyTorch reference within fp16 tolerance."""
        adapter = LigerSwiGLUAdapter()

        batch, seq, hidden_dim = 2, 64, 256
        torch.manual_seed(42)

        gate = torch.randn(batch, seq, hidden_dim, device=device, dtype=torch.float16)
        up = torch.randn(batch, seq, hidden_dim, device=device, dtype=torch.float16)

        # Liger output
        liger_output = adapter(gate=gate, up=up)

        # PyTorch reference: SwiGLU = SiLU(gate) * up
        ref_output = torch.nn.functional.silu(gate) * up

        # Compare within tolerance
        max_diff = (liger_output - ref_output).abs().max().item()
        assert max_diff < 0.01, f"Liger vs PyTorch SwiGLU max diff: {max_diff}"

    @pytest.mark.correctness
    @pytest.mark.skipif(
        not is_liger_available(),
        reason="Liger not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_swiglu_vs_pytorch_bf16(self, device: torch.device) -> None:
        """Liger SwiGLU matches PyTorch reference within bf16 tolerance."""
        adapter = LigerSwiGLUAdapter()

        batch, seq, hidden_dim = 2, 64, 256
        torch.manual_seed(42)

        gate = torch.randn(batch, seq, hidden_dim, device=device, dtype=torch.bfloat16)
        up = torch.randn(batch, seq, hidden_dim, device=device, dtype=torch.bfloat16)

        # Liger output
        liger_output = adapter(gate=gate, up=up)

        # PyTorch reference
        ref_output = torch.nn.functional.silu(gate) * up

        # BF16 has lower precision
        max_diff = (liger_output - ref_output).abs().max().item()
        assert max_diff < 0.02, f"Liger vs PyTorch SwiGLU max diff: {max_diff}"
