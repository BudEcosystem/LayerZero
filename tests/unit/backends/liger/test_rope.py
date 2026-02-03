"""Tests for Liger RoPE adapter."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.liger.rope import LigerRoPEAdapter
from layerzero.backends.liger.version import is_liger_available
from layerzero.backends.base import BaseKernel
from layerzero.models.kernel_spec import KernelSpec


class TestLigerRoPEAdapter:
    """Test Liger RoPE adapter."""

    def test_inherits_base_kernel(self) -> None:
        """Adapter inherits from BaseKernel."""
        adapter = LigerRoPEAdapter()
        assert isinstance(adapter, BaseKernel)

    def test_get_kernel_spec_returns_kernel_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        adapter = LigerRoPEAdapter()
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_id_contains_liger(self) -> None:
        """kernel_id contains liger."""
        adapter = LigerRoPEAdapter()
        spec = adapter.get_kernel_spec()
        assert "liger" in spec.kernel_id.lower()

    def test_operation_is_rope(self) -> None:
        """Operation is rope-related."""
        adapter = LigerRoPEAdapter()
        spec = adapter.get_kernel_spec()
        assert "rope" in spec.operation.lower() or "embedding" in spec.operation.lower()

    def test_source_is_liger(self) -> None:
        """Source is liger."""
        adapter = LigerRoPEAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.source == "liger"

    def test_supports_fp16_bf16_fp32(self) -> None:
        """Supports fp16, bf16, and fp32."""
        adapter = LigerRoPEAdapter()
        spec = adapter.get_kernel_spec()
        assert torch.float16 in spec.supported_dtypes
        assert torch.bfloat16 in spec.supported_dtypes
        assert torch.float32 in spec.supported_dtypes

    def test_is_available_property(self) -> None:
        """is_available returns bool."""
        adapter = LigerRoPEAdapter()
        assert isinstance(adapter.is_available, bool)

    def test_version_property(self) -> None:
        """version returns tuple or None."""
        adapter = LigerRoPEAdapter()
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
        sample_qk_for_rope: tuple[torch.Tensor, torch.Tensor],
        sample_rope_cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Adapter callable with valid input."""
        adapter = LigerRoPEAdapter()
        q, k = sample_qk_for_rope
        cos, sin = sample_rope_cos_sin

        q_out, k_out = adapter(q=q, k=k, cos=cos, sin=sin)

        # Output same shape as input
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_call_without_liger_raises(self) -> None:
        """Calling without liger raises RuntimeError."""
        if is_liger_available():
            pytest.skip("Liger is installed")

        adapter = LigerRoPEAdapter()
        q = torch.randn(2, 16, 8, 64, dtype=torch.float16)
        k = torch.randn(2, 16, 8, 64, dtype=torch.float16)
        cos = torch.ones(16, 64, dtype=torch.float16)
        sin = torch.zeros(16, 64, dtype=torch.float16)

        with pytest.raises(RuntimeError, match="[Ll]iger"):
            adapter(q=q, k=k, cos=cos, sin=sin)


class TestLigerRoPECorrectness:
    """Correctness tests comparing Liger RoPE to reference."""

    @pytest.mark.correctness
    @pytest.mark.skipif(
        not is_liger_available(),
        reason="Liger not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_rope_vs_pytorch_fp16(self, device: torch.device) -> None:
        """Liger RoPE matches PyTorch reference within fp16 tolerance."""
        adapter = LigerRoPEAdapter()

        batch, seq, heads, dim = 1, 32, 4, 64
        torch.manual_seed(42)

        q = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float16)

        # Create cos/sin
        positions = torch.arange(seq, device=device, dtype=torch.float32)
        freqs = torch.arange(dim // 2, device=device, dtype=torch.float32)
        freqs = 1.0 / (10000 ** (2 * freqs / dim))
        angles = positions[:, None] * freqs[None, :]
        cos = torch.cos(angles).repeat(1, 2).to(torch.float16)
        sin = torch.sin(angles).repeat(1, 2).to(torch.float16)

        # Liger output
        q_liger, k_liger = adapter(q=q, k=k, cos=cos, sin=sin)

        # PyTorch reference (standard RoPE formula)
        def apply_rope_pytorch(x, cos, sin):
            # Split into halves for rotation
            x1 = x[..., : dim // 2]
            x2 = x[..., dim // 2 :]
            cos_half = cos[:, : dim // 2].unsqueeze(0).unsqueeze(2)
            sin_half = sin[:, : dim // 2].unsqueeze(0).unsqueeze(2)
            out1 = x1 * cos_half - x2 * sin_half
            out2 = x2 * cos_half + x1 * sin_half
            return torch.cat([out1, out2], dim=-1)

        q_ref = apply_rope_pytorch(q, cos, sin)
        k_ref = apply_rope_pytorch(k, cos, sin)

        # Compare within tolerance
        max_diff_q = (q_liger - q_ref).abs().max().item()
        max_diff_k = (k_liger - k_ref).abs().max().item()
        assert max_diff_q < 0.01, f"Liger vs PyTorch RoPE Q max diff: {max_diff_q}"
        assert max_diff_k < 0.01, f"Liger vs PyTorch RoPE K max diff: {max_diff_k}"
