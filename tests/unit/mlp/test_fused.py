"""Fused MLP Tests for LayerZero.

Tests for fused MLP operations including:
- Gate + Up projection with activation (SwiGLU, GELU, etc.)
- Down projection
- Fused gate-up-act-down pattern (LLaMA-style MLP)
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F
from typing import Generator


class TestSwiGLUCorrectness:
    """Tests for SwiGLU activation correctness."""

    def test_swiglu_basic(self) -> None:
        """Basic SwiGLU computation: silu(gate) * up."""
        from layerzero.mlp.fused import swiglu

        batch_size = 2
        seq_len = 16
        hidden_dim = 64

        gate = torch.randn(batch_size, seq_len, hidden_dim)
        up = torch.randn(batch_size, seq_len, hidden_dim)

        result = swiglu(gate, up)

        # Reference implementation
        expected = F.silu(gate) * up

        assert result.shape == (batch_size, seq_len, hidden_dim)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_swiglu_inplace(self) -> None:
        """SwiGLU with inplace operation."""
        from layerzero.mlp.fused import swiglu

        gate = torch.randn(2, 16, 64)
        up = torch.randn(2, 16, 64)

        # Non-inplace
        result1 = swiglu(gate.clone(), up.clone(), inplace=False)

        # Inplace should give same result
        gate_clone = gate.clone()
        result2 = swiglu(gate_clone, up, inplace=True)

        assert torch.allclose(result1, result2, atol=1e-5)

    def test_swiglu_gradient(self) -> None:
        """SwiGLU supports gradient computation."""
        from layerzero.mlp.fused import swiglu

        gate = torch.randn(2, 16, 64, requires_grad=True)
        up = torch.randn(2, 16, 64, requires_grad=True)

        result = swiglu(gate, up)
        loss = result.sum()
        loss.backward()

        assert gate.grad is not None
        assert up.grad is not None
        assert not torch.isnan(gate.grad).any()
        assert not torch.isnan(up.grad).any()


class TestGELUGatedCorrectness:
    """Tests for GELU-gated (GeGLU) activation correctness."""

    def test_geglu_basic(self) -> None:
        """Basic GeGLU computation: gelu(gate) * up."""
        from layerzero.mlp.fused import geglu

        batch_size = 2
        seq_len = 16
        hidden_dim = 64

        gate = torch.randn(batch_size, seq_len, hidden_dim)
        up = torch.randn(batch_size, seq_len, hidden_dim)

        result = geglu(gate, up)

        # Reference implementation
        expected = F.gelu(gate) * up

        assert result.shape == (batch_size, seq_len, hidden_dim)
        assert torch.allclose(result, expected, atol=1e-5)


class TestReGLUCorrectness:
    """Tests for ReLU-gated (ReGLU) activation correctness."""

    def test_reglu_basic(self) -> None:
        """Basic ReGLU computation: relu(gate) * up."""
        from layerzero.mlp.fused import reglu

        batch_size = 2
        seq_len = 16
        hidden_dim = 64

        gate = torch.randn(batch_size, seq_len, hidden_dim)
        up = torch.randn(batch_size, seq_len, hidden_dim)

        result = reglu(gate, up)

        # Reference implementation
        expected = F.relu(gate) * up

        assert result.shape == (batch_size, seq_len, hidden_dim)
        assert torch.allclose(result, expected, atol=1e-5)


class TestFusedMLPCorrectness:
    """Tests for fused MLP layer correctness."""

    def test_fused_mlp_swiglu(self) -> None:
        """Fused MLP with SwiGLU activation."""
        from layerzero.mlp.fused import fused_mlp

        batch_size = 2
        seq_len = 16
        hidden_dim = 64
        intermediate_dim = 256

        x = torch.randn(batch_size, seq_len, hidden_dim)
        gate_proj = torch.randn(intermediate_dim, hidden_dim)
        up_proj = torch.randn(intermediate_dim, hidden_dim)
        down_proj = torch.randn(hidden_dim, intermediate_dim)

        result = fused_mlp(
            x, gate_proj, up_proj, down_proj,
            activation="swiglu"
        )

        # Reference implementation
        gate = x @ gate_proj.t()
        up = x @ up_proj.t()
        hidden = F.silu(gate) * up
        expected = hidden @ down_proj.t()

        assert result.shape == (batch_size, seq_len, hidden_dim)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_fused_mlp_gelu(self) -> None:
        """Fused MLP with GELU activation."""
        from layerzero.mlp.fused import fused_mlp

        batch_size = 2
        seq_len = 16
        hidden_dim = 64
        intermediate_dim = 256

        x = torch.randn(batch_size, seq_len, hidden_dim)
        gate_proj = torch.randn(intermediate_dim, hidden_dim)
        up_proj = torch.randn(intermediate_dim, hidden_dim)
        down_proj = torch.randn(hidden_dim, intermediate_dim)

        result = fused_mlp(
            x, gate_proj, up_proj, down_proj,
            activation="geglu"
        )

        # Reference implementation
        gate = x @ gate_proj.t()
        up = x @ up_proj.t()
        hidden = F.gelu(gate) * up
        expected = hidden @ down_proj.t()

        assert result.shape == (batch_size, seq_len, hidden_dim)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_fused_mlp_no_gate(self) -> None:
        """Fused MLP without gating (standard GELU)."""
        from layerzero.mlp.fused import fused_mlp

        batch_size = 2
        seq_len = 16
        hidden_dim = 64
        intermediate_dim = 256

        x = torch.randn(batch_size, seq_len, hidden_dim)
        up_proj = torch.randn(intermediate_dim, hidden_dim)
        down_proj = torch.randn(hidden_dim, intermediate_dim)

        result = fused_mlp(
            x, None, up_proj, down_proj,
            activation="gelu"
        )

        # Reference implementation (no gating)
        hidden = F.gelu(x @ up_proj.t())
        expected = hidden @ down_proj.t()

        assert result.shape == (batch_size, seq_len, hidden_dim)
        assert torch.allclose(result, expected, atol=1e-4)


class TestFusedMLPDtypes:
    """Data type tests for fused MLP."""

    def test_fused_mlp_float32(self) -> None:
        """Fused MLP with float32."""
        from layerzero.mlp.fused import fused_mlp

        x = torch.randn(2, 16, 64, dtype=torch.float32)
        gate = torch.randn(256, 64, dtype=torch.float32)
        up = torch.randn(256, 64, dtype=torch.float32)
        down = torch.randn(64, 256, dtype=torch.float32)

        result = fused_mlp(x, gate, up, down, activation="swiglu")
        assert result.dtype == torch.float32

    def test_fused_mlp_float16(self) -> None:
        """Fused MLP with float16."""
        from layerzero.mlp.fused import fused_mlp

        x = torch.randn(2, 16, 64, dtype=torch.float16)
        gate = torch.randn(256, 64, dtype=torch.float16)
        up = torch.randn(256, 64, dtype=torch.float16)
        down = torch.randn(64, 256, dtype=torch.float16)

        result = fused_mlp(x, gate, up, down, activation="swiglu")
        assert result.dtype == torch.float16

    def test_fused_mlp_bfloat16(self) -> None:
        """Fused MLP with bfloat16."""
        from layerzero.mlp.fused import fused_mlp

        x = torch.randn(2, 16, 64, dtype=torch.bfloat16)
        gate = torch.randn(256, 64, dtype=torch.bfloat16)
        up = torch.randn(256, 64, dtype=torch.bfloat16)
        down = torch.randn(64, 256, dtype=torch.bfloat16)

        result = fused_mlp(x, gate, up, down, activation="swiglu")
        assert result.dtype == torch.bfloat16


class TestFusedMLPDevices:
    """Device tests for fused MLP."""

    def test_fused_mlp_cpu(self) -> None:
        """Fused MLP works on CPU."""
        from layerzero.mlp.fused import fused_mlp

        x = torch.randn(2, 16, 64, device='cpu')
        gate = torch.randn(256, 64, device='cpu')
        up = torch.randn(256, 64, device='cpu')
        down = torch.randn(64, 256, device='cpu')

        result = fused_mlp(x, gate, up, down, activation="swiglu")
        assert result.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fused_mlp_cuda(self) -> None:
        """Fused MLP works on CUDA."""
        from layerzero.mlp.fused import fused_mlp

        x = torch.randn(2, 16, 64, device='cuda')
        gate = torch.randn(256, 64, device='cuda')
        up = torch.randn(256, 64, device='cuda')
        down = torch.randn(64, 256, device='cuda')

        result = fused_mlp(x, gate, up, down, activation="swiglu")
        assert result.device.type == 'cuda'


class TestLinearCorrectness:
    """Tests for linear/GEMM operation."""

    def test_linear_basic(self) -> None:
        """Basic linear operation."""
        from layerzero.mlp.linear import linear

        batch_size = 2
        seq_len = 16
        in_features = 64
        out_features = 128

        x = torch.randn(batch_size, seq_len, in_features)
        weight = torch.randn(out_features, in_features)
        bias = torch.randn(out_features)

        result = linear(x, weight, bias)

        # Reference
        expected = F.linear(x, weight, bias)

        assert result.shape == (batch_size, seq_len, out_features)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_linear_no_bias(self) -> None:
        """Linear without bias."""
        from layerzero.mlp.linear import linear

        x = torch.randn(2, 16, 64)
        weight = torch.randn(128, 64)

        result = linear(x, weight, bias=None)

        # Reference
        expected = F.linear(x, weight, None)

        assert result.shape == (2, 16, 128)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_linear_batched(self) -> None:
        """Batched linear operation."""
        from layerzero.mlp.linear import linear

        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 32, 64)
            weight = torch.randn(128, 64)

            result = linear(x, weight)
            assert result.shape == (batch_size, 32, 128)


class TestFusedMLPGradient:
    """Gradient tests for fused MLP."""

    def test_fused_mlp_gradient_flow(self) -> None:
        """Gradients flow through fused MLP."""
        from layerzero.mlp.fused import fused_mlp

        x = torch.randn(2, 16, 64, requires_grad=True)
        gate = torch.randn(256, 64, requires_grad=True)
        up = torch.randn(256, 64, requires_grad=True)
        down = torch.randn(64, 256, requires_grad=True)

        result = fused_mlp(x, gate, up, down, activation="swiglu")
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert gate.grad is not None
        assert up.grad is not None
        assert down.grad is not None

    def test_fused_mlp_gradient_correctness(self) -> None:
        """Gradients match unfused implementation."""
        from layerzero.mlp.fused import fused_mlp

        torch.manual_seed(42)

        # Fused version
        x1 = torch.randn(2, 16, 64, requires_grad=True)
        gate1 = torch.randn(256, 64, requires_grad=True)
        up1 = torch.randn(256, 64, requires_grad=True)
        down1 = torch.randn(64, 256, requires_grad=True)

        result1 = fused_mlp(x1, gate1, up1, down1, activation="swiglu")
        loss1 = result1.sum()
        loss1.backward()

        # Reference version
        torch.manual_seed(42)
        x2 = torch.randn(2, 16, 64, requires_grad=True)
        gate2 = torch.randn(256, 64, requires_grad=True)
        up2 = torch.randn(256, 64, requires_grad=True)
        down2 = torch.randn(64, 256, requires_grad=True)

        g = x2 @ gate2.t()
        u = x2 @ up2.t()
        h = F.silu(g) * u
        result2 = h @ down2.t()
        loss2 = result2.sum()
        loss2.backward()

        # Compare gradients
        assert torch.allclose(x1.grad, x2.grad, atol=1e-4)
        assert torch.allclose(gate1.grad, gate2.grad, atol=1e-4)
        assert torch.allclose(up1.grad, up2.grad, atol=1e-4)
        assert torch.allclose(down1.grad, down2.grad, atol=1e-4)


class TestFusedMLPIntegration:
    """Integration tests for fused MLP with LayerZero."""

    def test_mlp_fused_operation_spec_exists(self) -> None:
        """OperationSpec for mlp.fused exists."""
        from layerzero.models.operation_spec import mlp_fused_spec

        spec = mlp_fused_spec()
        assert spec.op_id == "mlp.fused"
        assert spec.has_fallback is True

    def test_mlp_linear_operation_spec_exists(self) -> None:
        """OperationSpec for mlp.linear exists."""
        from layerzero.models.operation_spec import mlp_linear_spec

        spec = mlp_linear_spec()
        assert spec.op_id == "mlp.linear"
        assert spec.has_fallback is True


class TestFusedMLPPerformance:
    """Performance tests for fused MLP."""

    @pytest.mark.stress
    def test_fused_mlp_large_batch(self) -> None:
        """Fused MLP with large batch."""
        from layerzero.mlp.fused import fused_mlp
        import time

        batch_size = 32
        seq_len = 2048
        hidden_dim = 4096
        intermediate_dim = 11008  # Llama-7B

        x = torch.randn(batch_size, seq_len, hidden_dim)
        gate = torch.randn(intermediate_dim, hidden_dim)
        up = torch.randn(intermediate_dim, hidden_dim)
        down = torch.randn(hidden_dim, intermediate_dim)

        # Warmup
        for _ in range(2):
            _ = fused_mlp(x[:1], gate, up, down, activation="swiglu")

        start = time.perf_counter()
        result = fused_mlp(x, gate, up, down, activation="swiglu")
        elapsed = time.perf_counter() - start

        assert result.shape == (batch_size, seq_len, hidden_dim)
        # Should complete in reasonable time on CPU (relaxed for large sizes)
        assert elapsed < 60.0, f"Fused MLP too slow: {elapsed:.3f}s"

    @pytest.mark.stress
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fused_mlp_cuda_performance(self) -> None:
        """Fused MLP CUDA performance."""
        from layerzero.mlp.fused import fused_mlp
        import time

        batch_size = 8
        seq_len = 1024
        hidden_dim = 4096
        intermediate_dim = 11008

        x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
        gate = torch.randn(intermediate_dim, hidden_dim, device='cuda')
        up = torch.randn(intermediate_dim, hidden_dim, device='cuda')
        down = torch.randn(hidden_dim, intermediate_dim, device='cuda')

        # Warmup
        for _ in range(5):
            _ = fused_mlp(x, gate, up, down, activation="swiglu")
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(10):
            _ = fused_mlp(x, gate, up, down, activation="swiglu")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # 10 iterations should complete in reasonable time on GPU
        assert elapsed < 10.0, f"CUDA fused MLP too slow: {elapsed:.3f}s for 10 iterations"
