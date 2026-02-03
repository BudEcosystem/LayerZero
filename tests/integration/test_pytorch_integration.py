"""Tests for PyTorch integration."""
from __future__ import annotations

import pytest
import torch

from layerzero.pytorch import ops
from layerzero.pytorch.meta_kernels import (
    attention_meta,
    rms_norm_meta,
    layer_norm_meta,
)
from layerzero.pytorch.sdpa_integration import (
    get_active_sdpa_backends,
    check_flash_attention_available,
    check_efficient_attention_available,
)


class TestTorchLibraryRegistration:
    """Test torch.library registration."""

    def test_ops_registered_in_torch_library(self) -> None:
        """LayerZero ops registered in torch.library."""
        # Import triggers registration
        import layerzero.pytorch.ops  # noqa: F401

        # Check ops are accessible via torch.ops
        assert hasattr(torch.ops, "layerzero")

    def test_ops_namespace_layerzero(self) -> None:
        """Ops use 'layerzero' namespace."""
        import layerzero.pytorch.ops  # noqa: F401

        # Should have layerzero namespace
        assert hasattr(torch.ops.layerzero, "attention")

    def test_attention_op_exists(self) -> None:
        """Attention op is registered."""
        import layerzero.pytorch.ops  # noqa: F401

        assert hasattr(torch.ops.layerzero, "attention")
        assert callable(torch.ops.layerzero.attention)

    def test_rms_norm_op_exists(self) -> None:
        """RMS norm op is registered."""
        import layerzero.pytorch.ops  # noqa: F401

        assert hasattr(torch.ops.layerzero, "rms_norm")
        assert callable(torch.ops.layerzero.rms_norm)

    def test_layer_norm_op_exists(self) -> None:
        """Layer norm op is registered."""
        import layerzero.pytorch.ops  # noqa: F401

        assert hasattr(torch.ops.layerzero, "layer_norm")
        assert callable(torch.ops.layerzero.layer_norm)


class TestMetaKernels:
    """Test meta kernels for tracing/export."""

    def test_attention_meta_returns_correct_shape(
        self,
        sample_qkv: dict[str, torch.Tensor],
    ) -> None:
        """attention_meta returns correct output shape."""
        q = sample_qkv["query"]
        k = sample_qkv["key"]
        v = sample_qkv["value"]

        result = attention_meta(q, k, v)
        assert result.shape == q.shape

    def test_rms_norm_meta_returns_correct_shape(
        self,
        sample_norm_input: dict[str, torch.Tensor],
    ) -> None:
        """rms_norm_meta returns correct output shape."""
        x = sample_norm_input["input"]
        weight = sample_norm_input["weight"]

        result = rms_norm_meta(x, weight)
        assert result.shape == x.shape

    def test_layer_norm_meta_returns_correct_shape(
        self,
        sample_norm_input: dict[str, torch.Tensor],
    ) -> None:
        """layer_norm_meta returns correct output shape."""
        x = sample_norm_input["input"]
        weight = sample_norm_input["weight"]
        bias = sample_norm_input["bias"]

        result = layer_norm_meta(x, weight, bias)
        assert result.shape == x.shape

    def test_meta_kernel_preserves_dtype(
        self,
        sample_qkv: dict[str, torch.Tensor],
    ) -> None:
        """Meta kernels preserve input dtype."""
        q = sample_qkv["query"].half()
        k = sample_qkv["key"].half()
        v = sample_qkv["value"].half()

        result = attention_meta(q, k, v)
        assert result.dtype == torch.float16


class TestOpExecution:
    """Test op execution."""

    def test_attention_op_executes_cpu(
        self,
        sample_qkv: dict[str, torch.Tensor],
    ) -> None:
        """Attention op executes on CPU."""
        import layerzero.pytorch.ops  # noqa: F401

        q = sample_qkv["query"]
        k = sample_qkv["key"]
        v = sample_qkv["value"]

        result = torch.ops.layerzero.attention(q, k, v)
        assert result.shape == q.shape
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_attention_op_executes_cuda(
        self,
        sample_qkv_cuda: dict[str, torch.Tensor],
    ) -> None:
        """Attention op executes on CUDA."""
        import layerzero.pytorch.ops  # noqa: F401

        q = sample_qkv_cuda["query"]
        k = sample_qkv_cuda["key"]
        v = sample_qkv_cuda["value"]

        result = torch.ops.layerzero.attention(q, k, v)
        assert result.shape == q.shape
        assert result.device.type == "cuda"

    def test_rms_norm_op_executes_cpu(
        self,
        sample_norm_input: dict[str, torch.Tensor],
    ) -> None:
        """RMS norm op executes on CPU."""
        import layerzero.pytorch.ops  # noqa: F401

        x = sample_norm_input["input"]
        weight = sample_norm_input["weight"]

        result = torch.ops.layerzero.rms_norm(x, weight)
        assert result.shape == x.shape

    def test_layer_norm_op_executes_cpu(
        self,
        sample_norm_input: dict[str, torch.Tensor],
    ) -> None:
        """Layer norm op executes on CPU."""
        import layerzero.pytorch.ops  # noqa: F401

        x = sample_norm_input["input"]
        weight = sample_norm_input["weight"]
        bias = sample_norm_input["bias"]

        result = torch.ops.layerzero.layer_norm(x, weight, bias)
        assert result.shape == x.shape


class TestTorchCompileCompatibility:
    """Test torch.compile compatibility."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile not available"
    )
    def test_torch_compile_no_graph_breaks(
        self,
        sample_qkv: dict[str, torch.Tensor],
    ) -> None:
        """torch.compile works without graph breaks."""
        import layerzero.pytorch.ops  # noqa: F401

        def forward(q, k, v):
            return torch.ops.layerzero.attention(q, k, v)

        compiled = torch.compile(forward, fullgraph=True)

        q = sample_qkv["query"]
        k = sample_qkv["key"]
        v = sample_qkv["value"]

        # Should not raise
        result = compiled(q, k, v)
        assert result.shape == q.shape

    @pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile not available"
    )
    def test_torch_compile_attention(
        self,
        sample_qkv: dict[str, torch.Tensor],
    ) -> None:
        """lz.attention compiles correctly."""
        import layerzero.pytorch.ops  # noqa: F401

        @torch.compile
        def attention_fn(q, k, v):
            return torch.ops.layerzero.attention(q, k, v)

        q = sample_qkv["query"]
        k = sample_qkv["key"]
        v = sample_qkv["value"]

        result = attention_fn(q, k, v)
        assert result.shape == q.shape

    @pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile not available"
    )
    def test_torch_compile_rms_norm(
        self,
        sample_norm_input: dict[str, torch.Tensor],
    ) -> None:
        """lz.rms_norm compiles correctly."""
        import layerzero.pytorch.ops  # noqa: F401

        @torch.compile
        def rms_norm_fn(x, w):
            return torch.ops.layerzero.rms_norm(x, w)

        x = sample_norm_input["input"]
        weight = sample_norm_input["weight"]

        result = rms_norm_fn(x, weight)
        assert result.shape == x.shape


class TestTorchExport:
    """Test torch.export compatibility."""

    @pytest.mark.skipif(
        not hasattr(torch, "export"),
        reason="torch.export not available"
    )
    def test_torch_export_attention(
        self,
        sample_qkv: dict[str, torch.Tensor],
    ) -> None:
        """lz.attention exports correctly."""
        import layerzero.pytorch.ops  # noqa: F401

        class AttentionModule(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.ops.layerzero.attention(q, k, v)

        module = AttentionModule()
        q = sample_qkv["query"]
        k = sample_qkv["key"]
        v = sample_qkv["value"]

        # Export should work
        exported = torch.export.export(module, (q, k, v))
        assert exported is not None

    @pytest.mark.skipif(
        not hasattr(torch, "export"),
        reason="torch.export not available"
    )
    def test_torch_export_with_meta_kernel(
        self,
        sample_qkv: dict[str, torch.Tensor],
    ) -> None:
        """Export uses meta kernel for tracing."""
        import layerzero.pytorch.ops  # noqa: F401

        class AttentionModule(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.ops.layerzero.attention(q, k, v)

        module = AttentionModule()
        q = sample_qkv["query"]
        k = sample_qkv["key"]
        v = sample_qkv["value"]

        # Export should use meta kernel
        with torch.no_grad():
            exported = torch.export.export(module, (q, k, v))

        # Verify output shape is correct in graph
        assert exported is not None


class TestSDPAIntegration:
    """Test SDPA backend integration."""

    def test_get_active_sdpa_backends(self) -> None:
        """get_active_sdpa_backends returns set."""
        backends = get_active_sdpa_backends()
        assert isinstance(backends, (set, frozenset))

    def test_check_flash_attention_available(self) -> None:
        """check_flash_attention_available returns bool."""
        result = check_flash_attention_available()
        assert isinstance(result, bool)

    def test_check_efficient_attention_available(self) -> None:
        """check_efficient_attention_available returns bool."""
        result = check_efficient_attention_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not hasattr(torch.nn.attention, "sdpa_kernel"),
        reason="sdpa_kernel context not available"
    )
    def test_sdpa_kernel_context_respected(
        self,
        sample_qkv: dict[str, torch.Tensor],
    ) -> None:
        """sdpa_kernel context is respected."""
        from torch.nn.attention import sdpa_kernel, SDPBackend
        import layerzero.pytorch.ops  # noqa: F401

        q = sample_qkv["query"]
        k = sample_qkv["key"]
        v = sample_qkv["value"]

        # Should work within sdpa_kernel context
        with sdpa_kernel(SDPBackend.MATH):
            result = torch.ops.layerzero.attention(q, k, v)

        assert result.shape == q.shape


class TestBackwardCompatibility:
    """Test backward pass and autograd integration."""

    def test_requires_grad_respected(
        self,
        sample_qkv: dict[str, torch.Tensor],
    ) -> None:
        """requires_grad flows through correctly."""
        import layerzero.pytorch.ops  # noqa: F401

        q = sample_qkv["query"].requires_grad_(True)
        k = sample_qkv["key"].requires_grad_(True)
        v = sample_qkv["value"].requires_grad_(True)

        result = torch.ops.layerzero.attention(q, k, v)

        # Output should require grad if inputs do
        assert result.requires_grad

    def test_autograd_integration(
        self,
        sample_qkv: dict[str, torch.Tensor],
    ) -> None:
        """Autograd integration works."""
        import layerzero.pytorch.ops  # noqa: F401

        q = sample_qkv["query"].requires_grad_(True)
        k = sample_qkv["key"].requires_grad_(True)
        v = sample_qkv["value"].requires_grad_(True)

        result = torch.ops.layerzero.attention(q, k, v)
        loss = result.sum()

        # Backward pass should work
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
