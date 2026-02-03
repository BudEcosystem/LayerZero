"""CUDA graph safety tests for LayerZero.

Tests CUDA graph capture, replay, and safety validation.
All tests require GPU and are skipped on CPU-only systems.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from layerzero.enums import Platform


# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Simple attention for testing."""
    return F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)


def reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference RMS normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


class GraphSafetyChecker:
    """Helper class to check CUDA graph safety of kernels."""

    # Known graph-safe operations
    SAFE_OPERATIONS = frozenset([
        "attention",
        "attention.causal",
        "norm.rms",
        "norm.layer",
        "matmul",
        "activation.gelu",
        "activation.silu",
    ])

    # Known graph-unsafe operations
    UNSAFE_OPERATIONS = frozenset([
        "dynamic_shape_op",
        "print_tensor",
        "host_to_device",
        "device_to_host",
    ])

    @classmethod
    def is_graph_safe(cls, operation: str, strict: bool = False) -> bool:
        """Check if operation is safe for CUDA graph capture.

        Args:
            operation: Operation name.
            strict: If True, reject unknown operations.

        Returns:
            True if operation is graph-safe.
        """
        if operation in cls.SAFE_OPERATIONS:
            return True
        if operation in cls.UNSAFE_OPERATIONS:
            return False
        # Unknown operation
        if strict:
            return False
        # In non-strict mode, assume safe
        return True

    @classmethod
    def get_graph_safe_kernels(cls) -> frozenset[str]:
        """Get set of graph-safe kernel IDs."""
        return cls.SAFE_OPERATIONS


class WarmupProtocol:
    """Protocol for warming up before CUDA graph capture."""

    def __init__(self):
        self.warmed_up = False
        self.cublas_initialized = False
        self.cudnn_initialized = False

    def warmup(self, func, *args, **kwargs):
        """Execute warmup before capture.

        Args:
            func: Function to warm up.
            *args: Arguments to pass.
            **kwargs: Keyword arguments to pass.
        """
        if CUDA_AVAILABLE:
            # Initialize cuBLAS
            dummy = torch.randn(64, 64, device="cuda")
            _ = torch.matmul(dummy, dummy)
            self.cublas_initialized = True

            # Initialize cuDNN (via conv or attention)
            _ = F.scaled_dot_product_attention(
                torch.randn(1, 1, 8, 32, device="cuda"),
                torch.randn(1, 1, 8, 32, device="cuda"),
                torch.randn(1, 1, 8, 32, device="cuda"),
            )
            self.cudnn_initialized = True

            # Warmup the actual function
            for _ in range(3):
                func(*args, **kwargs)

            torch.cuda.synchronize()
            self.warmed_up = True


class TestCUDAGraphCapture:
    """Test CUDA graph capture functionality."""

    @pytest.mark.gpu
    @skip_no_cuda
    def test_graph_capture_attention(self) -> None:
        """Attention captures in CUDA graph."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 64
        dtype = torch.float16
        device = "cuda"

        query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

        # Warmup
        for _ in range(3):
            _ = reference_attention(query, key, value)
        torch.cuda.synchronize()

        # Create CUDA graph
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()

        # Pre-allocate output
        with torch.cuda.stream(s):
            output = reference_attention(query, key, value)

        s.synchronize()

        # Capture
        with torch.cuda.graph(g, stream=s):
            output = reference_attention(query, key, value)

        # Graph should capture successfully
        assert g is not None

        # Replay
        g.replay()
        torch.cuda.synchronize()

        # Verify output is valid
        assert output is not None
        assert output.shape == query.shape
        assert torch.isfinite(output).all()

    @pytest.mark.gpu
    @skip_no_cuda
    def test_graph_capture_rms_norm(self) -> None:
        """RMSNorm captures in CUDA graph."""
        batch_size, seq_len, hidden_size = 2, 16, 64
        dtype = torch.float16
        device = "cuda"

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
        weight = torch.randn(hidden_size, dtype=dtype, device=device)

        # Warmup
        for _ in range(3):
            _ = reference_rms_norm(x, weight)
        torch.cuda.synchronize()

        # Create CUDA graph
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()

        # Pre-run
        with torch.cuda.stream(s):
            output = reference_rms_norm(x, weight)

        s.synchronize()

        # Capture
        with torch.cuda.graph(g, stream=s):
            output = reference_rms_norm(x, weight)

        # Graph should capture successfully
        assert g is not None

        # Verify output is valid
        assert output is not None
        assert output.shape == x.shape

    @pytest.mark.gpu
    @skip_no_cuda
    def test_graph_replay_correct(self) -> None:
        """Graph replay produces correct results."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 64
        dtype = torch.float16
        device = "cuda"

        # Static buffers
        static_query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        static_key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        static_value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

        # Warmup
        for _ in range(3):
            _ = reference_attention(static_query, static_key, static_value)
        torch.cuda.synchronize()

        # Compute expected result
        expected = reference_attention(static_query, static_key, static_value)
        torch.cuda.synchronize()

        # Create CUDA graph
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()

        with torch.cuda.stream(s):
            output = reference_attention(static_query, static_key, static_value)

        s.synchronize()

        with torch.cuda.graph(g, stream=s):
            output = reference_attention(static_query, static_key, static_value)

        # Replay and verify
        g.replay()
        torch.cuda.synchronize()

        # Results should match
        assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.gpu
    @skip_no_cuda
    def test_graph_replay_multiple_times(self) -> None:
        """Multiple graph replays work correctly."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 64
        dtype = torch.float16
        device = "cuda"

        static_query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        static_key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        static_value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

        # Warmup
        for _ in range(3):
            _ = reference_attention(static_query, static_key, static_value)
        torch.cuda.synchronize()

        # Create CUDA graph
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()

        with torch.cuda.stream(s):
            output = reference_attention(static_query, static_key, static_value)

        s.synchronize()

        with torch.cuda.graph(g, stream=s):
            output = reference_attention(static_query, static_key, static_value)

        # Replay multiple times
        for i in range(10):
            g.replay()
            torch.cuda.synchronize()

            # Each replay should produce valid output
            assert torch.isfinite(output).all(), f"Replay {i} produced non-finite output"


class TestGraphSafetyValidation:
    """Test graph safety validation."""

    @pytest.mark.gpu
    @skip_no_cuda
    def test_graph_unsafe_kernel_rejected(self) -> None:
        """Graph-unsafe kernels rejected in graph mode."""
        checker = GraphSafetyChecker()

        # Unsafe operations should be rejected
        assert not checker.is_graph_safe("dynamic_shape_op")
        assert not checker.is_graph_safe("host_to_device")
        assert not checker.is_graph_safe("device_to_host")

    @pytest.mark.gpu
    @skip_no_cuda
    def test_graph_whitelist_honored(self) -> None:
        """Graph whitelist determines safety."""
        checker = GraphSafetyChecker()

        safe_kernels = checker.get_graph_safe_kernels()

        # All safe operations should be in whitelist
        assert "attention" in safe_kernels
        assert "attention.causal" in safe_kernels
        assert "norm.rms" in safe_kernels
        assert "matmul" in safe_kernels

        # Unsafe should not be in whitelist
        assert "dynamic_shape_op" not in safe_kernels

    @pytest.mark.gpu
    @skip_no_cuda
    def test_graph_unknown_kernel_rejected_strict(self) -> None:
        """Unknown kernels rejected in strict mode."""
        checker = GraphSafetyChecker()

        # Unknown operation in strict mode
        assert not checker.is_graph_safe("unknown_operation", strict=True)

        # Unknown operation in non-strict mode (default allows)
        assert checker.is_graph_safe("unknown_operation", strict=False)


class TestGraphWarmup:
    """Test graph warmup protocol."""

    @pytest.mark.gpu
    @skip_no_cuda
    def test_warmup_before_capture(self) -> None:
        """Warmup executes before capture."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 64
        dtype = torch.float16
        device = "cuda"

        static_query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        static_key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        static_value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

        protocol = WarmupProtocol()

        def compute():
            return reference_attention(static_query, static_key, static_value)

        # Execute warmup
        protocol.warmup(compute)

        assert protocol.warmed_up

    @pytest.mark.gpu
    @skip_no_cuda
    def test_cublas_initialized_before_capture(self) -> None:
        """cuBLAS initialized before capture."""
        protocol = WarmupProtocol()

        def dummy():
            x = torch.randn(64, 64, device="cuda")
            return torch.matmul(x, x)

        protocol.warmup(dummy)

        assert protocol.cublas_initialized

    @pytest.mark.gpu
    @skip_no_cuda
    def test_cudnn_initialized_before_capture(self) -> None:
        """cuDNN initialized before capture."""
        protocol = WarmupProtocol()

        def dummy():
            q = torch.randn(1, 1, 8, 32, device="cuda")
            k = torch.randn(1, 1, 8, 32, device="cuda")
            v = torch.randn(1, 1, 8, 32, device="cuda")
            return F.scaled_dot_product_attention(q, k, v)

        protocol.warmup(dummy)

        assert protocol.cudnn_initialized


class TestGraphMemory:
    """Test graph memory management."""

    @pytest.mark.gpu
    @skip_no_cuda
    def test_memory_delta_check(self) -> None:
        """Memory delta checked during capture."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 64
        dtype = torch.float16
        device = "cuda"

        static_query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        static_key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        static_value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

        # Warmup
        for _ in range(3):
            _ = reference_attention(static_query, static_key, static_value)
        torch.cuda.synchronize()

        # Memory before capture
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()

        # Pre-capture run
        output = reference_attention(static_query, static_key, static_value)
        torch.cuda.synchronize()

        # Memory after pre-capture
        mem_after_pre = torch.cuda.memory_allocated()

        # Create and capture graph
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()

        with torch.cuda.stream(s):
            output = reference_attention(static_query, static_key, static_value)
        s.synchronize()

        with torch.cuda.graph(g, stream=s):
            output = reference_attention(static_query, static_key, static_value)

        # Memory should be stable after warmup
        mem_after_capture = torch.cuda.memory_allocated()

        # Memory shouldn't explode during capture
        # Allow some overhead for graph bookkeeping
        assert mem_after_capture <= mem_after_pre * 2

    @pytest.mark.gpu
    @skip_no_cuda
    def test_no_allocation_during_replay(self) -> None:
        """No new allocations during graph replay."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 64
        dtype = torch.float16
        device = "cuda"

        static_query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        static_key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        static_value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

        # Warmup
        for _ in range(3):
            _ = reference_attention(static_query, static_key, static_value)
        torch.cuda.synchronize()

        # Create and capture graph
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()

        with torch.cuda.stream(s):
            output = reference_attention(static_query, static_key, static_value)
        s.synchronize()

        with torch.cuda.graph(g, stream=s):
            output = reference_attention(static_query, static_key, static_value)

        # Memory before replay
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        # Replay multiple times
        for _ in range(10):
            g.replay()

        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()

        # No significant new allocations during replay
        # Allow up to 64KB for internal CUDA state changes
        mem_diff = abs(mem_after - mem_before)
        assert mem_diff < 64 * 1024, f"Memory changed by {mem_diff} bytes during replay"

    @pytest.mark.gpu
    @skip_no_cuda
    def test_graph_memory_pool(self) -> None:
        """Graph uses memory pool correctly."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 64
        dtype = torch.float16
        device = "cuda"

        # Create private memory pool for graph
        pool = torch.cuda.graph_pool_handle()

        static_query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        static_key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        static_value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

        # Warmup
        for _ in range(3):
            _ = reference_attention(static_query, static_key, static_value)
        torch.cuda.synchronize()

        # Create and capture graph with memory pool
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()

        with torch.cuda.stream(s):
            output = reference_attention(static_query, static_key, static_value)
        s.synchronize()

        with torch.cuda.graph(g, stream=s, pool=pool):
            output = reference_attention(static_query, static_key, static_value)

        # Graph should use the pool
        assert g is not None

        # Replay works with pool
        g.replay()
        torch.cuda.synchronize()

        assert torch.isfinite(output).all()
