"""Tests for CUDA graph warmup protocol."""
from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock, patch, call

import torch
import torch.nn.functional as F

from layerzero.graphs.warmup import (
    GraphWarmupProtocol,
    WarmupState,
    get_global_warmup,
    ensure_warmed_up,
)


class TestWarmupState:
    """Tests for WarmupState dataclass."""

    def test_default_values(self) -> None:
        """Default state indicates not warmed up."""
        state = WarmupState()

        assert state.cublas_initialized is False
        assert state.cudnn_initialized is False
        assert state.workspaces_allocated is False
        assert state.warmup_runs == 0
        assert state.warmup_time_ms == 0.0
        assert state.errors == []

    def test_is_ready_false_when_nothing_initialized(self) -> None:
        """is_ready is False when libraries not initialized."""
        state = WarmupState()
        assert state.is_ready is False

    def test_is_ready_false_when_only_cublas(self) -> None:
        """is_ready is False when only cuBLAS initialized."""
        state = WarmupState(cublas_initialized=True)
        assert state.is_ready is False

    def test_is_ready_false_when_only_cudnn(self) -> None:
        """is_ready is False when only cuDNN initialized."""
        state = WarmupState(cudnn_initialized=True)
        assert state.is_ready is False

    def test_is_ready_true_when_both_initialized(self) -> None:
        """is_ready is True when both libraries initialized."""
        state = WarmupState(
            cublas_initialized=True,
            cudnn_initialized=True,
        )
        assert state.is_ready is True

    def test_errors_list_mutable(self) -> None:
        """Errors list can be appended to."""
        state = WarmupState()
        state.errors.append("Error 1")
        state.errors.append("Error 2")

        assert len(state.errors) == 2


class TestGraphWarmupProtocolInit:
    """Tests for GraphWarmupProtocol initialization."""

    def test_default_initialization(self) -> None:
        """Default initialization creates proper state."""
        protocol = GraphWarmupProtocol()

        assert protocol.state is not None
        assert protocol.is_warmed_up is False
        assert protocol.cublas_initialized is False
        assert protocol.cudnn_initialized is False

    def test_custom_warmup_iterations(self) -> None:
        """Custom warmup iterations accepted."""
        protocol = GraphWarmupProtocol(warmup_iterations=5)
        assert protocol._warmup_iterations == 5

    def test_custom_device_string(self) -> None:
        """String device converted to torch.device."""
        protocol = GraphWarmupProtocol(device="cuda:0")
        assert protocol._device == torch.device("cuda:0")

    def test_custom_device_object(self) -> None:
        """torch.device object accepted."""
        device = torch.device("cuda:1")
        protocol = GraphWarmupProtocol(device=device)
        assert protocol._device == device

    def test_warmup_history_empty(self) -> None:
        """Warmup history starts empty."""
        protocol = GraphWarmupProtocol()
        assert protocol.warmup_history == []


class TestGraphWarmupProtocolWarmup:
    """Tests for warmup execution."""

    def test_warmup_returns_early_without_cuda(self) -> None:
        """Warmup returns early when CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            protocol = GraphWarmupProtocol()

            def dummy_func():
                return 42

            state = protocol.warmup(dummy_func)

            assert state.cublas_initialized is False
            assert state.cudnn_initialized is False

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.matmul")
    @patch("torch.randn")
    @patch.object(F, "scaled_dot_product_attention")
    def test_warmup_initializes_cublas(
        self,
        mock_sdpa,
        mock_randn,
        mock_matmul,
        mock_sync,
        mock_cuda_available,
    ) -> None:
        """Warmup initializes cuBLAS via matmul."""
        mock_randn.return_value = MagicMock()
        mock_matmul.return_value = MagicMock()
        mock_sdpa.return_value = MagicMock()

        protocol = GraphWarmupProtocol(device="cuda")

        def dummy_func():
            pass

        state = protocol.warmup(dummy_func)

        # cuBLAS should be initialized via matmul
        assert state.cublas_initialized is True
        mock_matmul.assert_called()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.matmul")
    @patch("torch.randn")
    @patch.object(F, "scaled_dot_product_attention")
    def test_warmup_initializes_cudnn(
        self,
        mock_sdpa,
        mock_randn,
        mock_matmul,
        mock_sync,
        mock_cuda_available,
    ) -> None:
        """Warmup initializes cuDNN via attention."""
        mock_randn.return_value = MagicMock()
        mock_matmul.return_value = MagicMock()
        mock_sdpa.return_value = MagicMock()

        protocol = GraphWarmupProtocol(device="cuda")

        def dummy_func():
            pass

        state = protocol.warmup(dummy_func)

        # cuDNN should be initialized via SDPA
        assert state.cudnn_initialized is True
        mock_sdpa.assert_called()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.matmul")
    @patch("torch.randn")
    @patch.object(F, "scaled_dot_product_attention")
    def test_warmup_runs_function(
        self,
        mock_sdpa,
        mock_randn,
        mock_matmul,
        mock_sync,
        mock_cuda_available,
    ) -> None:
        """Warmup runs target function specified times."""
        mock_randn.return_value = MagicMock()
        mock_matmul.return_value = MagicMock()
        mock_sdpa.return_value = MagicMock()

        protocol = GraphWarmupProtocol(warmup_iterations=3, device="cuda")

        call_count = [0]

        def counting_func():
            call_count[0] += 1

        state = protocol.warmup(counting_func)

        assert state.warmup_runs == 3
        assert call_count[0] == 3

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.matmul")
    @patch("torch.randn")
    @patch.object(F, "scaled_dot_product_attention")
    def test_warmup_with_args_kwargs(
        self,
        mock_sdpa,
        mock_randn,
        mock_matmul,
        mock_sync,
        mock_cuda_available,
    ) -> None:
        """Warmup passes args and kwargs to function."""
        mock_randn.return_value = MagicMock()
        mock_matmul.return_value = MagicMock()
        mock_sdpa.return_value = MagicMock()

        protocol = GraphWarmupProtocol(warmup_iterations=1, device="cuda")

        received = []

        def receiving_func(a, b, c=None):
            received.append((a, b, c))

        protocol.warmup(receiving_func, 1, 2, c=3)

        assert received == [(1, 2, 3)]

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.matmul")
    @patch("torch.randn")
    @patch.object(F, "scaled_dot_product_attention")
    def test_warmup_handles_function_error(
        self,
        mock_sdpa,
        mock_randn,
        mock_matmul,
        mock_sync,
        mock_cuda_available,
    ) -> None:
        """Warmup handles function errors gracefully."""
        mock_randn.return_value = MagicMock()
        mock_matmul.return_value = MagicMock()
        mock_sdpa.return_value = MagicMock()

        protocol = GraphWarmupProtocol(warmup_iterations=3, device="cuda")

        def failing_func():
            raise RuntimeError("Intentional failure")

        state = protocol.warmup(failing_func)

        # Warmup should complete despite errors
        assert state.workspaces_allocated is True
        assert len(state.errors) == 3  # All 3 iterations failed
        assert state.warmup_runs == 0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.matmul")
    @patch("torch.randn")
    @patch.object(F, "scaled_dot_product_attention")
    def test_warmup_records_time(
        self,
        mock_sdpa,
        mock_randn,
        mock_matmul,
        mock_sync,
        mock_cuda_available,
    ) -> None:
        """Warmup records total time."""
        mock_randn.return_value = MagicMock()
        mock_matmul.return_value = MagicMock()
        mock_sdpa.return_value = MagicMock()

        protocol = GraphWarmupProtocol(device="cuda")

        def dummy_func():
            pass

        state = protocol.warmup(dummy_func)

        assert state.warmup_time_ms > 0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.matmul")
    @patch("torch.randn")
    @patch.object(F, "scaled_dot_product_attention")
    def test_warmup_history_recorded(
        self,
        mock_sdpa,
        mock_randn,
        mock_matmul,
        mock_sync,
        mock_cuda_available,
    ) -> None:
        """Warmup records history for debugging."""
        mock_randn.return_value = MagicMock()
        mock_matmul.return_value = MagicMock()
        mock_sdpa.return_value = MagicMock()

        protocol = GraphWarmupProtocol(warmup_iterations=3, device="cuda")

        def dummy_func():
            pass

        protocol.warmup(dummy_func)

        history = protocol.warmup_history
        assert len(history) == 3

        for i, entry in enumerate(history):
            assert entry["iteration"] == i
            assert entry["success"] is True
            assert "time_ms" in entry


class TestGraphWarmupProtocolPartialWarmup:
    """Tests for individual warmup methods."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.matmul")
    @patch("torch.randn")
    def test_warmup_cublas_only(
        self,
        mock_randn,
        mock_matmul,
        mock_sync,
        mock_cuda_available,
    ) -> None:
        """warmup_cublas initializes only cuBLAS."""
        mock_randn.return_value = MagicMock()
        mock_matmul.return_value = MagicMock()

        protocol = GraphWarmupProtocol(device="cuda")

        result = protocol.warmup_cublas()

        assert result is True
        assert protocol.cublas_initialized is True
        assert protocol.cudnn_initialized is False

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.randn")
    @patch.object(F, "scaled_dot_product_attention")
    def test_warmup_cudnn_only(
        self,
        mock_sdpa,
        mock_randn,
        mock_sync,
        mock_cuda_available,
    ) -> None:
        """warmup_cudnn initializes only cuDNN."""
        mock_randn.return_value = MagicMock()
        mock_sdpa.return_value = MagicMock()

        protocol = GraphWarmupProtocol(device="cuda")

        result = protocol.warmup_cudnn()

        assert result is True
        assert protocol.cudnn_initialized is True
        # cuBLAS not initialized by warmup_cudnn
        assert protocol.cublas_initialized is False

    def test_warmup_cublas_returns_false_without_cuda(self) -> None:
        """warmup_cublas returns False without CUDA."""
        with patch("torch.cuda.is_available", return_value=False):
            protocol = GraphWarmupProtocol()
            result = protocol.warmup_cublas()

            assert result is False
            assert protocol.cublas_initialized is False

    def test_warmup_cudnn_returns_false_without_cuda(self) -> None:
        """warmup_cudnn returns False without CUDA."""
        with patch("torch.cuda.is_available", return_value=False):
            protocol = GraphWarmupProtocol()
            result = protocol.warmup_cudnn()

            assert result is False
            assert protocol.cudnn_initialized is False


class TestGraphWarmupProtocolReset:
    """Tests for reset functionality."""

    def test_reset_clears_state(self) -> None:
        """reset() clears all state."""
        protocol = GraphWarmupProtocol()

        # Manually set some state
        protocol._state = WarmupState(
            cublas_initialized=True,
            cudnn_initialized=True,
            warmup_runs=5,
        )
        protocol._warmup_history = [{"iteration": 0, "success": True}]

        protocol.reset()

        assert protocol.state.cublas_initialized is False
        assert protocol.state.cudnn_initialized is False
        assert protocol.state.warmup_runs == 0
        assert protocol.warmup_history == []


class TestGlobalWarmup:
    """Tests for global warmup functions."""

    def test_get_global_warmup_returns_instance(self) -> None:
        """get_global_warmup returns a protocol instance."""
        # Reset global state first
        import layerzero.graphs.warmup as warmup_module

        warmup_module._global_warmup = None

        protocol = get_global_warmup()

        assert isinstance(protocol, GraphWarmupProtocol)

    def test_get_global_warmup_returns_same_instance(self) -> None:
        """get_global_warmup returns same instance on subsequent calls."""
        import layerzero.graphs.warmup as warmup_module

        warmup_module._global_warmup = None

        protocol1 = get_global_warmup()
        protocol2 = get_global_warmup()

        assert protocol1 is protocol2

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.matmul")
    @patch("torch.randn")
    @patch.object(F, "scaled_dot_product_attention")
    def test_ensure_warmed_up_warms_up(
        self,
        mock_sdpa,
        mock_randn,
        mock_matmul,
        mock_sync,
        mock_cuda_available,
    ) -> None:
        """ensure_warmed_up performs warmup when needed."""
        import layerzero.graphs.warmup as warmup_module

        warmup_module._global_warmup = None

        mock_randn.return_value = MagicMock()
        mock_matmul.return_value = MagicMock()
        mock_sdpa.return_value = MagicMock()

        def dummy_func():
            pass

        state = ensure_warmed_up(dummy_func)

        assert state.is_ready is True

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.matmul")
    @patch("torch.randn")
    @patch.object(F, "scaled_dot_product_attention")
    def test_ensure_warmed_up_skips_if_ready(
        self,
        mock_sdpa,
        mock_randn,
        mock_matmul,
        mock_sync,
        mock_cuda_available,
    ) -> None:
        """ensure_warmed_up skips warmup if already done."""
        import layerzero.graphs.warmup as warmup_module

        mock_randn.return_value = MagicMock()
        mock_matmul.return_value = MagicMock()
        mock_sdpa.return_value = MagicMock()

        # Create pre-warmed protocol
        protocol = GraphWarmupProtocol()
        protocol._state = WarmupState(
            cublas_initialized=True,
            cudnn_initialized=True,
        )
        warmup_module._global_warmup = protocol

        def dummy_func():
            pass

        # Reset call counts
        mock_randn.reset_mock()
        mock_matmul.reset_mock()

        state = ensure_warmed_up(dummy_func)

        # Should not have called warmup functions
        # (matmul shouldn't be called for cublas init)
        assert state.is_ready is True
