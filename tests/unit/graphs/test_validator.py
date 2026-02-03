"""Tests for CUDA graph validator."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

import torch

from layerzero.graphs.config import GraphSafetyConfig, GraphValidationResult
from layerzero.graphs.whitelist import GraphWhitelist
from layerzero.graphs.validator import (
    GraphValidator,
    get_global_validator,
    is_graph_safe,
)


class TestGraphValidatorInit:
    """Tests for GraphValidator initialization."""

    def test_default_initialization(self) -> None:
        """Default initialization creates validator with defaults."""
        validator = GraphValidator()

        assert validator.config is not None
        assert validator.whitelist is not None
        assert isinstance(validator.config, GraphSafetyConfig)
        assert isinstance(validator.whitelist, GraphWhitelist)

    def test_custom_config(self) -> None:
        """Custom config is used."""
        config = GraphSafetyConfig(strict_mode=True, warmup_iterations=5)
        validator = GraphValidator(config=config)

        assert validator.config is config
        assert validator.config.strict_mode is True
        assert validator.config.warmup_iterations == 5

    def test_custom_whitelist(self) -> None:
        """Custom whitelist is used."""
        whitelist = GraphWhitelist(
            safe_kernels=frozenset(["my_kernel"]),
            default_safe=True,
        )
        validator = GraphValidator(whitelist=whitelist)

        assert validator.whitelist is whitelist
        assert validator.whitelist.is_graph_safe("my_kernel")


class TestGraphValidatorIsGraphSafe:
    """Tests for is_graph_safe method."""

    def test_safe_kernel_returns_true(self) -> None:
        """Known safe kernel returns True."""
        validator = GraphValidator()

        assert validator.is_graph_safe("attention") is True
        assert validator.is_graph_safe("norm.rms") is True

    def test_unsafe_kernel_returns_false(self) -> None:
        """Known unsafe kernel returns False."""
        validator = GraphValidator()

        assert validator.is_graph_safe("dynamic_shape_op") is False
        assert validator.is_graph_safe("host_to_device") is False

    def test_strict_mode_from_config(self) -> None:
        """Strict mode uses config value when not specified."""
        config = GraphSafetyConfig(strict_mode=True)
        validator = GraphValidator(config=config)

        # Unknown kernel should be rejected in strict mode
        assert validator.is_graph_safe("unknown_kernel") is False

    def test_strict_mode_override(self) -> None:
        """Strict mode can be overridden."""
        config = GraphSafetyConfig(strict_mode=True)
        validator = GraphValidator(config=config)

        # Override strict mode
        assert validator.is_graph_safe("unknown_kernel", strict=False) is True


class TestGraphValidatorValidateKernel:
    """Tests for validate_kernel method."""

    def test_safe_kernel_spec_passes(self) -> None:
        """Safe KernelSpec passes validation."""
        validator = GraphValidator()

        kernel = MagicMock()
        kernel.kernel_id = "attention"
        kernel.operation = "attention"
        kernel.is_cuda_graph_safe = None

        result = validator.validate_kernel(kernel)

        assert result.success is True
        assert result.kernel_id == "attention"
        assert result.operation == "attention"
        assert result.error is None

    def test_unsafe_kernel_spec_fails(self) -> None:
        """Unsafe KernelSpec fails validation."""
        config = GraphSafetyConfig(strict_mode=True)
        validator = GraphValidator(config=config)

        kernel = MagicMock()
        kernel.kernel_id = "dynamic_shape_op"
        kernel.operation = "dynamic_shape"
        kernel.is_cuda_graph_safe = None

        result = validator.validate_kernel(kernel)

        assert result.success is False
        assert "not graph-safe" in result.error

    def test_explicit_unsafe_flag_fails(self) -> None:
        """Kernel with is_cuda_graph_safe=False fails."""
        validator = GraphValidator()

        kernel = MagicMock()
        kernel.kernel_id = "attention"  # Would be safe by whitelist
        kernel.operation = "attention"
        kernel.is_cuda_graph_safe = False  # Explicitly marked unsafe

        result = validator.validate_kernel(kernel)

        assert result.success is False
        assert "explicitly marked as graph-unsafe" in result.error


class TestGraphValidatorValidateCapture:
    """Tests for validate_capture method."""

    def test_no_cuda_skips_validation(self) -> None:
        """Validation skipped when CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            validator = GraphValidator()

            result = validator.validate_capture(
                lambda: None,
                kernel_id="test",
                operation="test_op",
            )

            assert result.success is True
            assert result.has_warnings is True
            assert "CUDA not available" in result.warnings[0]

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.memory_allocated", return_value=1024)
    @patch("torch.cuda.memory_reserved", return_value=2048)
    @patch("torch.cuda.max_memory_allocated", return_value=1024)
    def test_successful_validation(self, *mocks) -> None:
        """Successful validation returns success result."""
        config = GraphSafetyConfig(
            validate_before_production=False,  # Skip dummy capture
            warmup_iterations=1,
        )
        validator = GraphValidator(config=config)

        # Mock warmup protocol
        with patch.object(validator._warmup_protocol, 'warmup') as mock_warmup:
            mock_state = MagicMock()
            mock_state.is_ready = True
            mock_state.cublas_initialized = True
            mock_state.cudnn_initialized = True
            mock_state.errors = []
            mock_warmup.return_value = mock_state

            result = validator.validate_capture(
                lambda: torch.randn(4, 4),
                kernel_id="test_kernel",
                operation="test_op",
            )

        assert result.success is True
        assert result.kernel_id == "test_kernel"
        assert result.operation == "test_op"
        assert result.capture_time_ms > 0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    def test_warmup_incomplete_warning(self, *mocks) -> None:
        """Incomplete warmup adds warning."""
        config = GraphSafetyConfig(
            validate_before_production=False,
            warmup_iterations=1,
        )
        validator = GraphValidator(config=config)

        # Mock warmup protocol to return incomplete state
        with patch.object(validator._warmup_protocol, 'warmup') as mock_warmup, \
             patch.object(validator._memory_tracker, 'snapshot'), \
             patch.object(validator._memory_tracker, 'check_memory_delta', return_value=(True, "OK")), \
             patch.object(validator._memory_tracker, 'get_delta', return_value=None):
            mock_state = MagicMock()
            mock_state.is_ready = False
            mock_state.cublas_initialized = True
            mock_state.cudnn_initialized = False
            mock_state.errors = ["cuDNN init failed"]
            mock_warmup.return_value = mock_state

            result = validator.validate_capture(
                lambda: None,
            )

        assert result.success is True
        assert result.has_warnings is True
        assert any("Warmup incomplete" in w for w in result.warnings)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    def test_memory_delta_warning(self, *mocks) -> None:
        """Memory delta above threshold adds warning."""
        config = GraphSafetyConfig(
            validate_before_production=False,
            memory_delta_warning_mb=1.0,
            warmup_iterations=1,
        )
        validator = GraphValidator(config=config)

        with patch.object(validator._warmup_protocol, 'warmup') as mock_warmup, \
             patch.object(validator._memory_tracker, 'snapshot'), \
             patch.object(validator._memory_tracker, 'reset'), \
             patch.object(validator._memory_tracker, 'check_memory_delta', return_value=(False, "Memory delta 5.00MB exceeds threshold 1.00MB")), \
             patch.object(validator._memory_tracker, 'get_delta') as mock_get_delta:
            mock_state = MagicMock()
            mock_state.is_ready = True
            mock_state.cublas_initialized = True
            mock_state.cudnn_initialized = True
            mock_state.errors = []
            mock_warmup.return_value = mock_state

            mock_delta = MagicMock()
            mock_delta.allocated_delta_mb = 5.0
            mock_get_delta.return_value = mock_delta

            result = validator.validate_capture(
                lambda: None,
            )

        assert result.success is True
        assert result.has_warnings is True
        assert any("exceeds" in w for w in result.warnings)

    @patch("torch.cuda.is_available", return_value=True)
    def test_validation_exception_fails(self, mock_cuda) -> None:
        """Exception during validation marks failure."""
        validator = GraphValidator()

        # Force an exception during warmup
        with patch.object(validator._warmup_protocol, 'warmup', side_effect=RuntimeError("Test error")):
            result = validator.validate_capture(
                lambda: None,
            )

        assert result.success is False
        assert "Validation failed" in result.error


class TestGraphValidatorDummyCapture:
    """Tests for dummy capture functionality."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    def test_strict_mode_runs_dummy_capture(self, *mocks) -> None:
        """Strict mode runs dummy capture."""
        config = GraphSafetyConfig(
            strict_mode=True,
            warmup_iterations=1,
        )
        validator = GraphValidator(config=config)

        with patch.object(validator._warmup_protocol, 'warmup') as mock_warmup, \
             patch.object(validator._memory_tracker, 'snapshot'), \
             patch.object(validator._memory_tracker, 'reset'), \
             patch.object(validator._memory_tracker, 'check_memory_delta', return_value=(True, "OK")), \
             patch.object(validator._memory_tracker, 'get_delta', return_value=None), \
             patch.object(validator, '_dummy_capture', return_value={"success": True}) as mock_capture:
            mock_state = MagicMock()
            mock_state.is_ready = True
            mock_state.cublas_initialized = True
            mock_state.cudnn_initialized = True
            mock_state.errors = []
            mock_warmup.return_value = mock_state

            validator.validate_capture(lambda: None)

            mock_capture.assert_called_once()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    def test_dummy_capture_failure_fails_validation(self, *mocks) -> None:
        """Failed dummy capture fails validation."""
        config = GraphSafetyConfig(
            strict_mode=True,
            warmup_iterations=1,
        )
        validator = GraphValidator(config=config)

        with patch.object(validator._warmup_protocol, 'warmup') as mock_warmup, \
             patch.object(validator._memory_tracker, 'snapshot'), \
             patch.object(validator._memory_tracker, 'reset'), \
             patch.object(validator, '_dummy_capture', return_value={"success": False, "error": "Capture failed"}):
            mock_state = MagicMock()
            mock_state.is_ready = True
            mock_state.cublas_initialized = True
            mock_state.cudnn_initialized = True
            mock_state.errors = []
            mock_warmup.return_value = mock_state

            result = validator.validate_capture(lambda: None)

        assert result.success is False
        assert "Capture failed" in result.error


class TestGraphValidatorValidateBatch:
    """Tests for validate_batch method."""

    def test_batch_validation(self) -> None:
        """validate_batch validates multiple kernels."""
        validator = GraphValidator()

        kernel1 = MagicMock()
        kernel1.kernel_id = "attention"
        kernel1.operation = "attention"
        kernel1.is_cuda_graph_safe = None

        kernel2 = MagicMock()
        kernel2.kernel_id = "matmul"
        kernel2.operation = "matmul"
        kernel2.is_cuda_graph_safe = None

        results = validator.validate_batch([kernel1, kernel2])

        assert len(results) == 2
        assert "attention" in results
        assert "matmul" in results
        assert results["attention"].success is True
        assert results["matmul"].success is True

    def test_batch_with_mixed_results(self) -> None:
        """validate_batch handles mixed safe/unsafe kernels."""
        config = GraphSafetyConfig(strict_mode=True)
        validator = GraphValidator(config=config)

        safe_kernel = MagicMock()
        safe_kernel.kernel_id = "attention"
        safe_kernel.operation = "attention"
        safe_kernel.is_cuda_graph_safe = None

        unsafe_kernel = MagicMock()
        unsafe_kernel.kernel_id = "dynamic_shape_op"
        unsafe_kernel.operation = "dynamic"
        unsafe_kernel.is_cuda_graph_safe = None

        results = validator.validate_batch([safe_kernel, unsafe_kernel])

        assert results["attention"].success is True
        assert results["dynamic_shape_op"].success is False


class TestGlobalValidator:
    """Tests for global validator functions."""

    def test_get_global_validator_returns_instance(self) -> None:
        """get_global_validator returns validator instance."""
        import layerzero.graphs.validator as validator_module

        validator_module._global_validator = None

        validator = get_global_validator()

        assert isinstance(validator, GraphValidator)

    def test_get_global_validator_returns_same_instance(self) -> None:
        """get_global_validator returns same instance."""
        import layerzero.graphs.validator as validator_module

        validator_module._global_validator = None

        v1 = get_global_validator()
        v2 = get_global_validator()

        assert v1 is v2

    def test_is_graph_safe_uses_global_validator(self) -> None:
        """is_graph_safe function uses global validator."""
        import layerzero.graphs.validator as validator_module

        validator_module._global_validator = None

        assert is_graph_safe("attention") is True
        assert is_graph_safe("dynamic_shape_op") is False

    def test_is_graph_safe_with_strict_mode(self) -> None:
        """is_graph_safe function accepts strict parameter."""
        import layerzero.graphs.validator as validator_module

        validator_module._global_validator = None

        assert is_graph_safe("unknown_kernel", strict=True) is False
        assert is_graph_safe("unknown_kernel", strict=False) is True
