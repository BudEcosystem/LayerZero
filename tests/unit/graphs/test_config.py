"""Tests for CUDA graph safety configuration."""
from __future__ import annotations

import os
import pytest
from unittest.mock import patch

from layerzero.graphs.config import GraphSafetyConfig, GraphValidationResult


class TestGraphSafetyConfig:
    """Tests for GraphSafetyConfig dataclass."""

    def test_default_values(self) -> None:
        """Default configuration values are correct."""
        config = GraphSafetyConfig()

        assert config.strict_mode is False
        assert config.memory_delta_warning_mb == 1.0
        assert config.warmup_iterations == 3
        assert config.default_graph_safe is False
        assert config.validate_before_production is True
        assert config.capture_timeout_ms == 30000.0

    def test_custom_values(self) -> None:
        """Custom configuration values are accepted."""
        config = GraphSafetyConfig(
            strict_mode=True,
            memory_delta_warning_mb=2.5,
            warmup_iterations=5,
            default_graph_safe=True,
            validate_before_production=False,
            capture_timeout_ms=60000.0,
        )

        assert config.strict_mode is True
        assert config.memory_delta_warning_mb == 2.5
        assert config.warmup_iterations == 5
        assert config.default_graph_safe is True
        assert config.validate_before_production is False
        assert config.capture_timeout_ms == 60000.0

    def test_frozen_immutability(self) -> None:
        """Config is frozen and immutable."""
        config = GraphSafetyConfig()

        with pytest.raises(AttributeError):
            config.strict_mode = True

    def test_invalid_memory_delta_negative(self) -> None:
        """Negative memory delta raises ValueError."""
        with pytest.raises(ValueError, match="memory_delta_warning_mb must be non-negative"):
            GraphSafetyConfig(memory_delta_warning_mb=-1.0)

    def test_invalid_warmup_iterations_negative(self) -> None:
        """Negative warmup iterations raises ValueError."""
        with pytest.raises(ValueError, match="warmup_iterations must be non-negative"):
            GraphSafetyConfig(warmup_iterations=-1)

    def test_invalid_timeout_zero(self) -> None:
        """Zero timeout raises ValueError."""
        with pytest.raises(ValueError, match="capture_timeout_ms must be positive"):
            GraphSafetyConfig(capture_timeout_ms=0)

    def test_invalid_timeout_negative(self) -> None:
        """Negative timeout raises ValueError."""
        with pytest.raises(ValueError, match="capture_timeout_ms must be positive"):
            GraphSafetyConfig(capture_timeout_ms=-1000)

    def test_from_env_defaults(self) -> None:
        """from_env uses defaults when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            config = GraphSafetyConfig.from_env()

        assert config.strict_mode is False
        assert config.memory_delta_warning_mb == 1.0
        assert config.warmup_iterations == 3
        assert config.default_graph_safe is False

    def test_from_env_strict_mode_true(self) -> None:
        """from_env parses LAYERZERO_GRAPH_STRICT_MODE."""
        with patch.dict(os.environ, {"LAYERZERO_GRAPH_STRICT_MODE": "1"}):
            config = GraphSafetyConfig.from_env()
            assert config.strict_mode is True

        with patch.dict(os.environ, {"LAYERZERO_GRAPH_STRICT_MODE": "true"}):
            config = GraphSafetyConfig.from_env()
            assert config.strict_mode is True

        with patch.dict(os.environ, {"LAYERZERO_GRAPH_STRICT_MODE": "yes"}):
            config = GraphSafetyConfig.from_env()
            assert config.strict_mode is True

    def test_from_env_strict_mode_false(self) -> None:
        """from_env parses LAYERZERO_GRAPH_STRICT_MODE as false."""
        with patch.dict(os.environ, {"LAYERZERO_GRAPH_STRICT_MODE": "0"}):
            config = GraphSafetyConfig.from_env()
            assert config.strict_mode is False

        with patch.dict(os.environ, {"LAYERZERO_GRAPH_STRICT_MODE": "false"}):
            config = GraphSafetyConfig.from_env()
            assert config.strict_mode is False

    def test_from_env_memory_warning(self) -> None:
        """from_env parses LAYERZERO_GRAPH_MEMORY_WARNING_MB."""
        with patch.dict(os.environ, {"LAYERZERO_GRAPH_MEMORY_WARNING_MB": "5.5"}):
            config = GraphSafetyConfig.from_env()
            assert config.memory_delta_warning_mb == 5.5

    def test_from_env_warmup_iterations(self) -> None:
        """from_env parses LAYERZERO_GRAPH_WARMUP_ITERATIONS."""
        with patch.dict(os.environ, {"LAYERZERO_GRAPH_WARMUP_ITERATIONS": "10"}):
            config = GraphSafetyConfig.from_env()
            assert config.warmup_iterations == 10

    def test_from_env_default_safe(self) -> None:
        """from_env parses LAYERZERO_GRAPH_DEFAULT_SAFE."""
        with patch.dict(os.environ, {"LAYERZERO_GRAPH_DEFAULT_SAFE": "1"}):
            config = GraphSafetyConfig.from_env()
            assert config.default_graph_safe is True


class TestGraphValidationResult:
    """Tests for GraphValidationResult dataclass."""

    def test_success_result(self) -> None:
        """Successful result has correct attributes."""
        result = GraphValidationResult(
            success=True,
            kernel_id="test_kernel",
            operation="attention",
        )

        assert result.success is True
        assert result.kernel_id == "test_kernel"
        assert result.operation == "attention"
        assert result.error is None
        assert result.warnings == []
        assert result.has_warnings is False

    def test_failure_result(self) -> None:
        """Failed result has correct attributes."""
        result = GraphValidationResult(
            success=False,
            kernel_id="bad_kernel",
            operation="dynamic_shape_op",
            error="Kernel not graph-safe",
        )

        assert result.success is False
        assert result.error == "Kernel not graph-safe"

    def test_add_warning(self) -> None:
        """add_warning appends to warnings list."""
        result = GraphValidationResult(success=True)

        result.add_warning("Warning 1")
        result.add_warning("Warning 2")

        assert len(result.warnings) == 2
        assert "Warning 1" in result.warnings
        assert "Warning 2" in result.warnings
        assert result.has_warnings is True

    def test_has_warnings_false(self) -> None:
        """has_warnings is False when no warnings."""
        result = GraphValidationResult(success=True)
        assert result.has_warnings is False

    def test_has_warnings_true(self) -> None:
        """has_warnings is True when warnings exist."""
        result = GraphValidationResult(success=True)
        result.warnings = ["Some warning"]
        assert result.has_warnings is True

    def test_memory_delta_default(self) -> None:
        """memory_delta_mb defaults to 0."""
        result = GraphValidationResult(success=True)
        assert result.memory_delta_mb == 0.0

    def test_capture_time_default(self) -> None:
        """capture_time_ms defaults to 0."""
        result = GraphValidationResult(success=True)
        assert result.capture_time_ms == 0.0

    def test_library_init_defaults(self) -> None:
        """Library init flags default to False."""
        result = GraphValidationResult(success=True)
        assert result.cublas_initialized is False
        assert result.cudnn_initialized is False

    def test_metadata_default(self) -> None:
        """metadata defaults to empty dict."""
        result = GraphValidationResult(success=True)
        assert result.metadata == {}

    def test_str_success(self) -> None:
        """String representation for success."""
        result = GraphValidationResult(
            success=True,
            kernel_id="test_kernel",
            operation="attention",
        )

        s = str(result)
        assert "SUCCESS" in s
        assert "test_kernel" in s
        assert "attention" in s

    def test_str_failure(self) -> None:
        """String representation for failure."""
        result = GraphValidationResult(
            success=False,
            kernel_id="bad_kernel",
            error="Not graph-safe",
        )

        s = str(result)
        assert "FAILED" in s
        assert "bad_kernel" in s
        assert "Not graph-safe" in s

    def test_str_with_memory_delta(self) -> None:
        """String representation includes memory delta."""
        result = GraphValidationResult(
            success=True,
            memory_delta_mb=2.5,
        )

        s = str(result)
        assert "2.50MB" in s

    def test_str_with_capture_time(self) -> None:
        """String representation includes capture time."""
        result = GraphValidationResult(
            success=True,
            capture_time_ms=150.5,
        )

        s = str(result)
        assert "150.5ms" in s
