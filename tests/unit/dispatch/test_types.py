"""
Unit tests for dispatch/types.py module.

Tests cover:
- DispatchMode enum values and behavior
- DispatchPhase enum values
- DispatchTiming dataclass properties and calculations
- DispatchResult dataclass and properties
- DispatchConfig validation and defaults
- Exception hierarchy (DispatchError, KernelExecutionError, etc.)
- Type aliases

All tests use pytest and follow standard conventions for the LayerZero project.
"""
from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import torch

from layerzero.dispatch.types import (
    DispatchMode,
    DispatchPhase,
    DispatchTiming,
    DispatchResult,
    DispatchConfig,
    DispatchError,
    KernelExecutionError,
    TransformError,
    FallbackChainExhaustedError,
    CircuitOpenError,
    ConfigurationError,
    HotReloadError,
    KernelId,
    OperationId,
    CacheKey,
    PolicyHash,
)
from layerzero.enums import Platform
from layerzero.models.kernel_spec import KernelSpec


# ============================================================================
# DispatchMode Tests
# ============================================================================


class TestDispatchMode:
    """Tests for DispatchMode enum."""

    def test_enum_values_exist(self) -> None:
        """All expected dispatch modes exist."""
        assert hasattr(DispatchMode, "STATIC")
        assert hasattr(DispatchMode, "DYNAMIC")
        assert hasattr(DispatchMode, "HOT_RELOAD")
        assert hasattr(DispatchMode, "CONFIG")
        assert hasattr(DispatchMode, "AUTO")

    def test_enum_values_are_unique(self) -> None:
        """All dispatch mode values are unique."""
        values = [
            DispatchMode.STATIC,
            DispatchMode.DYNAMIC,
            DispatchMode.HOT_RELOAD,
            DispatchMode.CONFIG,
            DispatchMode.AUTO,
        ]
        assert len(values) == len(set(values))

    def test_enum_is_hashable(self) -> None:
        """DispatchMode is hashable for use as dict keys."""
        modes_dict = {
            DispatchMode.STATIC: "static",
            DispatchMode.DYNAMIC: "dynamic",
        }
        assert modes_dict[DispatchMode.STATIC] == "static"

    def test_enum_comparison(self) -> None:
        """DispatchMode can be compared for equality."""
        assert DispatchMode.STATIC == DispatchMode.STATIC
        assert DispatchMode.STATIC != DispatchMode.DYNAMIC

    def test_enum_name_property(self) -> None:
        """DispatchMode has correct name property."""
        assert DispatchMode.STATIC.name == "STATIC"
        assert DispatchMode.DYNAMIC.name == "DYNAMIC"
        assert DispatchMode.HOT_RELOAD.name == "HOT_RELOAD"
        assert DispatchMode.CONFIG.name == "CONFIG"
        assert DispatchMode.AUTO.name == "AUTO"


# ============================================================================
# DispatchPhase Tests
# ============================================================================


class TestDispatchPhase:
    """Tests for DispatchPhase enum."""

    def test_enum_values_exist(self) -> None:
        """All expected dispatch phases exist."""
        assert hasattr(DispatchPhase, "SELECTION")
        assert hasattr(DispatchPhase, "PRE_TRANSFORM")
        assert hasattr(DispatchPhase, "EXECUTION")
        assert hasattr(DispatchPhase, "POST_TRANSFORM")
        assert hasattr(DispatchPhase, "FALLBACK")

    def test_enum_values_are_unique(self) -> None:
        """All dispatch phase values are unique."""
        values = [
            DispatchPhase.SELECTION,
            DispatchPhase.PRE_TRANSFORM,
            DispatchPhase.EXECUTION,
            DispatchPhase.POST_TRANSFORM,
            DispatchPhase.FALLBACK,
        ]
        assert len(values) == len(set(values))

    def test_enum_is_hashable(self) -> None:
        """DispatchPhase is hashable."""
        phases_set = {DispatchPhase.SELECTION, DispatchPhase.EXECUTION}
        assert len(phases_set) == 2

    def test_enum_name_property(self) -> None:
        """DispatchPhase has correct name property."""
        assert DispatchPhase.SELECTION.name == "SELECTION"
        assert DispatchPhase.EXECUTION.name == "EXECUTION"
        assert DispatchPhase.FALLBACK.name == "FALLBACK"


# ============================================================================
# DispatchTiming Tests
# ============================================================================


class TestDispatchTiming:
    """Tests for DispatchTiming dataclass."""

    def test_default_values(self) -> None:
        """DispatchTiming has correct default values."""
        timing = DispatchTiming()
        assert timing.selection_ns == 0
        assert timing.pre_transform_ns == 0
        assert timing.execution_ns == 0
        assert timing.post_transform_ns == 0
        assert timing.total_ns == 0

    def test_custom_values(self) -> None:
        """DispatchTiming accepts custom values."""
        timing = DispatchTiming(
            selection_ns=1000,
            pre_transform_ns=500,
            execution_ns=5000,
            post_transform_ns=200,
            total_ns=7000,
        )
        assert timing.selection_ns == 1000
        assert timing.pre_transform_ns == 500
        assert timing.execution_ns == 5000
        assert timing.post_transform_ns == 200
        assert timing.total_ns == 7000

    def test_selection_us_property(self) -> None:
        """selection_us converts nanoseconds to microseconds."""
        timing = DispatchTiming(selection_ns=1500)
        assert timing.selection_us == 1.5

    def test_execution_us_property(self) -> None:
        """execution_us converts nanoseconds to microseconds."""
        timing = DispatchTiming(execution_ns=2500)
        assert timing.execution_us == 2.5

    def test_total_us_property(self) -> None:
        """total_us converts nanoseconds to microseconds."""
        timing = DispatchTiming(total_ns=3500)
        assert timing.total_us == 3.5

    def test_overhead_ns_property(self) -> None:
        """overhead_ns calculates non-execution time."""
        timing = DispatchTiming(
            selection_ns=1000,
            pre_transform_ns=500,
            execution_ns=3000,
            post_transform_ns=200,
            total_ns=5000,
        )
        # overhead = total - execution = 5000 - 3000 = 2000
        assert timing.overhead_ns == 2000

    def test_is_frozen(self) -> None:
        """DispatchTiming is immutable (frozen)."""
        timing = DispatchTiming(selection_ns=1000)
        with pytest.raises(FrozenInstanceError):
            timing.selection_ns = 2000  # type: ignore

    def test_zero_division_safety(self) -> None:
        """Properties handle zero values correctly."""
        timing = DispatchTiming()
        assert timing.selection_us == 0.0
        assert timing.execution_us == 0.0
        assert timing.total_us == 0.0
        assert timing.overhead_ns == 0

    def test_large_values(self) -> None:
        """Handles large timing values (long-running operations)."""
        timing = DispatchTiming(
            selection_ns=1_000_000_000,  # 1 second
            execution_ns=10_000_000_000,  # 10 seconds
            total_ns=11_000_000_000,
        )
        assert timing.selection_us == 1_000_000.0
        assert timing.execution_us == 10_000_000.0


# ============================================================================
# DispatchResult Tests
# ============================================================================


class TestDispatchResult:
    """Tests for DispatchResult dataclass."""

    @pytest.fixture
    def mock_kernel_spec(self) -> KernelSpec:
        """Create a mock kernel spec for testing."""
        return KernelSpec(
            kernel_id="test_kernel.v1",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=None,
            platform=Platform.CUDA,
        )

    @pytest.fixture
    def mock_timing(self) -> DispatchTiming:
        """Create a mock timing for testing."""
        return DispatchTiming(
            selection_ns=1000,
            execution_ns=5000,
            total_ns=7000,
        )

    def test_required_fields(
        self, mock_kernel_spec: KernelSpec, mock_timing: DispatchTiming
    ) -> None:
        """DispatchResult requires all mandatory fields."""
        output = torch.zeros(1)
        result = DispatchResult(
            output=output,
            kernel_id="test_kernel.v1",
            kernel_spec=mock_kernel_spec,
            timing=mock_timing,
            mode=DispatchMode.DYNAMIC,
        )
        assert result.output is output
        assert result.kernel_id == "test_kernel.v1"
        assert result.kernel_spec is mock_kernel_spec
        assert result.timing is mock_timing
        assert result.mode == DispatchMode.DYNAMIC

    def test_default_optional_fields(
        self, mock_kernel_spec: KernelSpec, mock_timing: DispatchTiming
    ) -> None:
        """DispatchResult has correct default optional fields."""
        result = DispatchResult(
            output=torch.zeros(1),
            kernel_id="test",
            kernel_spec=mock_kernel_spec,
            timing=mock_timing,
            mode=DispatchMode.DYNAMIC,
        )
        assert result.cached is False
        assert result.fallback_used is False
        assert result.fallback_reason is None

    def test_custom_optional_fields(
        self, mock_kernel_spec: KernelSpec, mock_timing: DispatchTiming
    ) -> None:
        """DispatchResult accepts custom optional fields."""
        result = DispatchResult(
            output=torch.zeros(1),
            kernel_id="fallback_kernel",
            kernel_spec=mock_kernel_spec,
            timing=mock_timing,
            mode=DispatchMode.DYNAMIC,
            cached=True,
            fallback_used=True,
            fallback_reason="Primary kernel failed",
        )
        assert result.cached is True
        assert result.fallback_used is True
        assert result.fallback_reason == "Primary kernel failed"

    def test_overhead_us_property(
        self, mock_kernel_spec: KernelSpec
    ) -> None:
        """overhead_us calculates dispatch overhead correctly."""
        timing = DispatchTiming(
            selection_ns=1000,
            execution_ns=5000,
            total_ns=7000,
        )
        result = DispatchResult(
            output=torch.zeros(1),
            kernel_id="test",
            kernel_spec=mock_kernel_spec,
            timing=timing,
            mode=DispatchMode.DYNAMIC,
        )
        # overhead_ns = 7000 - 5000 = 2000, us = 2000 / 1000 = 2.0
        assert result.overhead_us == 2.0

    def test_is_frozen(
        self, mock_kernel_spec: KernelSpec, mock_timing: DispatchTiming
    ) -> None:
        """DispatchResult is immutable (frozen)."""
        result = DispatchResult(
            output=torch.zeros(1),
            kernel_id="test",
            kernel_spec=mock_kernel_spec,
            timing=mock_timing,
            mode=DispatchMode.DYNAMIC,
        )
        with pytest.raises(FrozenInstanceError):
            result.kernel_id = "modified"  # type: ignore


# ============================================================================
# DispatchConfig Tests
# ============================================================================


class TestDispatchConfig:
    """Tests for DispatchConfig dataclass."""

    def test_default_values(self) -> None:
        """DispatchConfig has correct default values."""
        config = DispatchConfig()
        assert config.mode == DispatchMode.DYNAMIC
        assert config.enable_cache is True
        assert config.cache_size == 10000
        assert config.cache_ttl_seconds == 3600.0
        assert config.enable_fallback is True
        assert config.max_fallback_attempts == 3
        assert config.fallback_timeout_ms == 100.0
        assert config.config_path is None
        assert config.watch_interval_seconds == 1.0
        assert config.validate_on_reload is True
        assert config.enable_transforms is True
        assert config.enable_cuda_graphs is False
        assert config.sync_after_execution is False
        assert config.enable_telemetry is True
        assert config.record_timing is True
        assert config.log_fallbacks is True
        assert config.circuit_breaker_enabled is True
        assert config.failure_threshold == 5
        assert config.recovery_timeout_seconds == 30.0
        assert config.static_kernel_map == {}

    def test_custom_values(self) -> None:
        """DispatchConfig accepts custom values."""
        config = DispatchConfig(
            mode=DispatchMode.STATIC,
            enable_cache=False,
            cache_size=5000,
            failure_threshold=10,
            static_kernel_map={"op1": "kernel1"},
        )
        assert config.mode == DispatchMode.STATIC
        assert config.enable_cache is False
        assert config.cache_size == 5000
        assert config.failure_threshold == 10
        assert config.static_kernel_map == {"op1": "kernel1"}

    def test_validation_negative_cache_size(self) -> None:
        """DispatchConfig rejects negative cache_size."""
        with pytest.raises(ValueError, match="cache_size must be non-negative"):
            DispatchConfig(cache_size=-1)

    def test_validation_zero_max_fallback_attempts(self) -> None:
        """DispatchConfig rejects max_fallback_attempts < 1."""
        with pytest.raises(ValueError, match="max_fallback_attempts must be at least 1"):
            DispatchConfig(max_fallback_attempts=0)

    def test_validation_negative_max_fallback_attempts(self) -> None:
        """DispatchConfig rejects negative max_fallback_attempts."""
        with pytest.raises(ValueError, match="max_fallback_attempts must be at least 1"):
            DispatchConfig(max_fallback_attempts=-5)

    def test_validation_zero_failure_threshold(self) -> None:
        """DispatchConfig rejects failure_threshold < 1."""
        with pytest.raises(ValueError, match="failure_threshold must be at least 1"):
            DispatchConfig(failure_threshold=0)

    def test_validation_negative_failure_threshold(self) -> None:
        """DispatchConfig rejects negative failure_threshold."""
        with pytest.raises(ValueError, match="failure_threshold must be at least 1"):
            DispatchConfig(failure_threshold=-1)

    def test_validation_allows_zero_cache_size(self) -> None:
        """DispatchConfig allows cache_size=0 (disabled)."""
        config = DispatchConfig(cache_size=0)
        assert config.cache_size == 0

    def test_is_mutable(self) -> None:
        """DispatchConfig is mutable (not frozen)."""
        config = DispatchConfig()
        config.enable_cache = False
        assert config.enable_cache is False

    def test_hot_reload_mode_with_config_path(self) -> None:
        """DispatchConfig can be configured for hot-reload mode."""
        config = DispatchConfig(
            mode=DispatchMode.HOT_RELOAD,
            config_path="/path/to/config.yaml",
            watch_interval_seconds=0.5,
        )
        assert config.mode == DispatchMode.HOT_RELOAD
        assert config.config_path == "/path/to/config.yaml"
        assert config.watch_interval_seconds == 0.5


# ============================================================================
# Exception Tests
# ============================================================================


class TestDispatchError:
    """Tests for DispatchError base exception."""

    def test_basic_message(self) -> None:
        """DispatchError can be created with just a message."""
        error = DispatchError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.operation is None
        assert error.kernel_id is None
        assert error.phase is None

    def test_full_attributes(self) -> None:
        """DispatchError accepts all optional attributes."""
        error = DispatchError(
            message="Kernel failed",
            operation="attention.causal",
            kernel_id="flash_attn.v3",
            phase=DispatchPhase.EXECUTION,
        )
        assert str(error) == "Kernel failed"
        assert error.operation == "attention.causal"
        assert error.kernel_id == "flash_attn.v3"
        assert error.phase == DispatchPhase.EXECUTION

    def test_is_exception(self) -> None:
        """DispatchError is a proper Exception subclass."""
        error = DispatchError("Test")
        assert isinstance(error, Exception)

        with pytest.raises(DispatchError):
            raise error


class TestKernelExecutionError:
    """Tests for KernelExecutionError."""

    def test_required_attributes(self) -> None:
        """KernelExecutionError requires operation and kernel_id."""
        error = KernelExecutionError(
            message="Execution failed",
            operation="attention.causal",
            kernel_id="test_kernel",
        )
        assert str(error) == "Execution failed"
        assert error.operation == "attention.causal"
        assert error.kernel_id == "test_kernel"
        assert error.phase == DispatchPhase.EXECUTION
        assert error.original_error is None

    def test_with_original_error(self) -> None:
        """KernelExecutionError can wrap an original error."""
        original = ValueError("Invalid input")
        error = KernelExecutionError(
            message="Kernel failed",
            operation="norm",
            kernel_id="liger_norm",
            original_error=original,
        )
        assert error.original_error is original

    def test_is_dispatch_error(self) -> None:
        """KernelExecutionError is a DispatchError subclass."""
        error = KernelExecutionError("msg", "op", "kernel")
        assert isinstance(error, DispatchError)


class TestTransformError:
    """Tests for TransformError."""

    def test_required_attributes(self) -> None:
        """TransformError requires transform_type."""
        error = TransformError(
            message="Transform failed",
            transform_type="layout_bhsd_to_bshd",
        )
        assert str(error) == "Transform failed"
        assert error.transform_type == "layout_bhsd_to_bshd"
        assert error.phase == DispatchPhase.PRE_TRANSFORM
        assert error.original_error is None

    def test_with_original_error(self) -> None:
        """TransformError can wrap an original error."""
        original = RuntimeError("Shape mismatch")
        error = TransformError(
            message="Failed",
            transform_type="dtype_cast",
            original_error=original,
        )
        assert error.original_error is original

    def test_is_dispatch_error(self) -> None:
        """TransformError is a DispatchError subclass."""
        error = TransformError("msg", "type")
        assert isinstance(error, DispatchError)


class TestFallbackChainExhaustedError:
    """Tests for FallbackChainExhaustedError."""

    def test_required_attributes(self) -> None:
        """FallbackChainExhaustedError requires operation, kernels, and errors."""
        errors = [ValueError("e1"), RuntimeError("e2")]
        error = FallbackChainExhaustedError(
            operation="attention",
            attempted_kernels=["k1", "k2"],
            errors=errors,
        )
        assert "attention" in str(error)
        assert error.operation == "attention"
        assert error.attempted_kernels == ["k1", "k2"]
        assert error.errors == errors
        assert error.phase == DispatchPhase.FALLBACK

    def test_message_format(self) -> None:
        """FallbackChainExhaustedError formats message with kernel list."""
        error = FallbackChainExhaustedError(
            operation="norm",
            attempted_kernels=["liger", "triton", "torch"],
            errors=[Exception("e1"), Exception("e2"), Exception("e3")],
        )
        message = str(error)
        assert "norm" in message
        assert "3 kernel(s)" in message
        assert "['liger', 'triton', 'torch']" in message

    def test_is_dispatch_error(self) -> None:
        """FallbackChainExhaustedError is a DispatchError subclass."""
        error = FallbackChainExhaustedError("op", [], [])
        assert isinstance(error, DispatchError)


class TestCircuitOpenError:
    """Tests for CircuitOpenError."""

    def test_required_attributes(self) -> None:
        """CircuitOpenError requires kernel_id and retry_after."""
        error = CircuitOpenError(
            kernel_id="flash_attn",
            retry_after_seconds=30.5,
        )
        assert error.kernel_id == "flash_attn"
        assert error.retry_after_seconds == 30.5

    def test_message_format(self) -> None:
        """CircuitOpenError formats message with retry time."""
        error = CircuitOpenError("test_kernel", 15.0)
        message = str(error)
        assert "test_kernel" in message
        assert "15.0" in message
        assert "Retry" in message or "retry" in message

    def test_is_dispatch_error(self) -> None:
        """CircuitOpenError is a DispatchError subclass."""
        error = CircuitOpenError("kernel", 0.0)
        assert isinstance(error, DispatchError)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_basic_message(self) -> None:
        """ConfigurationError can be created with just a message."""
        error = ConfigurationError("Invalid config")
        assert str(error) == "Invalid config"

    def test_is_dispatch_error(self) -> None:
        """ConfigurationError is a DispatchError subclass."""
        error = ConfigurationError("msg")
        assert isinstance(error, DispatchError)


class TestHotReloadError:
    """Tests for HotReloadError."""

    def test_required_attributes(self) -> None:
        """HotReloadError requires config_path."""
        error = HotReloadError(
            message="Failed to reload",
            config_path="/etc/config.yaml",
        )
        assert str(error) == "Failed to reload"
        assert error.config_path == "/etc/config.yaml"
        assert error.original_error is None

    def test_with_original_error(self) -> None:
        """HotReloadError can wrap an original error."""
        original = FileNotFoundError("File not found")
        error = HotReloadError(
            message="Reload failed",
            config_path="/missing/config.yaml",
            original_error=original,
        )
        assert error.original_error is original

    def test_is_dispatch_error(self) -> None:
        """HotReloadError is a DispatchError subclass."""
        error = HotReloadError("msg", "/path")
        assert isinstance(error, DispatchError)


# ============================================================================
# Type Alias Tests
# ============================================================================


class TestTypeAliases:
    """Tests for type aliases."""

    def test_kernel_id_is_string(self) -> None:
        """KernelId is aliased to str."""
        kernel_id: KernelId = "flash_attn.v3.causal"
        assert isinstance(kernel_id, str)

    def test_operation_id_is_string(self) -> None:
        """OperationId is aliased to str."""
        op_id: OperationId = "attention.causal"
        assert isinstance(op_id, str)

    def test_cache_key_is_string(self) -> None:
        """CacheKey is aliased to str."""
        cache_key: CacheKey = "hash_abc123"
        assert isinstance(cache_key, str)

    def test_policy_hash_is_string(self) -> None:
        """PolicyHash is aliased to str."""
        policy_hash: PolicyHash = "policy_v1_abc"
        assert isinstance(policy_hash, str)


# ============================================================================
# Exception Chaining Tests
# ============================================================================


class TestExceptionChaining:
    """Tests for exception chaining and context."""

    def test_kernel_execution_error_chain(self) -> None:
        """KernelExecutionError can be chained with raise from."""
        original = ValueError("Bad value")

        try:
            try:
                raise original
            except ValueError as e:
                raise KernelExecutionError(
                    "Execution failed",
                    "op",
                    "kernel",
                    original_error=e,
                ) from e
        except KernelExecutionError as err:
            assert err.__cause__ is original
            assert err.original_error is original

    def test_dispatch_error_as_context(self) -> None:
        """DispatchError can be used as context in try/except."""
        def failing_function() -> None:
            raise DispatchError("Inner error")

        with pytest.raises(DispatchError) as exc_info:
            failing_function()

        assert "Inner error" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
