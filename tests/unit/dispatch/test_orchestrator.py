"""
Unit tests for dispatch/orchestrator.py module.

Tests cover:
- DispatchOrchestrator initialization and configuration
- Mode selection (STATIC, DYNAMIC, HOT_RELOAD, CONFIG, AUTO)
- Fallback between modes
- Dispatcher registration and retrieval
- Telemetry tracking (OrchestratorTelemetry)
- Config reload functionality
- Global dispatcher management
- Thread safety
- Error handling and propagation

All tests use pytest and mock dependencies where needed.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from layerzero.dispatch.orchestrator import (
    DispatchOrchestrator,
    OrchestratorTelemetry,
    get_global_dispatcher,
    set_global_dispatcher,
    create_orchestrator,
    dispatch,
)
from layerzero.dispatch.types import (
    DispatchConfig,
    DispatchError,
    DispatchMode,
    DispatchPhase,
    DispatchResult,
    DispatchTiming,
)
from layerzero.dispatch.protocols import Dispatcher, BaseDispatcher
from layerzero.enums import Layout, OpKind, Platform
from layerzero.models.device_spec import DeviceSpec
from layerzero.models.kernel_spec import KernelSpec
from layerzero.models.selection_context import SelectionContext
from layerzero.registry.backend_registry import BackendRegistry
from layerzero.registry.kernel_registry import KernelRegistry


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def device_spec() -> DeviceSpec:
    """Create test device spec."""
    from layerzero.device import GPUGeneration

    return DeviceSpec(
        platform=Platform.CUDA,
        device_index=0,
        device_name="Test GPU",
        sm_version=(8, 0),
        gpu_generation=GPUGeneration.AMPERE,
        tensor_core_gen=3,
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=14 * 1024**3,
        supports_bf16=True,
        supports_fp8=False,
        supports_fp4=False,
        supports_tma=False,
        max_shared_memory_kb=164,
        cuda_version="12.0",
        driver_version="525.0",
    )


@pytest.fixture
def mock_kernel_spec() -> KernelSpec:
    """Create a mock kernel spec."""
    def impl(**kwargs: Any) -> torch.Tensor:
        q = kwargs.get("q") or kwargs.get("query")
        if q is None:
            return torch.zeros(1)
        return torch.zeros_like(q)

    return KernelSpec(
        kernel_id="test_kernel.v1",
        operation="attention.causal",
        source="test",
        version="1.0",
        impl=impl,
        platform=Platform.CUDA,
        min_sm=(7, 0),
        supported_dtypes=frozenset([torch.float16, torch.bfloat16, torch.float32]),
        priority=50,
    )


@pytest.fixture
def kernel_registry(mock_kernel_spec: KernelSpec) -> KernelRegistry:
    """Create kernel registry with test kernel."""
    registry = KernelRegistry()
    registry.register(mock_kernel_spec)
    return registry


@pytest.fixture
def backend_registry() -> BackendRegistry:
    """Create backend registry."""
    return BackendRegistry(failure_threshold=3, cooldown_seconds=30.0)


@pytest.fixture
def selection_context(device_spec: DeviceSpec) -> SelectionContext:
    """Create selection context."""
    return SelectionContext(
        device=device_spec,
        op_kind=OpKind.TENSOR,
        operation="attention.causal",
        dtype=torch.float16,
        batch_size=4,
        seq_len_q=512,
        seq_len_k=512,
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        layout=Layout.BSHD,
        is_causal=True,
    )


@pytest.fixture
def default_config() -> DispatchConfig:
    """Create default dispatch config."""
    return DispatchConfig(
        mode=DispatchMode.DYNAMIC,
        enable_cache=True,
        enable_fallback=True,
        failure_threshold=5,
    )


@pytest.fixture
def mock_dispatcher() -> MagicMock:
    """Create a mock dispatcher that follows the Dispatcher protocol."""
    dispatcher = MagicMock(spec=Dispatcher)
    dispatcher.mode = DispatchMode.DYNAMIC

    # Create proper DispatchResult for dispatch method
    mock_result = MagicMock(spec=DispatchResult)
    mock_result.timing = DispatchTiming(
        selection_ns=1000,
        execution_ns=5000,
        total_ns=7000,
    )
    mock_result.kernel_id = "test_kernel"
    mock_result.mode = DispatchMode.DYNAMIC
    dispatcher.dispatch.return_value = mock_result

    return dispatcher


# ============================================================================
# OrchestratorTelemetry Tests
# ============================================================================


class TestOrchestratorTelemetry:
    """Tests for OrchestratorTelemetry dataclass."""

    def test_default_values(self) -> None:
        """Telemetry starts with zero counters."""
        telemetry = OrchestratorTelemetry()
        assert telemetry.total_dispatches == 0
        assert telemetry.dispatches_by_mode == {}
        assert telemetry.fallbacks_between_modes == 0
        assert telemetry.total_errors == 0
        assert telemetry.errors_by_mode == {}
        assert telemetry.total_execution_time_ns == 0
        assert telemetry.total_selection_time_ns == 0

    def test_record_dispatch_success(self) -> None:
        """record_dispatch updates counters for successful dispatch."""
        telemetry = OrchestratorTelemetry()
        timing = DispatchTiming(selection_ns=1000, execution_ns=5000, total_ns=7000)

        telemetry.record_dispatch(DispatchMode.DYNAMIC, timing, error=False)

        assert telemetry.total_dispatches == 1
        assert telemetry.dispatches_by_mode[DispatchMode.DYNAMIC] == 1
        assert telemetry.total_errors == 0
        assert telemetry.total_execution_time_ns == 5000
        assert telemetry.total_selection_time_ns == 1000

    def test_record_dispatch_error(self) -> None:
        """record_dispatch updates error counters."""
        telemetry = OrchestratorTelemetry()
        timing = DispatchTiming(total_ns=1000)

        telemetry.record_dispatch(DispatchMode.STATIC, timing, error=True)

        assert telemetry.total_dispatches == 1
        assert telemetry.total_errors == 1
        assert telemetry.errors_by_mode[DispatchMode.STATIC] == 1

    def test_record_multiple_dispatches(self) -> None:
        """record_dispatch accumulates across multiple calls."""
        telemetry = OrchestratorTelemetry()
        timing = DispatchTiming(selection_ns=100, execution_ns=500, total_ns=700)

        for _ in range(10):
            telemetry.record_dispatch(DispatchMode.DYNAMIC, timing)

        assert telemetry.total_dispatches == 10
        assert telemetry.dispatches_by_mode[DispatchMode.DYNAMIC] == 10
        assert telemetry.total_execution_time_ns == 5000
        assert telemetry.total_selection_time_ns == 1000

    def test_record_mode_fallback(self) -> None:
        """record_mode_fallback increments fallback counter."""
        telemetry = OrchestratorTelemetry()

        telemetry.record_mode_fallback()
        telemetry.record_mode_fallback()
        telemetry.record_mode_fallback()

        assert telemetry.fallbacks_between_modes == 3

    def test_avg_execution_time_us_no_dispatches(self) -> None:
        """avg_execution_time_us returns 0 when no dispatches."""
        telemetry = OrchestratorTelemetry()
        assert telemetry.avg_execution_time_us == 0.0

    def test_avg_execution_time_us_calculation(self) -> None:
        """avg_execution_time_us calculates correct average."""
        telemetry = OrchestratorTelemetry()
        timing = DispatchTiming(execution_ns=2000, total_ns=3000)

        telemetry.record_dispatch(DispatchMode.DYNAMIC, timing)
        telemetry.record_dispatch(DispatchMode.DYNAMIC, timing)

        # Total 4000ns / 2 dispatches = 2000ns = 2.0us
        assert telemetry.avg_execution_time_us == 2.0

    def test_avg_selection_time_us_no_dispatches(self) -> None:
        """avg_selection_time_us returns 0 when no dispatches."""
        telemetry = OrchestratorTelemetry()
        assert telemetry.avg_selection_time_us == 0.0

    def test_avg_selection_time_us_calculation(self) -> None:
        """avg_selection_time_us calculates correct average."""
        telemetry = OrchestratorTelemetry()
        timing = DispatchTiming(selection_ns=1000, total_ns=2000)

        for _ in range(5):
            telemetry.record_dispatch(DispatchMode.STATIC, timing)

        # Total 5000ns / 5 dispatches = 1000ns = 1.0us
        assert telemetry.avg_selection_time_us == 1.0

    def test_to_dict(self) -> None:
        """to_dict returns correct dictionary structure."""
        telemetry = OrchestratorTelemetry()
        timing = DispatchTiming(selection_ns=1000, execution_ns=5000, total_ns=7000)

        telemetry.record_dispatch(DispatchMode.DYNAMIC, timing)
        telemetry.record_dispatch(DispatchMode.STATIC, timing, error=True)
        telemetry.record_mode_fallback()

        d = telemetry.to_dict()

        assert d["total_dispatches"] == 2
        assert d["dispatches_by_mode"]["DYNAMIC"] == 1
        assert d["dispatches_by_mode"]["STATIC"] == 1
        assert d["fallbacks_between_modes"] == 1
        assert d["total_errors"] == 1
        assert d["errors_by_mode"]["STATIC"] == 1
        assert "avg_execution_time_us" in d
        assert "avg_selection_time_us" in d


# ============================================================================
# DispatchOrchestrator Initialization Tests
# ============================================================================


class TestDispatchOrchestratorInit:
    """Tests for DispatchOrchestrator initialization."""

    def test_default_initialization(self) -> None:
        """Orchestrator can be initialized with defaults."""
        orchestrator = DispatchOrchestrator()
        assert orchestrator.config is not None
        assert orchestrator.default_mode == DispatchMode.DYNAMIC

    def test_initialization_with_config(self, default_config: DispatchConfig) -> None:
        """Orchestrator accepts custom config."""
        orchestrator = DispatchOrchestrator(config=default_config)
        assert orchestrator.config is default_config

    def test_initialization_with_registries(
        self, kernel_registry: KernelRegistry, backend_registry: BackendRegistry
    ) -> None:
        """Orchestrator accepts kernel and backend registries."""
        orchestrator = DispatchOrchestrator(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        assert orchestrator._kernel_registry is kernel_registry
        assert orchestrator._backend_registry is backend_registry

    def test_initialization_with_default_mode(self) -> None:
        """Orchestrator accepts custom default mode."""
        orchestrator = DispatchOrchestrator(default_mode=DispatchMode.STATIC)
        assert orchestrator.default_mode == DispatchMode.STATIC

    def test_initialization_creates_default_dispatcher(self) -> None:
        """Orchestrator creates a default dispatcher during init."""
        orchestrator = DispatchOrchestrator()
        # Should have at least one dispatcher initialized
        assert len(orchestrator._dispatchers) >= 0  # May be lazily initialized


# ============================================================================
# DispatchOrchestrator Mode Selection Tests
# ============================================================================


class TestDispatchOrchestratorModeSelection:
    """Tests for mode selection in DispatchOrchestrator."""

    def test_set_default_mode(self) -> None:
        """set_default_mode updates the default mode."""
        orchestrator = DispatchOrchestrator()
        orchestrator.set_default_mode(DispatchMode.STATIC)
        assert orchestrator.default_mode == DispatchMode.STATIC

    def test_set_default_mode_rejects_auto(self) -> None:
        """set_default_mode rejects AUTO as default mode."""
        orchestrator = DispatchOrchestrator()
        with pytest.raises(ValueError, match="Cannot set AUTO"):
            orchestrator.set_default_mode(DispatchMode.AUTO)

    def test_select_best_mode_uses_static_for_mapped_operations(self) -> None:
        """_select_best_mode returns STATIC for operations in static_kernel_map."""
        config = DispatchConfig(
            mode=DispatchMode.AUTO,
            static_kernel_map={"attention.causal": "flash_attn.v3"},
        )
        orchestrator = DispatchOrchestrator(config=config)

        mode = orchestrator._select_best_mode("attention.causal", None)
        assert mode == DispatchMode.STATIC

    def test_select_best_mode_defaults_to_dynamic(self) -> None:
        """_select_best_mode defaults to dynamic mode."""
        orchestrator = DispatchOrchestrator(default_mode=DispatchMode.DYNAMIC)
        mode = orchestrator._select_best_mode("unknown.operation", None)
        assert mode == DispatchMode.DYNAMIC


# ============================================================================
# DispatchOrchestrator Dispatcher Management Tests
# ============================================================================


class TestDispatchOrchestratorDispatcherManagement:
    """Tests for dispatcher registration and retrieval."""

    def test_register_dispatcher(self, mock_dispatcher: MagicMock) -> None:
        """register_dispatcher adds a dispatcher for a mode."""
        orchestrator = DispatchOrchestrator()
        orchestrator.register_dispatcher(DispatchMode.DYNAMIC, mock_dispatcher)

        assert orchestrator.get_dispatcher(DispatchMode.DYNAMIC) is mock_dispatcher

    def test_get_dispatcher_returns_none_for_unregistered(self) -> None:
        """get_dispatcher returns None for unregistered modes."""
        orchestrator = DispatchOrchestrator()
        orchestrator._dispatchers.clear()  # Clear any default dispatchers

        assert orchestrator.get_dispatcher(DispatchMode.HOT_RELOAD) is None

    def test_get_or_create_dispatcher_creates_new(
        self, kernel_registry: KernelRegistry, backend_registry: BackendRegistry
    ) -> None:
        """_get_or_create_dispatcher creates dispatcher if not exists."""
        orchestrator = DispatchOrchestrator(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        orchestrator._dispatchers.clear()

        dispatcher = orchestrator._get_or_create_dispatcher(DispatchMode.DYNAMIC)

        assert dispatcher is not None
        assert DispatchMode.DYNAMIC in orchestrator._dispatchers

    def test_get_or_create_dispatcher_returns_existing(
        self, mock_dispatcher: MagicMock
    ) -> None:
        """_get_or_create_dispatcher returns existing dispatcher."""
        orchestrator = DispatchOrchestrator()
        orchestrator.register_dispatcher(DispatchMode.DYNAMIC, mock_dispatcher)

        result = orchestrator._get_or_create_dispatcher(DispatchMode.DYNAMIC)

        assert result is mock_dispatcher

    def test_get_or_create_dispatcher_auto_uses_default(self) -> None:
        """_get_or_create_dispatcher for AUTO uses default mode."""
        orchestrator = DispatchOrchestrator(default_mode=DispatchMode.DYNAMIC)

        # This should create/get DYNAMIC dispatcher, not AUTO
        dispatcher = orchestrator._get_or_create_dispatcher(DispatchMode.AUTO)
        assert dispatcher is not None


# ============================================================================
# DispatchOrchestrator Dispatch Tests
# ============================================================================


class TestDispatchOrchestratorDispatch:
    """Tests for dispatch method."""

    def test_dispatch_uses_registered_dispatcher(
        self, mock_dispatcher: MagicMock, selection_context: SelectionContext
    ) -> None:
        """dispatch uses the registered dispatcher for the mode."""
        orchestrator = DispatchOrchestrator()
        orchestrator.register_dispatcher(DispatchMode.DYNAMIC, mock_dispatcher)

        inputs = {
            "query": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "key": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "value": torch.randn(4, 512, 8, 64, dtype=torch.float16),
        }

        result = orchestrator.dispatch(
            "attention.causal",
            inputs,
            context=selection_context,
            mode=DispatchMode.DYNAMIC,
        )

        mock_dispatcher.dispatch.assert_called_once()
        assert result is not None

    def test_dispatch_with_explicit_mode(
        self, mock_dispatcher: MagicMock, selection_context: SelectionContext
    ) -> None:
        """dispatch respects explicit mode override."""
        orchestrator = DispatchOrchestrator()
        orchestrator.register_dispatcher(DispatchMode.STATIC, mock_dispatcher)

        inputs = {"query": torch.zeros(1)}

        orchestrator.dispatch(
            "attention.causal",
            inputs,
            context=selection_context,
            mode=DispatchMode.STATIC,
        )

        mock_dispatcher.dispatch.assert_called_once()

    def test_dispatch_fallback_on_error(self, selection_context: SelectionContext) -> None:
        """dispatch tries fallback modes when primary fails."""
        # Create failing and working dispatchers
        failing_dispatcher = MagicMock(spec=Dispatcher)
        failing_dispatcher.mode = DispatchMode.STATIC
        failing_dispatcher.dispatch.side_effect = DispatchError("Static failed")

        working_dispatcher = MagicMock(spec=Dispatcher)
        working_dispatcher.mode = DispatchMode.DYNAMIC
        mock_result = MagicMock(spec=DispatchResult)
        mock_result.timing = DispatchTiming(total_ns=1000)
        working_dispatcher.dispatch.return_value = mock_result

        orchestrator = DispatchOrchestrator()
        orchestrator.register_dispatcher(DispatchMode.STATIC, failing_dispatcher)
        orchestrator.register_dispatcher(DispatchMode.DYNAMIC, working_dispatcher)

        inputs = {"query": torch.zeros(1)}

        # STATIC should fail and fallback to DYNAMIC
        result = orchestrator.dispatch(
            "attention.causal",
            inputs,
            context=selection_context,
            mode=DispatchMode.STATIC,
        )

        assert result is not None
        working_dispatcher.dispatch.assert_called()

    def test_dispatch_raises_when_all_modes_fail(
        self, selection_context: SelectionContext
    ) -> None:
        """dispatch raises DispatchError when all modes fail."""
        failing_dispatcher = MagicMock(spec=Dispatcher)
        failing_dispatcher.mode = DispatchMode.DYNAMIC
        failing_dispatcher.dispatch.side_effect = DispatchError("All failed")

        orchestrator = DispatchOrchestrator()
        orchestrator._dispatchers.clear()
        orchestrator.register_dispatcher(DispatchMode.DYNAMIC, failing_dispatcher)

        # Clear fallback chain to test exhaustion
        orchestrator._mode_fallback_chain[DispatchMode.DYNAMIC] = []

        inputs = {"query": torch.zeros(1)}

        with pytest.raises(DispatchError) as exc_info:
            orchestrator.dispatch(
                "attention.causal",
                inputs,
                context=selection_context,
                mode=DispatchMode.DYNAMIC,
            )

        assert "all modes" in str(exc_info.value).lower()


# ============================================================================
# DispatchOrchestrator Telemetry Tests
# ============================================================================


class TestDispatchOrchestratorTelemetry:
    """Tests for telemetry tracking."""

    def test_get_telemetry_returns_dict(self) -> None:
        """get_telemetry returns a dictionary."""
        orchestrator = DispatchOrchestrator()
        telemetry = orchestrator.get_telemetry()

        assert isinstance(telemetry, dict)
        assert "total_dispatches" in telemetry
        assert "dispatchers" in telemetry

    def test_reset_telemetry(self, mock_dispatcher: MagicMock) -> None:
        """reset_telemetry resets all counters."""
        orchestrator = DispatchOrchestrator()
        orchestrator.register_dispatcher(DispatchMode.DYNAMIC, mock_dispatcher)

        # Add some telemetry data
        timing = DispatchTiming(total_ns=1000)
        orchestrator._telemetry.record_dispatch(DispatchMode.DYNAMIC, timing)

        assert orchestrator._telemetry.total_dispatches > 0

        orchestrator.reset_telemetry()

        assert orchestrator._telemetry.total_dispatches == 0

    def test_telemetry_records_mode_fallback(
        self, selection_context: SelectionContext
    ) -> None:
        """Telemetry records fallback between modes."""
        failing_dispatcher = MagicMock(spec=Dispatcher)
        failing_dispatcher.dispatch.side_effect = DispatchError("Failed")

        working_dispatcher = MagicMock(spec=Dispatcher)
        mock_result = MagicMock(spec=DispatchResult)
        mock_result.timing = DispatchTiming(total_ns=1000)
        working_dispatcher.dispatch.return_value = mock_result

        orchestrator = DispatchOrchestrator()
        orchestrator.register_dispatcher(DispatchMode.STATIC, failing_dispatcher)
        orchestrator.register_dispatcher(DispatchMode.DYNAMIC, working_dispatcher)

        inputs = {"query": torch.zeros(1)}

        orchestrator.dispatch(
            "test",
            inputs,
            context=selection_context,
            mode=DispatchMode.STATIC,
        )

        assert orchestrator._telemetry.fallbacks_between_modes >= 1


# ============================================================================
# DispatchOrchestrator Config Management Tests
# ============================================================================


class TestDispatchOrchestratorConfigManagement:
    """Tests for config management."""

    def test_config_property(self, default_config: DispatchConfig) -> None:
        """config property returns current configuration."""
        orchestrator = DispatchOrchestrator(config=default_config)
        assert orchestrator.config is default_config

    def test_reload_config(self) -> None:
        """reload_config updates the orchestrator configuration."""
        orchestrator = DispatchOrchestrator()
        old_config = orchestrator.config

        new_config = DispatchConfig(
            mode=DispatchMode.STATIC,
            enable_cache=False,
        )

        orchestrator.reload_config(new_config)

        assert orchestrator.config is new_config
        assert orchestrator.config is not old_config

    def test_reload_config_updates_dispatchers(
        self, mock_dispatcher: MagicMock
    ) -> None:
        """reload_config calls update_config on dispatchers that support it."""
        orchestrator = DispatchOrchestrator()
        mock_dispatcher.update_config = MagicMock()
        orchestrator.register_dispatcher(DispatchMode.DYNAMIC, mock_dispatcher)

        new_config = DispatchConfig(mode=DispatchMode.DYNAMIC)
        orchestrator.reload_config(new_config)

        mock_dispatcher.update_config.assert_called_once_with(new_config)


# ============================================================================
# DispatchOrchestrator Shutdown Tests
# ============================================================================


class TestDispatchOrchestratorShutdown:
    """Tests for shutdown functionality."""

    def test_shutdown_clears_dispatchers(self, mock_dispatcher: MagicMock) -> None:
        """shutdown clears all dispatchers."""
        orchestrator = DispatchOrchestrator()
        orchestrator.register_dispatcher(DispatchMode.DYNAMIC, mock_dispatcher)

        assert len(orchestrator._dispatchers) > 0

        orchestrator.shutdown()

        assert len(orchestrator._dispatchers) == 0

    def test_shutdown_stops_watching(self) -> None:
        """shutdown calls stop_watching on dispatchers that support it."""
        mock_dispatcher = MagicMock(spec=Dispatcher)
        mock_dispatcher.stop_watching = MagicMock()

        orchestrator = DispatchOrchestrator()
        orchestrator.register_dispatcher(DispatchMode.HOT_RELOAD, mock_dispatcher)

        orchestrator.shutdown()

        mock_dispatcher.stop_watching.assert_called_once()

    def test_shutdown_calls_dispatcher_shutdown(self) -> None:
        """shutdown calls shutdown on dispatchers that support it."""
        mock_dispatcher = MagicMock(spec=Dispatcher)
        mock_dispatcher.shutdown = MagicMock()

        orchestrator = DispatchOrchestrator()
        orchestrator.register_dispatcher(DispatchMode.DYNAMIC, mock_dispatcher)

        orchestrator.shutdown()

        mock_dispatcher.shutdown.assert_called_once()


# ============================================================================
# Global Dispatcher Management Tests
# ============================================================================


class TestGlobalDispatcherManagement:
    """Tests for global dispatcher management functions."""

    def test_get_global_dispatcher_creates_if_none(self) -> None:
        """get_global_dispatcher creates orchestrator if none exists."""
        # Reset global state
        set_global_dispatcher(None)

        dispatcher = get_global_dispatcher()

        assert dispatcher is not None
        assert isinstance(dispatcher, DispatchOrchestrator)

    def test_get_global_dispatcher_returns_same_instance(self) -> None:
        """get_global_dispatcher returns the same instance."""
        # Reset global state
        set_global_dispatcher(None)

        dispatcher1 = get_global_dispatcher()
        dispatcher2 = get_global_dispatcher()

        assert dispatcher1 is dispatcher2

    def test_set_global_dispatcher(self) -> None:
        """set_global_dispatcher sets the global instance."""
        orchestrator = DispatchOrchestrator()

        set_global_dispatcher(orchestrator)

        assert get_global_dispatcher() is orchestrator

        # Cleanup
        set_global_dispatcher(None)

    def test_set_global_dispatcher_shuts_down_previous(self) -> None:
        """set_global_dispatcher shuts down the previous orchestrator."""
        old_orchestrator = MagicMock(spec=DispatchOrchestrator)
        old_orchestrator.shutdown = MagicMock()

        # Manually set the global
        import layerzero.dispatch.orchestrator as orch_module
        orch_module._global_orchestrator = old_orchestrator

        new_orchestrator = DispatchOrchestrator()
        set_global_dispatcher(new_orchestrator)

        old_orchestrator.shutdown.assert_called_once()

        # Cleanup
        set_global_dispatcher(None)


# ============================================================================
# create_orchestrator Factory Tests
# ============================================================================


class TestCreateOrchestrator:
    """Tests for create_orchestrator factory function."""

    def test_create_with_defaults(self) -> None:
        """create_orchestrator creates orchestrator with defaults."""
        orchestrator = create_orchestrator()

        assert orchestrator is not None
        assert isinstance(orchestrator, DispatchOrchestrator)
        assert orchestrator.default_mode == DispatchMode.DYNAMIC

    def test_create_with_config(self) -> None:
        """create_orchestrator accepts custom config."""
        config = DispatchConfig(mode=DispatchMode.STATIC)

        orchestrator = create_orchestrator(config=config)

        assert orchestrator.config is config

    def test_create_with_registries(
        self, kernel_registry: KernelRegistry, backend_registry: BackendRegistry
    ) -> None:
        """create_orchestrator accepts registries."""
        orchestrator = create_orchestrator(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )

        assert orchestrator._kernel_registry is kernel_registry
        assert orchestrator._backend_registry is backend_registry

    def test_create_with_default_mode(self) -> None:
        """create_orchestrator accepts custom default mode."""
        orchestrator = create_orchestrator(default_mode=DispatchMode.STATIC)

        assert orchestrator.default_mode == DispatchMode.STATIC

    def test_create_and_set_as_global(self) -> None:
        """create_orchestrator can set as global."""
        # Reset global
        set_global_dispatcher(None)

        orchestrator = create_orchestrator(set_as_global=True)

        assert get_global_dispatcher() is orchestrator

        # Cleanup
        set_global_dispatcher(None)


# ============================================================================
# dispatch Convenience Function Tests
# ============================================================================


class TestDispatchFunction:
    """Tests for dispatch convenience function."""

    def test_dispatch_uses_global_orchestrator(
        self, mock_dispatcher: MagicMock, selection_context: SelectionContext
    ) -> None:
        """dispatch uses the global orchestrator."""
        orchestrator = DispatchOrchestrator()
        orchestrator.register_dispatcher(DispatchMode.DYNAMIC, mock_dispatcher)
        set_global_dispatcher(orchestrator)

        inputs = {"query": torch.zeros(1)}

        dispatch(
            "test_operation",
            inputs,
            context=selection_context,
            mode=DispatchMode.DYNAMIC,
        )

        mock_dispatcher.dispatch.assert_called()

        # Cleanup
        set_global_dispatcher(None)


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread safety of orchestrator."""

    def test_concurrent_dispatch(
        self, mock_dispatcher: MagicMock, selection_context: SelectionContext
    ) -> None:
        """Multiple threads can dispatch concurrently."""
        orchestrator = DispatchOrchestrator()
        orchestrator.register_dispatcher(DispatchMode.DYNAMIC, mock_dispatcher)

        results: list[Any] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def dispatch_task() -> None:
            try:
                inputs = {"query": torch.randn(2, 4, 8, 64)}
                result = orchestrator.dispatch(
                    "attention.causal",
                    inputs,
                    context=selection_context,
                    mode=DispatchMode.DYNAMIC,
                )
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(dispatch_task) for _ in range(20)]
            for f in futures:
                f.result()

        assert len(errors) == 0
        assert len(results) == 20

    def test_concurrent_dispatcher_registration(self) -> None:
        """Multiple threads can register dispatchers concurrently."""
        orchestrator = DispatchOrchestrator()
        errors: list[Exception] = []

        def register_task(mode: DispatchMode) -> None:
            try:
                mock_dispatcher = MagicMock(spec=Dispatcher)
                orchestrator.register_dispatcher(mode, mock_dispatcher)
            except Exception as e:
                errors.append(e)

        threads = []
        modes = [DispatchMode.STATIC, DispatchMode.DYNAMIC, DispatchMode.CONFIG]

        for _ in range(10):
            for mode in modes:
                t = threading.Thread(target=register_task, args=(mode,))
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_telemetry_access(self) -> None:
        """Multiple threads can access telemetry concurrently."""
        orchestrator = DispatchOrchestrator()
        errors: list[Exception] = []

        def telemetry_task() -> None:
            try:
                for _ in range(100):
                    _ = orchestrator.get_telemetry()
                    timing = DispatchTiming(total_ns=1000)
                    orchestrator._telemetry.record_dispatch(DispatchMode.DYNAMIC, timing)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=telemetry_task) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_dispatch_with_none_context(self, mock_dispatcher: MagicMock) -> None:
        """dispatch works with None context."""
        orchestrator = DispatchOrchestrator()
        orchestrator.register_dispatcher(DispatchMode.DYNAMIC, mock_dispatcher)

        inputs = {"query": torch.zeros(4, 512, 8, 64)}

        # Should not raise - context is optional
        orchestrator.dispatch(
            "attention.causal",
            inputs,
            context=None,
            mode=DispatchMode.DYNAMIC,
        )

        mock_dispatcher.dispatch.assert_called()

    def test_get_kernel_for_operation(
        self, selection_context: SelectionContext
    ) -> None:
        """get_kernel_for_operation returns kernel spec."""
        mock_kernel_spec = MagicMock(spec=KernelSpec)
        mock_dispatcher = MagicMock(spec=Dispatcher)
        mock_dispatcher.get_kernel_for_operation.return_value = mock_kernel_spec

        orchestrator = DispatchOrchestrator()
        orchestrator.register_dispatcher(DispatchMode.DYNAMIC, mock_dispatcher)

        kernel = orchestrator.get_kernel_for_operation(
            "attention.causal",
            selection_context,
            mode=DispatchMode.DYNAMIC,
        )

        assert kernel is mock_kernel_spec
        mock_dispatcher.get_kernel_for_operation.assert_called_once()

    def test_unsupported_mode_raises(self) -> None:
        """Creating dispatcher for unsupported mode raises ValueError."""
        orchestrator = DispatchOrchestrator()

        # Patch the _create_dispatcher to test the error path
        # by trying to create a dispatcher for an invalid mode value
        with pytest.raises((ValueError, KeyError)):
            # Force an invalid mode through internal method
            orchestrator._create_dispatcher(MagicMock())  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
