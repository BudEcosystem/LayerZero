"""
Unified Kernel Dispatch Orchestrator

Coordinates all dispatch modes (static, dynamic, hot-reload, config-driven)
and provides a single entry point for kernel dispatch.

The orchestrator:
- Manages multiple dispatcher instances
- Automatically selects appropriate dispatch mode based on context
- Provides unified telemetry and monitoring
- Handles mode switching and fallback between modes
- Integrates with the existing SelectionEngine

Usage:
    from layerzero.dispatch import get_global_dispatcher, DispatchMode

    # Get global dispatcher
    dispatcher = get_global_dispatcher()

    # Dispatch with automatic mode selection
    result = dispatcher.dispatch("attention.causal", inputs, is_causal=True)

    # Dispatch with explicit mode
    result = dispatcher.dispatch(
        "attention.causal", inputs, mode=DispatchMode.STATIC, is_causal=True
    )
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.models.selection_context import SelectionContext
    from layerzero.registry.kernel_registry import KernelRegistry
    from layerzero.registry.backend_registry import BackendRegistry

from layerzero.dispatch.types import (
    DispatchConfig,
    DispatchError,
    DispatchMode,
    DispatchPhase,
    DispatchResult,
    DispatchTiming,
    FallbackChainExhaustedError,
    KernelExecutionError,
)
from layerzero.dispatch.protocols import BaseDispatcher, Dispatcher
from layerzero.dispatch.executor import KernelExecutorImpl

logger = logging.getLogger(__name__)


# Global orchestrator instance
_global_orchestrator: "DispatchOrchestrator | None" = None
_global_lock = threading.Lock()


@dataclass(slots=True)
class OrchestratorTelemetry:
    """Telemetry data for the orchestrator.

    Tracks per-mode statistics and overall dispatch metrics.
    """
    total_dispatches: int = 0
    dispatches_by_mode: dict[DispatchMode, int] = field(default_factory=dict)
    fallbacks_between_modes: int = 0
    total_errors: int = 0
    errors_by_mode: dict[DispatchMode, int] = field(default_factory=dict)
    total_execution_time_ns: int = 0
    total_selection_time_ns: int = 0

    def record_dispatch(
        self,
        mode: DispatchMode,
        timing: DispatchTiming,
        error: bool = False,
    ) -> None:
        """Record a dispatch event."""
        self.total_dispatches += 1
        self.dispatches_by_mode[mode] = self.dispatches_by_mode.get(mode, 0) + 1
        self.total_execution_time_ns += timing.execution_ns
        self.total_selection_time_ns += timing.selection_ns
        if error:
            self.total_errors += 1
            self.errors_by_mode[mode] = self.errors_by_mode.get(mode, 0) + 1

    def record_mode_fallback(self) -> None:
        """Record a fallback from one mode to another."""
        self.fallbacks_between_modes += 1

    @property
    def avg_execution_time_us(self) -> float:
        """Average execution time in microseconds."""
        if self.total_dispatches == 0:
            return 0.0
        return (self.total_execution_time_ns / self.total_dispatches) / 1000

    @property
    def avg_selection_time_us(self) -> float:
        """Average selection time in microseconds."""
        if self.total_dispatches == 0:
            return 0.0
        return (self.total_selection_time_ns / self.total_dispatches) / 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_dispatches": self.total_dispatches,
            "dispatches_by_mode": {m.name: c for m, c in self.dispatches_by_mode.items()},
            "fallbacks_between_modes": self.fallbacks_between_modes,
            "total_errors": self.total_errors,
            "errors_by_mode": {m.name: c for m, c in self.errors_by_mode.items()},
            "avg_execution_time_us": self.avg_execution_time_us,
            "avg_selection_time_us": self.avg_selection_time_us,
        }


class DispatchOrchestrator:
    """Unified dispatch orchestrator.

    Manages multiple dispatch modes and provides a single entry point
    for kernel dispatch operations.

    The orchestrator supports:
    - Automatic mode selection based on configuration and context
    - Explicit mode override per dispatch call
    - Fallback between modes when one fails
    - Unified telemetry across all modes
    - Thread-safe operation

    Attributes:
        config: Current dispatch configuration.
        default_mode: Default dispatch mode when AUTO is specified.
    """

    __slots__ = (
        "_config",
        "_dispatchers",
        "_executor",
        "_telemetry",
        "_lock",
        "_default_mode",
        "_mode_fallback_chain",
        "_selection_engine",
        "_kernel_registry",
        "_backend_registry",
    )

    def __init__(
        self,
        config: DispatchConfig | None = None,
        kernel_registry: "KernelRegistry | None" = None,
        backend_registry: "BackendRegistry | None" = None,
        selection_engine: Any = None,
        default_mode: DispatchMode = DispatchMode.DYNAMIC,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config: Dispatch configuration. If None, default config is used.
            kernel_registry: Kernel registry for dynamic/static dispatch.
            backend_registry: Backend registry for health tracking.
            selection_engine: Selection engine for dynamic dispatch.
            default_mode: Default mode when AUTO is specified.
        """
        self._config = config or DispatchConfig()
        self._dispatchers: dict[DispatchMode, Dispatcher] = {}
        self._executor = KernelExecutorImpl(backend_registry)
        self._telemetry = OrchestratorTelemetry()
        self._lock = threading.RLock()
        self._default_mode = default_mode
        self._selection_engine = selection_engine
        self._kernel_registry = kernel_registry
        self._backend_registry = backend_registry

        # Define fallback chain: if one mode fails, try next
        self._mode_fallback_chain: dict[DispatchMode, list[DispatchMode]] = {
            DispatchMode.STATIC: [DispatchMode.DYNAMIC],
            DispatchMode.DYNAMIC: [DispatchMode.CONFIG],
            DispatchMode.HOT_RELOAD: [DispatchMode.DYNAMIC],
            DispatchMode.CONFIG: [DispatchMode.DYNAMIC],
            DispatchMode.AUTO: [DispatchMode.DYNAMIC, DispatchMode.STATIC],
        }

        # Initialize dispatchers lazily
        self._initialize_default_dispatcher()

    def _initialize_default_dispatcher(self) -> None:
        """Initialize the default dispatcher."""
        mode = self._config.mode if self._config.mode != DispatchMode.AUTO else self._default_mode
        self._get_or_create_dispatcher(mode)

    def _get_or_create_dispatcher(self, mode: DispatchMode) -> Dispatcher:
        """Get or create a dispatcher for the given mode.

        Args:
            mode: Dispatch mode.

        Returns:
            Dispatcher instance for the mode.
        """
        if mode == DispatchMode.AUTO:
            mode = self._default_mode

        with self._lock:
            if mode in self._dispatchers:
                return self._dispatchers[mode]

            dispatcher = self._create_dispatcher(mode)
            self._dispatchers[mode] = dispatcher
            return dispatcher

    def _create_dispatcher(self, mode: DispatchMode) -> Dispatcher:
        """Create a new dispatcher for the given mode.

        Args:
            mode: Dispatch mode.

        Returns:
            New dispatcher instance.
        """
        if mode == DispatchMode.STATIC:
            from layerzero.dispatch.static import (
                StaticDispatcher,
                StaticKernelRegistry,
                create_static_dispatcher_from_config,
            )

            # Use static_kernel_map from config if available
            if self._config.static_kernel_map and self._kernel_registry:
                return create_static_dispatcher_from_config(
                    self._config,
                    self._kernel_registry,
                    self._executor,
                )

            # Create empty static dispatcher (kernels registered via global registry)
            registry = StaticKernelRegistry()
            return StaticDispatcher(
                registry=registry,
                config=self._config,
                executor=self._executor,
            )

        elif mode == DispatchMode.DYNAMIC:
            from layerzero.dispatch.dynamic import (
                DynamicDispatcher,
                create_dynamic_dispatcher,
            )

            # Create dynamic dispatcher with selection engine
            if self._kernel_registry and self._backend_registry:
                return create_dynamic_dispatcher(
                    kernel_registry=self._kernel_registry,
                    backend_registry=self._backend_registry,
                    config=self._config,
                )

            # Fallback: try to get global registries
            try:
                from layerzero.registry.kernel_registry import get_global_kernel_registry
                from layerzero.registry.backend_registry import get_global_backend_registry

                return create_dynamic_dispatcher(
                    kernel_registry=get_global_kernel_registry(),
                    backend_registry=get_global_backend_registry(),
                    config=self._config,
                )
            except (ImportError, AttributeError):
                # Create minimal dynamic dispatcher
                from layerzero.registry.kernel_registry import KernelRegistry
                from layerzero.registry.backend_registry import BackendRegistry

                return create_dynamic_dispatcher(
                    kernel_registry=KernelRegistry(),
                    backend_registry=BackendRegistry(),
                    config=self._config,
                )

        elif mode == DispatchMode.HOT_RELOAD:
            from layerzero.dispatch.hot_reload import (
                HotReloadDispatcher,
                create_hot_reload_dispatcher,
            )

            if self._config.config_path:
                return create_hot_reload_dispatcher(
                    config_path=self._config.config_path,
                    validate_on_reload=self._config.validate_on_reload,
                    watch_interval_seconds=self._config.watch_interval_seconds,
                    start_watching=True,
                    selection_engine=self._selection_engine,
                    kernel_registry=self._kernel_registry,
                )

            # No config path - create without file watching
            return HotReloadDispatcher(
                config=self._config,
                executor=self._executor,
            )

        elif mode == DispatchMode.CONFIG:
            from layerzero.dispatch.config_dispatch import (
                ConfigDrivenDispatcher,
                create_config_dispatcher,
            )

            if self._config.config_path:
                return create_config_dispatcher(
                    config_path=self._config.config_path,
                    kernel_registry=self._kernel_registry,
                    executor=self._executor,
                )

            # Create with empty config
            return ConfigDrivenDispatcher(
                config=self._config,
                executor=self._executor,
            )

        else:
            raise ValueError(f"Unsupported dispatch mode: {mode}")

    @property
    def config(self) -> DispatchConfig:
        """Get current configuration."""
        return self._config

    @property
    def default_mode(self) -> DispatchMode:
        """Get default dispatch mode."""
        return self._default_mode

    def set_default_mode(self, mode: DispatchMode) -> None:
        """Set the default dispatch mode.

        Args:
            mode: New default mode.
        """
        if mode == DispatchMode.AUTO:
            raise ValueError("Cannot set AUTO as default mode")
        self._default_mode = mode

    def dispatch(
        self,
        operation: str,
        inputs: dict[str, "torch.Tensor"],
        context: "SelectionContext | None" = None,
        mode: DispatchMode | None = None,
        **kwargs: Any,
    ) -> DispatchResult:
        """Dispatch an operation to the appropriate kernel.

        This is the main entry point for kernel dispatch.

        Args:
            operation: Operation identifier (e.g., "attention.causal").
            inputs: Dictionary of named input tensors.
            context: Optional pre-built selection context.
            mode: Optional mode override. If None, uses config.mode.
            **kwargs: Additional operation-specific arguments.

        Returns:
            DispatchResult with output tensor and metadata.

        Raises:
            DispatchError: If dispatch fails in all attempted modes.
        """
        start_time = time.perf_counter_ns()

        # Determine which mode to use
        effective_mode = mode or self._config.mode
        if effective_mode == DispatchMode.AUTO:
            effective_mode = self._select_best_mode(operation, context)

        # Try dispatch with fallback
        errors: list[tuple[DispatchMode, Exception]] = []
        modes_to_try = [effective_mode] + self._mode_fallback_chain.get(effective_mode, [])

        for attempt_mode in modes_to_try:
            try:
                dispatcher = self._get_or_create_dispatcher(attempt_mode)
                result = dispatcher.dispatch(operation, inputs, context, **kwargs)

                # Record telemetry
                self._telemetry.record_dispatch(attempt_mode, result.timing)

                # Record mode fallback if we tried multiple modes
                if len(errors) > 0:
                    self._telemetry.record_mode_fallback()

                return result

            except Exception as e:
                errors.append((attempt_mode, e))
                logger.warning(
                    f"Dispatch failed in mode {attempt_mode.name}: {e}. "
                    f"Trying fallback..."
                )
                continue

        # All modes failed
        error_msg = "; ".join(f"{m.name}: {e}" for m, e in errors)
        total_time = time.perf_counter_ns() - start_time

        # Record error telemetry
        self._telemetry.record_dispatch(
            effective_mode,
            DispatchTiming(total_ns=total_time),
            error=True,
        )

        raise DispatchError(
            f"Dispatch failed for operation '{operation}' in all modes: {error_msg}",
            operation=operation,
            phase=DispatchPhase.EXECUTION,
        )

    def _select_best_mode(
        self,
        operation: str,
        context: "SelectionContext | None",
    ) -> DispatchMode:
        """Select the best dispatch mode for the given operation.

        Uses heuristics to choose the most appropriate mode.

        Args:
            operation: Operation identifier.
            context: Selection context (if available).

        Returns:
            Selected dispatch mode.
        """
        # If static kernel map has this operation, use static
        if operation in self._config.static_kernel_map:
            return DispatchMode.STATIC

        # If hot-reload is enabled and config path exists, use hot-reload
        if self._config.config_path and self._config.mode == DispatchMode.HOT_RELOAD:
            return DispatchMode.HOT_RELOAD

        # Default to dynamic for flexibility
        return self._default_mode

    def get_kernel_for_operation(
        self,
        operation: str,
        context: "SelectionContext",
        mode: DispatchMode | None = None,
    ) -> "KernelSpec":
        """Get the kernel that would be used for an operation.

        Useful for inspection and debugging.

        Args:
            operation: Operation identifier.
            context: Selection context.
            mode: Optional mode override.

        Returns:
            KernelSpec that would be selected.
        """
        effective_mode = mode or self._config.mode
        if effective_mode == DispatchMode.AUTO:
            effective_mode = self._select_best_mode(operation, context)

        dispatcher = self._get_or_create_dispatcher(effective_mode)
        return dispatcher.get_kernel_for_operation(operation, context)

    def register_dispatcher(self, mode: DispatchMode, dispatcher: Dispatcher) -> None:
        """Register a custom dispatcher for a mode.

        Args:
            mode: Dispatch mode.
            dispatcher: Dispatcher instance to register.
        """
        with self._lock:
            self._dispatchers[mode] = dispatcher

    def get_dispatcher(self, mode: DispatchMode) -> Dispatcher | None:
        """Get the dispatcher for a mode.

        Args:
            mode: Dispatch mode.

        Returns:
            Dispatcher if registered, None otherwise.
        """
        return self._dispatchers.get(mode)

    def reload_config(self, config: DispatchConfig) -> None:
        """Reload the orchestrator configuration.

        This updates the config and reinitializes dispatchers as needed.

        Args:
            config: New configuration.
        """
        with self._lock:
            old_config = self._config
            self._config = config

            # Reinitialize dispatchers that depend on config
            for mode in list(self._dispatchers.keys()):
                if hasattr(self._dispatchers[mode], "update_config"):
                    self._dispatchers[mode].update_config(config)

    def get_telemetry(self) -> dict[str, Any]:
        """Get orchestrator telemetry.

        Returns:
            Dictionary of telemetry metrics.
        """
        result = self._telemetry.to_dict()

        # Add per-dispatcher telemetry
        result["dispatchers"] = {}
        for mode, dispatcher in self._dispatchers.items():
            if hasattr(dispatcher, "get_telemetry"):
                result["dispatchers"][mode.name] = dispatcher.get_telemetry()

        return result

    def reset_telemetry(self) -> None:
        """Reset all telemetry counters."""
        self._telemetry = OrchestratorTelemetry()

        for dispatcher in self._dispatchers.values():
            if hasattr(dispatcher, "reset_telemetry"):
                dispatcher.reset_telemetry()

    def shutdown(self) -> None:
        """Shutdown the orchestrator and all dispatchers.

        Stops any background tasks (file watching, etc.).
        """
        with self._lock:
            for mode, dispatcher in self._dispatchers.items():
                if hasattr(dispatcher, "stop_watching"):
                    dispatcher.stop_watching()
                if hasattr(dispatcher, "shutdown"):
                    dispatcher.shutdown()

            self._dispatchers.clear()


def get_global_dispatcher() -> DispatchOrchestrator:
    """Get the global dispatch orchestrator.

    Creates one if it doesn't exist.

    Returns:
        Global DispatchOrchestrator instance.
    """
    global _global_orchestrator

    if _global_orchestrator is None:
        with _global_lock:
            if _global_orchestrator is None:
                _global_orchestrator = DispatchOrchestrator()

    return _global_orchestrator


def set_global_dispatcher(orchestrator: DispatchOrchestrator | None) -> None:
    """Set the global dispatch orchestrator.

    Args:
        orchestrator: Orchestrator instance, or None to clear.
    """
    global _global_orchestrator

    with _global_lock:
        if _global_orchestrator is not None:
            _global_orchestrator.shutdown()
        _global_orchestrator = orchestrator


def create_orchestrator(
    config: DispatchConfig | None = None,
    kernel_registry: "KernelRegistry | None" = None,
    backend_registry: "BackendRegistry | None" = None,
    selection_engine: Any = None,
    default_mode: DispatchMode = DispatchMode.DYNAMIC,
    set_as_global: bool = False,
) -> DispatchOrchestrator:
    """Create a new dispatch orchestrator.

    Factory function for creating configured orchestrators.

    Args:
        config: Dispatch configuration.
        kernel_registry: Kernel registry.
        backend_registry: Backend registry.
        selection_engine: Selection engine.
        default_mode: Default dispatch mode.
        set_as_global: Whether to set as global orchestrator.

    Returns:
        New DispatchOrchestrator instance.
    """
    orchestrator = DispatchOrchestrator(
        config=config,
        kernel_registry=kernel_registry,
        backend_registry=backend_registry,
        selection_engine=selection_engine,
        default_mode=default_mode,
    )

    if set_as_global:
        set_global_dispatcher(orchestrator)

    return orchestrator


# Convenience function for direct dispatch
def dispatch(
    operation: str,
    inputs: dict[str, "torch.Tensor"],
    context: "SelectionContext | None" = None,
    mode: DispatchMode | None = None,
    **kwargs: Any,
) -> DispatchResult:
    """Dispatch an operation using the global orchestrator.

    Convenience function that uses the global orchestrator.

    Args:
        operation: Operation identifier.
        inputs: Input tensors.
        context: Selection context.
        mode: Dispatch mode override.
        **kwargs: Additional arguments.

    Returns:
        DispatchResult with output and metadata.
    """
    return get_global_dispatcher().dispatch(operation, inputs, context, mode, **kwargs)
