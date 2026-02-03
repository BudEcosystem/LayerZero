"""
Abstract protocols for the kernel dispatch system.

Defines the interfaces that all dispatchers must implement,
enabling polymorphic dispatch while maintaining type safety.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    import torch
    from layerzero.dispatch.types import DispatchMode, DispatchResult, DispatchConfig
    from layerzero.models.execution_plan import ExecutionPlan
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.models.selection_context import SelectionContext


@runtime_checkable
class KernelExecutor(Protocol):
    """Protocol for kernel execution.

    Any object that can execute a kernel must implement this interface.
    This allows for different execution strategies (direct call, CUDA graph,
    compiled kernel, etc.) while maintaining a uniform API.
    """

    def execute(
        self,
        kernel_spec: "KernelSpec",
        inputs: dict[str, "torch.Tensor"],
        **kwargs: Any,
    ) -> "torch.Tensor":
        """Execute a kernel with given inputs.

        Args:
            kernel_spec: Specification of the kernel to execute.
            inputs: Dictionary of named input tensors.
            **kwargs: Additional kernel-specific arguments.

        Returns:
            Output tensor from kernel execution.

        Raises:
            KernelExecutionError: If execution fails.
        """
        ...

    def supports_cuda_graph(self, kernel_spec: "KernelSpec") -> bool:
        """Check if kernel can be captured in CUDA graph.

        Args:
            kernel_spec: Kernel to check.

        Returns:
            True if kernel is CUDA graph safe.
        """
        ...


@runtime_checkable
class Dispatcher(Protocol):
    """Protocol for kernel dispatchers.

    All dispatch modes (static, dynamic, hot-reload, config-driven)
    implement this interface, allowing the orchestrator to use them
    interchangeably.
    """

    @property
    def mode(self) -> "DispatchMode":
        """Get the dispatch mode."""
        ...

    def dispatch(
        self,
        operation: str,
        inputs: dict[str, "torch.Tensor"],
        context: "SelectionContext | None" = None,
        **kwargs: Any,
    ) -> "DispatchResult":
        """Dispatch operation to appropriate kernel.

        Args:
            operation: Operation identifier (e.g., "attention.causal").
            inputs: Dictionary of named input tensors.
            context: Optional pre-built selection context.
            **kwargs: Additional operation-specific arguments.

        Returns:
            DispatchResult with output tensor and metadata.

        Raises:
            DispatchError: If dispatch fails.
        """
        ...

    def get_kernel_for_operation(
        self,
        operation: str,
        context: "SelectionContext",
    ) -> "KernelSpec":
        """Get the kernel that would be used for an operation.

        Useful for inspection and debugging without executing.

        Args:
            operation: Operation identifier.
            context: Selection context.

        Returns:
            KernelSpec that would be selected.

        Raises:
            NoKernelAvailableError: If no kernel matches.
        """
        ...


class BaseDispatcher(ABC):
    """Abstract base class for dispatchers.

    Provides common functionality and enforces the Dispatcher protocol.
    Subclasses implement specific dispatch strategies.
    """

    __slots__ = ("_config", "_executor", "_telemetry")

    def __init__(
        self,
        config: "DispatchConfig",
        executor: "KernelExecutor | None" = None,
    ) -> None:
        """Initialize dispatcher.

        Args:
            config: Dispatch configuration.
            executor: Kernel executor instance. If None, default is created.
        """
        self._config = config
        self._executor = executor
        self._telemetry: dict[str, Any] = {}

    @property
    @abstractmethod
    def mode(self) -> "DispatchMode":
        """Get dispatch mode for this dispatcher."""

    @abstractmethod
    def dispatch(
        self,
        operation: str,
        inputs: dict[str, "torch.Tensor"],
        context: "SelectionContext | None" = None,
        **kwargs: Any,
    ) -> "DispatchResult":
        """Execute dispatch. Must be implemented by subclasses."""

    @abstractmethod
    def get_kernel_for_operation(
        self,
        operation: str,
        context: "SelectionContext",
    ) -> "KernelSpec":
        """Get kernel for operation. Must be implemented by subclasses."""

    @property
    def config(self) -> "DispatchConfig":
        """Get configuration."""
        return self._config

    def update_config(self, config: "DispatchConfig") -> None:
        """Update configuration.

        Args:
            config: New configuration.
        """
        self._config = config

    def get_telemetry(self) -> dict[str, Any]:
        """Get telemetry data.

        Returns:
            Dict of telemetry metrics.
        """
        return self._telemetry.copy()

    def reset_telemetry(self) -> None:
        """Reset telemetry counters."""
        self._telemetry.clear()


class TransformProtocol(Protocol):
    """Protocol for tensor transformations.

    Transforms are applied before and after kernel execution
    to handle layout/dtype conversions.
    """

    def __call__(
        self,
        tensor: "torch.Tensor",
        **kwargs: Any,
    ) -> "torch.Tensor":
        """Apply transformation.

        Args:
            tensor: Input tensor.
            **kwargs: Transform-specific arguments.

        Returns:
            Transformed tensor.
        """
        ...

    @property
    def name(self) -> str:
        """Get transform name for logging."""
        ...

    @property
    def is_identity(self) -> bool:
        """Check if transform is identity (no-op)."""
        ...


class FallbackChain(Protocol):
    """Protocol for fallback chain management.

    Provides ordered list of fallback kernels when primary fails.
    """

    def get_fallbacks(
        self,
        operation: str,
        failed_kernel: str,
        context: "SelectionContext",
    ) -> list["KernelSpec"]:
        """Get ordered list of fallback kernels.

        Args:
            operation: Operation that failed.
            failed_kernel: Kernel that failed.
            context: Selection context.

        Returns:
            List of fallback kernels in priority order.
        """
        ...

    def record_failure(
        self,
        kernel_id: str,
        error: Exception,
    ) -> None:
        """Record kernel failure for future fallback decisions.

        Args:
            kernel_id: Kernel that failed.
            error: The error that occurred.
        """
        ...
