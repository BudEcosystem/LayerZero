"""
Core types for the kernel dispatch system.

Defines:
- DispatchMode: Enum of dispatch strategies
- DispatchResult: Result of kernel execution
- DispatchError: Base exception for dispatch failures
- DispatchConfig: Configuration dataclass
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.models.execution_plan import ExecutionPlan


class DispatchMode(Enum):
    """Kernel dispatch mode.

    Each mode has different characteristics:
    - STATIC: Zero overhead, compile-time kernel resolution via enum dispatch
    - DYNAMIC: Runtime selection, ~100-500ns overhead, full flexibility
    - HOT_RELOAD: Config file watching, instant updates, ~1-10ms reload
    - CONFIG: YAML-driven, ops-controlled, ~100ns lookup overhead
    - AUTO: Automatically choose best mode based on context
    """
    STATIC = auto()
    DYNAMIC = auto()
    HOT_RELOAD = auto()
    CONFIG = auto()
    AUTO = auto()


class DispatchPhase(Enum):
    """Phase of dispatch execution for telemetry."""
    SELECTION = auto()
    PRE_TRANSFORM = auto()
    EXECUTION = auto()
    POST_TRANSFORM = auto()
    FALLBACK = auto()


@dataclass(frozen=True, slots=True)
class DispatchTiming:
    """Timing information for dispatch execution.

    All times are in nanoseconds for precision.
    """
    selection_ns: int = 0
    pre_transform_ns: int = 0
    execution_ns: int = 0
    post_transform_ns: int = 0
    total_ns: int = 0

    @property
    def selection_us(self) -> float:
        """Selection time in microseconds."""
        return self.selection_ns / 1000

    @property
    def execution_us(self) -> float:
        """Execution time in microseconds."""
        return self.execution_ns / 1000

    @property
    def total_us(self) -> float:
        """Total time in microseconds."""
        return self.total_ns / 1000

    @property
    def overhead_ns(self) -> int:
        """Dispatch overhead (everything except execution)."""
        return self.total_ns - self.execution_ns


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """Result of kernel dispatch and execution.

    Encapsulates:
    - Output tensor from kernel execution
    - Selected kernel information
    - Timing metrics
    - Debug information
    """
    output: "torch.Tensor"
    kernel_id: str
    kernel_spec: "KernelSpec"
    timing: DispatchTiming
    mode: DispatchMode
    cached: bool = False
    fallback_used: bool = False
    fallback_reason: Optional[str] = None

    @property
    def overhead_us(self) -> float:
        """Dispatch overhead in microseconds."""
        return self.timing.overhead_ns / 1000


@dataclass(slots=True)
class DispatchConfig:
    """Configuration for the dispatch system.

    Controls behavior of selection, caching, fallback, and monitoring.

    Thread Safety:
        This class is designed to be thread-safe after construction.
        The static_kernel_map is stored as an immutable MappingProxyType.
    """
    # Dispatch mode
    mode: DispatchMode = DispatchMode.DYNAMIC

    # Cache settings
    enable_cache: bool = True
    cache_size: int = 10000
    cache_ttl_seconds: float = 3600.0

    # Fallback settings
    enable_fallback: bool = True
    max_fallback_attempts: int = 3
    fallback_timeout_ms: float = 100.0

    # Hot-reload settings
    config_path: Optional[str] = None
    watch_interval_seconds: float = 1.0
    validate_on_reload: bool = True

    # Execution settings
    enable_transforms: bool = True
    enable_cuda_graphs: bool = False
    sync_after_execution: bool = False

    # Monitoring settings
    enable_telemetry: bool = True
    record_timing: bool = True
    log_fallbacks: bool = True

    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0

    # Static dispatch settings (for STATIC mode)
    # Stored as immutable Mapping for thread safety
    static_kernel_map: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Validate configuration and ensure immutability."""
        if self.cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        if self.max_fallback_attempts < 1:
            raise ValueError("max_fallback_attempts must be at least 1")
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")

        # Ensure static_kernel_map is immutable
        if isinstance(self.static_kernel_map, dict):
            # Convert mutable dict to immutable MappingProxyType
            object.__setattr__(
                self,
                'static_kernel_map',
                MappingProxyType(dict(self.static_kernel_map))
            )


# Exception hierarchy

class DispatchError(Exception):
    """Base exception for dispatch failures."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        kernel_id: Optional[str] = None,
        phase: Optional[DispatchPhase] = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.kernel_id = kernel_id
        self.phase = phase


class KernelExecutionError(DispatchError):
    """Error during kernel execution."""

    def __init__(
        self,
        message: str,
        operation: str,
        kernel_id: str,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message,
            operation=operation,
            kernel_id=kernel_id,
            phase=DispatchPhase.EXECUTION,
        )
        self.original_error = original_error


class TransformError(DispatchError):
    """Error during tensor transformation."""

    def __init__(
        self,
        message: str,
        transform_type: str,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, phase=DispatchPhase.PRE_TRANSFORM)
        self.transform_type = transform_type
        self.original_error = original_error


class FallbackChainExhaustedError(DispatchError):
    """All fallback options have been exhausted."""

    def __init__(
        self,
        operation: str,
        attempted_kernels: list[str],
        errors: list[Exception],
    ) -> None:
        message = (
            f"All fallback options exhausted for operation '{operation}'. "
            f"Attempted {len(attempted_kernels)} kernel(s): {attempted_kernels}"
        )
        super().__init__(message, operation=operation, phase=DispatchPhase.FALLBACK)
        self.attempted_kernels = attempted_kernels
        self.errors = errors


class CircuitOpenError(DispatchError):
    """Circuit breaker is open for a kernel."""

    def __init__(
        self,
        kernel_id: str,
        retry_after_seconds: float,
    ) -> None:
        message = (
            f"Circuit breaker open for kernel '{kernel_id}'. "
            f"Retry after {retry_after_seconds:.1f} seconds."
        )
        super().__init__(message, kernel_id=kernel_id)
        self.retry_after_seconds = retry_after_seconds


class ConfigurationError(DispatchError):
    """Invalid dispatch configuration."""
    pass


class HotReloadError(DispatchError):
    """Error during hot-reload."""

    def __init__(
        self,
        message: str,
        config_path: str,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.config_path = config_path
        self.original_error = original_error


# Type aliases for clarity
KernelId = str
OperationId = str
CacheKey = str
PolicyHash = str
