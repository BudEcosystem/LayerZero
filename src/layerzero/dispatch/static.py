"""
LayerZero Static Kernel Dispatch System

Static dispatch achieves near-zero overhead through:
1. Compile-time/import-time kernel resolution via enum-like pattern
2. Direct function dispatch without virtual call overhead
3. Match statement dispatch (Python 3.10+) for O(1) dispatch

This module provides:
- StaticKernelEntry: Frozen dataclass for immutable kernel entries
- StaticKernelRegistry: Import-time kernel mapping with O(1) lookup
- StaticDispatcher: BaseDispatcher implementation for static dispatch mode

Thread Safety:
- All data structures are immutable after initialization
- No shared mutable state in dispatch path
- Uses __slots__ for memory efficiency

Performance:
- O(1) dict-based kernel lookup
- Match statement for type-safe dispatch
- Inline caching via frozen dataclasses
- Nanosecond precision timing with time.perf_counter_ns()
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Final, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.models.selection_context import SelectionContext

from layerzero.dispatch.types import (
    DispatchConfig,
    DispatchError,
    DispatchMode,
    DispatchPhase,
    DispatchResult,
    DispatchTiming,
    KernelExecutionError,
)
from layerzero.dispatch.protocols import BaseDispatcher, KernelExecutor
from layerzero.dispatch.executor import KernelExecutorImpl

logger = logging.getLogger(__name__)


# ============================================================================
# Error Types for Static Dispatch
# ============================================================================


class StaticDispatchError(DispatchError):
    """Error specific to static dispatch mode."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        kernel_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            operation=operation,
            kernel_id=kernel_id,
            phase=DispatchPhase.SELECTION,
        )


class KernelNotRegisteredError(StaticDispatchError):
    """Kernel is not registered in the static registry."""

    def __init__(self, operation: str) -> None:
        super().__init__(
            f"No kernel registered for operation '{operation}' in static registry",
            operation=operation,
        )
        self.operation = operation


class InvalidKernelEntryError(StaticDispatchError):
    """Kernel entry is invalid or corrupted."""

    def __init__(self, kernel_id: str, reason: str) -> None:
        super().__init__(
            f"Invalid kernel entry for '{kernel_id}': {reason}",
            kernel_id=kernel_id,
        )
        self.reason = reason


class RegistryFrozenError(StaticDispatchError):
    """Attempt to modify a frozen registry."""

    def __init__(self) -> None:
        super().__init__(
            "Cannot modify static registry after it has been frozen"
        )


# ============================================================================
# Operation Types for Match Dispatch
# ============================================================================


class OperationType(Enum):
    """Enumeration of operation types for match dispatch.

    Using enum for type-safe, O(1) operation classification.
    Match statement can dispatch on these values efficiently.
    """

    # Attention operations
    ATTENTION_CAUSAL = auto()
    ATTENTION_FULL = auto()
    ATTENTION_SLIDING_WINDOW = auto()
    ATTENTION_CROSS = auto()
    ATTENTION_PREFILL = auto()
    ATTENTION_DECODE = auto()

    # Normalization operations
    NORM_RMS = auto()
    NORM_LAYER = auto()
    NORM_GROUP = auto()

    # Position embedding operations
    ROPE = auto()
    ROPE_INTERLEAVED = auto()
    ALIBI = auto()

    # Activation operations
    SWIGLU = auto()
    GELU = auto()
    SILU = auto()

    # Loss operations
    CROSS_ENTROPY = auto()
    FUSED_CROSS_ENTROPY = auto()

    # Matmul operations
    MATMUL = auto()
    MATMUL_FP8 = auto()

    # Tokenization operations
    TOKENIZE = auto()
    DETOKENIZE = auto()

    # Sampling operations
    SAMPLE_TOP_K = auto()
    SAMPLE_TOP_P = auto()
    SAMPLE_GREEDY = auto()

    # Unknown/custom operations
    UNKNOWN = auto()


# Operation string to enum mapping for O(1) lookup
OPERATION_TYPE_MAP: Final[dict[str, OperationType]] = {
    # Attention
    "attention.causal": OperationType.ATTENTION_CAUSAL,
    "attention.full": OperationType.ATTENTION_FULL,
    "attention.sliding_window": OperationType.ATTENTION_SLIDING_WINDOW,
    "attention.cross": OperationType.ATTENTION_CROSS,
    "attention.prefill": OperationType.ATTENTION_PREFILL,
    "attention.decode": OperationType.ATTENTION_DECODE,
    # Normalization
    "norm.rms": OperationType.NORM_RMS,
    "rms_norm": OperationType.NORM_RMS,
    "norm.layer": OperationType.NORM_LAYER,
    "layer_norm": OperationType.NORM_LAYER,
    "norm.group": OperationType.NORM_GROUP,
    "group_norm": OperationType.NORM_GROUP,
    # Position embeddings
    "rope": OperationType.ROPE,
    "rope.standard": OperationType.ROPE,
    "rope.interleaved": OperationType.ROPE_INTERLEAVED,
    "alibi": OperationType.ALIBI,
    # Activations
    "swiglu": OperationType.SWIGLU,
    "activation.swiglu": OperationType.SWIGLU,
    "gelu": OperationType.GELU,
    "activation.gelu": OperationType.GELU,
    "silu": OperationType.SILU,
    "activation.silu": OperationType.SILU,
    # Loss
    "cross_entropy": OperationType.CROSS_ENTROPY,
    "loss.cross_entropy": OperationType.CROSS_ENTROPY,
    "fused_cross_entropy": OperationType.FUSED_CROSS_ENTROPY,
    "loss.fused_cross_entropy": OperationType.FUSED_CROSS_ENTROPY,
    # Matmul
    "matmul": OperationType.MATMUL,
    "matmul.fp8": OperationType.MATMUL_FP8,
    # Tokenization
    "tokenize": OperationType.TOKENIZE,
    "tokenization.encode": OperationType.TOKENIZE,
    "detokenize": OperationType.DETOKENIZE,
    "tokenization.decode": OperationType.DETOKENIZE,
    # Sampling
    "sample.top_k": OperationType.SAMPLE_TOP_K,
    "sample.top_p": OperationType.SAMPLE_TOP_P,
    "sample.greedy": OperationType.SAMPLE_GREEDY,
}


def get_operation_type(operation: str) -> OperationType:
    """Get operation type enum from operation string.

    O(1) dict lookup with fallback to UNKNOWN.

    Args:
        operation: Operation string identifier.

    Returns:
        OperationType enum value.
    """
    return OPERATION_TYPE_MAP.get(operation, OperationType.UNKNOWN)


# ============================================================================
# Static Kernel Entry (Frozen Dataclass)
# ============================================================================


@dataclass(frozen=True, slots=True)
class StaticKernelEntry:
    """Immutable kernel entry for static dispatch.

    Frozen dataclass provides:
    - Hashability for use in sets/dicts
    - Thread safety through immutability
    - Memory efficiency via __slots__
    - Compile-time error detection

    Attributes:
        kernel_id: Unique kernel identifier.
        operation: Operation this kernel implements.
        operation_type: Pre-computed operation type enum.
        kernel_spec: Full kernel specification.
        priority: Selection priority (higher = preferred).
        is_default: Whether this is the default kernel for operation.
    """

    kernel_id: str
    operation: str
    operation_type: OperationType
    kernel_spec: "KernelSpec"
    priority: int = 50
    is_default: bool = False

    def __post_init__(self) -> None:
        """Validate entry after initialization."""
        if not self.kernel_id:
            raise ValueError("kernel_id cannot be empty")
        if not self.operation:
            raise ValueError("operation cannot be empty")

    @classmethod
    def from_kernel_spec(
        cls,
        spec: "KernelSpec",
        *,
        is_default: bool = False,
    ) -> "StaticKernelEntry":
        """Create entry from KernelSpec.

        Args:
            spec: Kernel specification.
            is_default: Whether this is the default kernel.

        Returns:
            New StaticKernelEntry instance.
        """
        return cls(
            kernel_id=spec.kernel_id,
            operation=spec.operation,
            operation_type=get_operation_type(spec.operation),
            kernel_spec=spec,
            priority=spec.priority,
            is_default=is_default,
        )


# ============================================================================
# Static Kernel Registry
# ============================================================================


class StaticKernelRegistry:
    """Import-time kernel mapping with O(1) lookup.

    Designed for static dispatch where kernel selection is determined
    at initialization time, not runtime.

    Features:
    - O(1) lookup by operation or kernel_id
    - Immutable after freeze() is called
    - Pre-computed default kernels per operation
    - Memory efficient via __slots__

    Thread Safety:
    - Mutable during setup phase (before freeze)
    - Immutable after freeze (thread-safe reads)

    Usage:
        registry = StaticKernelRegistry()
        registry.register(entry1)
        registry.register(entry2)
        registry.freeze()  # No more modifications allowed

        kernel = registry.get_kernel_for_operation("attention.causal")
    """

    __slots__ = (
        "_by_operation",
        "_by_kernel_id",
        "_defaults",
        "_frozen",
        "_operation_count",
        "_kernel_count",
    )

    def __init__(self) -> None:
        """Initialize empty registry."""
        # O(1) lookup tables
        self._by_operation: dict[str, list[StaticKernelEntry]] = {}
        self._by_kernel_id: dict[str, StaticKernelEntry] = {}
        # Pre-computed defaults for each operation
        self._defaults: dict[str, StaticKernelEntry] = {}
        # Frozen flag
        self._frozen: bool = False
        # Counts for telemetry
        self._operation_count: int = 0
        self._kernel_count: int = 0

    @property
    def is_frozen(self) -> bool:
        """Check if registry is frozen."""
        return self._frozen

    @property
    def operation_count(self) -> int:
        """Number of registered operations."""
        return self._operation_count

    @property
    def kernel_count(self) -> int:
        """Number of registered kernels."""
        return self._kernel_count

    def register(self, entry: StaticKernelEntry) -> None:
        """Register a kernel entry.

        Args:
            entry: Kernel entry to register.

        Raises:
            RegistryFrozenError: If registry is frozen.
            ValueError: If kernel_id is already registered.
        """
        if self._frozen:
            raise RegistryFrozenError()

        if entry.kernel_id in self._by_kernel_id:
            raise ValueError(
                f"Kernel '{entry.kernel_id}' is already registered"
            )

        # Add to kernel_id index
        self._by_kernel_id[entry.kernel_id] = entry
        self._kernel_count += 1

        # Add to operation index
        if entry.operation not in self._by_operation:
            self._by_operation[entry.operation] = []
            self._operation_count += 1
        self._by_operation[entry.operation].append(entry)

        # Update default if this is marked as default or first for operation
        if entry.is_default or entry.operation not in self._defaults:
            self._defaults[entry.operation] = entry
        # Or if this has higher priority than current default
        elif entry.priority > self._defaults[entry.operation].priority:
            self._defaults[entry.operation] = entry

    def register_from_spec(
        self,
        spec: "KernelSpec",
        *,
        is_default: bool = False,
    ) -> StaticKernelEntry:
        """Register kernel from KernelSpec.

        Convenience method that creates entry and registers it.

        Args:
            spec: Kernel specification.
            is_default: Whether this is the default kernel.

        Returns:
            Created StaticKernelEntry.

        Raises:
            RegistryFrozenError: If registry is frozen.
            ValueError: If kernel_id is already registered.
        """
        entry = StaticKernelEntry.from_kernel_spec(spec, is_default=is_default)
        self.register(entry)
        return entry

    def register_many(self, entries: list[StaticKernelEntry]) -> None:
        """Register multiple entries atomically.

        If any registration fails, no entries are registered.

        Args:
            entries: List of entries to register.

        Raises:
            RegistryFrozenError: If registry is frozen.
            ValueError: If any kernel_id is already registered or duplicated.
        """
        if self._frozen:
            raise RegistryFrozenError()

        # Check for existing duplicates
        for entry in entries:
            if entry.kernel_id in self._by_kernel_id:
                raise ValueError(
                    f"Kernel '{entry.kernel_id}' is already registered"
                )

        # Check for duplicates within batch
        kernel_ids = [e.kernel_id for e in entries]
        if len(kernel_ids) != len(set(kernel_ids)):
            raise ValueError("Duplicate kernel_id in batch")

        # All checks passed, register
        for entry in entries:
            self.register(entry)

    def freeze(self) -> None:
        """Freeze registry to prevent further modifications.

        After freezing:
        - No more registrations allowed
        - Registry becomes thread-safe for reads
        - Default kernels are finalized
        """
        if self._frozen:
            return

        # Finalize defaults by selecting highest priority for each operation
        for operation, entries in self._by_operation.items():
            if entries:
                # Sort by priority descending, take first
                best = max(entries, key=lambda e: (e.is_default, e.priority))
                self._defaults[operation] = best

        self._frozen = True
        logger.info(
            f"Static kernel registry frozen with {self._kernel_count} kernels "
            f"for {self._operation_count} operations"
        )

    def get_by_kernel_id(self, kernel_id: str) -> StaticKernelEntry | None:
        """Get entry by kernel ID.

        O(1) dict lookup.

        Args:
            kernel_id: Kernel identifier.

        Returns:
            StaticKernelEntry if found, None otherwise.
        """
        return self._by_kernel_id.get(kernel_id)

    def get_by_operation(self, operation: str) -> list[StaticKernelEntry]:
        """Get all entries for an operation.

        O(1) dict lookup.

        Args:
            operation: Operation identifier.

        Returns:
            List of entries (empty if none).
        """
        return self._by_operation.get(operation, [])

    def get_default(self, operation: str) -> StaticKernelEntry | None:
        """Get default kernel for an operation.

        O(1) dict lookup.

        Args:
            operation: Operation identifier.

        Returns:
            Default StaticKernelEntry if found, None otherwise.
        """
        return self._defaults.get(operation)

    def get_kernel_for_operation(
        self,
        operation: str,
        context: "SelectionContext | None" = None,
    ) -> "KernelSpec":
        """Get kernel spec for operation.

        Primary lookup method for static dispatch.

        Args:
            operation: Operation identifier.
            context: Optional context (for future compatibility, not used).

        Returns:
            KernelSpec for the operation.

        Raises:
            KernelNotRegisteredError: If no kernel is registered.
        """
        entry = self._defaults.get(operation)
        if entry is None:
            raise KernelNotRegisteredError(operation)
        return entry.kernel_spec

    def operations(self) -> frozenset[str]:
        """Get all registered operations.

        Returns:
            Frozenset of operation identifiers.
        """
        return frozenset(self._by_operation.keys())

    def kernel_ids(self) -> frozenset[str]:
        """Get all registered kernel IDs.

        Returns:
            Frozenset of kernel identifiers.
        """
        return frozenset(self._by_kernel_id.keys())

    def to_mapping(self) -> dict[str, str]:
        """Export as operation -> kernel_id mapping.

        Useful for serialization and config export.

        Returns:
            Dict mapping operation to default kernel_id.
        """
        return {
            operation: entry.kernel_id
            for operation, entry in self._defaults.items()
        }

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[str, str],
        kernel_specs: dict[str, "KernelSpec"],
    ) -> "StaticKernelRegistry":
        """Create registry from operation -> kernel_id mapping.

        Args:
            mapping: Dict mapping operation to kernel_id.
            kernel_specs: Dict mapping kernel_id to KernelSpec.

        Returns:
            New StaticKernelRegistry instance.

        Raises:
            KeyError: If kernel_id not found in specs.
        """
        registry = cls()
        for operation, kernel_id in mapping.items():
            spec = kernel_specs[kernel_id]
            entry = StaticKernelEntry(
                kernel_id=kernel_id,
                operation=operation,
                operation_type=get_operation_type(operation),
                kernel_spec=spec,
                priority=spec.priority,
                is_default=True,
            )
            registry.register(entry)
        registry.freeze()
        return registry


# ============================================================================
# Global Registry Instance
# ============================================================================

import threading

# Module-level global registry for import-time initialization
_global_static_registry: StaticKernelRegistry | None = None
_global_registry_lock = threading.Lock()


def get_global_static_registry() -> StaticKernelRegistry:
    """Get or create the global static registry.

    Thread-safe using double-checked locking pattern.

    Returns:
        Global StaticKernelRegistry instance.
    """
    global _global_static_registry

    # Fast path: registry already exists
    if _global_static_registry is not None:
        return _global_static_registry

    # Slow path: need to create registry with lock
    with _global_registry_lock:
        # Double-check after acquiring lock
        if _global_static_registry is None:
            _global_static_registry = StaticKernelRegistry()
        return _global_static_registry


def set_global_static_registry(registry: StaticKernelRegistry) -> None:
    """Set the global static registry.

    Thread-safe assignment.

    Args:
        registry: Registry to set as global.
    """
    global _global_static_registry
    with _global_registry_lock:
        _global_static_registry = registry


# ============================================================================
# Static Dispatcher Implementation
# ============================================================================


class StaticDispatcher(BaseDispatcher):
    """Static dispatch mode implementation.

    Achieves near-zero overhead through:
    1. Pre-computed kernel mapping at initialization
    2. O(1) dict-based lookup
    3. Match statement for type-safe operation dispatch
    4. No runtime kernel selection logic

    Memory Efficiency:
    - Uses __slots__ for all instance attributes
    - Frozen dataclasses for kernel entries
    - Immutable registry after initialization

    Thread Safety:
    - No shared mutable state in dispatch path
    - Registry is frozen and immutable
    - All timing uses thread-local time.perf_counter_ns()

    Usage:
        config = DispatchConfig(mode=DispatchMode.STATIC)
        dispatcher = StaticDispatcher(config, registry)

        result = dispatcher.dispatch("attention.causal", inputs)
    """

    __slots__ = (
        "_registry",
        "_dispatch_count",
        "_total_dispatch_ns",
        "_kernel_cache",
        "_telemetry_lock",
    )

    def __init__(
        self,
        config: DispatchConfig,
        registry: StaticKernelRegistry | None = None,
        executor: KernelExecutor | None = None,
    ) -> None:
        """Initialize static dispatcher.

        Args:
            config: Dispatch configuration.
            registry: Static kernel registry. If None, global registry is used.
            executor: Kernel executor. If None, default is created.
        """
        super().__init__(config, executor)

        # Use provided registry or global
        self._registry = registry or get_global_static_registry()

        # Ensure registry is frozen
        if not self._registry.is_frozen:
            self._registry.freeze()

        # Telemetry counters (protected by lock for thread safety)
        self._telemetry_lock = threading.RLock()
        self._dispatch_count: int = 0
        self._total_dispatch_ns: int = 0

        # Inline cache for hot operations (operation -> kernel_spec)
        # This provides even faster lookup than the registry for repeated calls
        self._kernel_cache: dict[str, "KernelSpec"] = {}

        # Pre-warm cache with all registered operations
        for operation in self._registry.operations():
            entry = self._registry.get_default(operation)
            if entry is not None:
                self._kernel_cache[operation] = entry.kernel_spec

        logger.debug(
            f"StaticDispatcher initialized with {self._registry.kernel_count} kernels"
        )

    @property
    def mode(self) -> DispatchMode:
        """Get dispatch mode."""
        return DispatchMode.STATIC

    @property
    def registry(self) -> StaticKernelRegistry:
        """Get the static registry."""
        return self._registry

    def dispatch(
        self,
        operation: str,
        inputs: dict[str, "torch.Tensor"],
        context: "SelectionContext | None" = None,
        **kwargs: Any,
    ) -> DispatchResult:
        """Dispatch operation to kernel.

        Uses O(1) dict lookup and match statement for near-zero overhead.

        Args:
            operation: Operation identifier.
            inputs: Input tensors.
            context: Optional selection context (not used in static mode).
            **kwargs: Additional kernel arguments.

        Returns:
            DispatchResult with output and timing.

        Raises:
            KernelNotRegisteredError: If no kernel registered for operation.
            KernelExecutionError: If kernel execution fails.
        """
        # Start timing
        start_ns = time.perf_counter_ns()

        # O(1) kernel lookup from inline cache
        kernel_spec = self._kernel_cache.get(operation)

        if kernel_spec is None:
            # Cache miss - try registry lookup
            kernel_spec = self._get_kernel_with_match(operation)
            if kernel_spec is not None:
                # Populate cache for next time
                self._kernel_cache[operation] = kernel_spec
            else:
                raise KernelNotRegisteredError(operation)

        selection_ns = time.perf_counter_ns() - start_ns

        # Execute kernel
        execution_start = time.perf_counter_ns()

        if self._executor is None:
            self._executor = KernelExecutorImpl()

        try:
            output = self._executor.execute(kernel_spec, inputs, **kwargs)
        except KernelExecutionError:
            raise
        except Exception as e:
            raise KernelExecutionError(
                f"Static dispatch execution failed: {e}",
                operation=operation,
                kernel_id=kernel_spec.kernel_id,
                original_error=e,
            ) from e

        execution_ns = time.perf_counter_ns() - execution_start
        total_ns = time.perf_counter_ns() - start_ns

        # Create timing record
        timing = DispatchTiming(
            selection_ns=selection_ns,
            pre_transform_ns=0,
            execution_ns=execution_ns,
            post_transform_ns=0,
            total_ns=total_ns,
        )

        # Update telemetry (thread-safe, not on critical path)
        with self._telemetry_lock:
            self._dispatch_count += 1
            self._total_dispatch_ns += total_ns

        return DispatchResult(
            output=output,
            kernel_id=kernel_spec.kernel_id,
            kernel_spec=kernel_spec,
            timing=timing,
            mode=DispatchMode.STATIC,
            cached=operation in self._kernel_cache,
            fallback_used=False,
        )

    def _get_kernel_with_match(self, operation: str) -> "KernelSpec | None":
        """Get kernel using match statement dispatch.

        Provides type-safe dispatch with O(1) enum matching.

        Args:
            operation: Operation identifier.

        Returns:
            KernelSpec if found, None otherwise.
        """
        op_type = get_operation_type(operation)

        # Match statement for type-safe, O(1) dispatch
        match op_type:
            case OperationType.ATTENTION_CAUSAL:
                entry = self._registry.get_default("attention.causal")
            case OperationType.ATTENTION_FULL:
                entry = self._registry.get_default("attention.full")
            case OperationType.ATTENTION_SLIDING_WINDOW:
                entry = self._registry.get_default("attention.sliding_window")
            case OperationType.ATTENTION_CROSS:
                entry = self._registry.get_default("attention.cross")
            case OperationType.ATTENTION_PREFILL:
                entry = self._registry.get_default("attention.prefill")
            case OperationType.ATTENTION_DECODE:
                entry = self._registry.get_default("attention.decode")
            case OperationType.NORM_RMS:
                entry = self._registry.get_default("norm.rms") or \
                       self._registry.get_default("rms_norm")
            case OperationType.NORM_LAYER:
                entry = self._registry.get_default("norm.layer") or \
                       self._registry.get_default("layer_norm")
            case OperationType.NORM_GROUP:
                entry = self._registry.get_default("norm.group") or \
                       self._registry.get_default("group_norm")
            case OperationType.ROPE:
                entry = self._registry.get_default("rope") or \
                       self._registry.get_default("rope.standard")
            case OperationType.ROPE_INTERLEAVED:
                entry = self._registry.get_default("rope.interleaved")
            case OperationType.ALIBI:
                entry = self._registry.get_default("alibi")
            case OperationType.SWIGLU:
                entry = self._registry.get_default("swiglu") or \
                       self._registry.get_default("activation.swiglu")
            case OperationType.GELU:
                entry = self._registry.get_default("gelu") or \
                       self._registry.get_default("activation.gelu")
            case OperationType.SILU:
                entry = self._registry.get_default("silu") or \
                       self._registry.get_default("activation.silu")
            case OperationType.CROSS_ENTROPY:
                entry = self._registry.get_default("cross_entropy") or \
                       self._registry.get_default("loss.cross_entropy")
            case OperationType.FUSED_CROSS_ENTROPY:
                entry = self._registry.get_default("fused_cross_entropy") or \
                       self._registry.get_default("loss.fused_cross_entropy")
            case OperationType.MATMUL:
                entry = self._registry.get_default("matmul")
            case OperationType.MATMUL_FP8:
                entry = self._registry.get_default("matmul.fp8")
            case OperationType.TOKENIZE:
                entry = self._registry.get_default("tokenize") or \
                       self._registry.get_default("tokenization.encode")
            case OperationType.DETOKENIZE:
                entry = self._registry.get_default("detokenize") or \
                       self._registry.get_default("tokenization.decode")
            case OperationType.SAMPLE_TOP_K:
                entry = self._registry.get_default("sample.top_k")
            case OperationType.SAMPLE_TOP_P:
                entry = self._registry.get_default("sample.top_p")
            case OperationType.SAMPLE_GREEDY:
                entry = self._registry.get_default("sample.greedy")
            case OperationType.UNKNOWN:
                # Direct lookup for unknown operation types
                entry = self._registry.get_default(operation)
            case _:
                entry = None

        return entry.kernel_spec if entry else None

    def get_kernel_for_operation(
        self,
        operation: str,
        context: "SelectionContext",
    ) -> "KernelSpec":
        """Get kernel spec for operation without executing.

        Args:
            operation: Operation identifier.
            context: Selection context (not used in static mode).

        Returns:
            KernelSpec for the operation.

        Raises:
            KernelNotRegisteredError: If no kernel registered.
        """
        # First check inline cache
        if operation in self._kernel_cache:
            return self._kernel_cache[operation]

        # Fall back to registry
        return self._registry.get_kernel_for_operation(operation, context)

    def get_telemetry(self) -> dict[str, Any]:
        """Get telemetry data.

        Thread-safe access to telemetry counters.

        Returns:
            Dict with dispatch statistics.
        """
        base_telemetry = super().get_telemetry()
        with self._telemetry_lock:
            dispatch_count = self._dispatch_count
            total_dispatch_ns = self._total_dispatch_ns

        base_telemetry.update({
            "dispatch_count": dispatch_count,
            "total_dispatch_ns": total_dispatch_ns,
            "avg_dispatch_ns": (
                total_dispatch_ns / dispatch_count
                if dispatch_count > 0 else 0
            ),
            "kernel_count": self._registry.kernel_count,
            "operation_count": self._registry.operation_count,
            "cache_size": len(self._kernel_cache),
        })
        return base_telemetry

    def reset_telemetry(self) -> None:
        """Reset telemetry counters.

        Thread-safe reset.
        """
        super().reset_telemetry()
        with self._telemetry_lock:
            self._dispatch_count = 0
            self._total_dispatch_ns = 0


# ============================================================================
# Builder Pattern for Easy Setup
# ============================================================================


class StaticDispatcherBuilder:
    """Builder for constructing StaticDispatcher instances.

    Provides fluent API for setting up static dispatch with
    kernel registration and configuration.

    Usage:
        dispatcher = (
            StaticDispatcherBuilder()
            .with_kernel(flash_attn_spec, operation="attention.causal", default=True)
            .with_kernel(sdpa_spec, operation="attention.causal")
            .with_kernel(rms_norm_spec, operation="norm.rms", default=True)
            .with_config(enable_telemetry=True)
            .build()
        )
    """

    __slots__ = ("_registry", "_config", "_executor")

    def __init__(self) -> None:
        """Initialize builder."""
        self._registry = StaticKernelRegistry()
        self._config = DispatchConfig(mode=DispatchMode.STATIC)
        self._executor: KernelExecutor | None = None

    def with_kernel(
        self,
        spec: "KernelSpec",
        *,
        operation: str | None = None,
        default: bool = False,
    ) -> "StaticDispatcherBuilder":
        """Add kernel to registry.

        Args:
            spec: Kernel specification.
            operation: Override operation (defaults to spec.operation).
            default: Whether this is the default kernel for operation.

        Returns:
            Self for chaining.
        """
        entry = StaticKernelEntry(
            kernel_id=spec.kernel_id,
            operation=operation or spec.operation,
            operation_type=get_operation_type(operation or spec.operation),
            kernel_spec=spec,
            priority=spec.priority,
            is_default=default,
        )
        self._registry.register(entry)
        return self

    def with_kernels(
        self,
        specs: list["KernelSpec"],
        *,
        defaults: dict[str, str] | None = None,
    ) -> "StaticDispatcherBuilder":
        """Add multiple kernels to registry.

        Args:
            specs: List of kernel specifications.
            defaults: Dict mapping operation to default kernel_id.

        Returns:
            Self for chaining.
        """
        defaults = defaults or {}
        for spec in specs:
            is_default = defaults.get(spec.operation) == spec.kernel_id
            self.with_kernel(spec, default=is_default)
        return self

    def with_config(self, **kwargs: Any) -> "StaticDispatcherBuilder":
        """Update configuration.

        Args:
            **kwargs: Config fields to update.

        Returns:
            Self for chaining.
        """
        # Create new config with updates
        config_dict = {
            "mode": DispatchMode.STATIC,
            "enable_cache": self._config.enable_cache,
            "cache_size": self._config.cache_size,
            "cache_ttl_seconds": self._config.cache_ttl_seconds,
            "enable_fallback": self._config.enable_fallback,
            "max_fallback_attempts": self._config.max_fallback_attempts,
            "fallback_timeout_ms": self._config.fallback_timeout_ms,
            "config_path": self._config.config_path,
            "watch_interval_seconds": self._config.watch_interval_seconds,
            "validate_on_reload": self._config.validate_on_reload,
            "enable_transforms": self._config.enable_transforms,
            "enable_cuda_graphs": self._config.enable_cuda_graphs,
            "sync_after_execution": self._config.sync_after_execution,
            "enable_telemetry": self._config.enable_telemetry,
            "record_timing": self._config.record_timing,
            "log_fallbacks": self._config.log_fallbacks,
            "circuit_breaker_enabled": self._config.circuit_breaker_enabled,
            "failure_threshold": self._config.failure_threshold,
            "recovery_timeout_seconds": self._config.recovery_timeout_seconds,
            "static_kernel_map": self._config.static_kernel_map,
        }
        config_dict.update(kwargs)
        self._config = DispatchConfig(**config_dict)
        return self

    def with_executor(self, executor: KernelExecutor) -> "StaticDispatcherBuilder":
        """Set custom kernel executor.

        Args:
            executor: Kernel executor instance.

        Returns:
            Self for chaining.
        """
        self._executor = executor
        return self

    def with_registry(
        self,
        registry: StaticKernelRegistry,
    ) -> "StaticDispatcherBuilder":
        """Use existing registry.

        Args:
            registry: Pre-built registry.

        Returns:
            Self for chaining.
        """
        self._registry = registry
        return self

    def build(self) -> StaticDispatcher:
        """Build the dispatcher.

        Freezes the registry if not already frozen.

        Returns:
            Configured StaticDispatcher instance.
        """
        if not self._registry.is_frozen:
            self._registry.freeze()

        return StaticDispatcher(
            config=self._config,
            registry=self._registry,
            executor=self._executor,
        )


# ============================================================================
# Factory Functions
# ============================================================================


def create_static_dispatcher(
    kernel_specs: list["KernelSpec"],
    defaults: dict[str, str] | None = None,
    config: DispatchConfig | None = None,
    executor: KernelExecutor | None = None,
) -> StaticDispatcher:
    """Create static dispatcher from kernel specs.

    Convenience function for quick setup.

    Args:
        kernel_specs: List of kernel specifications.
        defaults: Dict mapping operation to default kernel_id.
        config: Optional dispatch configuration.
        executor: Optional kernel executor.

    Returns:
        Configured StaticDispatcher instance.
    """
    builder = StaticDispatcherBuilder()

    # Add kernels
    defaults = defaults or {}
    for spec in kernel_specs:
        is_default = defaults.get(spec.operation) == spec.kernel_id
        builder.with_kernel(spec, default=is_default)

    # Set config if provided
    if config is not None:
        builder._config = config

    # Set executor if provided
    if executor is not None:
        builder.with_executor(executor)

    return builder.build()


def create_static_dispatcher_from_config(
    config: DispatchConfig,
    kernel_registry: "KernelRegistry",
    executor: KernelExecutor | None = None,
) -> StaticDispatcher:
    """Create static dispatcher from config and kernel registry.

    Uses static_kernel_map from config to build static registry.

    Args:
        config: Dispatch configuration with static_kernel_map.
        kernel_registry: Kernel registry to pull specs from.
        executor: Optional kernel executor.

    Returns:
        Configured StaticDispatcher instance.

    Raises:
        ValueError: If kernel_id in map not found in registry.
    """
    from layerzero.registry.kernel_registry import KernelRegistry

    static_registry = StaticKernelRegistry()

    for operation, kernel_id in config.static_kernel_map.items():
        spec = kernel_registry.get(kernel_id)
        if spec is None:
            raise ValueError(
                f"Kernel '{kernel_id}' for operation '{operation}' "
                f"not found in kernel registry"
            )
        entry = StaticKernelEntry(
            kernel_id=kernel_id,
            operation=operation,
            operation_type=get_operation_type(operation),
            kernel_spec=spec,
            priority=spec.priority,
            is_default=True,
        )
        static_registry.register(entry)

    static_registry.freeze()

    return StaticDispatcher(
        config=config,
        registry=static_registry,
        executor=executor,
    )


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    # Error types
    "StaticDispatchError",
    "KernelNotRegisteredError",
    "InvalidKernelEntryError",
    "RegistryFrozenError",
    # Operation types
    "OperationType",
    "OPERATION_TYPE_MAP",
    "get_operation_type",
    # Kernel entry
    "StaticKernelEntry",
    # Registry
    "StaticKernelRegistry",
    "get_global_static_registry",
    "set_global_static_registry",
    # Dispatcher
    "StaticDispatcher",
    "StaticDispatcherBuilder",
    # Factory functions
    "create_static_dispatcher",
    "create_static_dispatcher_from_config",
]
