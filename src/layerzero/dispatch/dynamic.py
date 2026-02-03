"""
LayerZero Dynamic Kernel Dispatcher

Runtime kernel selection with ~100-500ns overhead.

Features:
- Integration with SelectionEngine for intelligent kernel selection
- Circuit breaker pattern for kernel health tracking
- Fallback chain when primary kernel fails
- Selection result caching with MVCC support
- Thread-safe operation

The dynamic dispatcher provides the most flexibility, selecting the best
kernel at runtime based on the current context (tensor shapes, dtypes,
device capabilities, etc.). The trade-off is a small overhead compared
to static dispatch.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from threading import RLock
from typing import Any, TYPE_CHECKING

from layerzero.dispatch.protocols import BaseDispatcher
from layerzero.dispatch.types import (
    CircuitOpenError,
    DispatchError,
    DispatchMode,
    DispatchPhase,
    DispatchResult,
    DispatchTiming,
    FallbackChainExhaustedError,
    KernelExecutionError,
)
from layerzero.dispatch.executor import KernelExecutorImpl
from layerzero.selection.engine import NoKernelAvailableError, SelectionEngine

if TYPE_CHECKING:
    import torch
    from layerzero.dispatch.types import DispatchConfig
    from layerzero.models.execution_plan import ExecutionPlan
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.models.selection_context import SelectionContext
    from layerzero.registry.backend_registry import BackendRegistry
    from layerzero.registry.kernel_registry import KernelRegistry
    from layerzero.selection.mvcc_cache import MVCCShardedCache

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker state for kernel health."""
    CLOSED = auto()    # Normal operation, allowing requests
    OPEN = auto()      # Failures exceeded threshold, blocking requests
    HALF_OPEN = auto() # Testing if kernel has recovered


@dataclass
class KernelCircuit:
    """Circuit breaker state for a single kernel.

    Tracks failures and controls whether a kernel should be tried.

    Attributes:
        kernel_id: Kernel identifier.
        state: Current circuit state.
        failure_count: Consecutive failure count.
        last_failure_time: Monotonic time of last failure.
        last_success_time: Monotonic time of last success.
        cooldown_until: Monotonic time when half-open transition occurs.
        total_failures: Lifetime failure count.
        total_successes: Lifetime success count.
    """
    __slots__ = (
        "kernel_id",
        "state",
        "failure_count",
        "last_failure_time",
        "last_success_time",
        "cooldown_until",
        "total_failures",
        "total_successes",
    )

    kernel_id: str
    state: CircuitState
    failure_count: int
    last_failure_time: float | None
    last_success_time: float | None
    cooldown_until: float | None
    total_failures: int
    total_successes: int

    def __init__(self, kernel_id: str) -> None:
        """Initialize circuit in closed state."""
        self.kernel_id = kernel_id
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.cooldown_until = None
        self.total_failures = 0
        self.total_successes = 0


class CircuitBreakerManager:
    """Manages circuit breakers for all kernels.

    Implements the circuit breaker pattern:
    1. CLOSED: Normal operation, requests pass through
    2. After N consecutive failures, circuit opens (OPEN)
    3. After cooldown period, circuit enters HALF_OPEN
    4. In HALF_OPEN: success -> CLOSED, failure -> OPEN

    Thread-safe for concurrent access.
    """

    __slots__ = (
        "_lock",
        "_circuits",
        "_failure_threshold",
        "_cooldown_seconds",
    )

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
    ) -> None:
        """Initialize circuit breaker manager.

        Args:
            failure_threshold: Consecutive failures to open circuit.
            cooldown_seconds: Seconds before half-open transition.
        """
        self._lock = RLock()
        self._circuits: dict[str, KernelCircuit] = {}
        self._failure_threshold = max(1, failure_threshold)
        self._cooldown_seconds = max(0.0, cooldown_seconds)

    def _get_or_create_circuit(self, kernel_id: str) -> KernelCircuit:
        """Get or create circuit for kernel (caller must hold lock)."""
        if kernel_id not in self._circuits:
            self._circuits[kernel_id] = KernelCircuit(kernel_id)
        return self._circuits[kernel_id]

    def is_allowed(self, kernel_id: str) -> bool:
        """Check if kernel is allowed (circuit not open).

        Also handles half-open transition based on cooldown.

        Args:
            kernel_id: Kernel identifier.

        Returns:
            True if kernel can be tried, False if circuit is open.
        """
        with self._lock:
            circuit = self._get_or_create_circuit(kernel_id)
            now = time.monotonic()

            if circuit.state == CircuitState.CLOSED:
                return True

            if circuit.state == CircuitState.HALF_OPEN:
                return True  # Allow one test request

            # OPEN state - check if cooldown has elapsed
            if circuit.cooldown_until is not None and now >= circuit.cooldown_until:
                circuit.state = CircuitState.HALF_OPEN
                logger.debug(
                    f"Circuit for kernel '{kernel_id}' transitioning to HALF_OPEN"
                )
                return True

            return False

    def get_retry_after(self, kernel_id: str) -> float | None:
        """Get seconds until circuit might allow retry.

        Args:
            kernel_id: Kernel identifier.

        Returns:
            Seconds until retry is allowed, or None if allowed now.
        """
        with self._lock:
            circuit = self._circuits.get(kernel_id)
            if circuit is None:
                return None

            if circuit.state != CircuitState.OPEN:
                return None

            if circuit.cooldown_until is None:
                return None

            now = time.monotonic()
            remaining = circuit.cooldown_until - now
            return max(0.0, remaining)

    def record_success(self, kernel_id: str) -> None:
        """Record successful kernel execution.

        Resets failure count and closes circuit.

        Args:
            kernel_id: Kernel identifier.
        """
        with self._lock:
            circuit = self._get_or_create_circuit(kernel_id)
            circuit.failure_count = 0
            circuit.last_success_time = time.monotonic()
            circuit.total_successes += 1

            if circuit.state != CircuitState.CLOSED:
                logger.info(
                    f"Circuit for kernel '{kernel_id}' closed after success"
                )
                circuit.state = CircuitState.CLOSED
                circuit.cooldown_until = None

    def record_failure(
        self,
        kernel_id: str,
        error: Exception | None = None,
    ) -> None:
        """Record failed kernel execution.

        Increments failure count. If threshold reached, opens circuit.

        Args:
            kernel_id: Kernel identifier.
            error: Optional exception that caused failure.
        """
        with self._lock:
            circuit = self._get_or_create_circuit(kernel_id)
            now = time.monotonic()

            circuit.failure_count += 1
            circuit.last_failure_time = now
            circuit.total_failures += 1

            if circuit.state == CircuitState.HALF_OPEN:
                # Test failed, go back to open
                circuit.state = CircuitState.OPEN
                circuit.cooldown_until = now + self._cooldown_seconds
                logger.warning(
                    f"Circuit for kernel '{kernel_id}' re-opened after "
                    f"half-open test failure"
                )
                return

            if circuit.failure_count >= self._failure_threshold:
                circuit.state = CircuitState.OPEN
                circuit.cooldown_until = now + self._cooldown_seconds
                error_msg = str(error) if error else "unknown"
                logger.warning(
                    f"Circuit for kernel '{kernel_id}' opened after "
                    f"{circuit.failure_count} failures. Last error: {error_msg}"
                )

    def reset(self, kernel_id: str) -> None:
        """Reset circuit for kernel to closed state.

        Args:
            kernel_id: Kernel identifier.
        """
        with self._lock:
            if kernel_id in self._circuits:
                circuit = self._circuits[kernel_id]
                circuit.state = CircuitState.CLOSED
                circuit.failure_count = 0
                circuit.cooldown_until = None

    def reset_all(self) -> None:
        """Reset all circuits to closed state."""
        with self._lock:
            for circuit in self._circuits.values():
                circuit.state = CircuitState.CLOSED
                circuit.failure_count = 0
                circuit.cooldown_until = None

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all circuits.

        Returns:
            Dict with circuit statistics.
        """
        with self._lock:
            total = len(self._circuits)
            open_count = sum(
                1 for c in self._circuits.values()
                if c.state == CircuitState.OPEN
            )
            half_open_count = sum(
                1 for c in self._circuits.values()
                if c.state == CircuitState.HALF_OPEN
            )

            return {
                "total_circuits": total,
                "open": open_count,
                "half_open": half_open_count,
                "closed": total - open_count - half_open_count,
                "failure_threshold": self._failure_threshold,
                "cooldown_seconds": self._cooldown_seconds,
            }


class FallbackChainImpl:
    """Fallback chain implementation for kernel failures.

    When a kernel fails, this provides alternative kernels to try
    in priority order. Tracks failure history to inform future decisions.
    """

    __slots__ = (
        "_selection_engine",
        "_circuit_breaker",
        "_max_fallbacks",
    )

    def __init__(
        self,
        selection_engine: SelectionEngine,
        circuit_breaker: CircuitBreakerManager,
        max_fallbacks: int = 3,
    ) -> None:
        """Initialize fallback chain.

        Args:
            selection_engine: Selection engine for finding alternatives.
            circuit_breaker: Circuit breaker for health tracking.
            max_fallbacks: Maximum number of fallback attempts.
        """
        self._selection_engine = selection_engine
        self._circuit_breaker = circuit_breaker
        self._max_fallbacks = max(1, max_fallbacks)

    def get_fallbacks(
        self,
        operation: str,
        failed_kernel_id: str,
        context: "SelectionContext",
    ) -> list["KernelSpec"]:
        """Get ordered list of fallback kernels.

        Excludes the failed kernel and any with open circuits.

        Args:
            operation: Operation that needs a fallback.
            failed_kernel_id: Kernel that already failed.
            context: Selection context.

        Returns:
            List of fallback kernels in priority order.
        """
        # Get all candidates for this operation
        kernel_registry = self._selection_engine.kernel_registry
        candidates = kernel_registry.get_by_operation(operation)

        fallbacks: list["KernelSpec"] = []

        for candidate in candidates:
            # Skip the failed kernel
            if candidate.kernel_id == failed_kernel_id:
                continue

            # Skip kernels with open circuits
            if not self._circuit_breaker.is_allowed(candidate.kernel_id):
                continue

            # Check if kernel is compatible with context
            reasons = candidate.check(context)
            if not reasons:
                fallbacks.append(candidate)

        # Sort by priority (higher is better)
        fallbacks.sort(key=lambda k: k.priority, reverse=True)

        # Limit number of fallbacks
        return fallbacks[:self._max_fallbacks]

    def record_failure(
        self,
        kernel_id: str,
        error: Exception,
    ) -> None:
        """Record kernel failure.

        Args:
            kernel_id: Kernel that failed.
            error: The error that occurred.
        """
        self._circuit_breaker.record_failure(kernel_id, error)


class DynamicDispatcher(BaseDispatcher):
    """Dynamic kernel dispatcher with runtime selection.

    Provides runtime kernel selection with ~100-500ns overhead.

    Features:
    - Integration with SelectionEngine for intelligent kernel selection
    - Circuit breaker pattern for kernel health tracking
    - Fallback chain when primary kernel fails
    - Selection result caching with MVCC support
    - Thread-safe operation

    Thread Safety:
    - All public methods are thread-safe
    - Uses fine-grained locking to minimize contention
    - Selection caching uses MVCC for concurrent access

    Usage:
        dispatcher = DynamicDispatcher(
            selection_engine=engine,
            backend_registry=registry,
            config=config,
        )

        result = dispatcher.dispatch(
            operation="attention.causal",
            inputs={"query": q, "key": k, "value": v},
            context=ctx,
            is_causal=True,
        )
    """

    __slots__ = (
        "_selection_engine",
        "_backend_registry",
        "_circuit_breaker",
        "_fallback_chain",
        "_mvcc_cache",
        "_telemetry_lock",
        "_dispatch_count",
        "_fallback_count",
        "_error_count",
        "_total_selection_ns",
        "_total_execution_ns",
    )

    def __init__(
        self,
        selection_engine: SelectionEngine,
        backend_registry: "BackendRegistry | None" = None,
        config: "DispatchConfig | None" = None,
        executor: KernelExecutorImpl | None = None,
        mvcc_cache: "MVCCShardedCache | None" = None,
    ) -> None:
        """Initialize dynamic dispatcher.

        Args:
            selection_engine: SelectionEngine for kernel selection.
            backend_registry: Optional backend registry for health tracking.
            config: Dispatch configuration. Uses defaults if None.
            executor: Kernel executor. Creates default if None.
            mvcc_cache: Optional MVCC cache for selection results.
        """
        from layerzero.dispatch.types import DispatchConfig

        # Use default config if not provided
        if config is None:
            config = DispatchConfig(mode=DispatchMode.DYNAMIC)

        # Create executor if not provided
        if executor is None:
            executor = KernelExecutorImpl(backend_registry)

        super().__init__(config, executor)

        self._selection_engine = selection_engine
        self._backend_registry = backend_registry

        # Initialize circuit breaker with config values
        self._circuit_breaker = CircuitBreakerManager(
            failure_threshold=config.failure_threshold,
            cooldown_seconds=config.recovery_timeout_seconds,
        )

        # Initialize fallback chain
        self._fallback_chain = FallbackChainImpl(
            selection_engine=selection_engine,
            circuit_breaker=self._circuit_breaker,
            max_fallbacks=config.max_fallback_attempts,
        )

        # Optional MVCC cache integration
        self._mvcc_cache = mvcc_cache

        # Telemetry (protected by lock)
        self._telemetry_lock = RLock()
        self._dispatch_count = 0
        self._fallback_count = 0
        self._error_count = 0
        self._total_selection_ns = 0
        self._total_execution_ns = 0

    @property
    def mode(self) -> DispatchMode:
        """Get dispatch mode (always DYNAMIC)."""
        return DispatchMode.DYNAMIC

    @property
    def selection_engine(self) -> SelectionEngine:
        """Get the selection engine."""
        return self._selection_engine

    @property
    def circuit_breaker(self) -> CircuitBreakerManager:
        """Get the circuit breaker manager."""
        return self._circuit_breaker

    def dispatch(
        self,
        operation: str,
        inputs: dict[str, "torch.Tensor"],
        context: "SelectionContext | None" = None,
        **kwargs: Any,
    ) -> DispatchResult:
        """Dispatch operation to the best available kernel.

        Pipeline:
        1. Build or use provided SelectionContext
        2. Check MVCC cache for cached selection
        3. Use SelectionEngine to select best kernel
        4. Check circuit breaker for kernel health
        5. Execute kernel with executor
        6. If failure, try fallback chain
        7. Update circuit breaker and telemetry
        8. Return DispatchResult with timing

        Args:
            operation: Operation identifier (e.g., "attention.causal").
            inputs: Dictionary of named input tensors.
            context: Optional pre-built selection context.
            **kwargs: Additional operation-specific arguments.

        Returns:
            DispatchResult with output tensor and metadata.

        Raises:
            DispatchError: If dispatch fails.
            FallbackChainExhaustedError: If all kernels fail.
        """
        total_start = time.perf_counter_ns()

        # Build context if not provided
        if context is None:
            context = self._build_context(operation, inputs, **kwargs)

        # Track timing
        selection_ns = 0
        pre_transform_ns = 0
        execution_ns = 0
        post_transform_ns = 0

        fallback_used = False
        fallback_reason: str | None = None
        cached = False

        # Select kernel
        selection_start = time.perf_counter_ns()

        try:
            plan = self._select_kernel(context)
            cached = plan.cached
        except NoKernelAvailableError as e:
            selection_ns = time.perf_counter_ns() - selection_start
            self._record_error()
            raise DispatchError(
                message=str(e),
                operation=operation,
                phase=DispatchPhase.SELECTION,
            ) from e

        selection_ns = time.perf_counter_ns() - selection_start

        # Get kernel spec
        kernel_spec = plan.kernel_spec
        kernel_id = kernel_spec.kernel_id

        # Check circuit breaker
        if not self._circuit_breaker.is_allowed(kernel_id):
            retry_after = self._circuit_breaker.get_retry_after(kernel_id)

            # Try to get a fallback
            if self._config.enable_fallback:
                fallbacks = self._fallback_chain.get_fallbacks(
                    operation, kernel_id, context
                )

                if fallbacks:
                    kernel_spec = fallbacks[0]
                    kernel_id = kernel_spec.kernel_id
                    fallback_used = True
                    fallback_reason = f"Primary kernel circuit open"
                    logger.debug(
                        f"Using fallback kernel '{kernel_id}' because "
                        f"primary kernel circuit is open"
                    )
                else:
                    raise CircuitOpenError(
                        kernel_id=plan.kernel_spec.kernel_id,
                        retry_after_seconds=retry_after or 0.0,
                    )
            else:
                raise CircuitOpenError(
                    kernel_id=kernel_id,
                    retry_after_seconds=retry_after or 0.0,
                )

        # Execute with fallback support
        attempted_kernels: list[str] = []
        errors: list[Exception] = []
        output: "torch.Tensor | None" = None

        # First attempt
        exec_start = time.perf_counter_ns()
        try:
            output = self._execute_kernel(kernel_spec, inputs, **kwargs)
            self._circuit_breaker.record_success(kernel_id)
            execution_ns = time.perf_counter_ns() - exec_start

        except Exception as e:
            execution_ns = time.perf_counter_ns() - exec_start
            attempted_kernels.append(kernel_id)
            errors.append(e)
            self._circuit_breaker.record_failure(kernel_id, e)

            logger.warning(
                f"Kernel '{kernel_id}' failed: {e}. "
                f"Attempting fallback..."
            )

            # Try fallbacks if enabled
            if self._config.enable_fallback:
                output, fallback_kernel, fallback_exec_ns = self._try_fallbacks(
                    operation=operation,
                    failed_kernel_id=kernel_id,
                    context=context,
                    inputs=inputs,
                    attempted_kernels=attempted_kernels,
                    errors=errors,
                    **kwargs,
                )

                if output is not None:
                    kernel_spec = fallback_kernel  # type: ignore
                    kernel_id = kernel_spec.kernel_id
                    execution_ns += fallback_exec_ns
                    fallback_used = True
                    fallback_reason = f"Primary kernel failed: {e}"

        # All attempts failed
        if output is None:
            self._record_error()
            raise FallbackChainExhaustedError(
                operation=operation,
                attempted_kernels=attempted_kernels,
                errors=errors,
            )

        total_ns = time.perf_counter_ns() - total_start

        # Update telemetry
        self._record_dispatch(
            selection_ns=selection_ns,
            execution_ns=execution_ns,
            fallback_used=fallback_used,
        )

        # Build result
        timing = DispatchTiming(
            selection_ns=selection_ns,
            pre_transform_ns=pre_transform_ns,
            execution_ns=execution_ns,
            post_transform_ns=post_transform_ns,
            total_ns=total_ns,
        )

        return DispatchResult(
            output=output,
            kernel_id=kernel_id,
            kernel_spec=kernel_spec,
            timing=timing,
            mode=DispatchMode.DYNAMIC,
            cached=cached,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    def get_kernel_for_operation(
        self,
        operation: str,
        context: "SelectionContext",
    ) -> "KernelSpec":
        """Get the kernel that would be selected for an operation.

        Useful for inspection and debugging without executing.
        Does not execute the kernel or affect circuit breaker state.

        Args:
            operation: Operation identifier.
            context: Selection context.

        Returns:
            KernelSpec that would be selected.

        Raises:
            NoKernelAvailableError: If no kernel matches.
        """
        plan = self._select_kernel(context)
        return plan.kernel_spec

    def _build_context(
        self,
        operation: str,
        inputs: dict[str, "torch.Tensor"],
        **kwargs: Any,
    ) -> "SelectionContext":
        """Build SelectionContext from inputs.

        Args:
            operation: Operation identifier.
            inputs: Input tensors.
            **kwargs: Additional context parameters.

        Returns:
            Built SelectionContext.
        """
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind

        # Get first tensor for device/dtype info
        first_tensor = next(iter(inputs.values()))
        device = DeviceSpec.detect(str(first_tensor.device))
        dtype = first_tensor.dtype
        batch_size = first_tensor.shape[0] if first_tensor.ndim > 0 else 1

        # Handle attention operations
        if operation.startswith("attention"):
            q = inputs.get("query") or inputs.get("q")
            k = inputs.get("key") or inputs.get("k")
            v = inputs.get("value") or inputs.get("v")

            if q is not None:
                return SelectionContext.from_tensors(
                    q=q,
                    k=k,
                    v=v,
                    is_causal=kwargs.get("is_causal", kwargs.get("causal", False)),
                    dropout_p=kwargs.get("dropout_p", 0.0),
                    scale=kwargs.get("scale"),
                    attn_mask=kwargs.get("attn_mask"),
                    device=device,
                    **{k: v for k, v in kwargs.items()
                       if k not in ("is_causal", "causal", "dropout_p", "scale", "attn_mask")},
                )

        # Handle normalization operations
        if operation.startswith("norm") or operation.startswith("rms_norm"):
            x = inputs.get("input") or inputs.get("x") or inputs.get("hidden_states")
            if x is not None:
                return SelectionContext.for_norm(
                    x=x,
                    operation=operation,
                    device=device,
                    **kwargs,
                )

        # Generic context
        return SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation=operation,
            dtype=dtype,
            batch_size=batch_size,
            is_contiguous=first_tensor.is_contiguous(),
            stride_last_dim=first_tensor.stride(-1) if first_tensor.ndim > 0 else 1,
            **{k: v for k, v in kwargs.items()
               if k in SelectionContext.__dataclass_fields__},
        )

    def _select_kernel(
        self,
        context: "SelectionContext",
    ) -> "ExecutionPlan":
        """Select kernel using SelectionEngine with optional MVCC cache.

        Args:
            context: Selection context.

        Returns:
            ExecutionPlan with selected kernel.

        Raises:
            NoKernelAvailableError: If no kernel matches.
        """
        cache_key = context.cache_key()
        policy_hash = self._selection_engine.policy.policy_hash

        # Try MVCC cache first
        if self._mvcc_cache is not None and self._config.enable_cache:
            cached_plan = self._mvcc_cache.get(cache_key, policy_hash)
            if cached_plan is not None:
                logger.debug(
                    f"Cache hit for {context.operation} "
                    f"(key={cache_key[:8]}...)"
                )
                return cached_plan

        # Use selection engine
        plan = self._selection_engine.select(
            context,
            use_cache=self._config.enable_cache,
            debug=False,
        )

        # Store in MVCC cache
        if self._mvcc_cache is not None and self._config.enable_cache:
            self._mvcc_cache.put(cache_key, policy_hash, plan)

        return plan

    def _execute_kernel(
        self,
        kernel_spec: "KernelSpec",
        inputs: dict[str, "torch.Tensor"],
        **kwargs: Any,
    ) -> "torch.Tensor":
        """Execute kernel with executor.

        Args:
            kernel_spec: Kernel to execute.
            inputs: Input tensors.
            **kwargs: Additional arguments.

        Returns:
            Output tensor.

        Raises:
            KernelExecutionError: If execution fails.
        """
        return self._executor.execute(kernel_spec, inputs, **kwargs)

    def _try_fallbacks(
        self,
        operation: str,
        failed_kernel_id: str,
        context: "SelectionContext",
        inputs: dict[str, "torch.Tensor"],
        attempted_kernels: list[str],
        errors: list[Exception],
        **kwargs: Any,
    ) -> tuple["torch.Tensor | None", "KernelSpec | None", int]:
        """Try fallback kernels after primary failure.

        Args:
            operation: Operation identifier.
            failed_kernel_id: Kernel that failed.
            context: Selection context.
            inputs: Input tensors.
            attempted_kernels: List to append attempted kernel IDs to.
            errors: List to append errors to.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (output tensor or None, kernel spec or None, execution time ns).
        """
        fallbacks = self._fallback_chain.get_fallbacks(
            operation, failed_kernel_id, context
        )

        total_exec_ns = 0

        for fallback in fallbacks:
            if fallback.kernel_id in attempted_kernels:
                continue

            logger.debug(f"Trying fallback kernel '{fallback.kernel_id}'")

            exec_start = time.perf_counter_ns()
            try:
                output = self._execute_kernel(fallback, inputs, **kwargs)
                exec_ns = time.perf_counter_ns() - exec_start
                total_exec_ns += exec_ns

                self._circuit_breaker.record_success(fallback.kernel_id)

                logger.info(
                    f"Fallback kernel '{fallback.kernel_id}' succeeded"
                )

                return output, fallback, total_exec_ns

            except Exception as e:
                exec_ns = time.perf_counter_ns() - exec_start
                total_exec_ns += exec_ns

                attempted_kernels.append(fallback.kernel_id)
                errors.append(e)

                self._circuit_breaker.record_failure(fallback.kernel_id, e)
                self._fallback_chain.record_failure(fallback.kernel_id, e)

                logger.warning(
                    f"Fallback kernel '{fallback.kernel_id}' failed: {e}"
                )

        return None, None, total_exec_ns

    def _record_dispatch(
        self,
        selection_ns: int,
        execution_ns: int,
        fallback_used: bool,
    ) -> None:
        """Record dispatch telemetry.

        Args:
            selection_ns: Selection time in nanoseconds.
            execution_ns: Execution time in nanoseconds.
            fallback_used: Whether fallback was used.
        """
        with self._telemetry_lock:
            self._dispatch_count += 1
            self._total_selection_ns += selection_ns
            self._total_execution_ns += execution_ns
            if fallback_used:
                self._fallback_count += 1

    def _record_error(self) -> None:
        """Record dispatch error."""
        with self._telemetry_lock:
            self._error_count += 1

    def get_telemetry(self) -> dict[str, Any]:
        """Get telemetry data.

        Returns:
            Dict of telemetry metrics including:
            - dispatch_count: Total dispatches
            - fallback_count: Dispatches using fallback
            - error_count: Failed dispatches
            - avg_selection_ns: Average selection time
            - avg_execution_ns: Average execution time
            - circuit_breaker_stats: Circuit breaker statistics
        """
        with self._telemetry_lock:
            dispatch_count = self._dispatch_count
            fallback_count = self._fallback_count
            error_count = self._error_count
            total_selection_ns = self._total_selection_ns
            total_execution_ns = self._total_execution_ns

        avg_selection_ns = (
            total_selection_ns / dispatch_count
            if dispatch_count > 0 else 0
        )
        avg_execution_ns = (
            total_execution_ns / dispatch_count
            if dispatch_count > 0 else 0
        )

        return {
            "mode": "dynamic",
            "dispatch_count": dispatch_count,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / dispatch_count if dispatch_count > 0 else 0.0,
            "error_count": error_count,
            "error_rate": error_count / (dispatch_count + error_count) if (dispatch_count + error_count) > 0 else 0.0,
            "avg_selection_ns": avg_selection_ns,
            "avg_selection_us": avg_selection_ns / 1000,
            "avg_execution_ns": avg_execution_ns,
            "avg_execution_us": avg_execution_ns / 1000,
            "total_selection_ns": total_selection_ns,
            "total_execution_ns": total_execution_ns,
            "circuit_breaker": self._circuit_breaker.get_stats(),
            "cache_enabled": self._mvcc_cache is not None and self._config.enable_cache,
        }

    def reset_telemetry(self) -> None:
        """Reset telemetry counters."""
        with self._telemetry_lock:
            self._dispatch_count = 0
            self._fallback_count = 0
            self._error_count = 0
            self._total_selection_ns = 0
            self._total_execution_ns = 0

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers to closed state."""
        self._circuit_breaker.reset_all()

    def invalidate_cache(self) -> None:
        """Invalidate all cached selections."""
        if self._mvcc_cache is not None:
            self._mvcc_cache.invalidate_all()

        # Also invalidate selection engine's cache
        self._selection_engine.cache.clear()


def create_dynamic_dispatcher(
    kernel_registry: "KernelRegistry",
    backend_registry: "BackendRegistry",
    config: "DispatchConfig | None" = None,
    use_mvcc_cache: bool = True,
) -> DynamicDispatcher:
    """Create a fully configured dynamic dispatcher.

    Factory function that sets up all components:
    - SelectionEngine with default policy
    - KernelExecutorImpl
    - MVCC cache (optional)
    - Circuit breaker

    Args:
        kernel_registry: Registry of available kernels.
        backend_registry: Registry of backends for health tracking.
        config: Dispatch configuration. Uses defaults if None.
        use_mvcc_cache: Whether to use MVCC cache for selections.

    Returns:
        Configured DynamicDispatcher instance.
    """
    from layerzero.dispatch.types import DispatchConfig
    from layerzero.selection.mvcc_cache import MVCCShardedCache

    # Use default config if not provided
    if config is None:
        config = DispatchConfig(mode=DispatchMode.DYNAMIC)

    # Create selection engine
    selection_engine = SelectionEngine(
        kernel_registry=kernel_registry,
        backend_registry=backend_registry,
    )

    # Create MVCC cache if enabled
    mvcc_cache = None
    if use_mvcc_cache and config.enable_cache:
        mvcc_cache = MVCCShardedCache(
            num_shards=256,
            max_entries_per_shard=config.cache_size // 256 + 1,
            ttl_seconds=config.cache_ttl_seconds,
        )

    # Create executor
    executor = KernelExecutorImpl(backend_registry)

    return DynamicDispatcher(
        selection_engine=selection_engine,
        backend_registry=backend_registry,
        config=config,
        executor=executor,
        mvcc_cache=mvcc_cache,
    )
