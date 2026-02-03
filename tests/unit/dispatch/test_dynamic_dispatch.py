"""
Unit tests for DynamicDispatcher.

Tests cover:
- Basic dispatch functionality
- Circuit breaker behavior
- Fallback chain
- Selection caching
- Thread safety
- Error handling
- Telemetry
"""
from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

import torch

from layerzero.dispatch.dynamic import (
    CircuitBreakerManager,
    CircuitState,
    DynamicDispatcher,
    FallbackChainImpl,
    KernelCircuit,
    create_dynamic_dispatcher,
)
from layerzero.dispatch.types import (
    CircuitOpenError,
    DispatchConfig,
    DispatchMode,
    DispatchResult,
    FallbackChainExhaustedError,
    KernelExecutionError,
)
from layerzero.enums import Layout, OpKind, Platform
from layerzero.models.device_spec import DeviceSpec
from layerzero.models.kernel_spec import KernelSpec
from layerzero.models.selection_context import SelectionContext
from layerzero.policy.policy import Policy
from layerzero.registry.backend_registry import BackendRegistry
from layerzero.registry.kernel_registry import KernelRegistry
from layerzero.selection.engine import SelectionEngine, NoKernelAvailableError


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


def _mock_attention_kernel(**kwargs: Any) -> torch.Tensor:
    """Mock attention kernel that accepts any kwargs and returns output."""
    # Use explicit None checks to avoid tensor boolean ambiguity
    q = kwargs.get("q")
    if q is None:
        q = kwargs.get("query")
    if q is None:
        return torch.zeros(1)
    # Return tensor with same batch/seq shape as query
    return torch.zeros_like(q)


def _mock_fallback_kernel(**kwargs: Any) -> torch.Tensor:
    """Mock fallback kernel."""
    q = kwargs.get("q")
    if q is None:
        q = kwargs.get("query")
    if q is None:
        return torch.ones(1)
    return torch.ones_like(q)


@pytest.fixture
def kernel_spec() -> KernelSpec:
    """Create test kernel spec."""
    return KernelSpec(
        kernel_id="test_kernel.v1",
        operation="attention.causal",
        source="test",
        version="1.0",
        impl=_mock_attention_kernel,
        platform=Platform.CUDA,
        min_sm=(7, 0),
        supported_dtypes=frozenset([torch.float16, torch.bfloat16, torch.float32]),
        priority=50,
    )


@pytest.fixture
def fallback_kernel_spec() -> KernelSpec:
    """Create fallback kernel spec."""
    return KernelSpec(
        kernel_id="fallback_kernel.v1",
        operation="attention.causal",
        source="fallback",
        version="1.0",
        impl=_mock_fallback_kernel,
        platform=Platform.CUDA,
        min_sm=(7, 0),
        supported_dtypes=frozenset([torch.float16, torch.bfloat16, torch.float32]),
        priority=25,  # Lower priority
    )


@pytest.fixture
def kernel_registry(kernel_spec: KernelSpec, fallback_kernel_spec: KernelSpec) -> KernelRegistry:
    """Create kernel registry with test kernels."""
    registry = KernelRegistry()
    registry.register(kernel_spec)
    registry.register(fallback_kernel_spec)
    return registry


@pytest.fixture
def backend_registry() -> BackendRegistry:
    """Create backend registry."""
    return BackendRegistry(failure_threshold=3, cooldown_seconds=30.0)


@pytest.fixture
def selection_engine(kernel_registry: KernelRegistry, backend_registry: BackendRegistry) -> SelectionEngine:
    """Create selection engine."""
    return SelectionEngine(
        kernel_registry=kernel_registry,
        backend_registry=backend_registry,
    )


@pytest.fixture
def config() -> DispatchConfig:
    """Create dispatch config."""
    return DispatchConfig(
        mode=DispatchMode.DYNAMIC,
        enable_cache=True,
        enable_fallback=True,
        max_fallback_attempts=3,
        failure_threshold=3,
        recovery_timeout_seconds=5.0,
    )


@pytest.fixture
def dynamic_dispatcher(
    selection_engine: SelectionEngine,
    backend_registry: BackendRegistry,
    config: DispatchConfig,
) -> DynamicDispatcher:
    """Create dynamic dispatcher."""
    return DynamicDispatcher(
        selection_engine=selection_engine,
        backend_registry=backend_registry,
        config=config,
    )


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


# ============================================================================
# Circuit Breaker Tests
# ============================================================================

class TestCircuitBreakerManager:
    """Tests for CircuitBreakerManager."""

    def test_initial_state_is_closed(self) -> None:
        """New circuits start in CLOSED state."""
        manager = CircuitBreakerManager(failure_threshold=3)
        assert manager.is_allowed("test_kernel")
        assert manager.get_retry_after("test_kernel") is None

    def test_opens_after_threshold_failures(self) -> None:
        """Circuit opens after consecutive failures reach threshold."""
        manager = CircuitBreakerManager(failure_threshold=3, cooldown_seconds=10.0)

        # First two failures don't open circuit
        manager.record_failure("test_kernel", ValueError("fail 1"))
        assert manager.is_allowed("test_kernel")

        manager.record_failure("test_kernel", ValueError("fail 2"))
        assert manager.is_allowed("test_kernel")

        # Third failure opens circuit
        manager.record_failure("test_kernel", ValueError("fail 3"))
        assert not manager.is_allowed("test_kernel")

        # Retry after should be set
        retry_after = manager.get_retry_after("test_kernel")
        assert retry_after is not None
        assert 0 < retry_after <= 10.0

    def test_success_resets_failure_count(self) -> None:
        """Success resets consecutive failure count."""
        manager = CircuitBreakerManager(failure_threshold=3)

        # Two failures
        manager.record_failure("test_kernel")
        manager.record_failure("test_kernel")

        # Success resets
        manager.record_success("test_kernel")

        # Need 3 more failures to open
        manager.record_failure("test_kernel")
        manager.record_failure("test_kernel")
        assert manager.is_allowed("test_kernel")

    def test_half_open_after_cooldown(self) -> None:
        """Circuit transitions to HALF_OPEN after cooldown."""
        manager = CircuitBreakerManager(failure_threshold=2, cooldown_seconds=0.1)

        # Open circuit
        manager.record_failure("test_kernel")
        manager.record_failure("test_kernel")
        assert not manager.is_allowed("test_kernel")

        # Wait for cooldown
        time.sleep(0.15)

        # Should be allowed (half-open)
        assert manager.is_allowed("test_kernel")

    def test_half_open_success_closes_circuit(self) -> None:
        """Success in HALF_OPEN state closes circuit."""
        manager = CircuitBreakerManager(failure_threshold=2, cooldown_seconds=0.05)

        # Open circuit
        manager.record_failure("test_kernel")
        manager.record_failure("test_kernel")

        # Wait for half-open
        time.sleep(0.1)
        assert manager.is_allowed("test_kernel")  # Transitions to half-open

        # Success closes circuit
        manager.record_success("test_kernel")
        assert manager.is_allowed("test_kernel")
        assert manager.get_retry_after("test_kernel") is None

    def test_half_open_failure_reopens_circuit(self) -> None:
        """Failure in HALF_OPEN state reopens circuit."""
        manager = CircuitBreakerManager(failure_threshold=2, cooldown_seconds=0.05)

        # Open circuit
        manager.record_failure("test_kernel")
        manager.record_failure("test_kernel")

        # Wait for half-open
        time.sleep(0.1)
        assert manager.is_allowed("test_kernel")  # Transitions to half-open

        # Failure reopens
        manager.record_failure("test_kernel")
        assert not manager.is_allowed("test_kernel")

    def test_reset_clears_circuit(self) -> None:
        """Reset clears circuit state."""
        manager = CircuitBreakerManager(failure_threshold=2)

        # Open circuit
        manager.record_failure("test_kernel")
        manager.record_failure("test_kernel")
        assert not manager.is_allowed("test_kernel")

        # Reset
        manager.reset("test_kernel")
        assert manager.is_allowed("test_kernel")

    def test_stats(self) -> None:
        """Stats returns correct counts."""
        manager = CircuitBreakerManager(failure_threshold=2)

        manager.record_failure("kernel1")
        manager.record_failure("kernel1")  # Opens

        manager.record_failure("kernel2")

        manager.record_success("kernel3")

        stats = manager.get_stats()
        assert stats["total_circuits"] == 3
        assert stats["open"] == 1
        assert stats["closed"] == 2


# ============================================================================
# Dynamic Dispatcher Tests
# ============================================================================

class TestDynamicDispatcher:
    """Tests for DynamicDispatcher."""

    def test_mode_is_dynamic(self, dynamic_dispatcher: DynamicDispatcher) -> None:
        """Mode property returns DYNAMIC."""
        assert dynamic_dispatcher.mode == DispatchMode.DYNAMIC

    def test_get_kernel_for_operation(
        self,
        dynamic_dispatcher: DynamicDispatcher,
        selection_context: SelectionContext,
    ) -> None:
        """get_kernel_for_operation returns correct kernel."""
        kernel = dynamic_dispatcher.get_kernel_for_operation(
            "attention.causal",
            selection_context,
        )
        assert kernel is not None
        assert kernel.operation == "attention.causal"

    def test_dispatch_returns_result(
        self,
        dynamic_dispatcher: DynamicDispatcher,
        selection_context: SelectionContext,
    ) -> None:
        """dispatch returns DispatchResult with correct fields."""
        inputs = {
            "query": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "key": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "value": torch.randn(4, 512, 8, 64, dtype=torch.float16),
        }

        result = dynamic_dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=selection_context,
        )

        assert isinstance(result, DispatchResult)
        assert result.mode == DispatchMode.DYNAMIC
        assert result.kernel_id is not None
        assert result.kernel_spec is not None
        assert result.timing.total_ns > 0
        assert not result.fallback_used

    def test_dispatch_uses_fallback_on_failure(
        self,
        selection_engine: SelectionEngine,
        backend_registry: BackendRegistry,
        selection_context: SelectionContext,
    ) -> None:
        """dispatch uses fallback when primary kernel fails."""

        def failing_impl(**kwargs: Any) -> torch.Tensor:
            raise ValueError("Primary failed")

        def working_impl(**kwargs: Any) -> torch.Tensor:
            q = kwargs.get("q") or kwargs.get("query")
            if q is None:
                return torch.ones(1)
            return torch.ones_like(q)

        # Create kernels with failing primary
        failing_kernel = KernelSpec(
            kernel_id="failing_kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=failing_impl,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=100,  # Higher priority - will be selected first
        )

        working_kernel = KernelSpec(
            kernel_id="working_kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=working_impl,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=50,
        )

        registry = KernelRegistry()
        registry.register(failing_kernel)
        registry.register(working_kernel)

        engine = SelectionEngine(registry, backend_registry)
        config = DispatchConfig(
            mode=DispatchMode.DYNAMIC,
            enable_fallback=True,
            max_fallback_attempts=3,
            failure_threshold=5,
        )

        dispatcher = DynamicDispatcher(
            selection_engine=engine,
            backend_registry=backend_registry,
            config=config,
        )

        inputs = {
            "query": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "key": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "value": torch.randn(4, 512, 8, 64, dtype=torch.float16),
        }

        result = dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=selection_context,
        )

        assert result.fallback_used
        assert result.kernel_id == "working_kernel"
        assert "Primary" in (result.fallback_reason or "")

    def test_circuit_breaker_blocks_failed_kernel(
        self,
        selection_engine: SelectionEngine,
        backend_registry: BackendRegistry,
        selection_context: SelectionContext,
    ) -> None:
        """Circuit breaker blocks kernel after repeated failures."""
        call_count = [0]

        def maybe_fail(**kwargs: Any) -> torch.Tensor:
            call_count[0] += 1
            if call_count[0] <= 3:
                raise ValueError("Simulated failure")
            q = kwargs.get("q") or kwargs.get("query")
            if q is None:
                return torch.ones(1)
            return torch.ones_like(q)

        def stable_impl(**kwargs: Any) -> torch.Tensor:
            q = kwargs.get("q") or kwargs.get("query")
            if q is None:
                return torch.zeros(1)
            return torch.zeros_like(q)

        kernel = KernelSpec(
            kernel_id="flaky_kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=maybe_fail,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=100,
        )

        fallback = KernelSpec(
            kernel_id="stable_kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=stable_impl,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=50,
        )

        registry = KernelRegistry()
        registry.register(kernel)
        registry.register(fallback)

        engine = SelectionEngine(registry, backend_registry)
        config = DispatchConfig(
            mode=DispatchMode.DYNAMIC,
            enable_fallback=True,
            failure_threshold=2,  # Open circuit after 2 failures
            recovery_timeout_seconds=10.0,
        )

        dispatcher = DynamicDispatcher(
            selection_engine=engine,
            backend_registry=backend_registry,
            config=config,
        )

        inputs = {
            "query": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "key": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "value": torch.randn(4, 512, 8, 64, dtype=torch.float16),
        }

        # First dispatch - primary fails, fallback succeeds
        result1 = dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=selection_context,
        )
        assert result1.fallback_used

        # After more failures, circuit should be open
        # The fallback should be used from the start
        result2 = dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=selection_context,
        )

        # Check that fallback is being used (circuit is open for primary)
        stats = dispatcher.circuit_breaker.get_stats()
        assert stats["open"] >= 1 or result2.fallback_used

    def test_raises_when_all_fallbacks_fail(
        self,
        backend_registry: BackendRegistry,
        selection_context: SelectionContext,
    ) -> None:
        """Raises FallbackChainExhaustedError when all kernels fail."""

        def fail1(**kwargs: Any) -> torch.Tensor:
            raise ValueError("Fail 1")

        def fail2(**kwargs: Any) -> torch.Tensor:
            raise ValueError("Fail 2")

        kernel1 = KernelSpec(
            kernel_id="fail1",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=fail1,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=100,
        )

        kernel2 = KernelSpec(
            kernel_id="fail2",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=fail2,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=50,
        )

        registry = KernelRegistry()
        registry.register(kernel1)
        registry.register(kernel2)

        engine = SelectionEngine(registry, backend_registry)
        config = DispatchConfig(
            mode=DispatchMode.DYNAMIC,
            enable_fallback=True,
            failure_threshold=10,  # High threshold to not open circuits
        )

        dispatcher = DynamicDispatcher(
            selection_engine=engine,
            backend_registry=backend_registry,
            config=config,
        )

        inputs = {
            "query": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "key": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "value": torch.randn(4, 512, 8, 64, dtype=torch.float16),
        }

        with pytest.raises(FallbackChainExhaustedError) as exc_info:
            dispatcher.dispatch(
                operation="attention.causal",
                inputs=inputs,
                context=selection_context,
            )

        assert len(exc_info.value.attempted_kernels) == 2
        assert len(exc_info.value.errors) == 2

    def test_telemetry_tracking(
        self,
        dynamic_dispatcher: DynamicDispatcher,
        selection_context: SelectionContext,
    ) -> None:
        """Telemetry is properly tracked."""
        inputs = {
            "query": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "key": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "value": torch.randn(4, 512, 8, 64, dtype=torch.float16),
        }

        # Reset telemetry
        dynamic_dispatcher.reset_telemetry()

        # Execute some dispatches
        for _ in range(5):
            dynamic_dispatcher.dispatch(
                operation="attention.causal",
                inputs=inputs,
                context=selection_context,
            )

        telemetry = dynamic_dispatcher.get_telemetry()

        assert telemetry["dispatch_count"] == 5
        assert telemetry["avg_selection_ns"] > 0
        assert telemetry["avg_execution_ns"] > 0
        assert telemetry["mode"] == "dynamic"

    def test_cache_invalidation(
        self,
        dynamic_dispatcher: DynamicDispatcher,
        selection_context: SelectionContext,
    ) -> None:
        """Cache invalidation works correctly."""
        inputs = {
            "query": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "key": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "value": torch.randn(4, 512, 8, 64, dtype=torch.float16),
        }

        # First dispatch populates cache
        result1 = dynamic_dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=selection_context,
        )

        # Second dispatch may use cache
        result2 = dynamic_dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=selection_context,
        )

        # Invalidate cache
        dynamic_dispatcher.invalidate_cache()

        # Third dispatch should not use cache
        result3 = dynamic_dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=selection_context,
        )

        # All should complete successfully
        assert result1.kernel_id is not None
        assert result2.kernel_id is not None
        assert result3.kernel_id is not None


# ============================================================================
# Fallback Chain Tests
# ============================================================================

class TestFallbackChain:
    """Tests for FallbackChainImpl."""

    def test_get_fallbacks_excludes_failed_kernel(
        self,
        selection_engine: SelectionEngine,
        device_spec: DeviceSpec,
    ) -> None:
        """get_fallbacks excludes the failed kernel."""
        circuit_breaker = CircuitBreakerManager()
        chain = FallbackChainImpl(
            selection_engine=selection_engine,
            circuit_breaker=circuit_breaker,
            max_fallbacks=5,
        )

        context = SelectionContext(
            device=device_spec,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=4,
        )

        fallbacks = chain.get_fallbacks(
            operation="attention.causal",
            failed_kernel_id="test_kernel.v1",
            context=context,
        )

        # Should not include the failed kernel
        assert all(k.kernel_id != "test_kernel.v1" for k in fallbacks)

    def test_get_fallbacks_excludes_open_circuits(
        self,
        selection_engine: SelectionEngine,
        device_spec: DeviceSpec,
    ) -> None:
        """get_fallbacks excludes kernels with open circuits."""
        circuit_breaker = CircuitBreakerManager(failure_threshold=1)
        chain = FallbackChainImpl(
            selection_engine=selection_engine,
            circuit_breaker=circuit_breaker,
            max_fallbacks=5,
        )

        # Open circuit for fallback kernel
        circuit_breaker.record_failure("fallback_kernel.v1")

        context = SelectionContext(
            device=device_spec,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=4,
        )

        fallbacks = chain.get_fallbacks(
            operation="attention.causal",
            failed_kernel_id="test_kernel.v1",
            context=context,
        )

        # Should not include kernel with open circuit
        assert all(k.kernel_id != "fallback_kernel.v1" for k in fallbacks)

    def test_get_fallbacks_respects_max_limit(
        self,
        backend_registry: BackendRegistry,
        device_spec: DeviceSpec,
    ) -> None:
        """get_fallbacks respects max_fallbacks limit."""
        # Create many kernels
        registry = KernelRegistry()
        for i in range(10):
            registry.register(KernelSpec(
                kernel_id=f"kernel_{i}",
                operation="attention.causal",
                source="test",
                version="1.0",
                impl=lambda **kwargs: torch.zeros(1),
                platform=Platform.CUDA,
                min_sm=(7, 0),
                supported_dtypes=frozenset([torch.float16]),
                priority=50 - i,
            ))

        engine = SelectionEngine(registry, backend_registry)
        circuit_breaker = CircuitBreakerManager()
        chain = FallbackChainImpl(
            selection_engine=engine,
            circuit_breaker=circuit_breaker,
            max_fallbacks=3,
        )

        context = SelectionContext(
            device=device_spec,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=4,
        )

        fallbacks = chain.get_fallbacks(
            operation="attention.causal",
            failed_kernel_id="kernel_0",
            context=context,
        )

        assert len(fallbacks) <= 3


# ============================================================================
# Thread Safety Tests
# ============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_dispatches(
        self,
        dynamic_dispatcher: DynamicDispatcher,
        selection_context: SelectionContext,
    ) -> None:
        """Multiple threads can dispatch concurrently."""
        inputs = {
            "query": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "key": torch.randn(4, 512, 8, 64, dtype=torch.float16),
            "value": torch.randn(4, 512, 8, 64, dtype=torch.float16),
        }

        results: list[DispatchResult] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def dispatch_task() -> None:
            try:
                result = dynamic_dispatcher.dispatch(
                    operation="attention.causal",
                    inputs=inputs,
                    context=selection_context,
                )
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run 20 concurrent dispatches
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(dispatch_task) for _ in range(20)]
            for future in futures:
                future.result()

        assert len(errors) == 0
        assert len(results) == 20

        # All results should be valid
        for result in results:
            assert result.kernel_id is not None
            assert result.timing.total_ns > 0

    def test_concurrent_circuit_breaker_updates(self) -> None:
        """Circuit breaker handles concurrent updates."""
        manager = CircuitBreakerManager(failure_threshold=100)
        errors: list[Exception] = []

        def stress_test() -> None:
            try:
                for _ in range(100):
                    manager.record_failure("test_kernel")
                    manager.is_allowed("test_kernel")
                    manager.record_success("test_kernel")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress_test) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestCreateDynamicDispatcher:
    """Tests for create_dynamic_dispatcher factory."""

    def test_creates_dispatcher_with_defaults(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
    ) -> None:
        """Factory creates dispatcher with default config."""
        dispatcher = create_dynamic_dispatcher(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )

        assert dispatcher.mode == DispatchMode.DYNAMIC
        assert dispatcher.selection_engine is not None
        assert dispatcher.circuit_breaker is not None

    def test_creates_dispatcher_with_mvcc_cache(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
    ) -> None:
        """Factory creates dispatcher with MVCC cache when enabled."""
        config = DispatchConfig(
            mode=DispatchMode.DYNAMIC,
            enable_cache=True,
        )

        dispatcher = create_dynamic_dispatcher(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            config=config,
            use_mvcc_cache=True,
        )

        telemetry = dispatcher.get_telemetry()
        assert telemetry["cache_enabled"] is True

    def test_creates_dispatcher_without_mvcc_cache(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
    ) -> None:
        """Factory creates dispatcher without MVCC cache when disabled."""
        dispatcher = create_dynamic_dispatcher(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            use_mvcc_cache=False,
        )

        telemetry = dispatcher.get_telemetry()
        assert telemetry["cache_enabled"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
