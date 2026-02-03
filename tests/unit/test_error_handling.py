"""Error handling tests for LayerZero.

Tests for proper error handling, circuit breaker, and recovery.
"""
from __future__ import annotations

import logging
import time
import pytest
import torch
from unittest.mock import patch, MagicMock
from typing import Any


class TestErrorHandling:
    """Tests for kernel error handling."""

    def test_kernel_runtime_error_handled(self) -> None:
        """Kernel runtime errors are handled gracefully."""
        from layerzero.registry.backend_registry import BackendRegistry, BackendState
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry(failure_threshold=3, cooldown_seconds=1.0)

        spec = BackendSpec(
            backend_id="error_prone_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="test_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        # Simulate a runtime error being reported
        registry.record_failure("error_prone_backend", "RuntimeError: CUDA error")

        health = registry.get_health("error_prone_backend")
        assert health is not None
        assert health.failure_count == 1
        assert health.total_failures == 1

    def test_circuit_breaker_opens(self) -> None:
        """Circuit breaker opens after consecutive failures."""
        from layerzero.registry.backend_registry import BackendRegistry, BackendState
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry(failure_threshold=3, cooldown_seconds=60.0)

        spec = BackendSpec(
            backend_id="flaky_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="test_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        # Should start healthy
        assert registry.is_available("flaky_backend")

        # Record failures up to threshold
        for i in range(3):
            registry.record_failure("flaky_backend", f"Error {i+1}")

        # Circuit should be open (backend unavailable)
        assert not registry.is_available("flaky_backend")

        health = registry.get_health("flaky_backend")
        assert health is not None
        assert health.state == BackendState.UNHEALTHY

    def test_circuit_breaker_recovery(self) -> None:
        """Circuit breaker recovers after cooldown."""
        from layerzero.registry.backend_registry import BackendRegistry, BackendState
        from layerzero.models.backend_spec import BackendSpec

        # Short cooldown for testing
        registry = BackendRegistry(failure_threshold=2, cooldown_seconds=0.1)

        spec = BackendSpec(
            backend_id="recovering_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="test_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        # Open circuit
        registry.record_failure("recovering_backend", "Error 1")
        registry.record_failure("recovering_backend", "Error 2")

        assert not registry.is_available("recovering_backend")

        # Wait for cooldown
        time.sleep(0.15)

        # Should be in cooldown state (ready for retry)
        health = registry.get_health("recovering_backend")
        assert health is not None
        assert health.state == BackendState.COOLDOWN

        # Record success to fully recover
        registry.record_success("recovering_backend")

        assert registry.is_available("recovering_backend")
        health = registry.get_health("recovering_backend")
        assert health.state == BackendState.HEALTHY

    def test_error_logged_with_context(self, caplog: pytest.LogCaptureFixture) -> None:
        """Errors are logged with full context."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()

        spec = BackendSpec(
            backend_id="logged_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="test_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        with caplog.at_level(logging.DEBUG):
            registry.record_failure(
                "logged_backend",
                "RuntimeError: CUDA out of memory"
            )

        # Error should be trackable via health metrics
        health = registry.get_health("logged_backend")
        assert health is not None
        assert health.total_failures == 1

    def test_fallback_after_error(self) -> None:
        """Fallback is used after kernel error."""
        from layerzero.selection.engine import SelectionEngine
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform, Layout
        from layerzero.device import GPUGeneration
        from layerzero.policy.policy import Policy

        kernel_registry = KernelRegistry()
        backend_registry = BackendRegistry(failure_threshold=1, cooldown_seconds=60.0)

        # Register primary kernel with backend
        primary_spec = KernelSpec(
            kernel_id="primary_kernel",
            operation="attention.causal",
            source="primary_backend",
            version="1.0",
            priority=100,
        )
        kernel_registry.register(primary_spec)

        # Register fallback kernel
        fallback_spec = KernelSpec(
            kernel_id="fallback_kernel",
            operation="attention.causal",
            source="fallback_backend",
            version="1.0",
            priority=10,
        )
        kernel_registry.register(fallback_spec)

        policy = Policy(version="1.0", locks=(), allows=(), denies=(), boosts=())
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            policy=policy,
        )

        device_spec = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="Test GPU",
            sm_version=(8, 0),
            gpu_generation=GPUGeneration.AMPERE,
            tensor_core_gen=3,
            total_memory_bytes=16 * 1024**3,
            available_memory_bytes=12 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=164,
            cuda_version="12.4",
            driver_version="550.54",
        )

        context = SelectionContext(
            device=device_spec,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=2,
            seq_len_q=64,
            seq_len_k=64,
            num_heads=4,
            head_dim=64,
            layout=Layout.BSHD,
        )

        # First selection - should get primary kernel
        plan1 = engine.select(context)
        assert plan1.kernel_id == "primary_kernel"


class TestErrorRecovery:
    """Tests for error recovery mechanisms."""

    def test_success_resets_failure_count(self) -> None:
        """Successful operation resets failure count."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry(failure_threshold=5)

        spec = BackendSpec(
            backend_id="test_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="test_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        # Record some failures
        for _ in range(3):
            registry.record_failure("test_backend", "Error")

        health = registry.get_health("test_backend")
        assert health.failure_count == 3

        # Record success
        registry.record_success("test_backend")

        health = registry.get_health("test_backend")
        assert health.failure_count == 0

    def test_intermittent_failures_dont_trigger_breaker(self) -> None:
        """Intermittent failures with successes don't trigger breaker."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry(failure_threshold=3)

        spec = BackendSpec(
            backend_id="intermittent_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="test_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        # Intermittent pattern: fail, fail, success, fail, fail, success
        for _ in range(3):
            registry.record_failure("intermittent_backend", "Error 1")
            registry.record_failure("intermittent_backend", "Error 2")
            registry.record_success("intermittent_backend")

        # Should still be available
        assert registry.is_available("intermittent_backend")


class TestErrorTypes:
    """Tests for different error type handling."""

    def test_cuda_oom_handling(self) -> None:
        """CUDA OOM errors are tracked."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()

        spec = BackendSpec(
            backend_id="memory_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="test_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        registry.record_failure(
            "memory_backend",
            "CUDA out of memory. Tried to allocate 2.00 GiB"
        )

        health = registry.get_health("memory_backend")
        assert health.total_failures == 1

    def test_numerical_error_handling(self) -> None:
        """Numerical errors (NaN, Inf) are tracked."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()

        spec = BackendSpec(
            backend_id="numerical_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="test_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        registry.record_failure(
            "numerical_backend",
            "Output contains NaN values"
        )

        health = registry.get_health("numerical_backend")
        assert health.total_failures == 1

    def test_timeout_handling(self) -> None:
        """Timeout errors are tracked."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()

        spec = BackendSpec(
            backend_id="slow_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="test_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        registry.record_failure(
            "slow_backend",
            "TimeoutError: Kernel execution timed out"
        )

        health = registry.get_health("slow_backend")
        assert health.total_failures == 1


class TestHealthMonitoring:
    """Tests for health monitoring system."""

    def test_health_statistics(self) -> None:
        """Health statistics are accurate."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()

        spec = BackendSpec(
            backend_id="stats_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="test_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        # Record mixed results
        for _ in range(10):
            registry.record_success("stats_backend")
        for _ in range(2):
            registry.record_failure("stats_backend", "Error")
        for _ in range(3):
            registry.record_success("stats_backend")

        health = registry.get_health("stats_backend")

        assert health.total_requests == 15
        assert health.total_failures == 2
        assert health.failure_count == 0  # Reset by last successes

    def test_multiple_backends_independent(self) -> None:
        """Multiple backends have independent health tracking."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry(failure_threshold=3)

        # Register two backends
        for backend_id in ["backend_a", "backend_b"]:
            spec = BackendSpec(
                backend_id=backend_id,
                version="1.0.0",
                installed=True,
                healthy=True,
                import_error=None,
                module_name="test_module",
                entry_point=None,
                supported_operations=frozenset(["attention"]),
                capabilities_schema_version="1.0",
            )
            registry.register(spec)

        # Fail backend_a
        for _ in range(5):
            registry.record_failure("backend_a", "Error")

        # backend_a should be unavailable
        assert not registry.is_available("backend_a")

        # backend_b should still be available
        assert registry.is_available("backend_b")

        health_a = registry.get_health("backend_a")
        health_b = registry.get_health("backend_b")

        assert health_a.total_failures == 5
        assert health_b.total_failures == 0
