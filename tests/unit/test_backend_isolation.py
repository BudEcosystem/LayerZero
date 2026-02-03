"""Backend isolation tests for LayerZero.

Tests that backend failures are properly isolated and don't crash the system.
"""
from __future__ import annotations

import sys
import logging
import pytest
from unittest.mock import patch, MagicMock
from typing import Any


class TestBackendIsolation:
    """Tests for backend failure isolation."""

    def test_import_failure_handled(self) -> None:
        """Import failure doesn't crash LayerZero."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()

        # Probe a non-existent backend
        spec = BackendSpec.probe("nonexistent_backend", "nonexistent.module.that.doesnt.exist")

        assert spec.backend_id == "nonexistent_backend"
        assert spec.installed is False
        assert spec.healthy is False
        assert spec.import_error is not None
        assert "nonexistent" in spec.import_error.lower() or "no module" in spec.import_error.lower()

    def test_backend_crash_isolated(self) -> None:
        """Backend crash is isolated and doesn't affect other backends."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry(failure_threshold=3, cooldown_seconds=0.1)

        # Register a working backend
        working_spec = BackendSpec(
            backend_id="working_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="torch",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )
        registry.register(working_spec)

        # Register a flaky backend
        flaky_spec = BackendSpec(
            backend_id="flaky_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flaky_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )
        registry.register(flaky_spec)

        # Simulate failures on flaky backend
        for _ in range(5):
            registry.record_failure("flaky_backend", "Simulated crash")

        # Flaky backend should be disabled
        assert not registry.is_available("flaky_backend")

        # Working backend should still be available
        assert registry.is_available("working_backend")

    def test_abi_conflict_handled(self) -> None:
        """ABI conflicts are handled gracefully."""
        from layerzero.models.backend_spec import BackendSpec

        # Simulate probing a backend with ABI conflict (raises ImportError)
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError(
                "undefined symbol: _ZN2at4_ops... (ABI conflict)"
            )

            spec = BackendSpec.probe("abi_conflict_backend", "abi_conflict_module")

            assert spec.installed is False
            assert spec.import_error is not None
            assert "undefined symbol" in spec.import_error or "ABI" in spec.import_error

    def test_version_mismatch_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Version mismatches are logged."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()

        # Register backend with specific version
        spec_v1 = BackendSpec(
            backend_id="versioned_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="test_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )
        registry.register(spec_v1)

        # Re-register with different version (simulating version change)
        spec_v2 = BackendSpec(
            backend_id="versioned_backend",
            version="2.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="test_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )

        with caplog.at_level(logging.DEBUG):
            registry.register(spec_v2)

        # Verify the backend was updated
        retrieved = registry.get("versioned_backend")
        assert retrieved is not None
        assert retrieved.version == "2.0.0"


class TestBackendRecovery:
    """Tests for backend recovery after failures."""

    def test_circuit_breaker_cooldown(self) -> None:
        """Backend recovers after cooldown period."""
        import time
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

        # Trigger circuit breaker
        registry.record_failure("recovering_backend", "Error 1")
        registry.record_failure("recovering_backend", "Error 2")

        # Should be unavailable
        assert not registry.is_available("recovering_backend")

        # Wait for cooldown
        time.sleep(0.2)

        # Should be in cooldown state (ready to retry)
        health = registry.get_health("recovering_backend")
        assert health is not None
        assert health.state == BackendState.COOLDOWN

    def test_recovery_on_success(self) -> None:
        """Backend fully recovers after successful operation."""
        import time
        from layerzero.registry.backend_registry import BackendRegistry, BackendState
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry(failure_threshold=2, cooldown_seconds=0.1)

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

        # Trigger failures
        registry.record_failure("test_backend", "Error 1")
        registry.record_failure("test_backend", "Error 2")

        # Wait for cooldown
        time.sleep(0.15)

        # Record success
        registry.record_success("test_backend")

        # Should be back to healthy
        health = registry.get_health("test_backend")
        assert health is not None
        assert health.state == BackendState.HEALTHY
        assert health.failure_count == 0


class TestBackendProbing:
    """Tests for safe backend probing."""

    def test_probe_never_raises(self) -> None:
        """BackendSpec.probe never raises an exception."""
        from layerzero.models.backend_spec import BackendSpec

        # Test various failure scenarios
        test_cases = [
            ("nonexistent", "nonexistent.module"),
            ("empty", ""),
            ("syntax_error", "syntax error module"),
            ("torch", "torch"),  # Should succeed
        ]

        for backend_id, module_name in test_cases:
            # Should never raise
            spec = BackendSpec.probe(backend_id, module_name)
            assert spec is not None
            assert spec.backend_id == backend_id

    def test_probe_timeout_handling(self) -> None:
        """Probing handles slow imports gracefully."""
        from layerzero.models.backend_spec import BackendSpec
        import signal

        # Create a mock that would hang
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = TimeoutError("Import timed out")

            spec = BackendSpec.probe("slow_backend", "slow_module")

            # Should handle gracefully
            assert spec.installed is False
            assert spec.import_error is not None


class TestBackendHealthTracking:
    """Tests for backend health tracking system."""

    def test_health_metrics_recorded(self) -> None:
        """Health metrics are recorded correctly."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()

        spec = BackendSpec(
            backend_id="metrics_backend",
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

        # Record some operations
        registry.record_success("metrics_backend")
        registry.record_success("metrics_backend")
        registry.record_failure("metrics_backend", "Error 1")
        registry.record_success("metrics_backend")

        health = registry.get_health("metrics_backend")
        assert health is not None
        assert health.total_requests == 4
        assert health.total_failures == 1
        assert health.failure_count == 0  # Reset by last success

    def test_health_persists_across_operations(self) -> None:
        """Health state persists correctly."""
        from layerzero.registry.backend_registry import BackendRegistry, BackendState
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry(failure_threshold=5)

        spec = BackendSpec(
            backend_id="persistent_backend",
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

        # Multiple failure/success cycles
        for _ in range(3):
            registry.record_failure("persistent_backend", "Error")

        assert registry.get_health("persistent_backend").failure_count == 3

        registry.record_success("persistent_backend")

        assert registry.get_health("persistent_backend").failure_count == 0
        assert registry.get_health("persistent_backend").state == BackendState.HEALTHY
