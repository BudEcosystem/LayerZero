"""
Test suite for BackendRegistry.

Tests backend registration, health tracking, and circuit breaker.
Following TDD methodology - tests define expected behavior.
"""
import pytest
import time
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor


class TestBackendRegistryCreation:
    """Test BackendRegistry construction."""

    def test_backend_registry_creation(self):
        """BackendRegistry can be created with defaults."""
        from layerzero.registry.backend_registry import BackendRegistry

        registry = BackendRegistry()
        assert len(registry.get_all()) == 0

    def test_backend_registry_custom_thresholds(self):
        """BackendRegistry accepts custom thresholds."""
        from layerzero.registry.backend_registry import BackendRegistry

        registry = BackendRegistry(
            failure_threshold=5,
            cooldown_seconds=120.0,
        )
        assert registry._failure_threshold == 5
        assert registry._cooldown_seconds == 120.0


class TestBackendRegistration:
    """Test backend registration."""

    def test_register_backend(self):
        """Register a backend successfully."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()
        spec = BackendSpec(
            backend_id="flash_attn",
            version="2.5.6",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(["attention.causal", "attention.full"]),
            capabilities_schema_version="1.0",
        )

        registry.register(spec)
        assert registry.get("flash_attn") is not None

    def test_register_updates_existing(self):
        """Registering existing backend updates it."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()
        spec1 = BackendSpec(
            backend_id="flash_attn",
            version="2.5.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )
        spec2 = BackendSpec(
            backend_id="flash_attn",
            version="2.6.0",  # Updated version
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )

        registry.register(spec1)
        registry.register(spec2)  # Should update, not error

        result = registry.get("flash_attn")
        assert result.version == "2.6.0"


class TestBackendProbe:
    """Test backend probe and registration."""

    def test_probe_and_register_installed(self):
        """Probe and register installed module."""
        from layerzero.registry.backend_registry import BackendRegistry

        registry = BackendRegistry()
        spec = registry.probe_and_register("json_module", "json")

        assert spec.backend_id == "json_module"
        assert spec.installed is True
        assert spec.healthy is True

    def test_probe_and_register_missing(self):
        """Probe and register missing module."""
        from layerzero.registry.backend_registry import BackendRegistry

        registry = BackendRegistry()
        spec = registry.probe_and_register("missing_module", "nonexistent_module_xyz")

        assert spec.backend_id == "missing_module"
        assert spec.installed is False
        assert spec.healthy is False
        assert spec.import_error is not None


class TestBackendAvailability:
    """Test backend availability checking."""

    def test_is_available_installed_healthy(self):
        """Installed and healthy backend is available."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()
        spec = BackendSpec(
            backend_id="flash_attn",
            version="2.5.6",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        assert registry.is_available("flash_attn") is True

    def test_is_available_not_installed(self):
        """Not installed backend is not available."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()
        spec = BackendSpec(
            backend_id="flash_attn",
            version="unknown",
            installed=False,
            healthy=False,
            import_error="ModuleNotFoundError",
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        assert registry.is_available("flash_attn") is False

    def test_is_available_unknown_backend(self):
        """Unknown backend is not available."""
        from layerzero.registry.backend_registry import BackendRegistry

        registry = BackendRegistry()
        assert registry.is_available("unknown_backend") is False

    def test_get_available_backends(self):
        """Get list of available backends."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()
        specs = [
            BackendSpec(
                backend_id="backend1",
                version="1.0.0",
                installed=True,
                healthy=True,
                import_error=None,
                module_name="mod1",
                entry_point=None,
                supported_operations=frozenset(),
                capabilities_schema_version="1.0",
            ),
            BackendSpec(
                backend_id="backend2",
                version="unknown",
                installed=False,
                healthy=False,
                import_error="Error",
                module_name="mod2",
                entry_point=None,
                supported_operations=frozenset(),
                capabilities_schema_version="1.0",
            ),
            BackendSpec(
                backend_id="backend3",
                version="2.0.0",
                installed=True,
                healthy=True,
                import_error=None,
                module_name="mod3",
                entry_point=None,
                supported_operations=frozenset(),
                capabilities_schema_version="1.0",
            ),
        ]
        for spec in specs:
            registry.register(spec)

        available = registry.get_available_backends()
        assert len(available) == 2
        assert all(b.installed and b.healthy for b in available)


class TestBackendHealthTracking:
    """Test backend health tracking."""

    def test_get_health(self):
        """Get health status for backend."""
        from layerzero.registry.backend_registry import BackendRegistry, BackendState
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()
        spec = BackendSpec(
            backend_id="flash_attn",
            version="2.5.6",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        health = registry.get_health("flash_attn")
        assert health is not None
        assert health.state == BackendState.HEALTHY
        assert health.failure_count == 0

    def test_get_health_unknown(self):
        """Get health for unknown backend returns None."""
        from layerzero.registry.backend_registry import BackendRegistry

        registry = BackendRegistry()
        health = registry.get_health("unknown")
        assert health is None

    def test_record_success_resets_failures(self):
        """Recording success resets failure count."""
        from layerzero.registry.backend_registry import BackendRegistry, BackendState
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry(failure_threshold=3)
        spec = BackendSpec(
            backend_id="flash_attn",
            version="2.5.6",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        # Record some failures
        registry.record_failure("flash_attn", "Error 1")
        registry.record_failure("flash_attn", "Error 2")

        health = registry.get_health("flash_attn")
        assert health.failure_count == 2

        # Record success
        registry.record_success("flash_attn")

        health = registry.get_health("flash_attn")
        assert health.failure_count == 0
        assert health.state == BackendState.HEALTHY


class TestBackendCircuitBreaker:
    """Test circuit breaker pattern."""

    def test_circuit_opens_after_threshold(self):
        """Circuit opens after failure threshold reached."""
        from layerzero.registry.backend_registry import BackendRegistry, BackendState
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry(failure_threshold=3)
        spec = BackendSpec(
            backend_id="flash_attn",
            version="2.5.6",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        # Initial state is healthy
        assert registry.get_health("flash_attn").state == BackendState.HEALTHY

        # Record failures up to threshold
        registry.record_failure("flash_attn", "Error 1")
        registry.record_failure("flash_attn", "Error 2")
        assert registry.get_health("flash_attn").state == BackendState.HEALTHY

        # Third failure opens circuit
        registry.record_failure("flash_attn", "Error 3")
        assert registry.get_health("flash_attn").state == BackendState.UNHEALTHY

    def test_circuit_disables_availability(self):
        """Open circuit makes backend unavailable."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry(failure_threshold=2)
        spec = BackendSpec(
            backend_id="flash_attn",
            version="2.5.6",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        assert registry.is_available("flash_attn") is True

        # Open circuit
        registry.record_failure("flash_attn", "Error 1")
        registry.record_failure("flash_attn", "Error 2")

        assert registry.is_available("flash_attn") is False

    def test_circuit_recovery_after_cooldown(self):
        """Circuit allows retry after cooldown."""
        from layerzero.registry.backend_registry import BackendRegistry, BackendState
        from layerzero.models.backend_spec import BackendSpec

        # Very short cooldown for testing
        registry = BackendRegistry(failure_threshold=2, cooldown_seconds=0.1)
        spec = BackendSpec(
            backend_id="flash_attn",
            version="2.5.6",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        # Open circuit
        registry.record_failure("flash_attn", "Error 1")
        registry.record_failure("flash_attn", "Error 2")
        assert registry.get_health("flash_attn").state == BackendState.UNHEALTHY

        # Wait for cooldown
        time.sleep(0.15)

        # Should be in cooldown state (ready to retry)
        health = registry.get_health("flash_attn")
        assert health.state == BackendState.COOLDOWN

    def test_circuit_closes_on_success_after_cooldown(self):
        """Circuit closes on success after cooldown."""
        from layerzero.registry.backend_registry import BackendRegistry, BackendState
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry(failure_threshold=2, cooldown_seconds=0.1)
        spec = BackendSpec(
            backend_id="flash_attn",
            version="2.5.6",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        # Open circuit
        registry.record_failure("flash_attn", "Error 1")
        registry.record_failure("flash_attn", "Error 2")

        # Wait for cooldown and record success
        time.sleep(0.15)
        registry.record_success("flash_attn")

        # Circuit should be closed
        health = registry.get_health("flash_attn")
        assert health.state == BackendState.HEALTHY
        assert registry.is_available("flash_attn") is True


class TestBackendManualControl:
    """Test manual backend control."""

    def test_mark_unhealthy(self):
        """Manually mark backend as unhealthy."""
        from layerzero.registry.backend_registry import BackendRegistry, BackendState
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()
        spec = BackendSpec(
            backend_id="flash_attn",
            version="2.5.6",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        registry.mark_unhealthy("flash_attn", "Manual disable for testing")

        health = registry.get_health("flash_attn")
        assert health.state == BackendState.UNHEALTHY
        assert registry.is_available("flash_attn") is False


class TestBackendRegistryClear:
    """Test registry clear operation."""

    def test_clear(self):
        """Clear removes all backends."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()
        spec = BackendSpec(
            backend_id="flash_attn",
            version="2.5.6",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)
        assert len(registry.get_all()) == 1

        registry.clear()
        assert len(registry.get_all()) == 0


class TestBackendRegistryThreadSafety:
    """Test thread safety of BackendRegistry."""

    def test_concurrent_health_updates(self):
        """Concurrent health updates are thread-safe."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry(failure_threshold=1000)  # High threshold
        spec = BackendSpec(
            backend_id="flash_attn",
            version="2.5.6",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )
        registry.register(spec)

        def update_health(i: int):
            if i % 2 == 0:
                registry.record_success("flash_attn")
            else:
                registry.record_failure("flash_attn", f"Error {i}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(update_health, range(100))

        # Should complete without errors
        health = registry.get_health("flash_attn")
        assert health.total_requests == 100
