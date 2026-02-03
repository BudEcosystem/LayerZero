"""
Unit tests for dispatch/circuit_breaker.py module.

Tests cover:
- CircuitState enum
- CircuitBreakerConfig validation
- CircuitStats tracking and calculations
- CircuitBreaker state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- CircuitBreaker thread safety
- CircuitBreaker listener notifications
- CircuitBreaker wrap() and protect() patterns
- CircuitBreakerRegistry management
- Global registry functions

All tests use pytest and cover normal operations, edge cases, and error conditions.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest

from layerzero.dispatch.circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitStats,
    CircuitBreaker,
    CircuitBreakerRegistry,
    get_global_circuit_registry,
    get_circuit,
)
from layerzero.dispatch.types import CircuitOpenError


# ============================================================================
# CircuitState Tests
# ============================================================================


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_enum_values_exist(self) -> None:
        """All expected circuit states exist."""
        assert hasattr(CircuitState, "CLOSED")
        assert hasattr(CircuitState, "OPEN")
        assert hasattr(CircuitState, "HALF_OPEN")

    def test_enum_values_are_unique(self) -> None:
        """All circuit state values are unique."""
        values = [CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN]
        assert len(values) == len(set(values))

    def test_enum_is_hashable(self) -> None:
        """CircuitState is hashable for use in dicts/sets."""
        states_dict = {CircuitState.CLOSED: "normal", CircuitState.OPEN: "blocked"}
        assert states_dict[CircuitState.CLOSED] == "normal"

    def test_enum_name_property(self) -> None:
        """CircuitState has correct name property."""
        assert CircuitState.CLOSED.name == "CLOSED"
        assert CircuitState.OPEN.name == "OPEN"
        assert CircuitState.HALF_OPEN.name == "HALF_OPEN"


# ============================================================================
# CircuitBreakerConfig Tests
# ============================================================================


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass."""

    def test_default_values(self) -> None:
        """Config has correct default values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.cooldown_seconds == 30.0
        assert config.half_open_max_calls == 3
        assert config.reset_timeout_seconds is None

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=3,
            cooldown_seconds=60.0,
            half_open_max_calls=5,
            reset_timeout_seconds=300.0,
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 3
        assert config.cooldown_seconds == 60.0
        assert config.half_open_max_calls == 5
        assert config.reset_timeout_seconds == 300.0

    def test_validation_failure_threshold_zero(self) -> None:
        """Config rejects failure_threshold < 1."""
        with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
            CircuitBreakerConfig(failure_threshold=0)

    def test_validation_failure_threshold_negative(self) -> None:
        """Config rejects negative failure_threshold."""
        with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
            CircuitBreakerConfig(failure_threshold=-5)

    def test_validation_success_threshold_zero(self) -> None:
        """Config rejects success_threshold < 1."""
        with pytest.raises(ValueError, match="success_threshold must be >= 1"):
            CircuitBreakerConfig(success_threshold=0)

    def test_validation_success_threshold_negative(self) -> None:
        """Config rejects negative success_threshold."""
        with pytest.raises(ValueError, match="success_threshold must be >= 1"):
            CircuitBreakerConfig(success_threshold=-1)

    def test_validation_cooldown_zero(self) -> None:
        """Config rejects cooldown_seconds <= 0."""
        with pytest.raises(ValueError, match="cooldown_seconds must be > 0"):
            CircuitBreakerConfig(cooldown_seconds=0.0)

    def test_validation_cooldown_negative(self) -> None:
        """Config rejects negative cooldown_seconds."""
        with pytest.raises(ValueError, match="cooldown_seconds must be > 0"):
            CircuitBreakerConfig(cooldown_seconds=-10.0)


# ============================================================================
# CircuitStats Tests
# ============================================================================


class TestCircuitStats:
    """Tests for CircuitStats dataclass."""

    def test_initial_values(self) -> None:
        """Stats start with zero counters."""
        stats = CircuitStats()
        assert stats.total_calls == 0
        assert stats.total_successes == 0
        assert stats.total_failures == 0
        assert stats.total_rejections == 0
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 0
        assert stats.last_failure_time is None
        assert stats.last_success_time is None
        assert stats.state_changes == 0

    def test_failure_rate_no_calls(self) -> None:
        """failure_rate returns 0.0 when no calls."""
        stats = CircuitStats()
        assert stats.failure_rate == 0.0

    def test_failure_rate_calculation(self) -> None:
        """failure_rate calculates correctly."""
        stats = CircuitStats()
        stats.total_calls = 100
        stats.total_failures = 25
        assert stats.failure_rate == 0.25

    def test_failure_rate_all_failures(self) -> None:
        """failure_rate returns 1.0 when all calls fail."""
        stats = CircuitStats()
        stats.total_calls = 10
        stats.total_failures = 10
        assert stats.failure_rate == 1.0

    def test_failure_rate_no_failures(self) -> None:
        """failure_rate returns 0.0 when no failures."""
        stats = CircuitStats()
        stats.total_calls = 100
        stats.total_failures = 0
        assert stats.failure_rate == 0.0

    def test_to_dict(self) -> None:
        """to_dict returns correct dictionary structure."""
        stats = CircuitStats()
        stats.total_calls = 100
        stats.total_successes = 80
        stats.total_failures = 15
        stats.total_rejections = 5
        stats.consecutive_failures = 2
        stats.consecutive_successes = 0
        stats.state_changes = 3

        d = stats.to_dict()

        assert d["total_calls"] == 100
        assert d["total_successes"] == 80
        assert d["total_failures"] == 15
        assert d["total_rejections"] == 5
        assert d["consecutive_failures"] == 2
        assert d["consecutive_successes"] == 0
        assert d["failure_rate"] == 0.15
        assert d["state_changes"] == 3


# ============================================================================
# CircuitBreaker Basic Tests
# ============================================================================


class TestCircuitBreakerBasic:
    """Basic tests for CircuitBreaker."""

    def test_initialization_defaults(self) -> None:
        """Circuit initializes with default config."""
        circuit = CircuitBreaker("test_circuit")
        assert circuit.name == "test_circuit"
        assert circuit.state == CircuitState.CLOSED
        assert circuit.stats.total_calls == 0

    def test_initialization_custom_config(self) -> None:
        """Circuit accepts custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            cooldown_seconds=10.0,
        )
        circuit = CircuitBreaker("test", config=config)
        assert circuit._config.failure_threshold == 3
        assert circuit._config.cooldown_seconds == 10.0

    def test_name_property(self) -> None:
        """name property returns circuit name."""
        circuit = CircuitBreaker("my_circuit")
        assert circuit.name == "my_circuit"

    def test_state_property(self) -> None:
        """state property returns current state."""
        circuit = CircuitBreaker("test")
        assert circuit.state == CircuitState.CLOSED

    def test_stats_property(self) -> None:
        """stats property returns circuit statistics."""
        circuit = CircuitBreaker("test")
        assert isinstance(circuit.stats, CircuitStats)


# ============================================================================
# CircuitBreaker State Transitions Tests
# ============================================================================


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    def test_can_execute_when_closed(self) -> None:
        """can_execute returns True in CLOSED state."""
        circuit = CircuitBreaker("test")
        assert circuit.can_execute() is True

    def test_closed_to_open_after_threshold(self) -> None:
        """Circuit opens after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=10.0)
        circuit = CircuitBreaker("test", config=config)

        # First two failures don't open
        circuit.record_failure(ValueError("fail 1"))
        assert circuit.state == CircuitState.CLOSED
        assert circuit.can_execute() is True

        circuit.record_failure(ValueError("fail 2"))
        assert circuit.state == CircuitState.CLOSED
        assert circuit.can_execute() is True

        # Third failure opens circuit
        circuit.record_failure(ValueError("fail 3"))
        assert circuit.state == CircuitState.OPEN
        assert circuit.can_execute() is False

    def test_success_resets_failure_count(self) -> None:
        """Success resets consecutive failure count."""
        config = CircuitBreakerConfig(failure_threshold=3)
        circuit = CircuitBreaker("test", config=config)

        circuit.record_failure()
        circuit.record_failure()
        assert circuit.stats.consecutive_failures == 2

        circuit.record_success()
        assert circuit.stats.consecutive_failures == 0
        assert circuit.stats.consecutive_successes == 1

        # Need 3 more failures to open now
        circuit.record_failure()
        circuit.record_failure()
        assert circuit.state == CircuitState.CLOSED

    def test_open_to_half_open_after_cooldown(self) -> None:
        """Circuit transitions to HALF_OPEN after cooldown."""
        config = CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=0.1)
        circuit = CircuitBreaker("test", config=config)

        # Open the circuit
        circuit.record_failure()
        circuit.record_failure()
        assert circuit.state == CircuitState.OPEN
        assert circuit.can_execute() is False

        # Wait for cooldown
        time.sleep(0.15)

        # Should transition to half-open on check
        assert circuit.can_execute() is True
        assert circuit.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self) -> None:
        """Successes in HALF_OPEN state close the circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            cooldown_seconds=0.05,
        )
        circuit = CircuitBreaker("test", config=config)

        # Open and wait for half-open
        circuit.record_failure()
        circuit.record_failure()
        time.sleep(0.1)
        circuit.can_execute()  # Triggers transition to half-open
        assert circuit.state == CircuitState.HALF_OPEN

        # First success
        circuit.record_success()
        assert circuit.state == CircuitState.HALF_OPEN

        # Second success should close
        circuit.record_success()
        assert circuit.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self) -> None:
        """Failure in HALF_OPEN state reopens the circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            cooldown_seconds=0.05,
        )
        circuit = CircuitBreaker("test", config=config)

        # Open and wait for half-open
        circuit.record_failure()
        circuit.record_failure()
        time.sleep(0.1)
        circuit.can_execute()  # Triggers transition to half-open
        assert circuit.state == CircuitState.HALF_OPEN

        # Failure should reopen
        circuit.record_failure()
        assert circuit.state == CircuitState.OPEN

    def test_half_open_limited_calls(self) -> None:
        """HALF_OPEN state limits concurrent calls."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            cooldown_seconds=0.01,
            half_open_max_calls=2,
        )
        circuit = CircuitBreaker("test", config=config)

        # Open and wait for half-open
        circuit.record_failure()
        time.sleep(0.02)
        assert circuit.can_execute()  # First allowed
        assert circuit.state == CircuitState.HALF_OPEN

        # Simulate calls in progress
        circuit._half_open_calls = 2

        # Third call should be rejected
        assert circuit.can_execute() is False


# ============================================================================
# CircuitBreaker Force Operations Tests
# ============================================================================


class TestCircuitBreakerForceOperations:
    """Tests for force_open, force_close, and reset."""

    def test_force_open(self) -> None:
        """force_open opens the circuit immediately."""
        circuit = CircuitBreaker("test")
        assert circuit.state == CircuitState.CLOSED

        circuit.force_open()

        assert circuit.state == CircuitState.OPEN
        assert circuit.can_execute() is False

    def test_force_close(self) -> None:
        """force_close closes the circuit immediately."""
        config = CircuitBreakerConfig(failure_threshold=1)
        circuit = CircuitBreaker("test", config=config)

        circuit.record_failure()
        assert circuit.state == CircuitState.OPEN

        circuit.force_close()

        assert circuit.state == CircuitState.CLOSED
        assert circuit.can_execute() is True

    def test_reset_clears_all_state(self) -> None:
        """reset clears circuit to initial state."""
        config = CircuitBreakerConfig(failure_threshold=2)
        circuit = CircuitBreaker("test", config=config)

        # Accumulate some state
        circuit.record_failure()
        circuit.record_failure()
        circuit.record_success()  # Won't close since already open

        circuit.reset()

        assert circuit.state == CircuitState.CLOSED
        assert circuit.stats.total_calls == 0
        assert circuit.stats.total_failures == 0
        assert circuit.stats.consecutive_failures == 0


# ============================================================================
# CircuitBreaker Listener Tests
# ============================================================================


class TestCircuitBreakerListeners:
    """Tests for state change listeners."""

    def test_add_listener(self) -> None:
        """add_listener registers a callback."""
        circuit = CircuitBreaker("test")
        listener = MagicMock()

        circuit.add_listener(listener)

        assert listener in circuit._listeners

    def test_listener_called_on_state_change(self) -> None:
        """Listeners are called when state changes."""
        config = CircuitBreakerConfig(failure_threshold=1)
        circuit = CircuitBreaker("test", config=config)

        listener = MagicMock()
        circuit.add_listener(listener)

        # Trigger state change
        circuit.record_failure()

        listener.assert_called_once()
        call_args = listener.call_args[0]
        assert call_args[0] == CircuitState.CLOSED  # old state
        assert call_args[1] == CircuitState.OPEN  # new state

    def test_multiple_listeners(self) -> None:
        """Multiple listeners are all called."""
        config = CircuitBreakerConfig(failure_threshold=1)
        circuit = CircuitBreaker("test", config=config)

        listener1 = MagicMock()
        listener2 = MagicMock()
        circuit.add_listener(listener1)
        circuit.add_listener(listener2)

        circuit.record_failure()

        listener1.assert_called_once()
        listener2.assert_called_once()

    def test_listener_exception_does_not_break_circuit(self) -> None:
        """Listener exceptions don't break the circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        circuit = CircuitBreaker("test", config=config)

        def failing_listener(old: CircuitState, new: CircuitState) -> None:
            raise ValueError("Listener error")

        circuit.add_listener(failing_listener)

        # Should not raise
        circuit.record_failure()

        assert circuit.state == CircuitState.OPEN


# ============================================================================
# CircuitBreaker wrap() Tests
# ============================================================================


class TestCircuitBreakerWrap:
    """Tests for CircuitBreaker.wrap() method."""

    def test_wrap_successful_function(self) -> None:
        """wrap executes and records success for successful functions."""
        circuit = CircuitBreaker("test")

        def success_fn(x: int) -> int:
            return x * 2

        wrapped = circuit.wrap(success_fn)
        result = wrapped(5)

        assert result == 10
        assert circuit.stats.total_successes == 1
        assert circuit.stats.total_failures == 0

    def test_wrap_failing_function(self) -> None:
        """wrap records failure when function raises."""
        circuit = CircuitBreaker("test")

        def fail_fn() -> None:
            raise ValueError("Failed")

        wrapped = circuit.wrap(fail_fn)

        with pytest.raises(ValueError):
            wrapped()

        assert circuit.stats.total_failures == 1
        assert circuit.stats.total_successes == 0

    def test_wrap_rejects_when_open(self) -> None:
        """wrap raises CircuitOpenError when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        circuit = CircuitBreaker("test", config=config)

        def success_fn() -> int:
            return 42

        wrapped = circuit.wrap(success_fn)

        # Open the circuit
        circuit.force_open()

        with pytest.raises(CircuitOpenError) as exc_info:
            wrapped()

        assert exc_info.value.kernel_id == "test"
        assert circuit.stats.total_rejections == 1

    def test_wrap_increments_half_open_calls(self) -> None:
        """wrap increments half_open_calls in HALF_OPEN state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            cooldown_seconds=0.01,
        )
        circuit = CircuitBreaker("test", config=config)

        def success_fn() -> int:
            # Check mid-execution
            assert circuit._half_open_calls == 1
            return 42

        wrapped = circuit.wrap(success_fn)

        # Open and wait for half-open
        circuit.force_open()
        time.sleep(0.02)

        # Should be allowed in half-open
        result = wrapped()
        assert result == 42


# ============================================================================
# CircuitBreaker to_dict() Tests
# ============================================================================


class TestCircuitBreakerToDict:
    """Tests for CircuitBreaker.to_dict() method."""

    def test_to_dict_basic(self) -> None:
        """to_dict returns correct basic structure."""
        circuit = CircuitBreaker("my_circuit")
        d = circuit.to_dict()

        assert d["name"] == "my_circuit"
        assert d["state"] == "CLOSED"
        assert "stats" in d
        assert "cooldown_remaining" in d

    def test_to_dict_with_cooldown(self) -> None:
        """to_dict shows cooldown remaining when open."""
        config = CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=10.0)
        circuit = CircuitBreaker("test", config=config)

        circuit.record_failure()
        assert circuit.state == CircuitState.OPEN

        d = circuit.to_dict()

        assert d["state"] == "OPEN"
        assert d["cooldown_remaining"] > 0
        assert d["cooldown_remaining"] <= 10.0

    def test_to_dict_no_cooldown_when_closed(self) -> None:
        """to_dict shows 0 cooldown when closed."""
        circuit = CircuitBreaker("test")
        d = circuit.to_dict()

        assert d["cooldown_remaining"] == 0


# ============================================================================
# CircuitBreakerRegistry Tests
# ============================================================================


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_initialization_defaults(self) -> None:
        """Registry initializes with default config."""
        registry = CircuitBreakerRegistry()
        assert registry._default_config is not None
        assert len(registry._circuits) == 0

    def test_initialization_custom_config(self) -> None:
        """Registry accepts custom default config."""
        config = CircuitBreakerConfig(failure_threshold=10)
        registry = CircuitBreakerRegistry(default_config=config)
        assert registry._default_config.failure_threshold == 10

    def test_get_or_create_new(self) -> None:
        """get_or_create creates new circuit if not exists."""
        registry = CircuitBreakerRegistry()

        circuit = registry.get_or_create("new_circuit")

        assert circuit is not None
        assert circuit.name == "new_circuit"
        assert "new_circuit" in registry._circuits

    def test_get_or_create_existing(self) -> None:
        """get_or_create returns existing circuit."""
        registry = CircuitBreakerRegistry()

        circuit1 = registry.get_or_create("test")
        circuit2 = registry.get_or_create("test")

        assert circuit1 is circuit2

    def test_get_or_create_with_custom_config(self) -> None:
        """get_or_create uses provided config."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(failure_threshold=3)

        circuit = registry.get_or_create("test", config=config)

        assert circuit._config.failure_threshold == 3

    def test_get_returns_existing(self) -> None:
        """get returns existing circuit."""
        registry = CircuitBreakerRegistry()
        registry.get_or_create("test")

        circuit = registry.get("test")

        assert circuit is not None
        assert circuit.name == "test"

    def test_get_returns_none_for_missing(self) -> None:
        """get returns None for non-existent circuit."""
        registry = CircuitBreakerRegistry()
        assert registry.get("nonexistent") is None

    def test_can_execute_no_circuit(self) -> None:
        """can_execute returns True if circuit doesn't exist."""
        registry = CircuitBreakerRegistry()
        assert registry.can_execute("unknown") is True

    def test_can_execute_with_circuit(self) -> None:
        """can_execute delegates to circuit."""
        registry = CircuitBreakerRegistry()
        circuit = registry.get_or_create("test")

        assert registry.can_execute("test") is True

        circuit.force_open()
        assert registry.can_execute("test") is False

    def test_record_success(self) -> None:
        """record_success updates circuit stats."""
        registry = CircuitBreakerRegistry()
        circuit = registry.get_or_create("test")

        registry.record_success("test")

        assert circuit.stats.total_successes == 1

    def test_record_success_no_circuit(self) -> None:
        """record_success is no-op for non-existent circuit."""
        registry = CircuitBreakerRegistry()
        # Should not raise
        registry.record_success("unknown")

    def test_record_failure(self) -> None:
        """record_failure updates circuit stats."""
        registry = CircuitBreakerRegistry()

        # Creates circuit if not exists
        registry.record_failure("test", ValueError("error"))

        circuit = registry.get("test")
        assert circuit is not None
        assert circuit.stats.total_failures == 1

    def test_reset_all(self) -> None:
        """reset_all resets all circuits."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(failure_threshold=1)

        circuit1 = registry.get_or_create("c1", config)
        circuit2 = registry.get_or_create("c2", config)

        circuit1.record_failure()
        circuit2.record_failure()
        assert circuit1.state == CircuitState.OPEN
        assert circuit2.state == CircuitState.OPEN

        registry.reset_all()

        assert circuit1.state == CircuitState.CLOSED
        assert circuit2.state == CircuitState.CLOSED

    def test_get_all_stats(self) -> None:
        """get_all_stats returns stats for all circuits."""
        registry = CircuitBreakerRegistry()
        registry.get_or_create("c1")
        registry.get_or_create("c2")

        stats = registry.get_all_stats()

        assert "c1" in stats
        assert "c2" in stats
        assert "state" in stats["c1"]

    def test_get_open_circuits(self) -> None:
        """get_open_circuits returns names of open circuits."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(failure_threshold=1)

        c1 = registry.get_or_create("c1", config)
        c2 = registry.get_or_create("c2", config)
        c3 = registry.get_or_create("c3", config)

        c1.record_failure()  # Opens
        c3.record_failure()  # Opens

        open_circuits = registry.get_open_circuits()

        assert "c1" in open_circuits
        assert "c2" not in open_circuits
        assert "c3" in open_circuits


# ============================================================================
# Global Registry Tests
# ============================================================================


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_global_circuit_registry_creates(self) -> None:
        """get_global_circuit_registry creates registry if none exists."""
        # Reset global state
        import layerzero.dispatch.circuit_breaker as cb_module
        cb_module._global_registry = None

        registry = get_global_circuit_registry()

        assert registry is not None
        assert isinstance(registry, CircuitBreakerRegistry)

    def test_get_global_circuit_registry_singleton(self) -> None:
        """get_global_circuit_registry returns same instance."""
        registry1 = get_global_circuit_registry()
        registry2 = get_global_circuit_registry()

        assert registry1 is registry2

    def test_get_circuit_uses_global_registry(self) -> None:
        """get_circuit uses global registry."""
        # Reset global
        import layerzero.dispatch.circuit_breaker as cb_module
        cb_module._global_registry = None

        circuit = get_circuit("test_circuit")

        assert circuit is not None
        assert circuit.name == "test_circuit"

        # Should be in global registry
        registry = get_global_circuit_registry()
        assert registry.get("test_circuit") is circuit


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread safety of circuit breaker."""

    def test_concurrent_record_operations(self) -> None:
        """Multiple threads can record success/failure concurrently."""
        config = CircuitBreakerConfig(failure_threshold=100)
        circuit = CircuitBreaker("test", config=config)
        errors: list[Exception] = []

        def stress_task() -> None:
            try:
                for _ in range(100):
                    circuit.record_failure()
                    circuit.can_execute()
                    circuit.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress_task) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have recorded many operations
        assert circuit.stats.total_calls > 0

    def test_concurrent_wrap_calls(self) -> None:
        """Multiple threads can call wrapped functions concurrently."""
        circuit = CircuitBreaker("test")
        results: list[int] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def success_fn(x: int) -> int:
            time.sleep(0.001)  # Small delay
            return x * 2

        wrapped = circuit.wrap(success_fn)

        def call_wrapped() -> None:
            try:
                result = wrapped(5)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(call_wrapped) for _ in range(20)]
            for f in futures:
                f.result()

        assert len(errors) == 0
        assert len(results) == 20
        assert all(r == 10 for r in results)

    def test_concurrent_registry_operations(self) -> None:
        """Multiple threads can access registry concurrently."""
        registry = CircuitBreakerRegistry()
        errors: list[Exception] = []

        def registry_task(name: str) -> None:
            try:
                for _ in range(50):
                    circuit = registry.get_or_create(name)
                    registry.record_success(name)
                    registry.record_failure(name)
                    registry.can_execute(name)
                    _ = registry.get_all_stats()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=registry_task, args=(f"circuit_{i}",))
            for i in range(5)
        ]
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

    def test_record_failure_with_none_error(self) -> None:
        """record_failure works with None error."""
        circuit = CircuitBreaker("test")
        circuit.record_failure(error=None)
        assert circuit.stats.total_failures == 1

    def test_stats_update_correctly_across_transitions(self) -> None:
        """Stats update correctly across all state transitions."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            cooldown_seconds=0.05,
        )
        circuit = CircuitBreaker("test", config=config)

        # CLOSED -> OPEN
        circuit.record_failure()
        circuit.record_failure()
        assert circuit.stats.state_changes == 1

        # Wait for HALF_OPEN
        time.sleep(0.1)
        circuit.can_execute()
        assert circuit.stats.state_changes == 2

        # HALF_OPEN -> OPEN (failure)
        circuit.record_failure()
        assert circuit.stats.state_changes == 3

        # Wait again for HALF_OPEN
        time.sleep(0.1)
        circuit.can_execute()

        # HALF_OPEN -> CLOSED (success x2)
        circuit.record_success()
        circuit.record_success()
        assert circuit.state == CircuitState.CLOSED
        assert circuit.stats.state_changes == 5

    def test_very_short_cooldown(self) -> None:
        """Very short cooldown works correctly."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            cooldown_seconds=0.001,  # 1ms
        )
        circuit = CircuitBreaker("test", config=config)

        circuit.record_failure()
        assert circuit.state == CircuitState.OPEN

        time.sleep(0.01)  # Wait > cooldown

        assert circuit.can_execute() is True
        assert circuit.state == CircuitState.HALF_OPEN

    def test_large_failure_threshold(self) -> None:
        """Large failure threshold works correctly."""
        config = CircuitBreakerConfig(failure_threshold=1000)
        circuit = CircuitBreaker("test", config=config)

        for _ in range(999):
            circuit.record_failure()
            assert circuit.state == CircuitState.CLOSED

        circuit.record_failure()
        assert circuit.state == CircuitState.OPEN

    def test_wrap_with_args_and_kwargs(self) -> None:
        """wrap correctly passes args and kwargs."""
        circuit = CircuitBreaker("test")

        def complex_fn(a: int, b: int, *, c: int = 0) -> int:
            return a + b + c

        wrapped = circuit.wrap(complex_fn)

        result = wrapped(1, 2, c=3)
        assert result == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
