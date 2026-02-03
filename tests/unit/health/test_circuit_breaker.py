"""Tests for circuit breaker pattern."""
from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock, patch

from layerzero.health.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_state_values(self) -> None:
        """CircuitState has expected values."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self) -> None:
        """Default config values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout_seconds == 30.0
        assert config.half_open_max_calls == 1

    def test_custom_values(self) -> None:
        """Custom config values accepted."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=3,
            timeout_seconds=60.0,
            half_open_max_calls=2,
        )

        assert config.failure_threshold == 10
        assert config.success_threshold == 3
        assert config.timeout_seconds == 60.0
        assert config.half_open_max_calls == 2

    def test_config_immutable(self) -> None:
        """Config is immutable."""
        config = CircuitBreakerConfig()

        with pytest.raises(AttributeError):
            config.failure_threshold = 10


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_circuit_closed_initially(self) -> None:
        """Circuit closed initially."""
        cb = CircuitBreaker("test_backend")

        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed is True
        assert cb.is_open is False

    def test_circuit_allows_calls_when_closed(self) -> None:
        """Circuit allows calls when closed."""
        cb = CircuitBreaker("test_backend")

        assert cb.can_execute() is True

    def test_circuit_opens_after_threshold(self, mock_time) -> None:
        """Circuit opens after N failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test_backend", config=config)

        # Record failures
        for _ in range(3):
            cb.record_failure()
            mock_time["advance"](1.0)

        assert cb.state == CircuitState.OPEN
        assert cb.is_open is True

    def test_circuit_blocks_calls_when_open(self, mock_time) -> None:
        """Circuit blocks calls when open."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test_backend", config=config)

        # Open the circuit
        for _ in range(3):
            cb.record_failure()
            mock_time["advance"](1.0)

        assert cb.can_execute() is False

    def test_circuit_half_open_after_cooldown(self, mock_time) -> None:
        """Circuit half-open after cooldown."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30.0,
        )
        cb = CircuitBreaker("test_backend", config=config)

        # Open the circuit
        for _ in range(3):
            cb.record_failure()
            mock_time["advance"](1.0)

        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        mock_time["advance"](35.0)  # Beyond 30s timeout

        # Should transition to half-open when checked
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.is_half_open is True

    def test_circuit_allows_limited_calls_when_half_open(self, mock_time) -> None:
        """Circuit allows limited calls in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30.0,
            half_open_max_calls=1,
        )
        cb = CircuitBreaker("test_backend", config=config)

        # Open and wait for half-open
        for _ in range(3):
            cb.record_failure()
            mock_time["advance"](1.0)

        mock_time["advance"](35.0)

        # Should allow one call
        assert cb.can_execute() is True

        # After one call, should block
        cb.record_call()
        assert cb.can_execute() is False

    def test_circuit_closes_on_success(self, mock_time) -> None:
        """Circuit closes on successful probe."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=30.0,
        )
        cb = CircuitBreaker("test_backend", config=config)

        # Open and wait for half-open
        for _ in range(3):
            cb.record_failure()
            mock_time["advance"](1.0)

        mock_time["advance"](35.0)

        # Half-open state
        assert cb.state == CircuitState.HALF_OPEN

        # Record successes
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # Still half-open

        cb.record_success()
        assert cb.state == CircuitState.CLOSED  # Closed now

    def test_circuit_reopens_on_probe_failure(self, mock_time) -> None:
        """Circuit reopens if probe fails."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30.0,
        )
        cb = CircuitBreaker("test_backend", config=config)

        # Open and wait for half-open
        for _ in range(3):
            cb.record_failure()
            mock_time["advance"](1.0)

        mock_time["advance"](35.0)

        # Half-open state
        assert cb.state == CircuitState.HALF_OPEN

        # Probe fails
        cb.record_failure()

        # Should reopen
        assert cb.state == CircuitState.OPEN

    def test_failure_count_reset_on_close(self, mock_time) -> None:
        """Failure count resets when circuit closes."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout_seconds=30.0,
        )
        cb = CircuitBreaker("test_backend", config=config)

        # Open the circuit
        for _ in range(3):
            cb.record_failure()
            mock_time["advance"](1.0)

        # Wait for half-open
        mock_time["advance"](35.0)
        _ = cb.state  # Trigger transition

        # Close with success
        cb.record_success()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_manual_reset(self, mock_time) -> None:
        """Manual reset closes circuit."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test_backend", config=config)

        # Open the circuit
        for _ in range(3):
            cb.record_failure()
            mock_time["advance"](1.0)

        assert cb.state == CircuitState.OPEN

        # Manual reset
        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_success_count_tracked_in_half_open(self, mock_time) -> None:
        """Success count tracked correctly in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=3,
            timeout_seconds=30.0,
        )
        cb = CircuitBreaker("test_backend", config=config)

        # Open and half-open
        for _ in range(3):
            cb.record_failure()
            mock_time["advance"](1.0)

        mock_time["advance"](35.0)
        _ = cb.state

        # Track successes
        cb.record_success()
        assert cb.half_open_success_count == 1

        cb.record_success()
        assert cb.half_open_success_count == 2

        cb.record_success()
        # Should have closed and reset
        assert cb.state == CircuitState.CLOSED

    def test_time_in_current_state(self, mock_time) -> None:
        """Time spent in current state tracked."""
        cb = CircuitBreaker("test_backend")

        mock_time["set"](100.0)
        # Force state change tracking
        cb._state_changed_at = 100.0

        mock_time["set"](150.0)

        assert cb.time_in_current_state == 50.0

    def test_open_until_time(self, mock_time) -> None:
        """open_until returns expected time."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30.0,
        )
        cb = CircuitBreaker("test_backend", config=config)

        mock_time["set"](100.0)

        # Open the circuit
        for _ in range(3):
            cb.record_failure()

        # Should be open until 100 + 30 = 130
        assert cb.open_until == 130.0

    def test_backend_id_accessible(self) -> None:
        """Backend ID is accessible."""
        cb = CircuitBreaker("flashinfer")
        assert cb.backend_id == "flashinfer"


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker."""

    def test_full_lifecycle(self, mock_time) -> None:
        """Full circuit breaker lifecycle."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=30.0,
        )
        cb = CircuitBreaker("test", config=config)

        # 1. Start closed
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

        # 2. Some successes
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

        # 3. Failures open circuit
        for _ in range(3):
            cb.record_failure()
            mock_time["advance"](1.0)

        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

        # 4. Wait for timeout -> half-open
        mock_time["advance"](35.0)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.can_execute() is True

        # 5. First probe fails -> reopen
        cb.record_call()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # 6. Wait again -> half-open
        mock_time["advance"](35.0)
        assert cb.state == CircuitState.HALF_OPEN

        # 7. Probe succeeds -> start closing
        cb.record_call()
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # Need 2 successes

        # Reset call count for second probe
        cb._half_open_calls = 0
        cb.record_call()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED  # Closed!

    def test_concurrent_safety(self, mock_time) -> None:
        """Circuit breaker is thread-safe."""
        from concurrent.futures import ThreadPoolExecutor

        config = CircuitBreakerConfig(failure_threshold=100)
        cb = CircuitBreaker("test", config=config)

        def record_failures():
            for _ in range(20):
                cb.record_failure()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(record_failures) for _ in range(4)]
            for f in futures:
                f.result()

        # Should have recorded all 80 failures
        assert cb.failure_count == 80
