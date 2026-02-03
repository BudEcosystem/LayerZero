"""
Circuit breaker pattern for backend fault tolerance.

This module provides:
- CircuitState: Circuit breaker states
- CircuitBreakerConfig: Configuration for circuit breaker
- CircuitBreaker: Circuit breaker implementation
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """States of a circuit breaker.

    Attributes:
        CLOSED: Circuit is closed, calls pass through normally.
        OPEN: Circuit is open, calls are blocked.
        HALF_OPEN: Circuit is testing, limited calls allowed.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuration for circuit breaker.

    Attributes:
        failure_threshold: Failures before circuit opens.
        success_threshold: Successes in half-open to close circuit.
        timeout_seconds: Time before open circuit becomes half-open.
        half_open_max_calls: Max calls allowed in half-open state.
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 1


class CircuitBreaker:
    """Circuit breaker for backend fault tolerance.

    Implements the circuit breaker pattern to prevent cascading failures
    when a backend is experiencing issues.

    States:
    - CLOSED: Normal operation, all calls pass through
    - OPEN: Backend failing, calls blocked for timeout period
    - HALF_OPEN: Testing recovery, limited calls allowed

    Thread-safe for concurrent access.

    Example:
        cb = CircuitBreaker("flashinfer")

        if cb.can_execute():
            try:
                result = call_backend()
                cb.record_success()
            except Exception:
                cb.record_failure()
        else:
            # Circuit is open, use fallback
            ...
    """

    def __init__(
        self,
        backend_id: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            backend_id: Backend identifier.
            config: Circuit breaker configuration.
        """
        self._backend_id = backend_id
        self._config = config or CircuitBreakerConfig()
        self._lock = threading.RLock()

        # State
        self._state = CircuitState.CLOSED
        self._state_changed_at = time.monotonic()

        # Counters
        self._failure_count = 0
        self._half_open_success_count = 0
        self._half_open_calls = 0

    @property
    def backend_id(self) -> str:
        """Get backend identifier."""
        return self._backend_id

    @property
    def config(self) -> CircuitBreakerConfig:
        """Get configuration."""
        return self._config

    @property
    def state(self) -> CircuitState:
        """Get current circuit state.

        Note: Accessing state may trigger automatic transitions
        (e.g., OPEN -> HALF_OPEN after timeout).

        Returns:
            Current CircuitState.
        """
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open."""
        return self.state == CircuitState.HALF_OPEN

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        with self._lock:
            return self._failure_count

    @property
    def half_open_success_count(self) -> int:
        """Get success count in half-open state."""
        with self._lock:
            return self._half_open_success_count

    @property
    def time_in_current_state(self) -> float:
        """Get time spent in current state (seconds)."""
        with self._lock:
            return time.monotonic() - self._state_changed_at

    @property
    def open_until(self) -> float:
        """Get time when open circuit will become half-open.

        Returns:
            Monotonic time when circuit will transition to half-open,
            or 0.0 if circuit is not open.
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                return self._state_changed_at + self._config.timeout_seconds
            return 0.0

    def can_execute(self) -> bool:
        """Check if a call can be executed.

        Returns:
            True if call should proceed, False if blocked.
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open
                return self._half_open_calls < self._config.half_open_max_calls

            return False

    def record_call(self) -> None:
        """Record that a call was made (for half-open tracking)."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1

            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self._config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        "Circuit breaker for %s opened after %d failures",
                        self._backend_id,
                        self._failure_count,
                    )

            elif self._state == CircuitState.HALF_OPEN:
                # Failure in half-open -> back to open
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "Circuit breaker for %s reopened after probe failure",
                    self._backend_id,
                )

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_success_count += 1

                if self._half_open_success_count >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info(
                        "Circuit breaker for %s closed after %d successes",
                        self._backend_id,
                        self._half_open_success_count,
                    )

            elif self._state == CircuitState.CLOSED:
                # Success in closed state - could reset failure count
                # but we keep failures to track overall health
                pass

    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            logger.info("Circuit breaker for %s manually reset", self._backend_id)

    def _check_state_transition(self) -> None:
        """Check and apply automatic state transitions.

        Must be called with lock held.
        """
        now = time.monotonic()

        if self._state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if now - self._state_changed_at >= self._config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
                logger.info(
                    "Circuit breaker for %s transitioned to half-open",
                    self._backend_id,
                )

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state.

        Must be called with lock held.

        Args:
            new_state: New circuit state.
        """
        old_state = self._state
        self._state = new_state
        self._state_changed_at = time.monotonic()

        # Reset counters based on transition
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._half_open_success_count = 0
            self._half_open_calls = 0

        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_success_count = 0
            self._half_open_calls = 0

        elif new_state == CircuitState.OPEN:
            self._half_open_success_count = 0
            self._half_open_calls = 0

        logger.debug(
            "Circuit breaker %s: %s -> %s",
            self._backend_id,
            old_state.value,
            new_state.value,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert circuit breaker state to dictionary.

        Returns:
            Dictionary with circuit breaker state.
        """
        with self._lock:
            self._check_state_transition()
            return {
                "backend_id": self._backend_id,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "half_open_success_count": self._half_open_success_count,
                "time_in_state_seconds": self.time_in_current_state,
                "open_until": self.open_until,
            }


class CircuitBreakerRegistry:
    """Registry for circuit breakers across backends.

    Provides a central place to manage circuit breakers for all backends.
    Thread-safe for concurrent access.

    Example:
        registry = CircuitBreakerRegistry()

        cb = registry.get_or_create("flashinfer")
        if cb.can_execute():
            ...
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """Initialize registry.

        Args:
            config: Default configuration for new circuit breakers.
        """
        self._config = config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def get_or_create(self, backend_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for backend.

        Args:
            backend_id: Backend identifier.

        Returns:
            CircuitBreaker for the backend.
        """
        with self._lock:
            if backend_id not in self._breakers:
                self._breakers[backend_id] = CircuitBreaker(
                    backend_id=backend_id,
                    config=self._config,
                )
            return self._breakers[backend_id]

    def can_execute(self, backend_id: str) -> bool:
        """Check if backend can execute.

        Args:
            backend_id: Backend identifier.

        Returns:
            True if circuit allows execution.
        """
        return self.get_or_create(backend_id).can_execute()

    def record_failure(self, backend_id: str) -> None:
        """Record failure for backend.

        Args:
            backend_id: Backend identifier.
        """
        self.get_or_create(backend_id).record_failure()

    def record_success(self, backend_id: str) -> None:
        """Record success for backend.

        Args:
            backend_id: Backend identifier.
        """
        self.get_or_create(backend_id).record_success()

    def reset_backend(self, backend_id: str) -> None:
        """Reset circuit breaker for backend.

        Args:
            backend_id: Backend identifier.
        """
        with self._lock:
            if backend_id in self._breakers:
                self._breakers[backend_id].reset()

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for cb in self._breakers.values():
                cb.reset()

    def get_open_circuits(self) -> list[str]:
        """Get list of backends with open circuits.

        Returns:
            List of backend IDs with open circuits.
        """
        with self._lock:
            return [
                bid for bid, cb in self._breakers.items()
                if cb.is_open
            ]

    def __len__(self) -> int:
        """Get number of circuit breakers."""
        with self._lock:
            return len(self._breakers)


# Global circuit breaker registry (thread-safe singleton pattern)
_global_registry: CircuitBreakerRegistry | None = None
_global_registry_lock = threading.Lock()


def get_global_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get global circuit breaker registry (thread-safe singleton).

    Uses double-checked locking pattern to ensure thread safety while
    minimizing lock contention after initialization.

    Returns:
        Global CircuitBreakerRegistry instance.
    """
    global _global_registry

    # Fast path: registry already initialized
    if _global_registry is not None:
        return _global_registry

    # Slow path: acquire lock and initialize if needed
    with _global_registry_lock:
        # Double-check inside lock to prevent race condition
        if _global_registry is None:
            _global_registry = CircuitBreakerRegistry()
        return _global_registry
