"""
Circuit Breaker Module

Standalone circuit breaker implementation for fault tolerance.
This module provides reusable circuit breaker patterns that can be
used by any dispatcher or external component.

Circuit Breaker States:
- CLOSED: Normal operation, requests pass through
- OPEN: Too many failures, requests are blocked
- HALF_OPEN: Testing recovery, limited requests allowed

Based on the Circuit Breaker pattern from "Release It!" by Michael Nygard.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import RLock
from typing import Any, Callable, Generic, TypeVar

from layerzero.dispatch.types import CircuitOpenError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker state."""
    CLOSED = auto()    # Normal operation
    OPEN = auto()      # Blocking requests
    HALF_OPEN = auto() # Testing recovery


@dataclass(slots=True)
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit.
        success_threshold: Number of successes in half-open before closing.
        cooldown_seconds: Time to wait before transitioning to half-open.
        half_open_max_calls: Max concurrent calls in half-open state.
        reset_timeout_seconds: Optional hard reset after this time.
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    cooldown_seconds: float = 30.0
    half_open_max_calls: int = 3
    reset_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")
        if self.cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be > 0")


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker.

    Attributes:
        total_calls: Total number of calls.
        total_successes: Successful calls.
        total_failures: Failed calls.
        total_rejections: Calls rejected due to open circuit.
        consecutive_failures: Current consecutive failure streak.
        consecutive_successes: Current consecutive success streak.
        last_failure_time: Monotonic time of last failure.
        last_success_time: Monotonic time of last success.
        state_changes: Number of state transitions.
    """
    __slots__ = (
        "total_calls",
        "total_successes",
        "total_failures",
        "total_rejections",
        "consecutive_failures",
        "consecutive_successes",
        "last_failure_time",
        "last_success_time",
        "state_changes",
    )

    total_calls: int
    total_successes: int
    total_failures: int
    total_rejections: int
    consecutive_failures: int
    consecutive_successes: int
    last_failure_time: float | None
    last_success_time: float | None
    state_changes: int

    def __init__(self) -> None:
        self.total_calls = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_rejections = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.state_changes = 0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate (0.0 to 1.0)."""
        if self.total_calls == 0:
            return 0.0
        return self.total_failures / self.total_calls

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "failure_rate": self.failure_rate,
            "state_changes": self.state_changes,
        }


class CircuitBreaker(Generic[T]):
    """Thread-safe circuit breaker.

    Protects against cascading failures by monitoring call success/failure
    and automatically blocking calls when too many failures occur.

    Usage:
        circuit = CircuitBreaker("my-service", config)

        # Check if call is allowed
        if circuit.can_execute():
            try:
                result = do_something()
                circuit.record_success()
            except Exception as e:
                circuit.record_failure(e)

        # Or use as context manager
        with circuit.protect():
            result = do_something()

        # Or wrap a callable
        safe_call = circuit.wrap(dangerous_function)
        result = safe_call(args)
    """

    __slots__ = (
        "_name",
        "_config",
        "_state",
        "_stats",
        "_lock",
        "_cooldown_until",
        "_half_open_calls",
        "_listeners",
    )

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Identifier for this circuit.
            config: Configuration options.
        """
        self._name = name
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._lock = RLock()
        self._cooldown_until: float | None = None
        self._half_open_calls = 0
        self._listeners: list[Callable[[CircuitState, CircuitState], None]] = []

    @property
    def name(self) -> str:
        """Get circuit name."""
        return self._name

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats

    def can_execute(self) -> bool:
        """Check if a call is currently allowed.

        Returns:
            True if call should proceed, False if blocked.
        """
        with self._lock:
            return self._can_execute_locked()

    def _can_execute_locked(self) -> bool:
        """Check execution (must hold lock)."""
        now = time.monotonic()

        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if cooldown period has elapsed
            if self._cooldown_until and now >= self._cooldown_until:
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open
            return self._half_open_calls < self._config.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.total_successes += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls -= 1
                if self._stats.consecutive_successes >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed call.

        Args:
            error: The exception that caused the failure.
        """
        with self._lock:
            self._stats.total_calls += 1
            self._stats.total_failures += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = time.monotonic()

            if self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self._config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._half_open_calls -= 1
                self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state (must hold lock)."""
        old_state = self._state
        if old_state == new_state:
            return

        logger.info(
            f"Circuit '{self._name}' transitioning: {old_state.name} -> {new_state.name}"
        )

        self._state = new_state
        self._stats.state_changes += 1

        if new_state == CircuitState.OPEN:
            self._cooldown_until = time.monotonic() + self._config.cooldown_seconds
            self._half_open_calls = 0

        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0

        elif new_state == CircuitState.CLOSED:
            self._cooldown_until = None
            self._half_open_calls = 0
            self._stats.consecutive_failures = 0

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(old_state, new_state)
            except Exception as e:
                logger.warning(f"Circuit listener error: {e}")

    def force_open(self) -> None:
        """Manually open the circuit."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)

    def force_close(self) -> None:
        """Manually close the circuit."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)

    def reset(self) -> None:
        """Reset circuit to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._cooldown_until = None
            self._half_open_calls = 0
            self._stats = CircuitStats()

    def add_listener(
        self,
        listener: Callable[[CircuitState, CircuitState], None],
    ) -> None:
        """Add state change listener.

        Args:
            listener: Callback receiving (old_state, new_state).
        """
        self._listeners.append(listener)

    def wrap(self, func: Callable[..., T]) -> Callable[..., T]:
        """Wrap a function with circuit breaker protection.

        Args:
            func: Function to wrap.

        Returns:
            Wrapped function that respects circuit state.
        """
        def wrapped(*args: Any, **kwargs: Any) -> T:
            if not self.can_execute():
                self._stats.total_rejections += 1
                raise CircuitOpenError(
                    self._name,
                    self._config.cooldown_seconds,
                )

            with self._lock:
                if self._state == CircuitState.HALF_OPEN:
                    self._half_open_calls += 1

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        return wrapped

    def to_dict(self) -> dict[str, Any]:
        """Get circuit status as dictionary."""
        with self._lock:
            return {
                "name": self._name,
                "state": self._state.name,
                "stats": self._stats.to_dict(),
                "cooldown_remaining": (
                    max(0, self._cooldown_until - time.monotonic())
                    if self._cooldown_until
                    else 0
                ),
            }


class CircuitBreakerRegistry:
    """Registry of circuit breakers for multiple resources.

    Provides a central place to manage circuit breakers for
    different services/kernels/backends.
    """

    __slots__ = ("_circuits", "_default_config", "_lock")

    def __init__(
        self,
        default_config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize registry.

        Args:
            default_config: Default config for new circuits.
        """
        self._circuits: dict[str, CircuitBreaker] = {}
        self._default_config = default_config or CircuitBreakerConfig()
        self._lock = RLock()

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get existing circuit or create new one.

        Args:
            name: Circuit identifier.
            config: Optional config (uses default if not provided).

        Returns:
            CircuitBreaker instance.
        """
        with self._lock:
            if name not in self._circuits:
                self._circuits[name] = CircuitBreaker(
                    name,
                    config or self._default_config,
                )
            return self._circuits[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit by name.

        Args:
            name: Circuit identifier.

        Returns:
            CircuitBreaker or None if not found.
        """
        return self._circuits.get(name)

    def can_execute(self, name: str) -> bool:
        """Check if circuit allows execution.

        Args:
            name: Circuit identifier.

        Returns:
            True if execution allowed (or circuit doesn't exist).
        """
        circuit = self._circuits.get(name)
        if circuit is None:
            return True
        return circuit.can_execute()

    def record_success(self, name: str) -> None:
        """Record success for circuit.

        Args:
            name: Circuit identifier.
        """
        circuit = self._circuits.get(name)
        if circuit:
            circuit.record_success()

    def record_failure(self, name: str, error: Exception | None = None) -> None:
        """Record failure for circuit.

        Args:
            name: Circuit identifier.
            error: The exception that occurred.
        """
        circuit = self.get_or_create(name)
        circuit.record_failure(error)

    def reset_all(self) -> None:
        """Reset all circuits."""
        with self._lock:
            for circuit in self._circuits.values():
                circuit.reset()

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all circuits.

        Returns:
            Dict of circuit_name -> status dict.
        """
        with self._lock:
            return {
                name: circuit.to_dict()
                for name, circuit in self._circuits.items()
            }

    def get_open_circuits(self) -> list[str]:
        """Get names of all open circuits.

        Returns:
            List of circuit names in OPEN state.
        """
        with self._lock:
            return [
                name
                for name, circuit in self._circuits.items()
                if circuit.state == CircuitState.OPEN
            ]


# Global registry instance
_global_registry: CircuitBreakerRegistry | None = None
_global_lock = RLock()


def get_global_circuit_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry.

    Returns:
        Global CircuitBreakerRegistry instance.
    """
    global _global_registry

    if _global_registry is None:
        with _global_lock:
            if _global_registry is None:
                _global_registry = CircuitBreakerRegistry()

    return _global_registry


def get_circuit(name: str) -> CircuitBreaker:
    """Get or create a circuit breaker by name.

    Convenience function using global registry.

    Args:
        name: Circuit identifier.

    Returns:
        CircuitBreaker instance.
    """
    return get_global_circuit_registry().get_or_create(name)
