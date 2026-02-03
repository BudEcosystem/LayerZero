"""
LayerZero Backend Registry

Registry of backend libraries with health tracking and circuit breaker.
Thread-safe backend registration and health monitoring.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, unique
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layerzero.models.backend_spec import BackendSpec


@unique
class BackendState(str, Enum):
    """Backend health state.

    Implements circuit breaker pattern:
    - HEALTHY: Backend is working normally
    - DEGRADED: Some failures but still usable (reserved for future use)
    - UNHEALTHY: Too many failures, backend is disabled
    - COOLDOWN: Cooldown elapsed, ready to retry
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    COOLDOWN = "cooldown"


@dataclass
class BackendHealth:
    """Health tracking for a backend.

    Tracks failure count, timing, and circuit breaker state.

    Attributes:
        backend_id: Backend identifier.
        state: Current health state.
        failure_count: Consecutive failure count.
        last_failure_time: Monotonic time of last failure.
        last_success_time: Monotonic time of last success.
        cooldown_until: Monotonic time when cooldown ends.
        total_requests: Total number of requests.
        total_failures: Total number of failures (lifetime).
    """

    backend_id: str
    state: BackendState = BackendState.HEALTHY
    failure_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    cooldown_until: float | None = None
    total_requests: int = 0
    total_failures: int = 0


class BackendRegistry:
    """Registry of backend libraries with health tracking.

    Implements circuit breaker pattern:
    - After N consecutive failures, backend is disabled (UNHEALTHY)
    - After cooldown period, backend enters COOLDOWN state
    - Single success after cooldown resets to HEALTHY
    - Single failure after cooldown returns to UNHEALTHY

    Thread-safe for concurrent access.

    Attributes:
        _failure_threshold: Number of failures before circuit opens.
        _cooldown_seconds: Seconds to wait before retrying.
    """

    __slots__ = (
        "_lock",
        "_backends",
        "_health",
        "_failure_threshold",
        "_cooldown_seconds",
    )

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
    ) -> None:
        """Initialize backend registry.

        Args:
            failure_threshold: Number of consecutive failures to open circuit.
            cooldown_seconds: Seconds to wait before allowing retry.
        """
        self._lock = RLock()
        self._backends: dict[str, "BackendSpec"] = {}
        self._health: dict[str, BackendHealth] = {}
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds

    def register(self, spec: "BackendSpec") -> None:
        """Register a backend specification.

        If backend already exists, updates it. Health state is preserved
        unless the new spec shows the backend is not installed.

        Args:
            spec: Backend specification to register.
        """
        with self._lock:
            existing_health = self._health.get(spec.backend_id)

            self._backends[spec.backend_id] = spec

            if existing_health is None:
                # New registration
                initial_state = (
                    BackendState.HEALTHY
                    if spec.installed and spec.healthy
                    else BackendState.UNHEALTHY
                )
                self._health[spec.backend_id] = BackendHealth(
                    backend_id=spec.backend_id,
                    state=initial_state,
                )
            elif not spec.installed:
                # Backend became unavailable
                existing_health.state = BackendState.UNHEALTHY

    def probe_and_register(
        self,
        backend_id: str,
        module_name: str,
    ) -> "BackendSpec":
        """Probe a backend and register it.

        Uses BackendSpec.probe() to detect availability and version.

        Args:
            backend_id: Backend identifier.
            module_name: Python module name to probe.

        Returns:
            Probed BackendSpec (also registered).
        """
        from layerzero.models.backend_spec import BackendSpec

        spec = BackendSpec.probe(backend_id, module_name)
        self.register(spec)
        return spec

    def get(self, backend_id: str) -> "BackendSpec | None":
        """Get backend by ID.

        Args:
            backend_id: Backend identifier.

        Returns:
            BackendSpec if found, None otherwise.
        """
        with self._lock:
            return self._backends.get(backend_id)

    def get_health(self, backend_id: str) -> BackendHealth | None:
        """Get health status for a backend.

        Also updates circuit breaker state based on cooldown.

        Args:
            backend_id: Backend identifier.

        Returns:
            BackendHealth if found, None otherwise.
        """
        with self._lock:
            health = self._health.get(backend_id)
            if health is None:
                return None

            # Check if cooldown has elapsed
            if health.state == BackendState.UNHEALTHY:
                if health.cooldown_until is not None:
                    now = time.monotonic()
                    if now >= health.cooldown_until:
                        health.state = BackendState.COOLDOWN

            return health

    def is_available(self, backend_id: str) -> bool:
        """Check if backend is available.

        A backend is available if:
        - It is registered
        - It is installed
        - Its health state is HEALTHY or COOLDOWN

        Args:
            backend_id: Backend identifier.

        Returns:
            True if available, False otherwise.
        """
        with self._lock:
            spec = self._backends.get(backend_id)
            if spec is None or not spec.installed:
                return False

            health = self.get_health(backend_id)
            if health is None:
                return False

            return health.state in (BackendState.HEALTHY, BackendState.COOLDOWN)

    def record_success(self, backend_id: str) -> None:
        """Record a successful operation.

        Resets consecutive failure count and closes circuit.

        Args:
            backend_id: Backend identifier.
        """
        with self._lock:
            health = self._health.get(backend_id)
            if health is None:
                return

            health.failure_count = 0
            health.total_requests += 1
            health.last_success_time = time.monotonic()
            health.state = BackendState.HEALTHY
            health.cooldown_until = None

    def record_failure(self, backend_id: str, error: str) -> None:
        """Record a failed operation.

        Increments failure count. If threshold is reached, opens circuit.

        Args:
            backend_id: Backend identifier.
            error: Error message (for logging/debugging).
        """
        with self._lock:
            health = self._health.get(backend_id)
            if health is None:
                return

            health.failure_count += 1
            health.total_failures += 1
            health.total_requests += 1
            health.last_failure_time = time.monotonic()

            if health.failure_count >= self._failure_threshold:
                # Open circuit
                health.state = BackendState.UNHEALTHY
                health.cooldown_until = time.monotonic() + self._cooldown_seconds

    def mark_unhealthy(self, backend_id: str, reason: str) -> None:
        """Manually mark backend as unhealthy.

        Opens circuit immediately regardless of failure count.

        Args:
            backend_id: Backend identifier.
            reason: Reason for marking unhealthy (for logging).
        """
        with self._lock:
            health = self._health.get(backend_id)
            if health is None:
                return

            health.state = BackendState.UNHEALTHY
            health.cooldown_until = time.monotonic() + self._cooldown_seconds
            health.last_failure_time = time.monotonic()

    def get_available_backends(self) -> list["BackendSpec"]:
        """Get all available (healthy) backends.

        Returns:
            List of BackendSpecs that are currently available.
        """
        with self._lock:
            return [
                spec
                for spec in self._backends.values()
                if self.is_available(spec.backend_id)
            ]

    def get_all(self) -> list["BackendSpec"]:
        """Get all registered backends.

        Returns:
            List of all registered BackendSpecs.
        """
        with self._lock:
            return list(self._backends.values())

    def clear(self) -> None:
        """Clear all registered backends."""
        with self._lock:
            self._backends.clear()
            self._health.clear()
