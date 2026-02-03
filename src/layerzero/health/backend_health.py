"""
Backend health status tracking.

This module provides:
- HealthStatus: Health status enum
- HealthConfig: Configuration for health tracking
- BackendHealth: Health state for individual backend
- BackendHealthTracker: Tracks health across all backends
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels for a backend.

    Attributes:
        HEALTHY: Backend is operating normally.
        DEGRADED: Backend has some failures but is still usable.
        UNHEALTHY: Backend has too many failures and should be avoided.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class HealthConfig:
    """Configuration for backend health tracking.

    Attributes:
        degraded_threshold: Number of failures before degraded status.
        unhealthy_threshold: Number of failures before unhealthy status.
        recovery_success_count: Successes needed to recover from degraded.
        failure_window_seconds: Window for counting recent failures.
        cooldown_seconds: Time to wait before retrying unhealthy backend.
    """

    degraded_threshold: int = 3
    unhealthy_threshold: int = 5
    recovery_success_count: int = 2
    failure_window_seconds: float = 60.0
    cooldown_seconds: float = 30.0


class BackendHealth:
    """Health tracking for a single backend.

    Tracks failures and successes within a time window and computes
    the current health status based on configured thresholds.

    Thread-safe for concurrent access.

    Example:
        health = BackendHealth("flashinfer")

        # Record outcomes
        health.record_failure()
        health.record_success()

        # Check status
        if health.status == HealthStatus.UNHEALTHY:
            # Skip this backend
            ...
    """

    def __init__(
        self,
        backend_id: str,
        config: HealthConfig | None = None,
    ) -> None:
        """Initialize backend health tracker.

        Args:
            backend_id: Unique identifier for the backend.
            config: Health configuration. Uses defaults if None.
        """
        self._backend_id = backend_id
        self._config = config or HealthConfig()
        self._lock = threading.RLock()

        # Failure timestamps within the window
        self._failure_times: list[float] = []

        # Consecutive success count for recovery
        self._consecutive_successes: int = 0

        # Total counts
        self._total_failures: int = 0
        self._total_successes: int = 0

        # Last event timestamps
        self._last_failure_time: float = 0.0
        self._last_success_time: float = 0.0

    @property
    def backend_id(self) -> str:
        """Get backend identifier."""
        return self._backend_id

    @property
    def config(self) -> HealthConfig:
        """Get configuration."""
        return self._config

    @property
    def status(self) -> HealthStatus:
        """Get current health status.

        Status is computed based on recent failures within the window.

        Returns:
            Current HealthStatus.
        """
        with self._lock:
            failures = self._get_recent_failures()

            if failures >= self._config.unhealthy_threshold:
                return HealthStatus.UNHEALTHY
            elif failures >= self._config.degraded_threshold:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY

    @property
    def failure_count(self) -> int:
        """Get number of recent failures within window."""
        with self._lock:
            return self._get_recent_failures()

    @property
    def success_count(self) -> int:
        """Get total success count."""
        with self._lock:
            return self._total_successes

    @property
    def consecutive_successes(self) -> int:
        """Get consecutive success count."""
        with self._lock:
            return self._consecutive_successes

    @property
    def last_failure_time(self) -> float:
        """Get timestamp of last failure."""
        with self._lock:
            return self._last_failure_time

    @property
    def last_success_time(self) -> float:
        """Get timestamp of last success."""
        with self._lock:
            return self._last_success_time

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a backend failure.

        Args:
            error: Optional exception that caused the failure.
        """
        now = time.monotonic()

        with self._lock:
            self._failure_times.append(now)
            self._total_failures += 1
            self._last_failure_time = now
            self._consecutive_successes = 0

            # Prune old failures
            self._prune_old_failures(now)

            logger.debug(
                "Backend %s failure recorded. Recent failures: %d",
                self._backend_id,
                self._get_recent_failures(),
            )

    def record_success(self) -> None:
        """Record a backend success."""
        now = time.monotonic()

        with self._lock:
            self._total_successes += 1
            self._last_success_time = now
            self._consecutive_successes += 1

            # Check if we've recovered
            if self._consecutive_successes >= self._config.recovery_success_count:
                # Clear failure history on recovery
                self._failure_times.clear()
                logger.debug(
                    "Backend %s recovered after %d consecutive successes",
                    self._backend_id,
                    self._consecutive_successes,
                )

            logger.debug(
                "Backend %s success recorded. Consecutive: %d",
                self._backend_id,
                self._consecutive_successes,
            )

    def reset(self) -> None:
        """Reset all health state."""
        with self._lock:
            self._failure_times.clear()
            self._consecutive_successes = 0
            self._total_failures = 0
            self._total_successes = 0
            self._last_failure_time = 0.0
            self._last_success_time = 0.0

            logger.debug("Backend %s health reset", self._backend_id)

    def _get_recent_failures(self) -> int:
        """Get count of failures within the window.

        Must be called with lock held.

        Returns:
            Number of recent failures.
        """
        now = time.monotonic()
        self._prune_old_failures(now)
        return len(self._failure_times)

    def _prune_old_failures(self, now: float) -> None:
        """Remove failures outside the window.

        Must be called with lock held.

        Args:
            now: Current monotonic time.
        """
        cutoff = now - self._config.failure_window_seconds
        self._failure_times = [t for t in self._failure_times if t > cutoff]

    def to_dict(self) -> dict[str, Any]:
        """Convert health state to dictionary.

        Returns:
            Dictionary with health state.
        """
        with self._lock:
            return {
                "backend_id": self._backend_id,
                "status": self.status.value,
                "failure_count": self.failure_count,
                "success_count": self._total_successes,
                "consecutive_successes": self._consecutive_successes,
                "last_failure_time": self._last_failure_time,
                "last_success_time": self._last_success_time,
            }


class BackendHealthTracker:
    """Tracks health status for all backends.

    Provides a central registry for tracking backend health across
    the system. Thread-safe for concurrent access.

    Example:
        tracker = BackendHealthTracker()

        # Record outcomes
        tracker.record_success("flashinfer")
        tracker.record_failure("torch_sdpa")

        # Check health
        healthy_backends = tracker.get_healthy_backends()
    """

    def __init__(self, config: HealthConfig | None = None) -> None:
        """Initialize health tracker.

        Args:
            config: Health configuration for all backends.
        """
        self._config = config or HealthConfig()
        self._backends: dict[str, BackendHealth] = {}
        self._lock = threading.RLock()

    @property
    def config(self) -> HealthConfig:
        """Get configuration."""
        return self._config

    def get_or_create(self, backend_id: str) -> BackendHealth:
        """Get or create health tracker for backend.

        Args:
            backend_id: Backend identifier.

        Returns:
            BackendHealth for the specified backend.
        """
        with self._lock:
            if backend_id not in self._backends:
                self._backends[backend_id] = BackendHealth(
                    backend_id=backend_id,
                    config=self._config,
                )
            return self._backends[backend_id]

    def record_failure(
        self,
        backend_id: str,
        error: Exception | None = None,
    ) -> None:
        """Record failure for backend.

        Args:
            backend_id: Backend identifier.
            error: Optional exception that caused the failure.
        """
        health = self.get_or_create(backend_id)
        health.record_failure(error=error)

    def record_success(self, backend_id: str) -> None:
        """Record success for backend.

        Args:
            backend_id: Backend identifier.
        """
        health = self.get_or_create(backend_id)
        health.record_success()

    def get_status(self, backend_id: str) -> HealthStatus:
        """Get health status for backend.

        Args:
            backend_id: Backend identifier.

        Returns:
            HealthStatus for the backend.
        """
        health = self.get_or_create(backend_id)
        return health.status

    def is_healthy(self, backend_id: str) -> bool:
        """Check if backend is healthy.

        Args:
            backend_id: Backend identifier.

        Returns:
            True if backend status is HEALTHY.
        """
        return self.get_status(backend_id) == HealthStatus.HEALTHY

    def get_healthy_backends(self) -> list[str]:
        """Get list of healthy backend IDs.

        Returns:
            List of backend IDs with HEALTHY status.
        """
        with self._lock:
            return [
                bid for bid, health in self._backends.items()
                if health.status == HealthStatus.HEALTHY
            ]

    def reset_backend(self, backend_id: str) -> None:
        """Reset health state for backend.

        Args:
            backend_id: Backend identifier.
        """
        with self._lock:
            if backend_id in self._backends:
                self._backends[backend_id].reset()

    def reset_all(self) -> None:
        """Reset health state for all backends."""
        with self._lock:
            self._backends.clear()

    def get_all_health(self) -> dict[str, BackendHealth]:
        """Get health for all tracked backends.

        Returns:
            Dictionary mapping backend_id to BackendHealth.
        """
        with self._lock:
            return self._backends.copy()

    def __len__(self) -> int:
        """Get number of tracked backends."""
        with self._lock:
            return len(self._backends)

    def __contains__(self, backend_id: str) -> bool:
        """Check if backend is tracked."""
        with self._lock:
            return backend_id in self._backends


# Global health tracker instance
_global_tracker: BackendHealthTracker | None = None


def get_global_health_tracker() -> BackendHealthTracker:
    """Get global health tracker instance.

    Returns:
        Global BackendHealthTracker instance.
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = BackendHealthTracker()
    return _global_tracker
