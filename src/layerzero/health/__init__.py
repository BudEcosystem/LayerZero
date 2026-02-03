"""
LayerZero Backend Health Tracking Module

Provides health status tracking and circuit breaker pattern for
backend resilience and fault tolerance.

Main components:
- BackendHealth: Health status for individual backend
- BackendHealthTracker: Tracks health across all backends
- CircuitBreaker: Circuit breaker pattern implementation

Usage:
    from layerzero.health import (
        BackendHealthTracker,
        CircuitBreaker,
        HealthStatus,
    )

    tracker = BackendHealthTracker()

    # Record outcomes
    tracker.record_success("flashinfer")
    tracker.record_failure("torch_sdpa")

    # Check health
    if tracker.is_healthy("flashinfer"):
        # Use backend
        ...
"""
from __future__ import annotations

from layerzero.health.backend_health import (
    BackendHealth,
    BackendHealthTracker,
    HealthConfig,
    HealthStatus,
)
from layerzero.health.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)

__all__ = [
    # Backend health
    "BackendHealth",
    "BackendHealthTracker",
    "HealthConfig",
    "HealthStatus",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
]
