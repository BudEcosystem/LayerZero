"""Tests for backend health tracking."""
from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock, patch

from layerzero.health.backend_health import (
    BackendHealth,
    BackendHealthTracker,
    HealthStatus,
    HealthConfig,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self) -> None:
        """HealthStatus has expected values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    def test_status_ordering(self) -> None:
        """HealthStatus can be compared."""
        # HEALTHY > DEGRADED > UNHEALTHY
        assert HealthStatus.HEALTHY != HealthStatus.DEGRADED
        assert HealthStatus.DEGRADED != HealthStatus.UNHEALTHY


class TestHealthConfig:
    """Tests for HealthConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config values."""
        config = HealthConfig()

        assert config.degraded_threshold == 3
        assert config.unhealthy_threshold == 5
        assert config.recovery_success_count == 2
        assert config.failure_window_seconds == 60.0
        assert config.cooldown_seconds == 30.0

    def test_custom_values(self) -> None:
        """Custom config values accepted."""
        config = HealthConfig(
            degraded_threshold=5,
            unhealthy_threshold=10,
            recovery_success_count=3,
            failure_window_seconds=120.0,
            cooldown_seconds=60.0,
        )

        assert config.degraded_threshold == 5
        assert config.unhealthy_threshold == 10
        assert config.recovery_success_count == 3
        assert config.failure_window_seconds == 120.0
        assert config.cooldown_seconds == 60.0

    def test_config_immutable(self) -> None:
        """Config is immutable (frozen dataclass)."""
        config = HealthConfig()

        with pytest.raises(AttributeError):
            config.degraded_threshold = 10


class TestBackendHealth:
    """Tests for BackendHealth."""

    def test_health_status_healthy(self) -> None:
        """Initially healthy status."""
        health = BackendHealth("flashinfer")

        assert health.backend_id == "flashinfer"
        assert health.status == HealthStatus.HEALTHY
        assert health.failure_count == 0
        assert health.success_count == 0

    def test_record_failure(self, mock_time) -> None:
        """Recording failure increments counter."""
        health = BackendHealth("flashinfer")

        health.record_failure()

        assert health.failure_count == 1

    def test_record_success(self) -> None:
        """Recording success increments counter."""
        health = BackendHealth("flashinfer")

        health.record_success()

        assert health.success_count == 1

    def test_health_status_degraded(self, mock_time) -> None:
        """Status degraded after some failures."""
        config = HealthConfig(degraded_threshold=3, unhealthy_threshold=5)
        health = BackendHealth("flashinfer", config=config)

        # Record failures
        for _ in range(3):
            health.record_failure()
            mock_time["advance"](1.0)

        assert health.status == HealthStatus.DEGRADED

    def test_health_status_unhealthy(self, mock_time) -> None:
        """Status unhealthy after many failures."""
        config = HealthConfig(degraded_threshold=3, unhealthy_threshold=5)
        health = BackendHealth("flashinfer", config=config)

        # Record failures
        for _ in range(5):
            health.record_failure()
            mock_time["advance"](1.0)

        assert health.status == HealthStatus.UNHEALTHY

    def test_failure_counter_increment(self, mock_time) -> None:
        """Failure counter increments correctly."""
        health = BackendHealth("flashinfer")

        health.record_failure()
        assert health.failure_count == 1

        health.record_failure()
        assert health.failure_count == 2

        health.record_failure()
        assert health.failure_count == 3

    def test_success_resets_failure_counter(self, mock_time) -> None:
        """Success resets failure counter."""
        config = HealthConfig(recovery_success_count=1)
        health = BackendHealth("flashinfer", config=config)

        # Record some failures
        for _ in range(3):
            health.record_failure()
            mock_time["advance"](1.0)

        assert health.failure_count == 3

        # Record success
        health.record_success()

        # Failure count should be reset
        assert health.failure_count == 0

    def test_consecutive_successes_recover_health(self, mock_time) -> None:
        """Consecutive successes recover health status."""
        config = HealthConfig(
            degraded_threshold=3,
            unhealthy_threshold=5,
            recovery_success_count=2,
        )
        health = BackendHealth("flashinfer", config=config)

        # Get to degraded state
        for _ in range(3):
            health.record_failure()
            mock_time["advance"](1.0)

        assert health.status == HealthStatus.DEGRADED

        # First success
        health.record_success()
        assert health.status == HealthStatus.DEGRADED  # Still degraded

        # Second success - should recover
        health.record_success()
        assert health.status == HealthStatus.HEALTHY

    def test_failure_window_expiry(self, mock_time) -> None:
        """Old failures outside window don't count."""
        config = HealthConfig(
            degraded_threshold=3,
            failure_window_seconds=60.0,
        )
        health = BackendHealth("flashinfer", config=config)

        # Record failures
        for _ in range(2):
            health.record_failure()
            mock_time["advance"](1.0)

        assert health.failure_count == 2

        # Wait for window to expire
        mock_time["advance"](65.0)  # Beyond 60s window

        # Old failures should have expired
        health.record_failure()

        # Should only count the new failure
        assert health.failure_count == 1

    def test_last_failure_time_tracked(self, mock_time) -> None:
        """Last failure time is tracked."""
        health = BackendHealth("flashinfer")

        mock_time["set"](100.0)
        health.record_failure()

        assert health.last_failure_time == 100.0

        mock_time["set"](200.0)
        health.record_failure()

        assert health.last_failure_time == 200.0

    def test_last_success_time_tracked(self, mock_time) -> None:
        """Last success time is tracked."""
        health = BackendHealth("flashinfer")

        mock_time["set"](100.0)
        health.record_success()

        assert health.last_success_time == 100.0

    def test_reset_health(self, mock_time) -> None:
        """reset() clears all health state."""
        health = BackendHealth("flashinfer")

        # Build up some state
        for _ in range(5):
            health.record_failure()
            mock_time["advance"](1.0)

        health.record_success()

        health.reset()

        assert health.status == HealthStatus.HEALTHY
        assert health.failure_count == 0
        assert health.success_count == 0


class TestBackendHealthTracker:
    """Tests for BackendHealthTracker."""

    def test_tracker_creation(self) -> None:
        """Tracker can be created."""
        tracker = BackendHealthTracker()

        assert tracker is not None
        assert len(tracker) == 0

    def test_get_or_create_backend(self) -> None:
        """get_or_create returns health for backend."""
        tracker = BackendHealthTracker()

        health = tracker.get_or_create("flashinfer")

        assert health.backend_id == "flashinfer"
        assert health.status == HealthStatus.HEALTHY

    def test_get_or_create_same_instance(self) -> None:
        """get_or_create returns same instance for same backend."""
        tracker = BackendHealthTracker()

        health1 = tracker.get_or_create("flashinfer")
        health2 = tracker.get_or_create("flashinfer")

        assert health1 is health2

    def test_record_failure_for_backend(self, mock_time) -> None:
        """record_failure updates backend health."""
        tracker = BackendHealthTracker()

        tracker.record_failure("flashinfer")
        tracker.record_failure("flashinfer")

        health = tracker.get_or_create("flashinfer")
        assert health.failure_count == 2

    def test_record_success_for_backend(self) -> None:
        """record_success updates backend health."""
        tracker = BackendHealthTracker()

        tracker.record_success("flashinfer")

        health = tracker.get_or_create("flashinfer")
        assert health.success_count == 1

    def test_get_status(self, mock_time) -> None:
        """get_status returns backend status."""
        config = HealthConfig(degraded_threshold=2)
        tracker = BackendHealthTracker(config=config)

        assert tracker.get_status("flashinfer") == HealthStatus.HEALTHY

        tracker.record_failure("flashinfer")
        mock_time["advance"](1.0)
        tracker.record_failure("flashinfer")

        assert tracker.get_status("flashinfer") == HealthStatus.DEGRADED

    def test_is_healthy(self, mock_time) -> None:
        """is_healthy returns correct value."""
        config = HealthConfig(degraded_threshold=2)
        tracker = BackendHealthTracker(config=config)

        assert tracker.is_healthy("flashinfer") is True

        tracker.record_failure("flashinfer")
        mock_time["advance"](1.0)
        tracker.record_failure("flashinfer")

        assert tracker.is_healthy("flashinfer") is False

    def test_multiple_backends(self, mock_time) -> None:
        """Tracker tracks multiple backends independently."""
        config = HealthConfig(degraded_threshold=2)
        tracker = BackendHealthTracker(config=config)

        # Fail one backend
        tracker.record_failure("flashinfer")
        mock_time["advance"](1.0)
        tracker.record_failure("flashinfer")

        # Other backend stays healthy
        tracker.record_success("flash_attn")

        assert tracker.get_status("flashinfer") == HealthStatus.DEGRADED
        assert tracker.get_status("flash_attn") == HealthStatus.HEALTHY

    def test_get_healthy_backends(self, mock_time) -> None:
        """get_healthy_backends returns only healthy backends."""
        config = HealthConfig(degraded_threshold=2)
        tracker = BackendHealthTracker(config=config)

        # Create backends
        tracker.record_success("flashinfer")
        tracker.record_success("flash_attn")
        tracker.record_success("torch_sdpa")

        # Degrade one
        tracker.record_failure("torch_sdpa")
        mock_time["advance"](1.0)
        tracker.record_failure("torch_sdpa")

        healthy = tracker.get_healthy_backends()

        assert "flashinfer" in healthy
        assert "flash_attn" in healthy
        assert "torch_sdpa" not in healthy

    def test_reset_backend(self, mock_time) -> None:
        """reset_backend clears backend state."""
        tracker = BackendHealthTracker()

        tracker.record_failure("flashinfer")
        tracker.record_failure("flashinfer")

        tracker.reset_backend("flashinfer")

        health = tracker.get_or_create("flashinfer")
        assert health.failure_count == 0

    def test_reset_all(self, mock_time) -> None:
        """reset_all clears all backends."""
        tracker = BackendHealthTracker()

        tracker.record_failure("flashinfer")
        tracker.record_failure("flash_attn")

        tracker.reset_all()

        assert len(tracker) == 0

    def test_len_returns_backend_count(self) -> None:
        """__len__ returns number of tracked backends."""
        tracker = BackendHealthTracker()

        assert len(tracker) == 0

        tracker.record_success("flashinfer")
        assert len(tracker) == 1

        tracker.record_success("flash_attn")
        assert len(tracker) == 2

    def test_get_all_health(self) -> None:
        """get_all_health returns health for all backends."""
        tracker = BackendHealthTracker()

        tracker.record_success("flashinfer")
        tracker.record_success("flash_attn")

        all_health = tracker.get_all_health()

        assert "flashinfer" in all_health
        assert "flash_attn" in all_health
        assert isinstance(all_health["flashinfer"], BackendHealth)
