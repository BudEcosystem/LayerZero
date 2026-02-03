"""Tests for subprocess backend isolation."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Any

from layerzero.isolation.subprocess_backend import (
    SubprocessBackend,
    SubprocessBackendConfig,
    SubprocessState,
    SubprocessError,
)


class TestSubprocessBackendConfig:
    """Tests for SubprocessBackendConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config values."""
        config = SubprocessBackendConfig(backend_id="test")

        assert config.backend_id == "test"
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 3
        assert config.restart_on_failure is True

    def test_custom_values(self) -> None:
        """Custom config values."""
        config = SubprocessBackendConfig(
            backend_id="custom",
            timeout_seconds=60.0,
            max_retries=5,
            restart_on_failure=False,
        )

        assert config.backend_id == "custom"
        assert config.timeout_seconds == 60.0
        assert config.max_retries == 5
        assert config.restart_on_failure is False

    def test_config_immutable(self) -> None:
        """Config is immutable."""
        config = SubprocessBackendConfig(backend_id="test")

        with pytest.raises(AttributeError):
            config.timeout_seconds = 100.0


class TestSubprocessState:
    """Tests for SubprocessState enum."""

    def test_not_started(self) -> None:
        """NOT_STARTED state."""
        assert SubprocessState.NOT_STARTED.value == "not_started"

    def test_running(self) -> None:
        """RUNNING state."""
        assert SubprocessState.RUNNING.value == "running"

    def test_stopped(self) -> None:
        """STOPPED state."""
        assert SubprocessState.STOPPED.value == "stopped"

    def test_failed(self) -> None:
        """FAILED state."""
        assert SubprocessState.FAILED.value == "failed"


class TestSubprocessBackend:
    """Tests for SubprocessBackend."""

    def test_subprocess_spawn(self, mock_isolated_backend_config) -> None:
        """Subprocess spawned for isolated backend."""
        config = SubprocessBackendConfig(
            backend_id=mock_isolated_backend_config["backend_id"],
        )
        backend = SubprocessBackend(config=config)

        with patch.object(backend, '_spawn_process') as mock_spawn:
            mock_spawn.return_value = MagicMock()
            backend.start()

        mock_spawn.assert_called_once()
        assert backend.state == SubprocessState.RUNNING

    def test_subprocess_communication(
        self,
        mock_isolated_backend_config,
        mock_request,
        mock_response,
    ) -> None:
        """IPC communication works."""
        config = SubprocessBackendConfig(
            backend_id=mock_isolated_backend_config["backend_id"],
        )
        backend = SubprocessBackend(config=config)

        with patch.object(backend, '_spawn_process'):
            backend.start()

        with patch.object(backend, '_send_request', return_value=mock_response):
            response = backend.execute(mock_request)

        assert response is not None
        assert response["status"] == "success"

    def test_subprocess_result_returned(
        self,
        mock_isolated_backend_config,
        mock_request,
        mock_response,
    ) -> None:
        """Result returned from subprocess."""
        config = SubprocessBackendConfig(
            backend_id=mock_isolated_backend_config["backend_id"],
        )
        backend = SubprocessBackend(config=config)

        with patch.object(backend, '_spawn_process'):
            backend.start()

        with patch.object(backend, '_send_request', return_value=mock_response):
            response = backend.execute(mock_request)

        assert response["output"] == "tensor_ref_4"
        assert response["metadata"]["kernel_time_ms"] == 1.5

    def test_subprocess_failure_handled(
        self,
        mock_isolated_backend_config,
        mock_request,
    ) -> None:
        """Subprocess failure handled gracefully."""
        config = SubprocessBackendConfig(
            backend_id=mock_isolated_backend_config["backend_id"],
            max_retries=2,
            restart_on_failure=True,
        )
        backend = SubprocessBackend(config=config)

        with patch.object(backend, '_spawn_process'):
            backend.start()

        # Simulate failure then recovery
        call_count = [0]
        def mock_send(request):
            call_count[0] += 1
            if call_count[0] < 2:
                raise SubprocessError("Connection lost")
            return {"status": "success", "output": "recovered"}

        with patch.object(backend, '_send_request', side_effect=mock_send):
            with patch.object(backend, '_restart_process'):
                response = backend.execute(mock_request)

        assert response["status"] == "success"

    def test_subprocess_stop(self, mock_isolated_backend_config) -> None:
        """Subprocess can be stopped."""
        config = SubprocessBackendConfig(
            backend_id=mock_isolated_backend_config["backend_id"],
        )
        backend = SubprocessBackend(config=config)

        mock_process = MagicMock()
        with patch.object(backend, '_spawn_process', return_value=mock_process):
            backend.start()

        backend.stop()

        assert backend.state == SubprocessState.STOPPED

    def test_subprocess_not_started_error(
        self,
        mock_isolated_backend_config,
        mock_request,
    ) -> None:
        """Error when executing on not-started subprocess."""
        config = SubprocessBackendConfig(
            backend_id=mock_isolated_backend_config["backend_id"],
        )
        backend = SubprocessBackend(config=config)

        # Don't start subprocess
        with pytest.raises(SubprocessError):
            backend.execute(mock_request)

    def test_subprocess_timeout_handled(
        self,
        mock_isolated_backend_config,
        mock_request,
    ) -> None:
        """Timeout is handled."""
        config = SubprocessBackendConfig(
            backend_id=mock_isolated_backend_config["backend_id"],
            timeout_seconds=0.1,
            max_retries=1,  # Single attempt to check timeout directly
        )
        backend = SubprocessBackend(config=config)

        with patch.object(backend, '_spawn_process'):
            backend.start()

        def mock_send_slow(request):
            import time
            time.sleep(0.5)
            return {"status": "success"}

        with patch.object(backend, '_send_request', side_effect=mock_send_slow):
            with pytest.raises(SubprocessError) as exc_info:
                backend.execute(mock_request)

        # After exhausting retries, should have failed due to timeout
        assert "timeout" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()


class TestSubprocessBackendContext:
    """Tests for subprocess backend context manager."""

    def test_context_manager_starts_stops(self, mock_isolated_backend_config) -> None:
        """Context manager starts and stops subprocess."""
        config = SubprocessBackendConfig(
            backend_id=mock_isolated_backend_config["backend_id"],
        )

        with patch.object(SubprocessBackend, '_spawn_process'):
            with SubprocessBackend(config=config) as backend:
                assert backend.state == SubprocessState.RUNNING

        assert backend.state == SubprocessState.STOPPED
