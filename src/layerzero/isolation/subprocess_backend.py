"""
Subprocess backend for isolated execution.

This module provides:
- SubprocessBackendConfig: Configuration for subprocess backend
- SubprocessState: State of subprocess
- SubprocessError: Error from subprocess
- SubprocessBackend: Isolated subprocess backend
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any

logger = logging.getLogger(__name__)


class SubprocessError(Exception):
    """Error from subprocess backend."""

    pass


@unique
class SubprocessState(str, Enum):
    """State of subprocess.

    Attributes:
        NOT_STARTED: Subprocess has not been started.
        RUNNING: Subprocess is running.
        STOPPED: Subprocess has been stopped.
        FAILED: Subprocess has failed.
    """

    NOT_STARTED = "not_started"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class SubprocessBackendConfig:
    """Configuration for subprocess backend.

    Attributes:
        backend_id: Unique backend identifier.
        timeout_seconds: Timeout for operations.
        max_retries: Maximum retry attempts.
        restart_on_failure: Restart subprocess on failure.
        python_executable: Python executable path.
    """

    backend_id: str
    timeout_seconds: float = 30.0
    max_retries: int = 3
    restart_on_failure: bool = True
    python_executable: str | None = None


class SubprocessBackend:
    """Isolated subprocess backend.

    Runs backend operations in a separate subprocess to handle
    ABI conflicts and provide isolation.

    Can be used as context manager for automatic start/stop.

    Example:
        config = SubprocessBackendConfig(backend_id="isolated_backend")

        with SubprocessBackend(config=config) as backend:
            result = backend.execute(request)
    """

    def __init__(self, config: SubprocessBackendConfig) -> None:
        """Initialize subprocess backend.

        Args:
            config: Backend configuration.
        """
        self._config = config
        self._state = SubprocessState.NOT_STARTED
        self._process: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def config(self) -> SubprocessBackendConfig:
        """Get configuration."""
        return self._config

    @property
    def state(self) -> SubprocessState:
        """Get current state."""
        with self._lock:
            return self._state

    def start(self) -> None:
        """Start subprocess.

        Raises:
            SubprocessError: If subprocess fails to start.
        """
        with self._lock:
            if self._state == SubprocessState.RUNNING:
                logger.warning("Subprocess already running")
                return

            try:
                self._process = self._spawn_process()
                self._state = SubprocessState.RUNNING
                logger.info(
                    "Started subprocess for backend %s",
                    self._config.backend_id,
                )
            except Exception as e:
                self._state = SubprocessState.FAILED
                raise SubprocessError(f"Failed to start subprocess: {e}") from e

    def stop(self) -> None:
        """Stop subprocess."""
        with self._lock:
            if self._process is not None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=5.0)
                except Exception as e:
                    logger.warning("Error stopping subprocess: %s", e)
                    self._process.kill()

                self._process = None

            self._state = SubprocessState.STOPPED
            logger.info(
                "Stopped subprocess for backend %s",
                self._config.backend_id,
            )

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute request in subprocess.

        Args:
            request: Request dictionary.

        Returns:
            Response dictionary.

        Raises:
            SubprocessError: If execution fails.
        """
        if self._state != SubprocessState.RUNNING:
            raise SubprocessError("Subprocess not running")

        for attempt in range(self._config.max_retries):
            try:
                return self._execute_with_timeout(request)
            except SubprocessError as e:
                logger.warning(
                    "Subprocess execution failed (attempt %d/%d): %s",
                    attempt + 1,
                    self._config.max_retries,
                    e,
                )

                if self._config.restart_on_failure:
                    self._restart_process()

        raise SubprocessError(
            f"Subprocess execution failed after {self._config.max_retries} retries"
        )

    def _execute_with_timeout(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute request with timeout.

        Args:
            request: Request dictionary.

        Returns:
            Response dictionary.

        Raises:
            SubprocessError: If timeout or error.
        """
        future = self._executor.submit(self._send_request, request)

        try:
            return future.result(timeout=self._config.timeout_seconds)
        except FuturesTimeoutError:
            raise SubprocessError(
                f"Request timeout after {self._config.timeout_seconds}s"
            )
        except Exception as e:
            raise SubprocessError(f"Request failed: {e}")

    def _spawn_process(self) -> subprocess.Popen:
        """Spawn subprocess.

        Returns:
            Subprocess Popen object.
        """
        python = self._config.python_executable or sys.executable

        # Spawn subprocess with backend worker
        return subprocess.Popen(
            [python, "-m", "layerzero.isolation.worker", self._config.backend_id],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _restart_process(self) -> None:
        """Restart subprocess after failure."""
        logger.info("Restarting subprocess for backend %s", self._config.backend_id)

        with self._lock:
            if self._process is not None:
                try:
                    self._process.kill()
                except Exception:
                    pass
                self._process = None

            try:
                self._process = self._spawn_process()
                self._state = SubprocessState.RUNNING
            except Exception as e:
                self._state = SubprocessState.FAILED
                raise SubprocessError(f"Failed to restart subprocess: {e}") from e

    def _send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send request to subprocess.

        Args:
            request: Request dictionary.

        Returns:
            Response dictionary.
        """
        # Placeholder - actual implementation would use IPC
        import json

        if self._process is None:
            raise SubprocessError("Process not started")

        # Send request
        request_json = json.dumps(request).encode('utf-8') + b'\n'
        self._process.stdin.write(request_json)
        self._process.stdin.flush()

        # Read response
        response_line = self._process.stdout.readline()
        if not response_line:
            raise SubprocessError("No response from subprocess")

        return json.loads(response_line.decode('utf-8'))

    def __enter__(self) -> SubprocessBackend:
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.stop()
