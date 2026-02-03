"""Pytest fixtures for backend health tracking tests."""
from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_time():
    """Mock time.monotonic for deterministic testing."""
    current_time = [0.0]

    def mock_monotonic():
        return current_time[0]

    def advance(seconds: float):
        current_time[0] += seconds

    with patch("time.monotonic", side_effect=mock_monotonic):
        yield {
            "advance": advance,
            "get": lambda: current_time[0],
            "set": lambda t: current_time.__setitem__(0, t),
        }
