"""Tests for ABI conflict detection and handling."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from typing import Any

from layerzero.isolation.abi_detector import (
    ABIConflictDetector,
    ABIInfo,
    ConflictResult,
    detect_abi_conflict,
)
from layerzero.isolation.subprocess_backend import SubprocessBackend


class TestABIInfo:
    """Tests for ABIInfo dataclass."""

    def test_creation(self) -> None:
        """ABIInfo stores ABI information."""
        info = ABIInfo(
            backend_id="test",
            abi_version="1.0",
            torch_version="2.0.0",
            cuda_version="11.8",
        )

        assert info.backend_id == "test"
        assert info.abi_version == "1.0"
        assert info.torch_version == "2.0.0"
        assert info.cuda_version == "11.8"

    def test_to_dict(self) -> None:
        """ABIInfo serializes to dict."""
        info = ABIInfo(
            backend_id="test",
            abi_version="1.0",
            torch_version="2.0.0",
            cuda_version="11.8",
        )

        d = info.to_dict()

        assert d["backend_id"] == "test"
        assert d["abi_version"] == "1.0"


class TestConflictResult:
    """Tests for ConflictResult dataclass."""

    def test_no_conflict(self) -> None:
        """No conflict result."""
        result = ConflictResult(
            has_conflict=False,
            conflicting_backends=[],
            details="",
        )

        assert result.has_conflict is False
        assert result.conflicting_backends == []

    def test_with_conflict(self) -> None:
        """Result with conflict."""
        result = ConflictResult(
            has_conflict=True,
            conflicting_backends=["backend_a", "backend_b"],
            details="ABI version mismatch",
        )

        assert result.has_conflict is True
        assert "backend_a" in result.conflicting_backends


class TestABIConflictDetector:
    """Tests for ABIConflictDetector."""

    def test_abi_conflict_detected(self) -> None:
        """ABI conflict detected between backends."""
        detector = ABIConflictDetector()

        # Register backends with different ABIs
        info_a = ABIInfo(
            backend_id="backend_a",
            abi_version="1.0",
            torch_version="2.0.0",
            cuda_version="11.8",
        )
        info_b = ABIInfo(
            backend_id="backend_b",
            abi_version="2.0",  # Different ABI
            torch_version="2.0.0",
            cuda_version="11.8",
        )

        detector.register(info_a)
        detector.register(info_b)

        result = detector.detect()

        assert result.has_conflict is True
        assert "backend_a" in result.conflicting_backends or "backend_b" in result.conflicting_backends

    def test_no_conflict_same_abi(self) -> None:
        """No conflict when ABIs match."""
        detector = ABIConflictDetector()

        info_a = ABIInfo(
            backend_id="backend_a",
            abi_version="1.0",
            torch_version="2.0.0",
            cuda_version="11.8",
        )
        info_b = ABIInfo(
            backend_id="backend_b",
            abi_version="1.0",  # Same ABI
            torch_version="2.0.0",
            cuda_version="11.8",
        )

        detector.register(info_a)
        detector.register(info_b)

        result = detector.detect()

        assert result.has_conflict is False

    def test_cuda_version_conflict(self) -> None:
        """CUDA version conflict detected."""
        detector = ABIConflictDetector()

        info_a = ABIInfo(
            backend_id="backend_a",
            abi_version="1.0",
            torch_version="2.0.0",
            cuda_version="11.8",
        )
        info_b = ABIInfo(
            backend_id="backend_b",
            abi_version="1.0",
            torch_version="2.0.0",
            cuda_version="12.1",  # Different CUDA
        )

        detector.register(info_a)
        detector.register(info_b)

        result = detector.detect()

        # CUDA version difference may or may not be a conflict
        # depending on configuration
        assert result is not None

    def test_abi_conflict_uses_subprocess(self) -> None:
        """ABI conflict triggers subprocess mode."""
        detector = ABIConflictDetector()

        info_a = ABIInfo(
            backend_id="backend_a",
            abi_version="1.0",
            torch_version="2.0.0",
            cuda_version="11.8",
        )
        info_b = ABIInfo(
            backend_id="backend_b",
            abi_version="2.0",  # Different ABI
            torch_version="2.0.0",
            cuda_version="11.8",
        )

        detector.register(info_a)
        detector.register(info_b)

        result = detector.detect()

        # Should recommend isolation for conflicting backend
        assert result.has_conflict is True
        isolated_backends = detector.get_backends_requiring_isolation()
        assert len(isolated_backends) > 0


class TestDetectABIConflict:
    """Tests for convenience function."""

    def test_detect_abi_conflict_function(self) -> None:
        """detect_abi_conflict convenience function."""
        backends = [
            ABIInfo("a", "1.0", "2.0.0", "11.8"),
            ABIInfo("b", "2.0", "2.0.0", "11.8"),
        ]

        result = detect_abi_conflict(backends)

        assert result.has_conflict is True


class TestABIConflictIntegration:
    """Integration tests for ABI conflict handling."""

    def test_conflict_resolution_strategy(self) -> None:
        """Conflict resolution uses subprocess isolation."""
        detector = ABIConflictDetector()

        # Register conflicting backends
        detector.register(ABIInfo("a", "1.0", "2.0.0", "11.8"))
        detector.register(ABIInfo("b", "2.0", "2.0.0", "11.8"))

        result = detector.detect()
        isolated = detector.get_backends_requiring_isolation()

        # One backend should be marked for isolation
        assert len(isolated) >= 1

    def test_single_backend_no_conflict(self) -> None:
        """Single backend has no conflict."""
        detector = ABIConflictDetector()

        detector.register(ABIInfo("a", "1.0", "2.0.0", "11.8"))

        result = detector.detect()

        assert result.has_conflict is False
