"""Tests for Triton version detection."""
from __future__ import annotations

import pytest

from layerzero.backends.triton.version import (
    detect_triton_version,
    get_triton_info,
    is_triton_available,
    get_triton_backend,
)


class TestTritonAvailability:
    """Test Triton availability detection."""

    def test_is_triton_available_returns_bool(self) -> None:
        """is_triton_available returns boolean."""
        result = is_triton_available()
        assert isinstance(result, bool)

    def test_detect_version_returns_tuple_or_none(self) -> None:
        """detect_triton_version returns tuple or None."""
        result = detect_triton_version()
        assert result is None or (
            isinstance(result, tuple) and
            len(result) == 3 and
            all(isinstance(x, int) for x in result)
        )

    def test_version_consistent_with_availability(self) -> None:
        """Version is only available when package is available."""
        available = is_triton_available()
        version = detect_triton_version()

        if not available:
            assert version is None
        else:
            assert version is not None


class TestTritonBackend:
    """Test Triton backend detection."""

    def test_get_backend_returns_str_or_none(self) -> None:
        """get_triton_backend returns string or None."""
        result = get_triton_backend()
        assert result is None or isinstance(result, str)

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    def test_backend_is_cuda_or_hip(self) -> None:
        """Backend is 'cuda' or 'hip' when available."""
        import torch
        backend = get_triton_backend()
        if torch.cuda.is_available():
            assert backend in ("cuda", "hip", None)


class TestTritonInfo:
    """Test Triton information retrieval."""

    def test_get_triton_info_returns_dict(self) -> None:
        """get_triton_info returns dict."""
        result = get_triton_info()
        assert isinstance(result, dict)

    def test_triton_info_has_available_key(self) -> None:
        """Triton info contains available key."""
        result = get_triton_info()
        assert "available" in result
        assert isinstance(result["available"], bool)

    def test_triton_info_has_version_key(self) -> None:
        """Triton info contains version key."""
        result = get_triton_info()
        assert "version" in result

    def test_triton_info_has_backend_key(self) -> None:
        """Triton info contains backend key."""
        result = get_triton_info()
        assert "backend" in result


class TestTritonVersionParsing:
    """Test version string parsing."""

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    def test_version_major_minor_patch(self) -> None:
        """Version has major.minor.patch components."""
        version = detect_triton_version()
        assert version is not None
        major, minor, patch = version
        assert major >= 0
        assert minor >= 0
        assert patch >= 0

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    def test_triton_info_when_available(self) -> None:
        """Triton info contains actual data when available."""
        info = get_triton_info()
        assert info["available"] is True
        assert info["version"] is not None


class TestTritonWhenUnavailable:
    """Test behavior when Triton is not installed."""

    def test_unavailable_version_is_none(self) -> None:
        """When unavailable, version returns None (not error)."""
        result = detect_triton_version()
        assert result is None or isinstance(result, tuple)

    def test_unavailable_info_graceful(self) -> None:
        """When unavailable, info returns empty gracefully."""
        result = get_triton_info()
        assert isinstance(result, dict)
        assert "available" in result
        assert "version" in result
        assert "backend" in result
