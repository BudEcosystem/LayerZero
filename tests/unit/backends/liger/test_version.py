"""Tests for Liger version detection."""
from __future__ import annotations

import pytest

from layerzero.backends.liger.version import (
    detect_liger_version,
    detect_triton_version,
    get_liger_info,
    is_liger_available,
    is_triton_available,
    check_triton_compatibility,
)


class TestLigerAvailability:
    """Test Liger availability detection."""

    def test_is_liger_available_returns_bool(self) -> None:
        """is_liger_available returns boolean."""
        result = is_liger_available()
        assert isinstance(result, bool)

    def test_detect_version_returns_tuple_or_none(self) -> None:
        """detect_liger_version returns tuple or None."""
        result = detect_liger_version()
        assert result is None or (
            isinstance(result, tuple) and
            len(result) == 3 and
            all(isinstance(x, int) for x in result)
        )

    def test_version_consistent_with_availability(self) -> None:
        """Version is only available when package is available."""
        available = is_liger_available()
        version = detect_liger_version()

        if not available:
            assert version is None
        else:
            assert version is not None


class TestTritonAvailability:
    """Test Triton availability detection."""

    def test_is_triton_available_returns_bool(self) -> None:
        """is_triton_available returns boolean."""
        result = is_triton_available()
        assert isinstance(result, bool)

    def test_detect_triton_version_returns_tuple_or_none(self) -> None:
        """detect_triton_version returns tuple or None."""
        result = detect_triton_version()
        assert result is None or (
            isinstance(result, tuple) and
            len(result) == 3 and
            all(isinstance(x, int) for x in result)
        )

    def test_triton_version_consistent_with_availability(self) -> None:
        """Triton version is only available when package is available."""
        available = is_triton_available()
        version = detect_triton_version()

        if not available:
            assert version is None
        else:
            assert version is not None

    def test_triton_compatibility_returns_bool(self) -> None:
        """check_triton_compatibility returns boolean."""
        result = check_triton_compatibility()
        assert isinstance(result, bool)

    def test_triton_compatibility_requires_availability(self) -> None:
        """Triton must be available for compatibility check to pass."""
        available = is_triton_available()
        compatible = check_triton_compatibility()

        if not available:
            # Can't be compatible if not available
            assert not compatible


class TestLigerInfo:
    """Test Liger information retrieval."""

    def test_get_liger_info_returns_dict(self) -> None:
        """get_liger_info returns dict."""
        result = get_liger_info()
        assert isinstance(result, dict)

    def test_liger_info_has_available_key(self) -> None:
        """Liger info contains available key."""
        result = get_liger_info()
        assert "available" in result
        assert isinstance(result["available"], bool)

    def test_liger_info_has_version_key(self) -> None:
        """Liger info contains version key."""
        result = get_liger_info()
        assert "version" in result

    def test_liger_info_has_triton_key(self) -> None:
        """Liger info contains triton info."""
        result = get_liger_info()
        assert "triton_available" in result
        assert isinstance(result["triton_available"], bool)

    def test_liger_info_has_kernels_key(self) -> None:
        """Liger info contains kernels list."""
        result = get_liger_info()
        assert "available_kernels" in result
        assert isinstance(result["available_kernels"], list)


class TestLigerVersionParsing:
    """Test version string parsing."""

    @pytest.mark.skipif(
        not is_liger_available(),
        reason="Liger not installed"
    )
    def test_version_major_minor_patch(self) -> None:
        """Version has major.minor.patch components."""
        version = detect_liger_version()
        assert version is not None
        major, minor, patch = version
        assert major >= 0
        assert minor >= 0
        assert patch >= 0

    @pytest.mark.skipif(
        not is_liger_available(),
        reason="Liger not installed"
    )
    def test_liger_info_when_available(self) -> None:
        """Liger info contains actual data when available."""
        info = get_liger_info()
        assert info["available"] is True
        assert info["version"] is not None


class TestLigerWhenUnavailable:
    """Test behavior when Liger is not installed."""

    def test_unavailable_version_is_none(self) -> None:
        """When unavailable, version returns None (not error)."""
        result = detect_liger_version()
        assert result is None or isinstance(result, tuple)

    def test_unavailable_info_graceful(self) -> None:
        """When unavailable, info returns empty gracefully."""
        result = get_liger_info()
        assert isinstance(result, dict)
        assert "available" in result
        assert "version" in result
        assert "triton_available" in result
        assert "available_kernels" in result
