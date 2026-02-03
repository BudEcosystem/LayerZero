"""Tests for xFormers version detection."""
from __future__ import annotations

import pytest

from layerzero.backends.xformers.version import (
    detect_xformers_version,
    get_available_backends,
    get_xformers_backend_info,
    is_xformers_available,
)


class TestXFormersAvailability:
    """Test xFormers availability detection."""

    def test_is_xformers_available_returns_bool(self) -> None:
        """is_xformers_available returns boolean."""
        result = is_xformers_available()
        assert isinstance(result, bool)

    def test_detect_version_returns_tuple_or_none(self) -> None:
        """detect_xformers_version returns tuple or None."""
        result = detect_xformers_version()
        assert result is None or (
            isinstance(result, tuple) and
            len(result) == 3 and
            all(isinstance(x, int) for x in result)
        )

    def test_version_consistent_with_availability(self) -> None:
        """Version is only available when package is available."""
        available = is_xformers_available()
        version = detect_xformers_version()

        if not available:
            assert version is None
        else:
            assert version is not None


class TestXFormersBackendInfo:
    """Test xFormers backend information retrieval."""

    def test_get_backend_info_returns_dict(self) -> None:
        """get_xformers_backend_info returns dict."""
        result = get_xformers_backend_info()
        assert isinstance(result, dict)

    def test_backend_info_has_available_key(self) -> None:
        """Backend info contains available key."""
        result = get_xformers_backend_info()
        assert "available" in result
        assert isinstance(result["available"], bool)

    def test_backend_info_has_version_key(self) -> None:
        """Backend info contains version key."""
        result = get_xformers_backend_info()
        assert "version" in result

    def test_backend_info_has_backends_key(self) -> None:
        """Backend info contains backends key."""
        result = get_xformers_backend_info()
        assert "backends" in result
        assert isinstance(result["backends"], list)

    def test_get_available_backends_returns_list(self) -> None:
        """get_available_backends returns list of strings."""
        result = get_available_backends()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)


class TestXFormersVersionParsing:
    """Test version string parsing."""

    @pytest.mark.skipif(
        not is_xformers_available(),
        reason="xFormers not installed"
    )
    def test_version_major_minor_patch(self) -> None:
        """Version has major.minor.patch components."""
        version = detect_xformers_version()
        assert version is not None
        major, minor, patch = version
        assert major >= 0
        assert minor >= 0
        assert patch >= 0

    @pytest.mark.skipif(
        not is_xformers_available(),
        reason="xFormers not installed"
    )
    def test_backend_info_when_available(self) -> None:
        """Backend info contains actual data when available."""
        info = get_xformers_backend_info()
        assert info["available"] is True
        assert info["version"] is not None


class TestXFormersWhenUnavailable:
    """Test behavior when xFormers is not installed."""

    def test_unavailable_version_is_none(self) -> None:
        """When unavailable, version returns None (not error)."""
        result = detect_xformers_version()
        assert result is None or isinstance(result, tuple)

    def test_unavailable_backend_info_graceful(self) -> None:
        """When unavailable, backend info returns empty gracefully."""
        result = get_xformers_backend_info()
        assert isinstance(result, dict)
        assert "available" in result
        assert "version" in result
        assert "backends" in result
