"""Tests for FlashInfer version detection."""
from __future__ import annotations

import pytest

from layerzero.backends.flashinfer.version import (
    detect_flashinfer_version,
    get_flashinfer_backend_info,
    is_flashinfer_available,
    is_jit_cache_available,
)


class TestFlashInferAvailability:
    """Test FlashInfer availability detection."""

    def test_is_flashinfer_available_returns_bool(self) -> None:
        """is_flashinfer_available returns boolean."""
        result = is_flashinfer_available()
        assert isinstance(result, bool)

    def test_detect_version_returns_tuple_or_none(self) -> None:
        """detect_flashinfer_version returns tuple or None."""
        result = detect_flashinfer_version()
        assert result is None or (
            isinstance(result, tuple) and
            len(result) == 3 and
            all(isinstance(x, int) for x in result)
        )

    def test_version_consistent_with_availability(self) -> None:
        """Version is only available when package is available."""
        available = is_flashinfer_available()
        version = detect_flashinfer_version()

        if not available:
            assert version is None
        else:
            assert version is not None

    def test_is_jit_cache_available_returns_bool(self) -> None:
        """is_jit_cache_available returns boolean."""
        result = is_jit_cache_available()
        assert isinstance(result, bool)


class TestFlashInferBackendInfo:
    """Test FlashInfer backend information retrieval."""

    def test_get_backend_info_returns_dict(self) -> None:
        """get_flashinfer_backend_info returns dict."""
        result = get_flashinfer_backend_info()
        assert isinstance(result, dict)

    def test_backend_info_has_available_key(self) -> None:
        """Backend info contains available key."""
        result = get_flashinfer_backend_info()
        assert "available" in result
        assert isinstance(result["available"], bool)

    def test_backend_info_has_version_key(self) -> None:
        """Backend info contains version key."""
        result = get_flashinfer_backend_info()
        assert "version" in result

    def test_backend_info_has_backends_key(self) -> None:
        """Backend info contains backends key."""
        result = get_flashinfer_backend_info()
        assert "backends" in result
        assert isinstance(result["backends"], list)

    def test_backend_info_has_jit_cache_key(self) -> None:
        """Backend info contains jit_cache key."""
        result = get_flashinfer_backend_info()
        assert "jit_cache" in result
        assert isinstance(result["jit_cache"], bool)


class TestFlashInferVersionParsing:
    """Test version string parsing."""

    @pytest.mark.skipif(
        not is_flashinfer_available(),
        reason="FlashInfer not installed"
    )
    def test_version_major_minor_patch(self) -> None:
        """Version has major.minor.patch components."""
        version = detect_flashinfer_version()
        assert version is not None
        major, minor, patch = version
        assert major >= 0
        assert minor >= 0
        assert patch >= 0

    @pytest.mark.skipif(
        not is_flashinfer_available(),
        reason="FlashInfer not installed"
    )
    def test_version_minimum_supported(self) -> None:
        """FlashInfer version is at least 0.5.0."""
        version = detect_flashinfer_version()
        assert version is not None
        # We support 0.5.0+
        assert version >= (0, 5, 0), f"FlashInfer {version} < 0.5.0"


class TestFlashInferWhenUnavailable:
    """Test behavior when FlashInfer is not installed."""

    def test_unavailable_version_is_none(self) -> None:
        """When unavailable, version returns None (not error)."""
        # This test always passes - it verifies graceful handling
        result = detect_flashinfer_version()
        # If not available, should be None, otherwise tuple
        assert result is None or isinstance(result, tuple)

    def test_unavailable_backend_info_graceful(self) -> None:
        """When unavailable, backend info returns empty gracefully."""
        result = get_flashinfer_backend_info()
        assert isinstance(result, dict)
        # Should always have these keys even when unavailable
        assert "available" in result
        assert "version" in result
        assert "backends" in result
