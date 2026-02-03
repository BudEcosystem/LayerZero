"""Tests for HuggingFace Kernel Hub version detection."""
from __future__ import annotations

import pytest

from layerzero.backends.hf_kernels.version import (
    detect_hf_kernels_version,
    get_hf_kernels_info,
    is_hf_kernels_available,
)


class TestHFKernelsAvailability:
    """Test HF kernels availability detection."""

    def test_is_hf_kernels_available_returns_bool(self) -> None:
        """is_hf_kernels_available returns boolean."""
        result = is_hf_kernels_available()
        assert isinstance(result, bool)

    def test_detect_version_returns_tuple_or_none(self) -> None:
        """detect_hf_kernels_version returns tuple or None."""
        result = detect_hf_kernels_version()
        assert result is None or (
            isinstance(result, tuple) and
            len(result) == 3 and
            all(isinstance(x, int) for x in result)
        )

    def test_version_consistent_with_availability(self) -> None:
        """Version is only available when package is available."""
        available = is_hf_kernels_available()
        version = detect_hf_kernels_version()

        if not available:
            assert version is None


class TestHFKernelsInfo:
    """Test HF kernels information retrieval."""

    def test_get_info_returns_dict(self) -> None:
        """get_hf_kernels_info returns dictionary."""
        result = get_hf_kernels_info()
        assert isinstance(result, dict)

    def test_info_has_available_key(self) -> None:
        """Info contains available key."""
        result = get_hf_kernels_info()
        assert "available" in result
        assert isinstance(result["available"], bool)

    def test_info_has_version_key(self) -> None:
        """Info contains version key."""
        result = get_hf_kernels_info()
        assert "version" in result

    def test_info_has_hub_url(self) -> None:
        """Info contains hub URL."""
        result = get_hf_kernels_info()
        assert "hub_url" in result


class TestHFKernelsWhenUnavailable:
    """Test behavior when HF kernels is not available."""

    def test_unavailable_version_is_none(self) -> None:
        """When unavailable, version returns None (not error)."""
        result = detect_hf_kernels_version()
        # Either a valid version or None, should not raise
        assert result is None or isinstance(result, tuple)

    def test_unavailable_info_graceful(self) -> None:
        """When unavailable, info returns gracefully."""
        result = get_hf_kernels_info()
        assert isinstance(result, dict)
        assert "available" in result
