"""Tests for FlashAttention version detection."""
from __future__ import annotations

import pytest

from layerzero.backends.flash_attn.version import (
    detect_flash_attn_version,
    is_flash_attn_available,
    select_fa_variant,
    FAVariant,
)


class TestFlashAttnVersionDetection:
    """Test FlashAttention version detection."""

    def test_detect_version_returns_tuple_or_none(self) -> None:
        """Test detect_flash_attn_version returns tuple or None."""
        version = detect_flash_attn_version()
        assert version is None or isinstance(version, tuple)

    def test_detect_version_tuple_has_3_parts(self) -> None:
        """Test version tuple has major, minor, patch."""
        version = detect_flash_attn_version()
        if version is not None:
            assert len(version) == 3
            assert all(isinstance(v, int) for v in version)

    def test_is_available_returns_bool(self) -> None:
        """Test is_flash_attn_available returns bool."""
        result = is_flash_attn_available()
        assert isinstance(result, bool)

    def test_is_available_matches_detect(self) -> None:
        """Test is_available matches detect result."""
        version = detect_flash_attn_version()
        available = is_flash_attn_available()

        if version is not None:
            assert available is True
        else:
            assert available is False


class TestFAVariant:
    """Test FAVariant enum."""

    def test_fa2_exists(self) -> None:
        """Test FA2 variant exists."""
        assert FAVariant.FA2 is not None

    def test_fa3_exists(self) -> None:
        """Test FA3 variant exists."""
        assert FAVariant.FA3 is not None

    def test_fa4_exists(self) -> None:
        """Test FA4 variant exists."""
        assert FAVariant.FA4 is not None


class TestSelectFAVariant:
    """Test FA variant selection by SM version."""

    def test_sm70_unsupported(self) -> None:
        """Test SM 7.0 (Volta) is unsupported."""
        variant = select_fa_variant(sm_version=(7, 0))
        assert variant is None

    def test_sm75_unsupported(self) -> None:
        """Test SM 7.5 (Turing) is unsupported."""
        variant = select_fa_variant(sm_version=(7, 5))
        assert variant is None

    def test_sm80_uses_fa2(self) -> None:
        """Test SM 8.0 (Ampere) uses FA2."""
        variant = select_fa_variant(sm_version=(8, 0))
        assert variant == FAVariant.FA2

    def test_sm86_uses_fa2(self) -> None:
        """Test SM 8.6 (Ampere) uses FA2."""
        variant = select_fa_variant(sm_version=(8, 6))
        assert variant == FAVariant.FA2

    def test_sm89_uses_fa2(self) -> None:
        """Test SM 8.9 (Ada) uses FA2."""
        variant = select_fa_variant(sm_version=(8, 9))
        assert variant == FAVariant.FA2

    def test_sm90_uses_fa3(self) -> None:
        """Test SM 9.0 (Hopper) uses FA3."""
        variant = select_fa_variant(sm_version=(9, 0))
        assert variant == FAVariant.FA3

    def test_sm100_uses_fa4(self) -> None:
        """Test SM 10.0 (Blackwell) uses FA4."""
        variant = select_fa_variant(sm_version=(10, 0))
        assert variant == FAVariant.FA4

    def test_sm120_uses_fa4(self) -> None:
        """Test future SM 12.0 uses FA4 (latest)."""
        variant = select_fa_variant(sm_version=(12, 0))
        assert variant == FAVariant.FA4
