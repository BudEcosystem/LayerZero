"""Tests for FlashAttention layout conversion."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.flash_attn.layout import (
    bhsd_to_bshd,
    bshd_to_bhsd,
    convert_layout,
)
from layerzero.enums import Layout


class TestBHSDtoBSHD:
    """Test BHSD to BSHD conversion."""

    def test_basic_conversion(self, device: torch.device) -> None:
        """Test basic BHSD to BSHD conversion."""
        # BHSD: (batch=2, heads=8, seq=16, dim=64)
        tensor = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)

        result = bhsd_to_bshd(tensor)

        # BSHD: (batch=2, seq=16, heads=8, dim=64)
        assert result.shape == (2, 16, 8, 64)

    def test_preserves_values(self, device: torch.device) -> None:
        """Test conversion preserves tensor values."""
        tensor = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)

        result = bhsd_to_bshd(tensor)

        # Value at [b, h, s, d] in BHSD should be at [b, s, h, d] in BSHD
        assert torch.allclose(tensor[0, 0, 5, :], result[0, 5, 0, :])
        assert torch.allclose(tensor[1, 3, 10, :], result[1, 10, 3, :])

    def test_preserves_dtype(self, device: torch.device) -> None:
        """Test conversion preserves dtype."""
        tensor = torch.randn(2, 8, 16, 64, device=device, dtype=torch.bfloat16)

        result = bhsd_to_bshd(tensor)

        assert result.dtype == torch.bfloat16

    def test_preserves_device(self, device: torch.device) -> None:
        """Test conversion preserves device."""
        tensor = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)

        result = bhsd_to_bshd(tensor)

        assert result.device == tensor.device


class TestBSHDtoBHSD:
    """Test BSHD to BHSD conversion."""

    def test_basic_conversion(self, device: torch.device) -> None:
        """Test basic BSHD to BHSD conversion."""
        # BSHD: (batch=2, seq=16, heads=8, dim=64)
        tensor = torch.randn(2, 16, 8, 64, device=device, dtype=torch.float16)

        result = bshd_to_bhsd(tensor)

        # BHSD: (batch=2, heads=8, seq=16, dim=64)
        assert result.shape == (2, 8, 16, 64)

    def test_preserves_values(self, device: torch.device) -> None:
        """Test conversion preserves tensor values."""
        tensor = torch.randn(2, 16, 8, 64, device=device, dtype=torch.float16)

        result = bshd_to_bhsd(tensor)

        # Value at [b, s, h, d] in BSHD should be at [b, h, s, d] in BHSD
        assert torch.allclose(tensor[0, 5, 0, :], result[0, 0, 5, :])
        assert torch.allclose(tensor[1, 10, 3, :], result[1, 3, 10, :])

    def test_roundtrip(self, device: torch.device) -> None:
        """Test roundtrip conversion."""
        original = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)

        # BHSD -> BSHD -> BHSD
        bshd = bhsd_to_bshd(original)
        result = bshd_to_bhsd(bshd)

        torch.testing.assert_close(result, original)


class TestConvertLayout:
    """Test generic layout conversion."""

    def test_bhsd_to_bshd_via_convert(self, device: torch.device) -> None:
        """Test BHSD to BSHD via convert_layout."""
        tensor = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)

        result = convert_layout(tensor, from_layout=Layout.BHSD, to_layout=Layout.BSHD)

        assert result.shape == (2, 16, 8, 64)

    def test_bshd_to_bhsd_via_convert(self, device: torch.device) -> None:
        """Test BSHD to BHSD via convert_layout."""
        tensor = torch.randn(2, 16, 8, 64, device=device, dtype=torch.float16)

        result = convert_layout(tensor, from_layout=Layout.BSHD, to_layout=Layout.BHSD)

        assert result.shape == (2, 8, 16, 64)

    def test_same_layout_no_op(self, device: torch.device) -> None:
        """Test same layout returns original tensor."""
        tensor = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)

        result = convert_layout(tensor, from_layout=Layout.BHSD, to_layout=Layout.BHSD)

        # Should be the same tensor (no copy)
        assert result is tensor

    def test_unsupported_layout_raises(self, device: torch.device) -> None:
        """Test unsupported layout raises ValueError."""
        tensor = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)

        # NHD is a ragged layout, not compatible with BHSD conversion
        with pytest.raises(ValueError, match="[Ll]ayout"):
            convert_layout(tensor, from_layout=Layout.BHSD, to_layout=Layout.NHD)
