"""Tests for FlashInfer layout conversion."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.flashinfer.layout import (
    bshd_to_nhd,
    bhsd_to_nhd,
    nhd_to_bshd,
    nhd_to_bhsd,
    hnd_to_nhd,
    nhd_to_hnd,
    convert_layout_for_flashinfer,
)
from layerzero.enums import Layout


class TestBSHDtoNHD:
    """Test BSHD to NHD conversion."""

    def test_basic_conversion(self, device: torch.device) -> None:
        """Test basic BSHD to NHD conversion."""
        # BSHD: (batch=2, seq=16, heads=8, dim=64)
        tensor = torch.randn(2, 16, 8, 64, device=device, dtype=torch.float16)

        result, seq_lens = bshd_to_nhd(tensor)

        # NHD: (num_tokens=32, heads=8, dim=64)
        assert result.shape == (32, 8, 64)
        assert seq_lens.tolist() == [16, 16]

    def test_variable_seq_lens(self, device: torch.device) -> None:
        """Test BSHD to NHD with variable sequence lengths."""
        tensor = torch.randn(2, 16, 8, 64, device=device, dtype=torch.float16)
        # Actual seq lens shorter than tensor dimension
        actual_seq_lens = torch.tensor([10, 12], device=device, dtype=torch.int32)

        result, _ = bshd_to_nhd(tensor, seq_lens=actual_seq_lens)

        # Should only include valid tokens: 10 + 12 = 22
        assert result.shape == (22, 8, 64)

    def test_preserves_values(self, device: torch.device) -> None:
        """Test conversion preserves tensor values."""
        batch, seq, heads, dim = 2, 4, 2, 8
        tensor = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float16)

        result, _ = bshd_to_nhd(tensor)

        # Value at [b=0, s=2, h=1, d=:] should be at [n=2, h=1, d=:]
        torch.testing.assert_close(tensor[0, 2, 1, :], result[2, 1, :])
        # Value at [b=1, s=1, h=0, d=:] should be at [n=seq+1=5, h=0, d=:]
        torch.testing.assert_close(tensor[1, 1, 0, :], result[seq + 1, 0, :])

    def test_preserves_dtype(self, device: torch.device) -> None:
        """Test conversion preserves dtype."""
        tensor = torch.randn(2, 16, 8, 64, device=device, dtype=torch.bfloat16)

        result, _ = bshd_to_nhd(tensor)

        assert result.dtype == torch.bfloat16

    def test_preserves_device(self, device: torch.device) -> None:
        """Test conversion preserves device."""
        tensor = torch.randn(2, 16, 8, 64, device=device, dtype=torch.float16)

        result, _ = bshd_to_nhd(tensor)

        assert result.device == tensor.device


class TestBHSDtoNHD:
    """Test BHSD to NHD conversion."""

    def test_basic_conversion(self, device: torch.device) -> None:
        """Test basic BHSD to NHD conversion."""
        # BHSD: (batch=2, heads=8, seq=16, dim=64)
        tensor = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)

        result, seq_lens = bhsd_to_nhd(tensor)

        # NHD: (num_tokens=32, heads=8, dim=64)
        assert result.shape == (32, 8, 64)
        assert seq_lens.tolist() == [16, 16]

    def test_preserves_values(self, device: torch.device) -> None:
        """Test conversion preserves tensor values."""
        batch, heads, seq, dim = 2, 2, 4, 8
        tensor = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.float16)

        result, _ = bhsd_to_nhd(tensor)

        # Value at [b=0, h=1, s=2, d=:] should be at [n=2, h=1, d=:]
        torch.testing.assert_close(tensor[0, 1, 2, :], result[2, 1, :])


class TestNHDtoBSHD:
    """Test NHD to BSHD conversion."""

    def test_basic_conversion(self, device: torch.device) -> None:
        """Test basic NHD to BSHD conversion."""
        # NHD: (num_tokens=32, heads=8, dim=64)
        tensor = torch.randn(32, 8, 64, device=device, dtype=torch.float16)
        seq_lens = torch.tensor([16, 16], device=device, dtype=torch.int32)

        result = nhd_to_bshd(tensor, seq_lens)

        # BSHD: (batch=2, seq=16, heads=8, dim=64)
        assert result.shape == (2, 16, 8, 64)

    def test_variable_seq_lens_padded(self, device: torch.device) -> None:
        """Test NHD to BSHD with variable seq lens (padded output)."""
        # NHD with different seq lens
        tensor = torch.randn(22, 8, 64, device=device, dtype=torch.float16)
        seq_lens = torch.tensor([10, 12], device=device, dtype=torch.int32)

        result = nhd_to_bshd(tensor, seq_lens)

        # Output padded to max seq len = 12
        assert result.shape == (2, 12, 8, 64)

    def test_roundtrip(self, device: torch.device) -> None:
        """Test BSHD -> NHD -> BSHD roundtrip."""
        original = torch.randn(2, 16, 8, 64, device=device, dtype=torch.float16)

        nhd, seq_lens = bshd_to_nhd(original)
        result = nhd_to_bshd(nhd, seq_lens)

        torch.testing.assert_close(result, original)


class TestNHDtoBHSD:
    """Test NHD to BHSD conversion."""

    def test_basic_conversion(self, device: torch.device) -> None:
        """Test basic NHD to BHSD conversion."""
        # NHD: (num_tokens=32, heads=8, dim=64)
        tensor = torch.randn(32, 8, 64, device=device, dtype=torch.float16)
        seq_lens = torch.tensor([16, 16], device=device, dtype=torch.int32)

        result = nhd_to_bhsd(tensor, seq_lens)

        # BHSD: (batch=2, heads=8, seq=16, dim=64)
        assert result.shape == (2, 8, 16, 64)


class TestHNDtoNHD:
    """Test HND to NHD conversion."""

    def test_basic_conversion(self, device: torch.device) -> None:
        """Test basic HND to NHD conversion."""
        # HND: (heads=8, num_tokens=32, dim=64)
        tensor = torch.randn(8, 32, 64, device=device, dtype=torch.float16)

        result = hnd_to_nhd(tensor)

        # NHD: (num_tokens=32, heads=8, dim=64)
        assert result.shape == (32, 8, 64)

    def test_preserves_values(self, device: torch.device) -> None:
        """Test conversion preserves tensor values."""
        tensor = torch.randn(4, 16, 32, device=device, dtype=torch.float16)

        result = hnd_to_nhd(tensor)

        # Value at [h=2, n=5, d=:] should be at [n=5, h=2, d=:]
        torch.testing.assert_close(tensor[2, 5, :], result[5, 2, :])


class TestNHDtoHND:
    """Test NHD to HND conversion."""

    def test_basic_conversion(self, device: torch.device) -> None:
        """Test basic NHD to HND conversion."""
        # NHD: (num_tokens=32, heads=8, dim=64)
        tensor = torch.randn(32, 8, 64, device=device, dtype=torch.float16)

        result = nhd_to_hnd(tensor)

        # HND: (heads=8, num_tokens=32, dim=64)
        assert result.shape == (8, 32, 64)

    def test_roundtrip(self, device: torch.device) -> None:
        """Test NHD -> HND -> NHD roundtrip."""
        original = torch.randn(32, 8, 64, device=device, dtype=torch.float16)

        hnd = nhd_to_hnd(original)
        result = hnd_to_nhd(hnd)

        torch.testing.assert_close(result, original)


class TestConvertLayoutForFlashInfer:
    """Test generic layout conversion function."""

    def test_bshd_to_nhd_via_convert(self, device: torch.device) -> None:
        """Test BSHD to NHD via convert_layout_for_flashinfer."""
        tensor = torch.randn(2, 16, 8, 64, device=device, dtype=torch.float16)

        result, metadata = convert_layout_for_flashinfer(
            tensor, from_layout=Layout.BSHD, to_layout=Layout.NHD
        )

        assert result.shape == (32, 8, 64)
        assert "seq_lens" in metadata

    def test_bhsd_to_nhd_via_convert(self, device: torch.device) -> None:
        """Test BHSD to NHD via convert_layout_for_flashinfer."""
        tensor = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)

        result, metadata = convert_layout_for_flashinfer(
            tensor, from_layout=Layout.BHSD, to_layout=Layout.NHD
        )

        assert result.shape == (32, 8, 64)

    def test_same_layout_no_op(self, device: torch.device) -> None:
        """Test same layout returns original tensor."""
        tensor = torch.randn(32, 8, 64, device=device, dtype=torch.float16)

        result, metadata = convert_layout_for_flashinfer(
            tensor, from_layout=Layout.NHD, to_layout=Layout.NHD
        )

        assert result is tensor

    def test_unsupported_layout_raises(self, device: torch.device) -> None:
        """Test unsupported layout raises ValueError."""
        tensor = torch.randn(32, 8, 64, device=device, dtype=torch.float16)

        # Direct HND to BSHD without seq_lens should fail
        with pytest.raises(ValueError, match="[Ll]ayout|[Ss]eq"):
            convert_layout_for_flashinfer(
                tensor, from_layout=Layout.HND, to_layout=Layout.BSHD
            )
