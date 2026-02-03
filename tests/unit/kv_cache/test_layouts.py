"""Tests for KV cache layout handling."""
from __future__ import annotations

import pytest
import torch


class TestKVCacheLayout:
    """Tests for KV cache layout enum."""

    def test_nhd_layout_defined(self) -> None:
        """NHD layout (batch, seq, heads, dim) is defined."""
        from layerzero.kv_cache.layouts import KVCacheLayout

        assert hasattr(KVCacheLayout, "NHD")
        assert KVCacheLayout.NHD.value == "nhd"

    def test_hnd_layout_defined(self) -> None:
        """HND layout (batch, heads, seq, dim) is defined."""
        from layerzero.kv_cache.layouts import KVCacheLayout

        assert hasattr(KVCacheLayout, "HND")
        assert KVCacheLayout.HND.value == "hnd"

    def test_bnhd_layout_defined(self) -> None:
        """BNHD layout (blocks, batch, heads, dim) for paged."""
        from layerzero.kv_cache.layouts import KVCacheLayout

        assert hasattr(KVCacheLayout, "BNHD")
        assert KVCacheLayout.BNHD.value == "bnhd"

    def test_all_layouts_unique(self) -> None:
        """All layout values are unique."""
        from layerzero.kv_cache.layouts import KVCacheLayout

        values = [l.value for l in KVCacheLayout]
        assert len(values) == len(set(values))


class TestLayoutConversion:
    """Tests for KV cache layout conversion."""

    def test_nhd_to_hnd_conversion(self, sample_kv_tensor_nhd: torch.Tensor) -> None:
        """Convert NHD layout to HND."""
        from layerzero.kv_cache.layouts import convert_layout, KVCacheLayout

        # Input: (batch, seq, heads, dim)
        input_tensor = sample_kv_tensor_nhd
        batch, seq, heads, dim = input_tensor.shape

        output = convert_layout(
            input_tensor,
            from_layout=KVCacheLayout.NHD,
            to_layout=KVCacheLayout.HND,
        )

        # Output: (batch, heads, seq, dim)
        assert output.shape == (batch, heads, seq, dim)

        # Verify data integrity
        assert torch.allclose(
            output[:, :, :, :], input_tensor.transpose(1, 2)
        )

    def test_hnd_to_nhd_conversion(self, sample_kv_tensor_hnd: torch.Tensor) -> None:
        """Convert HND layout to NHD."""
        from layerzero.kv_cache.layouts import convert_layout, KVCacheLayout

        # Input: (batch, heads, seq, dim)
        input_tensor = sample_kv_tensor_hnd
        batch, heads, seq, dim = input_tensor.shape

        output = convert_layout(
            input_tensor,
            from_layout=KVCacheLayout.HND,
            to_layout=KVCacheLayout.NHD,
        )

        # Output: (batch, seq, heads, dim)
        assert output.shape == (batch, seq, heads, dim)

    def test_same_layout_returns_view(self, sample_kv_tensor_nhd: torch.Tensor) -> None:
        """Same layout conversion returns view (no copy)."""
        from layerzero.kv_cache.layouts import convert_layout, KVCacheLayout

        input_tensor = sample_kv_tensor_nhd

        output = convert_layout(
            input_tensor,
            from_layout=KVCacheLayout.NHD,
            to_layout=KVCacheLayout.NHD,
        )

        # Should share storage
        assert output.data_ptr() == input_tensor.data_ptr()

    def test_roundtrip_conversion(self, sample_kv_tensor_nhd: torch.Tensor) -> None:
        """Roundtrip conversion preserves data."""
        from layerzero.kv_cache.layouts import convert_layout, KVCacheLayout

        original = sample_kv_tensor_nhd.clone()

        # NHD -> HND -> NHD
        intermediate = convert_layout(
            original,
            from_layout=KVCacheLayout.NHD,
            to_layout=KVCacheLayout.HND,
        )

        result = convert_layout(
            intermediate,
            from_layout=KVCacheLayout.HND,
            to_layout=KVCacheLayout.NHD,
        )

        assert torch.allclose(result, original)


class TestLayoutInfo:
    """Tests for layout information utilities."""

    def test_get_layout_dims_nhd(self) -> None:
        """Get dimension order for NHD layout."""
        from layerzero.kv_cache.layouts import get_layout_dim_order, KVCacheLayout

        order = get_layout_dim_order(KVCacheLayout.NHD)

        # NHD: batch, seq, heads, dim
        assert order == ("batch", "seq", "heads", "dim")

    def test_get_layout_dims_hnd(self) -> None:
        """Get dimension order for HND layout."""
        from layerzero.kv_cache.layouts import get_layout_dim_order, KVCacheLayout

        order = get_layout_dim_order(KVCacheLayout.HND)

        # HND: batch, heads, seq, dim
        assert order == ("batch", "heads", "seq", "dim")

    def test_get_seq_dim_nhd(self) -> None:
        """Get sequence dimension index for NHD."""
        from layerzero.kv_cache.layouts import get_seq_dim, KVCacheLayout

        assert get_seq_dim(KVCacheLayout.NHD) == 1

    def test_get_seq_dim_hnd(self) -> None:
        """Get sequence dimension index for HND."""
        from layerzero.kv_cache.layouts import get_seq_dim, KVCacheLayout

        assert get_seq_dim(KVCacheLayout.HND) == 2

    def test_get_head_dim_index(self) -> None:
        """Get head dimension index."""
        from layerzero.kv_cache.layouts import get_head_dim, KVCacheLayout

        assert get_head_dim(KVCacheLayout.NHD) == 2
        assert get_head_dim(KVCacheLayout.HND) == 1


class TestLayoutValidation:
    """Tests for layout validation."""

    def test_validate_tensor_shape_nhd(self) -> None:
        """Validate tensor has correct NHD shape."""
        from layerzero.kv_cache.layouts import validate_layout_shape, KVCacheLayout

        # Valid shape: (batch, seq, heads, dim)
        tensor = torch.randn(2, 128, 8, 64)
        assert validate_layout_shape(tensor, KVCacheLayout.NHD, num_heads=8, head_dim=64)

    def test_validate_tensor_shape_hnd(self) -> None:
        """Validate tensor has correct HND shape."""
        from layerzero.kv_cache.layouts import validate_layout_shape, KVCacheLayout

        # Valid shape: (batch, heads, seq, dim)
        tensor = torch.randn(2, 8, 128, 64)
        assert validate_layout_shape(tensor, KVCacheLayout.HND, num_heads=8, head_dim=64)

    def test_validate_wrong_head_count(self) -> None:
        """Reject tensor with wrong head count."""
        from layerzero.kv_cache.layouts import validate_layout_shape, KVCacheLayout

        tensor = torch.randn(2, 128, 8, 64)
        assert not validate_layout_shape(
            tensor, KVCacheLayout.NHD, num_heads=16, head_dim=64
        )

    def test_validate_wrong_head_dim(self) -> None:
        """Reject tensor with wrong head dimension."""
        from layerzero.kv_cache.layouts import validate_layout_shape, KVCacheLayout

        tensor = torch.randn(2, 128, 8, 64)
        assert not validate_layout_shape(
            tensor, KVCacheLayout.NHD, num_heads=8, head_dim=128
        )


class TestLayoutSerialization:
    """Tests for layout serialization."""

    def test_layout_from_string(self) -> None:
        """Parse layout from string."""
        from layerzero.kv_cache.layouts import layout_from_string, KVCacheLayout

        assert layout_from_string("nhd") == KVCacheLayout.NHD
        assert layout_from_string("hnd") == KVCacheLayout.HND
        assert layout_from_string("bnhd") == KVCacheLayout.BNHD

    def test_layout_from_string_case_insensitive(self) -> None:
        """Layout parsing is case-insensitive."""
        from layerzero.kv_cache.layouts import layout_from_string, KVCacheLayout

        assert layout_from_string("NHD") == KVCacheLayout.NHD
        assert layout_from_string("Hnd") == KVCacheLayout.HND

    def test_invalid_layout_string_raises(self) -> None:
        """Invalid layout string raises ValueError."""
        from layerzero.kv_cache.layouts import layout_from_string

        with pytest.raises(ValueError, match="Unknown layout"):
            layout_from_string("invalid")
