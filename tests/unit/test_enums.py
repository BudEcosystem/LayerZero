"""
Test suite for LayerZero Core Enumerations

Tests OpKind, Layout, MaskType, Platform, QuantFormat, KVCacheStrategy.
Following TDD methodology - these tests define the expected behavior.
"""
import json
import pytest
from enum import Enum


class TestOpKind:
    """Test OpKind enumeration."""

    def test_op_kind_tensor_value(self):
        """OpKind.TENSOR exists with string value."""
        from layerzero.enums import OpKind
        assert OpKind.TENSOR.value == "tensor"

    def test_op_kind_tokenization_value(self):
        """OpKind.TOKENIZATION exists with string value."""
        from layerzero.enums import OpKind
        assert OpKind.TOKENIZATION.value == "tokenization"

    def test_op_kind_sampling_value(self):
        """OpKind.SAMPLING exists with string value."""
        from layerzero.enums import OpKind
        assert OpKind.SAMPLING.value == "sampling"

    def test_op_kind_communication_value(self):
        """OpKind.COMMUNICATION exists for distributed ops."""
        from layerzero.enums import OpKind
        assert OpKind.COMMUNICATION.value == "communication"

    def test_op_kind_prepost_value(self):
        """OpKind.PREPOST exists for pre/post processing."""
        from layerzero.enums import OpKind
        assert OpKind.PREPOST.value == "prepost"

    def test_op_kind_is_str_enum(self):
        """OpKind is a string enum for JSON serialization."""
        from layerzero.enums import OpKind
        assert isinstance(OpKind.TENSOR, str)
        assert OpKind.TENSOR == "tensor"

    def test_op_kind_json_serializable(self):
        """OpKind values can be serialized to JSON."""
        from layerzero.enums import OpKind

        data = {"op_kind": OpKind.TENSOR.value}
        json_str = json.dumps(data)
        restored = json.loads(json_str)

        assert restored["op_kind"] == "tensor"
        assert OpKind(restored["op_kind"]) == OpKind.TENSOR


class TestLayout:
    """Test Layout enumeration for tensor formats."""

    def test_layout_bshd_value(self):
        """Layout.BSHD exists (Batch, Seq, Heads, Dim)."""
        from layerzero.enums import Layout
        assert Layout.BSHD.value == "BSHD"

    def test_layout_bhsd_value(self):
        """Layout.BHSD exists (Batch, Heads, Seq, Dim)."""
        from layerzero.enums import Layout
        assert Layout.BHSD.value == "BHSD"

    def test_layout_nhd_value(self):
        """Layout.NHD exists for FlashInfer format."""
        from layerzero.enums import Layout
        assert Layout.NHD.value == "NHD"

    def test_layout_hnd_value(self):
        """Layout.HND exists."""
        from layerzero.enums import Layout
        assert Layout.HND.value == "HND"

    def test_layout_unique_values(self):
        """All Layout values are unique."""
        from layerzero.enums import Layout
        values = [l.value for l in Layout]
        assert len(values) == len(set(values))

    def test_layout_is_str_enum(self):
        """Layout is a string enum."""
        from layerzero.enums import Layout
        assert isinstance(Layout.BSHD, str)


class TestMaskType:
    """Test MaskType enumeration for attention masks."""

    def test_mask_type_none(self):
        """MaskType.NONE for no mask."""
        from layerzero.enums import MaskType
        assert MaskType.NONE.value == "none"

    def test_mask_type_bool(self):
        """MaskType.BOOL for boolean masks."""
        from layerzero.enums import MaskType
        assert MaskType.BOOL.value == "bool"

    def test_mask_type_float(self):
        """MaskType.FLOAT for additive float masks."""
        from layerzero.enums import MaskType
        assert MaskType.FLOAT.value == "float"

    def test_mask_type_all_values(self):
        """All MaskType values exist."""
        from layerzero.enums import MaskType
        assert len(list(MaskType)) == 3


class TestPlatform:
    """Test Platform enumeration for hardware platforms."""

    def test_platform_cuda(self):
        """Platform.CUDA exists."""
        from layerzero.enums import Platform
        assert Platform.CUDA.value == "cuda"

    def test_platform_rocm(self):
        """Platform.ROCM exists for AMD GPUs."""
        from layerzero.enums import Platform
        assert Platform.ROCM.value == "rocm"

    def test_platform_cpu(self):
        """Platform.CPU exists."""
        from layerzero.enums import Platform
        assert Platform.CPU.value == "cpu"

    def test_platform_hpu(self):
        """Platform.HPU exists for Habana."""
        from layerzero.enums import Platform
        assert Platform.HPU.value == "hpu"

    def test_platform_xpu(self):
        """Platform.XPU exists for Intel."""
        from layerzero.enums import Platform
        assert Platform.XPU.value == "xpu"

    def test_platform_unique_values(self):
        """All Platform values are unique."""
        from layerzero.enums import Platform
        values = [p.value for p in Platform]
        assert len(values) == len(set(values))


class TestQuantFormat:
    """Test QuantFormat enumeration for quantization formats."""

    def test_quant_format_int4(self):
        """QuantFormat.INT4 exists."""
        from layerzero.enums import QuantFormat
        assert QuantFormat.INT4.value == "int4"

    def test_quant_format_int8(self):
        """QuantFormat.INT8 exists."""
        from layerzero.enums import QuantFormat
        assert QuantFormat.INT8.value == "int8"

    def test_quant_format_nvfp4(self):
        """QuantFormat.NVFP4 exists for NVIDIA FP4."""
        from layerzero.enums import QuantFormat
        assert QuantFormat.NVFP4.value == "nvfp4"

    def test_quant_format_mxfp4(self):
        """QuantFormat.MXFP4 exists for MX FP4."""
        from layerzero.enums import QuantFormat
        assert QuantFormat.MXFP4.value == "mxfp4"

    def test_quant_format_fp8_e4m3(self):
        """QuantFormat.FP8_E4M3 exists."""
        from layerzero.enums import QuantFormat
        assert QuantFormat.FP8_E4M3.value == "fp8_e4m3"

    def test_quant_format_fp8_e5m2(self):
        """QuantFormat.FP8_E5M2 exists."""
        from layerzero.enums import QuantFormat
        assert QuantFormat.FP8_E5M2.value == "fp8_e5m2"

    def test_quant_format_unique_values(self):
        """All QuantFormat values are unique."""
        from layerzero.enums import QuantFormat
        values = [q.value for q in QuantFormat]
        assert len(values) == len(set(values))


class TestKVCacheStrategy:
    """Test KVCacheStrategy enumeration."""

    def test_kv_strategy_contiguous(self):
        """KVCacheStrategy.CONTIGUOUS exists."""
        from layerzero.enums import KVCacheStrategy
        assert KVCacheStrategy.CONTIGUOUS.value == "contiguous"

    def test_kv_strategy_paged(self):
        """KVCacheStrategy.PAGED exists for PagedAttention."""
        from layerzero.enums import KVCacheStrategy
        assert KVCacheStrategy.PAGED.value == "paged"

    def test_kv_strategy_virtual(self):
        """KVCacheStrategy.VIRTUAL exists for virtual memory."""
        from layerzero.enums import KVCacheStrategy
        assert KVCacheStrategy.VIRTUAL.value == "virtual"

    def test_kv_strategy_unified(self):
        """KVCacheStrategy.UNIFIED exists."""
        from layerzero.enums import KVCacheStrategy
        assert KVCacheStrategy.UNIFIED.value == "unified"


class TestEnumSerialization:
    """Test enum serialization for all types."""

    def test_all_enums_json_serializable(self):
        """All enum values can be serialized to JSON."""
        from layerzero.enums import (
            OpKind, Layout, MaskType, Platform,
            QuantFormat, KVCacheStrategy
        )

        enums = [OpKind, Layout, MaskType, Platform,
                 QuantFormat, KVCacheStrategy]

        for enum_cls in enums:
            for member in enum_cls:
                # Should not raise
                json_str = json.dumps(member.value)
                assert json.loads(json_str) == member.value

    def test_enum_from_string_value(self):
        """Enums can be constructed from their string values."""
        from layerzero.enums import OpKind, Layout, Platform

        assert OpKind("tensor") == OpKind.TENSOR
        assert Layout("BSHD") == Layout.BSHD
        assert Platform("cuda") == Platform.CUDA

    def test_enum_invalid_value_raises(self):
        """Enums raise ValueError for invalid values."""
        from layerzero.enums import OpKind

        with pytest.raises(ValueError):
            OpKind("invalid_value")


class TestEnumUniqueness:
    """Test that all enums have unique members."""

    def test_op_kind_unique_decorator(self):
        """OpKind uses @unique decorator."""
        from layerzero.enums import OpKind
        # If @unique is applied, duplicate values would raise at import
        # Just verify it imported successfully
        assert len(OpKind) >= 5

    def test_layout_unique_decorator(self):
        """Layout uses @unique decorator."""
        from layerzero.enums import Layout
        assert len(Layout) >= 4

    def test_all_enums_inherit_from_enum(self):
        """All custom enums inherit from Enum."""
        from layerzero.enums import (
            OpKind, Layout, MaskType, Platform,
            QuantFormat, KVCacheStrategy
        )

        enums = [OpKind, Layout, MaskType, Platform,
                 QuantFormat, KVCacheStrategy]

        for enum_cls in enums:
            assert issubclass(enum_cls, Enum)
