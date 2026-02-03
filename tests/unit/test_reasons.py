"""
Test suite for LayerZero Reason Codes

Tests reason code uniqueness, serialization, and category coverage.
Following TDD methodology - these tests define the expected behavior.
"""
import json
import pytest
from typing import Set


class TestReasonCodes:
    """Test reason code constants and properties."""

    def test_reason_codes_unique(self):
        """All reason codes must have unique string values.

        Rationale: Duplicate codes would break serialization and debugging.
        """
        from layerzero.reasons import (
            PLATFORM_MISMATCH, SM_TOO_OLD, SM_TOO_NEW,
            GPU_GENERATION_UNSUPPORTED, TENSOR_CORE_GEN_UNSUPPORTED,
            INSTRUCTION_SET_MISMATCH, DEVICE_CAPABILITY_UNSUPPORTED,
            DTYPE_UNSUPPORTED, QUANT_FORMAT_UNSUPPORTED,
            QUANT_ACCURACY_THRESHOLD_EXCEEDED, REQUANTIZATION_REQUIRED,
            HEAD_DIM_INVALID, HEAD_DIM_ALIGNMENT, HEAD_DIM_TOO_LARGE,
            SEQ_TOO_LONG, GQA_UNSUPPORTED, GQA_HEADS_MISMATCH,
            ATTN_MASK_UNSUPPORTED, ATTN_MASK_INVALID,
            CUDA_GRAPH_UNSAFE, CUDA_BLOCK_LIMIT_EXCEEDED,
            CUDA_GRID_DIM_EXCEEDED, NON_DETERMINISTIC,
            NOT_INSTALLED, BACKEND_IMPORT_FAILED, JIT_DISABLED, BACKEND_ERROR,
            NOT_CONTIGUOUS, STRIDE_LAST_DIM, MEMORY_HEADROOM_EXCEEDED,
            PLAN_BUCKET_MISS, TOKENIZER_ID_MISMATCH, VOCAB_HASH_MISMATCH,
            NORMALIZER_MISMATCH, MERGES_HASH_MISMATCH, ADDED_TOKENS_MISMATCH,
            SPECIAL_TOKENS_MISMATCH, KV_CACHE_LAYOUT_MISMATCH,
            KV_CACHE_DTYPE_MISMATCH, KV_STRATEGY_UNSUPPORTED,
            DRIVER_VERSION_UNSUPPORTED, VIRTUAL_MEMORY_EXHAUSTED,
            PACKED_WEIGHTS_REQUIRED, TP_INVARIANCE_REQUIRED,
            TP_SIZE_EXCEEDED, REDUCTION_ORDER_MISMATCH,
            SPEC_DECODE_PP_INCOMPATIBLE, SPEC_DECODE_DRAFT_TP_CONSTRAINT,
            SPEC_DECODE_KV_INCOMPATIBLE, SPEC_DECODE_ALGORITHM_UNSUPPORTED,
            CAPABILITIES_SCHEMA_MISMATCH
        )

        all_codes = [
            PLATFORM_MISMATCH, SM_TOO_OLD, SM_TOO_NEW,
            GPU_GENERATION_UNSUPPORTED, TENSOR_CORE_GEN_UNSUPPORTED,
            INSTRUCTION_SET_MISMATCH, DEVICE_CAPABILITY_UNSUPPORTED,
            DTYPE_UNSUPPORTED, QUANT_FORMAT_UNSUPPORTED,
            QUANT_ACCURACY_THRESHOLD_EXCEEDED, REQUANTIZATION_REQUIRED,
            HEAD_DIM_INVALID, HEAD_DIM_ALIGNMENT, HEAD_DIM_TOO_LARGE,
            SEQ_TOO_LONG, GQA_UNSUPPORTED, GQA_HEADS_MISMATCH,
            ATTN_MASK_UNSUPPORTED, ATTN_MASK_INVALID,
            CUDA_GRAPH_UNSAFE, CUDA_BLOCK_LIMIT_EXCEEDED,
            CUDA_GRID_DIM_EXCEEDED, NON_DETERMINISTIC,
            NOT_INSTALLED, BACKEND_IMPORT_FAILED, JIT_DISABLED, BACKEND_ERROR,
            NOT_CONTIGUOUS, STRIDE_LAST_DIM, MEMORY_HEADROOM_EXCEEDED,
            PLAN_BUCKET_MISS, TOKENIZER_ID_MISMATCH, VOCAB_HASH_MISMATCH,
            NORMALIZER_MISMATCH, MERGES_HASH_MISMATCH, ADDED_TOKENS_MISMATCH,
            SPECIAL_TOKENS_MISMATCH, KV_CACHE_LAYOUT_MISMATCH,
            KV_CACHE_DTYPE_MISMATCH, KV_STRATEGY_UNSUPPORTED,
            DRIVER_VERSION_UNSUPPORTED, VIRTUAL_MEMORY_EXHAUSTED,
            PACKED_WEIGHTS_REQUIRED, TP_INVARIANCE_REQUIRED,
            TP_SIZE_EXCEEDED, REDUCTION_ORDER_MISMATCH,
            SPEC_DECODE_PP_INCOMPATIBLE, SPEC_DECODE_DRAFT_TP_CONSTRAINT,
            SPEC_DECODE_KV_INCOMPATIBLE, SPEC_DECODE_ALGORITHM_UNSUPPORTED,
            CAPABILITIES_SCHEMA_MISMATCH
        ]

        assert len(all_codes) == len(set(all_codes)), \
            "Duplicate reason codes found"

    def test_reason_codes_are_strings(self):
        """All reason codes must be string type for serialization."""
        from layerzero.reasons import (
            PLATFORM_MISMATCH, SM_TOO_OLD, DTYPE_UNSUPPORTED,
            HEAD_DIM_INVALID, CUDA_GRAPH_UNSAFE, NOT_INSTALLED
        )

        codes = [PLATFORM_MISMATCH, SM_TOO_OLD, DTYPE_UNSUPPORTED,
                 HEAD_DIM_INVALID, CUDA_GRAPH_UNSAFE, NOT_INSTALLED]

        for code in codes:
            assert isinstance(code, str), f"{code} is not a string"

    def test_reason_codes_uppercase(self):
        """All reason codes should be SCREAMING_SNAKE_CASE for consistency."""
        from layerzero.reasons import (
            PLATFORM_MISMATCH, SM_TOO_OLD, DTYPE_UNSUPPORTED
        )

        for code in [PLATFORM_MISMATCH, SM_TOO_OLD, DTYPE_UNSUPPORTED]:
            assert code == code.upper(), f"{code} is not uppercase"
            assert " " not in code, f"{code} contains spaces"

    def test_platform_mismatch_reason(self):
        """PLATFORM_MISMATCH code exists and is string."""
        from layerzero.reasons import PLATFORM_MISMATCH
        assert PLATFORM_MISMATCH == "PLATFORM_MISMATCH"

    def test_sm_too_old_reason(self):
        """SM_TOO_OLD code exists."""
        from layerzero.reasons import SM_TOO_OLD
        assert SM_TOO_OLD == "SM_TOO_OLD"

    def test_dtype_unsupported_reason(self):
        """DTYPE_UNSUPPORTED code exists."""
        from layerzero.reasons import DTYPE_UNSUPPORTED
        assert DTYPE_UNSUPPORTED == "DTYPE_UNSUPPORTED"

    def test_head_dim_invalid_reason(self):
        """HEAD_DIM_INVALID code exists."""
        from layerzero.reasons import HEAD_DIM_INVALID
        assert HEAD_DIM_INVALID == "HEAD_DIM_INVALID"

    def test_cuda_graph_unsafe_reason(self):
        """CUDA_GRAPH_UNSAFE code exists."""
        from layerzero.reasons import CUDA_GRAPH_UNSAFE
        assert CUDA_GRAPH_UNSAFE == "CUDA_GRAPH_UNSAFE"

    def test_cuda_block_limit_exceeded_reason(self):
        """CUDA_BLOCK_LIMIT_EXCEEDED code exists."""
        from layerzero.reasons import CUDA_BLOCK_LIMIT_EXCEEDED
        assert CUDA_BLOCK_LIMIT_EXCEEDED == "CUDA_BLOCK_LIMIT_EXCEEDED"

    def test_tp_invariance_required_reason(self):
        """TP_INVARIANCE_REQUIRED code exists."""
        from layerzero.reasons import TP_INVARIANCE_REQUIRED
        assert TP_INVARIANCE_REQUIRED == "TP_INVARIANCE_REQUIRED"

    def test_minimum_50_reason_codes(self):
        """At least 50 reason codes must be defined."""
        from layerzero import reasons

        # Count all uppercase string constants (reason codes convention)
        reason_codes = [
            attr for attr in dir(reasons)
            if attr.isupper() and isinstance(getattr(reasons, attr), str)
        ]

        assert len(reason_codes) >= 50, \
            f"Expected >= 50 reason codes, found {len(reason_codes)}"


class TestReasonClass:
    """Test the Reason dataclass."""

    def test_reason_creation(self):
        """Reason dataclass can be created with code, message, category."""
        from layerzero.reasons import Reason, ReasonCategory

        reason = Reason(
            code="TEST_CODE",
            message="Test message",
            category=ReasonCategory.HARDWARE
        )

        assert reason.code == "TEST_CODE"
        assert reason.message == "Test message"
        assert reason.category == ReasonCategory.HARDWARE

    def test_reason_is_frozen(self):
        """Reason dataclass is immutable (frozen)."""
        from layerzero.reasons import Reason, ReasonCategory

        reason = Reason(
            code="TEST_CODE",
            message="Test message",
            category=ReasonCategory.HARDWARE
        )

        with pytest.raises(AttributeError):
            reason.code = "NEW_CODE"  # type: ignore

    def test_reason_is_hashable(self):
        """Reason dataclass is hashable (can be used in sets/dicts)."""
        from layerzero.reasons import Reason, ReasonCategory

        reason1 = Reason(
            code="TEST_CODE",
            message="Test message",
            category=ReasonCategory.HARDWARE
        )
        reason2 = Reason(
            code="TEST_CODE",
            message="Test message",
            category=ReasonCategory.HARDWARE
        )

        # Hashable means can be added to a set
        reason_set: Set[Reason] = {reason1, reason2}
        assert len(reason_set) == 1  # Same content = same hash

    def test_reason_serialization_to_dict(self):
        """Reason can be serialized to dict."""
        from layerzero.reasons import Reason, ReasonCategory

        reason = Reason(
            code="TEST_CODE",
            message="Test message",
            category=ReasonCategory.HARDWARE
        )

        d = reason.to_dict()
        assert d["code"] == "TEST_CODE"
        assert d["message"] == "Test message"
        assert d["category"] == "hardware"

    def test_reason_deserialization_from_dict(self):
        """Reason can be deserialized from dict."""
        from layerzero.reasons import Reason, ReasonCategory

        d = {
            "code": "TEST_CODE",
            "message": "Test message",
            "category": "hardware"
        }

        reason = Reason.from_dict(d)
        assert reason.code == "TEST_CODE"
        assert reason.message == "Test message"
        assert reason.category == ReasonCategory.HARDWARE

    def test_reason_serialization_roundtrip(self):
        """Reason serialize/deserialize preserves all fields."""
        from layerzero.reasons import Reason, ReasonCategory

        original = Reason(
            code="ROUNDTRIP_TEST",
            message="Test roundtrip serialization",
            category=ReasonCategory.DTYPE
        )

        # Roundtrip through JSON
        json_str = json.dumps(original.to_dict())
        d = json.loads(json_str)
        restored = Reason.from_dict(d)

        assert restored.code == original.code
        assert restored.message == original.message
        assert restored.category == original.category
        assert restored == original

    def test_reason_str_representation(self):
        """Reason __str__ includes code and message."""
        from layerzero.reasons import Reason, ReasonCategory

        reason = Reason(
            code="TEST_CODE",
            message="Test message",
            category=ReasonCategory.HARDWARE
        )

        s = str(reason)
        assert "TEST_CODE" in s
        assert "Test message" in s


class TestReasonCategory:
    """Test ReasonCategory enum."""

    def test_reason_category_hardware(self):
        """ReasonCategory.HARDWARE exists."""
        from layerzero.reasons import ReasonCategory
        assert ReasonCategory.HARDWARE.value == "hardware"

    def test_reason_category_dtype(self):
        """ReasonCategory.DTYPE exists."""
        from layerzero.reasons import ReasonCategory
        assert ReasonCategory.DTYPE.value == "dtype"

    def test_reason_category_shape(self):
        """ReasonCategory.SHAPE exists."""
        from layerzero.reasons import ReasonCategory
        assert ReasonCategory.SHAPE.value == "shape"

    def test_reason_category_cuda(self):
        """ReasonCategory.CUDA exists."""
        from layerzero.reasons import ReasonCategory
        assert ReasonCategory.CUDA.value == "cuda"

    def test_reason_category_all_values(self):
        """All expected categories exist."""
        from layerzero.reasons import ReasonCategory

        expected_categories = [
            "hardware", "dtype", "shape", "attention", "cuda",
            "backend", "memory", "tokenizer", "kv_cache",
            "quantization", "distributed", "speculative", "schema"
        ]

        actual_values = [c.value for c in ReasonCategory]

        for expected in expected_categories:
            assert expected in actual_values, \
                f"Missing category: {expected}"

    def test_reason_code_categories_complete(self):
        """All categories have at least one reason code."""
        from layerzero.reasons import ReasonCategory, ALL_REASON_CODES

        # ALL_REASON_CODES should be a dict mapping code -> category
        categories_with_codes: Set[ReasonCategory] = set()

        for code, category in ALL_REASON_CODES.items():
            categories_with_codes.add(category)

        for category in ReasonCategory:
            assert category in categories_with_codes, \
                f"Category {category} has no reason codes"
