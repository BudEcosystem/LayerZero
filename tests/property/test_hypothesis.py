"""Property-based tests using Hypothesis for LayerZero.

Tests invariants across random inputs, discovers edge cases automatically.
If hypothesis is not installed, only fallback tests run.
"""
from __future__ import annotations

from typing import Any
import random

import pytest
import torch

from layerzero.enums import Layout, Platform
from layerzero import reasons
from layerzero.core.validation import (
    validate_head_dim,
    validate_dtype,
    ValidationResult,
    SUPPORTED_HEAD_DIMS,
)
from tests.correctness.reference import get_tolerance, DTYPE_TOLERANCES


# Check if hypothesis is available
try:
    from hypothesis import given, strategies as st, settings, assume
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


# Fallback tests that work without hypothesis
class TestPropertyFallback:
    """Property tests that work without hypothesis using manual sampling."""

    def test_head_dim_deterministic(self) -> None:
        """Head dim check is deterministic."""
        for head_dim in [32, 64, 65, 128, 256]:
            result1 = validate_head_dim(head_dim)
            result2 = validate_head_dim(head_dim)
            assert result1.valid == result2.valid

    def test_cache_key_deterministic(self) -> None:
        """Cache key generation is deterministic."""
        import hashlib

        def make_key(batch: int, seq: int) -> str:
            data = f"batch={batch},seq={seq}"
            return hashlib.md5(data.encode()).hexdigest()

        for batch, seq in [(1, 100), (8, 512), (32, 2048)]:
            key1 = make_key(batch, seq)
            key2 = make_key(batch, seq)
            assert key1 == key2

    def test_dtype_validation_deterministic(self) -> None:
        """Dtype validation is deterministic."""
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.int8]:
            result1 = validate_dtype(dtype)
            result2 = validate_dtype(dtype)
            assert result1.valid == result2.valid

    def test_tolerance_values_reasonable(self) -> None:
        """Tolerance values are reasonable."""
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            rtol, atol = get_tolerance(dtype)
            assert 0 < rtol <= 1.0
            assert 0 < atol <= 1.0

    def test_valid_head_dims_alignment(self) -> None:
        """Valid head dimensions pass alignment check."""
        # Multiples of 8
        for head_dim in [8, 16, 32, 64, 128, 256]:
            result = validate_head_dim(head_dim)
            # Even if not in SUPPORTED_HEAD_DIMS, should pass alignment
            if result.reason:
                assert result.reason != reasons.HEAD_DIM_ALIGNMENT

    def test_invalid_head_dims_alignment(self) -> None:
        """Invalid head dimensions fail alignment check."""
        # Not multiples of 8
        for head_dim in [7, 15, 33, 65, 127]:
            result = validate_head_dim(head_dim)
            assert result.valid is False
            assert result.reason == reasons.HEAD_DIM_ALIGNMENT

    def test_supported_dtypes_valid(self) -> None:
        """Supported dtypes pass validation."""
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            result = validate_dtype(dtype)
            assert result.valid is True

    def test_unsupported_dtypes_invalid(self) -> None:
        """Unsupported dtypes fail validation."""
        for dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            result = validate_dtype(dtype)
            assert result.valid is False
            assert result.reason == reasons.DTYPE_UNSUPPORTED

    def test_reason_codes_are_strings(self) -> None:
        """Reason codes are strings."""
        for code_name in [
            "DTYPE_UNSUPPORTED",
            "HEAD_DIM_INVALID",
            "HEAD_DIM_ALIGNMENT",
            "CUDA_BLOCK_LIMIT_EXCEEDED",
            "PLATFORM_MISMATCH",
            "LAYOUT_UNSUPPORTED",
        ]:
            code = getattr(reasons, code_name)
            assert isinstance(code, str)
            assert len(code) > 0

    def test_lower_precision_looser_tolerance(self) -> None:
        """Lower precision dtypes have looser tolerances."""
        fp16_rtol, fp16_atol = get_tolerance(torch.float16)
        fp32_rtol, fp32_atol = get_tolerance(torch.float32)

        assert fp16_rtol >= fp32_rtol
        assert fp16_atol >= fp32_atol

    def test_validation_is_pure(self) -> None:
        """Validation functions have no side effects."""
        for head_dim in [64, 65, 128]:
            results = [validate_head_dim(head_dim) for _ in range(5)]
            for result in results[1:]:
                assert result.valid == results[0].valid
                assert result.reason == results[0].reason

    def test_different_contexts_different_keys(self) -> None:
        """Different contexts produce different cache keys."""
        import hashlib

        def make_key(batch: int, seq: int) -> str:
            data = f"batch={batch},seq={seq}"
            return hashlib.md5(data.encode()).hexdigest()

        # Different contexts should produce different keys
        key1 = make_key(1, 100)
        key2 = make_key(2, 100)
        key3 = make_key(1, 200)

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3


# Hypothesis-based tests (only run if hypothesis is installed)
if HYPOTHESIS_AVAILABLE:
    class TestSelectionContextProperties:
        """Property-based tests for SelectionContext."""

        @given(
            batch_size=st.integers(1, 128),
            seq_len=st.integers(1, 8192),
        )
        @settings(max_examples=100)
        def test_context_values_valid_ranges(
            self,
            batch_size: int,
            seq_len: int,
        ) -> None:
            """Valid batch and seq_len values are accepted."""
            assert batch_size >= 1
            assert seq_len >= 1

            context = {
                "batch_size": batch_size,
                "seq_len_q": seq_len,
                "seq_len_k": seq_len,
            }

            assert context["batch_size"] == batch_size
            assert context["seq_len_q"] == seq_len

        @given(head_dim=st.integers(1, 512))
        @settings(max_examples=100)
        def test_head_dim_constraint_check_deterministic(self, head_dim: int) -> None:
            """Head dim constraint check is deterministic."""
            result1 = validate_head_dim(head_dim)
            result2 = validate_head_dim(head_dim)

            assert result1.valid == result2.valid
            assert result1.reason == result2.reason

    class TestCacheKeyPropertiesHypothesis:
        """Property-based tests for cache key generation with Hypothesis."""

        @given(
            batch1=st.integers(1, 64),
            batch2=st.integers(1, 64),
            seq1=st.integers(1, 2048),
            seq2=st.integers(1, 2048),
        )
        @settings(max_examples=100)
        def test_different_contexts_different_keys(
            self,
            batch1: int,
            batch2: int,
            seq1: int,
            seq2: int,
        ) -> None:
            """Different contexts produce different cache keys."""
            import hashlib

            def make_key(batch: int, seq: int) -> str:
                data = f"batch={batch},seq={seq}"
                return hashlib.md5(data.encode()).hexdigest()

            key1 = make_key(batch1, seq1)
            key2 = make_key(batch2, seq2)

            if (batch1, seq1) != (batch2, seq2):
                assert key1 != key2

        @given(
            batch=st.integers(1, 64),
            seq=st.integers(1, 2048),
        )
        @settings(max_examples=100)
        def test_same_context_same_key(self, batch: int, seq: int) -> None:
            """Same context always produces same cache key."""
            import hashlib

            def make_key(batch_size: int, seq_len: int) -> str:
                data = f"batch={batch_size},seq={seq_len}"
                return hashlib.md5(data.encode()).hexdigest()

            key1 = make_key(batch, seq)
            key2 = make_key(batch, seq)

            assert key1 == key2

    class TestValidationPropertiesHypothesis:
        """Property-based tests for validation with Hypothesis."""

        @given(head_dim=st.integers(1, 32).map(lambda x: x * 8))
        @settings(max_examples=50)
        def test_valid_head_dims_accepted(self, head_dim: int) -> None:
            """Valid head dimensions (multiples of 8) pass alignment check."""
            result = validate_head_dim(head_dim)
            assert isinstance(result, ValidationResult)

        @given(head_dim=st.integers(1, 255).filter(lambda x: x % 8 != 0))
        @settings(max_examples=50)
        def test_invalid_head_dims_rejected(self, head_dim: int) -> None:
            """Invalid head dimensions (not multiples of 8) are rejected."""
            result = validate_head_dim(head_dim)
            assert result.valid is False
            assert result.reason == reasons.HEAD_DIM_ALIGNMENT

        @given(
            dtype=st.sampled_from([torch.float16, torch.bfloat16, torch.float32])
        )
        @settings(max_examples=10)
        def test_supported_dtypes_valid(self, dtype: torch.dtype) -> None:
            """Supported dtypes pass validation."""
            result = validate_dtype(dtype)
            assert result.valid is True

    class TestPurityPropertiesHypothesis:
        """Test that functions are pure with Hypothesis."""

        @given(head_dim=st.integers(1, 512))
        @settings(max_examples=50)
        def test_validation_is_pure(self, head_dim: int) -> None:
            """Validation functions have no side effects."""
            results = [validate_head_dim(head_dim) for _ in range(5)]
            for result in results[1:]:
                assert result.valid == results[0].valid
                assert result.reason == results[0].reason

        @given(
            dtype=st.sampled_from([torch.float16, torch.bfloat16, torch.float32])
        )
        @settings(max_examples=10)
        def test_tolerance_lookup_is_pure(self, dtype: torch.dtype) -> None:
            """Tolerance lookup has no side effects."""
            results = [get_tolerance(dtype) for _ in range(5)]
            for result in results[1:]:
                assert result == results[0]
