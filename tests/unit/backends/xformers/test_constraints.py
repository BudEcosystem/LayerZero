"""Tests for xFormers constraint checking."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.xformers.constraints import (
    check_xformers_constraints,
    check_sm_version,
    check_head_dim,
    check_dtype,
    check_stride,
    XFORMERS_MIN_SM,
    XFORMERS_MIN_HEAD_DIM,
    XFORMERS_MAX_HEAD_DIM,
)
from layerzero.reasons import (
    SM_TOO_OLD,
    HEAD_DIM_TOO_LARGE,
    HEAD_DIM_TOO_SMALL,
    HEAD_DIM_ALIGNMENT,
    DTYPE_UNSUPPORTED,
    STRIDE_LAST_DIM,
)


class TestSMVersionChecks:
    """Test SM version constraint checks."""

    def test_sm70_volta_accepted(self) -> None:
        """SM 7.0 (Volta) is accepted."""
        reasons = check_sm_version((7, 0))
        assert len(reasons) == 0

    def test_sm75_turing_accepted(self) -> None:
        """SM 7.5 (Turing) is accepted."""
        reasons = check_sm_version((7, 5))
        assert len(reasons) == 0

    def test_sm80_ampere_accepted(self) -> None:
        """SM 8.0 (Ampere) is accepted."""
        reasons = check_sm_version((8, 0))
        assert len(reasons) == 0

    def test_sm86_ada_accepted(self) -> None:
        """SM 8.6 (Ada) is accepted."""
        reasons = check_sm_version((8, 6))
        assert len(reasons) == 0

    def test_sm90_hopper_accepted(self) -> None:
        """SM 9.0 (Hopper) is accepted."""
        reasons = check_sm_version((9, 0))
        assert len(reasons) == 0

    def test_sm61_pascal_rejected(self) -> None:
        """SM 6.1 (Pascal) is rejected."""
        reasons = check_sm_version((6, 1))
        assert len(reasons) == 1
        assert SM_TOO_OLD in reasons[0].code

    def test_sm50_maxwell_rejected(self) -> None:
        """SM 5.0 (Maxwell) is rejected."""
        reasons = check_sm_version((5, 0))
        assert len(reasons) == 1
        assert SM_TOO_OLD in reasons[0].code

    def test_min_sm_constant(self) -> None:
        """XFORMERS_MIN_SM is (7, 0)."""
        assert XFORMERS_MIN_SM == (7, 0)


class TestHeadDimChecks:
    """Test head dimension constraint checks."""

    def test_head_dim_64_accepted(self) -> None:
        """head_dim=64 is accepted."""
        reasons = check_head_dim(64)
        assert len(reasons) == 0

    def test_head_dim_128_accepted(self) -> None:
        """head_dim=128 is accepted."""
        reasons = check_head_dim(128)
        assert len(reasons) == 0

    def test_head_dim_256_accepted(self) -> None:
        """head_dim=256 is accepted."""
        reasons = check_head_dim(256)
        assert len(reasons) == 0

    def test_head_dim_8_accepted(self) -> None:
        """head_dim=8 is accepted (minimum)."""
        reasons = check_head_dim(8)
        assert len(reasons) == 0

    def test_head_dim_320_rejected(self) -> None:
        """head_dim=320 is rejected (too large)."""
        reasons = check_head_dim(320)
        assert len(reasons) == 1
        assert HEAD_DIM_TOO_LARGE in reasons[0].code

    def test_head_dim_4_rejected(self) -> None:
        """head_dim=4 is rejected (too small)."""
        reasons = check_head_dim(4)
        assert len(reasons) == 1
        assert HEAD_DIM_TOO_SMALL in reasons[0].code

    def test_head_dim_84_rejected_alignment(self) -> None:
        """head_dim=84 rejected (not multiple of 8)."""
        reasons = check_head_dim(84)
        assert len(reasons) == 1
        assert HEAD_DIM_ALIGNMENT in reasons[0].code

    def test_head_dim_100_rejected_alignment(self) -> None:
        """head_dim=100 rejected (not multiple of 8)."""
        reasons = check_head_dim(100)
        assert len(reasons) == 1
        assert HEAD_DIM_ALIGNMENT in reasons[0].code

    def test_min_max_head_dim_constants(self) -> None:
        """Min/max head dim constants are correct."""
        assert XFORMERS_MIN_HEAD_DIM == 8
        assert XFORMERS_MAX_HEAD_DIM == 256


class TestDtypeChecks:
    """Test dtype constraint checks."""

    def test_fp16_accepted(self) -> None:
        """torch.float16 is accepted."""
        reasons = check_dtype(torch.float16)
        assert len(reasons) == 0

    def test_bf16_accepted(self) -> None:
        """torch.bfloat16 is accepted."""
        reasons = check_dtype(torch.bfloat16)
        assert len(reasons) == 0

    def test_fp32_rejected(self) -> None:
        """torch.float32 is rejected."""
        reasons = check_dtype(torch.float32)
        assert len(reasons) == 1
        assert DTYPE_UNSUPPORTED in reasons[0].code

    def test_fp64_rejected(self) -> None:
        """torch.float64 is rejected."""
        reasons = check_dtype(torch.float64)
        assert len(reasons) == 1
        assert DTYPE_UNSUPPORTED in reasons[0].code

    def test_int8_rejected(self) -> None:
        """torch.int8 is rejected (no quantization support)."""
        reasons = check_dtype(torch.int8)
        assert len(reasons) == 1
        assert DTYPE_UNSUPPORTED in reasons[0].code


class TestStrideChecks:
    """Test stride constraint checks."""

    def test_stride_1_accepted(self) -> None:
        """stride[-1]=1 is accepted."""
        reasons = check_stride(1)
        assert len(reasons) == 0

    def test_stride_2_rejected(self) -> None:
        """stride[-1]=2 is rejected."""
        reasons = check_stride(2)
        assert len(reasons) == 1
        assert STRIDE_LAST_DIM in reasons[0].code

    def test_stride_4_rejected(self) -> None:
        """stride[-1]=4 is rejected."""
        reasons = check_stride(4)
        assert len(reasons) == 1
        assert STRIDE_LAST_DIM in reasons[0].code


class TestCombinedConstraintCheck:
    """Test combined constraint checking."""

    def test_all_valid_no_reasons(self) -> None:
        """Valid config returns no reasons."""
        reasons = check_xformers_constraints(
            sm_version=(8, 6),
            head_dim=128,
            dtype=torch.float16,
            stride_last_dim=1,
        )
        assert len(reasons) == 0

    def test_invalid_sm_returns_reason(self) -> None:
        """Invalid SM returns reason."""
        reasons = check_xformers_constraints(
            sm_version=(6, 1),
            head_dim=128,
            dtype=torch.float16,
            stride_last_dim=1,
        )
        assert len(reasons) >= 1
        assert any(SM_TOO_OLD in r.code for r in reasons)

    def test_invalid_head_dim_returns_reason(self) -> None:
        """Invalid head_dim returns reason."""
        reasons = check_xformers_constraints(
            sm_version=(8, 6),
            head_dim=320,
            dtype=torch.float16,
            stride_last_dim=1,
        )
        assert len(reasons) >= 1
        assert any(HEAD_DIM_TOO_LARGE in r.code for r in reasons)

    def test_invalid_dtype_returns_reason(self) -> None:
        """Invalid dtype returns reason."""
        reasons = check_xformers_constraints(
            sm_version=(8, 6),
            head_dim=128,
            dtype=torch.float32,
            stride_last_dim=1,
        )
        assert len(reasons) >= 1
        assert any(DTYPE_UNSUPPORTED in r.code for r in reasons)

    def test_invalid_stride_returns_reason(self) -> None:
        """Invalid stride returns reason."""
        reasons = check_xformers_constraints(
            sm_version=(8, 6),
            head_dim=128,
            dtype=torch.float16,
            stride_last_dim=2,
        )
        assert len(reasons) >= 1
        assert any(STRIDE_LAST_DIM in r.code for r in reasons)

    def test_multiple_invalids_returns_all_reasons(self) -> None:
        """Multiple invalid constraints return all reasons."""
        reasons = check_xformers_constraints(
            sm_version=(6, 1),  # Invalid
            head_dim=320,       # Invalid
            dtype=torch.float32,  # Invalid
            stride_last_dim=2,  # Invalid
        )
        # Should have 4 reasons
        assert len(reasons) >= 4
