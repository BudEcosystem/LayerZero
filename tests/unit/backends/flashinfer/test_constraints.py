"""Tests for FlashInfer constraint checking."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.flashinfer.constraints import (
    check_flashinfer_constraints,
    check_sm_version,
    check_head_dim,
    check_dtype,
    check_gqa_compatibility,
    FLASHINFER_MIN_SM,
    FLASHINFER_SUPPORTED_HEAD_DIMS,
)
from layerzero.reasons import (
    SM_TOO_OLD,
    HEAD_DIM_TOO_LARGE,
    HEAD_DIM_TOO_SMALL,
    HEAD_DIM_ALIGNMENT,
    DTYPE_UNSUPPORTED,
    GQA_UNSUPPORTED,
)


class TestSMVersionChecks:
    """Test SM version constraint checks."""

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

    def test_sm89_ada_accepted(self) -> None:
        """SM 8.9 (Ada L4) is accepted."""
        reasons = check_sm_version((8, 9))
        assert len(reasons) == 0

    def test_sm90_hopper_accepted(self) -> None:
        """SM 9.0 (Hopper) is accepted."""
        reasons = check_sm_version((9, 0))
        assert len(reasons) == 0

    def test_sm100_blackwell_accepted(self) -> None:
        """SM 10.0 (Blackwell) is accepted."""
        reasons = check_sm_version((10, 0))
        assert len(reasons) == 0

    def test_sm70_volta_rejected(self) -> None:
        """SM 7.0 (Volta) is rejected."""
        reasons = check_sm_version((7, 0))
        assert len(reasons) == 1
        assert SM_TOO_OLD in reasons[0].code

    def test_sm61_pascal_rejected(self) -> None:
        """SM 6.1 (Pascal) is rejected."""
        reasons = check_sm_version((6, 1))
        assert len(reasons) == 1
        assert SM_TOO_OLD in reasons[0].code

    def test_min_sm_constant(self) -> None:
        """FLASHINFER_MIN_SM is (7, 5)."""
        assert FLASHINFER_MIN_SM == (7, 5)


class TestHeadDimChecks:
    """Test head dimension constraint checks."""

    def test_head_dim_64_accepted(self) -> None:
        """head_dim=64 is accepted."""
        reasons = check_head_dim(64)
        assert len(reasons) == 0

    def test_head_dim_96_accepted(self) -> None:
        """head_dim=96 is accepted."""
        reasons = check_head_dim(96)
        assert len(reasons) == 0

    def test_head_dim_128_accepted(self) -> None:
        """head_dim=128 is accepted."""
        reasons = check_head_dim(128)
        assert len(reasons) == 0

    def test_head_dim_256_accepted(self) -> None:
        """head_dim=256 is accepted."""
        reasons = check_head_dim(256)
        assert len(reasons) == 0

    def test_head_dim_32_accepted(self) -> None:
        """head_dim=32 is accepted (minimum)."""
        reasons = check_head_dim(32)
        assert len(reasons) == 0

    def test_head_dim_320_rejected(self) -> None:
        """head_dim=320 is rejected (too large)."""
        reasons = check_head_dim(320)
        assert len(reasons) == 1
        assert HEAD_DIM_TOO_LARGE in reasons[0].code

    def test_head_dim_16_rejected(self) -> None:
        """head_dim=16 is rejected (too small for optimal)."""
        reasons = check_head_dim(16)
        # 16 is below minimum of 32
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

    def test_supported_head_dims_all_multiples_of_8(self) -> None:
        """All supported head dims are multiples of 8."""
        for dim in FLASHINFER_SUPPORTED_HEAD_DIMS:
            assert dim % 8 == 0, f"Head dim {dim} not multiple of 8"


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

    def test_int8_accepted_with_quantization(self) -> None:
        """torch.int8 is accepted (for quantized attention)."""
        reasons = check_dtype(torch.int8, allow_quantized=True)
        assert len(reasons) == 0

    def test_int8_rejected_without_quantization(self) -> None:
        """torch.int8 is rejected without quantization flag."""
        reasons = check_dtype(torch.int8, allow_quantized=False)
        assert len(reasons) == 1
        assert DTYPE_UNSUPPORTED in reasons[0].code


class TestGQAChecks:
    """Test GQA compatibility checks."""

    def test_gqa_group_size_1_accepted(self) -> None:
        """GQA with group size 1 (MHA) is accepted."""
        reasons = check_gqa_compatibility(q_heads=8, kv_heads=8)
        assert len(reasons) == 0

    def test_gqa_group_size_2_accepted(self) -> None:
        """GQA with group size 2 is accepted."""
        reasons = check_gqa_compatibility(q_heads=8, kv_heads=4)
        assert len(reasons) == 0

    def test_gqa_group_size_4_accepted(self) -> None:
        """GQA with group size 4 is accepted."""
        reasons = check_gqa_compatibility(q_heads=8, kv_heads=2)
        assert len(reasons) == 0

    def test_gqa_group_size_8_accepted(self) -> None:
        """GQA with group size 8 is accepted."""
        reasons = check_gqa_compatibility(q_heads=8, kv_heads=1)
        assert len(reasons) == 0

    def test_mqa_accepted(self) -> None:
        """MQA (single KV head) is accepted."""
        reasons = check_gqa_compatibility(q_heads=32, kv_heads=1)
        assert len(reasons) == 0

    def test_invalid_gqa_ratio_rejected(self) -> None:
        """Invalid GQA ratio (not divisible) is rejected."""
        reasons = check_gqa_compatibility(q_heads=8, kv_heads=3)
        assert len(reasons) == 1
        assert GQA_UNSUPPORTED in reasons[0].code

    def test_more_kv_than_q_rejected(self) -> None:
        """More KV heads than Q heads is rejected."""
        reasons = check_gqa_compatibility(q_heads=4, kv_heads=8)
        assert len(reasons) == 1
        assert GQA_UNSUPPORTED in reasons[0].code


class TestCombinedConstraintCheck:
    """Test combined constraint checking."""

    def test_all_valid_no_reasons(self) -> None:
        """Valid config returns no reasons."""
        reasons = check_flashinfer_constraints(
            sm_version=(8, 6),
            head_dim=128,
            dtype=torch.float16,
            q_heads=32,
            kv_heads=8,
        )
        assert len(reasons) == 0

    def test_invalid_sm_returns_reason(self) -> None:
        """Invalid SM returns reason."""
        reasons = check_flashinfer_constraints(
            sm_version=(7, 0),
            head_dim=128,
            dtype=torch.float16,
            q_heads=32,
            kv_heads=8,
        )
        assert len(reasons) >= 1
        assert any(SM_TOO_OLD in r.code for r in reasons)

    def test_invalid_head_dim_returns_reason(self) -> None:
        """Invalid head_dim returns reason."""
        reasons = check_flashinfer_constraints(
            sm_version=(8, 6),
            head_dim=320,
            dtype=torch.float16,
            q_heads=32,
            kv_heads=8,
        )
        assert len(reasons) >= 1
        assert any(HEAD_DIM_TOO_LARGE in r.code for r in reasons)

    def test_invalid_dtype_returns_reason(self) -> None:
        """Invalid dtype returns reason."""
        reasons = check_flashinfer_constraints(
            sm_version=(8, 6),
            head_dim=128,
            dtype=torch.float32,
            q_heads=32,
            kv_heads=8,
        )
        assert len(reasons) >= 1
        assert any(DTYPE_UNSUPPORTED in r.code for r in reasons)

    def test_multiple_invalids_returns_all_reasons(self) -> None:
        """Multiple invalid constraints return all reasons."""
        reasons = check_flashinfer_constraints(
            sm_version=(7, 0),  # Invalid
            head_dim=320,       # Invalid
            dtype=torch.float32,  # Invalid
            q_heads=8,
            kv_heads=3,         # Invalid
        )
        # Should have at least 4 reasons
        assert len(reasons) >= 4
