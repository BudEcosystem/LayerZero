"""Tests for FlashAttention constraints."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.flash_attn.constraints import (
    check_fa_constraints,
    check_fa2_constraints,
    check_fa3_constraints,
    check_fa4_constraints,
)
from layerzero.backends.flash_attn.version import FAVariant


class TestFA2Constraints:
    """Test FA2 (Ampere/Ada) constraints."""

    def test_fa2_accepts_sm80(self) -> None:
        """Test FA2 accepts SM 8.0."""
        violations = check_fa2_constraints(
            sm_version=(8, 0),
            dtype=torch.float16,
            head_dim=64,
        )
        assert len(violations) == 0

    def test_fa2_accepts_sm89(self) -> None:
        """Test FA2 accepts SM 8.9 (Ada)."""
        violations = check_fa2_constraints(
            sm_version=(8, 9),
            dtype=torch.float16,
            head_dim=64,
        )
        assert len(violations) == 0

    def test_fa2_rejects_sm70(self) -> None:
        """Test FA2 rejects SM 7.0."""
        violations = check_fa2_constraints(
            sm_version=(7, 0),
            dtype=torch.float16,
            head_dim=64,
        )
        assert len(violations) > 0

    def test_fa2_rejects_sm90(self) -> None:
        """Test FA2 rejects SM 9.0 (use FA3)."""
        violations = check_fa2_constraints(
            sm_version=(9, 0),
            dtype=torch.float16,
            head_dim=64,
        )
        assert len(violations) > 0

    def test_fa2_accepts_fp16(self) -> None:
        """Test FA2 accepts fp16."""
        violations = check_fa2_constraints(
            sm_version=(8, 0),
            dtype=torch.float16,
            head_dim=64,
        )
        assert len(violations) == 0

    def test_fa2_accepts_bf16(self) -> None:
        """Test FA2 accepts bf16."""
        violations = check_fa2_constraints(
            sm_version=(8, 0),
            dtype=torch.bfloat16,
            head_dim=64,
        )
        assert len(violations) == 0

    def test_fa2_rejects_fp32(self) -> None:
        """Test FA2 rejects fp32."""
        violations = check_fa2_constraints(
            sm_version=(8, 0),
            dtype=torch.float32,
            head_dim=64,
        )
        assert len(violations) > 0

    def test_fa2_accepts_head_dim_256(self) -> None:
        """Test FA2 accepts head_dim=256."""
        violations = check_fa2_constraints(
            sm_version=(8, 0),
            dtype=torch.float16,
            head_dim=256,
        )
        assert len(violations) == 0

    def test_fa2_rejects_head_dim_320(self) -> None:
        """Test FA2 rejects head_dim=320."""
        violations = check_fa2_constraints(
            sm_version=(8, 0),
            dtype=torch.float16,
            head_dim=320,
        )
        assert len(violations) > 0

    def test_fa2_head_dim_multiple_8(self) -> None:
        """Test FA2 requires head_dim multiple of 8."""
        violations = check_fa2_constraints(
            sm_version=(8, 0),
            dtype=torch.float16,
            head_dim=65,  # Not multiple of 8
        )
        assert len(violations) > 0


class TestFA3Constraints:
    """Test FA3 (Hopper) constraints."""

    def test_fa3_accepts_sm90(self) -> None:
        """Test FA3 accepts SM 9.0."""
        violations = check_fa3_constraints(
            sm_version=(9, 0),
            dtype=torch.float16,
            head_dim=64,
        )
        assert len(violations) == 0

    def test_fa3_rejects_sm80(self) -> None:
        """Test FA3 rejects SM 8.0 (use FA2)."""
        violations = check_fa3_constraints(
            sm_version=(8, 0),
            dtype=torch.float16,
            head_dim=64,
        )
        assert len(violations) > 0

    def test_fa3_rejects_sm100(self) -> None:
        """Test FA3 rejects SM 10.0 (use FA4)."""
        violations = check_fa3_constraints(
            sm_version=(10, 0),
            dtype=torch.float16,
            head_dim=64,
        )
        assert len(violations) > 0


class TestFA4Constraints:
    """Test FA4 (Blackwell+) constraints."""

    def test_fa4_accepts_sm100(self) -> None:
        """Test FA4 accepts SM 10.0."""
        violations = check_fa4_constraints(
            sm_version=(10, 0),
            dtype=torch.float16,
            head_dim=64,
        )
        assert len(violations) == 0

    def test_fa4_accepts_sm120(self) -> None:
        """Test FA4 accepts future SM 12.0."""
        violations = check_fa4_constraints(
            sm_version=(12, 0),
            dtype=torch.float16,
            head_dim=64,
        )
        assert len(violations) == 0

    def test_fa4_rejects_sm90(self) -> None:
        """Test FA4 rejects SM 9.0 (use FA3)."""
        violations = check_fa4_constraints(
            sm_version=(9, 0),
            dtype=torch.float16,
            head_dim=64,
        )
        assert len(violations) > 0


class TestGenericFAConstraints:
    """Test generic FA constraint checker."""

    def test_check_fa_constraints_auto_selects_fa2(self) -> None:
        """Test auto-selects FA2 for SM 8.0."""
        violations = check_fa_constraints(
            sm_version=(8, 0),
            dtype=torch.float16,
            head_dim=64,
        )
        assert len(violations) == 0

    def test_check_fa_constraints_auto_selects_fa3(self) -> None:
        """Test auto-selects FA3 for SM 9.0."""
        violations = check_fa_constraints(
            sm_version=(9, 0),
            dtype=torch.float16,
            head_dim=64,
        )
        assert len(violations) == 0

    def test_check_fa_constraints_rejects_unsupported_sm(self) -> None:
        """Test rejects unsupported SM version."""
        violations = check_fa_constraints(
            sm_version=(7, 0),
            dtype=torch.float16,
            head_dim=64,
        )
        assert len(violations) > 0
