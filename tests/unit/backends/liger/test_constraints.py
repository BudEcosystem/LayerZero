"""Tests for Liger constraint checking."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.liger.constraints import (
    check_liger_constraints,
    check_triton_version,
    check_dtype,
    check_platform,
    LIGER_MIN_TRITON_VERSION,
    LIGER_SUPPORTED_DTYPES,
)
from layerzero.reasons import (
    DTYPE_UNSUPPORTED,
    BACKEND_VERSION_MISMATCH,
    PLATFORM_MISMATCH,
)


class TestTritonVersionChecks:
    """Test Triton version constraint checks."""

    def test_triton_2_0_0_accepted(self) -> None:
        """Triton 2.0.0 is accepted."""
        reasons = check_triton_version((2, 0, 0))
        assert len(reasons) == 0

    def test_triton_2_1_0_accepted(self) -> None:
        """Triton 2.1.0 is accepted."""
        reasons = check_triton_version((2, 1, 0))
        assert len(reasons) == 0

    def test_triton_3_0_0_accepted(self) -> None:
        """Triton 3.0.0 is accepted."""
        reasons = check_triton_version((3, 0, 0))
        assert len(reasons) == 0

    def test_triton_1_4_0_rejected(self) -> None:
        """Triton 1.4.0 is rejected (too old)."""
        reasons = check_triton_version((1, 4, 0))
        assert len(reasons) == 1
        assert BACKEND_VERSION_MISMATCH in reasons[0].code

    def test_triton_1_0_0_rejected(self) -> None:
        """Triton 1.0.0 is rejected (too old)."""
        reasons = check_triton_version((1, 0, 0))
        assert len(reasons) == 1
        assert BACKEND_VERSION_MISMATCH in reasons[0].code

    def test_min_triton_version_constant(self) -> None:
        """LIGER_MIN_TRITON_VERSION is (2, 0, 0)."""
        assert LIGER_MIN_TRITON_VERSION == (2, 0, 0)


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

    def test_fp32_accepted(self) -> None:
        """torch.float32 is accepted."""
        reasons = check_dtype(torch.float32)
        assert len(reasons) == 0

    def test_fp64_rejected(self) -> None:
        """torch.float64 is rejected."""
        reasons = check_dtype(torch.float64)
        assert len(reasons) == 1
        assert DTYPE_UNSUPPORTED in reasons[0].code

    def test_int8_rejected(self) -> None:
        """torch.int8 is rejected."""
        reasons = check_dtype(torch.int8)
        assert len(reasons) == 1
        assert DTYPE_UNSUPPORTED in reasons[0].code

    def test_supported_dtypes_constant(self) -> None:
        """LIGER_SUPPORTED_DTYPES includes fp16, bf16, fp32."""
        assert torch.float16 in LIGER_SUPPORTED_DTYPES
        assert torch.bfloat16 in LIGER_SUPPORTED_DTYPES
        assert torch.float32 in LIGER_SUPPORTED_DTYPES


class TestPlatformChecks:
    """Test platform constraint checks."""

    def test_cuda_accepted(self) -> None:
        """CUDA platform is accepted."""
        from layerzero.enums import Platform
        reasons = check_platform(Platform.CUDA)
        assert len(reasons) == 0

    def test_rocm_accepted(self) -> None:
        """ROCm platform is accepted (Triton supports it)."""
        from layerzero.enums import Platform
        reasons = check_platform(Platform.ROCM)
        assert len(reasons) == 0

    def test_cpu_rejected(self) -> None:
        """CPU platform is rejected (Triton requires GPU)."""
        from layerzero.enums import Platform
        reasons = check_platform(Platform.CPU)
        assert len(reasons) == 1
        assert PLATFORM_MISMATCH in reasons[0].code


class TestCombinedConstraintCheck:
    """Test combined constraint checking."""

    def test_all_valid_no_reasons(self) -> None:
        """Valid config returns no reasons."""
        from layerzero.enums import Platform
        reasons = check_liger_constraints(
            triton_version=(2, 1, 0),
            dtype=torch.float16,
            platform=Platform.CUDA,
        )
        assert len(reasons) == 0

    def test_invalid_triton_returns_reason(self) -> None:
        """Invalid Triton version returns reason."""
        from layerzero.enums import Platform
        reasons = check_liger_constraints(
            triton_version=(1, 0, 0),
            dtype=torch.float16,
            platform=Platform.CUDA,
        )
        assert len(reasons) >= 1
        assert any(BACKEND_VERSION_MISMATCH in r.code for r in reasons)

    def test_invalid_dtype_returns_reason(self) -> None:
        """Invalid dtype returns reason."""
        from layerzero.enums import Platform
        reasons = check_liger_constraints(
            triton_version=(2, 1, 0),
            dtype=torch.float64,
            platform=Platform.CUDA,
        )
        assert len(reasons) >= 1
        assert any(DTYPE_UNSUPPORTED in r.code for r in reasons)

    def test_invalid_platform_returns_reason(self) -> None:
        """Invalid platform returns reason."""
        from layerzero.enums import Platform
        reasons = check_liger_constraints(
            triton_version=(2, 1, 0),
            dtype=torch.float16,
            platform=Platform.CPU,
        )
        assert len(reasons) >= 1
        assert any(PLATFORM_MISMATCH in r.code for r in reasons)

    def test_multiple_invalids_returns_all_reasons(self) -> None:
        """Multiple invalid constraints return all reasons."""
        from layerzero.enums import Platform
        reasons = check_liger_constraints(
            triton_version=(1, 0, 0),  # Invalid
            dtype=torch.float64,       # Invalid
            platform=Platform.CPU,     # Invalid
        )
        assert len(reasons) >= 3
