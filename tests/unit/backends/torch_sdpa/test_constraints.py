"""Tests for SDPA backend constraints."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.torch_sdpa.constraints import (
    SDPABackendType,
    check_flash_constraints,
    check_efficient_constraints,
    check_cudnn_constraints,
    get_available_backends,
    can_use_backend,
)


class TestSDPABackendType:
    """Test SDPABackendType enum."""

    def test_flash_backend_exists(self) -> None:
        """Test FLASH backend type exists."""
        assert SDPABackendType.FLASH is not None

    def test_efficient_backend_exists(self) -> None:
        """Test EFFICIENT backend type exists."""
        assert SDPABackendType.EFFICIENT is not None

    def test_cudnn_backend_exists(self) -> None:
        """Test CUDNN backend type exists."""
        assert SDPABackendType.CUDNN is not None

    def test_math_backend_exists(self) -> None:
        """Test MATH backend type exists."""
        assert SDPABackendType.MATH is not None


class TestFlashConstraints:
    """Test FlashAttention backend constraints."""

    def test_flash_requires_sm80_plus(self) -> None:
        """Test Flash requires SM 8.0+."""
        # SM 7.5 should fail
        reasons = check_flash_constraints(
            sm_version=(7, 5),
            dtype=torch.float16,
            head_dim=64,
            is_causal=False,
            has_mask=False,
        )
        assert len(reasons) > 0

    def test_flash_accepts_sm80(self) -> None:
        """Test Flash accepts SM 8.0."""
        reasons = check_flash_constraints(
            sm_version=(8, 0),
            dtype=torch.float16,
            head_dim=64,
            is_causal=False,
            has_mask=False,
        )
        assert len(reasons) == 0

    def test_flash_rejects_fp32(self) -> None:
        """Test Flash rejects fp32."""
        reasons = check_flash_constraints(
            sm_version=(8, 0),
            dtype=torch.float32,
            head_dim=64,
            is_causal=False,
            has_mask=False,
        )
        assert len(reasons) > 0

    def test_flash_accepts_fp16(self) -> None:
        """Test Flash accepts fp16."""
        reasons = check_flash_constraints(
            sm_version=(8, 0),
            dtype=torch.float16,
            head_dim=64,
            is_causal=False,
            has_mask=False,
        )
        assert len(reasons) == 0

    def test_flash_accepts_bf16(self) -> None:
        """Test Flash accepts bf16."""
        reasons = check_flash_constraints(
            sm_version=(8, 0),
            dtype=torch.bfloat16,
            head_dim=64,
            is_causal=False,
            has_mask=False,
        )
        assert len(reasons) == 0

    def test_flash_rejects_mask_plus_causal(self) -> None:
        """Test Flash rejects mask + is_causal together."""
        reasons = check_flash_constraints(
            sm_version=(8, 0),
            dtype=torch.float16,
            head_dim=64,
            is_causal=True,
            has_mask=True,
        )
        assert len(reasons) > 0

    def test_flash_rejects_head_dim_too_large(self) -> None:
        """Test Flash rejects head_dim > 256."""
        reasons = check_flash_constraints(
            sm_version=(8, 0),
            dtype=torch.float16,
            head_dim=512,
            is_causal=False,
            has_mask=False,
        )
        assert len(reasons) > 0


class TestEfficientConstraints:
    """Test Memory Efficient backend constraints."""

    def test_efficient_accepts_sm50(self) -> None:
        """Test Efficient accepts SM 5.0+."""
        reasons = check_efficient_constraints(
            sm_version=(5, 0),
            dtype=torch.float16,
            head_dim=64,
            is_causal=False,
            has_mask=False,
        )
        assert len(reasons) == 0

    def test_efficient_accepts_fp32(self) -> None:
        """Test Efficient accepts fp32."""
        reasons = check_efficient_constraints(
            sm_version=(8, 0),
            dtype=torch.float32,
            head_dim=64,
            is_causal=False,
            has_mask=False,
        )
        assert len(reasons) == 0

    def test_efficient_rejects_mask_plus_causal(self) -> None:
        """Test Efficient rejects mask + is_causal together."""
        reasons = check_efficient_constraints(
            sm_version=(8, 0),
            dtype=torch.float16,
            head_dim=64,
            is_causal=True,
            has_mask=True,
        )
        assert len(reasons) > 0


class TestCudnnConstraints:
    """Test cuDNN backend constraints."""

    def test_cudnn_requires_sm80_plus(self) -> None:
        """Test cuDNN requires SM 8.0+."""
        reasons = check_cudnn_constraints(
            sm_version=(7, 5),
            dtype=torch.float16,
            head_dim=64,
            is_causal=False,
            has_mask=False,
        )
        assert len(reasons) > 0

    def test_cudnn_rejects_fp32(self) -> None:
        """Test cuDNN rejects fp32."""
        reasons = check_cudnn_constraints(
            sm_version=(8, 0),
            dtype=torch.float32,
            head_dim=64,
            is_causal=False,
            has_mask=False,
        )
        assert len(reasons) > 0

    def test_cudnn_head_dim_128_limit(self) -> None:
        """Test cuDNN has head_dim <= 128."""
        reasons = check_cudnn_constraints(
            sm_version=(8, 0),
            dtype=torch.float16,
            head_dim=256,
            is_causal=False,
            has_mask=False,
        )
        assert len(reasons) > 0


class TestGetAvailableBackends:
    """Test getting available backends."""

    def test_math_always_available(self) -> None:
        """Test MATH backend is always available."""
        backends = get_available_backends(
            sm_version=(5, 0),
            dtype=torch.float32,
            head_dim=64,
            is_causal=False,
            has_mask=False,
        )
        assert SDPABackendType.MATH in backends

    def test_multiple_backends_on_sm80_fp16(self) -> None:
        """Test multiple backends available on SM80 with fp16."""
        backends = get_available_backends(
            sm_version=(8, 0),
            dtype=torch.float16,
            head_dim=64,
            is_causal=False,
            has_mask=False,
        )
        assert len(backends) >= 2  # At least MATH and something else


class TestCanUseBackend:
    """Test can_use_backend helper."""

    def test_can_use_math_always(self) -> None:
        """Test MATH backend can always be used."""
        can_use = can_use_backend(
            backend=SDPABackendType.MATH,
            sm_version=(5, 0),
            dtype=torch.float32,
            head_dim=64,
            is_causal=False,
            has_mask=False,
        )
        assert can_use is True

    def test_cannot_use_flash_on_sm70(self) -> None:
        """Test Flash cannot be used on SM 7.0."""
        can_use = can_use_backend(
            backend=SDPABackendType.FLASH,
            sm_version=(7, 0),
            dtype=torch.float16,
            head_dim=64,
            is_causal=False,
            has_mask=False,
        )
        assert can_use is False
