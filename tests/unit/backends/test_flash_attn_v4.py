"""Tests for FlashAttention 4 backend integration.

Tests FA4 availability detection, hardware requirements,
routing, constraints, and correctness.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from typing import Any

import torch

from layerzero.device import GPUGeneration


class TestFA4Availability:
    """Tests for FA4 availability detection."""

    def test_fa4_installation_detection(self) -> None:
        """Detect FA4 installation."""
        from layerzero.backends.flash_attn_v4.availability import is_fa4_available

        # This tests the detection logic, not actual installation
        # which depends on the environment
        result = is_fa4_available()
        assert isinstance(result, bool)

    def test_fa4_version_detection(self) -> None:
        """Detect FA4 version >= 3.0."""
        from layerzero.backends.flash_attn_v4.availability import (
            detect_fa4_version,
            FA4_MIN_VERSION,
        )

        # Test version parsing logic with mock
        with patch("layerzero.backends.flash_attn_v4.availability._get_flash_attn_version") as mock_ver:
            mock_ver.return_value = (3, 0, 0)
            version = detect_fa4_version()
            assert version is not None
            assert version >= FA4_MIN_VERSION

    def test_fa4_version_too_old(self) -> None:
        """FA4 not available if version < 3.0."""
        from layerzero.backends.flash_attn_v4.availability import detect_fa4_version

        with patch("layerzero.backends.flash_attn_v4.availability._get_flash_attn_version") as mock_ver:
            mock_ver.return_value = (2, 5, 6)
            version = detect_fa4_version()
            assert version is None

    def test_fa4_cuda_129_required(self) -> None:
        """FA4 requires CUDA 12.9+."""
        from layerzero.backends.flash_attn_v4.availability import (
            check_cuda_version_for_fa4,
            FA4_MIN_CUDA_VERSION,
        )

        # Test CUDA version check
        assert check_cuda_version_for_fa4("12.9.0") is True
        assert check_cuda_version_for_fa4("13.0.0") is True
        assert check_cuda_version_for_fa4("12.4.0") is False
        assert check_cuda_version_for_fa4("11.8.0") is False

    def test_fa4_not_installed(self) -> None:
        """Handle FA4 not installed."""
        from layerzero.backends.flash_attn_v4.availability import detect_fa4_version

        with patch("layerzero.backends.flash_attn_v4.availability._get_flash_attn_version") as mock_ver:
            mock_ver.return_value = None
            version = detect_fa4_version()
            assert version is None


class TestFA4HardwareRequirements:
    """Tests for FA4 hardware requirements."""

    def test_fa4_sm100_supported(self) -> None:
        """FA4 supported on SM100."""
        from layerzero.backends.flash_attn_v4.hardware import is_fa4_compatible_sm

        assert is_fa4_compatible_sm((10, 0)) is True

    def test_fa4_sm120_supported(self) -> None:
        """FA4 supported on SM120 (RTX 50xx)."""
        from layerzero.backends.flash_attn_v4.hardware import is_fa4_compatible_sm

        assert is_fa4_compatible_sm((12, 0)) is True

    def test_fa4_sm90_rejected(self) -> None:
        """FA4 rejected on Hopper (use FA3)."""
        from layerzero.backends.flash_attn_v4.hardware import is_fa4_compatible_sm

        assert is_fa4_compatible_sm((9, 0)) is False

    def test_fa4_sm86_rejected(self) -> None:
        """FA4 rejected on Ampere."""
        from layerzero.backends.flash_attn_v4.hardware import is_fa4_compatible_sm

        assert is_fa4_compatible_sm((8, 6)) is False

    def test_fa4_tcgen05_required(self) -> None:
        """FA4 requires tcgen05.mma support."""
        from layerzero.backends.flash_attn_v4.hardware import (
            has_tcgen05_support,
        )

        # tcgen05 available on Blackwell (SM 10.0+)
        assert has_tcgen05_support((10, 0)) is True
        assert has_tcgen05_support((9, 0)) is False

    def test_fa4_minimum_memory(self) -> None:
        """FA4 has minimum memory requirements."""
        from layerzero.backends.flash_attn_v4.hardware import (
            check_fa4_memory_requirements,
        )

        # Test minimum memory (FA4 needs more shared memory)
        result = check_fa4_memory_requirements(
            available_memory_bytes=16 * 1024**3,  # 16GB
            sequence_length=4096,
            head_dim=128,
            batch_size=32,
        )
        assert result.is_sufficient is True


class TestFA4Routing:
    """Tests for FA4 routing."""

    def test_blackwell_routes_to_fa4(self) -> None:
        """Blackwell GPU routes to FA4."""
        from layerzero.backends.flash_attn.version import select_fa_variant, FAVariant

        assert select_fa_variant((10, 0)) == FAVariant.FA4
        assert select_fa_variant((12, 0)) == FAVariant.FA4

    def test_hopper_routes_to_fa3(self) -> None:
        """Hopper GPU routes to FA3."""
        from layerzero.backends.flash_attn.version import select_fa_variant, FAVariant

        assert select_fa_variant((9, 0)) == FAVariant.FA3

    def test_ampere_routes_to_fa2(self) -> None:
        """Ampere GPU routes to FA2."""
        from layerzero.backends.flash_attn.version import select_fa_variant, FAVariant

        assert select_fa_variant((8, 0)) == FAVariant.FA2
        assert select_fa_variant((8, 6)) == FAVariant.FA2

    def test_ada_routes_to_fa2(self) -> None:
        """Ada Lovelace GPU routes to FA2."""
        from layerzero.backends.flash_attn.version import select_fa_variant, FAVariant

        assert select_fa_variant((8, 9)) == FAVariant.FA2


class TestFA4Constraints:
    """Tests for FA4 constraints."""

    def test_fa4_head_dim_constraints(self) -> None:
        """FA4 head_dim constraints validated."""
        from layerzero.backends.flash_attn.constraints import check_fa4_constraints

        # Valid head_dim
        violations = check_fa4_constraints(
            sm_version=(10, 0),
            dtype=torch.float16,
            head_dim=128,
        )
        assert len(violations) == 0

        # Invalid head_dim (too large)
        violations = check_fa4_constraints(
            sm_version=(10, 0),
            dtype=torch.float16,
            head_dim=512,
        )
        assert any(v.code == "HEAD_DIM_TOO_LARGE" for v in violations)

        # Invalid head_dim (not multiple of 8)
        violations = check_fa4_constraints(
            sm_version=(10, 0),
            dtype=torch.float16,
            head_dim=65,
        )
        assert any(v.code == "HEAD_DIM_ALIGNMENT" for v in violations)

    def test_fa4_dtype_fp16_bf16(self) -> None:
        """FA4 supports fp16/bf16."""
        from layerzero.backends.flash_attn.constraints import check_fa4_constraints

        # fp16 supported
        violations = check_fa4_constraints(
            sm_version=(10, 0),
            dtype=torch.float16,
            head_dim=128,
        )
        assert len(violations) == 0

        # bf16 supported
        violations = check_fa4_constraints(
            sm_version=(10, 0),
            dtype=torch.bfloat16,
            head_dim=128,
        )
        assert len(violations) == 0

    def test_fa4_dtype_fp32_rejected(self) -> None:
        """FA4 rejects fp32."""
        from layerzero.backends.flash_attn.constraints import check_fa4_constraints

        violations = check_fa4_constraints(
            sm_version=(10, 0),
            dtype=torch.float32,
            head_dim=128,
        )
        assert any(v.code == "DTYPE_UNSUPPORTED" for v in violations)

    def test_fa4_fp8_support(self) -> None:
        """FA4 supports FP8 on Blackwell."""
        from layerzero.backends.flash_attn_v4.constraints import (
            check_fa4_fp8_constraints,
        )

        # FP8 supported on Blackwell
        violations = check_fa4_fp8_constraints(
            sm_version=(10, 0),
            dtype_str="fp8_e4m3",
            head_dim=128,
        )
        assert len(violations) == 0

    def test_fa4_sm_too_old(self) -> None:
        """FA4 rejects SM < 10.0."""
        from layerzero.backends.flash_attn.constraints import check_fa4_constraints

        violations = check_fa4_constraints(
            sm_version=(9, 0),
            dtype=torch.float16,
            head_dim=128,
        )
        assert any(v.code == "SM_TOO_OLD" for v in violations)


class TestFA4Adapter:
    """Tests for FA4 adapter class."""

    def test_fa4_adapter_creation(self) -> None:
        """FA4 adapter can be created."""
        from layerzero.backends.flash_attn_v4.adapter import FlashAttnV4Adapter

        adapter = FlashAttnV4Adapter()
        assert adapter is not None

    def test_fa4_adapter_kernel_spec(self) -> None:
        """FA4 adapter provides kernel spec."""
        from layerzero.backends.flash_attn_v4.adapter import FlashAttnV4Adapter

        adapter = FlashAttnV4Adapter()
        spec = adapter.get_kernel_spec()

        assert spec.kernel_id == "flash_attn.v4"
        assert spec.min_sm == (10, 0)
        assert GPUGeneration.BLACKWELL in spec.supported_generations

    def test_fa4_adapter_priority(self) -> None:
        """FA4 adapter has high priority."""
        from layerzero.backends.flash_attn_v4.adapter import FlashAttnV4Adapter

        adapter = FlashAttnV4Adapter()
        spec = adapter.get_kernel_spec()

        # FA4 should have higher priority than FA3/FA2
        assert spec.priority >= 95


class TestFA4Specs:
    """Tests for FA4 kernel specifications."""

    def test_fa4_spec_supported_generations(self) -> None:
        """FA4 spec lists Blackwell as supported."""
        from layerzero.backends.flash_attn_v4.specs import FA4_KERNEL_SPEC

        assert GPUGeneration.BLACKWELL in FA4_KERNEL_SPEC.supported_generations
        assert GPUGeneration.HOPPER not in FA4_KERNEL_SPEC.supported_generations

    def test_fa4_spec_min_sm(self) -> None:
        """FA4 spec has correct min SM."""
        from layerzero.backends.flash_attn_v4.specs import FA4_KERNEL_SPEC

        assert FA4_KERNEL_SPEC.min_sm == (10, 0)

    def test_fa4_spec_dtypes(self) -> None:
        """FA4 spec lists supported dtypes."""
        from layerzero.backends.flash_attn_v4.specs import FA4_KERNEL_SPEC

        assert torch.float16 in FA4_KERNEL_SPEC.supported_dtypes
        assert torch.bfloat16 in FA4_KERNEL_SPEC.supported_dtypes

    def test_fa4_spec_features(self) -> None:
        """FA4 spec has correct features."""
        from layerzero.backends.flash_attn_v4.specs import FA4_KERNEL_SPEC

        assert FA4_KERNEL_SPEC.supports_gqa is True
        assert FA4_KERNEL_SPEC.is_cuda_graph_safe is True


class TestFA4ThreadSafety:
    """Tests for FA4 thread safety."""

    def test_fa4_adapter_concurrent_access(self) -> None:
        """FA4 adapter handles concurrent access."""
        import threading

        from layerzero.backends.flash_attn_v4.adapter import FlashAttnV4Adapter

        adapter = FlashAttnV4Adapter()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(100):
                    spec = adapter.get_kernel_spec()
                    _ = adapter.is_available
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
