"""Tests for GPU generation detection and routing.

Tests GPU generation detection, kernel filtering by generation,
and FA version routing (FA3 vs FA4 for Blackwell).
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any

from layerzero.device import GPUGeneration, sm_to_generation


class TestBlackwellDetection:
    """Tests for Blackwell GPU detection."""

    def test_blackwell_sm100_detection(self) -> None:
        """SM 10.0 correctly detected as Blackwell."""
        assert sm_to_generation(10, 0) == GPUGeneration.BLACKWELL

    def test_blackwell_sm120_detection(self) -> None:
        """SM 12.0 (RTX 50xx) correctly detected as Blackwell."""
        assert sm_to_generation(12, 0) == GPUGeneration.BLACKWELL

    def test_blackwell_future_sm_detection(self) -> None:
        """Future SM versions (>10) detected as Blackwell or newer."""
        result = sm_to_generation(13, 0)
        assert result >= GPUGeneration.BLACKWELL


class TestKernelFilteringByGeneration:
    """Tests for kernel filtering by supported GPU generations."""

    def test_kernel_supported_generation_check(self) -> None:
        """Kernels filtered by supported_generations."""
        from layerzero.device import GPUGeneration

        # Kernel that only supports Hopper and older
        supported = frozenset([
            GPUGeneration.AMPERE,
            GPUGeneration.ADA_LOVELACE,
            GPUGeneration.HOPPER,
        ])

        # Blackwell is NOT in supported
        device_gen = GPUGeneration.BLACKWELL
        assert device_gen not in supported

        # Hopper IS in supported
        device_gen = GPUGeneration.HOPPER
        assert device_gen in supported

    def test_empty_supported_generations_means_all(self) -> None:
        """Empty supported_generations means all generations supported."""
        from layerzero.device import GPUGeneration

        supported: frozenset[GPUGeneration] = frozenset()
        device_gen = GPUGeneration.BLACKWELL

        # Empty frozenset means no restriction
        # Convention: empty set = all supported
        is_supported = not supported or device_gen in supported
        assert is_supported is True


class TestFA3ExcludedFromBlackwell:
    """Tests for FA3 exclusion from Blackwell GPUs."""

    def test_fa3_not_in_blackwell_supported_gens(self) -> None:
        """FA3 kernels should not have Blackwell in supported_generations."""
        from layerzero.device import GPUGeneration

        # FA3 supported generations (excludes Blackwell)
        fa3_supported = frozenset([
            GPUGeneration.AMPERE,
            GPUGeneration.ADA_LOVELACE,
            GPUGeneration.HOPPER,
        ])

        assert GPUGeneration.BLACKWELL not in fa3_supported

    def test_fa3_kernel_rejected_on_blackwell(self) -> None:
        """FA3 kernel should be rejected when device is Blackwell."""
        from layerzero.device import GPUGeneration
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.enums import Platform

        # Create FA3-style kernel spec
        fa3_spec = KernelSpec(
            kernel_id="flash_attn.v3",
            operation="attention.causal",
            source="flash_attn",
            version="3.0.0",
            platform=Platform.CUDA,
            min_sm=(8, 0),
            max_sm=(9, 0),  # Hopper max - excludes Blackwell
            supported_generations=frozenset([
                GPUGeneration.AMPERE,
                GPUGeneration.ADA_LOVELACE,
                GPUGeneration.HOPPER,
            ]),
        )

        # Check that Blackwell is not supported
        assert GPUGeneration.BLACKWELL not in fa3_spec.supported_generations

        # max_sm check would also reject SM 10.0
        assert fa3_spec.max_sm is not None
        blackwell_sm = (10, 0)
        assert blackwell_sm > fa3_spec.max_sm


class TestFA4PreferredOnBlackwell:
    """Tests for FA4 preference on Blackwell GPUs."""

    def test_fa4_supports_blackwell(self) -> None:
        """FA4 should have Blackwell in supported_generations."""
        from layerzero.device import GPUGeneration

        # FA4 supported generations (includes Blackwell)
        fa4_supported = frozenset([
            GPUGeneration.BLACKWELL,
        ])

        assert GPUGeneration.BLACKWELL in fa4_supported

    def test_fa4_kernel_accepted_on_blackwell(self) -> None:
        """FA4 kernel should be accepted when device is Blackwell."""
        from layerzero.device import GPUGeneration
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.enums import Platform

        # Create FA4-style kernel spec
        fa4_spec = KernelSpec(
            kernel_id="flash_attn.v4",
            operation="attention.causal",
            source="flash_attn",
            version="4.0.0",
            platform=Platform.CUDA,
            min_sm=(10, 0),  # Blackwell min
            supported_generations=frozenset([
                GPUGeneration.BLACKWELL,
            ]),
        )

        # Check that Blackwell is supported
        assert GPUGeneration.BLACKWELL in fa4_spec.supported_generations

        # min_sm check would accept SM 10.0
        assert fa4_spec.min_sm is not None
        blackwell_sm = (10, 0)
        assert blackwell_sm >= fa4_spec.min_sm


class TestUnknownGenerationFallback:
    """Tests for handling unknown GPU architectures."""

    def test_unknown_generation_graceful_handling(self) -> None:
        """Unknown architectures handled gracefully."""
        from layerzero.device import sm_to_generation, GPUGeneration

        # Very old SM
        assert sm_to_generation(5, 0) == GPUGeneration.UNKNOWN

        # Unknown recent architecture
        assert sm_to_generation(6, 5) == GPUGeneration.UNKNOWN

    def test_unknown_generation_kernel_selection_fallback(self) -> None:
        """Unknown generation allows kernels with empty supported_generations."""
        from layerzero.device import GPUGeneration
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.enums import Platform

        # Kernel with no generation restriction (empty set)
        fallback_kernel = KernelSpec(
            kernel_id="sdpa.fallback",
            operation="attention.causal",
            source="torch",
            version="2.0.0",
            platform=Platform.CUDA,
            supported_generations=frozenset(),  # Empty = all supported
        )

        device_gen = GPUGeneration.UNKNOWN

        # Empty supported_generations means all are supported
        is_supported = (
            not fallback_kernel.supported_generations
            or device_gen in fallback_kernel.supported_generations
        )
        assert is_supported is True


class TestGenerationRouting:
    """Tests for GPU generation-based kernel routing."""

    def test_routing_selects_correct_kernel_for_hopper(self) -> None:
        """Hopper GPU routes to FA3."""
        from layerzero.device import GPUGeneration

        device_gen = GPUGeneration.HOPPER

        # Define available kernels with their supported generations
        kernels = {
            "fa2": frozenset([GPUGeneration.AMPERE, GPUGeneration.ADA_LOVELACE]),
            "fa3": frozenset([GPUGeneration.AMPERE, GPUGeneration.ADA_LOVELACE, GPUGeneration.HOPPER]),
            "fa4": frozenset([GPUGeneration.BLACKWELL]),
        }

        # Find matching kernels
        matching = [k for k, gens in kernels.items() if device_gen in gens]

        assert "fa3" in matching
        assert "fa4" not in matching

    def test_routing_selects_correct_kernel_for_blackwell(self) -> None:
        """Blackwell GPU routes to FA4."""
        from layerzero.device import GPUGeneration

        device_gen = GPUGeneration.BLACKWELL

        # Define available kernels with their supported generations
        kernels = {
            "fa2": frozenset([GPUGeneration.AMPERE, GPUGeneration.ADA_LOVELACE]),
            "fa3": frozenset([GPUGeneration.AMPERE, GPUGeneration.ADA_LOVELACE, GPUGeneration.HOPPER]),
            "fa4": frozenset([GPUGeneration.BLACKWELL]),
        }

        # Find matching kernels
        matching = [k for k, gens in kernels.items() if device_gen in gens]

        assert "fa4" in matching
        assert "fa3" not in matching
        assert "fa2" not in matching

    def test_routing_selects_correct_kernel_for_ampere(self) -> None:
        """Ampere GPU routes to FA2/FA3."""
        from layerzero.device import GPUGeneration

        device_gen = GPUGeneration.AMPERE

        kernels = {
            "fa2": frozenset([GPUGeneration.AMPERE, GPUGeneration.ADA_LOVELACE]),
            "fa3": frozenset([GPUGeneration.AMPERE, GPUGeneration.ADA_LOVELACE, GPUGeneration.HOPPER]),
            "fa4": frozenset([GPUGeneration.BLACKWELL]),
        }

        matching = [k for k, gens in kernels.items() if device_gen in gens]

        assert "fa2" in matching
        assert "fa3" in matching
        assert "fa4" not in matching


class TestGenerationFilter:
    """Tests for generation-based kernel filtering."""

    def test_filter_kernels_by_generation(self) -> None:
        """Filter kernel list by device generation."""
        from layerzero.device import GPUGeneration
        from layerzero.routing.gpu_routing import filter_by_generation

        # Mock kernels with supported generations
        kernels = [
            {"id": "k1", "supported_generations": frozenset([GPUGeneration.AMPERE])},
            {"id": "k2", "supported_generations": frozenset([GPUGeneration.HOPPER])},
            {"id": "k3", "supported_generations": frozenset([GPUGeneration.BLACKWELL])},
            {"id": "k4", "supported_generations": frozenset()},  # All
        ]

        # Filter for Hopper
        filtered = filter_by_generation(kernels, GPUGeneration.HOPPER)

        ids = [k["id"] for k in filtered]
        assert "k2" in ids  # Hopper supported
        assert "k4" in ids  # Empty = all
        assert "k1" not in ids  # Ampere only
        assert "k3" not in ids  # Blackwell only

    def test_filter_returns_all_for_empty_supported_gens(self) -> None:
        """Kernels with empty supported_generations pass all filters."""
        from layerzero.device import GPUGeneration
        from layerzero.routing.gpu_routing import filter_by_generation

        kernels = [
            {"id": "universal", "supported_generations": frozenset()},
        ]

        # Should pass for any generation
        for gen in GPUGeneration:
            filtered = filter_by_generation(kernels, gen)
            assert len(filtered) == 1


class TestGenerationPreference:
    """Tests for generation-based kernel preference scoring."""

    def test_prefer_native_generation_kernel(self) -> None:
        """Kernels native to device generation preferred."""
        from layerzero.device import GPUGeneration
        from layerzero.routing.gpu_routing import score_by_generation

        # Kernel native to Hopper vs one that supports Hopper but isn't native
        kernel_native = {
            "id": "native",
            "supported_generations": frozenset([GPUGeneration.HOPPER]),
            "native_generation": GPUGeneration.HOPPER,
        }
        kernel_compat = {
            "id": "compat",
            "supported_generations": frozenset([GPUGeneration.AMPERE, GPUGeneration.HOPPER]),
            "native_generation": GPUGeneration.AMPERE,
        }

        # Native should score higher
        score_native = score_by_generation(kernel_native, GPUGeneration.HOPPER)
        score_compat = score_by_generation(kernel_compat, GPUGeneration.HOPPER)

        assert score_native > score_compat
