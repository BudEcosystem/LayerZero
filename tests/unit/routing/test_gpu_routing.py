"""Tests for GPU routing module."""
from __future__ import annotations

import pytest
from typing import Any

from layerzero.device import GPUGeneration
from layerzero.routing.gpu_routing import (
    filter_by_generation,
    score_by_generation,
    select_best_for_generation,
    GenerationRouter,
    RouterConfig,
)


class TestRouterConfig:
    """Tests for RouterConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config values."""
        config = RouterConfig()

        assert config.prefer_native is True
        assert config.native_bonus == 100
        assert config.fallback_to_universal is True
        assert config.strict_mode is False

    def test_custom_values(self) -> None:
        """Custom config values."""
        config = RouterConfig(
            prefer_native=False,
            native_bonus=50,
            fallback_to_universal=False,
            strict_mode=True,
        )

        assert config.prefer_native is False
        assert config.native_bonus == 50
        assert config.fallback_to_universal is False
        assert config.strict_mode is True

    def test_config_immutable(self) -> None:
        """Config is immutable."""
        config = RouterConfig()

        with pytest.raises(AttributeError):
            config.native_bonus = 200


class TestFilterByGeneration:
    """Tests for filter_by_generation function."""

    def test_filter_hopper(self, fa_kernels) -> None:
        """Filter for Hopper returns FA2, FA3."""
        filtered = filter_by_generation(fa_kernels, GPUGeneration.HOPPER)

        ids = [k["id"] for k in filtered]
        assert "fa3" in ids
        assert "fa2" not in ids  # FA2 doesn't explicitly support Hopper
        assert "fa4" not in ids

    def test_filter_blackwell(self, fa_kernels) -> None:
        """Filter for Blackwell returns FA4 only."""
        filtered = filter_by_generation(fa_kernels, GPUGeneration.BLACKWELL)

        ids = [k["id"] for k in filtered]
        assert "fa4" in ids
        assert "fa3" not in ids
        assert "fa2" not in ids

    def test_filter_ampere(self, fa_kernels) -> None:
        """Filter for Ampere returns FA2, FA3."""
        filtered = filter_by_generation(fa_kernels, GPUGeneration.AMPERE)

        ids = [k["id"] for k in filtered]
        assert "fa2" in ids
        assert "fa3" in ids
        assert "fa4" not in ids

    def test_filter_includes_universal(self, mixed_kernels) -> None:
        """Universal kernels included by default."""
        filtered = filter_by_generation(mixed_kernels, GPUGeneration.TURING)

        ids = [k["id"] for k in filtered]
        assert "sdpa_universal" in ids

    def test_filter_excludes_universal_when_disabled(self, mixed_kernels) -> None:
        """Universal kernels excluded when fallback disabled."""
        config = RouterConfig(fallback_to_universal=False)
        filtered = filter_by_generation(
            mixed_kernels,
            GPUGeneration.TURING,
            config,
        )

        ids = [k["id"] for k in filtered]
        assert "sdpa_universal" not in ids

    def test_filter_empty_list(self) -> None:
        """Empty kernel list returns empty."""
        filtered = filter_by_generation([], GPUGeneration.HOPPER)

        assert len(filtered) == 0

    def test_filter_unknown_generation(self, mixed_kernels) -> None:
        """Unknown generation only matches universal kernels."""
        filtered = filter_by_generation(mixed_kernels, GPUGeneration.UNKNOWN)

        ids = [k["id"] for k in filtered]
        assert "sdpa_universal" in ids
        # FA kernels don't support UNKNOWN
        assert "fa2" not in ids
        assert "fa3" not in ids
        assert "fa4" not in ids


class TestScoreByGeneration:
    """Tests for score_by_generation function."""

    def test_native_generation_scores_higher(self) -> None:
        """Native generation kernel scores higher."""
        kernel_native = {
            "id": "native",
            "supported_generations": frozenset([GPUGeneration.HOPPER]),
            "native_generation": GPUGeneration.HOPPER,
        }
        kernel_compat = {
            "id": "compat",
            "supported_generations": frozenset([
                GPUGeneration.AMPERE,
                GPUGeneration.HOPPER,
            ]),
            "native_generation": GPUGeneration.AMPERE,
        }

        score_native = score_by_generation(kernel_native, GPUGeneration.HOPPER)
        score_compat = score_by_generation(kernel_compat, GPUGeneration.HOPPER)

        assert score_native > score_compat

    def test_universal_kernel_base_score(self, universal_kernel) -> None:
        """Universal kernel gets base score."""
        score = score_by_generation(universal_kernel, GPUGeneration.BLACKWELL)

        # Base score for supporting (via empty set)
        assert score >= 50

    def test_strict_mode_penalizes_universal(self, universal_kernel) -> None:
        """Strict mode penalizes universal kernels."""
        config_normal = RouterConfig(strict_mode=False)
        config_strict = RouterConfig(strict_mode=True)

        score_normal = score_by_generation(
            universal_kernel,
            GPUGeneration.BLACKWELL,
            config_normal,
        )
        score_strict = score_by_generation(
            universal_kernel,
            GPUGeneration.BLACKWELL,
            config_strict,
        )

        assert score_strict < score_normal

    def test_prefer_native_disabled(self) -> None:
        """Prefer native disabled removes bonus."""
        kernel = {
            "id": "native",
            "supported_generations": frozenset([GPUGeneration.HOPPER]),
            "native_generation": GPUGeneration.HOPPER,
        }

        config_prefer = RouterConfig(prefer_native=True, native_bonus=100)
        config_no_prefer = RouterConfig(prefer_native=False, native_bonus=100)

        score_prefer = score_by_generation(kernel, GPUGeneration.HOPPER, config_prefer)
        score_no_prefer = score_by_generation(kernel, GPUGeneration.HOPPER, config_no_prefer)

        assert score_prefer > score_no_prefer


class TestSelectBestForGeneration:
    """Tests for select_best_for_generation function."""

    def test_select_fa3_for_hopper(self, fa_kernels) -> None:
        """Select FA3 for Hopper (native)."""
        best = select_best_for_generation(fa_kernels, GPUGeneration.HOPPER)

        assert best is not None
        assert best["id"] == "fa3"

    def test_select_fa4_for_blackwell(self, fa_kernels) -> None:
        """Select FA4 for Blackwell."""
        best = select_best_for_generation(fa_kernels, GPUGeneration.BLACKWELL)

        assert best is not None
        assert best["id"] == "fa4"

    def test_select_universal_when_no_native(self, mixed_kernels) -> None:
        """Select universal kernel when no native available."""
        best = select_best_for_generation(mixed_kernels, GPUGeneration.TURING)

        assert best is not None
        assert best["id"] == "sdpa_universal"

    def test_select_returns_none_for_empty(self) -> None:
        """Returns None for empty kernel list."""
        best = select_best_for_generation([], GPUGeneration.HOPPER)

        assert best is None

    def test_select_returns_none_when_no_compatible(self, fa_kernels) -> None:
        """Returns None when no compatible kernels."""
        config = RouterConfig(fallback_to_universal=False)
        best = select_best_for_generation(
            fa_kernels,
            GPUGeneration.TURING,
            config,
        )

        assert best is None


class TestGenerationRouter:
    """Tests for GenerationRouter class."""

    def test_router_creation(self) -> None:
        """Router created with default config."""
        router = GenerationRouter()

        assert router.config.prefer_native is True

    def test_router_custom_config(self) -> None:
        """Router uses custom config."""
        config = RouterConfig(native_bonus=200)
        router = GenerationRouter(config=config)

        assert router.config.native_bonus == 200

    def test_route_selects_best(self, fa_kernels) -> None:
        """Route selects best kernel for generation."""
        router = GenerationRouter()

        result = router.route("attention", fa_kernels, GPUGeneration.HOPPER)

        assert result is not None
        assert result["id"] == "fa3"

    def test_route_caches_result(self, fa_kernels) -> None:
        """Route caches results."""
        router = GenerationRouter()

        result1 = router.route("attention", fa_kernels, GPUGeneration.HOPPER)
        result2 = router.route("attention", fa_kernels, GPUGeneration.HOPPER)

        assert result1 is result2  # Same object from cache

    def test_clear_cache(self, fa_kernels) -> None:
        """Clear cache removes entries."""
        router = GenerationRouter()

        router.route("attention", fa_kernels, GPUGeneration.HOPPER)
        count = router.clear_cache()

        assert count == 1

    def test_get_fa_kernel_for_blackwell(self) -> None:
        """FA4 returned for Blackwell."""
        router = GenerationRouter()

        result = router.get_fa_kernel_for_generation(GPUGeneration.BLACKWELL)

        assert result == "fa4"

    def test_get_fa_kernel_for_hopper(self) -> None:
        """FA3 returned for Hopper."""
        router = GenerationRouter()

        result = router.get_fa_kernel_for_generation(GPUGeneration.HOPPER)

        assert result == "fa3"

    def test_get_fa_kernel_for_ampere(self) -> None:
        """FA2 returned for Ampere."""
        router = GenerationRouter()

        result = router.get_fa_kernel_for_generation(GPUGeneration.AMPERE)

        assert result == "fa2"

    def test_is_fa3_compatible_hopper(self) -> None:
        """FA3 compatible with Hopper."""
        router = GenerationRouter()

        assert router.is_fa3_compatible(GPUGeneration.HOPPER) is True

    def test_is_fa3_incompatible_blackwell(self) -> None:
        """FA3 incompatible with Blackwell."""
        router = GenerationRouter()

        assert router.is_fa3_compatible(GPUGeneration.BLACKWELL) is False

    def test_is_fa4_compatible_blackwell(self) -> None:
        """FA4 compatible with Blackwell."""
        router = GenerationRouter()

        assert router.is_fa4_compatible(GPUGeneration.BLACKWELL) is True

    def test_is_fa4_incompatible_hopper(self) -> None:
        """FA4 incompatible with Hopper."""
        router = GenerationRouter()

        assert router.is_fa4_compatible(GPUGeneration.HOPPER) is False


class TestGenerationRouterThreadSafety:
    """Tests for router thread safety."""

    def test_concurrent_routing(self, fa_kernels) -> None:
        """Router handles concurrent access."""
        import threading

        router = GenerationRouter()
        errors: list[Exception] = []

        def worker(gen: GPUGeneration) -> None:
            try:
                for _ in range(100):
                    router.route("attention", fa_kernels, gen)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(GPUGeneration.HOPPER,)),
            threading.Thread(target=worker, args=(GPUGeneration.BLACKWELL,)),
            threading.Thread(target=worker, args=(GPUGeneration.AMPERE,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
