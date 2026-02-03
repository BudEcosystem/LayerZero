"""Tests for speculative decoding kernel coordination."""
from __future__ import annotations

import pytest
from typing import Any

from layerzero.device import GPUGeneration


class TestSpeculativeConfig:
    """Tests for speculative decoding configuration."""

    def test_config_default_values(self) -> None:
        """Config has sensible defaults."""
        from layerzero.speculative.coordination import SpeculativeConfig

        config = SpeculativeConfig()

        assert config.speculation_length == 5
        assert config.use_tree_attention is False
        assert config.max_tree_width == 1

    def test_config_custom_speculation_length(self) -> None:
        """Custom speculation length."""
        from layerzero.speculative.coordination import SpeculativeConfig

        config = SpeculativeConfig(speculation_length=8)
        assert config.speculation_length == 8

    def test_config_tree_speculation(self) -> None:
        """Tree speculation configuration."""
        from layerzero.speculative.coordination import SpeculativeConfig

        config = SpeculativeConfig(
            use_tree_attention=True,
            max_tree_width=4,
        )

        assert config.use_tree_attention is True
        assert config.max_tree_width == 4


class TestSpeculativeCoordination:
    """Tests for draft/target model kernel coordination."""

    def test_draft_model_kernel_selection(
        self, draft_model_config: dict, hopper_device: GPUGeneration
    ) -> None:
        """Draft model kernel selection works."""
        from layerzero.speculative.coordination import (
            SpeculativeCoordinator,
            SpeculativeConfig,
        )

        config = SpeculativeConfig(speculation_length=5)
        coordinator = SpeculativeCoordinator(config)

        kernel_id = coordinator.select_draft_kernel(
            model_config=draft_model_config,
            gpu_generation=hopper_device,
        )

        assert kernel_id is not None
        # Draft should prefer low-latency kernels
        assert isinstance(kernel_id, str)

    def test_target_model_kernel_selection(
        self, target_model_config: dict, hopper_device: GPUGeneration
    ) -> None:
        """Target model kernel selection works."""
        from layerzero.speculative.coordination import (
            SpeculativeCoordinator,
            SpeculativeConfig,
        )

        config = SpeculativeConfig(speculation_length=5)
        coordinator = SpeculativeCoordinator(config)

        kernel_id = coordinator.select_target_kernel(
            model_config=target_model_config,
            gpu_generation=hopper_device,
        )

        assert kernel_id is not None
        # Target can use high-throughput kernels
        assert isinstance(kernel_id, str)

    def test_verification_kernel_selection(
        self, target_model_config: dict, hopper_device: GPUGeneration
    ) -> None:
        """Verification kernel selection works."""
        from layerzero.speculative.coordination import (
            SpeculativeCoordinator,
            SpeculativeConfig,
        )

        config = SpeculativeConfig(speculation_length=5)
        coordinator = SpeculativeCoordinator(config)

        kernel_id = coordinator.select_verification_kernel(
            model_config=target_model_config,
            gpu_generation=hopper_device,
            batch_size=1,
            speculation_length=5,
        )

        assert kernel_id is not None

    def test_draft_prefers_low_latency(
        self, draft_model_config: dict, hopper_device: GPUGeneration
    ) -> None:
        """Draft model prefers low-latency kernels."""
        from layerzero.speculative.coordination import (
            SpeculativeCoordinator,
            SpeculativeConfig,
        )

        config = SpeculativeConfig()
        coordinator = SpeculativeCoordinator(config)

        requirements = coordinator.get_draft_kernel_requirements()

        assert requirements["prefer_low_latency"] is True
        assert requirements["prefer_throughput"] is False

    def test_target_prefers_throughput(
        self, target_model_config: dict, hopper_device: GPUGeneration
    ) -> None:
        """Target model prefers throughput kernels."""
        from layerzero.speculative.coordination import (
            SpeculativeCoordinator,
            SpeculativeConfig,
        )

        config = SpeculativeConfig()
        coordinator = SpeculativeCoordinator(config)

        requirements = coordinator.get_target_kernel_requirements()

        assert requirements["prefer_throughput"] is True


class TestSpeculativeRequirements:
    """Tests for speculative decoding requirements."""

    def test_speculation_length_affects_selection(
        self, target_model_config: dict, hopper_device: GPUGeneration
    ) -> None:
        """Speculation length affects kernel selection."""
        from layerzero.speculative.coordination import (
            SpeculativeCoordinator,
            SpeculativeConfig,
        )

        # Short speculation
        config_short = SpeculativeConfig(speculation_length=2)
        coordinator_short = SpeculativeCoordinator(config_short)

        # Long speculation
        config_long = SpeculativeConfig(speculation_length=16)
        coordinator_long = SpeculativeCoordinator(config_long)

        # Requirements should differ
        req_short = coordinator_short.get_verification_requirements()
        req_long = coordinator_long.get_verification_requirements()

        assert req_short["max_expansion"] == 2
        assert req_long["max_expansion"] == 16

    def test_batch_expansion_handled(
        self, target_model_config: dict, hopper_device: GPUGeneration
    ) -> None:
        """Batch expansion for verification handled."""
        from layerzero.speculative.coordination import (
            SpeculativeCoordinator,
            SpeculativeConfig,
            compute_verification_batch_size,
        )

        config = SpeculativeConfig(speculation_length=5)

        # Batch of 4 sequences, each with 5 speculative tokens
        expanded_batch = compute_verification_batch_size(
            batch_size=4,
            speculation_length=5,
        )

        # Need to verify 5 tokens per sequence -> expanded batch
        assert expanded_batch == 4 * 6  # original + 5 speculative per seq

    def test_kv_cache_shared_between_draft_target(self) -> None:
        """KV cache sharing requirements."""
        from layerzero.speculative.coordination import (
            SpeculativeCoordinator,
            SpeculativeConfig,
        )

        config = SpeculativeConfig()
        coordinator = SpeculativeCoordinator(config)

        sharing_info = coordinator.get_kv_cache_sharing_requirements()

        # Draft and target should share verified prefix
        assert sharing_info["share_verified_prefix"] is True


class TestTreeSpeculation:
    """Tests for tree-based speculative decoding."""

    def test_tree_based_speculation_supported(self) -> None:
        """Tree-based speculation is supported."""
        from layerzero.speculative.coordination import (
            SpeculativeCoordinator,
            SpeculativeConfig,
        )

        config = SpeculativeConfig(
            use_tree_attention=True,
            max_tree_width=4,
            max_tree_depth=3,
        )

        coordinator = SpeculativeCoordinator(config)

        assert coordinator.supports_tree_speculation()

    def test_tree_attention_kernel(self, hopper_device: GPUGeneration) -> None:
        """Tree attention kernel works."""
        from layerzero.speculative.coordination import (
            SpeculativeCoordinator,
            SpeculativeConfig,
        )

        config = SpeculativeConfig(
            use_tree_attention=True,
            max_tree_width=4,
        )

        coordinator = SpeculativeCoordinator(config)

        kernel_id = coordinator.select_tree_attention_kernel(
            gpu_generation=hopper_device,
        )

        assert kernel_id is not None

    def test_tree_mask_requirements(self) -> None:
        """Tree attention mask requirements."""
        from layerzero.speculative.coordination import (
            SpeculativeCoordinator,
            SpeculativeConfig,
            compute_tree_attention_mask_size,
        )

        config = SpeculativeConfig(
            use_tree_attention=True,
            max_tree_width=4,
            max_tree_depth=3,
        )

        # For tree with width=4, depth=3
        # Total nodes: 1 + 4 + 16 = 21
        mask_size = compute_tree_attention_mask_size(
            tree_width=4,
            tree_depth=3,
        )

        expected_nodes = 1 + 4 + 16  # root + level1 + level2
        assert mask_size >= expected_nodes


class TestMedusaIntegration:
    """Tests for Medusa-style parallel drafting."""

    def test_medusa_heads_config(self) -> None:
        """Medusa heads configuration."""
        from layerzero.speculative.coordination import (
            SpeculativeConfig,
            MedusaConfig,
        )

        medusa = MedusaConfig(
            num_heads=4,
            head_predictions=5,
        )

        config = SpeculativeConfig(
            medusa_config=medusa,
        )

        assert config.medusa_config is not None
        assert config.medusa_config.num_heads == 4
        assert config.medusa_config.head_predictions == 5

    def test_medusa_kernel_selection(self, hopper_device: GPUGeneration) -> None:
        """Medusa-specific kernel selection."""
        from layerzero.speculative.coordination import (
            SpeculativeCoordinator,
            SpeculativeConfig,
            MedusaConfig,
        )

        medusa = MedusaConfig(num_heads=4, head_predictions=5)
        config = SpeculativeConfig(medusa_config=medusa)
        coordinator = SpeculativeCoordinator(config)

        # Medusa needs parallel head execution
        requirements = coordinator.get_medusa_requirements()

        assert requirements["parallel_heads"] == 4
        assert requirements["predictions_per_head"] == 5


class TestSpeculativeModelPair:
    """Tests for draft/target model pairing."""

    def test_validate_model_pair(
        self, draft_model_config: dict, target_model_config: dict
    ) -> None:
        """Validate draft and target model compatibility."""
        from layerzero.speculative.coordination import validate_model_pair

        is_valid, reason = validate_model_pair(
            draft_config=draft_model_config,
            target_config=target_model_config,
        )

        # Same vocab size required
        assert is_valid is True

    def test_reject_vocab_mismatch(
        self, draft_model_config: dict, target_model_config: dict
    ) -> None:
        """Reject models with vocabulary mismatch."""
        from layerzero.speculative.coordination import validate_model_pair

        # Modify vocab size
        draft_model_config["vocab_size"] = 50000

        is_valid, reason = validate_model_pair(
            draft_config=draft_model_config,
            target_config=target_model_config,
        )

        assert is_valid is False
        assert "vocab" in reason.lower()

    def test_compute_size_ratio(
        self, draft_model_config: dict, target_model_config: dict
    ) -> None:
        """Compute draft/target size ratio."""
        from layerzero.speculative.coordination import compute_model_size_ratio

        ratio = compute_model_size_ratio(
            draft_config=draft_model_config,
            target_config=target_model_config,
        )

        # Draft should be much smaller
        assert ratio < 0.1  # Draft is less than 10% of target
