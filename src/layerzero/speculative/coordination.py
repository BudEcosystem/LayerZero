"""Speculative decoding kernel coordination.

This module provides:
- SpeculativeConfig for configuration
- SpeculativeCoordinator for draft/target kernel selection
- Model pair validation
- Tree and Medusa speculative decoding support
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from layerzero.device import GPUGeneration

logger = logging.getLogger(__name__)


@dataclass
class MedusaConfig:
    """Configuration for Medusa-style parallel drafting.

    Attributes:
        num_heads: Number of Medusa heads.
        head_predictions: Predictions per head.
    """

    num_heads: int = 4
    head_predictions: int = 5


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.

    Attributes:
        speculation_length: Number of tokens to speculate.
        use_tree_attention: Enable tree-based speculation.
        max_tree_width: Maximum tree branching factor.
        max_tree_depth: Maximum tree depth.
        medusa_config: Optional Medusa configuration.
    """

    speculation_length: int = 5
    use_tree_attention: bool = False
    max_tree_width: int = 1
    max_tree_depth: int = 1
    medusa_config: MedusaConfig | None = None


class SpeculativeCoordinator:
    """Coordinates kernel selection for speculative decoding.

    Handles draft model (low-latency) and target model (high-throughput)
    kernel selection, as well as verification kernel requirements.
    """

    def __init__(self, config: SpeculativeConfig) -> None:
        """Initialize coordinator.

        Args:
            config: Speculative decoding configuration.
        """
        self._config = config
        self._lock = RLock()

        logger.debug(
            "SpeculativeCoordinator initialized: speculation_length=%d, tree=%s",
            config.speculation_length,
            config.use_tree_attention,
        )

    def select_draft_kernel(
        self,
        model_config: dict[str, Any],
        gpu_generation: GPUGeneration,
    ) -> str:
        """Select kernel for draft model.

        Args:
            model_config: Draft model configuration.
            gpu_generation: GPU generation.

        Returns:
            Kernel ID for draft model.
        """
        with self._lock:
            # Draft model prioritizes latency over throughput
            # Small models benefit from lightweight kernels

            num_heads = model_config.get("num_heads", 4)
            head_dim = model_config.get("head_dim", 64)

            # For small models, use simple flash attention
            if gpu_generation >= GPUGeneration.HOPPER:
                kernel_id = "flash_attn_v3_draft"
            elif gpu_generation >= GPUGeneration.AMPERE:
                kernel_id = "flash_attn_v2_draft"
            else:
                kernel_id = "sdpa_draft"

            logger.debug(
                "Selected draft kernel: %s for %s",
                kernel_id,
                gpu_generation.value,
            )

            return kernel_id

    def select_target_kernel(
        self,
        model_config: dict[str, Any],
        gpu_generation: GPUGeneration,
    ) -> str:
        """Select kernel for target model.

        Args:
            model_config: Target model configuration.
            gpu_generation: GPU generation.

        Returns:
            Kernel ID for target model.
        """
        with self._lock:
            # Target model can use high-throughput kernels
            # GQA optimization is important for large models

            num_kv_heads = model_config.get("num_kv_heads", 8)
            num_heads = model_config.get("num_heads", 64)

            # Check if GQA
            is_gqa = num_kv_heads < num_heads

            if gpu_generation >= GPUGeneration.HOPPER:
                kernel_id = "flash_attn_v3_gqa" if is_gqa else "flash_attn_v3"
            elif gpu_generation >= GPUGeneration.AMPERE:
                kernel_id = "flash_attn_v2_gqa" if is_gqa else "flash_attn_v2"
            else:
                kernel_id = "sdpa_target"

            logger.debug(
                "Selected target kernel: %s (GQA=%s)",
                kernel_id,
                is_gqa,
            )

            return kernel_id

    def select_verification_kernel(
        self,
        model_config: dict[str, Any],
        gpu_generation: GPUGeneration,
        batch_size: int,
        speculation_length: int,
    ) -> str:
        """Select kernel for verification.

        Args:
            model_config: Target model configuration.
            gpu_generation: GPU generation.
            batch_size: Batch size.
            speculation_length: Number of speculated tokens.

        Returns:
            Kernel ID for verification.
        """
        with self._lock:
            # Verification needs to handle expanded batch
            expanded_batch = compute_verification_batch_size(
                batch_size, speculation_length
            )

            if self._config.use_tree_attention:
                kernel_id = "tree_attention_verify"
            elif gpu_generation >= GPUGeneration.HOPPER:
                kernel_id = "flash_attn_v3_verify"
            else:
                kernel_id = "flash_attn_v2_verify"

            logger.debug(
                "Selected verification kernel: %s for batch=%d->%d",
                kernel_id,
                batch_size,
                expanded_batch,
            )

            return kernel_id

    def select_tree_attention_kernel(
        self,
        gpu_generation: GPUGeneration,
    ) -> str:
        """Select kernel for tree attention.

        Args:
            gpu_generation: GPU generation.

        Returns:
            Kernel ID for tree attention.
        """
        with self._lock:
            if gpu_generation >= GPUGeneration.HOPPER:
                return "tree_attention_hopper"
            else:
                return "tree_attention_generic"

    def get_draft_kernel_requirements(self) -> dict[str, Any]:
        """Get requirements for draft model kernel.

        Returns:
            Dictionary of kernel requirements.
        """
        return {
            "prefer_low_latency": True,
            "prefer_throughput": False,
            "max_batch_size": 8,
            "supports_variable_length": True,
        }

    def get_target_kernel_requirements(self) -> dict[str, Any]:
        """Get requirements for target model kernel.

        Returns:
            Dictionary of kernel requirements.
        """
        return {
            "prefer_low_latency": False,
            "prefer_throughput": True,
            "supports_gqa": True,
            "supports_paged_kv": True,
        }

    def get_verification_requirements(self) -> dict[str, Any]:
        """Get requirements for verification kernel.

        Returns:
            Dictionary of kernel requirements.
        """
        return {
            "max_expansion": self._config.speculation_length,
            "supports_batch_expansion": True,
            "supports_rejection_sampling": True,
        }

    def get_kv_cache_sharing_requirements(self) -> dict[str, Any]:
        """Get KV cache sharing requirements.

        Returns:
            Dictionary of sharing requirements.
        """
        return {
            "share_verified_prefix": True,
            "draft_cache_separate": True,
            "reuse_target_cache": True,
        }

    def supports_tree_speculation(self) -> bool:
        """Check if tree speculation is supported.

        Returns:
            True if tree speculation is enabled.
        """
        return self._config.use_tree_attention

    def get_medusa_requirements(self) -> dict[str, Any]:
        """Get Medusa-specific requirements.

        Returns:
            Dictionary of Medusa requirements.
        """
        if self._config.medusa_config is None:
            return {
                "parallel_heads": 0,
                "predictions_per_head": 0,
            }

        return {
            "parallel_heads": self._config.medusa_config.num_heads,
            "predictions_per_head": self._config.medusa_config.head_predictions,
        }


# ============================================================================
# Model Pair Validation
# ============================================================================


def validate_model_pair(
    draft_config: dict[str, Any],
    target_config: dict[str, Any],
) -> tuple[bool, str]:
    """Validate draft and target model compatibility.

    Args:
        draft_config: Draft model configuration.
        target_config: Target model configuration.

    Returns:
        Tuple of (is_valid, reason).
    """
    # Check vocabulary size match
    draft_vocab = draft_config.get("vocab_size", 0)
    target_vocab = target_config.get("vocab_size", 0)

    if draft_vocab != target_vocab:
        return False, f"Vocab size mismatch: draft={draft_vocab}, target={target_vocab}"

    # Additional validation can be added here

    return True, "Valid model pair"


def compute_model_size_ratio(
    draft_config: dict[str, Any],
    target_config: dict[str, Any],
) -> float:
    """Compute draft/target model size ratio.

    Args:
        draft_config: Draft model configuration.
        target_config: Target model configuration.

    Returns:
        Size ratio (draft_params / target_params).
    """
    def estimate_params(config: dict[str, Any]) -> int:
        layers = config.get("num_layers", 1)
        hidden = config.get("hidden_size", 256)
        vocab = config.get("vocab_size", 32000)

        # Rough estimate: embeddings + attention + FFN per layer
        embeddings = vocab * hidden
        per_layer = 4 * hidden * hidden + 8 * hidden * hidden  # Attention + FFN
        total = embeddings + layers * per_layer

        return total

    draft_params = estimate_params(draft_config)
    target_params = estimate_params(target_config)

    if target_params == 0:
        return 0.0

    return draft_params / target_params


# ============================================================================
# Batch Size and Mask Utilities
# ============================================================================


def compute_verification_batch_size(
    batch_size: int,
    speculation_length: int,
) -> int:
    """Compute expanded batch size for verification.

    For each sequence, we need to verify speculation_length + 1 positions
    (original + speculated tokens).

    Args:
        batch_size: Original batch size.
        speculation_length: Number of speculated tokens.

    Returns:
        Expanded batch size.
    """
    return batch_size * (speculation_length + 1)


def compute_tree_attention_mask_size(
    tree_width: int,
    tree_depth: int,
) -> int:
    """Compute attention mask size for tree speculation.

    Args:
        tree_width: Branching factor.
        tree_depth: Tree depth.

    Returns:
        Number of nodes in tree.
    """
    # Geometric series: 1 + w + w^2 + ... + w^(d-1)
    if tree_width == 1:
        return tree_depth

    total_nodes = (tree_width ** tree_depth - 1) // (tree_width - 1)
    return total_nodes
