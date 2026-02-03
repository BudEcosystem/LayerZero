"""Speculative decoding kernel coordination.

This module provides:
- SpeculativeConfig for speculative decoding configuration
- SpeculativeCoordinator for draft/target kernel coordination
- Verification utilities for rejection sampling
- Tree-based speculative decoding support
- Medusa-style parallel drafting support
"""
from __future__ import annotations

from layerzero.speculative.coordination import (
    SpeculativeConfig,
    SpeculativeCoordinator,
    MedusaConfig,
    validate_model_pair,
    compute_model_size_ratio,
    compute_verification_batch_size,
    compute_tree_attention_mask_size,
)
from layerzero.speculative.verification import (
    VerificationConfig,
    VerificationKernelSelector,
    VerificationMetrics,
    prepare_verification_batch,
    rejection_sample,
    greedy_verify,
    compute_accepted_tokens,
    get_first_rejection_index,
    select_continuation_tokens,
    estimate_effective_speedup,
)

__all__ = [
    # Coordination
    "SpeculativeConfig",
    "SpeculativeCoordinator",
    "MedusaConfig",
    "validate_model_pair",
    "compute_model_size_ratio",
    "compute_verification_batch_size",
    "compute_tree_attention_mask_size",
    # Verification
    "VerificationConfig",
    "VerificationKernelSelector",
    "VerificationMetrics",
    "prepare_verification_batch",
    "rejection_sample",
    "greedy_verify",
    "compute_accepted_tokens",
    "get_first_rejection_index",
    "select_continuation_tokens",
    "estimate_effective_speedup",
]
