"""FlashAttention 4 kernel specifications.

This module defines the KernelSpec for FA4.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from layerzero.device import GPUGeneration
from layerzero.enums import Layout, MaskType, Platform
from layerzero.models.kernel_spec import KernelSpec

if TYPE_CHECKING:
    from layerzero.backends.flash_attn_v4.adapter import FlashAttnV4Adapter

logger = logging.getLogger(__name__)


def create_fa4_kernel_spec(
    impl: "FlashAttnV4Adapter | None" = None,
    version_str: str = "4.0.0",
) -> KernelSpec:
    """Create FA4 kernel specification.

    Args:
        impl: Implementation adapter (optional).
        version_str: Version string.

    Returns:
        KernelSpec for FA4.
    """
    return KernelSpec(
        kernel_id="flash_attn.v4",
        operation="attention",
        source="flash_attn",
        version=version_str,
        impl=impl,
        platform=Platform.CUDA,
        # FA4 requires SM 10.0+ (Blackwell)
        min_sm=(10, 0),
        max_sm=None,  # No upper limit
        # Only Blackwell and newer
        supported_generations=frozenset([GPUGeneration.BLACKWELL]),
        # TC gen 5 (Blackwell)
        min_tensor_core_gen=5,
        # Supported dtypes
        supported_dtypes=frozenset([
            torch.float16,
            torch.bfloat16,
            # FP8 support (as additional dtypes when available)
        ]),
        # Head dim constraints
        min_head_dim=8,
        max_head_dim=256,
        head_dim_multiple=8,
        # Sequence constraints
        min_seq_len=1,
        max_seq_len=None,  # FA4 handles variable seq
        # Feature support
        supports_gqa=True,
        supports_mqa=True,
        supports_attn_mask=False,  # FA4 has different mask API
        supported_attn_mask_types=frozenset([MaskType.NONE]),
        supports_dropout=True,
        supports_scale=True,
        supports_alibi=True,
        # Layout support - FA4 uses BSHD
        requires_last_dim_stride1=True,
        requires_contiguous=True,
        requires_layouts=frozenset([Layout.BSHD]),
        produces_layout=Layout.BSHD,
        # Execution properties
        is_cuda_graph_safe=True,
        deterministic=False,
        workspace_bytes=0,
        # Highest priority for Blackwell
        priority=100,
    )


# Default FA4 kernel spec
FA4_KERNEL_SPEC = create_fa4_kernel_spec()
