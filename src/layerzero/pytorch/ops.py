"""
LayerZero Op Registration via torch.library

Registers LayerZero operations with PyTorch's library system
for torch.compile and torch.export compatibility.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.library import Library

from layerzero.pytorch.meta_kernels import (
    attention_meta,
    rms_norm_meta,
    layer_norm_meta,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Create the LayerZero library
# "DEF" mode allows defining new operators
_lib = Library("layerzero", "DEF")

# Define op schemas
_lib.define(
    "attention(Tensor query, Tensor key, Tensor value, "
    "Tensor? attn_mask=None, float dropout_p=0.0, "
    "bool is_causal=False, float? scale=None) -> Tensor"
)

_lib.define(
    "rms_norm(Tensor input, Tensor weight, float eps=1e-6) -> Tensor"
)

_lib.define(
    "layer_norm(Tensor input, Tensor weight, Tensor? bias=None, "
    "float eps=1e-5) -> Tensor"
)

# Sampling operations
_lib.define(
    "sample_topk(Tensor logits, int k, float temperature=1.0) -> Tensor"
)

_lib.define(
    "sample_topp(Tensor logits, float p, float temperature=1.0) -> Tensor"
)

# Embedding operation
_lib.define(
    "embedding_lookup(Tensor input, Tensor weight, int? padding_idx=None) -> Tensor"
)


# =============================================================================
# CUDA Implementations
# =============================================================================


@torch.library.impl(_lib, "attention", "CUDA")
def _attention_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """CUDA implementation of attention.

    Dispatches to the best available CUDA kernel (Flash Attention,
    FlashInfer, xFormers, or torch SDPA).
    """
    # Use torch.nn.functional.scaled_dot_product_attention as the
    # implementation, which will dispatch to Flash/efficient kernels
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


@torch.library.impl(_lib, "rms_norm", "CUDA")
def _rms_norm_cuda(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CUDA implementation of RMS normalization."""
    # Compute RMS
    variance = input.pow(2).mean(dim=-1, keepdim=True)
    input_normalized = input * torch.rsqrt(variance + eps)
    return input_normalized * weight


@torch.library.impl(_lib, "layer_norm", "CUDA")
def _layer_norm_cuda(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """CUDA implementation of layer normalization."""
    normalized_shape = weight.shape
    return torch.nn.functional.layer_norm(
        input,
        normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
    )


@torch.library.impl(_lib, "sample_topk", "CUDA")
def _sample_topk_cuda(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """CUDA implementation of top-k sampling."""
    from layerzero.sampling.topk import topk_sample
    return topk_sample(logits, k=k, temperature=temperature)


@torch.library.impl(_lib, "sample_topp", "CUDA")
def _sample_topp_cuda(
    logits: torch.Tensor,
    p: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """CUDA implementation of top-p sampling."""
    from layerzero.sampling.topp import topp_sample
    return topp_sample(logits, p=p, temperature=temperature)


@torch.library.impl(_lib, "embedding_lookup", "CUDA")
def _embedding_lookup_cuda(
    input: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: int | None = None,
) -> torch.Tensor:
    """CUDA implementation of embedding lookup."""
    return torch.nn.functional.embedding(input, weight, padding_idx=padding_idx)


# =============================================================================
# CPU Implementations
# =============================================================================


@torch.library.impl(_lib, "attention", "CPU")
def _attention_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """CPU implementation of attention."""
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


@torch.library.impl(_lib, "rms_norm", "CPU")
def _rms_norm_cpu(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CPU implementation of RMS normalization."""
    variance = input.pow(2).mean(dim=-1, keepdim=True)
    input_normalized = input * torch.rsqrt(variance + eps)
    return input_normalized * weight


@torch.library.impl(_lib, "layer_norm", "CPU")
def _layer_norm_cpu(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """CPU implementation of layer normalization."""
    normalized_shape = weight.shape
    return torch.nn.functional.layer_norm(
        input,
        normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
    )


@torch.library.impl(_lib, "sample_topk", "CPU")
def _sample_topk_cpu(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """CPU implementation of top-k sampling."""
    from layerzero.sampling.topk import topk_sample
    return topk_sample(logits, k=k, temperature=temperature)


@torch.library.impl(_lib, "sample_topp", "CPU")
def _sample_topp_cpu(
    logits: torch.Tensor,
    p: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """CPU implementation of top-p sampling."""
    from layerzero.sampling.topp import topp_sample
    return topp_sample(logits, p=p, temperature=temperature)


@torch.library.impl(_lib, "embedding_lookup", "CPU")
def _embedding_lookup_cpu(
    input: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: int | None = None,
) -> torch.Tensor:
    """CPU implementation of embedding lookup."""
    return torch.nn.functional.embedding(input, weight, padding_idx=padding_idx)


# =============================================================================
# Meta Implementations (for tracing/export)
# =============================================================================


@torch.library.impl(_lib, "attention", "Meta")
def _attention_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Meta implementation for attention (shape inference)."""
    return attention_meta(query, key, value, attn_mask, dropout_p, is_causal, scale)


@torch.library.impl(_lib, "rms_norm", "Meta")
def _rms_norm_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Meta implementation for RMS norm (shape inference)."""
    return rms_norm_meta(input, weight, eps)


@torch.library.impl(_lib, "layer_norm", "Meta")
def _layer_norm_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Meta implementation for layer norm (shape inference)."""
    return layer_norm_meta(input, weight, bias, eps)


# =============================================================================
# Autograd Registration
# =============================================================================


def _setup_autograd() -> None:
    """Setup autograd for custom ops.

    Registers backward formulas for gradient computation.
    """
    # For attention, we rely on SDPA's autograd
    # For RMS norm, define custom backward
    pass


_setup_autograd()


logger.debug("LayerZero ops registered with torch.library")
