"""
LayerZero Sampling Module

Provides efficient sampling operations for token generation:
- Top-K sampling
- Top-P (nucleus) sampling
- Combined top-K + top-P sampling
- Temperature scaling utilities
"""
from __future__ import annotations

from layerzero.sampling.topk import topk_sample
from layerzero.sampling.topp import topp_sample
from layerzero.sampling.combined import topk_topp_sample
from layerzero.sampling.temperature import apply_temperature

__all__ = [
    "topk_sample",
    "topp_sample",
    "topk_topp_sample",
    "apply_temperature",
]
