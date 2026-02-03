"""
LayerZero Core Enumerations

Type-safe enums for operations, layouts, masks, platforms, and quantization.
All enums inherit from (str, Enum) for JSON serialization compatibility.

This module provides:
- OpKind: Classification of operations (tensor, tokenization, sampling)
- Layout: Tensor layout formats (BSHD, BHSD, NHD, HND)
- MaskType: Attention mask types (none, bool, float)
- Platform: Hardware platforms (cuda, rocm, cpu, hpu, xpu)
- QuantFormat: Quantization formats (int4, int8, fp8, etc.)
- KVCacheStrategy: KV cache management strategies
"""
from __future__ import annotations

from enum import Enum, unique


@unique
class OpKind(str, Enum):
    """Operation kind classification.

    Used to categorize operations for kernel lookup and selection.

    Members:
        TENSOR: Tensor operations (attention, matmul, norms, etc.)
        TOKENIZATION: Text tokenization operations
        SAMPLING: Token sampling operations (top-k, top-p, etc.)
        COMMUNICATION: Distributed communication (all-reduce, etc.)
        PREPOST: Pre/post processing operations
    """

    TENSOR = "tensor"
    TOKENIZATION = "tokenization"
    SAMPLING = "sampling"
    COMMUNICATION = "communication"
    PREPOST = "prepost"


@unique
class Layout(str, Enum):
    """Tensor layout formats for attention operations.

    Different backends expect different tensor layouts. LayerZero
    handles layout detection and transformation.

    Members:
        BSHD: Batch, Sequence, Heads, Dim (PyTorch SDPA default)
        BHSD: Batch, Heads, Sequence, Dim (FlashAttention default)
        NHD: Num_tokens, Heads, Dim (FlashInfer ragged batch)
        HND: Heads, Num_tokens, Dim (Alternative ragged format)
    """

    BSHD = "BSHD"
    BHSD = "BHSD"
    NHD = "NHD"
    HND = "HND"


@unique
class MaskType(str, Enum):
    """Attention mask types.

    Different kernels support different mask types. LayerZero
    validates mask compatibility during selection.

    Members:
        NONE: No attention mask
        BOOL: Boolean mask (True = attend, False = ignore)
        FLOAT: Additive float mask (added to attention scores)
    """

    NONE = "none"
    BOOL = "bool"
    FLOAT = "float"


@unique
class Platform(str, Enum):
    """Hardware platform types.

    Used for platform-specific kernel filtering and backend selection.

    Members:
        CUDA: NVIDIA GPUs via CUDA
        ROCM: AMD GPUs via ROCm
        CPU: CPU-only (Intel/AMD/ARM)
        HPU: Intel Habana Gaudi
        XPU: Intel discrete GPUs
    """

    CUDA = "cuda"
    ROCM = "rocm"
    CPU = "cpu"
    HPU = "hpu"
    XPU = "xpu"


@unique
class QuantFormat(str, Enum):
    """Quantization formats.

    Different quantization formats have different precision,
    performance, and hardware support characteristics.

    Members:
        INT4: 4-bit integer (GPTQ, AWQ style)
        INT8: 8-bit integer (SmoothQuant, LLM.int8 style)
        NVFP4: NVIDIA's native FP4 (Blackwell+)
        MXFP4: Microscaling FP4 (OCP standard)
        FP8_E4M3: 8-bit float with 4 exponent, 3 mantissa bits
        FP8_E5M2: 8-bit float with 5 exponent, 2 mantissa bits
    """

    INT4 = "int4"
    INT8 = "int8"
    NVFP4 = "nvfp4"
    MXFP4 = "mxfp4"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


@unique
class KVCacheStrategy(str, Enum):
    """KV cache management strategies.

    Different strategies trade off between memory efficiency,
    latency, and implementation complexity.

    Members:
        CONTIGUOUS: Contiguous memory allocation (simple but wasteful)
        PAGED: PagedAttention-style block allocation (vLLM)
        VIRTUAL: Virtual memory-based (FlashInfer cascade)
        UNIFIED: Unified prefix/suffix caching (SGLang)
    """

    CONTIGUOUS = "contiguous"
    PAGED = "paged"
    VIRTUAL = "virtual"
    UNIFIED = "unified"


@unique
class AttentionVariant(str, Enum):
    """Attention computation variants.

    Different attention algorithms optimized for different use cases.

    Members:
        PREFILL: Full prefill/prompt attention
        DECODE: Single token decode attention
        PREFILL_WITH_PAGED_KV: Prefill with paged KV cache
        DECODE_WITH_PAGED_KV: Decode with paged KV cache
        CROSS: Cross attention (encoder-decoder)
    """

    PREFILL = "prefill"
    DECODE = "decode"
    PREFILL_WITH_PAGED_KV = "prefill_paged"
    DECODE_WITH_PAGED_KV = "decode_paged"
    CROSS = "cross"


@unique
class PositionalEncoding(str, Enum):
    """Positional encoding schemes.

    Different models use different positional encoding methods.

    Members:
        NONE: No positional encoding
        ROPE: Rotary Position Embedding
        ROPE_INTERLEAVED: RoPE with interleaved layout
        ALIBI: Attention with Linear Biases
        XPOS: Extended RoPE variant
        SINUSOIDAL: Classic sinusoidal encoding
    """

    NONE = "none"
    ROPE = "rope"
    ROPE_INTERLEAVED = "rope_interleaved"
    ALIBI = "alibi"
    XPOS = "xpos"
    SINUSOIDAL = "sinusoidal"


@unique
class NormType(str, Enum):
    """Normalization layer types.

    Different normalization methods used in transformer architectures.

    Members:
        LAYER_NORM: Standard LayerNorm
        RMS_NORM: Root Mean Square normalization (LLaMA)
        GROUP_NORM: Group normalization (vision models)
        BATCH_NORM: Batch normalization
    """

    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    GROUP_NORM = "group_norm"
    BATCH_NORM = "batch_norm"


@unique
class ActivationType(str, Enum):
    """Activation function types.

    Common activation functions used in transformer MLPs.

    Members:
        GELU: Gaussian Error Linear Unit
        GELU_TANH: GELU approximation using tanh
        SILU: Sigmoid Linear Unit (SwiGLU)
        RELU: Rectified Linear Unit
        QUICK_GELU: Fast GELU approximation
    """

    GELU = "gelu"
    GELU_TANH = "gelu_tanh"
    SILU = "silu"
    RELU = "relu"
    QUICK_GELU = "quick_gelu"
