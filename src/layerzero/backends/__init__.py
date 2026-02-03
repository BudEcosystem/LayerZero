"""
LayerZero Backend Adapters

Kernel adapters for various attention backend libraries.
"""
from layerzero.backends.base import BaseKernel
from layerzero.backends.flash_attn import FlashAttnAdapter
from layerzero.backends.flashinfer import (
    FlashInferPrefillAdapter,
    FlashInferDecodeAdapter,
    FlashInferPagedAdapter,
)
from layerzero.backends.torch_sdpa import TorchSDPAAdapter
from layerzero.backends.xformers import XFormersAdapter
from layerzero.backends.liger import (
    LigerRMSNormAdapter,
    LigerRoPEAdapter,
    LigerSwiGLUAdapter,
    LigerCrossEntropyAdapter,
)
from layerzero.backends.triton import (
    TritonKernelAdapter,
    TritonKernelRegistry,
    get_registry as get_triton_registry,
    register_triton_kernel,
)
from layerzero.backends.cpu import (
    OneDNNMatmulAdapter,
    OneDNNLayerNormAdapter,
    ZenDNNMatmulAdapter,
    IPEXMatmulAdapter,
    IPEXAttentionAdapter,
)
from layerzero.backends.hf_kernels import (
    HFKernelAdapter,
    HFKernelLoader,
    KernelLockfile,
)

__all__ = [
    "BaseKernel",
    "FlashAttnAdapter",
    "FlashInferPrefillAdapter",
    "FlashInferDecodeAdapter",
    "FlashInferPagedAdapter",
    "TorchSDPAAdapter",
    "XFormersAdapter",
    "LigerRMSNormAdapter",
    "LigerRoPEAdapter",
    "LigerSwiGLUAdapter",
    "LigerCrossEntropyAdapter",
    "TritonKernelAdapter",
    "TritonKernelRegistry",
    "get_triton_registry",
    "register_triton_kernel",
    "OneDNNMatmulAdapter",
    "OneDNNLayerNormAdapter",
    "ZenDNNMatmulAdapter",
    "IPEXMatmulAdapter",
    "IPEXAttentionAdapter",
    "HFKernelAdapter",
    "HFKernelLoader",
    "KernelLockfile",
]
