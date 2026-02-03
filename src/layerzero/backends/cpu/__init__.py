"""
LayerZero CPU Backend Adapters

Support for CPU-optimized backends including:
- Intel oneDNN for Intel CPU optimization
- AMD ZenDNN for AMD EPYC optimization
- Intel Extension for PyTorch (IPEX) for Intel hardware
"""
from __future__ import annotations

from layerzero.backends.cpu.detection import (
    CPUVendor,
    ISAFeature,
    detect_cpu_vendor,
    detect_isa_features,
    get_cpu_info,
    get_optimal_cpu_backend,
)
from layerzero.backends.cpu.onednn import (
    OneDNNLayerNormAdapter,
    OneDNNMatmulAdapter,
    detect_onednn_version,
    is_onednn_available,
)
from layerzero.backends.cpu.zendnn import (
    ZenDNNMatmulAdapter,
    detect_zendnn_version,
    is_aocl_blas_available,
    is_zendnn_available,
)
from layerzero.backends.cpu.ipex import (
    IPEXAttentionAdapter,
    IPEXMatmulAdapter,
    detect_ipex_version,
    is_ipex_available,
    is_xpu_available,
)

__all__ = [
    # Detection
    "CPUVendor",
    "ISAFeature",
    "detect_cpu_vendor",
    "detect_isa_features",
    "get_cpu_info",
    "get_optimal_cpu_backend",
    # oneDNN
    "OneDNNLayerNormAdapter",
    "OneDNNMatmulAdapter",
    "detect_onednn_version",
    "is_onednn_available",
    # ZenDNN
    "ZenDNNMatmulAdapter",
    "detect_zendnn_version",
    "is_aocl_blas_available",
    "is_zendnn_available",
    # IPEX
    "IPEXAttentionAdapter",
    "IPEXMatmulAdapter",
    "detect_ipex_version",
    "is_ipex_available",
    "is_xpu_available",
]
