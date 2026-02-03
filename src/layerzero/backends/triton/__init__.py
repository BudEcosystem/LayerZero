"""
LayerZero Triton Custom Kernel Backend

Support for registering and managing custom Triton kernels.
Users can register their own Triton kernels with LayerZero for use
in the kernel selection system.

Key features:
- Registration API for custom Triton kernels
- Automatic KernelSpec generation
- Grid/block configuration validation
- CUDA and ROCm support via Triton
"""
from __future__ import annotations

from layerzero.backends.triton.version import (
    detect_triton_version,
    get_triton_backend,
    get_triton_info,
    is_triton_available,
)
from layerzero.backends.triton.registry import (
    TritonKernelRegistry,
    get_registry,
    register_triton_kernel,
)
from layerzero.backends.triton.config import (
    DEFAULT_MAX_GRID,
    DEFAULT_MAX_THREADS,
    GridConfig,
    validate_block_config,
    validate_grid_config,
)
from layerzero.backends.triton.adapter import TritonKernelAdapter

__all__ = [
    # Version detection
    "detect_triton_version",
    "get_triton_backend",
    "get_triton_info",
    "is_triton_available",
    # Registry
    "TritonKernelRegistry",
    "get_registry",
    "register_triton_kernel",
    # Config
    "DEFAULT_MAX_GRID",
    "DEFAULT_MAX_THREADS",
    "GridConfig",
    "validate_block_config",
    "validate_grid_config",
    # Adapter
    "TritonKernelAdapter",
]
