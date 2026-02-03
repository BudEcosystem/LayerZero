"""
LayerZero HuggingFace Kernel Hub Integration

Support for dynamically loading pre-compiled kernels from HuggingFace Hub.

Key features:
- Dynamic kernel loading from HuggingFace Hub
- ABI3 and manylinux_2_28 compatibility validation
- torch.ops namespace management
- Kernel lockfiles for reproducibility
"""
from __future__ import annotations

from layerzero.backends.hf_kernels.version import (
    detect_hf_kernels_version,
    get_hf_kernels_info,
    is_hf_kernels_available,
)
from layerzero.backends.hf_kernels.loader import (
    HFKernelLoader,
    LoadedKernel,
)
from layerzero.backends.hf_kernels.validator import (
    ABIValidator,
    validate_abi3_compatibility,
    validate_manylinux_compatibility,
    validate_torch_ops_namespace,
)
from layerzero.backends.hf_kernels.lockfile import (
    KernelLockEntry,
    KernelLockfile,
)
from layerzero.backends.hf_kernels.adapter import HFKernelAdapter

__all__ = [
    # Version detection
    "detect_hf_kernels_version",
    "get_hf_kernels_info",
    "is_hf_kernels_available",
    # Loader
    "HFKernelLoader",
    "LoadedKernel",
    # Validator
    "ABIValidator",
    "validate_abi3_compatibility",
    "validate_manylinux_compatibility",
    "validate_torch_ops_namespace",
    # Lockfile
    "KernelLockEntry",
    "KernelLockfile",
    # Adapter
    "HFKernelAdapter",
]
