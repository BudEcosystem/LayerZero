"""
LayerZero CPU Detection

Runtime detection of CPU vendor and ISA features.
"""
from __future__ import annotations

import logging
import os
import platform
import struct
from enum import Enum, unique
from typing import Any

logger = logging.getLogger(__name__)


@unique
class CPUVendor(str, Enum):
    """CPU vendor identification.

    Members:
        INTEL: Intel Corporation
        AMD: Advanced Micro Devices
        ARM: ARM Holdings (includes Apple Silicon)
        UNKNOWN: Unknown or unsupported vendor
    """

    INTEL = "intel"
    AMD = "amd"
    ARM = "arm"
    UNKNOWN = "unknown"


@unique
class ISAFeature(str, Enum):
    """CPU ISA (Instruction Set Architecture) features.

    Members for x86-64:
        SSE4_2: Streaming SIMD Extensions 4.2
        AVX2: Advanced Vector Extensions 2
        AVX512F: AVX-512 Foundation
        AVX512_BF16: AVX-512 BFloat16
        AVX512_VNNI: AVX-512 Vector Neural Network Instructions
        AMX: Advanced Matrix Extensions

    Members for ARM:
        NEON: ARM Advanced SIMD
        SVE: ARM Scalable Vector Extension
    """

    SSE4_2 = "sse4_2"
    AVX2 = "avx2"
    AVX512F = "avx512f"
    AVX512_BF16 = "avx512_bf16"
    AVX512_VNNI = "avx512_vnni"
    AMX = "amx"
    NEON = "neon"
    SVE = "sve"


def _detect_vendor_from_cpuinfo() -> CPUVendor:
    """Detect CPU vendor from /proc/cpuinfo (Linux)."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            content = f.read().lower()

        if "genuineintel" in content:
            return CPUVendor.INTEL
        elif "authenticamd" in content:
            return CPUVendor.AMD
        elif "arm" in content or "aarch64" in content:
            return CPUVendor.ARM

    except (OSError, IOError):
        pass

    return CPUVendor.UNKNOWN


def _detect_vendor_from_platform() -> CPUVendor:
    """Detect CPU vendor from platform module."""
    machine = platform.machine().lower()
    processor = platform.processor().lower()

    # ARM detection
    if machine in ("arm64", "aarch64", "armv8", "armv7l"):
        return CPUVendor.ARM

    # x86 - need to check processor string
    if "intel" in processor:
        return CPUVendor.INTEL
    elif "amd" in processor:
        return CPUVendor.AMD

    return CPUVendor.UNKNOWN


def detect_cpu_vendor() -> CPUVendor:
    """Detect the CPU vendor.

    Tries multiple methods to detect the CPU vendor:
    1. Read /proc/cpuinfo (Linux)
    2. Check platform.processor() and platform.machine()

    Returns:
        CPUVendor enum value.
    """
    # Try /proc/cpuinfo first (most reliable on Linux)
    vendor = _detect_vendor_from_cpuinfo()
    if vendor != CPUVendor.UNKNOWN:
        return vendor

    # Fallback to platform module
    vendor = _detect_vendor_from_platform()
    if vendor != CPUVendor.UNKNOWN:
        return vendor

    # Check if we're on macOS with Apple Silicon
    if platform.system() == "Darwin":
        machine = platform.machine()
        if machine == "arm64":
            return CPUVendor.ARM

    return CPUVendor.UNKNOWN


def _detect_features_from_cpuinfo() -> set[ISAFeature]:
    """Detect ISA features from /proc/cpuinfo (Linux)."""
    features: set[ISAFeature] = set()

    try:
        with open("/proc/cpuinfo", "r") as f:
            content = f.read().lower()

        # Find the flags line
        for line in content.split("\n"):
            if line.startswith("flags") or line.startswith("features"):
                flags = line.split(":", 1)[1] if ":" in line else ""

                # x86 features
                if "sse4_2" in flags:
                    features.add(ISAFeature.SSE4_2)
                if "avx2" in flags:
                    features.add(ISAFeature.AVX2)
                if "avx512f" in flags:
                    features.add(ISAFeature.AVX512F)
                if "avx512_bf16" in flags:
                    features.add(ISAFeature.AVX512_BF16)
                if "avx512_vnni" in flags or "avx512vnni" in flags:
                    features.add(ISAFeature.AVX512_VNNI)
                if "amx" in flags:
                    features.add(ISAFeature.AMX)

                # ARM features
                if "neon" in flags or "asimd" in flags:
                    features.add(ISAFeature.NEON)
                if "sve" in flags:
                    features.add(ISAFeature.SVE)

                break

    except (OSError, IOError):
        pass

    return features


def _detect_features_from_torch() -> set[ISAFeature]:
    """Detect ISA features from PyTorch."""
    features: set[ISAFeature] = set()

    try:
        import torch

        # PyTorch exposes some CPU capability info
        if hasattr(torch.backends, "cpu") and hasattr(torch.backends.cpu, "get_cpu_capability"):
            capability = torch.backends.cpu.get_cpu_capability()
            if "avx2" in capability.lower():
                features.add(ISAFeature.AVX2)
            if "avx512" in capability.lower():
                features.add(ISAFeature.AVX512F)

    except (ImportError, AttributeError):
        pass

    return features


def detect_isa_features() -> set[ISAFeature]:
    """Detect supported ISA features.

    Combines information from /proc/cpuinfo and PyTorch.

    Returns:
        Set of supported ISAFeature values.
    """
    features: set[ISAFeature] = set()

    # Try /proc/cpuinfo first
    features.update(_detect_features_from_cpuinfo())

    # Add PyTorch detection
    features.update(_detect_features_from_torch())

    # ARM detection fallback
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        # ARM64 always has NEON
        features.add(ISAFeature.NEON)

    return features


def get_cpu_info() -> dict[str, Any]:
    """Get comprehensive CPU information.

    Returns:
        Dictionary containing:
        - vendor: CPUVendor enum
        - features: Set of ISAFeature values
        - core_count: Number of CPU cores
        - model: CPU model string if available
    """
    vendor = detect_cpu_vendor()
    features = detect_isa_features()

    # Get core count
    core_count = os.cpu_count() or 1

    # Get model string if available
    model = None
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except (OSError, IOError):
        pass

    return {
        "vendor": vendor,
        "features": features,
        "core_count": core_count,
        "model": model,
    }


def get_optimal_cpu_backend() -> str:
    """Determine the optimal CPU backend for this system.

    Selection logic:
    1. If Intel CPU with IPEX available: return "ipex"
    2. If Intel CPU with oneDNN available: return "onednn"
    3. If AMD EPYC with ZenDNN available: return "zendnn"
    4. Otherwise: return "pytorch"

    Returns:
        Name of optimal CPU backend.
    """
    vendor = detect_cpu_vendor()

    # Intel CPU: prefer IPEX, then oneDNN
    if vendor == CPUVendor.INTEL:
        try:
            import intel_extension_for_pytorch  # noqa: F401
            return "ipex"
        except ImportError:
            pass

        # oneDNN is usually bundled with PyTorch on Intel
        try:
            import torch
            if hasattr(torch.backends, "mkldnn") and torch.backends.mkldnn.is_available():
                return "onednn"
        except (ImportError, AttributeError):
            pass

    # AMD CPU: prefer ZenDNN
    if vendor == CPUVendor.AMD:
        try:
            # Check for ZenDNN-optimized PyTorch
            # ZenDNN is typically integrated at build time
            import torch
            # Check for ZENDNN env variable or specific build
            if os.environ.get("ZENDNN_ENABLE", "").lower() in ("1", "true"):
                return "zendnn"
        except ImportError:
            pass

    # Fallback to PyTorch
    return "pytorch"
