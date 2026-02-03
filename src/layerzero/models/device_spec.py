"""
LayerZero Device Specification

Dataclass representing hardware device capabilities.
Supports auto-detection via DeviceSpec.detect().
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

from layerzero.device import (
    GPUGeneration,
    get_tensor_core_gen,
    sm_to_generation,
    supports_bf16,
    supports_fp4,
    supports_fp8,
    supports_tma,
    get_max_shared_memory_kb,
)
from layerzero.enums import Platform


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """Hardware device specification.

    Captures device capabilities for kernel selection.
    Immutable (frozen) for hashability and thread safety.

    Attributes:
        platform: Hardware platform (CUDA, ROCm, CPU, etc.)
        device_index: Device index (0 for CPU, GPU index otherwise)
        device_name: Human-readable device name
        sm_version: NVIDIA SM version as (major, minor), None for non-CUDA
        gpu_generation: GPU architecture generation
        tensor_core_gen: Tensor core generation (0-5)
        total_memory_bytes: Total device memory in bytes
        available_memory_bytes: Available device memory in bytes
        supports_bf16: Whether device supports bfloat16
        supports_fp8: Whether device supports FP8 (E4M3/E5M2)
        supports_fp4: Whether device supports FP4/NVFP4
        supports_tma: Whether device supports Tensor Memory Access
        max_shared_memory_kb: Maximum shared memory per SM in KB
        cuda_version: CUDA version string (e.g., "12.4")
        driver_version: Driver version string (e.g., "550.54")
    """

    platform: Platform
    device_index: int
    device_name: str
    sm_version: tuple[int, int] | None
    gpu_generation: GPUGeneration
    tensor_core_gen: int
    total_memory_bytes: int
    available_memory_bytes: int
    supports_bf16: bool
    supports_fp8: bool
    supports_fp4: bool
    supports_tma: bool
    max_shared_memory_kb: int
    cuda_version: str | None
    driver_version: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON compatibility.

        Returns:
            Dict with all device spec fields.
        """
        return {
            "platform": self.platform.value,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "sm_version": list(self.sm_version) if self.sm_version else None,
            "gpu_generation": self.gpu_generation.value,
            "tensor_core_gen": self.tensor_core_gen,
            "total_memory_bytes": self.total_memory_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "supports_bf16": self.supports_bf16,
            "supports_fp8": self.supports_fp8,
            "supports_fp4": self.supports_fp4,
            "supports_tma": self.supports_tma,
            "max_shared_memory_kb": self.max_shared_memory_kb,
            "cuda_version": self.cuda_version,
            "driver_version": self.driver_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DeviceSpec:
        """Deserialize from dictionary.

        Args:
            d: Dict with device spec fields.

        Returns:
            New DeviceSpec instance.
        """
        return cls(
            platform=Platform(d["platform"]),
            device_index=d["device_index"],
            device_name=d["device_name"],
            sm_version=tuple(d["sm_version"]) if d["sm_version"] else None,
            gpu_generation=GPUGeneration(d["gpu_generation"]),
            tensor_core_gen=d["tensor_core_gen"],
            total_memory_bytes=d["total_memory_bytes"],
            available_memory_bytes=d["available_memory_bytes"],
            supports_bf16=d["supports_bf16"],
            supports_fp8=d["supports_fp8"],
            supports_fp4=d["supports_fp4"],
            supports_tma=d["supports_tma"],
            max_shared_memory_kb=d["max_shared_memory_kb"],
            cuda_version=d.get("cuda_version"),
            driver_version=d.get("driver_version"),
        )

    def cache_key(self) -> str:
        """Generate cache key for this device.

        Returns:
            String cache key based on device properties.
        """
        key_parts = [
            self.platform.value,
            str(self.device_index),
            str(self.sm_version) if self.sm_version else "cpu",
            self.gpu_generation.value,
        ]
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()[:16]

    @classmethod
    def detect(cls, device: str | None = None) -> DeviceSpec:
        """Auto-detect device capabilities.

        Args:
            device: Device string ("cuda:0", "cpu", etc.) or None for default.

        Returns:
            DeviceSpec for the specified or default device.
        """
        # Import torch lazily to avoid import errors when not installed
        try:
            import torch
        except ImportError:
            return cls.cpu()

        # Parse device string
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"

        if isinstance(device, str):
            device_obj = torch.device(device)
        else:
            device_obj = device

        if device_obj.type == "cpu":
            return cls.cpu()

        if device_obj.type == "cuda":
            return cls._detect_cuda(device_obj.index or 0)

        # Fallback for other device types
        return cls.cpu()

    @classmethod
    def _detect_cuda(cls, device_index: int) -> DeviceSpec:
        """Detect CUDA device capabilities.

        Args:
            device_index: CUDA device index.

        Returns:
            DeviceSpec for the CUDA device.
        """
        import torch

        props = torch.cuda.get_device_properties(device_index)
        sm_version = (props.major, props.minor)
        gpu_gen = sm_to_generation(props.major, props.minor)
        tc_gen = get_tensor_core_gen(gpu_gen)

        # Get memory info
        total_mem = props.total_memory
        try:
            torch.cuda.set_device(device_index)
            free_mem, _ = torch.cuda.mem_get_info(device_index)
        except Exception:
            free_mem = total_mem

        # Get CUDA version
        cuda_version = None
        try:
            cuda_version = torch.version.cuda
        except Exception:
            pass

        # Get driver version (from environment or nvidia-smi)
        driver_version = os.environ.get("NVIDIA_DRIVER_CAPABILITIES", None)

        return cls(
            platform=Platform.CUDA,
            device_index=device_index,
            device_name=props.name,
            sm_version=sm_version,
            gpu_generation=gpu_gen,
            tensor_core_gen=tc_gen,
            total_memory_bytes=total_mem,
            available_memory_bytes=free_mem,
            supports_bf16=supports_bf16(gpu_gen),
            supports_fp8=supports_fp8(gpu_gen),
            supports_fp4=supports_fp4(gpu_gen),
            supports_tma=supports_tma(gpu_gen),
            max_shared_memory_kb=get_max_shared_memory_kb(gpu_gen),
            cuda_version=cuda_version,
            driver_version=driver_version,
        )

    @classmethod
    def cpu(cls) -> DeviceSpec:
        """Create CPU device spec.

        Returns:
            DeviceSpec for CPU.
        """
        import platform
        import os

        # Try to get CPU info
        cpu_name = platform.processor() or "Unknown CPU"

        # Estimate available memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_mem = mem.total
            available_mem = mem.available
        except ImportError:
            # Fallback: assume 16GB
            total_mem = 16 * 1024**3
            available_mem = 8 * 1024**3

        return cls(
            platform=Platform.CPU,
            device_index=0,
            device_name=cpu_name,
            sm_version=None,
            gpu_generation=GPUGeneration.UNKNOWN,
            tensor_core_gen=0,
            total_memory_bytes=total_mem,
            available_memory_bytes=available_mem,
            supports_bf16=True,  # CPU supports bf16 via emulation
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=0,
            cuda_version=None,
            driver_version=None,
        )
