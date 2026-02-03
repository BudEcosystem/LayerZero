"""
Device fixtures for LayerZero tests.

Provides device-related test utilities and skip conditions.
"""
from __future__ import annotations

import pytest
import torch

# Extended timeout for stress tests (5 minutes)
STRESS_TEST_TIMEOUT = 300


def gpu_required() -> None:
    """Skip test if GPU is not available.

    Raises:
        pytest.skip.Exception: If CUDA is not available.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for this test")


def multigpu_required() -> None:
    """Skip test if less than 2 GPUs are available.

    Raises:
        pytest.skip.Exception: If less than 2 GPUs are available.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for multigpu test")
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required for this test")


def get_cuda_device() -> torch.device:
    """Get CUDA device or skip if unavailable.

    Returns:
        torch.device: CUDA device.

    Raises:
        pytest.skip.Exception: If CUDA is not available.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


def reset_cuda_state() -> None:
    """Reset CUDA state between tests.

    Clears CUDA cache and synchronizes devices.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_device() -> torch.device:
    """Get the best available device.

    Prefers CUDA if available, otherwise CPU.

    Returns:
        torch.device: Best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_all_devices() -> list[torch.device]:
    """Get all available devices.

    Returns:
        list[torch.device]: List of available devices.
    """
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device(f"cuda:{i}"))
    return devices


def is_ampere_or_newer() -> bool:
    """Check if GPU is Ampere architecture or newer.

    Returns:
        bool: True if GPU is Ampere (SM 8.0+) or newer.
    """
    if not torch.cuda.is_available():
        return False

    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def is_hopper_or_newer() -> bool:
    """Check if GPU is Hopper architecture or newer.

    Returns:
        bool: True if GPU is Hopper (SM 9.0+) or newer.
    """
    if not torch.cuda.is_available():
        return False

    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def get_gpu_memory_gb() -> float | None:
    """Get total GPU memory in GB.

    Returns:
        float | None: GPU memory in GB, or None if no GPU.
    """
    if not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 ** 3)


def skip_if_not_ampere() -> None:
    """Skip test if GPU is not Ampere or newer.

    Raises:
        pytest.skip.Exception: If GPU is older than Ampere.
    """
    if not is_ampere_or_newer():
        pytest.skip("Ampere or newer GPU required")


def skip_if_not_hopper() -> None:
    """Skip test if GPU is not Hopper or newer.

    Raises:
        pytest.skip.Exception: If GPU is older than Hopper.
    """
    if not is_hopper_or_newer():
        pytest.skip("Hopper or newer GPU required")


def skip_if_low_memory(min_gb: float = 8.0) -> None:
    """Skip test if GPU has less than specified memory.

    Args:
        min_gb: Minimum required GPU memory in GB.

    Raises:
        pytest.skip.Exception: If GPU memory is insufficient.
    """
    memory = get_gpu_memory_gb()
    if memory is None or memory < min_gb:
        pytest.skip(f"At least {min_gb}GB GPU memory required")
