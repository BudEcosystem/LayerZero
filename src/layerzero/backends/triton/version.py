"""
LayerZero Triton Version Detection

Functions for detecting Triton installation, version, and backend.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def is_triton_available() -> bool:
    """Check if Triton is installed and importable.

    Returns:
        True if Triton can be imported, False otherwise.
    """
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"Triton import check failed: {e}")
        return False


def detect_triton_version() -> tuple[int, int, int] | None:
    """Detect installed Triton version.

    Returns:
        Tuple of (major, minor, patch) version numbers, or None if not installed.
    """
    if not is_triton_available():
        return None

    try:
        import triton
        version_str = triton.__version__

        # Parse version string (e.g., "2.1.0", "3.0.0")
        # Handle versions with suffixes like "2.1.0+cu121"
        base_version = version_str.split("+")[0]

        # Handle versions with "post" or "dev" suffixes
        for suffix in (".post", ".dev", "rc", "a", "b"):
            if suffix in base_version:
                base_version = base_version.split(suffix)[0]

        parts = base_version.split(".")
        if len(parts) >= 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            return (int(parts[0]), int(parts[1]), 0)
        else:
            return (int(parts[0]), 0, 0)

    except (ImportError, AttributeError, ValueError) as e:
        logger.warning(f"Failed to parse Triton version: {e}")
        return None


def get_triton_backend() -> str | None:
    """Get the Triton backend type.

    Triton supports CUDA (NVIDIA) and HIP (AMD ROCm) backends.

    Returns:
        'cuda' for NVIDIA, 'hip' for AMD, or None if unavailable.
    """
    if not is_triton_available():
        return None

    try:
        import torch

        # Check if CUDA is available
        if torch.cuda.is_available():
            # Check if it's ROCm (AMD)
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                return "hip"
            return "cuda"

        return None

    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to detect Triton backend: {e}")
        return None


def get_triton_info() -> dict[str, Any]:
    """Get comprehensive Triton information.

    Returns:
        Dictionary containing:
        - available: Whether Triton is installed
        - version: Version tuple or None
        - version_string: Version string or None
        - backend: Backend type ('cuda', 'hip') or None
    """
    available = is_triton_available()
    version = detect_triton_version()

    info: dict[str, Any] = {
        "available": available,
        "version": version,
        "version_string": None,
        "backend": get_triton_backend(),
    }

    if not available:
        return info

    try:
        import triton
        info["version_string"] = triton.__version__
    except (ImportError, AttributeError):
        pass

    return info
