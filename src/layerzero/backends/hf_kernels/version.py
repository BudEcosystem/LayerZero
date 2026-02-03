"""
LayerZero HuggingFace Kernel Hub Version Detection

Functions for detecting HuggingFace Kernels library availability and version.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# HuggingFace Hub base URL for kernels
HF_KERNELS_HUB_URL = "https://huggingface.co/kernels"


def is_hf_kernels_available() -> bool:
    """Check if HuggingFace kernels library is available.

    The HF kernels library provides dynamic kernel loading from the Hub.

    Returns:
        True if HF kernels library is available, False otherwise.
    """
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"HF kernels availability check failed: {e}")
        return False


def detect_hf_kernels_version() -> tuple[int, int, int] | None:
    """Detect HuggingFace Hub library version.

    Returns:
        Tuple of (major, minor, patch) or None if not available.
    """
    if not is_hf_kernels_available():
        return None

    try:
        import huggingface_hub
        version_str = huggingface_hub.__version__

        # Parse version string (e.g., "0.20.0", "0.21.0")
        base_version = version_str.split("+")[0]

        # Handle versions with suffixes
        for suffix in (".post", ".dev", "rc", "a", "b"):
            if suffix in base_version:
                base_version = base_version.split(suffix)[0]

        parts = base_version.split(".")
        if len(parts) >= 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            return (int(parts[0]), int(parts[1]), 0)
        return (int(parts[0]), 0, 0)

    except (ImportError, ValueError, AttributeError) as e:
        logger.warning(f"Failed to detect HF kernels version: {e}")
        return None


def get_hf_kernels_info() -> dict[str, Any]:
    """Get comprehensive HF kernels information.

    Returns:
        Dictionary containing:
        - available: Whether HF hub is available
        - version: Version tuple or None
        - version_string: Version string or None
        - hub_url: Hub URL for kernels
    """
    available = is_hf_kernels_available()
    version = detect_hf_kernels_version()

    info: dict[str, Any] = {
        "available": available,
        "version": version,
        "version_string": None,
        "hub_url": HF_KERNELS_HUB_URL,
    }

    if not available:
        return info

    try:
        import huggingface_hub
        info["version_string"] = huggingface_hub.__version__
    except (ImportError, AttributeError):
        pass

    return info
