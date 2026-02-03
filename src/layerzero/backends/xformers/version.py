"""
LayerZero xFormers Version Detection

Functions for detecting xFormers installation and available backends.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def is_xformers_available() -> bool:
    """Check if xFormers is installed and importable.

    Returns:
        True if xFormers can be imported, False otherwise.
    """
    try:
        import xformers  # noqa: F401
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"xFormers import check failed: {e}")
        return False


def detect_xformers_version() -> tuple[int, int, int] | None:
    """Detect installed xFormers version.

    Returns:
        Tuple of (major, minor, patch) version numbers, or None if not installed.
    """
    if not is_xformers_available():
        return None

    try:
        import xformers
        version_str = xformers.__version__

        # Parse version string (e.g., "0.0.23", "0.0.23+cu118.d20231210")
        # Remove any build metadata
        base_version = version_str.split("+")[0]

        parts = base_version.split(".")
        if len(parts) >= 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            return (int(parts[0]), int(parts[1]), 0)
        else:
            return (int(parts[0]), 0, 0)

    except (ImportError, AttributeError, ValueError) as e:
        logger.warning(f"Failed to parse xFormers version: {e}")
        return None


def get_available_backends() -> list[str]:
    """Get list of available xFormers attention backends.

    Returns:
        List of backend names available in current installation.
    """
    if not is_xformers_available():
        return []

    backends: list[str] = []

    try:
        # Check for memory efficient attention
        from xformers.ops import memory_efficient_attention  # noqa: F401
        backends.append("memory_efficient")
    except ImportError:
        pass

    try:
        # Check for fused attention
        from xformers.ops import fmha  # noqa: F401
        backends.append("fmha")
    except ImportError:
        pass

    # Try to get info from xformers.info module if available
    try:
        from xformers import info
        if hasattr(info, "get_operators"):
            for op in info.get_operators():
                if hasattr(op, "NAME") and op.NAME not in backends:
                    backends.append(op.NAME)
    except (ImportError, AttributeError):
        pass

    return backends


def get_xformers_backend_info() -> dict[str, Any]:
    """Get comprehensive xFormers backend information.

    Returns:
        Dictionary containing:
        - available: Whether xFormers is installed
        - version: Version tuple or None
        - version_string: Version string or None
        - backends: List of available backend names
        - cuda_available: Whether CUDA backend is available
    """
    available = is_xformers_available()
    version = detect_xformers_version()

    info: dict[str, Any] = {
        "available": available,
        "version": version,
        "version_string": None,
        "backends": [],
        "cuda_available": False,
    }

    if not available:
        return info

    try:
        import xformers
        info["version_string"] = xformers.__version__
    except (ImportError, AttributeError):
        pass

    info["backends"] = get_available_backends()

    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            # Verify xFormers can actually use CUDA
            try:
                from xformers.ops import memory_efficient_attention
                # Create minimal test tensors
                device = torch.device("cuda")
                dtype = torch.float16
                q = torch.zeros(1, 1, 1, 64, device=device, dtype=dtype)
                k = torch.zeros(1, 1, 1, 64, device=device, dtype=dtype)
                v = torch.zeros(1, 1, 1, 64, device=device, dtype=dtype)
                # This will fail if CUDA kernels aren't available
                _ = memory_efficient_attention(q, k, v)
                info["cuda_available"] = True
            except Exception:
                info["cuda_available"] = False
    except ImportError:
        pass

    return info
