"""
LayerZero Liger Version Detection

Functions for detecting Liger and Triton installation and compatibility.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def is_liger_available() -> bool:
    """Check if Liger is installed and importable.

    Returns:
        True if Liger can be imported, False otherwise.
    """
    try:
        import liger_kernel  # noqa: F401
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"Liger import check failed: {e}")
        return False


def detect_liger_version() -> tuple[int, int, int] | None:
    """Detect installed Liger version.

    Returns:
        Tuple of (major, minor, patch) version numbers, or None if not installed.
    """
    if not is_liger_available():
        return None

    try:
        import liger_kernel
        version_str = liger_kernel.__version__

        # Parse version string (e.g., "0.1.0", "0.2.1")
        base_version = version_str.split("+")[0]

        parts = base_version.split(".")
        if len(parts) >= 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            return (int(parts[0]), int(parts[1]), 0)
        else:
            return (int(parts[0]), 0, 0)

    except (ImportError, AttributeError, ValueError) as e:
        logger.warning(f"Failed to parse Liger version: {e}")
        return None


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
        base_version = version_str.split("+")[0]

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


def check_triton_compatibility() -> bool:
    """Check if Triton version is compatible with Liger.

    Liger requires Triton >= 2.0.0.

    Returns:
        True if Triton is available and version >= 2.0.0, False otherwise.
    """
    version = detect_triton_version()
    if version is None:
        return False

    # Liger requires Triton 2.0.0+
    return version >= (2, 0, 0)


def get_available_kernels() -> list[str]:
    """Get list of available Liger kernels.

    Returns:
        List of kernel names available in current installation.
    """
    if not is_liger_available():
        return []

    kernels: list[str] = []

    # Check for each kernel
    try:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction  # noqa: F401
        kernels.append("rms_norm")
    except ImportError:
        pass

    try:
        from liger_kernel.ops.layer_norm import LigerLayerNormFunction  # noqa: F401
        kernels.append("layer_norm")
    except ImportError:
        pass

    try:
        from liger_kernel.ops.rope import LigerRopeFunction  # noqa: F401
        kernels.append("rope")
    except ImportError:
        pass

    try:
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction  # noqa: F401
        kernels.append("swiglu")
    except ImportError:
        pass

    try:
        from liger_kernel.ops.geglu import LigerGELUMulFunction  # noqa: F401
        kernels.append("geglu")
    except ImportError:
        pass

    try:
        from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction  # noqa: F401
        kernels.append("cross_entropy")
    except ImportError:
        pass

    try:
        from liger_kernel.ops.fused_linear_cross_entropy import (  # noqa: F401
            LigerFusedLinearCrossEntropyFunction,
        )
        kernels.append("fused_linear_cross_entropy")
    except ImportError:
        pass

    return kernels


def get_liger_info() -> dict[str, Any]:
    """Get comprehensive Liger information.

    Returns:
        Dictionary containing:
        - available: Whether Liger is installed
        - version: Version tuple or None
        - version_string: Version string or None
        - triton_available: Whether Triton is installed
        - triton_version: Triton version tuple or None
        - triton_compatible: Whether Triton version is compatible
        - available_kernels: List of available kernel names
    """
    available = is_liger_available()
    version = detect_liger_version()

    info: dict[str, Any] = {
        "available": available,
        "version": version,
        "version_string": None,
        "triton_available": is_triton_available(),
        "triton_version": detect_triton_version(),
        "triton_compatible": check_triton_compatibility(),
        "available_kernels": [],
    }

    if not available:
        return info

    try:
        import liger_kernel
        info["version_string"] = liger_kernel.__version__
    except (ImportError, AttributeError):
        pass

    info["available_kernels"] = get_available_kernels()

    return info
