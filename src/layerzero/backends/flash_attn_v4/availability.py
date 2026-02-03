"""FlashAttention 4 availability detection.

This module handles detection of FA4 installation,
version checking, and CUDA version requirements.
"""
from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)

# FA4 version requirements
FA4_MIN_VERSION: Final[tuple[int, int, int]] = (3, 0, 0)
FA4_MIN_CUDA_VERSION: Final[str] = "12.9"


def _get_flash_attn_version() -> tuple[int, int, int] | None:
    """Get installed flash_attn version.

    Returns:
        Version tuple (major, minor, patch) or None if not installed.
    """
    try:
        import flash_attn
        version_str = flash_attn.__version__

        # Handle version strings like "3.0.0" or "3.0.0.post1"
        parts = version_str.split(".")[:3]

        # Parse integers
        parsed = []
        for part in parts:
            digits = ""
            for char in part:
                if char.isdigit():
                    digits += char
                else:
                    break
            parsed.append(int(digits) if digits else 0)

        while len(parsed) < 3:
            parsed.append(0)

        return (parsed[0], parsed[1], parsed[2])
    except ImportError:
        return None
    except (ValueError, IndexError, AttributeError):
        return None


def detect_fa4_version() -> tuple[int, int, int] | None:
    """Detect if FA4 (version >= 3.0) is available.

    Returns:
        Version tuple if FA4 is available, None otherwise.
    """
    version = _get_flash_attn_version()

    if version is None:
        logger.debug("flash_attn not installed")
        return None

    if version < FA4_MIN_VERSION:
        logger.debug(
            "flash_attn version %s < FA4 minimum %s",
            version,
            FA4_MIN_VERSION,
        )
        return None

    logger.debug("FA4 version detected: %s", version)
    return version


def is_fa4_available() -> bool:
    """Check if FA4 is installed and available.

    Returns:
        True if FA4 (flash_attn >= 3.0) is available.
    """
    return detect_fa4_version() is not None


def check_cuda_version_for_fa4(cuda_version: str | None) -> bool:
    """Check if CUDA version meets FA4 requirements.

    FA4 requires CUDA 12.9+ for tcgen05 support.

    Args:
        cuda_version: CUDA version string (e.g., "12.9.0").

    Returns:
        True if CUDA version is sufficient.
    """
    if cuda_version is None:
        logger.debug("No CUDA version provided")
        return False

    try:
        # Parse version string
        parts = cuda_version.split(".")[:2]
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0

        # Parse minimum version
        min_parts = FA4_MIN_CUDA_VERSION.split(".")
        min_major = int(min_parts[0])
        min_minor = int(min_parts[1]) if len(min_parts) > 1 else 0

        # Compare versions
        if major > min_major:
            return True
        if major == min_major and minor >= min_minor:
            return True

        logger.debug(
            "CUDA %s does not meet FA4 requirement of %s+",
            cuda_version,
            FA4_MIN_CUDA_VERSION,
        )
        return False

    except (ValueError, IndexError):
        logger.warning("Failed to parse CUDA version: %s", cuda_version)
        return False
