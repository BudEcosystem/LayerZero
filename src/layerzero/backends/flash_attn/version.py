"""
LayerZero FlashAttention Version Detection

Functions for detecting FlashAttention version and selecting appropriate variant.
"""
from __future__ import annotations

from enum import Enum, unique


@unique
class FAVariant(str, Enum):
    """FlashAttention variant types.

    FA2: For Ampere/Ada (SM 8.0-8.9)
    FA3: For Hopper (SM 9.0)
    FA4: For Blackwell and beyond (SM 10.0+)
    """

    FA2 = "fa2"
    FA3 = "fa3"
    FA4 = "fa4"


def detect_flash_attn_version() -> tuple[int, int, int] | None:
    """Detect installed FlashAttention version.

    Returns:
        Version tuple (major, minor, patch) or None if not installed.
    """
    try:
        import flash_attn
        version_str = flash_attn.__version__

        # Handle version strings like "2.5.6" or "2.5.6.post1"
        parts = version_str.split(".")[:3]

        # Parse integers, handle suffixes like "6post1"
        parsed = []
        for part in parts:
            # Extract leading digits
            digits = ""
            for char in part:
                if char.isdigit():
                    digits += char
                else:
                    break
            parsed.append(int(digits) if digits else 0)

        # Pad to 3 elements
        while len(parsed) < 3:
            parsed.append(0)

        return (parsed[0], parsed[1], parsed[2])

    except ImportError:
        return None
    except (ValueError, IndexError, AttributeError):
        return None


def is_flash_attn_available() -> bool:
    """Check if FlashAttention is installed and available.

    Returns:
        True if flash_attn is installed, False otherwise.
    """
    return detect_flash_attn_version() is not None


def select_fa_variant(sm_version: tuple[int, int]) -> FAVariant | None:
    """Select appropriate FA variant for GPU SM version.

    Args:
        sm_version: GPU SM version as (major, minor).

    Returns:
        FAVariant to use, or None if SM version not supported.
    """
    major, minor = sm_version

    # SM < 8.0: FlashAttention not supported
    if major < 8:
        return None

    # SM 8.x (Ampere/Ada): Use FA2
    if major == 8:
        return FAVariant.FA2

    # SM 9.0 (Hopper): Use FA3
    if major == 9 and minor < 10:
        return FAVariant.FA3

    # SM 10.0+ (Blackwell and beyond): Use FA4
    return FAVariant.FA4
