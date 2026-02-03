"""
LayerZero FlashInfer Version Detection

Functions for detecting FlashInfer installation and version.
"""
from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@functools.lru_cache(maxsize=1)
def is_flashinfer_available() -> bool:
    """Check if FlashInfer is installed.

    Returns:
        True if flashinfer can be imported.
    """
    try:
        import flashinfer  # noqa: F401
        return True
    except ImportError:
        return False


@functools.lru_cache(maxsize=1)
def detect_flashinfer_version() -> tuple[int, int, int] | None:
    """Detect installed FlashInfer version.

    Returns:
        Tuple of (major, minor, patch) or None if not installed.
    """
    if not is_flashinfer_available():
        return None

    try:
        import flashinfer
        version_str = getattr(flashinfer, "__version__", "0.0.0")

        # Parse version string (e.g., "0.5.3" or "0.5.3+cu124")
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))

        return (0, 0, 0)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def is_jit_cache_available() -> bool:
    """Check if FlashInfer JIT cache package is available.

    The flashinfer-jit-cache package provides persistent caching
    of JIT compiled kernels across processes.

    Returns:
        True if JIT cache is available.
    """
    if not is_flashinfer_available():
        return False

    try:
        # Check for JIT cache module
        import flashinfer
        # FlashInfer 0.5+ has built-in JIT caching via env var or explicit API
        return hasattr(flashinfer, "get_jit_cache_dir") or hasattr(flashinfer, "jit")
    except Exception:
        return False


def get_flashinfer_backend_info() -> dict:
    """Get detailed FlashInfer backend information.

    Returns:
        Dict with:
        - available: bool
        - version: str or None
        - backends: list of available internal backends
        - jit_cache: bool
        - features: dict of supported features
    """
    info = {
        "available": is_flashinfer_available(),
        "version": None,
        "backends": [],
        "jit_cache": is_jit_cache_available(),
        "features": {},
    }

    if not info["available"]:
        return info

    try:
        import flashinfer

        # Version
        version = detect_flashinfer_version()
        if version:
            info["version"] = f"{version[0]}.{version[1]}.{version[2]}"

        # Detect available backends/features
        backends = []

        # Check for prefill attention
        if hasattr(flashinfer, "single_prefill_with_kv_cache"):
            backends.append("prefill")
        if hasattr(flashinfer, "BatchPrefillWithPagedKVCacheWrapper"):
            backends.append("batch_prefill_paged")

        # Check for decode attention
        if hasattr(flashinfer, "single_decode_with_kv_cache"):
            backends.append("decode")
        if hasattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper"):
            backends.append("batch_decode_paged")

        # Check for cascade attention (new in 0.5+)
        if hasattr(flashinfer, "cascade"):
            backends.append("cascade")

        # Check for quantization support
        if hasattr(flashinfer, "quantization"):
            backends.append("quantization")
            info["features"]["int8"] = True
            info["features"]["fp8"] = True

        info["backends"] = backends

        # Feature detection
        info["features"]["gqa"] = True  # FlashInfer supports GQA
        info["features"]["mqa"] = True  # FlashInfer supports MQA
        info["features"]["alibi"] = hasattr(flashinfer, "alibi_slopes")
        info["features"]["sliding_window"] = True  # Supported in paged API

    except Exception:
        pass

    return info


def get_flashinfer_sm_support() -> tuple[tuple[int, int], tuple[int, int] | None]:
    """Get FlashInfer SM version support range.

    Returns:
        Tuple of (min_sm, max_sm) where max_sm may be None (no upper limit).
    """
    # FlashInfer supports SM 7.5 (Turing) through Blackwell
    return ((7, 5), None)
