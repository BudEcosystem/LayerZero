"""
SDPA Backend Integration

Integration with torch.nn.attention.sdpa_kernel context
for backend selection and compatibility checking.
"""
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Generator


def get_active_sdpa_backends() -> frozenset[Any]:
    """Get currently enabled SDPA backends.

    Returns the set of backends that are currently enabled
    in the SDPA context.

    Returns:
        Frozenset of enabled SDPBackend values.
    """
    try:
        from torch.nn.attention import SDPBackend

        # Check which backends are available
        available = set()

        # Check Flash Attention
        if check_flash_attention_available():
            available.add(SDPBackend.FLASH_ATTENTION)

        # Check Efficient Attention (mem efficient)
        if check_efficient_attention_available():
            available.add(SDPBackend.EFFICIENT_ATTENTION)

        # Math backend is always available
        available.add(SDPBackend.MATH)

        return frozenset(available)

    except ImportError:
        # SDPBackend not available in older PyTorch
        return frozenset()


def check_flash_attention_available() -> bool:
    """Check if Flash Attention can be used.

    Checks both hardware capability and library availability.

    Returns:
        True if Flash Attention is usable.
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Check CUDA capability
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)

        # Flash Attention requires SM80+ (Ampere or newer)
        if capability[0] < 8:
            return False

        # Check if the backend is available in PyTorch
        try:
            from torch.nn.attention import SDPBackend

            # Try to check via internal API
            if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
                return torch.backends.cuda.flash_sdp_enabled()

            return True
        except ImportError:
            return False

    except Exception:
        return False


def check_efficient_attention_available() -> bool:
    """Check if efficient attention (memory efficient) can be used.

    Returns:
        True if efficient attention is usable.
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Check if the backend is available
        if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled"):
            return torch.backends.cuda.mem_efficient_sdp_enabled()

        # Assume available if on CUDA
        return True

    except Exception:
        return False


def check_cudnn_attention_available() -> bool:
    """Check if cuDNN attention can be used.

    Returns:
        True if cuDNN attention is usable.
    """
    if not torch.cuda.is_available():
        return False

    try:
        if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
            return torch.backends.cuda.cudnn_sdp_enabled()
        return False
    except Exception:
        return False


@contextlib.contextmanager
def layerzero_sdpa_context(
    enable_flash: bool = True,
    enable_efficient: bool = True,
    enable_math: bool = True,
    enable_cudnn: bool = True,
) -> Generator[None, None, None]:
    """Context manager for LayerZero SDPA backend selection.

    Controls which SDPA backends are enabled within the context.

    Args:
        enable_flash: Enable Flash Attention backend
        enable_efficient: Enable Memory Efficient backend
        enable_math: Enable Math backend
        enable_cudnn: Enable cuDNN backend

    Example:
        ```python
        with layerzero_sdpa_context(enable_flash=True, enable_math=False):
            # Only Flash and Efficient attention available
            output = lz.attention(q, k, v)
        ```
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend

        backends = []
        if enable_flash:
            backends.append(SDPBackend.FLASH_ATTENTION)
        if enable_efficient:
            backends.append(SDPBackend.EFFICIENT_ATTENTION)
        if enable_math:
            backends.append(SDPBackend.MATH)
        if enable_cudnn and hasattr(SDPBackend, "CUDNN_ATTENTION"):
            backends.append(SDPBackend.CUDNN_ATTENTION)

        with sdpa_kernel(backends):
            yield

    except ImportError:
        # sdpa_kernel not available, just yield
        yield


def get_sdpa_backend_for_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> str:
    """Determine the best SDPA backend for given inputs.

    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether causal masking is used

    Returns:
        Backend name: "flash", "efficient", "math", or "unknown"
    """
    if not torch.cuda.is_available():
        return "math"

    device = query.device
    if device.type != "cuda":
        return "math"

    dtype = query.dtype

    # Flash Attention checks
    if dtype in (torch.float16, torch.bfloat16):
        if check_flash_attention_available():
            # Flash attention has specific requirements
            batch_size = query.size(0)
            num_heads = query.size(1)
            seq_len = query.size(2)
            head_dim = query.size(3)

            # Flash attention works best with certain configurations
            if head_dim <= 128 and head_dim % 8 == 0:
                return "flash"

    # Memory efficient attention
    if check_efficient_attention_available():
        return "efficient"

    return "math"
