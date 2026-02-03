"""
LayerZero PyTorch Integration

Provides PyTorch integration via torch.library for:
- torch.compile compatibility (no graph breaks)
- torch.export support via meta kernels
- SDPA backend integration
"""
from layerzero.pytorch import ops
from layerzero.pytorch.meta_kernels import (
    attention_meta,
    rms_norm_meta,
    layer_norm_meta,
)
from layerzero.pytorch.sdpa_integration import (
    get_active_sdpa_backends,
    check_flash_attention_available,
    check_efficient_attention_available,
    layerzero_sdpa_context,
)
from layerzero.pytorch.compile_compat import (
    ensure_no_graph_breaks,
    register_for_compile,
)

__all__ = [
    "ops",
    "attention_meta",
    "rms_norm_meta",
    "layer_norm_meta",
    "get_active_sdpa_backends",
    "check_flash_attention_available",
    "check_efficient_attention_available",
    "layerzero_sdpa_context",
    "ensure_no_graph_breaks",
    "register_for_compile",
]
