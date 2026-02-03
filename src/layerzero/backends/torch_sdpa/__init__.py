"""
LayerZero Torch SDPA Backend

Adapter for torch.nn.functional.scaled_dot_product_attention.
"""
from layerzero.backends.torch_sdpa.adapter import TorchSDPAAdapter
from layerzero.backends.torch_sdpa.constraints import (
    SDPABackendType,
    check_flash_constraints,
    check_efficient_constraints,
    check_cudnn_constraints,
    get_available_backends,
    can_use_backend,
)
from layerzero.backends.torch_sdpa.kernel import (
    sdpa_forward,
    SDPAConfig,
)

__all__ = [
    "TorchSDPAAdapter",
    "SDPABackendType",
    "SDPAConfig",
    "check_flash_constraints",
    "check_efficient_constraints",
    "check_cudnn_constraints",
    "get_available_backends",
    "can_use_backend",
    "sdpa_forward",
]
