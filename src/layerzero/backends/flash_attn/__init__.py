"""
LayerZero FlashAttention Backend

Adapter for FlashAttention (FA2/FA3/FA4) library.
"""
from layerzero.backends.flash_attn.adapter import FlashAttnAdapter
from layerzero.backends.flash_attn.constraints import (
    check_fa_constraints,
    check_fa2_constraints,
    check_fa3_constraints,
    check_fa4_constraints,
)
from layerzero.backends.flash_attn.layout import (
    bhsd_to_bshd,
    bshd_to_bhsd,
    convert_layout,
)
from layerzero.backends.flash_attn.version import (
    detect_flash_attn_version,
    is_flash_attn_available,
    select_fa_variant,
    FAVariant,
)

__all__ = [
    "FlashAttnAdapter",
    "FAVariant",
    "bhsd_to_bshd",
    "bshd_to_bhsd",
    "check_fa_constraints",
    "check_fa2_constraints",
    "check_fa3_constraints",
    "check_fa4_constraints",
    "convert_layout",
    "detect_flash_attn_version",
    "is_flash_attn_available",
    "select_fa_variant",
]
