"""LayerZero Public API.

This module provides the user-facing API for LayerZero:
- Operation dispatch (attention, rms_norm, layer_norm, etc.)
- Configuration management (configure, lock, unlock)
- Inspection and debugging (select, explain, which, list_kernels)
- System utilities (doctor, readiness_check, compile, dry_run)
"""
from __future__ import annotations

from layerzero.api.operations import (
    attention,
    rms_norm,
    layer_norm,
    sample_topk,
    sample_topp,
    tokenize,
    detokenize,
)
from layerzero.api.config import (
    configure,
    get_config,
    load_config,
    lock,
    unlock,
    get_locks,
)
from layerzero.api.inspection import (
    select,
    explain,
    which,
    list_kernels,
    validate,
)
from layerzero.api.system import (
    doctor,
    readiness_check,
    compile,
    dry_run,
    solve,
    tune,
)

__all__ = [
    # Operations
    "attention",
    "rms_norm",
    "layer_norm",
    "sample_topk",
    "sample_topp",
    "tokenize",
    "detokenize",
    # Configuration
    "configure",
    "get_config",
    "load_config",
    "lock",
    "unlock",
    "get_locks",
    # Inspection
    "select",
    "explain",
    "which",
    "list_kernels",
    "validate",
    # System
    "doctor",
    "readiness_check",
    "compile",
    "dry_run",
    "solve",
    "tune",
]
