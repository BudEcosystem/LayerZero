"""
Distributed selection consistency module.

This module provides:
- ConsistencyConfig: Configuration for distributed consistency
- ConsistencyMode: Strict/relaxed/disabled modes
- SelectionHash: Hash for selection verification
- VersionChecker: Checks LayerZero version across ranks
- SelectionSynchronizer: Synchronizes selection across ranks
- TPConfig: Configuration for tensor parallel invariance
- TPInvarianceFilter: Filters kernels by TP invariance
- TPContext: Tensor parallel context
"""
from __future__ import annotations

from layerzero.distributed.consistency import (
    ConsistencyConfig,
    ConsistencyError,
    ConsistencyMode,
    DistributedContext,
    SelectionHash,
    SelectionSynchronizer,
    VersionChecker,
    get_distributed_context,
    is_distributed,
)
from layerzero.distributed.tp_invariance import (
    TPConfig,
    TPContext,
    TPInvarianceFilter,
    get_tp_context,
    is_tp_enabled,
    require_tp_invariant,
)

__all__ = [
    # Consistency
    "ConsistencyConfig",
    "ConsistencyError",
    "ConsistencyMode",
    "DistributedContext",
    "SelectionHash",
    "SelectionSynchronizer",
    "VersionChecker",
    "get_distributed_context",
    "is_distributed",
    # TP Invariance
    "TPConfig",
    "TPContext",
    "TPInvarianceFilter",
    "get_tp_context",
    "is_tp_enabled",
    "require_tp_invariant",
]
