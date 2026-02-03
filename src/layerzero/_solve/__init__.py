"""
Build-time solver module (lz.solve).

This module provides:
- Solver: Build-time solver for generating dispatch tables
- SolverConfig: Configuration for solver
- SolverResult: Result of solve operation
- DispatchTable: Dispatch table for kernel selection
- DispatchEntry: Entry in dispatch table
- ShapeBucket: Shape bucket for bucketed dispatch
- BucketRange: Range for a dimension in a bucket
- solve: Convenience function for solving
"""
from __future__ import annotations

from layerzero._solve.dispatch_table import (
    BucketRange,
    DispatchEntry,
    DispatchTable,
    ShapeBucket,
)
from layerzero._solve.solver import (
    Solver,
    SolverConfig,
    SolverResult,
    solve,
)

__all__ = [
    # Solver
    "Solver",
    "SolverConfig",
    "SolverResult",
    "solve",
    # Dispatch table
    "BucketRange",
    "DispatchEntry",
    "DispatchTable",
    "ShapeBucket",
]
