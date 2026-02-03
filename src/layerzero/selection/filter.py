"""
LayerZero Filter Phase

Filters kernel candidates by compatibility with SelectionContext.
First stage of the selection pipeline.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.models.selection_context import SelectionContext
    from layerzero.reasons import Reason


class FilterPhase:
    """Filter kernels by compatibility with context.

    Checks:
    - Platform compatibility (CUDA/ROCm/CPU)
    - SM version (min/max)
    - GPU generation
    - Dtype support
    - Shape constraints (head_dim, seq_len, batch_size)
    - Layout requirements
    - Feature support (GQA, dropout, mask)
    - CUDA graph safety
    - Determinism requirements

    Delegates actual checking to KernelSpec.check() method.
    Thread-safe (stateless).
    """

    __slots__ = ()

    def filter(
        self,
        candidates: list["KernelSpec"],
        ctx: "SelectionContext",
    ) -> tuple[list["KernelSpec"], dict[str, list["Reason"]]]:
        """Filter candidates by context compatibility.

        Args:
            candidates: List of candidate kernel specs.
            ctx: Selection context with runtime requirements.

        Returns:
            Tuple of:
            - valid: List of kernels that pass all checks.
            - filtered_out: Dict mapping kernel_id to list of failure reasons.
        """
        valid: list["KernelSpec"] = []
        filtered_out: dict[str, list["Reason"]] = {}

        for kernel in candidates:
            # Delegate to KernelSpec.check() for all compatibility checks
            reasons = kernel.check(ctx)

            if reasons:
                # Kernel failed one or more checks
                filtered_out[kernel.kernel_id] = reasons
            else:
                # Kernel passes all checks
                valid.append(kernel)

        return valid, filtered_out
