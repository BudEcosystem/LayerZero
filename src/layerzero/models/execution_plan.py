"""
LayerZero Execution Plan

Dataclasses representing selected kernel execution plans and
selection reports for debugging/explainability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.reasons import Reason


@dataclass(frozen=True, slots=True)
class SelectionReport:
    """Detailed report of kernel selection process.

    For debugging and explainability. Captures all candidates considered,
    why some were filtered out, scores assigned, and the final selection.

    Attributes:
        operation: Operation being performed (e.g., "attention.causal")
        context_summary: Key context fields used in selection
        candidates: All kernel IDs considered
        filtered_out: Map of kernel_id -> list of failure reasons
        scores: Map of kernel_id -> score for valid candidates
        selected_kernel: Final selected kernel ID
        selection_reason: Reason for selection (e.g., "highest_priority")
        selection_time_us: Time taken for selection in microseconds
    """

    operation: str
    context_summary: dict[str, Any]
    candidates: tuple[str, ...]
    filtered_out: dict[str, list["Reason"]]
    scores: dict[str, float]
    selected_kernel: str
    selection_reason: str
    selection_time_us: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON compatibility.

        Note: Reason objects in filtered_out are serialized to their dict form.

        Returns:
            Dict with all report fields.
        """
        # Serialize filtered_out reasons
        filtered_out_serialized: dict[str, list[dict[str, Any]]] = {}
        for kernel_id, reasons in self.filtered_out.items():
            filtered_out_serialized[kernel_id] = [
                {
                    "code": r.code,
                    "message": r.message,
                    "category": r.category.value,
                }
                for r in reasons
            ]

        return {
            "operation": self.operation,
            "context_summary": self.context_summary,
            "candidates": list(self.candidates),
            "filtered_out": filtered_out_serialized,
            "scores": self.scores,
            "selected_kernel": self.selected_kernel,
            "selection_reason": self.selection_reason,
            "selection_time_us": self.selection_time_us,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SelectionReport":
        """Deserialize from dictionary.

        Note: Reason objects in filtered_out are reconstructed from dicts.

        Args:
            d: Dict with report fields.

        Returns:
            New SelectionReport instance.
        """
        from layerzero.reasons import Reason, ReasonCategory

        # Deserialize filtered_out reasons
        filtered_out: dict[str, list[Reason]] = {}
        for kernel_id, reason_dicts in d.get("filtered_out", {}).items():
            filtered_out[kernel_id] = [
                Reason(
                    code=r["code"],
                    message=r["message"],
                    category=ReasonCategory(r["category"]),
                )
                for r in reason_dicts
            ]

        return cls(
            operation=d["operation"],
            context_summary=d.get("context_summary", {}),
            candidates=tuple(d.get("candidates", [])),
            filtered_out=filtered_out,
            scores=d.get("scores", {}),
            selected_kernel=d["selected_kernel"],
            selection_reason=d.get("selection_reason", "unknown"),
            selection_time_us=d.get("selection_time_us", 0),
        )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Execution plan for a selected kernel.

    Contains the selected kernel specification plus any transforms
    needed to convert inputs/outputs to the kernel's expected format.
    Immutable (frozen) for hashability and thread safety.

    Attributes:
        kernel_id: Selected kernel identifier
        kernel_spec: Full kernel specification
        pre_transforms: Transform names to apply before kernel execution
        post_transforms: Transform names to apply after kernel execution
        debug_info: Selection report for debugging (optional)
        cached: Whether this plan was retrieved from cache
        cache_key: Cache key used for this plan (if cached)
    """

    kernel_id: str
    kernel_spec: "KernelSpec"
    pre_transforms: tuple[str, ...] = ()
    post_transforms: tuple[str, ...] = ()
    debug_info: SelectionReport | None = None
    cached: bool = False
    cache_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON compatibility.

        Note: kernel_spec is serialized via its to_dict method.
        debug_info is serialized if present.

        Returns:
            Dict with all plan fields.
        """
        result = {
            "kernel_id": self.kernel_id,
            "kernel_spec": self.kernel_spec.to_dict(),
            "pre_transforms": list(self.pre_transforms),
            "post_transforms": list(self.post_transforms),
            "cached": self.cached,
            "cache_key": self.cache_key,
        }

        if self.debug_info is not None:
            result["debug_info"] = self.debug_info.to_dict()
        else:
            result["debug_info"] = None

        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExecutionPlan":
        """Deserialize from dictionary.

        Note: kernel_spec.impl will be None after deserialization.
        debug_info is reconstructed if present.

        Args:
            d: Dict with plan fields.

        Returns:
            New ExecutionPlan instance.
        """
        from layerzero.models.kernel_spec import KernelSpec

        debug_info = None
        if d.get("debug_info") is not None:
            debug_info = SelectionReport.from_dict(d["debug_info"])

        return cls(
            kernel_id=d["kernel_id"],
            kernel_spec=KernelSpec.from_dict(d["kernel_spec"]),
            pre_transforms=tuple(d.get("pre_transforms", [])),
            post_transforms=tuple(d.get("post_transforms", [])),
            debug_info=debug_info,
            cached=d.get("cached", False),
            cache_key=d.get("cache_key"),
        )

    def has_transforms(self) -> bool:
        """Check if this plan has any transforms.

        Returns:
            True if there are pre or post transforms.
        """
        return len(self.pre_transforms) > 0 or len(self.post_transforms) > 0

    def transform_count(self) -> int:
        """Get total number of transforms.

        Returns:
            Count of pre + post transforms.
        """
        return len(self.pre_transforms) + len(self.post_transforms)
