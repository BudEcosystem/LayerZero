"""
SelectionReport - Full Trace of Kernel Selection Decision

Provides detailed information about why a kernel was selected,
including all candidates, rejection reasons, and scores.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class KernelCandidate:
    """A candidate kernel considered during selection.

    Represents a single kernel that was evaluated during the
    selection process, along with its score and rejection status.

    Attributes:
        kernel_id: Unique identifier for the kernel.
        score: Selection score (0-1), or None if rejected.
        rejected: Whether the kernel was rejected.
        rejection_reasons: Tuple of reason codes for rejection.
        metadata: Additional kernel metadata.
    """

    kernel_id: str
    score: float | None
    rejected: bool
    rejection_reasons: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "kernel_id": self.kernel_id,
            "score": self.score,
            "rejected": self.rejected,
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KernelCandidate:
        """Create from dictionary representation."""
        return cls(
            kernel_id=data["kernel_id"],
            score=data.get("score"),
            rejected=data.get("rejected", False),
            rejection_reasons=tuple(data.get("rejection_reasons", [])),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class SelectionReport:
    """Full trace of kernel selection decision.

    Contains complete information about a kernel selection,
    including all candidates considered, their scores, and
    rejection reasons.

    Attributes:
        operation: The operation type (e.g., "attention", "matmul").
        chosen_kernel_id: ID of the selected kernel, or None if none selected.
        candidates: All kernel candidates that were evaluated.
        selection_latency_ns: Time taken for selection in nanoseconds.
        cache_hit: Whether the selection was a cache hit.
        timestamp: Unix timestamp when selection occurred.
        context: Additional context about the selection.

    Example:
        ```python
        report = SelectionReport(
            operation="attention",
            chosen_kernel_id="flash_attn.fwd",
            candidates=(...),
            selection_latency_ns=1500,
            cache_hit=False,
            timestamp=time.time(),
            context={"batch_size": 8},
        )

        # Print rejection reasons
        for c in report.candidates:
            if c.rejected:
                print(f"{c.kernel_id}: {c.rejection_reasons}")
        ```
    """

    operation: str
    chosen_kernel_id: str | None
    candidates: tuple[KernelCandidate, ...]
    selection_latency_ns: int
    cache_hit: bool
    timestamp: float
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "operation": self.operation,
            "chosen_kernel_id": self.chosen_kernel_id,
            "candidates": [c.to_dict() for c in self.candidates],
            "selection_latency_ns": self.selection_latency_ns,
            "cache_hit": self.cache_hit,
            "timestamp": self.timestamp,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SelectionReport:
        """Create from dictionary representation."""
        candidates = tuple(
            KernelCandidate.from_dict(c)
            for c in data.get("candidates", [])
        )
        return cls(
            operation=data["operation"],
            chosen_kernel_id=data.get("chosen_kernel_id"),
            candidates=candidates,
            selection_latency_ns=data.get("selection_latency_ns", 0),
            cache_hit=data.get("cache_hit", False),
            timestamp=data.get("timestamp", 0.0),
            context=data.get("context", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string.

        Returns:
            JSON representation of the report.
        """
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, data: str) -> SelectionReport:
        """Deserialize from JSON string.

        Args:
            data: JSON string representation.

        Returns:
            SelectionReport instance.
        """
        return cls.from_dict(json.loads(data))

    def summary(self) -> str:
        """Get a concise summary of the selection.

        Returns:
            Human-readable summary string.
        """
        lines = [
            f"Operation: {self.operation}",
            f"Chosen: {self.chosen_kernel_id or 'None'}",
            f"Latency: {self.selection_latency_ns}ns",
            f"Cache hit: {self.cache_hit}",
            f"Candidates: {len(self.candidates)}",
        ]

        # Show valid candidates
        valid = [c for c in self.candidates if not c.rejected]
        if valid:
            lines.append("\nValid candidates:")
            for c in valid:
                score_str = f"{c.score:.2f}" if c.score is not None else "N/A"
                chosen = " [CHOSEN]" if c.kernel_id == self.chosen_kernel_id else ""
                lines.append(f"  - {c.kernel_id}: score={score_str}{chosen}")

        # Show rejected candidates
        rejected = [c for c in self.candidates if c.rejected]
        if rejected:
            lines.append("\nRejected candidates:")
            for c in rejected:
                reasons = ", ".join(c.rejection_reasons) if c.rejection_reasons else "unknown"
                lines.append(f"  - {c.kernel_id}: {reasons}")

        return "\n".join(lines)

    def pretty_print(self) -> str:
        """Get formatted output for display.

        Returns:
            Formatted string for terminal display.
        """
        return self.summary()

    def format(self) -> str:
        """Alias for pretty_print."""
        return self.pretty_print()

    def __str__(self) -> str:
        """String representation."""
        return self.summary()

    @property
    def rejected_candidates(self) -> tuple[KernelCandidate, ...]:
        """Get all rejected candidates."""
        return tuple(c for c in self.candidates if c.rejected)

    @property
    def valid_candidates(self) -> tuple[KernelCandidate, ...]:
        """Get all valid (non-rejected) candidates."""
        return tuple(c for c in self.candidates if not c.rejected)

    @property
    def chosen_candidate(self) -> KernelCandidate | None:
        """Get the chosen candidate, if any."""
        if self.chosen_kernel_id is None:
            return None
        for c in self.candidates:
            if c.kernel_id == self.chosen_kernel_id:
                return c
        return None
