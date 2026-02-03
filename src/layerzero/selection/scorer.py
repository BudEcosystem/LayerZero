"""
LayerZero Scoring Phase

Scores kernel candidates for selection priority.
Second stage of the selection pipeline.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.models.selection_context import SelectionContext
    from layerzero.policy.engine import RuleEngine


class ScoringPhase:
    """Score kernels for selection priority.

    Score formula:
        final_score = (base_priority + additive_boost) * multiplicative_boost - transform_cost

    Score factors:
    - Base priority (from KernelSpec.priority)
    - Policy additive boosts (summed)
    - Policy multiplicative boosts (multiplied)
    - Transform cost penalty (KernelSpec.transform_cost_hint)

    Thread-safe (stateless evaluation).
    """

    __slots__ = ("_rule_engine",)

    def __init__(self, rule_engine: "RuleEngine") -> None:
        """Initialize scoring phase with rule engine.

        Args:
            rule_engine: Rule engine for policy boost evaluation.
        """
        self._rule_engine = rule_engine

    @property
    def rule_engine(self) -> "RuleEngine":
        """Get the rule engine being used."""
        return self._rule_engine

    def score(
        self,
        candidates: list["KernelSpec"],
        ctx: "SelectionContext",
    ) -> dict[str, float]:
        """Score each candidate kernel.

        Args:
            candidates: Filtered candidate kernels.
            ctx: Selection context for conditional boosts.

        Returns:
            Dict of kernel_id -> score (higher = better).
        """
        scores: dict[str, float] = {}

        for kernel in candidates:
            # Start with base priority
            score = float(kernel.priority)

            # Get policy boosts
            add_boost, mult_boost = self._rule_engine.get_priority_boost(
                kernel.kernel_id, ctx
            )

            # Apply additive boost first, then multiplicative
            score = (score + add_boost) * mult_boost

            # Subtract transform cost penalty
            score -= kernel.transform_cost_hint

            scores[kernel.kernel_id] = score

        return scores
