"""
LayerZero Rule Engine

Evaluates policy rules against selection context.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from layerzero.policy.rule import RuleType

if TYPE_CHECKING:
    from layerzero.models.selection_context import SelectionContext
    from layerzero.policy.policy import Policy


class RuleEngine:
    """Evaluate policy rules against context.

    Evaluation order:
    1. Locks (if any match, return locked kernel)
    2. Denies (filter out denied kernels)
    3. Allows (if any allows, filter to allowed only)
    4. Boosts (modify priorities of matching kernels)

    Thread-safe (immutable policy, stateless evaluation).
    """

    __slots__ = ("_policy",)

    def __init__(self, policy: "Policy") -> None:
        """Initialize rule engine with policy.

        Args:
            policy: Policy containing rules to evaluate.
        """
        self._policy = policy

    @property
    def policy(self) -> "Policy":
        """Get the policy being used."""
        return self._policy

    def get_locked_kernel(
        self,
        ctx: "SelectionContext",
    ) -> str | None:
        """Get locked kernel ID if any lock rule matches.

        Lock rules are evaluated in order. First match wins.

        Args:
            ctx: Selection context to evaluate.

        Returns:
            Kernel ID if locked, None otherwise.
        """
        for rule in self._policy.locks:
            # Check if rule operation matches context operation
            if rule.matches_operation(ctx.operation):
                # Check if conditions match
                if rule.matches_context(ctx):
                    return rule.value
        return None

    def is_denied(
        self,
        kernel_id: str,
        ctx: "SelectionContext",
    ) -> bool:
        """Check if kernel is denied for context.

        Args:
            kernel_id: Kernel identifier to check.
            ctx: Selection context.

        Returns:
            True if kernel is denied.
        """
        for rule in self._policy.denies:
            if rule.matches_target(kernel_id):
                if rule.matches_context(ctx):
                    return True
        return False

    def is_allowed(
        self,
        kernel_id: str,
        ctx: "SelectionContext",
    ) -> bool:
        """Check if kernel is allowed for context.

        If no allow rules exist, all kernels are allowed.
        If allow rules exist, only matching kernels are allowed.

        Args:
            kernel_id: Kernel identifier to check.
            ctx: Selection context.

        Returns:
            True if kernel is allowed.
        """
        # If no allow rules, everything is allowed
        if not self._policy.allows:
            return True

        # With allow rules, check if any match
        for rule in self._policy.allows:
            if rule.matches_target(kernel_id):
                if rule.matches_context(ctx):
                    return True

        return False

    def get_priority_boost(
        self,
        kernel_id: str,
        ctx: "SelectionContext",
    ) -> tuple[int, float]:
        """Get priority boosts for a kernel.

        Multiple boost rules can apply. Additive boosts are summed,
        multiplicative boosts are multiplied.

        Args:
            kernel_id: Kernel identifier.
            ctx: Selection context.

        Returns:
            Tuple of (additive_boost, multiplicative_boost).
        """
        add_boost = 0
        mult_boost = 1.0

        for rule in self._policy.boosts:
            if rule.matches_target(kernel_id):
                if rule.matches_context(ctx):
                    if rule.rule_type == RuleType.BOOST_ADD:
                        add_boost += rule.value
                    elif rule.rule_type == RuleType.BOOST_MULTIPLY:
                        mult_boost *= rule.value

        return add_boost, mult_boost

    def filter_kernels(
        self,
        kernel_ids: list[str],
        ctx: "SelectionContext",
    ) -> list[str]:
        """Filter kernel list by allow/deny rules.

        Order of filtering:
        1. Remove denied kernels
        2. If allow rules exist, keep only allowed kernels

        Args:
            kernel_ids: List of kernel identifiers.
            ctx: Selection context.

        Returns:
            Filtered list of kernel identifiers.
        """
        result = []

        for kernel_id in kernel_ids:
            # Check deny rules
            if self.is_denied(kernel_id, ctx):
                continue

            # Check allow rules
            if not self.is_allowed(kernel_id, ctx):
                continue

            result.append(kernel_id)

        return result
