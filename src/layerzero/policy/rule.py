"""
LayerZero Policy Rule

Rule and Condition classes for policy matching.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from layerzero.models.selection_context import SelectionContext


@unique
class ConditionOp(str, Enum):
    """Condition comparison operators."""

    EQ = "=="        # Equal
    NE = "!="        # Not equal
    GT = ">"         # Greater than
    GE = ">="        # Greater or equal
    LT = "<"         # Less than
    LE = "<="        # Less or equal
    IN = "in"        # In set
    NOT_IN = "not_in"  # Not in set
    MATCH = "match"    # Regex/glob match


@unique
class RuleType(str, Enum):
    """Rule action types."""

    LOCK = "lock"               # Force specific kernel
    ALLOW = "allow"             # Permit kernel/backend
    DENY = "deny"               # Block kernel/backend
    BOOST_ADD = "boost_add"           # Add to priority
    BOOST_MULTIPLY = "boost_multiply"  # Multiply priority


@dataclass(frozen=True, slots=True)
class Condition:
    """Single match condition.

    Evaluates whether a context field matches a value using
    a comparison operator.

    Attributes:
        field: Context field name (e.g., "head_dim", "operation")
        op: Comparison operator
        value: Value to compare against
    """

    field: str
    op: ConditionOp
    value: Any

    def matches(self, ctx: "SelectionContext") -> bool:
        """Check if condition matches context.

        Args:
            ctx: Selection context to evaluate.

        Returns:
            True if condition matches, False otherwise.
        """
        actual = getattr(ctx, self.field, None)
        if actual is None:
            return False

        if self.op == ConditionOp.EQ:
            return actual == self.value
        elif self.op == ConditionOp.NE:
            return actual != self.value
        elif self.op == ConditionOp.GT:
            return actual > self.value
        elif self.op == ConditionOp.GE:
            return actual >= self.value
        elif self.op == ConditionOp.LT:
            return actual < self.value
        elif self.op == ConditionOp.LE:
            return actual <= self.value
        elif self.op == ConditionOp.IN:
            return actual in self.value
        elif self.op == ConditionOp.NOT_IN:
            return actual not in self.value
        elif self.op == ConditionOp.MATCH:
            # Use fnmatch for glob patterns
            if isinstance(actual, str):
                return fnmatch.fnmatch(actual, self.value)
            return False

        return False


@dataclass(frozen=True, slots=True)
class Rule:
    """Policy rule.

    Defines an action to take (lock, allow, deny, boost) for
    a target pattern when conditions are met.

    Attributes:
        rule_type: Type of rule action
        target: Target pattern (kernel_id, backend_id, operation glob)
        conditions: Tuple of conditions (AND-ed together)
        value: Rule-specific value (kernel_id for lock, amount for boost)
    """

    rule_type: RuleType
    target: str  # Glob pattern for matching
    conditions: tuple[Condition, ...]
    value: Any  # Lock: kernel_id, Boost: amount

    def matches_context(self, ctx: "SelectionContext") -> bool:
        """Check if all conditions match context.

        Args:
            ctx: Selection context to evaluate.

        Returns:
            True if all conditions match (empty conditions = always match).
        """
        if not self.conditions:
            return True
        return all(c.matches(ctx) for c in self.conditions)

    def matches_target(self, target_id: str) -> bool:
        """Check if target pattern matches ID.

        Uses fnmatch for glob-style matching.

        Args:
            target_id: Target identifier to check.

        Returns:
            True if pattern matches target.
        """
        return fnmatch.fnmatch(target_id, self.target)

    def matches_operation(self, operation: str) -> bool:
        """Check if rule target matches operation.

        For lock rules, the target is an operation pattern.

        Args:
            operation: Operation identifier.

        Returns:
            True if target pattern matches operation.
        """
        return fnmatch.fnmatch(operation, self.target)


def parse_condition(field: str, value: str) -> Condition:
    """Parse condition string to Condition object.

    Supports operators: >=, <=, !=, ==, >, <
    Supports glob patterns: * and ?
    Plain values default to equality.

    Args:
        field: Context field name.
        value: Condition value string (e.g., ">=64", "attention.*")

    Returns:
        Parsed Condition object.

    Examples:
        parse_condition("head_dim", ">=64")
        -> Condition("head_dim", ConditionOp.GE, 64)

        parse_condition("operation", "attention.*")
        -> Condition("operation", ConditionOp.MATCH, "attention.*")
    """
    value = value.strip()

    # Check operators in order of length (longest first)
    operators = [
        (">=", ConditionOp.GE),
        ("<=", ConditionOp.LE),
        ("!=", ConditionOp.NE),
        ("==", ConditionOp.EQ),
        (">", ConditionOp.GT),
        ("<", ConditionOp.LT),
    ]

    for op_str, op in operators:
        if value.startswith(op_str):
            raw_value = value[len(op_str):].strip()
            parsed = _parse_value(raw_value)
            return Condition(field, op, parsed)

    # Check for glob pattern
    if "*" in value or "?" in value:
        return Condition(field, ConditionOp.MATCH, value)

    # Default to equality
    return Condition(field, ConditionOp.EQ, _parse_value(value))


def _parse_value(raw: str) -> Any:
    """Parse raw string value to appropriate type.

    Tries int, then float, then returns string.

    Args:
        raw: Raw string value.

    Returns:
        Parsed value (int, float, or str).
    """
    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        pass

    return raw
