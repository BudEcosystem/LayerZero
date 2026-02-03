"""
LayerZero Policy Container

Policy class containing all rules.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layerzero.policy.rule import Rule


@dataclass
class Policy:
    """Collection of policy rules.

    Contains all rules organized by type for efficient evaluation.
    Computes a hash for cache invalidation when rules change.

    Attributes:
        version: Policy schema version
        locks: Lock rules (evaluated first, force specific kernel)
        allows: Allow rules (whitelist)
        denies: Deny rules (blacklist)
        boosts: Boost rules (priority modifiers)
    """

    version: str
    locks: tuple["Rule", ...]
    allows: tuple["Rule", ...]
    denies: tuple["Rule", ...]
    boosts: tuple["Rule", ...]

    _hash: str | None = field(default=None, repr=False, compare=False)

    @property
    def is_empty(self) -> bool:
        """Check if policy has no rules.

        Returns:
            True if no rules defined.
        """
        return (
            len(self.locks) == 0
            and len(self.allows) == 0
            and len(self.denies) == 0
            and len(self.boosts) == 0
        )

    @property
    def policy_hash(self) -> str:
        """Get deterministic hash of policy content.

        Used for cache invalidation when policy changes.

        Returns:
            SHA256 hash string.
        """
        if self._hash is not None:
            return self._hash

        # Build normalized representation
        content = {
            "version": self.version,
            "locks": [self._rule_to_dict(r) for r in self.locks],
            "allows": [self._rule_to_dict(r) for r in self.allows],
            "denies": [self._rule_to_dict(r) for r in self.denies],
            "boosts": [self._rule_to_dict(r) for r in self.boosts],
        }

        # Sort keys for determinism
        json_str = json.dumps(content, sort_keys=True)
        hash_value = hashlib.sha256(json_str.encode()).hexdigest()

        # Cache the hash
        object.__setattr__(self, "_hash", hash_value)
        return hash_value

    @staticmethod
    def _rule_to_dict(rule: "Rule") -> dict:
        """Convert rule to dict for hashing.

        Args:
            rule: Rule to convert.

        Returns:
            Dict representation.
        """
        return {
            "type": rule.rule_type.value,
            "target": rule.target,
            "conditions": [
                {
                    "field": c.field,
                    "op": c.op.value,
                    "value": str(c.value),
                }
                for c in rule.conditions
            ],
            "value": str(rule.value) if rule.value is not None else None,
        }

    @property
    def rule_count(self) -> int:
        """Total number of rules.

        Returns:
            Sum of all rule counts.
        """
        return (
            len(self.locks)
            + len(self.allows)
            + len(self.denies)
            + len(self.boosts)
        )
