"""
LayerZero Policy Loader

Loads policy from YAML files and environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from layerzero.policy.policy import Policy
from layerzero.policy.rule import (
    Condition,
    ConditionOp,
    Rule,
    RuleType,
    parse_condition,
)


class PolicyLoader:
    """Load policy from YAML files and environment variables.

    Priority (highest first):
    1. Environment variables
    2. YAML file
    3. Defaults (empty policy)

    Environment Variable Format:
        LAYERZERO_LOCK_{operation}={kernel}
        LAYERZERO_DENY={kernel1},{kernel2}
        LAYERZERO_ALLOW={backend1},{backend2}
        LAYERZERO_BOOST_{kernel}={+/-}{amount}
    """

    def __init__(self) -> None:
        """Initialize policy loader."""
        pass

    def load(
        self,
        yaml_path: Path | None = None,
        env_prefix: str = "LAYERZERO_",
    ) -> Policy:
        """Load policy from all sources.

        Args:
            yaml_path: Path to YAML config file (optional).
            env_prefix: Environment variable prefix.

        Returns:
            Compiled Policy object.
        """
        # Start with empty policy
        yaml_dict: dict[str, Any] = {"version": "1.0"}
        env_dict: dict[str, Any] = {}

        # Load YAML if provided
        if yaml_path is not None and yaml_path.exists():
            yaml_dict = self.load_yaml(yaml_path)

        # Load environment variables
        env_dict = self.load_env(env_prefix)

        # Merge sources (env overrides yaml)
        merged = self.merge(yaml_dict, env_dict)

        # Compile to Policy
        return self.compile(merged)

    def load_yaml(self, path: Path) -> dict[str, Any]:
        """Load and parse YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            Parsed YAML as dict.

        Raises:
            FileNotFoundError: If file doesn't exist.
            yaml.YAMLError: If YAML is invalid.
        """
        try:
            import yaml
        except ImportError:
            # PyYAML not installed, return empty
            return {"version": "1.0"}

        with open(path, "r") as f:
            content = yaml.safe_load(f)

        if content is None:
            return {"version": "1.0"}

        return content

    def load_env(self, prefix: str) -> dict[str, Any]:
        """Load policy rules from environment variables.

        Args:
            prefix: Environment variable prefix.

        Returns:
            Dict with rules from environment.
        """
        result: dict[str, Any] = {
            "locks": [],
            "denies": [],
            "allows": [],
            "boosts": [],
        }

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            suffix = key[len(prefix):]

            # LAYERZERO_LOCK_{operation}={kernel}
            if suffix.startswith("LOCK_"):
                operation = suffix[5:].replace("_", ".")
                result["locks"].append({
                    "operation": operation,
                    "kernel": value,
                })

            # LAYERZERO_DENY={kernel1},{kernel2}
            elif suffix == "DENY":
                for kernel in value.split(","):
                    kernel = kernel.strip()
                    if kernel:
                        result["denies"].append({"kernel": kernel})

            # LAYERZERO_ALLOW={backend1},{backend2}
            elif suffix == "ALLOW":
                for backend in value.split(","):
                    backend = backend.strip()
                    if backend:
                        result["allows"].append({"backend": backend})

            # LAYERZERO_BOOST_{kernel}={+/-}{amount}
            elif suffix.startswith("BOOST_"):
                kernel = suffix[6:].replace("_", ".")
                amount = self._parse_boost_value(value)
                result["boosts"].append({
                    "kernel": kernel,
                    "priority_add": amount,
                })

        return result

    def merge(self, *sources: dict[str, Any]) -> dict[str, Any]:
        """Merge policy sources with precedence.

        Later sources override earlier ones.

        Args:
            *sources: Dicts to merge.

        Returns:
            Merged dict.
        """
        result: dict[str, Any] = {
            "version": "1.0",
            "locks": [],
            "deny": [],
            "allow": [],
            "boosts": [],
        }

        for source in sources:
            if "version" in source:
                result["version"] = source["version"]

            # Merge lists (append)
            if "locks" in source:
                result["locks"].extend(source["locks"])
            if "deny" in source:
                result["deny"].extend(source["deny"])
            if "denies" in source:
                result["deny"].extend(source["denies"])
            if "allow" in source:
                result["allow"].extend(source["allow"])
            if "allows" in source:
                result["allow"].extend(source["allows"])
            if "boosts" in source:
                result["boosts"].extend(source["boosts"])

        return result

    def compile(self, raw: dict[str, Any]) -> Policy:
        """Compile raw dict to Policy with Rules.

        Args:
            raw: Raw policy dict.

        Returns:
            Compiled Policy object.
        """
        locks: list[Rule] = []
        allows: list[Rule] = []
        denies: list[Rule] = []
        boosts: list[Rule] = []

        # Compile lock rules
        for item in raw.get("locks", []):
            operation = item.get("operation", "*")
            kernel = item.get("kernel", "*")
            conditions = self._compile_conditions(item.get("when", {}))
            locks.append(Rule(
                rule_type=RuleType.LOCK,
                target=operation,
                conditions=tuple(conditions),
                value=kernel,
            ))

        # Compile deny rules
        for item in raw.get("deny", []):
            target = item.get("kernel") or item.get("backend", "*")
            conditions = self._compile_conditions(item.get("when", {}))
            denies.append(Rule(
                rule_type=RuleType.DENY,
                target=target,
                conditions=tuple(conditions),
                value=None,
            ))

        # Compile allow rules
        for item in raw.get("allow", []):
            target = item.get("kernel") or item.get("backend", "*")
            conditions = self._compile_conditions(item.get("when", {}))
            allows.append(Rule(
                rule_type=RuleType.ALLOW,
                target=target,
                conditions=tuple(conditions),
                value=None,
            ))

        # Compile boost rules
        for item in raw.get("boosts", []):
            target = item.get("kernel") or item.get("backend", "*")
            conditions = self._compile_conditions(item.get("when", {}))

            # Handle additive boost
            if "priority_add" in item:
                boosts.append(Rule(
                    rule_type=RuleType.BOOST_ADD,
                    target=target,
                    conditions=tuple(conditions),
                    value=item["priority_add"],
                ))

            # Handle multiplicative boost
            if "priority_multiply" in item:
                boosts.append(Rule(
                    rule_type=RuleType.BOOST_MULTIPLY,
                    target=target,
                    conditions=tuple(conditions),
                    value=item["priority_multiply"],
                ))

        return Policy(
            version=raw.get("version", "1.0"),
            locks=tuple(locks),
            allows=tuple(allows),
            denies=tuple(denies),
            boosts=tuple(boosts),
        )

    def _compile_conditions(
        self,
        when_dict: dict[str, str],
    ) -> list[Condition]:
        """Compile 'when' dict to Condition objects.

        Args:
            when_dict: Dict of field -> condition string.

        Returns:
            List of Condition objects.
        """
        conditions: list[Condition] = []
        for field, value in when_dict.items():
            conditions.append(parse_condition(field, str(value)))
        return conditions

    def _parse_boost_value(self, value: str) -> int:
        """Parse boost value from string.

        Args:
            value: Boost value string (e.g., "+20", "-10", "20")

        Returns:
            Parsed integer value.
        """
        value = value.strip()
        if value.startswith("+"):
            return int(value[1:])
        return int(value)
