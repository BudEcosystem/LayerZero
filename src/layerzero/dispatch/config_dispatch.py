"""
Config-Driven Kernel Dispatch Implementation.

Provides YAML-based configuration for ops-controlled kernel deployment.
Enables declarative dispatch rules with conditions, priority overrides,
kernel locks, whitelisting, and blacklisting.

Features:
- Schema validation for config files
- Rule-based kernel selection with condition evaluation
- Kernel whitelisting/blacklisting via patterns
- Priority overrides from config
- Fallback chains defined in config
- Efficient rule evaluation with caching
- Thread-safe implementation

Config Format:
```yaml
version: "1.0"
defaults:
  fallback_policy: torch_sdpa
dispatch_rules:
  - operation: "attention.*"
    conditions:
      batch_size_gte: 8
      dtype: [float16, bfloat16]
    kernel: flash_attention_v2
    priority: 100
  - operation: "attention.causal"
    conditions:
      seq_len_gt: 8192
    kernel: xformers_memory_efficient
    priority: 90
kernel_locks:
  attention.causal: flash_attention_v2
kernel_denies:
  - "*_experimental"
kernel_allows:
  - "flash_attn.*"
  - "torch_sdpa"
fallback_chains:
  attention.causal:
    - flash_attention_v2
    - xformers_memory_efficient
    - torch_sdpa
```
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, ClassVar, Final, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from layerzero.dispatch.executor import KernelExecutorImpl
    from layerzero.dispatch.protocols import KernelExecutor
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.models.selection_context import SelectionContext
    from layerzero.policy.policy import Policy
    from layerzero.registry.kernel_registry import KernelRegistry

from layerzero.dispatch.protocols import BaseDispatcher
from layerzero.dispatch.types import (
    ConfigurationError,
    DispatchConfig,
    DispatchMode,
    DispatchResult,
    DispatchTiming,
    FallbackChainExhaustedError,
    KernelExecutionError,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Config Schema Version
# ============================================================================

SCHEMA_VERSION: Final[str] = "1.0"
SUPPORTED_VERSIONS: Final[frozenset[str]] = frozenset({"1.0"})


# ============================================================================
# Condition Operators
# ============================================================================

class ConditionOperator(str, Enum):
    """Operators for dispatch rule conditions."""

    EQ = "eq"           # Equal
    NE = "ne"           # Not equal
    GT = "gt"           # Greater than
    GTE = "gte"         # Greater than or equal
    LT = "lt"           # Less than
    LTE = "lte"         # Less than or equal
    IN = "in"           # In list
    NOT_IN = "not_in"   # Not in list
    MATCH = "match"     # Pattern match (glob)
    REGEX = "regex"     # Regex match


# Condition field suffix to operator mapping
CONDITION_SUFFIXES: Final[dict[str, ConditionOperator]] = {
    "_eq": ConditionOperator.EQ,
    "_ne": ConditionOperator.NE,
    "_gt": ConditionOperator.GT,
    "_gte": ConditionOperator.GTE,
    "_lt": ConditionOperator.LT,
    "_lte": ConditionOperator.LTE,
    "_in": ConditionOperator.IN,
    "_not_in": ConditionOperator.NOT_IN,
    "_match": ConditionOperator.MATCH,
    "_regex": ConditionOperator.REGEX,
}


# ============================================================================
# Schema Validation
# ============================================================================

@dataclass(frozen=True, slots=True)
class SchemaError:
    """Schema validation error."""

    path: str
    message: str
    value: Any = None

    def __str__(self) -> str:
        if self.value is not None:
            return f"{self.path}: {self.message} (got: {self.value!r})"
        return f"{self.path}: {self.message}"


class ConfigSchema:
    """Schema validator for dispatch config files.

    Validates:
    - Version compatibility
    - Required fields
    - Field types
    - Value constraints

    Thread-safe (stateless validation).
    """

    __slots__ = ()

    # Valid condition field names (without suffix)
    VALID_CONDITION_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "batch_size",
        "seq_len",
        "seq_len_q",
        "seq_len_k",
        "head_dim",
        "num_heads",
        "num_kv_heads",
        "dtype",
        "device",
        "platform",
        "layout",
        "is_causal",
        "enable_gqa",
        "dropout_p",
        "is_cuda_graph_capturing",
        "requires_deterministic",
        "tp_size",
        "pp_size",
    })

    # Valid dtype values
    VALID_DTYPES: ClassVar[frozenset[str]] = frozenset({
        "float16", "float32", "float64", "bfloat16",
        "int8", "int16", "int32", "int64", "uint8", "bool",
    })

    # Valid platform values
    VALID_PLATFORMS: ClassVar[frozenset[str]] = frozenset({
        "cuda", "rocm", "cpu", "hpu", "xpu",
    })

    # Valid layout values
    VALID_LAYOUTS: ClassVar[frozenset[str]] = frozenset({
        "BSHD", "BHSD", "NHD", "HND",
    })

    def validate(self, config: dict[str, Any]) -> list[SchemaError]:
        """Validate config against schema.

        Args:
            config: Raw config dict to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[SchemaError] = []

        # Validate version
        version = config.get("version")
        if version is None:
            errors.append(SchemaError("version", "Missing required field"))
        elif version not in SUPPORTED_VERSIONS:
            errors.append(SchemaError(
                "version",
                f"Unsupported version, must be one of {sorted(SUPPORTED_VERSIONS)}",
                version,
            ))

        # Validate defaults
        if "defaults" in config:
            errors.extend(self._validate_defaults(config["defaults"]))

        # Validate dispatch_rules
        if "dispatch_rules" in config:
            rules = config["dispatch_rules"]
            if not isinstance(rules, list):
                errors.append(SchemaError(
                    "dispatch_rules", "Must be a list", type(rules).__name__
                ))
            else:
                for i, rule in enumerate(rules):
                    errors.extend(self._validate_dispatch_rule(rule, f"dispatch_rules[{i}]"))

        # Validate kernel_locks
        if "kernel_locks" in config:
            locks = config["kernel_locks"]
            if not isinstance(locks, dict):
                errors.append(SchemaError(
                    "kernel_locks", "Must be a dict", type(locks).__name__
                ))
            else:
                for op, kernel in locks.items():
                    if not isinstance(op, str):
                        errors.append(SchemaError(
                            f"kernel_locks.{op}", "Key must be string", type(op).__name__
                        ))
                    if not isinstance(kernel, str):
                        errors.append(SchemaError(
                            f"kernel_locks.{op}", "Value must be string", type(kernel).__name__
                        ))

        # Validate kernel_denies
        if "kernel_denies" in config:
            denies = config["kernel_denies"]
            if not isinstance(denies, list):
                errors.append(SchemaError(
                    "kernel_denies", "Must be a list", type(denies).__name__
                ))
            else:
                for i, pattern in enumerate(denies):
                    if not isinstance(pattern, str):
                        errors.append(SchemaError(
                            f"kernel_denies[{i}]", "Must be string", type(pattern).__name__
                        ))

        # Validate kernel_allows
        if "kernel_allows" in config:
            allows = config["kernel_allows"]
            if not isinstance(allows, list):
                errors.append(SchemaError(
                    "kernel_allows", "Must be a list", type(allows).__name__
                ))
            else:
                for i, pattern in enumerate(allows):
                    if not isinstance(pattern, str):
                        errors.append(SchemaError(
                            f"kernel_allows[{i}]", "Must be string", type(pattern).__name__
                        ))

        # Validate fallback_chains
        if "fallback_chains" in config:
            chains = config["fallback_chains"]
            if not isinstance(chains, dict):
                errors.append(SchemaError(
                    "fallback_chains", "Must be a dict", type(chains).__name__
                ))
            else:
                for op, kernels in chains.items():
                    if not isinstance(op, str):
                        errors.append(SchemaError(
                            f"fallback_chains.{op}", "Key must be string", type(op).__name__
                        ))
                    if not isinstance(kernels, list):
                        errors.append(SchemaError(
                            f"fallback_chains.{op}", "Value must be list", type(kernels).__name__
                        ))
                    else:
                        for i, k in enumerate(kernels):
                            if not isinstance(k, str):
                                errors.append(SchemaError(
                                    f"fallback_chains.{op}[{i}]",
                                    "Must be string",
                                    type(k).__name__,
                                ))

        return errors

    def _validate_defaults(self, defaults: Any) -> list[SchemaError]:
        """Validate defaults section."""
        errors: list[SchemaError] = []

        if not isinstance(defaults, dict):
            errors.append(SchemaError(
                "defaults", "Must be a dict", type(defaults).__name__
            ))
            return errors

        # Validate fallback_policy
        if "fallback_policy" in defaults:
            policy = defaults["fallback_policy"]
            if not isinstance(policy, str):
                errors.append(SchemaError(
                    "defaults.fallback_policy",
                    "Must be string",
                    type(policy).__name__,
                ))

        # Validate default_priority
        if "default_priority" in defaults:
            priority = defaults["default_priority"]
            if not isinstance(priority, int):
                errors.append(SchemaError(
                    "defaults.default_priority",
                    "Must be integer",
                    type(priority).__name__,
                ))
            elif not (0 <= priority <= 1000):
                errors.append(SchemaError(
                    "defaults.default_priority",
                    "Must be between 0 and 1000",
                    priority,
                ))

        return errors

    def _validate_dispatch_rule(self, rule: Any, path: str) -> list[SchemaError]:
        """Validate a single dispatch rule."""
        errors: list[SchemaError] = []

        if not isinstance(rule, dict):
            errors.append(SchemaError(path, "Must be a dict", type(rule).__name__))
            return errors

        # Required: operation
        if "operation" not in rule:
            errors.append(SchemaError(f"{path}.operation", "Missing required field"))
        elif not isinstance(rule["operation"], str):
            errors.append(SchemaError(
                f"{path}.operation",
                "Must be string",
                type(rule["operation"]).__name__,
            ))

        # Required: kernel
        if "kernel" not in rule:
            errors.append(SchemaError(f"{path}.kernel", "Missing required field"))
        elif not isinstance(rule["kernel"], str):
            errors.append(SchemaError(
                f"{path}.kernel",
                "Must be string",
                type(rule["kernel"]).__name__,
            ))

        # Optional: conditions
        if "conditions" in rule:
            errors.extend(self._validate_conditions(rule["conditions"], f"{path}.conditions"))

        # Optional: priority
        if "priority" in rule:
            priority = rule["priority"]
            if not isinstance(priority, int):
                errors.append(SchemaError(
                    f"{path}.priority",
                    "Must be integer",
                    type(priority).__name__,
                ))
            elif not (0 <= priority <= 1000):
                errors.append(SchemaError(
                    f"{path}.priority",
                    "Must be between 0 and 1000",
                    priority,
                ))

        return errors

    def _validate_conditions(self, conditions: Any, path: str) -> list[SchemaError]:
        """Validate rule conditions."""
        errors: list[SchemaError] = []

        if not isinstance(conditions, dict):
            errors.append(SchemaError(path, "Must be a dict", type(conditions).__name__))
            return errors

        for cond_name, cond_value in conditions.items():
            # Parse condition name
            base_field, op = self._parse_condition_name(cond_name)

            # Validate field name
            if base_field not in self.VALID_CONDITION_FIELDS:
                errors.append(SchemaError(
                    f"{path}.{cond_name}",
                    f"Unknown condition field. Valid fields: {sorted(self.VALID_CONDITION_FIELDS)}",
                    base_field,
                ))
                continue

            # Validate value based on field type and operator
            errors.extend(self._validate_condition_value(
                cond_value, base_field, op, f"{path}.{cond_name}"
            ))

        return errors

    def _parse_condition_name(self, name: str) -> tuple[str, ConditionOperator]:
        """Parse condition name into field and operator.

        Args:
            name: Condition name (e.g., "batch_size_gte", "dtype")

        Returns:
            Tuple of (base_field, operator).
        """
        for suffix, op in CONDITION_SUFFIXES.items():
            if name.endswith(suffix):
                return name[:-len(suffix)], op

        # Default to equality
        return name, ConditionOperator.EQ

    def _validate_condition_value(
        self,
        value: Any,
        field: str,
        op: ConditionOperator,
        path: str,
    ) -> list[SchemaError]:
        """Validate condition value based on field type and operator."""
        errors: list[SchemaError] = []

        # Handle list operators - require list
        if op in (ConditionOperator.IN, ConditionOperator.NOT_IN):
            if not isinstance(value, list):
                errors.append(SchemaError(
                    path,
                    f"Operator '{op.value}' requires list value",
                    type(value).__name__,
                ))
                return errors
            values = value
        # For dtypes and similar fields, a list indicates implicit IN operator
        elif isinstance(value, list):
            # List value with non-IN operator is treated as implicit IN for certain fields
            if field in ("dtype", "platform", "layout", "device"):
                values = value
            else:
                errors.append(SchemaError(
                    path,
                    f"Unexpected list value for operator '{op.value}'",
                    type(value).__name__,
                ))
                return errors
        else:
            values = [value]

        # Validate each value based on field type
        if field == "dtype":
            for v in values:
                if not isinstance(v, str):
                    errors.append(SchemaError(path, "dtype must be string", type(v).__name__))
                elif v not in self.VALID_DTYPES:
                    errors.append(SchemaError(
                        path,
                        f"Invalid dtype. Valid values: {sorted(self.VALID_DTYPES)}",
                        v,
                    ))

        elif field == "platform":
            for v in values:
                if not isinstance(v, str):
                    errors.append(SchemaError(path, "platform must be string", type(v).__name__))
                elif v not in self.VALID_PLATFORMS:
                    errors.append(SchemaError(
                        path,
                        f"Invalid platform. Valid values: {sorted(self.VALID_PLATFORMS)}",
                        v,
                    ))

        elif field == "layout":
            for v in values:
                if not isinstance(v, str):
                    errors.append(SchemaError(path, "layout must be string", type(v).__name__))
                elif v not in self.VALID_LAYOUTS:
                    errors.append(SchemaError(
                        path,
                        f"Invalid layout. Valid values: {sorted(self.VALID_LAYOUTS)}",
                        v,
                    ))

        elif field in ("is_causal", "enable_gqa", "is_cuda_graph_capturing", "requires_deterministic"):
            for v in values:
                if not isinstance(v, bool):
                    errors.append(SchemaError(path, f"{field} must be boolean", type(v).__name__))

        elif field in ("batch_size", "seq_len", "seq_len_q", "seq_len_k", "head_dim",
                       "num_heads", "num_kv_heads", "tp_size", "pp_size"):
            for v in values:
                if not isinstance(v, int):
                    errors.append(SchemaError(path, f"{field} must be integer", type(v).__name__))
                elif v < 0:
                    errors.append(SchemaError(path, f"{field} must be non-negative", v))

        elif field == "dropout_p":
            for v in values:
                if not isinstance(v, (int, float)):
                    errors.append(SchemaError(path, "dropout_p must be number", type(v).__name__))
                elif not (0.0 <= float(v) <= 1.0):
                    errors.append(SchemaError(path, "dropout_p must be between 0 and 1", v))

        elif field == "device":
            for v in values:
                if not isinstance(v, str):
                    errors.append(SchemaError(path, "device must be string", type(v).__name__))

        return errors


# ============================================================================
# Compiled Condition
# ============================================================================

@dataclass(frozen=True, slots=True)
class CompiledCondition:
    """Pre-compiled condition for efficient evaluation.

    Attributes:
        field: Context field to check
        operator: Comparison operator
        value: Value to compare against
        evaluator: Pre-compiled evaluator function
    """

    field: str
    operator: ConditionOperator
    value: Any
    evaluator: Callable[["SelectionContext"], bool]

    def matches(self, ctx: "SelectionContext") -> bool:
        """Evaluate condition against context.

        Args:
            ctx: Selection context to evaluate.

        Returns:
            True if condition matches.
        """
        return self.evaluator(ctx)


def _compile_condition(
    field: str,
    operator: ConditionOperator,
    value: Any,
) -> CompiledCondition:
    """Compile a condition into an optimized evaluator.

    Args:
        field: Context field name.
        operator: Comparison operator.
        value: Value to compare against.

    Returns:
        CompiledCondition with pre-compiled evaluator.
    """
    # Handle dtype string to torch.dtype conversion
    dtype_value = value
    if field == "dtype":
        dtype_value = _convert_dtype_value(value)

    # Handle platform string to Platform enum conversion
    platform_value = value
    if field == "platform":
        platform_value = _convert_platform_value(value)

    # Handle layout string to Layout enum conversion
    layout_value = value
    if field == "layout":
        layout_value = _convert_layout_value(value)

    # Create evaluator based on operator
    if operator == ConditionOperator.EQ:
        if field == "dtype":
            evaluator = lambda ctx, v=dtype_value: getattr(ctx, field, None) == v
        elif field == "platform":
            evaluator = lambda ctx, v=platform_value: ctx.device.platform == v
        elif field == "layout":
            evaluator = lambda ctx, v=layout_value: getattr(ctx, field, None) == v
        elif field == "device":
            evaluator = lambda ctx, v=value: str(ctx.device) == v
        else:
            evaluator = lambda ctx, f=field, v=value: getattr(ctx, f, None) == v

    elif operator == ConditionOperator.NE:
        if field == "dtype":
            evaluator = lambda ctx, v=dtype_value: getattr(ctx, field, None) != v
        elif field == "platform":
            evaluator = lambda ctx, v=platform_value: ctx.device.platform != v
        elif field == "layout":
            evaluator = lambda ctx, v=layout_value: getattr(ctx, field, None) != v
        else:
            evaluator = lambda ctx, f=field, v=value: getattr(ctx, f, None) != v

    elif operator == ConditionOperator.GT:
        evaluator = lambda ctx, f=field, v=value: _safe_gt(getattr(ctx, f, None), v)

    elif operator == ConditionOperator.GTE:
        evaluator = lambda ctx, f=field, v=value: _safe_gte(getattr(ctx, f, None), v)

    elif operator == ConditionOperator.LT:
        evaluator = lambda ctx, f=field, v=value: _safe_lt(getattr(ctx, f, None), v)

    elif operator == ConditionOperator.LTE:
        evaluator = lambda ctx, f=field, v=value: _safe_lte(getattr(ctx, f, None), v)

    elif operator == ConditionOperator.IN:
        if field == "dtype":
            converted = _convert_dtype_list(value)
            evaluator = lambda ctx, v=converted: getattr(ctx, field, None) in v
        elif field == "platform":
            converted = _convert_platform_list(value)
            evaluator = lambda ctx, v=converted: ctx.device.platform in v
        elif field == "layout":
            converted = _convert_layout_list(value)
            evaluator = lambda ctx, v=converted: getattr(ctx, field, None) in v
        else:
            value_set = frozenset(value)
            evaluator = lambda ctx, f=field, v=value_set: getattr(ctx, f, None) in v

    elif operator == ConditionOperator.NOT_IN:
        if field == "dtype":
            converted = _convert_dtype_list(value)
            evaluator = lambda ctx, v=converted: getattr(ctx, field, None) not in v
        elif field == "platform":
            converted = _convert_platform_list(value)
            evaluator = lambda ctx, v=converted: ctx.device.platform not in v
        elif field == "layout":
            converted = _convert_layout_list(value)
            evaluator = lambda ctx, v=converted: getattr(ctx, field, None) not in v
        else:
            value_set = frozenset(value)
            evaluator = lambda ctx, f=field, v=value_set: getattr(ctx, f, None) not in v

    elif operator == ConditionOperator.MATCH:
        evaluator = lambda ctx, f=field, v=value: _fnmatch_field(ctx, f, v)

    elif operator == ConditionOperator.REGEX:
        pattern = re.compile(value)
        evaluator = lambda ctx, f=field, p=pattern: _regex_match_field(ctx, f, p)

    else:
        # Default: always false
        evaluator = lambda ctx: False

    return CompiledCondition(
        field=field,
        operator=operator,
        value=value,
        evaluator=evaluator,
    )


def _safe_gt(a: Any, b: Any) -> bool:
    """Safe greater-than comparison."""
    if a is None:
        return False
    try:
        return a > b
    except TypeError:
        return False


def _safe_gte(a: Any, b: Any) -> bool:
    """Safe greater-than-or-equal comparison."""
    if a is None:
        return False
    try:
        return a >= b
    except TypeError:
        return False


def _safe_lt(a: Any, b: Any) -> bool:
    """Safe less-than comparison."""
    if a is None:
        return False
    try:
        return a < b
    except TypeError:
        return False


def _safe_lte(a: Any, b: Any) -> bool:
    """Safe less-than-or-equal comparison."""
    if a is None:
        return False
    try:
        return a <= b
    except TypeError:
        return False


def _fnmatch_field(ctx: "SelectionContext", field: str, pattern: str) -> bool:
    """Match field value against glob pattern."""
    value = getattr(ctx, field, None)
    if value is None:
        return False
    return fnmatch.fnmatch(str(value), pattern)


def _regex_match_field(ctx: "SelectionContext", field: str, pattern: re.Pattern) -> bool:
    """Match field value against regex pattern."""
    value = getattr(ctx, field, None)
    if value is None:
        return False
    return pattern.search(str(value)) is not None


def _convert_dtype_value(value: str | list[str]) -> Any:
    """Convert dtype string(s) to torch.dtype."""
    import torch
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if isinstance(value, str):
        return dtype_map.get(value, value)
    return value


def _convert_dtype_list(values: list[str]) -> frozenset:
    """Convert list of dtype strings to frozenset of torch.dtypes."""
    import torch
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    return frozenset(dtype_map.get(v, v) for v in values)


def _convert_platform_value(value: str) -> Any:
    """Convert platform string to Platform enum."""
    from layerzero.enums import Platform
    try:
        return Platform(value)
    except ValueError:
        return value


def _convert_platform_list(values: list[str]) -> frozenset:
    """Convert list of platform strings to frozenset of Platform enums."""
    from layerzero.enums import Platform
    result = []
    for v in values:
        try:
            result.append(Platform(v))
        except ValueError:
            result.append(v)
    return frozenset(result)


def _convert_layout_value(value: str) -> Any:
    """Convert layout string to Layout enum."""
    from layerzero.enums import Layout
    try:
        return Layout(value)
    except ValueError:
        return value


def _convert_layout_list(values: list[str]) -> frozenset:
    """Convert list of layout strings to frozenset of Layout enums."""
    from layerzero.enums import Layout
    result = []
    for v in values:
        try:
            result.append(Layout(v))
        except ValueError:
            result.append(v)
    return frozenset(result)


# ============================================================================
# Compiled Dispatch Rule
# ============================================================================

@dataclass(frozen=True, slots=True)
class CompiledDispatchRule:
    """Pre-compiled dispatch rule for efficient evaluation.

    Rules are compiled from YAML config during load to enable
    fast runtime evaluation without repeated parsing.

    Attributes:
        operation_pattern: Glob pattern for operation matching
        kernel_id: Target kernel to select
        priority: Rule priority (higher = evaluate first)
        conditions: Pre-compiled conditions (all must match)
        _operation_regex: Pre-compiled regex for operation pattern
    """

    operation_pattern: str
    kernel_id: str
    priority: int
    conditions: tuple[CompiledCondition, ...]
    _operation_regex: re.Pattern

    def matches_operation(self, operation: str) -> bool:
        """Check if rule's operation pattern matches operation.

        Args:
            operation: Operation identifier to check.

        Returns:
            True if pattern matches operation.
        """
        return self._operation_regex.match(operation) is not None

    def matches_context(self, ctx: "SelectionContext") -> bool:
        """Check if all conditions match context.

        Short-circuits on first non-match for efficiency.

        Args:
            ctx: Selection context to evaluate.

        Returns:
            True if all conditions match (empty = always match).
        """
        for condition in self.conditions:
            if not condition.matches(ctx):
                return False
        return True

    def matches(self, operation: str, ctx: "SelectionContext") -> bool:
        """Check if rule matches operation and context.

        Args:
            operation: Operation identifier.
            ctx: Selection context.

        Returns:
            True if rule matches both operation and context.
        """
        return self.matches_operation(operation) and self.matches_context(ctx)


def _glob_to_regex(pattern: str) -> re.Pattern:
    """Convert glob pattern to regex for efficient matching.

    Args:
        pattern: Glob pattern with * and ? wildcards.

    Returns:
        Compiled regex pattern.
    """
    # Escape special regex chars except * and ?
    escaped = re.escape(pattern)
    # Convert glob wildcards to regex
    regex_pattern = escaped.replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile(f"^{regex_pattern}$")


# ============================================================================
# Compiled Config
# ============================================================================

@dataclass(slots=True)
class CompiledConfig:
    """Pre-compiled dispatch configuration for runtime efficiency.

    Compiles all rules and patterns during load to enable
    O(1) lookups where possible and efficient rule evaluation.

    Thread-safe (immutable after construction).

    Attributes:
        version: Config schema version
        dispatch_rules: Compiled dispatch rules sorted by priority
        kernel_locks: Operation -> kernel_id map for hard locks
        kernel_denies: Compiled deny patterns
        kernel_allows: Compiled allow patterns (None = allow all)
        fallback_chains: Operation -> [kernel_ids] fallback chains
        fallback_policy: Default fallback kernel
        default_priority: Default priority for unspecified rules
        config_hash: Hash of config for cache invalidation
    """

    version: str
    dispatch_rules: tuple[CompiledDispatchRule, ...]
    kernel_locks: dict[str, str]  # operation pattern -> kernel_id
    kernel_denies: tuple[re.Pattern, ...]
    kernel_allows: tuple[re.Pattern, ...] | None  # None = allow all
    fallback_chains: dict[str, tuple[str, ...]]  # operation -> [kernel_ids]
    fallback_policy: str | None
    default_priority: int
    config_hash: str

    # Lock pattern -> regex cache
    _lock_regexes: dict[str, re.Pattern] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_locked_kernel(self, operation: str) -> str | None:
        """Get locked kernel for operation if any.

        Args:
            operation: Operation identifier.

        Returns:
            Locked kernel_id or None.
        """
        # Check exact match first (fast path)
        if operation in self.kernel_locks:
            return self.kernel_locks[operation]

        # Check pattern matches
        for pattern, kernel_id in self.kernel_locks.items():
            if "*" in pattern or "?" in pattern:
                regex = self._get_lock_regex(pattern)
                if regex.match(operation):
                    return kernel_id

        return None

    def _get_lock_regex(self, pattern: str) -> re.Pattern:
        """Get or compile regex for lock pattern."""
        with self._lock:
            if pattern not in self._lock_regexes:
                self._lock_regexes[pattern] = _glob_to_regex(pattern)
            return self._lock_regexes[pattern]

    def is_kernel_denied(self, kernel_id: str) -> bool:
        """Check if kernel is denied by any deny pattern.

        Args:
            kernel_id: Kernel identifier to check.

        Returns:
            True if kernel is denied.
        """
        for pattern in self.kernel_denies:
            if pattern.match(kernel_id):
                return True
        return False

    def is_kernel_allowed(self, kernel_id: str) -> bool:
        """Check if kernel is allowed by allow patterns.

        If no allow patterns defined, all kernels are allowed.
        If allow patterns defined, kernel must match at least one.

        Args:
            kernel_id: Kernel identifier to check.

        Returns:
            True if kernel is allowed.
        """
        if self.kernel_allows is None:
            return True

        for pattern in self.kernel_allows:
            if pattern.match(kernel_id):
                return True
        return False

    def get_matching_rules(
        self,
        operation: str,
        ctx: "SelectionContext",
    ) -> list[CompiledDispatchRule]:
        """Get all rules matching operation and context.

        Returns rules in priority order (highest first).

        Args:
            operation: Operation identifier.
            ctx: Selection context.

        Returns:
            List of matching rules sorted by priority (descending).
        """
        matching = []
        for rule in self.dispatch_rules:
            if rule.matches(operation, ctx):
                matching.append(rule)
        return matching

    def get_fallback_chain(self, operation: str) -> tuple[str, ...]:
        """Get fallback chain for operation.

        First checks exact match, then pattern matches, then uses
        default fallback policy if available.

        Args:
            operation: Operation identifier.

        Returns:
            Tuple of fallback kernel_ids (may be empty).
        """
        # Check exact match
        if operation in self.fallback_chains:
            return self.fallback_chains[operation]

        # Check pattern matches
        for pattern, chain in self.fallback_chains.items():
            if "*" in pattern or "?" in pattern:
                if fnmatch.fnmatch(operation, pattern):
                    return chain

        # Use default fallback policy
        if self.fallback_policy:
            return (self.fallback_policy,)

        return ()


def compile_config(raw: dict[str, Any]) -> CompiledConfig:
    """Compile raw config dict into optimized CompiledConfig.

    Args:
        raw: Raw config dict (should be validated first).

    Returns:
        CompiledConfig ready for efficient runtime use.
    """
    # Extract defaults
    defaults = raw.get("defaults", {})
    fallback_policy = defaults.get("fallback_policy")
    default_priority = defaults.get("default_priority", 50)

    # Compile dispatch rules
    rules: list[CompiledDispatchRule] = []
    for rule_dict in raw.get("dispatch_rules", []):
        operation_pattern = rule_dict["operation"]
        kernel_id = rule_dict["kernel"]
        priority = rule_dict.get("priority", default_priority)

        # Compile conditions
        conditions: list[CompiledCondition] = []
        for cond_name, cond_value in rule_dict.get("conditions", {}).items():
            # Parse condition name into field and operator
            base_field, op = _parse_condition_name(cond_name)
            # If value is a list and operator is EQ, convert to IN (implicit IN syntax)
            if isinstance(cond_value, list) and op == ConditionOperator.EQ:
                op = ConditionOperator.IN
            conditions.append(_compile_condition(base_field, op, cond_value))

        rules.append(CompiledDispatchRule(
            operation_pattern=operation_pattern,
            kernel_id=kernel_id,
            priority=priority,
            conditions=tuple(conditions),
            _operation_regex=_glob_to_regex(operation_pattern),
        ))

    # Sort rules by priority (highest first)
    rules.sort(key=lambda r: r.priority, reverse=True)

    # Process kernel locks
    kernel_locks = dict(raw.get("kernel_locks", {}))

    # Compile deny patterns
    deny_patterns: list[re.Pattern] = []
    for pattern in raw.get("kernel_denies", []):
        deny_patterns.append(_glob_to_regex(pattern))

    # Compile allow patterns (None if not specified)
    allow_patterns: list[re.Pattern] | None = None
    if "kernel_allows" in raw and raw["kernel_allows"]:
        allow_patterns = []
        for pattern in raw["kernel_allows"]:
            allow_patterns.append(_glob_to_regex(pattern))

    # Process fallback chains
    fallback_chains: dict[str, tuple[str, ...]] = {}
    for op, chain in raw.get("fallback_chains", {}).items():
        fallback_chains[op] = tuple(chain)

    # Compute config hash for cache invalidation
    config_hash = _compute_config_hash(raw)

    return CompiledConfig(
        version=raw.get("version", SCHEMA_VERSION),
        dispatch_rules=tuple(rules),
        kernel_locks=kernel_locks,
        kernel_denies=tuple(deny_patterns),
        kernel_allows=tuple(allow_patterns) if allow_patterns is not None else None,
        fallback_chains=fallback_chains,
        fallback_policy=fallback_policy,
        default_priority=default_priority,
        config_hash=config_hash,
    )


def _parse_condition_name(name: str) -> tuple[str, ConditionOperator]:
    """Parse condition name into field and operator."""
    for suffix, op in CONDITION_SUFFIXES.items():
        if name.endswith(suffix):
            return name[:-len(suffix)], op
    return name, ConditionOperator.EQ


def _compute_config_hash(config: dict[str, Any]) -> str:
    """Compute deterministic hash of config for cache invalidation."""
    json_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


# ============================================================================
# Rule Evaluation Cache
# ============================================================================

@dataclass(slots=True)
class RuleEvaluationResult:
    """Cached result of rule evaluation.

    Attributes:
        kernel_id: Selected kernel_id (or None if no match)
        rule_priority: Priority of matching rule
        timestamp: Creation time for TTL expiration
    """

    kernel_id: str | None
    rule_priority: int
    timestamp: float


class RuleEvaluationCache:
    """LRU cache for rule evaluation results.

    Caches results of rule evaluation to avoid repeated condition
    evaluation for identical contexts.

    Thread-safe implementation using a lock.

    Attributes:
        max_size: Maximum cache entries
        ttl_seconds: Time-to-live for cache entries
    """

    __slots__ = ("_max_size", "_ttl_seconds", "_cache", "_lock", "_hits", "_misses")

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: float = 60.0,
    ) -> None:
        """Initialize cache.

        Args:
            max_size: Maximum number of cached entries.
            ttl_seconds: Entry TTL in seconds.
        """
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: dict[str, RuleEvaluationResult] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> RuleEvaluationResult | None:
        """Get cached result.

        Args:
            key: Cache key.

        Returns:
            Cached result or None if not found/expired.
        """
        with self._lock:
            result = self._cache.get(key)
            if result is None:
                self._misses += 1
                return None

            # Check TTL
            if time.monotonic() - result.timestamp > self._ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            return result

    def put(self, key: str, kernel_id: str | None, priority: int) -> None:
        """Store result in cache.

        Args:
            key: Cache key.
            kernel_id: Selected kernel_id.
            priority: Rule priority.
        """
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size:
                self._evict_oldest()

            self._cache[key] = RuleEvaluationResult(
                kernel_id=kernel_id,
                rule_priority=priority,
                timestamp=time.monotonic(),
            )

    def invalidate(self, config_hash: str) -> None:
        """Invalidate all entries (config changed).

        Args:
            config_hash: New config hash (unused, clears all).
        """
        with self._lock:
            self._cache.clear()

    def _evict_oldest(self) -> None:
        """Evict oldest entry (called while holding lock)."""
        if not self._cache:
            return

        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        del self._cache[oldest_key]

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, size, hit_rate.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "max_size": self._max_size,
                "hit_rate": hit_rate,
            }

    def clear(self) -> None:
        """Clear all cache entries and reset stats."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# ============================================================================
# Config-Driven Dispatcher
# ============================================================================

class ConfigDrivenDispatcher(BaseDispatcher):
    """YAML config-driven kernel dispatcher.

    Enables ops-controlled deployment via declarative configuration.
    Supports:
    - Rule-based kernel selection with conditions
    - Kernel locks (force specific kernel for operation)
    - Whitelisting/blacklisting via patterns
    - Priority overrides
    - Fallback chains

    Thread-safe implementation suitable for production use.

    Typical overhead: ~100ns for cached lookups, ~1-10us for rule evaluation.
    """

    __slots__ = (
        "_compiled_config",
        "_rule_cache",
        "_kernel_registry",
        "_config_path",
        "_last_config_mtime",
    )

    def __init__(
        self,
        config: DispatchConfig,
        executor: "KernelExecutor | None" = None,
        kernel_registry: "KernelRegistry | None" = None,
        yaml_config: dict[str, Any] | None = None,
        config_path: Path | str | None = None,
    ) -> None:
        """Initialize config-driven dispatcher.

        Args:
            config: Base dispatch configuration.
            executor: Kernel executor instance.
            kernel_registry: Kernel registry for kernel lookup.
            yaml_config: Pre-loaded YAML config dict.
            config_path: Path to YAML config file (for hot-reload).

        Raises:
            ConfigurationError: If config validation fails.
        """
        super().__init__(config, executor)

        self._kernel_registry = kernel_registry
        self._config_path = Path(config_path) if config_path else None
        self._last_config_mtime: float = 0.0

        # Load and compile config
        if yaml_config is not None:
            raw_config = yaml_config
        elif self._config_path is not None and self._config_path.exists():
            raw_config = self._load_yaml_config(self._config_path)
            self._last_config_mtime = self._config_path.stat().st_mtime
        else:
            # Empty config
            raw_config = {"version": SCHEMA_VERSION}

        # Validate and compile
        self._compiled_config = self._validate_and_compile(raw_config)

        # Initialize rule evaluation cache
        self._rule_cache = RuleEvaluationCache(
            max_size=config.cache_size,
            ttl_seconds=config.cache_ttl_seconds,
        )

        logger.info(
            "ConfigDrivenDispatcher initialized with %d rules, %d locks, config_hash=%s",
            len(self._compiled_config.dispatch_rules),
            len(self._compiled_config.kernel_locks),
            self._compiled_config.config_hash,
        )

    @property
    def mode(self) -> DispatchMode:
        """Get dispatch mode."""
        return DispatchMode.CONFIG

    @property
    def compiled_config(self) -> CompiledConfig:
        """Get compiled configuration."""
        return self._compiled_config

    def dispatch(
        self,
        operation: str,
        inputs: dict[str, "torch.Tensor"],
        context: "SelectionContext | None" = None,
        **kwargs: Any,
    ) -> DispatchResult:
        """Dispatch operation to kernel selected by config rules.

        Selection order:
        1. Check kernel locks (operation -> kernel_id)
        2. Check rule cache (context-based)
        3. Evaluate dispatch rules (highest priority first)
        4. Apply allow/deny filters
        5. Use fallback chain if no rule matches

        Args:
            operation: Operation identifier (e.g., "attention.causal").
            inputs: Dictionary of named input tensors.
            context: Optional pre-built selection context.
            **kwargs: Additional operation-specific arguments.

        Returns:
            DispatchResult with output tensor and metadata.

        Raises:
            DispatchError: If dispatch fails.
            FallbackChainExhaustedError: If all fallbacks fail.
        """
        start_time = time.perf_counter_ns()

        # Build context if not provided
        if context is None:
            context = self._build_context(operation, inputs, **kwargs)

        selection_start = time.perf_counter_ns()

        # 1. Check kernel locks (highest priority)
        locked_kernel = self._compiled_config.get_locked_kernel(operation)
        if locked_kernel is not None:
            kernel_spec = self._get_kernel_spec(locked_kernel)
            if kernel_spec is not None:
                selection_ns = time.perf_counter_ns() - selection_start
                return self._execute_kernel(
                    kernel_spec=kernel_spec,
                    inputs=inputs,
                    context=context,
                    selection_ns=selection_ns,
                    start_time=start_time,
                    cached=False,
                    **kwargs,
                )
            else:
                logger.warning(
                    "Locked kernel '%s' not found in registry for operation '%s'",
                    locked_kernel, operation,
                )

        # 2. Check rule cache
        cache_key = self._make_cache_key(operation, context)
        cached_result = self._rule_cache.get(cache_key)

        if cached_result is not None and cached_result.kernel_id is not None:
            kernel_spec = self._get_kernel_spec(cached_result.kernel_id)
            if kernel_spec is not None:
                selection_ns = time.perf_counter_ns() - selection_start
                return self._execute_kernel(
                    kernel_spec=kernel_spec,
                    inputs=inputs,
                    context=context,
                    selection_ns=selection_ns,
                    start_time=start_time,
                    cached=True,
                    **kwargs,
                )

        # 3. Evaluate dispatch rules
        matching_rules = self._compiled_config.get_matching_rules(operation, context)

        for rule in matching_rules:
            kernel_id = rule.kernel_id

            # Check allow/deny filters
            if self._compiled_config.is_kernel_denied(kernel_id):
                logger.debug(
                    "Kernel '%s' denied by config for operation '%s'",
                    kernel_id, operation,
                )
                continue

            if not self._compiled_config.is_kernel_allowed(kernel_id):
                logger.debug(
                    "Kernel '%s' not in allow list for operation '%s'",
                    kernel_id, operation,
                )
                continue

            # Get kernel spec
            kernel_spec = self._get_kernel_spec(kernel_id)
            if kernel_spec is None:
                logger.debug(
                    "Kernel '%s' not found in registry for operation '%s'",
                    kernel_id, operation,
                )
                continue

            # Cache result
            self._rule_cache.put(cache_key, kernel_id, rule.priority)

            selection_ns = time.perf_counter_ns() - selection_start
            return self._execute_kernel(
                kernel_spec=kernel_spec,
                inputs=inputs,
                context=context,
                selection_ns=selection_ns,
                start_time=start_time,
                cached=False,
                **kwargs,
            )

        # 4. Use fallback chain
        fallback_chain = self._compiled_config.get_fallback_chain(operation)
        attempted_kernels: list[str] = []
        errors: list[Exception] = []

        for fallback_kernel_id in fallback_chain:
            # Check allow/deny
            if self._compiled_config.is_kernel_denied(fallback_kernel_id):
                continue
            if not self._compiled_config.is_kernel_allowed(fallback_kernel_id):
                continue

            kernel_spec = self._get_kernel_spec(fallback_kernel_id)
            if kernel_spec is None:
                continue

            attempted_kernels.append(fallback_kernel_id)

            try:
                selection_ns = time.perf_counter_ns() - selection_start
                return self._execute_kernel(
                    kernel_spec=kernel_spec,
                    inputs=inputs,
                    context=context,
                    selection_ns=selection_ns,
                    start_time=start_time,
                    cached=False,
                    fallback_used=True,
                    **kwargs,
                )
            except KernelExecutionError as e:
                errors.append(e)
                logger.warning(
                    "Fallback kernel '%s' failed for operation '%s': %s",
                    fallback_kernel_id, operation, e,
                )
                continue

        # No kernel found
        raise FallbackChainExhaustedError(
            operation=operation,
            attempted_kernels=attempted_kernels,
            errors=errors,
        )

    def get_kernel_for_operation(
        self,
        operation: str,
        context: "SelectionContext",
    ) -> "KernelSpec":
        """Get the kernel that would be used for an operation.

        Useful for inspection and debugging without executing.

        Args:
            operation: Operation identifier.
            context: Selection context.

        Returns:
            KernelSpec that would be selected.

        Raises:
            NoKernelAvailableError: If no kernel matches.
        """
        from layerzero.selection.engine import NoKernelAvailableError

        # Check kernel locks
        locked_kernel = self._compiled_config.get_locked_kernel(operation)
        if locked_kernel is not None:
            kernel_spec = self._get_kernel_spec(locked_kernel)
            if kernel_spec is not None:
                return kernel_spec

        # Evaluate dispatch rules
        matching_rules = self._compiled_config.get_matching_rules(operation, context)

        for rule in matching_rules:
            kernel_id = rule.kernel_id

            if self._compiled_config.is_kernel_denied(kernel_id):
                continue
            if not self._compiled_config.is_kernel_allowed(kernel_id):
                continue

            kernel_spec = self._get_kernel_spec(kernel_id)
            if kernel_spec is not None:
                return kernel_spec

        # Check fallback chain
        fallback_chain = self._compiled_config.get_fallback_chain(operation)
        for fallback_kernel_id in fallback_chain:
            if self._compiled_config.is_kernel_denied(fallback_kernel_id):
                continue
            if not self._compiled_config.is_kernel_allowed(fallback_kernel_id):
                continue

            kernel_spec = self._get_kernel_spec(fallback_kernel_id)
            if kernel_spec is not None:
                return kernel_spec

        raise NoKernelAvailableError(operation)

    def reload_config(self, config_path: Path | str | None = None) -> bool:
        """Reload configuration from file.

        Thread-safe hot-reload of configuration.

        Args:
            config_path: Path to config file (uses existing if None).

        Returns:
            True if config was reloaded, False if unchanged.

        Raises:
            ConfigurationError: If new config is invalid.
        """
        path = Path(config_path) if config_path else self._config_path
        if path is None or not path.exists():
            logger.warning("No config path available for reload")
            return False

        # Check if file modified
        current_mtime = path.stat().st_mtime
        if current_mtime == self._last_config_mtime:
            return False

        logger.info("Reloading config from %s", path)

        # Load and validate new config
        raw_config = self._load_yaml_config(path)
        new_compiled = self._validate_and_compile(raw_config)

        # Check if config actually changed
        if new_compiled.config_hash == self._compiled_config.config_hash:
            self._last_config_mtime = current_mtime
            return False

        # Atomic swap
        old_hash = self._compiled_config.config_hash
        self._compiled_config = new_compiled
        self._last_config_mtime = current_mtime

        # Invalidate cache
        self._rule_cache.invalidate(new_compiled.config_hash)

        logger.info(
            "Config reloaded: %s -> %s, %d rules",
            old_hash, new_compiled.config_hash, len(new_compiled.dispatch_rules),
        )

        return True

    def update_yaml_config(self, yaml_config: dict[str, Any]) -> None:
        """Update configuration from dict (for programmatic updates).

        Args:
            yaml_config: New config dict.

        Raises:
            ConfigurationError: If config is invalid.
        """
        new_compiled = self._validate_and_compile(yaml_config)

        old_hash = self._compiled_config.config_hash
        self._compiled_config = new_compiled
        self._rule_cache.invalidate(new_compiled.config_hash)

        logger.info(
            "Config updated programmatically: %s -> %s",
            old_hash, new_compiled.config_hash,
        )

    def get_telemetry(self) -> dict[str, Any]:
        """Get dispatcher telemetry.

        Returns:
            Dict with cache stats and config info.
        """
        base_telemetry = super().get_telemetry()
        base_telemetry.update({
            "config_hash": self._compiled_config.config_hash,
            "rule_count": len(self._compiled_config.dispatch_rules),
            "lock_count": len(self._compiled_config.kernel_locks),
            "deny_pattern_count": len(self._compiled_config.kernel_denies),
            "allow_pattern_count": (
                len(self._compiled_config.kernel_allows)
                if self._compiled_config.kernel_allows else 0
            ),
            "fallback_chain_count": len(self._compiled_config.fallback_chains),
            "rule_cache": self._rule_cache.stats(),
        })
        return base_telemetry

    def explain_selection(
        self,
        operation: str,
        context: "SelectionContext",
    ) -> dict[str, Any]:
        """Explain kernel selection for debugging.

        Args:
            operation: Operation identifier.
            context: Selection context.

        Returns:
            Dict with selection explanation.
        """
        explanation: dict[str, Any] = {
            "operation": operation,
            "config_hash": self._compiled_config.config_hash,
            "selection_path": [],
            "selected_kernel": None,
        }

        # Check locks
        locked_kernel = self._compiled_config.get_locked_kernel(operation)
        if locked_kernel is not None:
            explanation["selection_path"].append({
                "step": "kernel_lock",
                "pattern": operation,
                "kernel": locked_kernel,
            })
            kernel_spec = self._get_kernel_spec(locked_kernel)
            if kernel_spec is not None:
                explanation["selected_kernel"] = locked_kernel
                explanation["reason"] = "kernel_lock"
                return explanation
            else:
                explanation["selection_path"][-1]["error"] = "kernel_not_found"

        # Evaluate rules
        for rule in self._compiled_config.dispatch_rules:
            rule_info: dict[str, Any] = {
                "step": "dispatch_rule",
                "operation_pattern": rule.operation_pattern,
                "kernel": rule.kernel_id,
                "priority": rule.priority,
            }

            if not rule.matches_operation(operation):
                rule_info["result"] = "operation_mismatch"
                explanation["selection_path"].append(rule_info)
                continue

            if not rule.matches_context(context):
                rule_info["result"] = "condition_mismatch"
                # Include which conditions failed
                failed_conditions = []
                for cond in rule.conditions:
                    if not cond.matches(context):
                        failed_conditions.append({
                            "field": cond.field,
                            "operator": cond.operator.value,
                            "expected": cond.value,
                            "actual": getattr(context, cond.field, None),
                        })
                rule_info["failed_conditions"] = failed_conditions
                explanation["selection_path"].append(rule_info)
                continue

            # Check allow/deny
            if self._compiled_config.is_kernel_denied(rule.kernel_id):
                rule_info["result"] = "kernel_denied"
                explanation["selection_path"].append(rule_info)
                continue

            if not self._compiled_config.is_kernel_allowed(rule.kernel_id):
                rule_info["result"] = "kernel_not_allowed"
                explanation["selection_path"].append(rule_info)
                continue

            # Check if kernel exists
            kernel_spec = self._get_kernel_spec(rule.kernel_id)
            if kernel_spec is None:
                rule_info["result"] = "kernel_not_found"
                explanation["selection_path"].append(rule_info)
                continue

            rule_info["result"] = "selected"
            explanation["selection_path"].append(rule_info)
            explanation["selected_kernel"] = rule.kernel_id
            explanation["reason"] = "dispatch_rule"
            return explanation

        # Fallback chain
        fallback_chain = self._compiled_config.get_fallback_chain(operation)
        for fallback_kernel_id in fallback_chain:
            fallback_info: dict[str, Any] = {
                "step": "fallback_chain",
                "kernel": fallback_kernel_id,
            }

            if self._compiled_config.is_kernel_denied(fallback_kernel_id):
                fallback_info["result"] = "kernel_denied"
                explanation["selection_path"].append(fallback_info)
                continue

            if not self._compiled_config.is_kernel_allowed(fallback_kernel_id):
                fallback_info["result"] = "kernel_not_allowed"
                explanation["selection_path"].append(fallback_info)
                continue

            kernel_spec = self._get_kernel_spec(fallback_kernel_id)
            if kernel_spec is None:
                fallback_info["result"] = "kernel_not_found"
                explanation["selection_path"].append(fallback_info)
                continue

            fallback_info["result"] = "selected"
            explanation["selection_path"].append(fallback_info)
            explanation["selected_kernel"] = fallback_kernel_id
            explanation["reason"] = "fallback_chain"
            return explanation

        explanation["reason"] = "no_kernel_available"
        return explanation

    def _load_yaml_config(self, path: Path) -> dict[str, Any]:
        """Load YAML config file.

        Args:
            path: Path to YAML file.

        Returns:
            Parsed YAML dict.

        Raises:
            ConfigurationError: If file cannot be loaded.
        """
        try:
            import yaml
        except ImportError as e:
            raise ConfigurationError(
                "PyYAML not installed. Install with: pip install pyyaml"
            ) from e

        try:
            with open(path, "r") as f:
                content = yaml.safe_load(f)
            return content if content is not None else {"version": SCHEMA_VERSION}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e
        except OSError as e:
            raise ConfigurationError(f"Cannot read config file {path}: {e}") from e

    def _validate_and_compile(self, raw_config: dict[str, Any]) -> CompiledConfig:
        """Validate and compile raw config.

        Args:
            raw_config: Raw config dict.

        Returns:
            CompiledConfig.

        Raises:
            ConfigurationError: If validation fails.
        """
        schema = ConfigSchema()
        errors = schema.validate(raw_config)

        if errors:
            error_messages = "\n  ".join(str(e) for e in errors)
            raise ConfigurationError(
                f"Config validation failed with {len(errors)} error(s):\n  {error_messages}"
            )

        return compile_config(raw_config)

    def _get_kernel_spec(self, kernel_id: str) -> "KernelSpec | None":
        """Get kernel spec from registry.

        Args:
            kernel_id: Kernel identifier.

        Returns:
            KernelSpec or None if not found.
        """
        if self._kernel_registry is None:
            return None
        return self._kernel_registry.get(kernel_id)

    def _build_context(
        self,
        operation: str,
        inputs: dict[str, "torch.Tensor"],
        **kwargs: Any,
    ) -> "SelectionContext":
        """Build selection context from inputs.

        Args:
            operation: Operation identifier.
            inputs: Input tensors.
            **kwargs: Additional context parameters.

        Returns:
            SelectionContext for kernel selection.
        """
        from layerzero.enums import Layout, OpKind
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.models.selection_context import SelectionContext

        # Get first tensor for device/dtype detection
        first_tensor = next(iter(inputs.values()))
        device = DeviceSpec.detect(str(first_tensor.device))

        # Detect batch size
        batch_size = first_tensor.shape[0] if first_tensor.ndim > 0 else 1

        # Build context with provided kwargs
        return SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation=operation,
            dtype=first_tensor.dtype,
            batch_size=kwargs.get("batch_size", batch_size),
            seq_len_q=kwargs.get("seq_len_q") or kwargs.get("seq_len"),
            seq_len_k=kwargs.get("seq_len_k"),
            num_heads=kwargs.get("num_heads"),
            num_kv_heads=kwargs.get("num_kv_heads"),
            head_dim=kwargs.get("head_dim"),
            layout=kwargs.get("layout", Layout.BSHD),
            is_causal=kwargs.get("is_causal", kwargs.get("causal", False)),
            dropout_p=kwargs.get("dropout_p", 0.0),
            enable_gqa=kwargs.get("enable_gqa", False),
            is_cuda_graph_capturing=kwargs.get("is_cuda_graph_capturing", False),
            requires_deterministic=kwargs.get("requires_deterministic", False),
            tp_size=kwargs.get("tp_size", 1),
            pp_size=kwargs.get("pp_size", 1),
        )

    def _make_cache_key(self, operation: str, context: "SelectionContext") -> str:
        """Generate cache key for rule evaluation.

        Args:
            operation: Operation identifier.
            context: Selection context.

        Returns:
            Cache key string.
        """
        # Use context's cache_key which includes all relevant fields
        return f"{self._compiled_config.config_hash}:{operation}:{context.cache_key()}"

    def _execute_kernel(
        self,
        kernel_spec: "KernelSpec",
        inputs: dict[str, "torch.Tensor"],
        context: "SelectionContext",
        selection_ns: int,
        start_time: int,
        cached: bool,
        fallback_used: bool = False,
        **kwargs: Any,
    ) -> DispatchResult:
        """Execute kernel and build result.

        Args:
            kernel_spec: Kernel to execute.
            inputs: Input tensors.
            context: Selection context.
            selection_ns: Selection time in nanoseconds.
            start_time: Dispatch start time (perf_counter_ns).
            cached: Whether selection was cached.
            fallback_used: Whether this is a fallback kernel.
            **kwargs: Additional kernel arguments.

        Returns:
            DispatchResult with output tensor and metadata.
        """
        from layerzero.dispatch.executor import KernelExecutorImpl

        # Get or create executor
        executor = self._executor
        if executor is None:
            executor = KernelExecutorImpl()

        # Execute kernel with timing
        exec_start = time.perf_counter_ns()
        output, exec_ns = executor.execute_with_timing(kernel_spec, inputs, **kwargs)

        total_ns = time.perf_counter_ns() - start_time

        timing = DispatchTiming(
            selection_ns=selection_ns,
            pre_transform_ns=0,
            execution_ns=exec_ns,
            post_transform_ns=0,
            total_ns=total_ns,
        )

        return DispatchResult(
            output=output,
            kernel_id=kernel_spec.kernel_id,
            kernel_spec=kernel_spec,
            timing=timing,
            mode=DispatchMode.CONFIG,
            cached=cached,
            fallback_used=fallback_used,
            fallback_reason="no_rule_match" if fallback_used else None,
        )


# ============================================================================
# Factory Function
# ============================================================================

def create_config_dispatcher(
    config_path: Path | str | None = None,
    yaml_config: dict[str, Any] | None = None,
    kernel_registry: "KernelRegistry | None" = None,
    cache_size: int = 10000,
    cache_ttl_seconds: float = 60.0,
) -> ConfigDrivenDispatcher:
    """Create a config-driven dispatcher.

    Convenience factory function for creating ConfigDrivenDispatcher.

    Args:
        config_path: Path to YAML config file.
        yaml_config: Pre-loaded config dict (overrides config_path).
        kernel_registry: Kernel registry for kernel lookup.
        cache_size: Maximum cache entries.
        cache_ttl_seconds: Cache entry TTL.

    Returns:
        Configured ConfigDrivenDispatcher instance.

    Raises:
        ConfigurationError: If config is invalid.
    """
    dispatch_config = DispatchConfig(
        mode=DispatchMode.CONFIG,
        enable_cache=True,
        cache_size=cache_size,
        cache_ttl_seconds=cache_ttl_seconds,
    )

    return ConfigDrivenDispatcher(
        config=dispatch_config,
        kernel_registry=kernel_registry,
        yaml_config=yaml_config,
        config_path=config_path,
    )


# ============================================================================
# Integration with Policy System
# ============================================================================

def config_to_policy(compiled_config: CompiledConfig) -> "Policy":
    """Convert compiled config to Policy for integration.

    Enables using ConfigDrivenDispatcher's config with
    the existing PolicyEngine and SelectionEngine.

    Args:
        compiled_config: Compiled dispatch config.

    Returns:
        Policy object with equivalent rules.
    """
    from layerzero.policy.policy import Policy
    from layerzero.policy.rule import Condition, ConditionOp, Rule, RuleType

    locks: list[Rule] = []
    denies: list[Rule] = []
    allows: list[Rule] = []
    boosts: list[Rule] = []

    # Convert kernel locks
    for operation_pattern, kernel_id in compiled_config.kernel_locks.items():
        locks.append(Rule(
            rule_type=RuleType.LOCK,
            target=operation_pattern,
            conditions=(),
            value=kernel_id,
        ))

    # Convert deny patterns
    for pattern in compiled_config.kernel_denies:
        denies.append(Rule(
            rule_type=RuleType.DENY,
            target=pattern.pattern.replace("^", "").replace("$", "").replace(".*", "*").replace(".", "?"),
            conditions=(),
            value=None,
        ))

    # Convert allow patterns
    if compiled_config.kernel_allows is not None:
        for pattern in compiled_config.kernel_allows:
            allows.append(Rule(
                rule_type=RuleType.ALLOW,
                target=pattern.pattern.replace("^", "").replace("$", "").replace(".*", "*").replace(".", "?"),
                conditions=(),
                value=None,
            ))

    # Convert dispatch rules to boost rules
    # Higher priority rules get higher boost
    for rule in compiled_config.dispatch_rules:
        # Convert conditions
        conditions: list[Condition] = []
        for cc in rule.conditions:
            # Map operator
            op_map = {
                ConditionOperator.EQ: ConditionOp.EQ,
                ConditionOperator.NE: ConditionOp.NE,
                ConditionOperator.GT: ConditionOp.GT,
                ConditionOperator.GTE: ConditionOp.GE,
                ConditionOperator.LT: ConditionOp.LT,
                ConditionOperator.LTE: ConditionOp.LE,
                ConditionOperator.IN: ConditionOp.IN,
                ConditionOperator.NOT_IN: ConditionOp.NOT_IN,
                ConditionOperator.MATCH: ConditionOp.MATCH,
                ConditionOperator.REGEX: ConditionOp.MATCH,
            }
            conditions.append(Condition(
                field=cc.field,
                op=op_map.get(cc.operator, ConditionOp.EQ),
                value=cc.value,
            ))

        boosts.append(Rule(
            rule_type=RuleType.BOOST_ADD,
            target=rule.kernel_id,
            conditions=tuple(conditions),
            value=rule.priority - compiled_config.default_priority,
        ))

    return Policy(
        version=compiled_config.version,
        locks=tuple(locks),
        allows=tuple(allows),
        denies=tuple(denies),
        boosts=tuple(boosts),
    )
