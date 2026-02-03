"""
Test suite for Rule and Condition classes.

Tests rule matching and condition evaluation.
Following TDD methodology - tests define expected behavior.
"""
import pytest
import torch

from layerzero.models.device_spec import DeviceSpec
from layerzero.models.selection_context import SelectionContext
from layerzero.enums import OpKind, Layout


def make_context(**kwargs) -> SelectionContext:
    """Create test SelectionContext with defaults."""
    defaults = {
        "device": DeviceSpec.cpu(),
        "op_kind": OpKind.TENSOR,
        "operation": "attention.causal",
        "dtype": torch.float16,
        "batch_size": 32,
        "seq_len_q": 512,
        "seq_len_k": 512,
        "num_heads": 8,
        "num_kv_heads": 8,
        "head_dim": 64,
    }
    defaults.update(kwargs)
    return SelectionContext(**defaults)


class TestConditionCreation:
    """Test Condition construction."""

    def test_condition_creation(self):
        """Condition can be created."""
        from layerzero.policy.rule import Condition, ConditionOp

        cond = Condition(
            field="head_dim",
            op=ConditionOp.GE,
            value=64,
        )
        assert cond.field == "head_dim"
        assert cond.op == ConditionOp.GE
        assert cond.value == 64

    def test_condition_is_frozen(self):
        """Condition is immutable."""
        from layerzero.policy.rule import Condition, ConditionOp

        cond = Condition("head_dim", ConditionOp.EQ, 64)
        with pytest.raises(AttributeError):
            cond.value = 128  # type: ignore


class TestConditionMatching:
    """Test Condition.matches()."""

    def test_condition_eq_matches(self):
        """EQ condition matches equal values."""
        from layerzero.policy.rule import Condition, ConditionOp

        cond = Condition("head_dim", ConditionOp.EQ, 64)
        ctx = make_context(head_dim=64)
        assert cond.matches(ctx) is True

    def test_condition_eq_no_match(self):
        """EQ condition doesn't match unequal values."""
        from layerzero.policy.rule import Condition, ConditionOp

        cond = Condition("head_dim", ConditionOp.EQ, 64)
        ctx = make_context(head_dim=128)
        assert cond.matches(ctx) is False

    def test_condition_gt_matches(self):
        """GT condition matches greater values."""
        from layerzero.policy.rule import Condition, ConditionOp

        cond = Condition("head_dim", ConditionOp.GT, 32)
        ctx = make_context(head_dim=64)
        assert cond.matches(ctx) is True

    def test_condition_gt_no_match(self):
        """GT condition doesn't match equal or lesser values."""
        from layerzero.policy.rule import Condition, ConditionOp

        cond = Condition("head_dim", ConditionOp.GT, 64)
        ctx = make_context(head_dim=64)
        assert cond.matches(ctx) is False

    def test_condition_ge_matches(self):
        """GE condition matches equal or greater values."""
        from layerzero.policy.rule import Condition, ConditionOp

        cond = Condition("seq_len_q", ConditionOp.GE, 512)
        ctx = make_context(seq_len_q=512)
        assert cond.matches(ctx) is True

    def test_condition_lt_matches(self):
        """LT condition matches lesser values."""
        from layerzero.policy.rule import Condition, ConditionOp

        cond = Condition("batch_size", ConditionOp.LT, 64)
        ctx = make_context(batch_size=32)
        assert cond.matches(ctx) is True

    def test_condition_le_matches(self):
        """LE condition matches equal or lesser values."""
        from layerzero.policy.rule import Condition, ConditionOp

        cond = Condition("batch_size", ConditionOp.LE, 32)
        ctx = make_context(batch_size=32)
        assert cond.matches(ctx) is True

    def test_condition_ne_matches(self):
        """NE condition matches unequal values."""
        from layerzero.policy.rule import Condition, ConditionOp

        cond = Condition("head_dim", ConditionOp.NE, 128)
        ctx = make_context(head_dim=64)
        assert cond.matches(ctx) is True

    def test_condition_match_glob(self):
        """MATCH condition matches glob patterns."""
        from layerzero.policy.rule import Condition, ConditionOp

        cond = Condition("operation", ConditionOp.MATCH, "attention.*")
        ctx = make_context(operation="attention.causal")
        assert cond.matches(ctx) is True

    def test_condition_match_glob_no_match(self):
        """MATCH condition doesn't match non-matching patterns."""
        from layerzero.policy.rule import Condition, ConditionOp

        cond = Condition("operation", ConditionOp.MATCH, "attention.*")
        ctx = make_context(operation="norm.rms")
        assert cond.matches(ctx) is False

    def test_condition_missing_field(self):
        """Condition returns False for missing field."""
        from layerzero.policy.rule import Condition, ConditionOp

        cond = Condition("nonexistent_field", ConditionOp.EQ, 64)
        ctx = make_context()
        assert cond.matches(ctx) is False


class TestConditionParsing:
    """Test parse_condition function."""

    def test_parse_ge_condition(self):
        """Parse '>=' condition."""
        from layerzero.policy.rule import parse_condition, ConditionOp

        cond = parse_condition("head_dim", ">=64")
        assert cond.op == ConditionOp.GE
        assert cond.value == 64

    def test_parse_le_condition(self):
        """Parse '<=' condition."""
        from layerzero.policy.rule import parse_condition, ConditionOp

        cond = parse_condition("seq_len_q", "<=1024")
        assert cond.op == ConditionOp.LE
        assert cond.value == 1024

    def test_parse_gt_condition(self):
        """Parse '>' condition."""
        from layerzero.policy.rule import parse_condition, ConditionOp

        cond = parse_condition("batch_size", ">8")
        assert cond.op == ConditionOp.GT
        assert cond.value == 8

    def test_parse_lt_condition(self):
        """Parse '<' condition."""
        from layerzero.policy.rule import parse_condition, ConditionOp

        cond = parse_condition("num_heads", "<32")
        assert cond.op == ConditionOp.LT
        assert cond.value == 32

    def test_parse_eq_condition(self):
        """Parse '==' condition."""
        from layerzero.policy.rule import parse_condition, ConditionOp

        cond = parse_condition("head_dim", "==64")
        assert cond.op == ConditionOp.EQ
        assert cond.value == 64

    def test_parse_ne_condition(self):
        """Parse '!=' condition."""
        from layerzero.policy.rule import parse_condition, ConditionOp

        cond = parse_condition("layout", "!=BHSD")
        assert cond.op == ConditionOp.NE
        assert cond.value == "BHSD"

    def test_parse_glob_pattern(self):
        """Parse glob pattern as MATCH condition."""
        from layerzero.policy.rule import parse_condition, ConditionOp

        cond = parse_condition("operation", "attention.*")
        assert cond.op == ConditionOp.MATCH
        assert cond.value == "attention.*"

    def test_parse_plain_value(self):
        """Parse plain value as EQ condition."""
        from layerzero.policy.rule import parse_condition, ConditionOp

        cond = parse_condition("layout", "BSHD")
        assert cond.op == ConditionOp.EQ
        assert cond.value == "BSHD"

    def test_parse_float_value(self):
        """Parse float value."""
        from layerzero.policy.rule import parse_condition, ConditionOp

        cond = parse_condition("dropout_p", ">=0.1")
        assert cond.op == ConditionOp.GE
        assert cond.value == 0.1


class TestRuleCreation:
    """Test Rule construction."""

    def test_rule_creation(self):
        """Rule can be created."""
        from layerzero.policy.rule import Rule, RuleType, Condition, ConditionOp

        rule = Rule(
            rule_type=RuleType.LOCK,
            target="attention.causal",
            conditions=(
                Condition("head_dim", ConditionOp.GE, 64),
            ),
            value="flash_attn.v3.causal",
        )
        assert rule.rule_type == RuleType.LOCK
        assert rule.target == "attention.causal"
        assert rule.value == "flash_attn.v3.causal"

    def test_rule_is_frozen(self):
        """Rule is immutable."""
        from layerzero.policy.rule import Rule, RuleType

        rule = Rule(RuleType.DENY, "torch.sdpa", (), None)
        with pytest.raises(AttributeError):
            rule.target = "other"  # type: ignore


class TestRuleMatching:
    """Test Rule matching methods."""

    def test_rule_matches_context_all_conditions(self):
        """Rule matches when all conditions match."""
        from layerzero.policy.rule import Rule, RuleType, Condition, ConditionOp

        rule = Rule(
            rule_type=RuleType.LOCK,
            target="attention.*",
            conditions=(
                Condition("head_dim", ConditionOp.GE, 64),
                Condition("seq_len_q", ConditionOp.GE, 128),
            ),
            value="flash_attn.v3.causal",
        )
        ctx = make_context(head_dim=64, seq_len_q=512)
        assert rule.matches_context(ctx) is True

    def test_rule_no_match_partial_conditions(self):
        """Rule doesn't match when any condition fails."""
        from layerzero.policy.rule import Rule, RuleType, Condition, ConditionOp

        rule = Rule(
            rule_type=RuleType.LOCK,
            target="attention.*",
            conditions=(
                Condition("head_dim", ConditionOp.GE, 128),  # Fails
                Condition("seq_len_q", ConditionOp.GE, 128),
            ),
            value="flash_attn.v3.causal",
        )
        ctx = make_context(head_dim=64, seq_len_q=512)
        assert rule.matches_context(ctx) is False

    def test_rule_matches_context_no_conditions(self):
        """Rule with no conditions always matches context."""
        from layerzero.policy.rule import Rule, RuleType

        rule = Rule(RuleType.DENY, "torch.sdpa", (), None)
        ctx = make_context()
        assert rule.matches_context(ctx) is True

    def test_rule_matches_target_exact(self):
        """Rule matches exact target."""
        from layerzero.policy.rule import Rule, RuleType

        rule = Rule(RuleType.DENY, "flash_attn.v3.causal", (), None)
        assert rule.matches_target("flash_attn.v3.causal") is True
        assert rule.matches_target("flash_attn.v2.causal") is False

    def test_rule_matches_target_glob(self):
        """Rule matches glob target pattern."""
        from layerzero.policy.rule import Rule, RuleType

        rule = Rule(RuleType.ALLOW, "flash_attn.*", (), None)
        assert rule.matches_target("flash_attn.v3.causal") is True
        assert rule.matches_target("flash_attn.v2.full") is True
        assert rule.matches_target("xformers.causal") is False
