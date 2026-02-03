"""
Test suite for RuleEngine.

Tests rule evaluation and filtering.
Following TDD methodology - tests define expected behavior.
"""
import pytest
import torch

from layerzero.models.device_spec import DeviceSpec
from layerzero.models.selection_context import SelectionContext
from layerzero.enums import OpKind


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


class TestRuleEngineCreation:
    """Test RuleEngine construction."""

    def test_rule_engine_creation(self):
        """RuleEngine can be created with policy."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(),
        )
        engine = RuleEngine(policy)
        assert engine is not None


class TestRuleEngineLocks:
    """Test lock rule evaluation."""

    def test_get_locked_kernel_no_locks(self):
        """Returns None when no lock rules match."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(),
        )
        engine = RuleEngine(policy)
        ctx = make_context()

        result = engine.get_locked_kernel(ctx)
        assert result is None

    def test_get_locked_kernel_matches(self):
        """Returns kernel ID when lock rule matches."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType

        policy = Policy(
            version="1.0",
            locks=(
                Rule(RuleType.LOCK, "attention.causal", (), "flash_attn.v3.causal"),
            ),
            allows=(),
            denies=(),
            boosts=(),
        )
        engine = RuleEngine(policy)
        ctx = make_context(operation="attention.causal")

        result = engine.get_locked_kernel(ctx)
        assert result == "flash_attn.v3.causal"

    def test_get_locked_kernel_conditional(self):
        """Lock rule respects conditions."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType, Condition, ConditionOp

        policy = Policy(
            version="1.0",
            locks=(
                Rule(
                    RuleType.LOCK,
                    "attention.causal",
                    (Condition("head_dim", ConditionOp.GE, 128),),  # Won't match
                    "flash_attn.v3.causal",
                ),
            ),
            allows=(),
            denies=(),
            boosts=(),
        )
        engine = RuleEngine(policy)
        ctx = make_context(head_dim=64)  # Less than 128

        result = engine.get_locked_kernel(ctx)
        assert result is None

    def test_get_locked_kernel_glob_match(self):
        """Lock rule with glob pattern matches."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType

        policy = Policy(
            version="1.0",
            locks=(
                Rule(RuleType.LOCK, "attention.*", (), "flash_attn.v3"),
            ),
            allows=(),
            denies=(),
            boosts=(),
        )
        engine = RuleEngine(policy)

        ctx_causal = make_context(operation="attention.causal")
        ctx_full = make_context(operation="attention.full")
        ctx_norm = make_context(operation="norm.rms")

        assert engine.get_locked_kernel(ctx_causal) == "flash_attn.v3"
        assert engine.get_locked_kernel(ctx_full) == "flash_attn.v3"
        assert engine.get_locked_kernel(ctx_norm) is None


class TestRuleEngineDeny:
    """Test deny rule evaluation."""

    def test_is_denied_no_denies(self):
        """Returns False when no deny rules."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(),
        )
        engine = RuleEngine(policy)
        ctx = make_context()

        assert engine.is_denied("any_kernel", ctx) is False

    def test_is_denied_matches(self):
        """Returns True when deny rule matches."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(Rule(RuleType.DENY, "torch.sdpa", (), None),),
            boosts=(),
        )
        engine = RuleEngine(policy)
        ctx = make_context()

        assert engine.is_denied("torch.sdpa", ctx) is True
        assert engine.is_denied("flash_attn.v3", ctx) is False

    def test_is_denied_glob(self):
        """Deny rule with glob pattern."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(Rule(RuleType.DENY, "triton.*", (), None),),
            boosts=(),
        )
        engine = RuleEngine(policy)
        ctx = make_context()

        assert engine.is_denied("triton.attention", ctx) is True
        assert engine.is_denied("triton.norm", ctx) is True
        assert engine.is_denied("flash_attn.v3", ctx) is False

    def test_is_denied_conditional(self):
        """Deny rule with condition."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType, Condition, ConditionOp

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(
                Rule(
                    RuleType.DENY,
                    "triton.*",
                    (Condition("batch_size", ConditionOp.LT, 4),),
                    None,
                ),
            ),
            boosts=(),
        )
        engine = RuleEngine(policy)

        ctx_small = make_context(batch_size=2)
        ctx_large = make_context(batch_size=32)

        assert engine.is_denied("triton.attention", ctx_small) is True
        assert engine.is_denied("triton.attention", ctx_large) is False


class TestRuleEngineAllow:
    """Test allow rule evaluation."""

    def test_is_allowed_no_allows(self):
        """Returns True when no allow rules (default allow all)."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(),
        )
        engine = RuleEngine(policy)
        ctx = make_context()

        assert engine.is_allowed("any_kernel", ctx) is True

    def test_is_allowed_whitelist(self):
        """When allow rules exist, only allowed kernels pass."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(
                Rule(RuleType.ALLOW, "flash_attn.*", (), None),
                Rule(RuleType.ALLOW, "xformers.*", (), None),
            ),
            denies=(),
            boosts=(),
        )
        engine = RuleEngine(policy)
        ctx = make_context()

        assert engine.is_allowed("flash_attn.v3.causal", ctx) is True
        assert engine.is_allowed("xformers.causal", ctx) is True
        assert engine.is_allowed("torch.sdpa", ctx) is False


class TestRuleEngineBoosts:
    """Test boost rule evaluation."""

    def test_get_priority_boost_no_boosts(self):
        """Returns (0, 1.0) when no boost rules."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(),
        )
        engine = RuleEngine(policy)
        ctx = make_context()

        add, mult = engine.get_priority_boost("any_kernel", ctx)
        assert add == 0
        assert mult == 1.0

    def test_get_priority_boost_additive(self):
        """Additive boost adds to priority."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(Rule(RuleType.BOOST_ADD, "flash_attn.*", (), 20),),
        )
        engine = RuleEngine(policy)
        ctx = make_context()

        add, mult = engine.get_priority_boost("flash_attn.v3.causal", ctx)
        assert add == 20
        assert mult == 1.0

    def test_get_priority_boost_multiplicative(self):
        """Multiplicative boost multiplies priority."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(Rule(RuleType.BOOST_MULTIPLY, "liger.*", (), 1.5),),
        )
        engine = RuleEngine(policy)
        ctx = make_context()

        add, mult = engine.get_priority_boost("liger.rms", ctx)
        assert add == 0
        assert mult == 1.5


class TestRuleEngineFilter:
    """Test kernel filtering."""

    def test_filter_kernels_empty_policy(self):
        """Empty policy returns all kernels."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(),
        )
        engine = RuleEngine(policy)
        ctx = make_context()

        kernels = ["flash_attn.v3", "xformers.causal", "torch.sdpa"]
        result = engine.filter_kernels(kernels, ctx)
        assert result == kernels

    def test_filter_kernels_deny(self):
        """Denied kernels are filtered out."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(Rule(RuleType.DENY, "torch.sdpa", (), None),),
            boosts=(),
        )
        engine = RuleEngine(policy)
        ctx = make_context()

        kernels = ["flash_attn.v3", "xformers.causal", "torch.sdpa"]
        result = engine.filter_kernels(kernels, ctx)
        assert "torch.sdpa" not in result
        assert len(result) == 2

    def test_filter_kernels_allow(self):
        """Only allowed kernels pass."""
        from layerzero.policy.engine import RuleEngine
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(Rule(RuleType.ALLOW, "flash_attn.*", (), None),),
            denies=(),
            boosts=(),
        )
        engine = RuleEngine(policy)
        ctx = make_context()

        kernels = ["flash_attn.v3", "flash_attn.v2", "torch.sdpa"]
        result = engine.filter_kernels(kernels, ctx)
        assert "torch.sdpa" not in result
        assert len(result) == 2
