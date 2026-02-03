"""
Tests for ScoringPhase.

TDD tests for kernel scoring based on priority, policy boosts, and transform cost.
"""
from __future__ import annotations

import pytest
import torch

from layerzero.models.device_spec import DeviceSpec
from layerzero.models.kernel_spec import KernelSpec
from layerzero.policy.policy import Policy
from layerzero.policy.rule import Rule, RuleType, Condition, ConditionOp
from layerzero.policy.engine import RuleEngine
from layerzero.selection.scorer import ScoringPhase

from .conftest import make_selection_context


class TestScoringPhaseInit:
    """Test ScoringPhase initialization."""

    def test_init_with_rule_engine(
        self,
        empty_policy: Policy,
    ) -> None:
        """Test initialization with rule engine."""
        rule_engine = RuleEngine(empty_policy)
        scorer = ScoringPhase(rule_engine)
        assert scorer is not None

    def test_rule_engine_property(
        self,
        empty_policy: Policy,
    ) -> None:
        """Test rule engine is accessible."""
        rule_engine = RuleEngine(empty_policy)
        scorer = ScoringPhase(rule_engine)
        assert scorer.rule_engine is rule_engine


class TestScoringPhaseBasePriority:
    """Test scoring with base priority only."""

    def test_score_single_kernel(
        self,
        empty_policy: Policy,
        high_priority_kernel: KernelSpec,
        device_spec: DeviceSpec,
    ) -> None:
        """Test scoring a single kernel."""
        rule_engine = RuleEngine(empty_policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        scores = scorer.score([high_priority_kernel], ctx)

        assert high_priority_kernel.kernel_id in scores
        # Base priority 90, no boosts, no transform cost
        assert scores[high_priority_kernel.kernel_id] == 90.0

    def test_score_multiple_kernels(
        self,
        empty_policy: Policy,
        high_priority_kernel: KernelSpec,
        medium_priority_kernel: KernelSpec,
        low_priority_kernel: KernelSpec,
        device_spec: DeviceSpec,
    ) -> None:
        """Test scoring multiple kernels."""
        rule_engine = RuleEngine(empty_policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        candidates = [high_priority_kernel, medium_priority_kernel, low_priority_kernel]
        scores = scorer.score(candidates, ctx)

        assert len(scores) == 3
        # High priority (90) > medium (50-5=45) > low (20-10=10)
        assert scores[high_priority_kernel.kernel_id] > scores[medium_priority_kernel.kernel_id]
        assert scores[medium_priority_kernel.kernel_id] > scores[low_priority_kernel.kernel_id]

    def test_score_empty_candidates(
        self,
        empty_policy: Policy,
        device_spec: DeviceSpec,
    ) -> None:
        """Test scoring empty candidate list."""
        rule_engine = RuleEngine(empty_policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        scores = scorer.score([], ctx)

        assert len(scores) == 0


class TestScoringPhaseTransformCost:
    """Test transform cost penalty."""

    def test_transform_cost_reduces_score(
        self,
        empty_policy: Policy,
        device_spec: DeviceSpec,
    ) -> None:
        """Test transform cost reduces kernel score."""
        kernel_no_cost = KernelSpec(
            kernel_id="kernel.no_cost",
            operation="attention.causal",
            source="test",
            version="1.0",
            priority=50,
            transform_cost_hint=0,
        )
        kernel_with_cost = KernelSpec(
            kernel_id="kernel.with_cost",
            operation="attention.causal",
            source="test",
            version="1.0",
            priority=50,
            transform_cost_hint=15,
        )

        rule_engine = RuleEngine(empty_policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        scores = scorer.score([kernel_no_cost, kernel_with_cost], ctx)

        # Same priority but different transform cost
        assert scores[kernel_no_cost.kernel_id] == 50.0
        assert scores[kernel_with_cost.kernel_id] == 35.0  # 50 - 15


class TestScoringPhaseAdditiveBoost:
    """Test additive boost rules."""

    def test_additive_boost_increases_score(
        self,
        medium_priority_kernel: KernelSpec,
        device_spec: DeviceSpec,
    ) -> None:
        """Test additive boost increases kernel score."""
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(
                Rule(
                    rule_type=RuleType.BOOST_ADD,
                    target="sdpa.*",
                    conditions=(),
                    value=20,
                ),
            ),
        )
        rule_engine = RuleEngine(policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        scores = scorer.score([medium_priority_kernel], ctx)

        # Base 50, +20 boost, -5 transform cost = 65
        assert scores[medium_priority_kernel.kernel_id] == 65.0

    def test_negative_additive_boost(
        self,
        high_priority_kernel: KernelSpec,
        device_spec: DeviceSpec,
    ) -> None:
        """Test negative additive boost decreases score."""
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(
                Rule(
                    rule_type=RuleType.BOOST_ADD,
                    target="flash_attn.*",
                    conditions=(),
                    value=-30,
                ),
            ),
        )
        rule_engine = RuleEngine(policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        scores = scorer.score([high_priority_kernel], ctx)

        # Base 90, -30 boost = 60
        assert scores[high_priority_kernel.kernel_id] == 60.0

    def test_multiple_additive_boosts(
        self,
        medium_priority_kernel: KernelSpec,
        device_spec: DeviceSpec,
    ) -> None:
        """Test multiple additive boosts are summed."""
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(
                Rule(
                    rule_type=RuleType.BOOST_ADD,
                    target="sdpa.*",
                    conditions=(),
                    value=10,
                ),
                Rule(
                    rule_type=RuleType.BOOST_ADD,
                    target="*",
                    conditions=(),
                    value=5,
                ),
            ),
        )
        rule_engine = RuleEngine(policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        scores = scorer.score([medium_priority_kernel], ctx)

        # Base 50, +10 +5 boosts, -5 transform cost = 60
        assert scores[medium_priority_kernel.kernel_id] == 60.0


class TestScoringPhaseMultiplicativeBoost:
    """Test multiplicative boost rules."""

    def test_multiplicative_boost_increases_score(
        self,
        device_spec: DeviceSpec,
    ) -> None:
        """Test multiplicative boost scales kernel score."""
        kernel = KernelSpec(
            kernel_id="test.kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            priority=50,
            transform_cost_hint=0,
        )
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(
                Rule(
                    rule_type=RuleType.BOOST_MULTIPLY,
                    target="test.*",
                    conditions=(),
                    value=2.0,
                ),
            ),
        )
        rule_engine = RuleEngine(policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        scores = scorer.score([kernel], ctx)

        # Base 50, * 2.0 = 100
        assert scores[kernel.kernel_id] == 100.0

    def test_fractional_multiplicative_boost(
        self,
        device_spec: DeviceSpec,
    ) -> None:
        """Test fractional multiplicative boost reduces score."""
        kernel = KernelSpec(
            kernel_id="test.kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            priority=100,
            transform_cost_hint=0,
        )
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(
                Rule(
                    rule_type=RuleType.BOOST_MULTIPLY,
                    target="test.*",
                    conditions=(),
                    value=0.5,
                ),
            ),
        )
        rule_engine = RuleEngine(policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        scores = scorer.score([kernel], ctx)

        # Base 100, * 0.5 = 50
        assert scores[kernel.kernel_id] == 50.0

    def test_multiple_multiplicative_boosts(
        self,
        device_spec: DeviceSpec,
    ) -> None:
        """Test multiple multiplicative boosts are multiplied together."""
        kernel = KernelSpec(
            kernel_id="test.kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            priority=50,
            transform_cost_hint=0,
        )
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(
                Rule(
                    rule_type=RuleType.BOOST_MULTIPLY,
                    target="test.*",
                    conditions=(),
                    value=2.0,
                ),
                Rule(
                    rule_type=RuleType.BOOST_MULTIPLY,
                    target="*",
                    conditions=(),
                    value=1.5,
                ),
            ),
        )
        rule_engine = RuleEngine(policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        scores = scorer.score([kernel], ctx)

        # Base 50, * 2.0 * 1.5 = 150
        assert scores[kernel.kernel_id] == 150.0


class TestScoringPhaseBoostOrder:
    """Test boost application order."""

    def test_add_before_multiply(
        self,
        device_spec: DeviceSpec,
    ) -> None:
        """Test additive boost applied before multiplicative."""
        kernel = KernelSpec(
            kernel_id="test.kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            priority=50,
            transform_cost_hint=0,
        )
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(
                Rule(
                    rule_type=RuleType.BOOST_ADD,
                    target="test.*",
                    conditions=(),
                    value=10,
                ),
                Rule(
                    rule_type=RuleType.BOOST_MULTIPLY,
                    target="test.*",
                    conditions=(),
                    value=2.0,
                ),
            ),
        )
        rule_engine = RuleEngine(policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        scores = scorer.score([kernel], ctx)

        # (50 + 10) * 2.0 = 120
        assert scores[kernel.kernel_id] == 120.0


class TestScoringPhaseConditionalBoosts:
    """Test conditional boost rules."""

    def test_conditional_boost_applies_when_matched(
        self,
        device_spec: DeviceSpec,
    ) -> None:
        """Test conditional boost applies when condition matches."""
        kernel = KernelSpec(
            kernel_id="test.kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            priority=50,
            transform_cost_hint=0,
        )
        # Boost when head_dim >= 64
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(
                Rule(
                    rule_type=RuleType.BOOST_ADD,
                    target="test.*",
                    conditions=(
                        Condition(field="head_dim", op=ConditionOp.GE, value=64),
                    ),
                    value=25,
                ),
            ),
        )
        rule_engine = RuleEngine(policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec, head_dim=64)

        scores = scorer.score([kernel], ctx)

        # 50 + 25 = 75
        assert scores[kernel.kernel_id] == 75.0

    def test_conditional_boost_skipped_when_not_matched(
        self,
        device_spec: DeviceSpec,
    ) -> None:
        """Test conditional boost not applied when condition doesn't match."""
        kernel = KernelSpec(
            kernel_id="test.kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            priority=50,
            transform_cost_hint=0,
        )
        # Boost when head_dim >= 128
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(
                Rule(
                    rule_type=RuleType.BOOST_ADD,
                    target="test.*",
                    conditions=(
                        Condition(field="head_dim", op=ConditionOp.GE, value=128),
                    ),
                    value=25,
                ),
            ),
        )
        rule_engine = RuleEngine(policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec, head_dim=64)

        scores = scorer.score([kernel], ctx)

        # 50 (no boost applied)
        assert scores[kernel.kernel_id] == 50.0


class TestScoringPhaseTargetMatching:
    """Test target pattern matching for boosts."""

    def test_exact_target_match(
        self,
        device_spec: DeviceSpec,
    ) -> None:
        """Test exact kernel ID target matching."""
        kernel = KernelSpec(
            kernel_id="flash_attn.v3",
            operation="attention.causal",
            source="flash_attn",
            version="3.0",
            priority=50,
            transform_cost_hint=0,
        )
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(
                Rule(
                    rule_type=RuleType.BOOST_ADD,
                    target="flash_attn.v3",
                    conditions=(),
                    value=10,
                ),
            ),
        )
        rule_engine = RuleEngine(policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        scores = scorer.score([kernel], ctx)

        assert scores[kernel.kernel_id] == 60.0

    def test_wildcard_target_match(
        self,
        device_spec: DeviceSpec,
    ) -> None:
        """Test wildcard target matching."""
        kernel1 = KernelSpec(
            kernel_id="flash_attn.v2",
            operation="attention.causal",
            source="flash_attn",
            version="2.0",
            priority=50,
            transform_cost_hint=0,
        )
        kernel2 = KernelSpec(
            kernel_id="flash_attn.v3",
            operation="attention.causal",
            source="flash_attn",
            version="3.0",
            priority=50,
            transform_cost_hint=0,
        )
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(
                Rule(
                    rule_type=RuleType.BOOST_ADD,
                    target="flash_attn.*",
                    conditions=(),
                    value=10,
                ),
            ),
        )
        rule_engine = RuleEngine(policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        scores = scorer.score([kernel1, kernel2], ctx)

        assert scores[kernel1.kernel_id] == 60.0
        assert scores[kernel2.kernel_id] == 60.0

    def test_no_target_match(
        self,
        device_spec: DeviceSpec,
    ) -> None:
        """Test no target match doesn't apply boost."""
        kernel = KernelSpec(
            kernel_id="sdpa.default",
            operation="attention.causal",
            source="torch",
            version="2.0",
            priority=50,
            transform_cost_hint=0,
        )
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(
                Rule(
                    rule_type=RuleType.BOOST_ADD,
                    target="flash_attn.*",
                    conditions=(),
                    value=10,
                ),
            ),
        )
        rule_engine = RuleEngine(policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        scores = scorer.score([kernel], ctx)

        # No boost applied
        assert scores[kernel.kernel_id] == 50.0


class TestScoringPhaseScoreOrdering:
    """Test score ordering for selection."""

    def test_scores_return_correct_order(
        self,
        empty_policy: Policy,
        high_priority_kernel: KernelSpec,
        medium_priority_kernel: KernelSpec,
        low_priority_kernel: KernelSpec,
        device_spec: DeviceSpec,
    ) -> None:
        """Test scores can be used to determine kernel order."""
        rule_engine = RuleEngine(empty_policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        candidates = [low_priority_kernel, high_priority_kernel, medium_priority_kernel]
        scores = scorer.score(candidates, ctx)

        # Sort candidates by score descending
        sorted_candidates = sorted(candidates, key=lambda k: scores[k.kernel_id], reverse=True)

        assert sorted_candidates[0].kernel_id == high_priority_kernel.kernel_id
        assert sorted_candidates[1].kernel_id == medium_priority_kernel.kernel_id
        assert sorted_candidates[2].kernel_id == low_priority_kernel.kernel_id

    def test_policy_boost_changes_order(
        self,
        high_priority_kernel: KernelSpec,
        low_priority_kernel: KernelSpec,
        device_spec: DeviceSpec,
    ) -> None:
        """Test policy boost can change kernel ordering."""
        # Boost low priority kernel enough to surpass high priority
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(
                Rule(
                    rule_type=RuleType.BOOST_ADD,
                    target="legacy.*",
                    conditions=(),
                    value=100,  # 20 + 100 - 10 = 110 > 90
                ),
            ),
        )
        rule_engine = RuleEngine(policy)
        scorer = ScoringPhase(rule_engine)
        ctx = make_selection_context(device_spec)

        candidates = [high_priority_kernel, low_priority_kernel]
        scores = scorer.score(candidates, ctx)

        # Low priority kernel now has higher score due to boost
        assert scores[low_priority_kernel.kernel_id] > scores[high_priority_kernel.kernel_id]
