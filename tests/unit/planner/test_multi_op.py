"""Tests for multi-op planner."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from typing import Any

from layerzero.planner.multi_op import (
    MultiOpPlanner,
    PlannerConfig,
    OpPlan,
    MultiOpPlan,
    TransformCost,
)


class TestPlannerConfig:
    """Tests for PlannerConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config values."""
        config = PlannerConfig()

        assert config.transform_penalty == 1.0
        assert config.layout_transform_cost_ms == 0.5
        assert config.dtype_transform_cost_ms == 0.3
        assert config.optimize_jointly is True

    def test_custom_values(self) -> None:
        """Custom config values."""
        config = PlannerConfig(
            transform_penalty=2.0,
            layout_transform_cost_ms=1.0,
            dtype_transform_cost_ms=0.5,
            optimize_jointly=False,
        )

        assert config.transform_penalty == 2.0
        assert config.layout_transform_cost_ms == 1.0
        assert config.dtype_transform_cost_ms == 0.5
        assert config.optimize_jointly is False

    def test_config_immutable(self) -> None:
        """Config is immutable."""
        config = PlannerConfig()

        with pytest.raises(AttributeError):
            config.transform_penalty = 5.0


class TestTransformCost:
    """Tests for TransformCost."""

    def test_no_transform(self) -> None:
        """No transform needed."""
        cost = TransformCost(
            from_layout="BSHD",
            to_layout="BSHD",
            from_dtype="float16",
            to_dtype="float16",
        )

        assert cost.requires_layout_transform is False
        assert cost.requires_dtype_transform is False
        assert cost.total_cost_ms == 0.0

    def test_layout_transform(self) -> None:
        """Layout transform required."""
        cost = TransformCost(
            from_layout="BSHD",
            to_layout="BHSD",
            from_dtype="float16",
            to_dtype="float16",
            layout_cost_ms=0.5,
        )

        assert cost.requires_layout_transform is True
        assert cost.requires_dtype_transform is False
        assert cost.total_cost_ms == 0.5

    def test_dtype_transform(self) -> None:
        """Dtype transform required."""
        cost = TransformCost(
            from_layout="BSHD",
            to_layout="BSHD",
            from_dtype="float16",
            to_dtype="float32",
            dtype_cost_ms=0.3,
        )

        assert cost.requires_layout_transform is False
        assert cost.requires_dtype_transform is True
        assert cost.total_cost_ms == 0.3

    def test_both_transforms(self) -> None:
        """Both transforms required."""
        cost = TransformCost(
            from_layout="BSHD",
            to_layout="BHSD",
            from_dtype="float16",
            to_dtype="float32",
            layout_cost_ms=0.5,
            dtype_cost_ms=0.3,
        )

        assert cost.requires_layout_transform is True
        assert cost.requires_dtype_transform is True
        assert cost.total_cost_ms == 0.8


class TestOpPlan:
    """Tests for OpPlan."""

    def test_creation(self) -> None:
        """OpPlan stores operation plan."""
        plan = OpPlan(
            op_type="attention",
            kernel_id="flash_attn",
            input_layout="BSHD",
            output_layout="BSHD",
            estimated_latency_ms=1.0,
        )

        assert plan.op_type == "attention"
        assert plan.kernel_id == "flash_attn"
        assert plan.estimated_latency_ms == 1.0

    def test_to_dict(self) -> None:
        """OpPlan serializes to dict."""
        plan = OpPlan(
            op_type="attention",
            kernel_id="flash_attn",
            input_layout="BSHD",
            output_layout="BSHD",
            estimated_latency_ms=1.0,
        )

        d = plan.to_dict()

        assert d["op_type"] == "attention"
        assert d["kernel_id"] == "flash_attn"


class TestMultiOpPlan:
    """Tests for MultiOpPlan."""

    def test_creation(self) -> None:
        """MultiOpPlan stores multiple op plans."""
        ops = [
            OpPlan("attention", "flash_attn", "BSHD", "BSHD", 1.0),
            OpPlan("layernorm", "triton_ln", "BSHD", "BSHD", 0.2),
        ]
        plan = MultiOpPlan(ops=ops)

        assert len(plan.ops) == 2

    def test_total_latency(self) -> None:
        """total_latency sums op latencies."""
        ops = [
            OpPlan("attention", "flash_attn", "BSHD", "BSHD", 1.0),
            OpPlan("layernorm", "triton_ln", "BSHD", "BSHD", 0.2),
            OpPlan("mlp", "fused_mlp", "BSH", "BSH", 1.5),
        ]
        plan = MultiOpPlan(ops=ops)

        assert plan.total_latency_ms == 2.7

    def test_total_transform_cost(self) -> None:
        """total_transform_cost sums transform costs."""
        ops = [
            OpPlan("attention", "flash_attn", "BSHD", "BSHD", 1.0, transform_cost_ms=0.0),
            OpPlan("layernorm", "triton_ln", "BHSD", "BSHD", 0.2, transform_cost_ms=0.5),
        ]
        plan = MultiOpPlan(ops=ops)

        assert plan.total_transform_cost_ms == 0.5

    def test_to_dict(self) -> None:
        """MultiOpPlan serializes to dict."""
        ops = [
            OpPlan("attention", "flash_attn", "BSHD", "BSHD", 1.0),
        ]
        plan = MultiOpPlan(ops=ops)

        d = plan.to_dict()

        assert "ops" in d
        assert "total_latency_ms" in d


class TestMultiOpPlanner:
    """Tests for MultiOpPlanner."""

    def test_plan_attention_norm_mlp(self, transformer_block) -> None:
        """Plan attention + norm + MLP block."""
        planner = MultiOpPlanner()

        with patch.object(planner, '_get_candidate_kernels') as mock_kernels:
            mock_kernels.side_effect = lambda op: {
                "attention": [
                    {"kernel_id": "flash_attn", "input_layout": "BSHD", "output_layout": "BSHD", "latency_ms": 1.0}
                ],
                "layernorm": [
                    {"kernel_id": "triton_ln", "input_layout": "BSHD", "output_layout": "BSHD", "latency_ms": 0.2}
                ],
                "mlp": [
                    {"kernel_id": "fused_mlp", "input_layout": "BSH", "output_layout": "BSH", "latency_ms": 1.5}
                ],
            }.get(op["op_type"], [])

            plan = planner.plan(transformer_block)

        assert plan is not None
        assert len(plan.ops) == 3

    def test_plan_minimizes_transforms(self) -> None:
        """Plan minimizes layout/dtype transforms."""
        config = PlannerConfig(
            layout_transform_cost_ms=1.0,  # Expensive transform
            transform_penalty=2.0,
        )
        planner = MultiOpPlanner(config=config)

        ops = [
            {"op_type": "attention", "input_layout": "BSHD", "output_layout": "BSHD",
             "input_dtype": "float16", "output_dtype": "float16"},
            {"op_type": "layernorm", "input_layout": "BSHD", "output_layout": "BSHD",
             "input_dtype": "float16", "output_dtype": "float16"},
        ]

        with patch.object(planner, '_get_candidate_kernels') as mock_kernels:
            mock_kernels.side_effect = lambda op: [
                {"kernel_id": f"{op['op_type']}_kernel", "input_layout": "BSHD",
                 "output_layout": "BSHD", "latency_ms": 1.0}
            ]

            plan = planner.plan(ops)

        # Plan should minimize transforms
        assert plan.total_transform_cost_ms == 0.0

    def test_plan_joint_selection(self) -> None:
        """Kernels selected jointly for better overall latency."""
        config = PlannerConfig(optimize_jointly=True)
        planner = MultiOpPlanner(config=config)

        ops = [
            {"op_type": "attention", "input_layout": "BSHD", "output_layout": "BSHD",
             "input_dtype": "float16", "output_dtype": "float16"},
            {"op_type": "layernorm", "input_layout": "BSHD", "output_layout": "BSHD",
             "input_dtype": "float16", "output_dtype": "float16"},
        ]

        with patch.object(planner, '_get_candidate_kernels') as mock_kernels:
            mock_kernels.side_effect = lambda op: [
                {"kernel_id": f"{op['op_type']}_fast", "input_layout": "BSHD",
                 "output_layout": "BSHD", "latency_ms": 0.5},
                {"kernel_id": f"{op['op_type']}_slow", "input_layout": "BHSD",
                 "output_layout": "BHSD", "latency_ms": 0.3},  # Faster but needs transform
            ]

            plan = planner.plan(ops)

        # Joint selection should consider transform costs
        assert plan is not None
        assert len(plan.ops) == 2

    def test_plan_total_latency_optimized(self) -> None:
        """Total latency optimized, not per-op."""
        config = PlannerConfig(optimize_jointly=True)
        planner = MultiOpPlanner(config=config)

        ops = [
            {"op_type": "op1", "input_layout": "BSHD", "output_layout": "BSHD",
             "input_dtype": "float16", "output_dtype": "float16"},
            {"op_type": "op2", "input_layout": "BSHD", "output_layout": "BSHD",
             "input_dtype": "float16", "output_dtype": "float16"},
        ]

        with patch.object(planner, '_get_candidate_kernels') as mock_kernels:
            mock_kernels.side_effect = lambda op: [
                {"kernel_id": "matching_layout", "input_layout": "BSHD",
                 "output_layout": "BSHD", "latency_ms": 1.0}
            ]

            plan = planner.plan(ops)

        # Total latency should be optimized (no transform overhead)
        assert plan.total_latency_ms <= 2.0

    def test_plan_empty_ops(self) -> None:
        """Handle empty operations list."""
        planner = MultiOpPlanner()

        plan = planner.plan([])

        assert plan is not None
        assert len(plan.ops) == 0
        assert plan.total_latency_ms == 0.0
