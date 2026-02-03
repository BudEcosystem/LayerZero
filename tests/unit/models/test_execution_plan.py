"""
Test suite for ExecutionPlan and SelectionReport dataclasses.

Tests execution plan structure and serialization.
Following TDD methodology - tests define expected behavior.
"""
import json
import pytest


class TestExecutionPlanCreation:
    """Test ExecutionPlan construction."""

    def test_execution_plan_required_fields(self):
        """ExecutionPlan must have kernel_id and kernel_spec."""
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec

        kernel_spec = KernelSpec(
            kernel_id="flash_attn.v3.causal",
            operation="attention.causal",
            source="flash_attn",
            version="3.0.0",
        )

        plan = ExecutionPlan(
            kernel_id="flash_attn.v3.causal",
            kernel_spec=kernel_spec,
        )

        assert plan.kernel_id == "flash_attn.v3.causal"
        assert plan.kernel_spec is not None

    def test_execution_plan_defaults(self):
        """ExecutionPlan has default empty transforms."""
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec

        kernel_spec = KernelSpec(
            kernel_id="test.kernel",
            operation="attention.causal",
            source="test",
            version="1.0.0",
        )

        plan = ExecutionPlan(
            kernel_id="test.kernel",
            kernel_spec=kernel_spec,
        )

        assert plan.pre_transforms == ()
        assert plan.post_transforms == ()
        assert plan.debug_info is None
        assert plan.cached is False
        assert plan.cache_key is None

    def test_execution_plan_with_transforms(self):
        """ExecutionPlan can have pre/post transforms."""
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec

        kernel_spec = KernelSpec(
            kernel_id="flash_attn.v3.causal",
            operation="attention.causal",
            source="flash_attn",
            version="3.0.0",
        )

        plan = ExecutionPlan(
            kernel_id="flash_attn.v3.causal",
            kernel_spec=kernel_spec,
            pre_transforms=("layout_BSHD_to_BHSD",),
            post_transforms=("layout_BHSD_to_BSHD",),
        )

        assert plan.pre_transforms == ("layout_BSHD_to_BHSD",)
        assert plan.post_transforms == ("layout_BHSD_to_BSHD",)

    def test_execution_plan_is_frozen(self):
        """ExecutionPlan is immutable (frozen)."""
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec

        kernel_spec = KernelSpec(
            kernel_id="test.kernel",
            operation="attention.causal",
            source="test",
            version="1.0.0",
        )

        plan = ExecutionPlan(
            kernel_id="test.kernel",
            kernel_spec=kernel_spec,
        )

        with pytest.raises(AttributeError):
            plan.kernel_id = "other.kernel"  # type: ignore


class TestExecutionPlanSerialization:
    """Test ExecutionPlan JSON serialization."""

    def test_execution_plan_to_dict(self):
        """ExecutionPlan can be serialized to dict."""
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec

        kernel_spec = KernelSpec(
            kernel_id="flash_attn.v3.causal",
            operation="attention.causal",
            source="flash_attn",
            version="3.0.0",
            priority=100,
        )

        plan = ExecutionPlan(
            kernel_id="flash_attn.v3.causal",
            kernel_spec=kernel_spec,
            pre_transforms=("layout_convert",),
            cached=True,
            cache_key="abc123",
        )

        d = plan.to_dict()
        assert d["kernel_id"] == "flash_attn.v3.causal"
        assert d["pre_transforms"] == ["layout_convert"]
        assert d["cached"] is True
        assert d["cache_key"] == "abc123"


class TestSelectionReportCreation:
    """Test SelectionReport construction."""

    def test_selection_report_required_fields(self):
        """SelectionReport captures selection decision details."""
        from layerzero.models.execution_plan import SelectionReport

        report = SelectionReport(
            operation="attention.causal",
            context_summary={"dtype": "float16", "head_dim": 64},
            candidates=("flash_attn.v3", "flash_attn.v2", "torch.sdpa"),
            filtered_out={},
            scores={"flash_attn.v3": 100.0, "flash_attn.v2": 90.0, "torch.sdpa": 50.0},
            selected_kernel="flash_attn.v3",
            selection_reason="highest_priority",
            selection_time_us=150,
        )

        assert report.operation == "attention.causal"
        assert report.selected_kernel == "flash_attn.v3"
        assert report.selection_time_us == 150

    def test_selection_report_with_filtered_out(self):
        """SelectionReport tracks filtered candidates with reasons."""
        from layerzero.models.execution_plan import SelectionReport
        from layerzero.reasons import Reason, ReasonCategory, SM_TOO_OLD

        report = SelectionReport(
            operation="attention.causal",
            context_summary={"sm_version": (8, 6)},
            candidates=("flash_attn.v3", "flash_attn.v2"),
            filtered_out={
                "flash_attn.v3": [
                    Reason(
                        code=SM_TOO_OLD,
                        message="Requires SM 9.0+",
                        category=ReasonCategory.HARDWARE
                    )
                ]
            },
            scores={"flash_attn.v2": 90.0},
            selected_kernel="flash_attn.v2",
            selection_reason="only_valid_candidate",
            selection_time_us=200,
        )

        assert "flash_attn.v3" in report.filtered_out
        assert len(report.filtered_out["flash_attn.v3"]) == 1
        assert report.filtered_out["flash_attn.v3"][0].code == SM_TOO_OLD

    def test_selection_report_is_frozen(self):
        """SelectionReport is immutable (frozen)."""
        from layerzero.models.execution_plan import SelectionReport

        report = SelectionReport(
            operation="attention.causal",
            context_summary={},
            candidates=(),
            filtered_out={},
            scores={},
            selected_kernel="torch.sdpa",
            selection_reason="fallback",
            selection_time_us=100,
        )

        with pytest.raises(AttributeError):
            report.selected_kernel = "other"  # type: ignore


class TestSelectionReportSerialization:
    """Test SelectionReport JSON serialization."""

    def test_selection_report_to_dict(self):
        """SelectionReport can be serialized to dict."""
        from layerzero.models.execution_plan import SelectionReport

        report = SelectionReport(
            operation="norm.rms",
            context_summary={"batch_size": 32, "dtype": "bfloat16"},
            candidates=("liger.rms", "apex.rms", "torch.rms"),
            filtered_out={},
            scores={"liger.rms": 100.0, "apex.rms": 90.0, "torch.rms": 10.0},
            selected_kernel="liger.rms",
            selection_reason="highest_priority",
            selection_time_us=50,
        )

        d = report.to_dict()
        assert d["operation"] == "norm.rms"
        assert d["selected_kernel"] == "liger.rms"
        assert d["selection_time_us"] == 50
        assert "liger.rms" in d["scores"]

    def test_selection_report_json_roundtrip(self):
        """SelectionReport serialize/deserialize preserves fields."""
        from layerzero.models.execution_plan import SelectionReport

        original = SelectionReport(
            operation="attention.causal",
            context_summary={"head_dim": 128},
            candidates=("flash_attn.v2",),
            filtered_out={},
            scores={"flash_attn.v2": 90.0},
            selected_kernel="flash_attn.v2",
            selection_reason="only_candidate",
            selection_time_us=75,
        )

        json_str = json.dumps(original.to_dict())
        d = json.loads(json_str)
        restored = SelectionReport.from_dict(d)

        assert restored.operation == original.operation
        assert restored.selected_kernel == original.selected_kernel
        assert restored.selection_time_us == original.selection_time_us
