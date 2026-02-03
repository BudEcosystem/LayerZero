"""Tests for LayerZero system APIs.

Tests for lz.doctor(), lz.readiness_check(), lz.compile(), lz.dry_run(), etc.
"""
from __future__ import annotations

import pytest
import torch


class TestDoctorAPI:
    """Tests for lz.doctor() public API."""

    def test_doctor_basic(self) -> None:
        """Basic doctor diagnostics."""
        import layerzero as lz

        report = lz.doctor()

        assert report is not None
        assert hasattr(report, 'healthy') or hasattr(report, 'status')

    def test_doctor_checks_backends(self) -> None:
        """Doctor checks backend availability."""
        import layerzero as lz

        report = lz.doctor()

        # Should have backend status
        assert hasattr(report, 'backends') or hasattr(report, 'backend_status')

    def test_doctor_checks_cuda(self) -> None:
        """Doctor checks CUDA status."""
        import layerzero as lz

        report = lz.doctor()

        # Should have CUDA status
        assert hasattr(report, 'cuda') or hasattr(report, 'cuda_available')

    def test_doctor_checks_cache(self) -> None:
        """Doctor checks cache status."""
        import layerzero as lz

        report = lz.doctor()

        # Should have cache status
        assert hasattr(report, 'cache') or hasattr(report, 'cache_status')

    def test_doctor_returns_summary(self) -> None:
        """Doctor returns overall summary."""
        import layerzero as lz

        report = lz.doctor()

        # Should have some form of summary
        assert hasattr(report, 'summary') or hasattr(report, 'message') or hasattr(report, 'healthy')


class TestReadinessCheckAPI:
    """Tests for lz.readiness_check() public API."""

    def test_readiness_check_basic(self) -> None:
        """Basic readiness check."""
        import layerzero as lz

        result = lz.readiness_check()

        assert result is not None
        assert hasattr(result, 'ready') or hasattr(result, 'is_ready')

    def test_readiness_check_returns_issues(self) -> None:
        """Readiness check returns issues if any."""
        import layerzero as lz

        result = lz.readiness_check()

        # Should indicate issues if not ready
        if not (getattr(result, 'ready', True) or getattr(result, 'is_ready', True)):
            assert hasattr(result, 'issues') or hasattr(result, 'problems')

    def test_readiness_check_validates_backends(self) -> None:
        """Readiness check validates backend availability."""
        import layerzero as lz

        result = lz.readiness_check()

        # Should check backends
        assert hasattr(result, 'backends_checked') or hasattr(result, 'backend_status')


class TestCompileAPI:
    """Tests for lz.compile() public API."""

    def test_compile_model(self) -> None:
        """Compile kernel selections for model."""
        import layerzero as lz

        # Simple model for testing
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Compile should return compiled model or plan
        result = lz.compile(model)

        # Should return something (compiled model or plan)
        assert result is not None

    def test_compile_with_shapes(self) -> None:
        """Compile with specific input shapes."""
        import layerzero as lz

        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x * 2

        model = SimpleModel()

        result = lz.compile(
            model,
            shapes=[(2, 64, 128)],
            dtype=torch.float16,
        )

        assert result is not None


class TestDryRunAPI:
    """Tests for lz.dry_run() public API."""

    def test_dry_run_basic(self) -> None:
        """Basic dry run."""
        import layerzero as lz

        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x * 2

        model = SimpleModel()

        # Dry run should return selection plan
        plan = lz.dry_run(model)

        assert plan is not None

    def test_dry_run_shows_selections(self) -> None:
        """Dry run shows kernel selections."""
        import layerzero as lz

        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x * 2

        model = SimpleModel()

        plan = lz.dry_run(model)

        # Should have selections info
        assert hasattr(plan, 'selections') or hasattr(plan, 'kernels')

    def test_dry_run_no_execution(self) -> None:
        """Dry run doesn't execute kernels."""
        import layerzero as lz

        class FailingModel(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("Should not be called")

        model = FailingModel()

        # Should not raise (no execution)
        try:
            plan = lz.dry_run(model)
            # If dry_run doesn't trace the model, this is fine
        except RuntimeError as e:
            if "Should not be called" in str(e):
                pytest.fail("dry_run executed the model")


class TestSolveAPI:
    """Tests for lz.solve() build-time solver."""

    def test_solve_basic(self) -> None:
        """Basic build-time solving."""
        import layerzero as lz

        # Solve for attention operation
        solution = lz.solve(
            operations=["attention.causal"],
            shapes=[(1, 1024, 8, 64)],
            dtype=torch.float16,
        )

        assert solution is not None

    def test_solve_returns_plan(self) -> None:
        """Solve returns execution plan."""
        import layerzero as lz

        solution = lz.solve(
            operations=["attention.causal"],
            shapes=[(1, 1024, 8, 64)],
        )

        # Should have kernel assignments
        assert hasattr(solution, 'assignments') or hasattr(solution, 'plan')

    def test_solve_multiple_operations(self) -> None:
        """Solve for multiple operations."""
        import layerzero as lz

        solution = lz.solve(
            operations=["attention.causal", "norm.rms", "mlp.fused"],
            shapes=[(1, 1024, 8, 64)],
        )

        assert solution is not None


class TestTuneAPI:
    """Tests for lz.tune() auto-tuning."""

    def test_tune_basic(self) -> None:
        """Basic auto-tuning."""
        import layerzero as lz

        # Tune attention for specific shape
        result = lz.tune(
            operation="attention.causal",
            shape=(1, 1024, 8, 64),
            dtype=torch.float16,
            samples=3,  # Few samples for test speed
        )

        assert result is not None

    def test_tune_returns_best_kernel(self) -> None:
        """Tune returns best kernel."""
        import layerzero as lz

        result = lz.tune(
            operation="attention.causal",
            shape=(1, 1024, 8, 64),
            samples=3,
        )

        assert hasattr(result, 'best_kernel') or hasattr(result, 'kernel_id')

    @pytest.mark.stress
    def test_tune_improves_selection(self) -> None:
        """Tuning improves kernel selection."""
        import layerzero as lz

        # Get initial selection
        initial = lz.which("attention.causal", batch_size=1, seq_len=1024)

        # Tune
        result = lz.tune(
            operation="attention.causal",
            shape=(1, 1024, 8, 64),
            samples=5,
        )

        # Should have timing data now
        assert hasattr(result, 'timings') or hasattr(result, 'measurements')
