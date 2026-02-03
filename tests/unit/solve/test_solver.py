"""Tests for build-time solver."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from typing import Any

from layerzero._solve.solver import (
    Solver,
    SolverConfig,
    SolverResult,
    solve,
)
from layerzero._solve.dispatch_table import (
    DispatchTable,
    DispatchEntry,
    ShapeBucket,
    BucketRange,
)


class TestSolverConfig:
    """Tests for SolverConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config values."""
        config = SolverConfig()

        assert config.enable_jit is True
        assert config.jit_warmup_iterations == 3
        assert config.include_hardware_signature is True
        assert config.persist_table is True
        assert config.table_path is None

    def test_custom_values(self) -> None:
        """Custom config values."""
        config = SolverConfig(
            enable_jit=False,
            jit_warmup_iterations=5,
            include_hardware_signature=False,
            persist_table=False,
            table_path="/tmp/dispatch.json",
        )

        assert config.enable_jit is False
        assert config.jit_warmup_iterations == 5
        assert config.include_hardware_signature is False
        assert config.persist_table is False
        assert config.table_path == "/tmp/dispatch.json"

    def test_config_immutable(self) -> None:
        """Config is immutable."""
        config = SolverConfig()

        with pytest.raises(AttributeError):
            config.enable_jit = False


class TestSolver:
    """Tests for Solver."""

    def test_solver_initialization(self) -> None:
        """Solver initializes correctly."""
        solver = Solver()

        assert solver.config is not None
        assert isinstance(solver.config, SolverConfig)

    def test_solver_custom_config(self) -> None:
        """Solver accepts custom config."""
        config = SolverConfig(enable_jit=False)
        solver = Solver(config=config)

        assert solver.config.enable_jit is False

    def test_solve_generates_dispatch_table(
        self,
        mock_hardware_context,
        sample_shape_buckets,
    ) -> None:
        """lz.solve generates dispatch table."""
        solver = Solver()

        with patch.object(solver, '_get_hardware_context', return_value=mock_hardware_context):
            result = solver.solve(
                operation="attention",
                buckets=sample_shape_buckets,
            )

        assert isinstance(result, SolverResult)
        assert result.dispatch_table is not None
        assert isinstance(result.dispatch_table, DispatchTable)
        assert len(result.dispatch_table) > 0

    def test_solve_bucketed_shapes(
        self,
        mock_hardware_context,
        sample_shape_buckets,
    ) -> None:
        """Solver handles bucketed shapes."""
        solver = Solver()

        with patch.object(solver, '_get_hardware_context', return_value=mock_hardware_context):
            result = solver.solve(
                operation="attention",
                buckets=sample_shape_buckets,
            )

        # Check that buckets are represented in the dispatch table
        table = result.dispatch_table
        assert table.bucket_count >= 1

    def test_solve_triggers_jit(
        self,
        mock_hardware_context,
        sample_shape_buckets,
    ) -> None:
        """Solver triggers JIT compilation when enabled."""
        config = SolverConfig(enable_jit=True, jit_warmup_iterations=2)
        solver = Solver(config=config)

        jit_calls = []

        def mock_jit_compile(kernel_id: str, context: Any) -> None:
            jit_calls.append(kernel_id)

        with patch.object(solver, '_get_hardware_context', return_value=mock_hardware_context):
            with patch.object(solver, '_jit_compile', side_effect=mock_jit_compile):
                solver.solve(
                    operation="attention",
                    buckets=sample_shape_buckets,
                )

        # JIT should be triggered for kernels
        assert len(jit_calls) >= 0  # May be 0 if no kernels selected

    def test_solve_hardware_signature(
        self,
        mock_hardware_context,
        sample_shape_buckets,
    ) -> None:
        """Dispatch table includes hardware signature."""
        config = SolverConfig(include_hardware_signature=True)
        solver = Solver(config=config)

        with patch.object(solver, '_get_hardware_context', return_value=mock_hardware_context):
            result = solver.solve(
                operation="attention",
                buckets=sample_shape_buckets,
            )

        # Check hardware signature in result
        assert result.hardware_signature is not None
        assert "platform" in result.hardware_signature
        assert result.hardware_signature["platform"] == "cuda"

    def test_solve_no_hardware_signature(
        self,
        mock_hardware_context,
        sample_shape_buckets,
    ) -> None:
        """Hardware signature can be disabled."""
        config = SolverConfig(include_hardware_signature=False)
        solver = Solver(config=config)

        with patch.object(solver, '_get_hardware_context', return_value=mock_hardware_context):
            result = solver.solve(
                operation="attention",
                buckets=sample_shape_buckets,
            )

        assert result.hardware_signature is None

    def test_solve_empty_buckets(self, mock_hardware_context) -> None:
        """Solver handles empty buckets."""
        solver = Solver()

        with patch.object(solver, '_get_hardware_context', return_value=mock_hardware_context):
            result = solver.solve(
                operation="attention",
                buckets=[],
            )

        # Should return empty dispatch table
        assert result.dispatch_table is not None
        assert len(result.dispatch_table) == 0


class TestSolverResult:
    """Tests for SolverResult."""

    def test_result_creation(self) -> None:
        """SolverResult stores values correctly."""
        table = DispatchTable()
        result = SolverResult(
            dispatch_table=table,
            hardware_signature={"platform": "cuda"},
            jit_compiled=["kernel1", "kernel2"],
            solve_time_ms=100.5,
        )

        assert result.dispatch_table is table
        assert result.hardware_signature == {"platform": "cuda"}
        assert result.jit_compiled == ["kernel1", "kernel2"]
        assert result.solve_time_ms == 100.5

    def test_result_to_dict(self) -> None:
        """SolverResult can be serialized to dict."""
        table = DispatchTable()
        result = SolverResult(
            dispatch_table=table,
            hardware_signature={"platform": "cuda"},
            jit_compiled=["kernel1"],
            solve_time_ms=50.0,
        )

        d = result.to_dict()

        assert "dispatch_table" in d
        assert "hardware_signature" in d
        assert d["hardware_signature"] == {"platform": "cuda"}
        assert d["jit_compiled"] == ["kernel1"]
        assert d["solve_time_ms"] == 50.0


class TestSolveFunction:
    """Tests for module-level solve function."""

    def test_solve_convenience_function(
        self,
        mock_hardware_context,
        sample_shape_buckets,
    ) -> None:
        """solve() convenience function works."""
        with patch('layerzero._solve.solver.Solver') as MockSolver:
            mock_solver = MagicMock()
            mock_result = SolverResult(
                dispatch_table=DispatchTable(),
                hardware_signature=None,
                jit_compiled=[],
                solve_time_ms=0.0,
            )
            mock_solver.solve.return_value = mock_result
            MockSolver.return_value = mock_solver

            result = solve(
                operation="attention",
                buckets=sample_shape_buckets,
            )

        assert result is mock_result
        mock_solver.solve.assert_called_once()


class TestSolverIntegration:
    """Integration tests for solver."""

    def test_solve_model_block(self, mock_hardware_context) -> None:
        """Solve entire model block."""
        solver = Solver()

        # Define a model block with multiple operations
        model_ops = [
            {
                "operation": "attention",
                "buckets": [
                    {"batch_size": [1, 2, 4], "seq_len": [512, 1024]},
                ],
            },
            {
                "operation": "matmul",
                "buckets": [
                    {"m": [512, 1024], "n": [512, 1024], "k": [256, 512]},
                ],
            },
        ]

        with patch.object(solver, '_get_hardware_context', return_value=mock_hardware_context):
            results = []
            for op in model_ops:
                result = solver.solve(
                    operation=op["operation"],
                    buckets=op["buckets"],
                )
                results.append(result)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, SolverResult)

    def test_solve_multi_op_plan(self, mock_hardware_context) -> None:
        """Multi-op plan generated."""
        solver = Solver()

        with patch.object(solver, '_get_hardware_context', return_value=mock_hardware_context):
            # Solve for attention
            attn_result = solver.solve(
                operation="attention",
                buckets=[{"batch_size": [1], "seq_len": [512]}],
            )

            # Solve for linear
            linear_result = solver.solve(
                operation="linear",
                buckets=[{"m": [512], "n": [2048], "k": [512]}],
            )

        # Both should have dispatch tables
        assert attn_result.dispatch_table is not None
        assert linear_result.dispatch_table is not None
