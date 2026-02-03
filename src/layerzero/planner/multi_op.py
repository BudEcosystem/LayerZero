"""Multi-operation planner for joint kernel selection.

This module implements plan-aware kernel selection that considers
the cost of layout/dtype transforms between consecutive operations
to find the globally optimal kernel assignment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlannerConfig:
    """Configuration for the multi-op planner.

    Attributes:
        transform_penalty: Multiplier applied to transform costs.
        layout_transform_cost_ms: Base cost for layout transforms in ms.
        dtype_transform_cost_ms: Base cost for dtype transforms in ms.
        optimize_jointly: If True, select kernels jointly; if False, greedily.
    """

    transform_penalty: float = 1.0
    layout_transform_cost_ms: float = 0.5
    dtype_transform_cost_ms: float = 0.3
    optimize_jointly: bool = True


@dataclass(frozen=True)
class TransformCost:
    """Cost of transforming between layouts/dtypes.

    Attributes:
        from_layout: Source layout format.
        to_layout: Target layout format.
        from_dtype: Source data type.
        to_dtype: Target data type.
        layout_cost_ms: Cost of layout transform in ms.
        dtype_cost_ms: Cost of dtype transform in ms.
    """

    from_layout: str
    to_layout: str
    from_dtype: str
    to_dtype: str
    layout_cost_ms: float = 0.0
    dtype_cost_ms: float = 0.0

    @property
    def requires_layout_transform(self) -> bool:
        """Check if layout transform is needed."""
        return self.from_layout != self.to_layout

    @property
    def requires_dtype_transform(self) -> bool:
        """Check if dtype transform is needed."""
        return self.from_dtype != self.to_dtype

    @property
    def total_cost_ms(self) -> float:
        """Total transform cost in milliseconds."""
        cost = 0.0
        if self.requires_layout_transform:
            cost += self.layout_cost_ms
        if self.requires_dtype_transform:
            cost += self.dtype_cost_ms
        return cost


@dataclass
class OpPlan:
    """Plan for a single operation.

    Attributes:
        op_type: Type of operation (e.g., "attention", "layernorm").
        kernel_id: Selected kernel identifier.
        input_layout: Input tensor layout.
        output_layout: Output tensor layout.
        estimated_latency_ms: Estimated kernel execution time.
        input_dtype: Input data type.
        output_dtype: Output data type.
        transform_cost_ms: Cost of any required transforms.
        metadata: Additional operation metadata.
    """

    op_type: str
    kernel_id: str
    input_layout: str
    output_layout: str
    estimated_latency_ms: float
    input_dtype: str = "float16"
    output_dtype: str = "float16"
    transform_cost_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "op_type": self.op_type,
            "kernel_id": self.kernel_id,
            "input_layout": self.input_layout,
            "output_layout": self.output_layout,
            "estimated_latency_ms": self.estimated_latency_ms,
            "input_dtype": self.input_dtype,
            "output_dtype": self.output_dtype,
            "transform_cost_ms": self.transform_cost_ms,
            "metadata": self.metadata,
        }


@dataclass
class MultiOpPlan:
    """Plan for multiple consecutive operations.

    Attributes:
        ops: List of operation plans.
        metadata: Additional plan metadata.
    """

    ops: list[OpPlan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        """Total estimated latency including transforms."""
        return sum(op.estimated_latency_ms for op in self.ops)

    @property
    def total_transform_cost_ms(self) -> float:
        """Total transform cost across all operations."""
        return sum(op.transform_cost_ms for op in self.ops)

    @property
    def total_cost_ms(self) -> float:
        """Total cost including kernel latency and transforms."""
        return self.total_latency_ms + self.total_transform_cost_ms

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "ops": [op.to_dict() for op in self.ops],
            "total_latency_ms": self.total_latency_ms,
            "total_transform_cost_ms": self.total_transform_cost_ms,
            "total_cost_ms": self.total_cost_ms,
            "metadata": self.metadata,
        }


class MultiOpPlanner:
    """Planner for joint kernel selection across multiple operations.

    This planner considers the cost of layout/dtype transforms between
    consecutive operations to find the globally optimal kernel assignment.
    When optimize_jointly=True, it evaluates all combinations of kernels
    to minimize total cost (kernel latency + transform overhead).
    """

    def __init__(
        self,
        config: PlannerConfig | None = None,
    ) -> None:
        """Initialize the multi-op planner.

        Args:
            config: Planner configuration. Uses defaults if None.
        """
        self._config = config or PlannerConfig()
        self._lock = RLock()

        logger.debug(
            "MultiOpPlanner initialized with config: %s",
            self._config,
        )

    @property
    def config(self) -> PlannerConfig:
        """Get planner configuration."""
        return self._config

    def plan(self, operations: list[dict[str, Any]]) -> MultiOpPlan:
        """Create an execution plan for a sequence of operations.

        Args:
            operations: List of operation specifications. Each spec should
                contain: op_type, input_layout, output_layout, input_dtype,
                output_dtype, and optional config.

        Returns:
            MultiOpPlan with selected kernels for each operation.
        """
        with self._lock:
            if not operations:
                logger.debug("Empty operations list, returning empty plan")
                return MultiOpPlan()

            # Get candidate kernels for each operation
            candidates_per_op = []
            for op in operations:
                candidates = self._get_candidate_kernels(op)
                if not candidates:
                    logger.warning(
                        "No candidate kernels for operation: %s",
                        op.get("op_type", "unknown"),
                    )
                    # Create a dummy candidate as fallback
                    candidates = [
                        {
                            "kernel_id": f"{op.get('op_type', 'unknown')}_fallback",
                            "input_layout": op.get("input_layout", "BSHD"),
                            "output_layout": op.get("output_layout", "BSHD"),
                            "latency_ms": 1.0,
                        }
                    ]
                candidates_per_op.append(candidates)

            if self._config.optimize_jointly:
                return self._plan_jointly(operations, candidates_per_op)
            else:
                return self._plan_greedy(operations, candidates_per_op)

    def _plan_jointly(
        self,
        operations: list[dict[str, Any]],
        candidates_per_op: list[list[dict[str, Any]]],
    ) -> MultiOpPlan:
        """Select kernels jointly to minimize total cost.

        This evaluates all combinations of kernels across operations
        and selects the combination with minimum total cost.
        """
        best_plan: MultiOpPlan | None = None
        best_cost = float("inf")

        # Enumerate all combinations of kernels
        for combination in product(*candidates_per_op):
            plan = self._evaluate_combination(operations, list(combination))
            cost = plan.total_cost_ms

            if cost < best_cost:
                best_cost = cost
                best_plan = plan

        if best_plan is None:
            logger.error("No valid plan found for operations")
            return MultiOpPlan()

        logger.debug(
            "Joint planning selected plan with total cost: %.3f ms",
            best_cost,
        )
        return best_plan

    def _plan_greedy(
        self,
        operations: list[dict[str, Any]],
        candidates_per_op: list[list[dict[str, Any]]],
    ) -> MultiOpPlan:
        """Select kernels greedily, one at a time.

        This selects the best kernel for each operation considering
        only the transform cost from the previous operation.
        """
        op_plans: list[OpPlan] = []
        prev_layout: str | None = None
        prev_dtype: str | None = None

        for i, (op, candidates) in enumerate(zip(operations, candidates_per_op)):
            best_kernel: dict[str, Any] | None = None
            best_cost = float("inf")

            for kernel in candidates:
                # Calculate transform cost from previous op
                transform_cost = 0.0
                if prev_layout is not None:
                    if prev_layout != kernel.get("input_layout", "BSHD"):
                        transform_cost += (
                            self._config.layout_transform_cost_ms
                            * self._config.transform_penalty
                        )
                    if prev_dtype != op.get("input_dtype", "float16"):
                        transform_cost += (
                            self._config.dtype_transform_cost_ms
                            * self._config.transform_penalty
                        )

                total_cost = kernel.get("latency_ms", 1.0) + transform_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_kernel = kernel

            if best_kernel is None:
                logger.warning("No kernel selected for operation %d", i)
                continue

            # Calculate actual transform cost for the selected kernel
            transform_cost_ms = 0.0
            if prev_layout is not None:
                if prev_layout != best_kernel.get("input_layout", "BSHD"):
                    transform_cost_ms += (
                        self._config.layout_transform_cost_ms
                        * self._config.transform_penalty
                    )
                if prev_dtype != op.get("input_dtype", "float16"):
                    transform_cost_ms += (
                        self._config.dtype_transform_cost_ms
                        * self._config.transform_penalty
                    )

            op_plan = OpPlan(
                op_type=op.get("op_type", "unknown"),
                kernel_id=best_kernel.get("kernel_id", "unknown"),
                input_layout=best_kernel.get("input_layout", "BSHD"),
                output_layout=best_kernel.get("output_layout", "BSHD"),
                estimated_latency_ms=best_kernel.get("latency_ms", 1.0),
                input_dtype=op.get("input_dtype", "float16"),
                output_dtype=op.get("output_dtype", "float16"),
                transform_cost_ms=transform_cost_ms,
            )
            op_plans.append(op_plan)

            prev_layout = best_kernel.get("output_layout", "BSHD")
            prev_dtype = op.get("output_dtype", "float16")

        return MultiOpPlan(ops=op_plans)

    def _evaluate_combination(
        self,
        operations: list[dict[str, Any]],
        kernels: list[dict[str, Any]],
    ) -> MultiOpPlan:
        """Evaluate a specific combination of kernels.

        Args:
            operations: Operation specifications.
            kernels: Selected kernel for each operation.

        Returns:
            MultiOpPlan with computed costs.
        """
        op_plans: list[OpPlan] = []
        prev_layout: str | None = None
        prev_dtype: str | None = None

        for op, kernel in zip(operations, kernels):
            # Calculate transform cost from previous operation
            transform_cost_ms = 0.0
            if prev_layout is not None:
                kernel_input_layout = kernel.get("input_layout", "BSHD")
                if prev_layout != kernel_input_layout:
                    transform_cost_ms += (
                        self._config.layout_transform_cost_ms
                        * self._config.transform_penalty
                    )

                op_input_dtype = op.get("input_dtype", "float16")
                if prev_dtype != op_input_dtype:
                    transform_cost_ms += (
                        self._config.dtype_transform_cost_ms
                        * self._config.transform_penalty
                    )

            op_plan = OpPlan(
                op_type=op.get("op_type", "unknown"),
                kernel_id=kernel.get("kernel_id", "unknown"),
                input_layout=kernel.get("input_layout", "BSHD"),
                output_layout=kernel.get("output_layout", "BSHD"),
                estimated_latency_ms=kernel.get("latency_ms", 1.0),
                input_dtype=op.get("input_dtype", "float16"),
                output_dtype=op.get("output_dtype", "float16"),
                transform_cost_ms=transform_cost_ms,
            )
            op_plans.append(op_plan)

            prev_layout = kernel.get("output_layout", "BSHD")
            prev_dtype = op.get("output_dtype", "float16")

        return MultiOpPlan(ops=op_plans)

    def _get_candidate_kernels(
        self,
        operation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Get candidate kernels for an operation.

        This is a stub that should be overridden or patched in tests.
        In production, this would query the kernel registry.

        Args:
            operation: Operation specification.

        Returns:
            List of candidate kernel specifications.
        """
        # Default implementation returns empty list
        # This is designed to be patched/mocked in tests
        return []

    def _compute_transform_cost(
        self,
        from_layout: str,
        to_layout: str,
        from_dtype: str,
        to_dtype: str,
    ) -> TransformCost:
        """Compute the cost of transforming between layouts/dtypes.

        Args:
            from_layout: Source layout format.
            to_layout: Target layout format.
            from_dtype: Source data type.
            to_dtype: Target data type.

        Returns:
            TransformCost with computed costs.
        """
        layout_cost = 0.0
        if from_layout != to_layout:
            layout_cost = (
                self._config.layout_transform_cost_ms
                * self._config.transform_penalty
            )

        dtype_cost = 0.0
        if from_dtype != to_dtype:
            dtype_cost = (
                self._config.dtype_transform_cost_ms
                * self._config.transform_penalty
            )

        return TransformCost(
            from_layout=from_layout,
            to_layout=to_layout,
            from_dtype=from_dtype,
            to_dtype=to_dtype,
            layout_cost_ms=layout_cost,
            dtype_cost_ms=dtype_cost,
        )
