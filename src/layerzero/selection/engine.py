"""
LayerZero Selection Engine

Core kernel selection engine implementing the filter → score → select → cache pipeline.
"""
from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

from layerzero.policy.engine import RuleEngine
from layerzero.policy.policy import Policy
from layerzero.selection.cache import SelectionCache
from layerzero.selection.filter import FilterPhase
from layerzero.selection.scorer import ScoringPhase

if TYPE_CHECKING:
    from layerzero.models.execution_plan import ExecutionPlan, SelectionReport
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.models.selection_context import SelectionContext
    from layerzero.reasons import Reason
    from layerzero.registry.backend_registry import BackendRegistry
    from layerzero.registry.kernel_registry import KernelRegistry


class NoKernelAvailableError(Exception):
    """Raised when no kernel is available for a selection context.

    Contains information about what candidates were considered and
    why they were filtered out.
    """

    def __init__(
        self,
        operation: str,
        filtered_out: dict[str, list["Reason"]] | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize error.

        Args:
            operation: Operation that had no available kernel.
            filtered_out: Dict of kernel_id -> reasons for filtering.
            message: Optional custom message.
        """
        self.operation = operation
        self.filtered_out = filtered_out or {}

        if message:
            super().__init__(message)
        else:
            if filtered_out:
                super().__init__(
                    f"No kernel available for operation '{operation}'. "
                    f"{len(filtered_out)} candidate(s) were filtered out."
                )
            else:
                super().__init__(
                    f"No kernel available for operation '{operation}'. "
                    "No candidates found."
                )


class SelectionEngine:
    """Core kernel selection engine.

    Implements the filter → score → select → cache pipeline.
    Thread-safe for concurrent selection requests.

    Pipeline stages:
    1. Check policy lock (if any lock rule matches, return locked kernel)
    2. Check cache (if enabled and cache hit, return cached plan)
    3. Get candidates (from kernel registry by operation)
    4. Apply policy filters (allow/deny rules)
    5. Filter by compatibility (platform, dtype, shape, etc.)
    6. Score remaining candidates (priority + boosts - transform cost)
    7. Select highest-scoring kernel
    8. Cache result (if enabled)
    9. Return ExecutionPlan

    Attributes:
        policy: Current selection policy.
        cache: Selection cache instance.
    """

    __slots__ = (
        "_kernel_registry",
        "_backend_registry",
        "_policy",
        "_rule_engine",
        "_cache",
        "_filter",
        "_scorer",
    )

    def __init__(
        self,
        kernel_registry: "KernelRegistry",
        backend_registry: "BackendRegistry",
        policy: Policy | None = None,
        cache: SelectionCache | None = None,
    ) -> None:
        """Initialize selection engine.

        Args:
            kernel_registry: Registry of available kernels.
            backend_registry: Registry of backends (for health checks).
            policy: Selection policy. If None, empty policy is created.
            cache: Selection cache. If None, default cache is created.
        """
        self._kernel_registry = kernel_registry
        self._backend_registry = backend_registry

        # Use empty policy if not provided
        self._policy = policy or Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(),
        )

        self._rule_engine = RuleEngine(self._policy)
        self._cache = cache or SelectionCache()
        self._filter = FilterPhase()
        self._scorer = ScoringPhase(self._rule_engine)

    @property
    def policy(self) -> Policy:
        """Get current policy."""
        return self._policy

    @property
    def cache(self) -> SelectionCache:
        """Get selection cache."""
        return self._cache

    @property
    def kernel_registry(self) -> "KernelRegistry":
        """Get the kernel registry.

        Provides read-only access to the kernel registry used by this engine.
        """
        return self._kernel_registry

    @property
    def backend_registry(self) -> "BackendRegistry":
        """Get the backend registry.

        Provides read-only access to the backend registry used by this engine.
        """
        return self._backend_registry

    def update_policy(self, policy: Policy) -> None:
        """Update selection policy.

        Invalidates cache entries for old policy hash.

        Args:
            policy: New policy to use.
        """
        # Invalidate old policy's cache entries
        old_hash = self._policy.policy_hash
        self._cache.invalidate(old_hash)

        # Update policy and rule engine
        self._policy = policy
        self._rule_engine = RuleEngine(self._policy)
        self._scorer = ScoringPhase(self._rule_engine)

    def select(
        self,
        ctx: "SelectionContext",
        *,
        use_cache: bool = True,
        debug: bool = False,
    ) -> "ExecutionPlan":
        """Select best kernel for context.

        Args:
            ctx: Selection context describing the operation.
            use_cache: Whether to use cached selections.
            debug: Whether to include debug info in plan.

        Returns:
            ExecutionPlan with selected kernel.

        Raises:
            NoKernelAvailableError: If no kernel is available.
        """
        from layerzero.models.execution_plan import ExecutionPlan, SelectionReport

        start_time = time.perf_counter_ns()

        # 1. Check for policy lock
        locked_kernel_id = self._rule_engine.get_locked_kernel(ctx)
        if locked_kernel_id:
            kernel = self._kernel_registry.get(locked_kernel_id)
            if kernel:
                return self._make_plan(
                    kernel,
                    ctx,
                    "policy_lock",
                    debug=debug,
                    start_time=start_time,
                )

        # 2. Check cache
        cache_key = ctx.cache_key()
        policy_hash = self._policy.policy_hash

        if use_cache:
            cached = self._cache.get(cache_key, policy_hash)
            if cached:
                # Return cached plan with cached=True
                return ExecutionPlan(
                    kernel_id=cached.kernel_id,
                    kernel_spec=cached.kernel_spec,
                    pre_transforms=cached.pre_transforms,
                    post_transforms=cached.post_transforms,
                    debug_info=cached.debug_info,
                    cached=True,
                    cache_key=cache_key,
                )

        # 3. Get candidates for operation
        candidates = self._kernel_registry.get_by_operation(ctx.operation)
        if not candidates:
            raise NoKernelAvailableError(ctx.operation)

        candidate_ids = [k.kernel_id for k in candidates]

        # 4. Apply policy filters (allow/deny)
        allowed_ids = self._rule_engine.filter_kernels(candidate_ids, ctx)
        candidates = [k for k in candidates if k.kernel_id in allowed_ids]

        if not candidates:
            raise NoKernelAvailableError(
                ctx.operation,
                message=f"No kernel available for operation '{ctx.operation}'. "
                        "All candidates denied by policy.",
            )

        # 5. Filter by context compatibility
        valid, filtered_out = self._filter.filter(candidates, ctx)

        if not valid:
            raise NoKernelAvailableError(ctx.operation, filtered_out)

        # 6. Score remaining candidates
        scores = self._scorer.score(valid, ctx)

        # 7. Select highest-scoring kernel
        best_kernel = max(valid, key=lambda k: scores[k.kernel_id])

        # 8. Build execution plan
        plan = self._make_plan(
            best_kernel,
            ctx,
            "highest_score",
            scores=scores,
            candidates=candidate_ids,
            filtered_out=filtered_out,
            debug=debug,
            start_time=start_time,
            cache_key=cache_key,
        )

        # 9. Cache result
        if use_cache:
            self._cache.put(cache_key, policy_hash, plan)

        return plan

    def select_batch(
        self,
        contexts: list["SelectionContext"],
        *,
        use_cache: bool = True,
        debug: bool = False,
    ) -> list["ExecutionPlan"]:
        """Select kernels for multiple contexts.

        Args:
            contexts: List of selection contexts.
            use_cache: Whether to use cached selections.
            debug: Whether to include debug info.

        Returns:
            List of ExecutionPlans, one per context.
        """
        return [
            self.select(ctx, use_cache=use_cache, debug=debug)
            for ctx in contexts
        ]

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache hit_rate, size, etc.
        """
        return self._cache.stats()

    def _make_plan(
        self,
        kernel: "KernelSpec",
        ctx: "SelectionContext",
        selection_reason: str,
        *,
        scores: dict[str, float] | None = None,
        candidates: list[str] | None = None,
        filtered_out: dict[str, list["Reason"]] | None = None,
        debug: bool = False,
        start_time: int | None = None,
        cache_key: str | None = None,
    ) -> "ExecutionPlan":
        """Create ExecutionPlan from selected kernel.

        Args:
            kernel: Selected kernel spec.
            ctx: Selection context.
            selection_reason: Reason for selection.
            scores: Score dict (for debug report).
            candidates: All candidate IDs (for debug report).
            filtered_out: Filtered kernels and reasons (for debug report).
            debug: Whether to include debug report.
            start_time: Start time in nanoseconds (for timing).
            cache_key: Cache key for this selection.

        Returns:
            ExecutionPlan with optional debug info.
        """
        from layerzero.models.execution_plan import ExecutionPlan, SelectionReport

        debug_info: SelectionReport | None = None

        if debug:
            # Calculate selection time
            selection_time_us = 0
            if start_time is not None:
                elapsed_ns = time.perf_counter_ns() - start_time
                selection_time_us = elapsed_ns // 1000

            # Build context summary
            context_summary = {
                "operation": ctx.operation,
                "dtype": str(ctx.dtype).replace("torch.", ""),
                "batch_size": ctx.batch_size,
                "head_dim": ctx.head_dim,
                "seq_len_q": ctx.seq_len_q,
                "layout": ctx.layout.value,
            }

            debug_info = SelectionReport(
                operation=ctx.operation,
                context_summary=context_summary,
                candidates=tuple(candidates or []),
                filtered_out=filtered_out or {},
                scores=scores or {},
                selected_kernel=kernel.kernel_id,
                selection_reason=selection_reason,
                selection_time_us=selection_time_us,
            )

        return ExecutionPlan(
            kernel_id=kernel.kernel_id,
            kernel_spec=kernel,
            pre_transforms=(),
            post_transforms=(),
            debug_info=debug_info,
            cached=False,
            cache_key=cache_key,
        )
