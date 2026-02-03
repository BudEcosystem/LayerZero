"""
Tensor parallel invariance filtering.

This module provides:
- TPConfig: Configuration for tensor parallel invariance
- TPContext: Tensor parallel context
- TPInvarianceFilter: Filters kernels by TP invariance requirements
"""
from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import torch

from layerzero.reasons import (
    TP_INVARIANCE_REQUIRED,
    TP_INVARIANT_KERNEL_REQUIRED,
    TP_SIZE_EXCEEDED,
    Reason,
    ReasonCategory,
)

if TYPE_CHECKING:
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.models.selection_context import SelectionContext

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class TPConfig:
    """Configuration for tensor parallel invariance.

    Attributes:
        require_invariant: Require TP-invariant kernels.
        max_tp_size: Maximum allowed TP size.
        allow_non_invariant_inference: Allow non-invariant in inference.
        strict_training: Strict invariance requirement in training.
    """

    require_invariant: bool = False
    max_tp_size: int | None = None
    allow_non_invariant_inference: bool = True
    strict_training: bool = True


@dataclass(frozen=True)
class TPContext:
    """Tensor parallel context.

    Attributes:
        tp_size: Tensor parallel size.
        tp_rank: Tensor parallel rank.
        enabled: Whether TP is enabled.
    """

    tp_size: int
    tp_rank: int
    enabled: bool


def is_tp_enabled() -> bool:
    """Check if tensor parallelism is enabled.

    Checks environment variables and torch.distributed state.

    Returns:
        True if TP is enabled.
    """
    # Check environment variables
    tp_size = os.environ.get("TP_SIZE", "1")
    try:
        if int(tp_size) > 1:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return True
    except ValueError:
        pass

    return False


def get_tp_context() -> TPContext:
    """Get current tensor parallel context.

    Returns:
        TPContext with current TP settings.
    """
    if not is_tp_enabled():
        return TPContext(
            tp_size=1,
            tp_rank=0,
            enabled=False,
        )

    # Get from environment
    tp_size = int(os.environ.get("TP_SIZE", "1"))
    tp_rank = int(os.environ.get("TP_RANK", "0"))

    return TPContext(
        tp_size=tp_size,
        tp_rank=tp_rank,
        enabled=True,
    )


def require_tp_invariant(func: F) -> F:
    """Decorator to mark function as requiring TP invariance.

    When applied, the function is marked as requiring tensor parallel
    invariant execution. This helps the selection system choose
    appropriate kernels.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function with _tp_invariant attribute.

    Example:
        @require_tp_invariant
        def my_attention_kernel(...):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    wrapper._tp_invariant = True  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


class TPInvarianceFilter:
    """Filters kernels by tensor parallel invariance requirements.

    Ensures that kernels used in TP mode produce deterministic
    results regardless of the TP configuration.

    Example:
        config = TPConfig(require_invariant=True)
        filter_ = TPInvarianceFilter(config=config)

        passed, reason = filter_.check(kernel, ctx)
        if not passed:
            # Use fallback kernel
            ...
    """

    def __init__(self, config: TPConfig | None = None) -> None:
        """Initialize TP invariance filter.

        Args:
            config: TP configuration.
        """
        self._config = config or TPConfig()

    @property
    def config(self) -> TPConfig:
        """Get configuration."""
        return self._config

    def update_config(self, config: TPConfig) -> None:
        """Update configuration.

        Args:
            config: New configuration.
        """
        self._config = config

    def check(
        self,
        kernel: "KernelSpec",
        ctx: "SelectionContext",
    ) -> tuple[bool, Reason | None]:
        """Check if kernel meets TP invariance requirements.

        Args:
            kernel: Kernel specification.
            ctx: Selection context.

        Returns:
            Tuple of (passed, reason). Reason is None if passed.
        """
        # Get TP size from context
        tp_size = getattr(ctx, 'tp_size', 1)
        is_training = getattr(ctx, 'is_training', False)

        # No TP means no filtering
        if tp_size <= 1:
            return True, None

        # Check max TP size
        if self._config.max_tp_size is not None:
            if tp_size > self._config.max_tp_size:
                reason = Reason(
                    code=TP_SIZE_EXCEEDED,
                    message=(
                        f"TP size {tp_size} exceeds maximum "
                        f"{self._config.max_tp_size}"
                    ),
                    category=ReasonCategory.DISTRIBUTED,
                )
                logger.debug(
                    "TP filter rejected kernel %s: %s",
                    getattr(kernel, 'kernel_id', 'unknown'),
                    reason.message,
                )
                return False, reason

        # Check invariance requirement
        if not self._config.require_invariant:
            return True, None

        # Check if kernel is TP-invariant
        kernel_tp_invariant = getattr(kernel, 'tp_invariant', False)

        if not kernel_tp_invariant:
            # Non-invariant kernel
            if is_training and self._config.strict_training:
                reason = Reason(
                    code=TP_INVARIANT_KERNEL_REQUIRED,
                    message=(
                        f"Kernel {getattr(kernel, 'kernel_id', 'unknown')} "
                        f"is not TP-invariant but require_invariant=True "
                        f"and strict_training=True"
                    ),
                    category=ReasonCategory.DISTRIBUTED,
                )
                logger.debug(
                    "TP filter rejected kernel %s: %s",
                    getattr(kernel, 'kernel_id', 'unknown'),
                    reason.message,
                )
                return False, reason

            if not is_training and not self._config.allow_non_invariant_inference:
                reason = Reason(
                    code=TP_INVARIANT_KERNEL_REQUIRED,
                    message=(
                        f"Kernel {getattr(kernel, 'kernel_id', 'unknown')} "
                        f"is not TP-invariant and allow_non_invariant_inference=False"
                    ),
                    category=ReasonCategory.DISTRIBUTED,
                )
                logger.debug(
                    "TP filter rejected kernel %s: %s",
                    getattr(kernel, 'kernel_id', 'unknown'),
                    reason.message,
                )
                return False, reason

        return True, None


# Global filter instance
_global_tp_filter: TPInvarianceFilter | None = None


def get_global_tp_filter() -> TPInvarianceFilter:
    """Get global TP invariance filter.

    Returns:
        Global TPInvarianceFilter instance.
    """
    global _global_tp_filter
    if _global_tp_filter is None:
        _global_tp_filter = TPInvarianceFilter()
    return _global_tp_filter


def check_tp_invariance(
    kernel: "KernelSpec",
    ctx: "SelectionContext",
) -> tuple[bool, Reason | None]:
    """Check TP invariance using global filter.

    Args:
        kernel: Kernel specification.
        ctx: Selection context.

    Returns:
        Tuple of (passed, reason).
    """
    return get_global_tp_filter().check(kernel, ctx)
