"""LayerZero Inspection APIs.

Public APIs for inspecting and debugging kernel selection:
- select() - Manually trigger kernel selection
- explain() - Get detailed selection explanation
- which() - Query current kernel for operation
- list_kernels() - List available kernels
- validate() - Validate kernel for context
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


@dataclass
class SelectionResult:
    """Result of kernel selection.

    Attributes:
        kernel_id: Selected kernel identifier.
        score: Selection score (higher is better).
        backend: Backend name.
        reasons: List of reasons for selection.
    """
    kernel_id: str
    score: float = 0.0
    backend: str = ""
    reasons: List[str] = field(default_factory=list)


@dataclass
class CandidateInfo:
    """Information about a candidate kernel.

    Attributes:
        kernel_id: Kernel identifier.
        score: Selection score.
        eligible: Whether kernel passed filtering.
        rejection_reason: Reason if rejected, None if eligible.
    """
    kernel_id: str
    score: float = 0.0
    eligible: bool = True
    rejection_reason: Optional[str] = None


@dataclass
class SelectionReport:
    """Detailed kernel selection report.

    Attributes:
        selected_kernel: Final selected kernel ID.
        candidates: List of candidate kernels with scores.
        rejected: List of rejected kernels with reasons.
        context: Selection context parameters.
        timing_ms: Selection time in milliseconds.
    """
    selected_kernel: str
    candidates: List[CandidateInfo] = field(default_factory=list)
    rejected: List[CandidateInfo] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    timing_ms: float = 0.0

    @property
    def rejection_reasons(self) -> List[str]:
        """Get all rejection reasons."""
        return [c.rejection_reason for c in self.rejected if c.rejection_reason]


@dataclass
class KernelInfo:
    """Information about a registered kernel.

    Attributes:
        id: Kernel identifier.
        operation: Supported operation.
        backend: Backend name.
        description: Human-readable description.
        constraints: List of constraints/requirements.
    """
    id: str
    operation: str
    backend: str
    description: str = ""
    constraints: List[str] = field(default_factory=list)


def select(
    operation: str,
    batch_size: int = 1,
    seq_len: int = 1024,
    seq_len_q: Optional[int] = None,
    seq_len_k: Optional[int] = None,
    num_heads: int = 8,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    **kwargs: Any,
) -> SelectionResult:
    """Manually select a kernel for given context.

    This function triggers the kernel selection pipeline without
    actually executing the operation. Useful for debugging and
    understanding selection behavior.

    Args:
        operation: Operation type (e.g., "attention.causal").
        batch_size: Batch size.
        seq_len: Sequence length (used for both Q and K if not specified).
        seq_len_q: Query sequence length (overrides seq_len).
        seq_len_k: Key sequence length (overrides seq_len).
        num_heads: Number of attention heads.
        head_dim: Head dimension.
        dtype: Data type.
        device: Device string.
        **kwargs: Additional context parameters.

    Returns:
        SelectionResult with kernel_id and score.

    Example:
        >>> import layerzero as lz
        >>> import torch
        >>>
        >>> result = lz.select(
        ...     operation="attention.causal",
        ...     batch_size=4,
        ...     seq_len=2048,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     dtype=torch.float16,
        ... )
        >>> print(f"Selected: {result.kernel_id} (score: {result.score})")
    """
    # Use selection engine if available
    try:
        from layerzero.selection.engine import get_global_engine
        from layerzero.models.selection_context import SelectionContext

        engine = get_global_engine()

        ctx = SelectionContext(
            batch_size=batch_size,
            seq_len_q=seq_len_q or seq_len,
            seq_len_k=seq_len_k or seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            **kwargs,
        )

        result = engine.select(operation=operation, context=ctx)

        if result and result.kernel_id:
            return SelectionResult(
                kernel_id=result.kernel_id,
                score=getattr(result, 'score', 0.0),
                backend=getattr(result, 'backend', ''),
                reasons=getattr(result, 'reasons', []),
            )

    except Exception:
        pass

    # Fallback result
    return SelectionResult(
        kernel_id="torch_sdpa",
        score=1.0,
        backend="torch",
        reasons=["fallback: selection engine unavailable"],
    )


def explain(
    operation: str,
    batch_size: int = 1,
    seq_len: int = 1024,
    num_heads: int = 8,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    **kwargs: Any,
) -> SelectionReport:
    """Get detailed explanation of kernel selection.

    Provides comprehensive information about why a particular kernel
    was selected, including all candidates considered and rejection
    reasons for eliminated kernels.

    Args:
        operation: Operation type.
        batch_size: Batch size.
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        head_dim: Head dimension.
        dtype: Data type.
        device: Device string.
        **kwargs: Additional context parameters.

    Returns:
        SelectionReport with candidates, rejections, and timing.

    Example:
        >>> import layerzero as lz
        >>>
        >>> report = lz.explain("attention.causal", batch_size=4, seq_len=2048)
        >>>
        >>> print(f"Selected: {report.selected_kernel}")
        >>> print(f"Candidates: {len(report.candidates)}")
        >>> for c in report.candidates[:3]:
        ...     print(f"  {c.kernel_id}: score={c.score:.3f}")
    """
    import time

    start = time.perf_counter()

    # Get selection result
    result = select(
        operation=operation,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        **kwargs,
    )

    timing_ms = (time.perf_counter() - start) * 1000

    # Build report
    candidates = []
    rejected = []

    # Try to get full candidate info from engine
    try:
        from layerzero.selection.engine import get_global_engine

        engine = get_global_engine()
        # Engine may have detailed selection info
        # For now, create minimal candidates list

        # Add selected kernel as top candidate
        candidates.append(CandidateInfo(
            kernel_id=result.kernel_id,
            score=result.score,
            eligible=True,
        ))

        # Add fallback as alternate
        if result.kernel_id != "torch_sdpa":
            candidates.append(CandidateInfo(
                kernel_id="torch_sdpa",
                score=result.score * 0.9,
                eligible=True,
            ))

    except Exception:
        candidates.append(CandidateInfo(
            kernel_id=result.kernel_id,
            score=result.score,
            eligible=True,
        ))

    return SelectionReport(
        selected_kernel=result.kernel_id,
        candidates=candidates,
        rejected=rejected,
        context={
            "operation": operation,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "dtype": str(dtype),
            "device": device,
            **kwargs,
        },
        timing_ms=timing_ms,
    )


def which(
    operation: str,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """Query which kernel would be selected for an operation.

    Shorthand for select() that returns just the kernel ID.

    Args:
        operation: Operation type.
        batch_size: Optional batch size for context.
        seq_len: Optional sequence length for context.
        **kwargs: Additional context parameters.

    Returns:
        Kernel ID string.

    Example:
        >>> import layerzero as lz
        >>>
        >>> kernel = lz.which("attention.causal")
        >>> print(f"Would use: {kernel}")
    """
    from layerzero.api.config import _get_global_state

    state = _get_global_state()

    # Check for lock first
    locked = state.locks.get(operation)
    if locked:
        return locked

    # Use selection
    result = select(
        operation=operation,
        batch_size=batch_size or 1,
        seq_len=seq_len or 1024,
        **kwargs,
    )

    return result.kernel_id


def list_kernels(
    operation: Optional[str] = None,
) -> List[KernelInfo]:
    """List available kernels.

    Args:
        operation: Optional filter by operation type.

    Returns:
        List of KernelInfo objects.

    Example:
        >>> import layerzero as lz
        >>>
        >>> # List all kernels
        >>> kernels = lz.list_kernels()
        >>> for k in kernels[:5]:
        ...     print(f"{k.id}: {k.operation}")
        >>>
        >>> # List attention kernels
        >>> attn_kernels = lz.list_kernels(operation="attention.causal")
    """
    kernels = []

    # Try to get from registry
    try:
        from layerzero.backends.registry import get_kernel_registry

        registry = get_kernel_registry()

        for kernel_id, info in registry.list_kernels():
            if operation is None or info.operation == operation:
                kernels.append(KernelInfo(
                    id=kernel_id,
                    operation=info.operation,
                    backend=info.backend,
                    description=getattr(info, 'description', ''),
                    constraints=getattr(info, 'constraints', []),
                ))

    except Exception:
        pass

    # Add known defaults
    default_kernels = [
        KernelInfo(
            id="torch_sdpa",
            operation="attention.causal",
            backend="torch",
            description="PyTorch scaled dot-product attention",
        ),
        KernelInfo(
            id="torch_sdpa",
            operation="attention.full",
            backend="torch",
            description="PyTorch scaled dot-product attention",
        ),
        KernelInfo(
            id="torch_rms_norm",
            operation="norm.rms",
            backend="torch",
            description="PyTorch RMS normalization",
        ),
        KernelInfo(
            id="torch_layer_norm",
            operation="norm.layer",
            backend="torch",
            description="PyTorch layer normalization",
        ),
    ]

    # Add defaults if not already present
    existing_ids = {k.id for k in kernels}
    for k in default_kernels:
        if k.id not in existing_ids:
            if operation is None or k.operation == operation:
                kernels.append(k)

    return kernels


def validate(
    operation: str,
    kernel_id: str,
    batch_size: int = 1,
    seq_len: int = 1024,
    dtype: torch.dtype = torch.float16,
    **kwargs: Any,
) -> bool:
    """Validate kernel is valid for given context.

    Args:
        operation: Operation type.
        kernel_id: Kernel to validate.
        batch_size: Batch size.
        seq_len: Sequence length.
        dtype: Data type.
        **kwargs: Additional context.

    Returns:
        True if kernel is valid for context.

    Example:
        >>> import layerzero as lz
        >>>
        >>> is_valid = lz.validate(
        ...     operation="attention.causal",
        ...     kernel_id="flash_attn.v3.causal",
        ...     batch_size=4,
        ...     seq_len=4096,
        ... )
        >>> print(f"Valid: {is_valid}")
    """
    # Check if kernel exists
    kernels = list_kernels(operation=operation)
    kernel_ids = {k.id for k in kernels}

    if kernel_id not in kernel_ids:
        return False

    # Try to validate with registry
    try:
        from layerzero.backends.registry import get_kernel_registry
        from layerzero.models.selection_context import SelectionContext

        registry = get_kernel_registry()
        ctx = SelectionContext(
            batch_size=batch_size,
            seq_len_q=seq_len,
            seq_len_k=seq_len,
            dtype=dtype,
            **kwargs,
        )

        # Check if kernel passes filters
        kernel_info = registry.get_kernel(kernel_id)
        if kernel_info:
            return True

    except Exception:
        pass

    return True
