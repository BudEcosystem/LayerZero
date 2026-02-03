"""
Explain API

Provides lz.explain() API for debugging kernel selection decisions.
Returns detailed SelectionReport with all candidates and rejection reasons.
"""
from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

import torch

from layerzero.telemetry.selection_report import (
    KernelCandidate,
    SelectionReport,
)

if TYPE_CHECKING:
    pass


# Registry of known kernel adapters per operation
# This would be populated by the kernel registry
_OPERATION_CANDIDATES: dict[str, list[dict[str, Any]]] = {
    "attention": [
        {
            "kernel_id": "flash_attn.fwd",
            "source": "flash_attn",
            "priority": 90,
            "requirements": ["cuda", "fp16_or_bf16"],
        },
        {
            "kernel_id": "flashinfer.prefill",
            "source": "flashinfer",
            "priority": 85,
            "requirements": ["cuda", "fp16_or_bf16"],
        },
        {
            "kernel_id": "xformers.cutlass",
            "source": "xformers",
            "priority": 80,
            "requirements": ["cuda"],
        },
        {
            "kernel_id": "torch.sdpa",
            "source": "torch",
            "priority": 50,
            "requirements": [],
        },
    ],
    "matmul": [
        {
            "kernel_id": "torch.mm",
            "source": "torch",
            "priority": 50,
            "requirements": [],
        },
        {
            "kernel_id": "ipex.matmul",
            "source": "ipex",
            "priority": 45,
            "requirements": ["intel_cpu"],
        },
        {
            "kernel_id": "onednn.matmul",
            "source": "onednn",
            "priority": 40,
            "requirements": ["cpu"],
        },
    ],
    "layer_norm": [
        {
            "kernel_id": "liger.rms_norm",
            "source": "liger",
            "priority": 80,
            "requirements": ["cuda"],
        },
        {
            "kernel_id": "torch.layer_norm",
            "source": "torch",
            "priority": 50,
            "requirements": [],
        },
    ],
    "rope": [
        {
            "kernel_id": "liger.rope",
            "source": "liger",
            "priority": 80,
            "requirements": ["cuda"],
        },
        {
            "kernel_id": "torch.rope",
            "source": "torch",
            "priority": 50,
            "requirements": [],
        },
    ],
    "softmax": [
        {
            "kernel_id": "torch.softmax",
            "source": "torch",
            "priority": 50,
            "requirements": [],
        },
    ],
}


def _check_requirements(
    requirements: list[str],
    context: dict[str, Any],
) -> list[str]:
    """Check requirements and return list of unmet requirements.

    Args:
        requirements: List of requirement strings.
        context: Context with device, dtype info.

    Returns:
        List of unmet requirement codes.
    """
    unmet: list[str] = []

    device = context.get("device", "cpu")
    dtype = context.get("dtype", "float32")
    inferred_device = context.get("inferred_device", device)

    for req in requirements:
        if req == "cuda":
            if not torch.cuda.is_available():
                unmet.append("CUDA_NOT_AVAILABLE")
            elif "cuda" not in str(inferred_device).lower():
                unmet.append("DEVICE_MISMATCH")
        elif req == "fp16_or_bf16":
            if dtype not in ("float16", "bfloat16", torch.float16, torch.bfloat16):
                unmet.append("DTYPE_NOT_SUPPORTED")
        elif req == "intel_cpu":
            # Check for Intel CPU
            try:
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                if "intel" not in info.get("vendor_id_raw", "").lower():
                    unmet.append("VENDOR_MISMATCH")
            except ImportError:
                # Assume not Intel if can't check
                unmet.append("CANNOT_DETECT_VENDOR")
        elif req == "cpu":
            if "cpu" not in str(inferred_device).lower():
                unmet.append("DEVICE_MISMATCH")

    return unmet


def _score_candidate(
    candidate_info: dict[str, Any],
    context: dict[str, Any],
) -> float:
    """Calculate score for a candidate kernel.

    Args:
        candidate_info: Kernel candidate information.
        context: Selection context.

    Returns:
        Score between 0 and 1.
    """
    # Base score from priority (0-100 -> 0-1)
    priority = candidate_info.get("priority", 50)
    base_score = priority / 100.0

    # Adjust based on context matches
    # (In a real implementation, this would be more sophisticated)
    return min(1.0, max(0.0, base_score))


def explain(
    operation: str,
    *tensors: torch.Tensor,
    **kwargs: Any,
) -> SelectionReport:
    """Explain kernel selection for given operation and inputs.

    Simulates the kernel selection process and returns a detailed
    report of all candidates considered, their scores, and any
    rejection reasons.

    Args:
        operation: The operation type (e.g., "attention", "matmul").
        *tensors: Input tensors (optional, for context inference).
        **kwargs: Additional context (batch_size, seq_len, dtype, etc.).

    Returns:
        SelectionReport with full selection trace.

    Example:
        ```python
        # Simple explain
        report = lz.explain("attention")
        print(report.summary())

        # With tensors for context inference
        report = lz.explain("attention", query, key, value)
        for c in report.candidates:
            if c.rejected:
                print(f"{c.kernel_id}: {c.rejection_reasons}")
        ```
    """
    start_time = time.monotonic_ns()

    # Build context from kwargs and tensors
    context: dict[str, Any] = dict(kwargs)

    # Infer from tensors if provided
    if tensors:
        first_tensor = tensors[0]
        context["inferred_dtype"] = str(first_tensor.dtype)
        context["inferred_device"] = str(first_tensor.device)
        context["dtype"] = context.get("dtype", first_tensor.dtype)
        context["device"] = context.get("device", str(first_tensor.device))

        # Infer shape info
        if first_tensor.dim() >= 2:
            context["batch_size"] = context.get("batch_size", first_tensor.size(0))
        if first_tensor.dim() >= 3:
            context["seq_len"] = context.get("seq_len", first_tensor.size(-2))

    # Get candidate kernels for this operation
    candidate_infos = _OPERATION_CANDIDATES.get(operation, [])

    # Evaluate each candidate
    candidates: list[KernelCandidate] = []
    chosen_kernel_id: str | None = None
    best_score: float = -1.0

    for info in candidate_infos:
        kernel_id = info["kernel_id"]
        requirements = info.get("requirements", [])

        # Check requirements
        unmet = _check_requirements(requirements, context)

        if unmet:
            # Rejected
            candidates.append(KernelCandidate(
                kernel_id=kernel_id,
                score=None,
                rejected=True,
                rejection_reasons=tuple(unmet),
                metadata={"source": info.get("source", "unknown")},
            ))
        else:
            # Valid candidate, calculate score
            score = _score_candidate(info, context)
            candidates.append(KernelCandidate(
                kernel_id=kernel_id,
                score=score,
                rejected=False,
                rejection_reasons=(),
                metadata={"source": info.get("source", "unknown")},
            ))

            # Track best
            if score > best_score:
                best_score = score
                chosen_kernel_id = kernel_id

    # Calculate latency
    end_time = time.monotonic_ns()
    latency_ns = end_time - start_time

    # Build report
    return SelectionReport(
        operation=operation,
        chosen_kernel_id=chosen_kernel_id,
        candidates=tuple(candidates),
        selection_latency_ns=latency_ns,
        cache_hit=False,  # Explain never hits cache
        timestamp=time.time(),
        context=context,
    )
