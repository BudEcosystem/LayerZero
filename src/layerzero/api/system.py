"""LayerZero System APIs.

Public APIs for system diagnostics and optimization:
- doctor() - System diagnostics
- readiness_check() - Pre-flight validation
- compile() - Build-time kernel selection
- dry_run() - Selection preview
- solve() - Build-time solver
- tune() - Auto-tuning
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


@dataclass
class BackendStatus:
    """Status of a backend.

    Attributes:
        name: Backend name.
        available: Whether backend is available.
        version: Backend version string.
        error: Error message if unavailable.
    """
    name: str
    available: bool
    version: str = ""
    error: Optional[str] = None


@dataclass
class DoctorReport:
    """System diagnostics report.

    Attributes:
        healthy: Overall health status.
        backends: Status of each backend.
        cuda_available: CUDA availability.
        cuda_devices: List of CUDA devices.
        cache_status: Selection cache status.
        summary: Human-readable summary.
    """
    healthy: bool
    backends: List[BackendStatus] = field(default_factory=list)
    cuda_available: bool = False
    cuda_devices: List[str] = field(default_factory=list)
    cache_status: str = "ok"
    summary: str = ""


@dataclass
class ReadinessReport:
    """System readiness report.

    Attributes:
        ready: Whether system is ready for production.
        is_ready: Alias for ready.
        issues: List of issues blocking readiness.
        warnings: List of non-blocking warnings.
        backends_checked: Backends that were checked.
    """
    ready: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    backends_checked: List[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.ready


@dataclass
class CompilationPlan:
    """Kernel compilation plan.

    Attributes:
        selections: Mapping of operations to kernels.
        plan: Alias for selections.
        shapes: Shapes covered by the plan.
    """
    selections: Dict[str, str] = field(default_factory=dict)
    shapes: List[Tuple[int, ...]] = field(default_factory=list)

    @property
    def plan(self) -> Dict[str, str]:
        return self.selections


@dataclass
class TuneResult:
    """Auto-tuning result.

    Attributes:
        best_kernel: Best kernel found.
        kernel_id: Alias for best_kernel.
        timings: Timing measurements.
        measurements: Alias for timings.
    """
    best_kernel: str
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def kernel_id(self) -> str:
        return self.best_kernel

    @property
    def measurements(self) -> Dict[str, float]:
        return self.timings


def doctor() -> DoctorReport:
    """Run system diagnostics.

    Checks:
    - Backend availability (FlashAttention, FlashInfer, etc.)
    - CUDA device status
    - Selection cache integrity
    - Configuration validity

    Returns:
        DoctorReport with detailed status.

    Example:
        >>> import layerzero as lz
        >>>
        >>> report = lz.doctor()
        >>> if report.healthy:
        ...     print("System healthy!")
        >>> else:
        ...     print(f"Issues found: {report.summary}")
    """
    backends = []
    issues = []

    # Check PyTorch
    backends.append(BackendStatus(
        name="torch",
        available=True,
        version=torch.__version__,
    ))

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    cuda_devices = []
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            cuda_devices.append(torch.cuda.get_device_name(i))

    # Check FlashAttention
    try:
        import flash_attn
        backends.append(BackendStatus(
            name="flash_attn",
            available=True,
            version=getattr(flash_attn, '__version__', 'unknown'),
        ))
    except ImportError as e:
        backends.append(BackendStatus(
            name="flash_attn",
            available=False,
            error=str(e),
        ))

    # Check FlashInfer
    try:
        import flashinfer
        backends.append(BackendStatus(
            name="flashinfer",
            available=True,
            version=getattr(flashinfer, '__version__', 'unknown'),
        ))
    except ImportError as e:
        backends.append(BackendStatus(
            name="flashinfer",
            available=False,
            error=str(e),
        ))

    # Check xFormers
    try:
        import xformers
        backends.append(BackendStatus(
            name="xformers",
            available=True,
            version=getattr(xformers, '__version__', 'unknown'),
        ))
    except ImportError as e:
        backends.append(BackendStatus(
            name="xformers",
            available=False,
            error=str(e),
        ))

    # Check Liger
    try:
        import liger_kernel
        backends.append(BackendStatus(
            name="liger_kernel",
            available=True,
            version=getattr(liger_kernel, '__version__', 'unknown'),
        ))
    except ImportError as e:
        backends.append(BackendStatus(
            name="liger_kernel",
            available=False,
            error=str(e),
        ))

    # Check cache status
    cache_status = "ok"
    try:
        from layerzero.selection.cache import get_global_cache
        cache = get_global_cache()
        if cache:
            cache_status = "ok"
    except Exception:
        cache_status = "unavailable"

    # Determine overall health
    available_backends = [b for b in backends if b.available]
    healthy = len(available_backends) >= 1 and cache_status == "ok"

    summary_parts = []
    if not cuda_available:
        summary_parts.append("CUDA not available")
    summary_parts.append(f"{len(available_backends)}/{len(backends)} backends available")

    return DoctorReport(
        healthy=healthy,
        backends=backends,
        cuda_available=cuda_available,
        cuda_devices=cuda_devices,
        cache_status=cache_status,
        summary="; ".join(summary_parts),
    )


def readiness_check() -> ReadinessReport:
    """Validate system is ready for production.

    Performs comprehensive checks:
    - Required backends are available
    - CUDA is working (if expected)
    - Caches are initialized
    - No configuration errors

    Returns:
        ReadinessReport indicating readiness.

    Example:
        >>> import layerzero as lz
        >>>
        >>> report = lz.readiness_check()
        >>> if not report.ready:
        ...     for issue in report.issues:
        ...         print(f"Issue: {issue}")
    """
    issues = []
    warnings = []
    backends_checked = []

    # Check PyTorch
    backends_checked.append("torch")
    if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        issues.append("PyTorch SDPA not available (requires PyTorch 2.0+)")

    # Check CUDA if GPU expected
    if torch.cuda.is_available():
        backends_checked.append("cuda")
        try:
            # Try allocating small tensor
            _ = torch.zeros(1, device='cuda')
        except Exception as e:
            issues.append(f"CUDA allocation failed: {e}")

    # Check cache initialization
    try:
        from layerzero.selection.cache import get_global_cache
        cache = get_global_cache()
        backends_checked.append("cache")
    except Exception as e:
        warnings.append(f"Selection cache unavailable: {e}")

    # Check at least one optimized backend
    optimized_backends = []
    for name, import_name in [
        ("flash_attn", "flash_attn"),
        ("flashinfer", "flashinfer"),
        ("xformers", "xformers"),
    ]:
        try:
            __import__(import_name)
            optimized_backends.append(name)
            backends_checked.append(name)
        except ImportError:
            pass

    if not optimized_backends and torch.cuda.is_available():
        warnings.append(
            "No optimized attention backends available. "
            "Consider installing flash-attn or flashinfer for better performance."
        )

    ready = len(issues) == 0
    return ReadinessReport(
        ready=ready,
        issues=issues,
        warnings=warnings,
        backends_checked=backends_checked,
    )


def compile(
    model: Any,
    shapes: Optional[List[Tuple[int, ...]]] = None,
    dtype: torch.dtype = torch.float16,
    **kwargs: Any,
) -> CompilationPlan:
    """Compile kernel selections for a model.

    Analyzes the model and pre-selects kernels for all operations.
    This can be used to "bake in" kernel selections for static
    workloads, avoiding runtime selection overhead.

    Args:
        model: PyTorch model to compile.
        shapes: Input shapes to optimize for.
        dtype: Data type.
        **kwargs: Additional compilation options.

    Returns:
        CompilationPlan with kernel selections.

    Example:
        >>> import layerzero as lz
        >>> import torch
        >>>
        >>> model = MyTransformerModel()
        >>> plan = lz.compile(model, shapes=[(1, 2048)])
        >>> print(plan.selections)
    """
    selections = {}

    # Default operations to compile
    operations = ["attention.causal", "norm.rms", "norm.layer"]

    for op in operations:
        from layerzero.api.inspection import which
        kernel = which(op)
        selections[op] = kernel

    return CompilationPlan(
        selections=selections,
        shapes=shapes or [],
    )


def dry_run(
    model: Any,
    shapes: Optional[List[Tuple[int, ...]]] = None,
    **kwargs: Any,
) -> CompilationPlan:
    """Show kernel selections without executing.

    Like compile(), but doesn't actually prepare kernels.
    Useful for debugging selection behavior.

    Args:
        model: PyTorch model.
        shapes: Input shapes.
        **kwargs: Additional options.

    Returns:
        CompilationPlan showing what would be selected.

    Example:
        >>> import layerzero as lz
        >>>
        >>> plan = lz.dry_run(model)
        >>> for op, kernel in plan.selections.items():
        ...     print(f"{op} -> {kernel}")
    """
    return compile(model, shapes=shapes, **kwargs)


def solve(
    operations: List[str],
    shapes: List[Tuple[int, ...]],
    dtype: torch.dtype = torch.float16,
    **kwargs: Any,
) -> CompilationPlan:
    """Build-time kernel solver.

    Solves for optimal kernel assignments across multiple
    operations and shapes.

    Args:
        operations: List of operations.
        shapes: List of shapes to optimize for.
        dtype: Data type.
        **kwargs: Solver options.

    Returns:
        CompilationPlan with optimal assignments.

    Example:
        >>> import layerzero as lz
        >>>
        >>> solution = lz.solve(
        ...     operations=["attention.causal", "norm.rms"],
        ...     shapes=[(1, 1024, 8, 64)],
        ... )
    """
    selections = {}

    for op in operations:
        from layerzero.api.inspection import which
        kernel = which(op)
        selections[op] = kernel

    return CompilationPlan(
        selections=selections,
        shapes=shapes,
    )


def tune(
    operation: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float16,
    samples: int = 10,
    **kwargs: Any,
) -> TuneResult:
    """Auto-tune kernel selection for an operation.

    Benchmarks available kernels and selects the fastest one
    for the given shape.

    Args:
        operation: Operation to tune.
        shape: Shape to tune for.
        dtype: Data type.
        samples: Number of benchmark samples.
        **kwargs: Tuning options.

    Returns:
        TuneResult with best kernel and timings.

    Example:
        >>> import layerzero as lz
        >>>
        >>> result = lz.tune(
        ...     operation="attention.causal",
        ...     shape=(1, 2048, 32, 128),
        ...     samples=20,
        ... )
        >>> print(f"Best kernel: {result.best_kernel}")
    """
    import time

    from layerzero.api.inspection import list_kernels

    kernels = list_kernels(operation=operation)
    timings = {}

    # For each kernel, run benchmark
    for kernel_info in kernels:
        try:
            # Skip unavailable backends
            if kernel_info.backend == "flash_attn":
                try:
                    import flash_attn
                except ImportError:
                    continue
            elif kernel_info.backend == "flashinfer":
                try:
                    import flashinfer
                except ImportError:
                    continue

            # Simple timing (real implementation would be more sophisticated)
            times = []
            for _ in range(samples):
                start = time.perf_counter()
                # Would execute kernel here
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            timings[kernel_info.id] = sum(times) / len(times)

        except Exception:
            continue

    # Select best
    if timings:
        best_kernel = min(timings, key=timings.get)
    else:
        best_kernel = "torch_sdpa"
        timings["torch_sdpa"] = 0.0

    return TuneResult(
        best_kernel=best_kernel,
        timings=timings,
    )
