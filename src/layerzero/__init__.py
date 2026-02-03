"""
LayerZero - Kernel Orchestration Layer for LLM Inference

A production-grade kernel selection and dispatching system that
provides unified abstractions across multiple GPU backends.

Main APIs:
- lz.warmup(): Precompile JIT kernels for specified shapes
- lz.select(): Select best kernel for given context
- lz.explain(): Get detailed selection report
- lz.is_graph_safe(): Check if kernel is safe for CUDA graph capture
- lz.validate_graph_capture(): Validate function can be captured in CUDA graph
"""

__version__ = "0.1.0"

from layerzero.device import (
    GPUGeneration,
    get_tensor_core_gen,
    sm_to_generation,
)
from layerzero.enums import (
    KVCacheStrategy,
    Layout,
    MaskType,
    OpKind,
    Platform,
    QuantFormat,
)
from layerzero.reasons import (
    ALL_REASON_CODES,
    Reason,
    ReasonCategory,
)

# Import warmup components lazily to avoid circular imports
def warmup(
    operation: str | None = None,
    shapes: list[tuple[int, ...]] | None = None,
    dtype=None,  # torch.dtype
    backends: list[str] | None = None,
    blocking: bool = True,
    timeout_ms: float = 30000.0,
):
    """Precompile JIT kernels for specified shapes.

    This prevents JIT compilation latency spikes in production by
    pre-warming kernel caches for common shapes.

    Args:
        operation: Operation to warmup (e.g., "attention.causal").
                   If None, warmup all operations.
        shapes: List of shapes (batch, seq_len, heads, head_dim).
                If None, use default production shapes.
        dtype: Data type to warmup for (torch.float16, torch.bfloat16, etc.).
        backends: Backends to warmup. If None, warmup all available.
        blocking: If True, block until warmup complete.
        timeout_ms: Timeout per shape in milliseconds.

    Returns:
        WarmupReport with timing and any errors.

    Example:
        import torch
        import layerzero as lz

        # Warmup attention for common shapes
        report = lz.warmup(
            operation="attention.causal",
            shapes=[(1, 1024, 16, 128), (1, 2048, 16, 128)],
            dtype=torch.float16,
        )

        if report.success_rate < 1.0:
            print(f"Warmup had {report.failed_shapes} failures")
    """
    import torch as _torch
    from layerzero.warmup import (
        JITWarmupProtocol,
        ShapeManifest,
        ShapeSignature,
        WarmupConfig,
    )

    # Use default dtype if not specified
    if dtype is None:
        dtype = _torch.float16

    # Create config
    config = WarmupConfig(
        enabled=True,
        blocking=blocking,
        timeout_ms=timeout_ms,
    )

    # Create protocol
    protocol = JITWarmupProtocol(config)

    # Build manifest
    manifest = ShapeManifest()

    if shapes is not None:
        # User provided explicit shapes
        for shape in shapes:
            if len(shape) == 4:
                batch, seq_len, heads, head_dim = shape
            else:
                raise ValueError(f"Shape must be (batch, seq_len, heads, head_dim), got {shape}")

            sig = ShapeSignature(
                operation=operation or "attention.causal",
                dtype=dtype,
                batch_size_bucket=batch,
                seq_len_bucket=seq_len,
                head_dim=head_dim,
                num_heads=heads,
            )
            manifest.add_shape(sig, critical=True)
    else:
        # Use default shapes for common LLM configurations
        default_shapes = [
            (1, 512, 32, 128),
            (1, 1024, 32, 128),
            (1, 2048, 32, 128),
            (4, 512, 32, 128),
            (4, 1024, 32, 128),
            (8, 512, 32, 128),
        ]
        for batch, seq_len, heads, head_dim in default_shapes:
            sig = ShapeSignature(
                operation=operation or "attention.causal",
                dtype=dtype,
                batch_size_bucket=batch,
                seq_len_bucket=seq_len,
                head_dim=head_dim,
                num_heads=heads,
            )
            manifest.add_shape(sig, critical=batch <= 4 and seq_len <= 1024)

    return protocol.warmup(manifest, backends=backends)


def is_graph_safe(
    kernel_id: str,
    strict: bool | None = None,
) -> bool:
    """Check if kernel is safe for CUDA graph capture.

    CUDA graphs have restrictions on what operations can be captured.
    This function checks a kernel ID against the whitelist of known
    safe and unsafe operations.

    Args:
        kernel_id: Kernel identifier or operation name.
        strict: If True, reject unknown kernels. If None, uses config default.

    Returns:
        True if kernel is safe for graph capture.

    Example:
        import layerzero as lz

        if lz.is_graph_safe("attention.causal"):
            # Safe to capture
            with torch.cuda.graph(g):
                output = attention(q, k, v)
    """
    from layerzero.graphs.validator import is_graph_safe as _is_graph_safe

    return _is_graph_safe(kernel_id, strict=strict)


def validate_graph_capture(
    func,
    *args,
    kernel_id: str | None = None,
    operation: str | None = None,
    strict: bool | None = None,
    **kwargs,
):
    """Validate function can be captured in CUDA graph.

    Performs full validation including:
    1. Warmup (initializes cuBLAS/cuDNN)
    2. Memory tracking (detects unexpected allocations)
    3. Optional dummy capture (in strict mode)

    Args:
        func: Function to validate.
        *args: Arguments to pass to function.
        kernel_id: Optional kernel ID for reporting.
        operation: Optional operation name for reporting.
        strict: If True, perform strict validation with dummy capture.
        **kwargs: Keyword arguments to pass to function.

    Returns:
        GraphValidationResult with validation status.

    Example:
        import layerzero as lz
        import torch

        def my_attention(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        result = lz.validate_graph_capture(
            my_attention, q, k, v,
            kernel_id="my_attention",
            operation="attention",
        )

        if not result.success:
            print(f"Validation failed: {result.error}")
        elif result.has_warnings:
            for warning in result.warnings:
                print(f"Warning: {warning}")
    """
    from layerzero.graphs.config import GraphSafetyConfig
    from layerzero.graphs.validator import GraphValidator

    # Create config
    config = GraphSafetyConfig(
        strict_mode=strict if strict is not None else False,
    )

    # Create validator
    validator = GraphValidator(config=config)

    return validator.validate_capture(
        func,
        *args,
        kernel_id=kernel_id,
        operation=operation,
        **kwargs,
    )


# Import exception classes
from layerzero.exceptions import (
    LayerZeroError,
    NoKernelFoundError,
    CudaGraphUnsafeError,
    KernelExecutionError,
    PolicyValidationError,
    BackendNotAvailableError,
    ConfigurationError,
    SelectionTimeoutError,
    CacheCorruptionError,
)

# Import public API functions
from layerzero.api.operations import (
    attention,
    paged_attention,
    rms_norm,
    layer_norm,
    rope,
    sample_topk,
    sample_topp,
    quantize,
    tokenize,
    detokenize,
)
from layerzero.api.config import (
    configure,
    get_config,
    load_config,
    lock,
    unlock,
    get_locks,
    prefer,
    disabled,
)
from layerzero.api.inspection import (
    select,
    explain,
    which,
    list_kernels,
    validate,
)
from layerzero.api.system import (
    doctor,
    readiness_check,
    compile,
    dry_run,
    solve,
    tune,
)


__all__ = [
    # Version
    "__version__",
    # Device/GPU
    "GPUGeneration",
    "sm_to_generation",
    "get_tensor_core_gen",
    # Enums
    "OpKind",
    "Layout",
    "MaskType",
    "Platform",
    "QuantFormat",
    "KVCacheStrategy",
    # Reasons
    "Reason",
    "ReasonCategory",
    "ALL_REASON_CODES",
    # Warmup
    "warmup",
    # Graph validation
    "is_graph_safe",
    "validate_graph_capture",
    # Exceptions
    "LayerZeroError",
    "NoKernelFoundError",
    "CudaGraphUnsafeError",
    "KernelExecutionError",
    "PolicyValidationError",
    "BackendNotAvailableError",
    "ConfigurationError",
    "SelectionTimeoutError",
    "CacheCorruptionError",
    # Public API - Operations
    "attention",
    "paged_attention",
    "rms_norm",
    "layer_norm",
    "rope",
    "sample_topk",
    "sample_topp",
    "quantize",
    "tokenize",
    "detokenize",
    # Public API - Configuration
    "configure",
    "get_config",
    "load_config",
    "lock",
    "unlock",
    "get_locks",
    "prefer",
    "disabled",
    # Public API - Inspection
    "select",
    "explain",
    "which",
    "list_kernels",
    "validate",
    # Public API - System
    "doctor",
    "readiness_check",
    "compile",
    "dry_run",
    "solve",
    "tune",
]
