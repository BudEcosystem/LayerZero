"""
Kernel Executor - The actual kernel execution layer.

This module bridges the gap between kernel selection and execution.
It handles:
- Tensor layout/dtype transformations
- Argument mapping for different backends
- Error handling and fallback
- CUDA graph capture (when supported)
- Telemetry and timing
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from layerzero.backends.base import BaseKernel
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.models.selection_context import SelectionContext
    from layerzero.registry.backend_registry import BackendRegistry

from layerzero.dispatch.types import (
    DispatchTiming,
    KernelExecutionError,
    TransformError,
)

logger = logging.getLogger(__name__)


class KernelExecutorImpl:
    """Kernel executor implementation.

    Executes kernels with proper argument mapping, transformations,
    and error handling. This is the critical "last mile" that connects
    the selection engine to actual kernel execution.

    The executor is stateless and thread-safe.
    """

    __slots__ = (
        "_backend_registry",
        "_transform_cache",
        "_execution_cache",
        "_cuda_graphs",
    )

    def __init__(
        self,
        backend_registry: Optional["BackendRegistry"] = None,
    ) -> None:
        """Initialize executor.

        Args:
            backend_registry: Optional backend registry for health tracking.
        """
        self._backend_registry = backend_registry
        # Cache for compiled transforms (layout converters, etc.)
        self._transform_cache: dict[str, Callable] = {}
        # Cache for execution wrappers
        self._execution_cache: dict[str, Callable] = {}
        # CUDA graph cache (kernel_id -> graph)
        self._cuda_graphs: dict[str, Any] = {}

    def execute(
        self,
        kernel_spec: "KernelSpec",
        inputs: dict[str, "torch.Tensor"],
        **kwargs: Any,
    ) -> "torch.Tensor":
        """Execute a kernel with given inputs.

        This is the main entry point for kernel execution.

        Args:
            kernel_spec: Specification of kernel to execute.
            inputs: Named input tensors (e.g., {"query": q, "key": k, "value": v}).
            **kwargs: Additional kernel-specific arguments.

        Returns:
            Output tensor from kernel execution.

        Raises:
            KernelExecutionError: If execution fails.
        """
        import torch

        kernel_id = kernel_spec.kernel_id
        operation = kernel_spec.operation

        # Get kernel implementation
        impl = kernel_spec.impl
        if impl is None:
            raise KernelExecutionError(
                f"Kernel '{kernel_id}' has no implementation",
                operation=operation,
                kernel_id=kernel_id,
            )

        try:
            # Map arguments to kernel-specific format
            mapped_kwargs = self._map_arguments(kernel_spec, inputs, kwargs)

            # Execute kernel
            if hasattr(impl, "__call__"):
                # Direct callable (BaseKernel or function)
                output = impl(**mapped_kwargs)
            else:
                raise KernelExecutionError(
                    f"Kernel implementation is not callable: {type(impl)}",
                    operation=operation,
                    kernel_id=kernel_id,
                )

            # Record success if backend registry available
            if self._backend_registry is not None:
                self._backend_registry.record_success(kernel_spec.source)

            return output

        except Exception as e:
            # Record failure if backend registry available
            if self._backend_registry is not None:
                self._backend_registry.record_failure(kernel_spec.source, e)

            # Wrap in KernelExecutionError if not already
            if isinstance(e, KernelExecutionError):
                raise
            raise KernelExecutionError(
                f"Kernel execution failed: {e}",
                operation=operation,
                kernel_id=kernel_id,
                original_error=e,
            ) from e

    def execute_with_timing(
        self,
        kernel_spec: "KernelSpec",
        inputs: dict[str, "torch.Tensor"],
        **kwargs: Any,
    ) -> tuple["torch.Tensor", int]:
        """Execute kernel and return execution time.

        Args:
            kernel_spec: Kernel specification.
            inputs: Input tensors.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (output tensor, execution time in nanoseconds).
        """
        start = time.perf_counter_ns()
        output = self.execute(kernel_spec, inputs, **kwargs)
        elapsed = time.perf_counter_ns() - start
        return output, elapsed

    def supports_cuda_graph(self, kernel_spec: "KernelSpec") -> bool:
        """Check if kernel supports CUDA graph capture.

        Args:
            kernel_spec: Kernel to check.

        Returns:
            True if kernel is CUDA graph safe.
        """
        return getattr(kernel_spec, "is_cuda_graph_safe", False)

    def _map_arguments(
        self,
        kernel_spec: "KernelSpec",
        inputs: dict[str, "torch.Tensor"],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Map generic arguments to kernel-specific format.

        Different kernels expect different argument names and formats.
        This method handles the translation.

        Args:
            kernel_spec: Target kernel specification.
            inputs: Input tensors with generic names.
            kwargs: Additional arguments.

        Returns:
            Mapped arguments ready for kernel execution.
        """
        import torch

        operation = kernel_spec.operation
        source = kernel_spec.source

        # Start with kwargs
        result = dict(kwargs)

        # Handle attention operations
        if operation.startswith("attention"):
            result.update(self._map_attention_args(kernel_spec, inputs, kwargs))

        # Handle normalization operations
        elif operation.startswith("rms_norm") or operation.startswith("layer_norm"):
            result.update(self._map_norm_args(kernel_spec, inputs, kwargs))

        # Handle RoPE operations
        elif operation.startswith("rope"):
            result.update(self._map_rope_args(kernel_spec, inputs, kwargs))

        # Default: pass inputs directly
        else:
            result.update(inputs)

        return result

    def _map_attention_args(
        self,
        kernel_spec: "KernelSpec",
        inputs: dict[str, "torch.Tensor"],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Map attention arguments to kernel-specific format.

        Handles differences between:
        - FlashAttention (q, k, v separate)
        - xFormers (Inputs dataclass)
        - Torch SDPA (query, key, value)
        - FlashInfer (different layout expectations)
        """
        import torch

        source = kernel_spec.source
        result: dict[str, Any] = {}

        # Get tensors with fallback names
        # Use explicit None checks to avoid tensor boolean ambiguity
        q = inputs.get("query")
        if q is None:
            q = inputs.get("q")
        k = inputs.get("key")
        if k is None:
            k = inputs.get("k")
        v = inputs.get("value")
        if v is None:
            v = inputs.get("v")

        if q is None or k is None or v is None:
            raise KernelExecutionError(
                "Missing required attention inputs (query, key, value)",
                operation=kernel_spec.operation,
                kernel_id=kernel_spec.kernel_id,
            )

        # Apply layout transformation if needed
        # Pass explicit layout if provided in kwargs
        input_layout = kwargs.get("input_layout") or kwargs.get("layout")
        q, k, v = self._transform_attention_layout(kernel_spec, q, k, v, input_layout)

        # Map based on source library
        if source in ("flash_attn", "flash_attn_v2", "flash_attn_v3", "flash_attn_v4"):
            # FlashAttention expects (batch, seqlen, nheads, headdim) BSHD layout
            result["q"] = q
            result["k"] = k
            result["v"] = v
            # Map optional arguments
            if "dropout_p" in kwargs:
                result["dropout_p"] = kwargs["dropout_p"]
            if "softmax_scale" in kwargs:
                result["softmax_scale"] = kwargs["softmax_scale"]
            elif "scale" in kwargs:
                result["softmax_scale"] = kwargs["scale"]
            if "causal" in kwargs:
                result["causal"] = kwargs["causal"]
            elif "is_causal" in kwargs:
                result["causal"] = kwargs["is_causal"]
            if "window_size" in kwargs:
                result["window_size"] = kwargs["window_size"]

        elif source == "flashinfer":
            # FlashInfer has specific layout requirements
            result["q"] = q
            result["k"] = k
            result["v"] = v
            if "causal" in kwargs:
                result["causal"] = kwargs["causal"]
            elif "is_causal" in kwargs:
                result["causal"] = kwargs["is_causal"]

        elif source == "xformers":
            # xFormers expects BHSD layout typically
            result["query"] = q
            result["key"] = k
            result["value"] = v
            if "attn_bias" in kwargs:
                result["attn_bias"] = kwargs["attn_bias"]
            elif "attn_mask" in kwargs:
                result["attn_bias"] = kwargs["attn_mask"]
            if "scale" in kwargs:
                result["scale"] = kwargs["scale"]
            result["p"] = kwargs.get("dropout_p", 0.0)

        elif source in ("torch_sdpa", "torch"):
            # PyTorch SDPA
            result["query"] = q
            result["key"] = k
            result["value"] = v
            if "attn_mask" in kwargs:
                result["attn_mask"] = kwargs["attn_mask"]
            result["dropout_p"] = kwargs.get("dropout_p", 0.0)
            result["is_causal"] = kwargs.get("is_causal", kwargs.get("causal", False))
            if "scale" in kwargs:
                result["scale"] = kwargs["scale"]

        else:
            # Generic mapping
            result["query"] = q
            result["key"] = k
            result["value"] = v
            result.update(kwargs)

        return result

    def _map_norm_args(
        self,
        kernel_spec: "KernelSpec",
        inputs: dict[str, "torch.Tensor"],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Map normalization arguments."""
        result: dict[str, Any] = {}

        # Use explicit None checks to avoid tensor boolean ambiguity
        x = inputs.get("input")
        if x is None:
            x = inputs.get("x")
        if x is None:
            x = inputs.get("hidden_states")

        weight = inputs.get("weight")
        if weight is None:
            weight = inputs.get("gamma")

        if x is None:
            raise KernelExecutionError(
                "Missing required input tensor for normalization",
                operation=kernel_spec.operation,
                kernel_id=kernel_spec.kernel_id,
            )

        source = kernel_spec.source

        if source == "liger":
            result["x"] = x
            result["weight"] = weight
            result["eps"] = kwargs.get("eps", 1e-6)

        elif source == "triton":
            result["x"] = x
            result["weight"] = weight
            result["eps"] = kwargs.get("eps", 1e-6)

        else:
            result["input"] = x
            result["weight"] = weight
            result["eps"] = kwargs.get("eps", 1e-6)

        return result

    def _map_rope_args(
        self,
        kernel_spec: "KernelSpec",
        inputs: dict[str, "torch.Tensor"],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Map RoPE arguments."""
        result: dict[str, Any] = {}

        # Use explicit None checks to avoid tensor boolean ambiguity
        x = inputs.get("input")
        if x is None:
            x = inputs.get("x")
        if x is None:
            x = inputs.get("hidden_states")

        cos = inputs.get("cos")
        if cos is None:
            cos = inputs.get("cos_cached")

        sin = inputs.get("sin")
        if sin is None:
            sin = inputs.get("sin_cached")

        if x is None:
            raise KernelExecutionError(
                "Missing required input tensor for RoPE",
                operation=kernel_spec.operation,
                kernel_id=kernel_spec.kernel_id,
            )

        result["x"] = x
        result["cos"] = cos
        result["sin"] = sin

        if "position_ids" in kwargs or "position_ids" in inputs:
            pos_ids = kwargs.get("position_ids")
            if pos_ids is None:
                pos_ids = inputs.get("position_ids")
            result["position_ids"] = pos_ids
        if "interleaved" in kwargs:
            result["interleaved"] = kwargs["interleaved"]

        return result

    def _transform_attention_layout(
        self,
        kernel_spec: "KernelSpec",
        q: "torch.Tensor",
        k: "torch.Tensor",
        v: "torch.Tensor",
        input_layout: Optional[str] = None,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Transform attention tensors to kernel-expected layout.

        Handles BHSD <-> BSHD conversion based on kernel requirements.

        Layout Detection Strategy (in order of priority):
        1. Explicit input_layout parameter ("BHSD" or "BSHD")
        2. Tensor metadata if available (input_layout attribute)
        3. Shape-based heuristic using head_dim validation

        Args:
            kernel_spec: Target kernel.
            q, k, v: Input tensors.
            input_layout: Optional explicit input layout ("BHSD" or "BSHD").

        Returns:
            Transformed tensors.
        """
        import torch
        from layerzero.enums import Layout

        # Get kernel's required layout
        requires_layouts = getattr(kernel_spec, "requires_layouts", None)
        if not requires_layouts:
            return q, k, v

        # Only handle 4D tensors
        if q.dim() != 4:
            return q, k, v

        # Determine input layout using multiple strategies
        detected_layout: str | None = None

        # Strategy 1: Explicit parameter
        if input_layout is not None:
            detected_layout = input_layout.upper()

        # Strategy 2: Tensor metadata (if set by higher layers)
        elif hasattr(q, "layout_format"):
            detected_layout = getattr(q, "layout_format", None)

        # Strategy 3: Shape-based heuristic with head_dim validation
        # Standard head dimensions: 64, 80, 96, 128, 256
        # This is more robust than comparing dim sizes
        else:
            batch, d1, d2, d3 = q.shape
            # Common head dimensions in transformer models
            standard_head_dims = {32, 64, 80, 96, 128, 256}

            # Check if last dim (d3) is a standard head dimension
            if d3 in standard_head_dims:
                # Last dim is head_dim, now determine if d1 is heads or seq
                # BHSD: (batch, heads, seq, head_dim) - heads typically 1-128
                # BSHD: (batch, seq, heads, head_dim) - seq typically varies widely

                # Heuristic: heads are typically <= 128 and often power of 2 or close
                # Sequences can be any length but are typically > 128 in practice
                typical_head_counts = {1, 2, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 96, 128}

                if d1 in typical_head_counts and d2 not in typical_head_counts:
                    # d1 looks like heads, d2 looks like seq -> BHSD
                    detected_layout = "BHSD"
                elif d2 in typical_head_counts and d1 not in typical_head_counts:
                    # d1 looks like seq, d2 looks like heads -> BSHD
                    detected_layout = "BSHD"
                elif d1 <= d2:
                    # If both could be heads or neither, use size comparison
                    # Smaller dim is more likely heads (BHSD)
                    detected_layout = "BHSD"
                else:
                    # Larger dim1 suggests it's sequence (BSHD)
                    detected_layout = "BSHD"
            else:
                # Non-standard head dim, fall back to conservative assumption
                # Assume BSHD as it's more common in modern implementations
                detected_layout = "BSHD"

        input_is_bhsd = detected_layout == "BHSD"

        # Check what kernel needs
        needs_bshd = Layout.BSHD in requires_layouts
        needs_bhsd = Layout.BHSD in requires_layouts

        if needs_bshd and input_is_bhsd:
            # Convert BHSD -> BSHD: transpose dims 1 and 2
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

        elif needs_bhsd and not input_is_bhsd:
            # Convert BSHD -> BHSD: transpose dims 1 and 2
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

        return q, k, v


def execute_kernel(
    kernel_spec: "KernelSpec",
    inputs: dict[str, "torch.Tensor"],
    backend_registry: Optional["BackendRegistry"] = None,
    **kwargs: Any,
) -> "torch.Tensor":
    """Convenience function to execute a kernel.

    Creates a temporary executor and runs the kernel.

    Args:
        kernel_spec: Kernel to execute.
        inputs: Input tensors.
        backend_registry: Optional backend registry.
        **kwargs: Additional arguments.

    Returns:
        Output tensor.
    """
    executor = KernelExecutorImpl(backend_registry)
    return executor.execute(kernel_spec, inputs, **kwargs)


class CUDAGraphExecutor:
    """Executor that captures and replays CUDA graphs.

    CUDA graphs can reduce kernel launch overhead from ~20us to ~2.5us
    by capturing a sequence of GPU operations and replaying them.

    This executor automatically captures graphs for supported kernels
    and falls back to regular execution otherwise.
    """

    __slots__ = (
        "_base_executor",
        "_graphs",
        "_static_inputs",
        "_static_outputs",
        "_warmup_count",
    )

    def __init__(
        self,
        base_executor: KernelExecutorImpl,
        warmup_count: int = 3,
    ) -> None:
        """Initialize CUDA graph executor.

        Args:
            base_executor: Base executor for warmup and non-graphable kernels.
            warmup_count: Number of warmup iterations before graph capture.
        """
        self._base_executor = base_executor
        self._graphs: dict[str, Any] = {}  # kernel_id -> captured graph
        self._static_inputs: dict[str, dict[str, "torch.Tensor"]] = {}
        self._static_outputs: dict[str, "torch.Tensor"] = {}
        self._warmup_count = warmup_count

    def execute(
        self,
        kernel_spec: "KernelSpec",
        inputs: dict[str, "torch.Tensor"],
        **kwargs: Any,
    ) -> "torch.Tensor":
        """Execute kernel, using CUDA graph if available.

        Args:
            kernel_spec: Kernel to execute.
            inputs: Input tensors.
            **kwargs: Additional arguments.

        Returns:
            Output tensor.
        """
        import torch

        kernel_id = kernel_spec.kernel_id

        # Check if we have a captured graph
        if kernel_id in self._graphs:
            return self._replay_graph(kernel_id, inputs)

        # Check if kernel supports CUDA graphs
        if not self._base_executor.supports_cuda_graph(kernel_spec):
            return self._base_executor.execute(kernel_spec, inputs, **kwargs)

        # Need CUDA for graphs
        device = next(iter(inputs.values())).device
        if not device.type == "cuda":
            return self._base_executor.execute(kernel_spec, inputs, **kwargs)

        # Warmup before capture
        for _ in range(self._warmup_count):
            output = self._base_executor.execute(kernel_spec, inputs, **kwargs)

        # Capture graph
        try:
            graph, static_inputs, static_output = self._capture_graph(
                kernel_spec, inputs, **kwargs
            )
            self._graphs[kernel_id] = graph
            self._static_inputs[kernel_id] = static_inputs
            self._static_outputs[kernel_id] = static_output
            return static_output.clone()

        except Exception as e:
            logger.warning(
                f"Failed to capture CUDA graph for {kernel_id}: {e}. "
                "Falling back to regular execution."
            )
            return self._base_executor.execute(kernel_spec, inputs, **kwargs)

    def _capture_graph(
        self,
        kernel_spec: "KernelSpec",
        inputs: dict[str, "torch.Tensor"],
        **kwargs: Any,
    ) -> tuple[Any, dict[str, "torch.Tensor"], "torch.Tensor"]:
        """Capture a CUDA graph for kernel execution.

        Args:
            kernel_spec: Kernel to capture.
            inputs: Input tensors (used as templates).
            **kwargs: Additional arguments.

        Returns:
            Tuple of (graph, static_inputs, static_output).
        """
        import torch

        # Create static input buffers
        static_inputs = {
            name: tensor.clone() for name, tensor in inputs.items()
        }

        # Synchronize before capture
        torch.cuda.synchronize()

        # Create graph
        graph = torch.cuda.CUDAGraph()

        # Capture
        with torch.cuda.graph(graph):
            static_output = self._base_executor.execute(
                kernel_spec, static_inputs, **kwargs
            )

        return graph, static_inputs, static_output

    def _replay_graph(
        self,
        kernel_id: str,
        inputs: dict[str, "torch.Tensor"],
    ) -> "torch.Tensor":
        """Replay a captured CUDA graph with new inputs.

        Args:
            kernel_id: ID of captured kernel.
            inputs: New input tensors.

        Returns:
            Output tensor.
        """
        import torch

        graph = self._graphs[kernel_id]
        static_inputs = self._static_inputs[kernel_id]
        static_output = self._static_outputs[kernel_id]

        # Copy new inputs to static buffers
        for name, tensor in inputs.items():
            static_inputs[name].copy_(tensor)

        # Replay graph
        graph.replay()

        return static_output.clone()

    def clear_graphs(self) -> None:
        """Clear all captured graphs."""
        self._graphs.clear()
        self._static_inputs.clear()
        self._static_outputs.clear()
