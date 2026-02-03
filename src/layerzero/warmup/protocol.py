"""
JIT Warmup Protocol implementation.

This module provides:
- JITWarmupProtocol: Main warmup execution logic
- WarmupStatus: Current warmup state tracking
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Callable, TYPE_CHECKING

from layerzero.warmup.config import ShapeWarmupResult, WarmupConfig, WarmupReport
from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

if TYPE_CHECKING:
    from layerzero.registry.backend_registry import BackendRegistry

logger = logging.getLogger(__name__)


@dataclass
class WarmupStatus:
    """Current warmup state.

    Attributes:
        critical_complete: Whether all critical shapes are warmed up.
        all_complete: Whether all shapes are warmed up.
        critical_total: Total critical shapes.
        critical_pending: Critical shapes not yet warmed up.
        total_shapes: Total shapes in manifest.
        warmed_shapes: Number of shapes warmed up.
        in_progress: Number of shapes currently being compiled.
    """

    critical_complete: bool = False
    all_complete: bool = False
    critical_total: int = 0
    critical_pending: int = 0
    total_shapes: int = 0
    warmed_shapes: int = 0
    in_progress: int = 0


@dataclass
class _CompileTask:
    """Internal compile task."""

    signature: ShapeSignature
    backend: str
    critical: bool
    priority: float


class JITWarmupProtocol:
    """JIT warmup protocol implementation.

    Handles JIT compilation warmup with:
    - Blocking until critical shapes compiled (optional)
    - Concurrent compilation with limits
    - Timeout handling with fallback
    - Background compilation for non-critical shapes
    - Cache persistence

    Example:
        config = WarmupConfig(timeout_ms=30000.0, max_concurrent_jit=2)
        protocol = JITWarmupProtocol(config)

        manifest = ShapeManifest.from_model_config(model_config)
        report = protocol.warmup(manifest)

        print(f"Warmed up {report.compiled_shapes} shapes")
    """

    # Note: No __slots__ to allow mocking in tests

    def __init__(
        self,
        config: WarmupConfig,
        backend_registry: "BackendRegistry | None" = None,
    ) -> None:
        """Initialize warmup protocol.

        Args:
            config: Warmup configuration.
            backend_registry: Registry of available backends.
        """
        self._config = config
        self._manifest: ShapeManifest | None = None
        self._warmed_shapes: set[str] = set()
        self._shape_lock = threading.Lock()
        self._compile_semaphore = threading.Semaphore(config.max_concurrent_jit)
        self._background_queue: Queue[_CompileTask] = Queue()
        self._background_thread: threading.Thread | None = None
        self._background_running = False
        self._backend_registry = backend_registry
        self.on_progress: Callable[[int, int], None] | None = None

    @property
    def config(self) -> WarmupConfig:
        """Get warmup configuration."""
        return self._config

    def warmup(
        self,
        manifest: ShapeManifest,
        backends: list[str] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> WarmupReport:
        """Execute warmup for shapes in manifest.

        Args:
            manifest: Shape manifest to warmup.
            backends: Backends to warmup for. If None, warmup all available.
            progress_callback: Called with (total, done) for progress updates.

        Returns:
            WarmupReport with results.
        """
        if not self._config.enabled:
            return WarmupReport(
                total_shapes=0,
                compiled_shapes=0,
                cached_shapes=0,
                failed_shapes=0,
                total_time_ms=0.0,
            )

        self._manifest = manifest
        start_time = time.perf_counter()

        # Get ordered shapes
        shapes = manifest.get_ordered_shapes(
            critical_first=self._config.critical_shapes_first
        )
        total_shapes = len(shapes)

        if total_shapes == 0:
            return WarmupReport(
                total_shapes=0,
                compiled_shapes=0,
                cached_shapes=0,
                failed_shapes=0,
                total_time_ms=0.0,
            )

        # Determine backends to use
        if backends is None:
            backends = self._get_available_backends()

        # Split into critical and non-critical
        critical_shapes = manifest.get_shapes(critical_only=True)
        critical_keys = {s.to_key() for s in critical_shapes}

        compiled_count = 0
        cached_count = 0
        failed_count = 0
        errors: list[str] = []
        results: list[ShapeWarmupResult] = []

        # Process shapes
        if self._config.blocking:
            # Blocking mode: compile all critical shapes synchronously
            compiled_count, cached_count, failed_count, errors, results = (
                self._warmup_blocking(shapes, backends, critical_keys, progress_callback)
            )
        else:
            # Non-blocking mode: start background compilation
            self._start_background_compiler()
            for shape in shapes:
                for backend in backends:
                    task = _CompileTask(
                        signature=shape,
                        backend=backend,
                        critical=shape.to_key() in critical_keys,
                        priority=1.0 if shape.to_key() in critical_keys else 0.5,
                    )
                    self._background_queue.put(task)

            # Wait briefly for critical shapes
            critical_timeout_ms = self._config.timeout_ms * len(critical_shapes)
            critical_timeout = critical_timeout_ms / 1000.0
            deadline = time.perf_counter() + critical_timeout

            while time.perf_counter() < deadline:
                with self._shape_lock:
                    warmed_critical = sum(
                        1 for k in critical_keys if k in self._warmed_shapes
                    )
                if warmed_critical >= len(critical_keys):
                    break
                time.sleep(0.1)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return WarmupReport(
            total_shapes=total_shapes,
            compiled_shapes=compiled_count,
            cached_shapes=cached_count,
            failed_shapes=failed_count,
            total_time_ms=elapsed_ms,
            errors=errors,
            shape_results=results,
        )

    def _warmup_blocking(
        self,
        shapes: list[ShapeSignature],
        backends: list[str],
        critical_keys: set[str],
        progress_callback: Callable[[int, int], None] | None,
    ) -> tuple[int, int, int, list[str], list[ShapeWarmupResult]]:
        """Execute blocking warmup.

        Args:
            shapes: Shapes to warmup (ordered).
            backends: Backends to use.
            critical_keys: Keys of critical shapes.
            progress_callback: Progress callback.

        Returns:
            Tuple of (compiled, cached, failed, errors, results).
        """
        compiled_count = 0
        cached_count = 0
        failed_count = 0
        errors: list[str] = []
        results: list[ShapeWarmupResult] = []

        total = len(shapes) * len(backends)
        done = 0

        # Use thread pool for concurrent compilation
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_concurrent_jit
        ) as executor:
            futures: dict[concurrent.futures.Future, tuple[ShapeSignature, str]] = {}

            for shape in shapes:
                for backend in backends:
                    future = executor.submit(
                        self._compile_shape_with_timeout,
                        shape,
                        backend,
                        self._config.timeout_ms,
                    )
                    futures[future] = (shape, backend)

            for future in concurrent.futures.as_completed(futures):
                shape, backend = futures[future]
                done += 1

                try:
                    success, compile_time, error = future.result()
                except Exception as e:
                    success = False
                    compile_time = 0.0
                    error = str(e)

                # Determine if cached (very fast compile time indicates cache hit)
                cached = success and compile_time < 1.0

                result = ShapeWarmupResult(
                    shape_key=shape.to_key(),
                    success=success,
                    cached=cached,
                    compile_time_ms=compile_time,
                    error=error,
                    backend=backend,
                )
                results.append(result)

                if success:
                    if cached:
                        cached_count += 1
                    else:
                        compiled_count += 1

                    with self._shape_lock:
                        self._warmed_shapes.add(shape.to_key())
                else:
                    failed_count += 1
                    if error:
                        errors.append(f"{shape.to_key()}: {error}")

                # Report progress
                if progress_callback:
                    progress_callback(total, done)

        return compiled_count, cached_count, failed_count, errors, results

    def _compile_shape_with_timeout(
        self,
        shape: ShapeSignature,
        backend: str,
        timeout_ms: float,
    ) -> tuple[bool, float, str | None]:
        """Compile shape with timeout.

        Args:
            shape: Shape to compile.
            backend: Backend to use.
            timeout_ms: Timeout in milliseconds.

        Returns:
            Tuple of (success, compile_time_ms, error).
        """
        # Acquire semaphore to limit concurrency
        if not self._compile_semaphore.acquire(timeout=timeout_ms / 1000.0):
            return (False, timeout_ms, "Timeout waiting for compilation slot")

        try:
            start = time.perf_counter()
            success, compile_time, error = self._compile_shape(shape, backend)
            elapsed = (time.perf_counter() - start) * 1000

            # Check if we exceeded timeout
            if elapsed > timeout_ms:
                return (False, elapsed, f"JIT compilation timed out after {elapsed:.1f}ms")

            return (success, elapsed, error)
        finally:
            self._compile_semaphore.release()

    def _compile_shape(
        self,
        shape: ShapeSignature,
        backend: str,
    ) -> tuple[bool, float, str | None]:
        """Actually compile a shape.

        This method should be overridden or the backend adapters should
        provide warmup_jit methods.

        Args:
            shape: Shape to compile.
            backend: Backend to use.

        Returns:
            Tuple of (success, compile_time_ms, error).
        """
        # Check if already warmed
        with self._shape_lock:
            if shape.to_key() in self._warmed_shapes:
                return (True, 0.0, None)

        # Try to get backend adapter and call warmup_jit
        try:
            adapter = self._get_backend_adapter(backend)
            if adapter is None:
                return (False, 0.0, f"Backend {backend} not available")

            if hasattr(adapter, "warmup_jit"):
                start = time.perf_counter()
                success = adapter.warmup_jit(shape, timeout_ms=self._config.timeout_ms)
                elapsed_ms = (time.perf_counter() - start) * 1000
                return (success, elapsed_ms, None if success else "warmup_jit returned False")
            else:
                # Backend doesn't support explicit warmup
                # Try to trigger JIT by calling with dummy inputs
                return self._trigger_jit_compile(shape, backend)

        except Exception as e:
            logger.warning(f"Warmup failed for {shape.to_key()} on {backend}: {e}")
            return (False, 0.0, str(e))

    def _trigger_jit_compile(
        self,
        shape: ShapeSignature,
        backend: str,
    ) -> tuple[bool, float, str | None]:
        """Trigger JIT compilation by running with dummy inputs.

        Args:
            shape: Shape to compile.
            backend: Backend to use.

        Returns:
            Tuple of (success, compile_time_ms, error).
        """
        import torch

        try:
            # Create dummy inputs based on shape
            B = shape.batch_size_bucket
            S = shape.seq_len_bucket
            H = shape.num_heads
            D = shape.head_dim
            dtype = shape.dtype
            device = "cuda" if torch.cuda.is_available() else "cpu"

            q = torch.randn(B, H, S, D, dtype=dtype, device=device)
            k = torch.randn(B, H, S, D, dtype=dtype, device=device)
            v = torch.randn(B, H, S, D, dtype=dtype, device=device)

            # Time the compilation (first call triggers JIT)
            start = time.perf_counter()

            # Call the attention function (triggers JIT)
            if "attention" in shape.operation:
                import torch.nn.functional as F
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            if device == "cuda":
                torch.cuda.synchronize()

            elapsed_ms = (time.perf_counter() - start) * 1000
            return (True, elapsed_ms, None)

        except Exception as e:
            return (False, 0.0, str(e))

    def _get_backend_adapter(self, backend: str) -> Any | None:
        """Get backend adapter by name.

        Args:
            backend: Backend name.

        Returns:
            Backend adapter or None if not available.
        """
        if self._backend_registry is not None:
            return self._backend_registry.get_adapter(backend)
        return None

    def _get_available_backends(self) -> list[str]:
        """Get list of available backends.

        Returns:
            List of backend names.
        """
        if self._backend_registry is not None:
            return self._backend_registry.list_available()

        # Default backends if no registry
        backends = []

        # Check common backends
        try:
            import flash_attn
            backends.append("flash_attn")
        except ImportError:
            pass

        try:
            import flashinfer
            backends.append("flashinfer")
        except ImportError:
            pass

        try:
            import triton
            backends.append("triton")
        except ImportError:
            pass

        # Always include torch SDPA as fallback
        backends.append("torch_sdpa")

        return backends

    def _start_background_compiler(self) -> None:
        """Start background compilation thread."""
        if self._background_thread is not None and self._background_thread.is_alive():
            return

        self._background_running = True
        self._background_thread = threading.Thread(
            target=self._background_compile_loop,
            daemon=True,
            name="LayerZero-JIT-Background",
        )
        self._background_thread.start()

    def _stop_background_compiler(self) -> None:
        """Stop background compilation thread."""
        self._background_running = False
        if self._background_thread is not None:
            # Signal thread to stop
            self._background_queue.put(None)  # type: ignore
            self._background_thread.join(timeout=5.0)
            self._background_thread = None

    def _background_compile_loop(self) -> None:
        """Background compilation loop."""
        while self._background_running:
            try:
                task = self._background_queue.get(timeout=1.0)
            except Empty:
                continue

            if task is None:  # Shutdown signal
                break

            # Compile the shape
            try:
                self._compile_shape(task.signature, task.backend)
            except Exception as e:
                logger.warning(
                    f"Background compile failed for {task.signature.to_key()}: {e}"
                )

    def warmup_shape(
        self,
        shape: ShapeSignature,
        backend: str,
        timeout_ms: float | None = None,
    ) -> ShapeWarmupResult:
        """Warmup a single shape.

        Args:
            shape: Shape to warmup.
            backend: Backend to use.
            timeout_ms: Timeout in milliseconds. Uses config default if None.

        Returns:
            ShapeWarmupResult with result.
        """
        timeout = timeout_ms or self._config.timeout_ms
        success, compile_time, error = self._compile_shape_with_timeout(
            shape, backend, timeout
        )

        cached = success and compile_time < 1.0

        return ShapeWarmupResult(
            shape_key=shape.to_key(),
            success=success,
            cached=cached,
            compile_time_ms=compile_time,
            error=error,
            backend=backend,
        )

    def is_warmed_up(self, shape: ShapeSignature, backend: str) -> bool:
        """Check if shape is warmed up.

        Args:
            shape: Shape to check.
            backend: Backend to check for.

        Returns:
            True if shape is warmed up for backend.
        """
        with self._shape_lock:
            return shape.to_key() in self._warmed_shapes

    def get_warmup_status(self) -> WarmupStatus:
        """Get current warmup status.

        Returns:
            WarmupStatus with current state.
        """
        if self._manifest is None:
            return WarmupStatus()

        critical_shapes = self._manifest.get_shapes(critical_only=True)
        all_shapes = self._manifest.get_shapes(critical_only=False)

        with self._shape_lock:
            warmed_critical = sum(
                1 for s in critical_shapes if s.to_key() in self._warmed_shapes
            )
            warmed_all = len(self._warmed_shapes)

        return WarmupStatus(
            critical_complete=warmed_critical >= len(critical_shapes),
            all_complete=warmed_all >= len(all_shapes),
            critical_total=len(critical_shapes),
            critical_pending=len(critical_shapes) - warmed_critical,
            total_shapes=len(all_shapes),
            warmed_shapes=warmed_all,
        )

    def queue_background_compile(
        self,
        shape: ShapeSignature,
        backend: str,
    ) -> None:
        """Queue shape for background compilation.

        Args:
            shape: Shape to compile.
            backend: Backend to use.
        """
        self._start_background_compiler()

        task = _CompileTask(
            signature=shape,
            backend=backend,
            critical=False,
            priority=0.5,
        )
        self._background_queue.put(task)

    def __del__(self) -> None:
        """Cleanup background thread on deletion."""
        self._stop_background_compiler()
