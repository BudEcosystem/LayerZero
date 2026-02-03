# Code Examples

This document provides comprehensive code examples for common use cases with the LayerZero dispatch system.

## Basic Dispatch

### Simple Dispatch with Global Orchestrator

```python
import torch
from layerzero.dispatch import dispatch

# Create input tensors
batch_size, seq_len, num_heads, head_dim = 8, 2048, 32, 128
device = "cuda"
dtype = torch.float16

query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

# Dispatch attention operation
result = dispatch(
    operation="attention.causal",
    inputs={"query": query, "key": key, "value": value},
    is_causal=True,
)

# Access output
output = result.output
print(f"Output shape: {output.shape}")

# Check timing
print(f"Selection time: {result.timing.selection_us:.2f}us")
print(f"Execution time: {result.timing.execution_us:.2f}us")
print(f"Kernel used: {result.kernel_id}")
```

### Dispatch with Explicit Context

```python
from layerzero.dispatch import dispatch
from layerzero.selection import SelectionContext

# Build context with full control
context = SelectionContext(
    batch_size=8,
    seq_len=2048,
    seq_len_q=2048,
    seq_len_k=2048,
    head_dim=128,
    num_heads=32,
    num_kv_heads=32,
    dtype="float16",
    device="cuda:0",
    platform="cuda",
    is_causal=True,
    enable_gqa=False,
    dropout_p=0.0,
)

result = dispatch(
    operation="attention.causal",
    inputs={"query": query, "key": key, "value": value},
    context=context,
)
```

### Dispatch with Mode Override

```python
from layerzero.dispatch import dispatch, DispatchMode

# Force static dispatch for this call
result = dispatch(
    operation="attention.causal",
    inputs=inputs,
    mode=DispatchMode.STATIC,
)

# Force dynamic dispatch
result = dispatch(
    operation="attention.causal",
    inputs=inputs,
    mode=DispatchMode.DYNAMIC,
)
```

## Static Dispatch Setup

### Using StaticDispatcherBuilder

```python
from layerzero.dispatch import (
    StaticDispatcherBuilder,
    DispatchConfig,
    DispatchMode,
)
from layerzero.registry import get_global_kernel_registry

# Get kernel specifications from registry
registry = get_global_kernel_registry()
flash_attn = registry.get("flash_attn_v2_attention")
sdpa = registry.get("torch_sdpa_attention")
liger_rms = registry.get("liger_rms_norm")
triton_rope = registry.get("triton_rope")

# Build static dispatcher
dispatcher = (
    StaticDispatcherBuilder()
    # Register attention kernels
    .with_kernel(flash_attn, operation="attention.causal", default=True)
    .with_kernel(sdpa, operation="attention.causal")
    .with_kernel(flash_attn, operation="attention.cross", default=True)
    # Register normalization kernels
    .with_kernel(liger_rms, operation="rms_norm", default=True)
    # Register position encoding kernels
    .with_kernel(triton_rope, operation="rope", default=True)
    # Configuration
    .with_config(
        enable_telemetry=True,
        record_timing=True,
    )
    .build()
)

# Use the dispatcher
result = dispatcher.dispatch("attention.causal", inputs)
```

### Using Static Kernel Registry

```python
from layerzero.dispatch import (
    StaticDispatcher,
    StaticKernelRegistry,
    DispatchConfig,
    DispatchMode,
)

# Create and populate registry
registry = StaticKernelRegistry()

# Register kernels from specs
registry.register_from_spec(flash_attn_spec, is_default=True)
registry.register_from_spec(sdpa_spec)
registry.register_from_spec(xformers_spec)

# Freeze registry (no more modifications allowed)
registry.freeze()

# Create dispatcher
config = DispatchConfig(
    mode=DispatchMode.STATIC,
    enable_telemetry=True,
)

dispatcher = StaticDispatcher(config, registry=registry)

# Dispatch
result = dispatcher.dispatch("attention.causal", inputs)
```

### Using Config-Based Static Dispatch

```python
from layerzero.dispatch import (
    create_orchestrator,
    DispatchConfig,
    DispatchMode,
)
from layerzero.registry import get_global_kernel_registry, get_global_backend_registry

# Define static kernel mapping
config = DispatchConfig(
    mode=DispatchMode.STATIC,
    static_kernel_map={
        "attention.causal": "flash_attn_v2_attention",
        "attention.cross": "flash_attn_v2_attention",
        "attention.decode": "flashinfer_decode_attention",
        "rms_norm": "liger_rms_norm",
        "layer_norm": "torch_layer_norm",
        "rope": "liger_rope",
        "swiglu": "liger_swiglu",
    },
    enable_telemetry=False,  # Minimal overhead
    circuit_breaker_enabled=False,
)

# Create orchestrator
orchestrator = create_orchestrator(
    config=config,
    kernel_registry=get_global_kernel_registry(),
    backend_registry=get_global_backend_registry(),
)

# Use for dispatch
result = orchestrator.dispatch("attention.causal", inputs)
```

## Hot-Reload Configuration

### Basic Hot-Reload Setup

```python
from layerzero.dispatch import (
    HotReloadDispatcher,
    create_hot_reload_dispatcher,
    DispatchConfig,
    DispatchMode,
)
from layerzero.registry import get_global_kernel_registry

# Method 1: Using factory function
dispatcher = create_hot_reload_dispatcher(
    config_path="config/kernels.yaml",
    validate_on_reload=True,
    watch_interval_seconds=1.0,
    start_watching=True,
    kernel_registry=get_global_kernel_registry(),
)

# Method 2: Using config
config = DispatchConfig(
    mode=DispatchMode.HOT_RELOAD,
    config_path="config/kernels.yaml",
    watch_interval_seconds=1.0,
    validate_on_reload=True,
)

dispatcher = HotReloadDispatcher(config)
dispatcher.watch()  # Start file watching

# Dispatch as usual
result = dispatcher.dispatch("attention.causal", inputs)
```

### Hot-Reload with Callbacks

```python
from layerzero.dispatch import HotReloadDispatcher, ConfigVersion

dispatcher = HotReloadDispatcher(config)

# Register reload callbacks
def on_config_reloaded(version: ConfigVersion):
    """Called when config is successfully reloaded."""
    print(f"Config reloaded to version {version.version}")
    print(f"Config hash: {version.config_hash}")
    print(f"Loaded at: {version.loaded_at}")

    # Update metrics, notify other systems, etc.
    metrics.config_version.set(version.version)

def on_reload_error(error: Exception):
    """Called when reload fails."""
    print(f"Reload failed: {error}")

    # Alert, log error, etc.
    alert_channel.send(f"Config reload failed: {error}")

dispatcher.on_reload(on_config_reloaded)
dispatcher.on_error(on_reload_error)

# Start watching
dispatcher.watch()
```

### Manual Reload and Rollback

```python
from layerzero.dispatch import HotReloadDispatcher

dispatcher = HotReloadDispatcher(config)

# Manual reload with validation
try:
    dispatcher.reload(
        validate=True,
        grace_period_seconds=2.0,  # Wait for in-flight requests
    )
    print("Reload successful")
except Exception as e:
    print(f"Reload failed: {e}")

# Rollback to previous config
try:
    dispatcher.rollback()
    print("Rollback successful")
except Exception as e:
    print(f"Rollback failed: {e}")

# Check reload statistics
stats = dispatcher.stats
print(f"Total reloads: {stats.total_reloads}")
print(f"Failed reloads: {stats.failed_reloads}")
print(f"Last reload time: {stats.last_reload_time_ms}ms")
print(f"Current version: {stats.current_version}")
```

### Hot-Reload YAML Configuration

```yaml
# config/kernels.yaml
version: "1.0"

defaults:
  fallback_policy: torch_sdpa
  default_priority: 50

dispatch_rules:
  # High-performance attention for large batches
  - operation: "attention.*"
    conditions:
      batch_size_gte: 8
      dtype: [float16, bfloat16]
      platform: cuda
    kernel: flash_attention_v2
    priority: 100

  # Memory-efficient for long sequences
  - operation: "attention.*"
    conditions:
      seq_len_gt: 8192
    kernel: xformers_memory_efficient
    priority: 90

  # Decode attention
  - operation: "attention.decode"
    conditions:
      platform: cuda
    kernel: flashinfer_decode
    priority: 100

  # RMS normalization
  - operation: "rms_norm"
    conditions:
      platform: cuda
    kernel: liger_rms_norm
    priority: 100

kernel_locks:
  # Always use FlashAttention for prefill
  attention.prefill: flash_attention_v2

kernel_denies:
  - "*_experimental"
  - "*_debug"

kernel_allows:
  - "flash_attn_*"
  - "flashinfer_*"
  - "xformers_*"
  - "torch_sdpa"
  - "liger_*"

fallback_chains:
  attention.causal:
    - flash_attention_v2
    - xformers_memory_efficient
    - torch_sdpa

  attention.decode:
    - flashinfer_decode
    - flash_attention_v2
    - torch_sdpa

  rms_norm:
    - liger_rms_norm
    - triton_rms_norm
    - torch_rms_norm
```

## Circuit Breaker Usage

### Basic Circuit Breaker

```python
from layerzero.dispatch import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)

# Configure circuit breaker
config = CircuitBreakerConfig(
    failure_threshold=5,       # Open after 5 failures
    success_threshold=2,       # Close after 2 successes in half-open
    cooldown_seconds=30.0,     # Wait 30s before half-open
    half_open_max_calls=3,     # Allow 3 test calls in half-open
)

# Create circuit breaker for a kernel
circuit = CircuitBreaker("flash_attn_v2_attention", config)

# Check if call is allowed
if circuit.allow_request():
    try:
        # Execute kernel
        result = kernel_impl(**kwargs)
        circuit.record_success()
    except Exception as e:
        circuit.record_failure(e)
        raise
else:
    # Circuit is open, use fallback
    result = fallback_kernel(**kwargs)

# Check circuit state
print(f"State: {circuit.state}")  # CLOSED, OPEN, or HALF_OPEN
print(f"Failure count: {circuit.failure_count}")
print(f"Success count: {circuit.success_count}")
```

### Circuit Breaker with State Listeners

```python
from layerzero.dispatch import CircuitBreaker, CircuitState

circuit = CircuitBreaker("my-kernel")

def on_state_change(old_state: CircuitState, new_state: CircuitState):
    """Log circuit state changes."""
    print(f"Circuit state changed: {old_state.name} -> {new_state.name}")

    if new_state == CircuitState.OPEN:
        # Alert: kernel is failing
        alert_channel.send(f"Circuit opened for {circuit.kernel_id}")
    elif new_state == CircuitState.CLOSED:
        # Recovery complete
        alert_channel.send(f"Circuit recovered for {circuit.kernel_id}")

circuit.add_listener(on_state_change)
```

### Circuit Breaker Manager

```python
from layerzero.dispatch import CircuitBreakerManager, CircuitBreakerConfig

# Create manager for multiple kernels
manager = CircuitBreakerManager(
    default_config=CircuitBreakerConfig(
        failure_threshold=5,
        cooldown_seconds=30.0,
    )
)

# Get or create circuit for kernel
circuit = manager.get_circuit("flash_attn_v2_attention")

# Use circuit
if circuit.allow_request():
    try:
        result = execute_kernel()
        circuit.record_success()
    except Exception as e:
        circuit.record_failure(e)

# Get all circuit stats
for kernel_id, stats in manager.get_all_stats().items():
    print(f"{kernel_id}: {stats.state.name}, failures={stats.failure_count}")

# Reset a specific circuit
manager.reset_circuit("flash_attn_v2_attention")

# Reset all circuits
manager.reset_all()
```

### Circuit Breaker with Dispatch Integration

```python
from layerzero.dispatch import (
    create_orchestrator,
    DispatchConfig,
    DispatchMode,
)

# Configure with circuit breaker
config = DispatchConfig(
    mode=DispatchMode.DYNAMIC,
    circuit_breaker_enabled=True,
    failure_threshold=5,
    recovery_timeout_seconds=30.0,
    enable_fallback=True,
    max_fallback_attempts=3,
)

orchestrator = create_orchestrator(
    config=config,
    kernel_registry=kernel_registry,
    backend_registry=backend_registry,
)

# Dispatch with automatic circuit breaker handling
try:
    result = orchestrator.dispatch("attention.causal", inputs)
except CircuitOpenError as e:
    print(f"All circuits open: {e}")
    # Handle complete failure
except FallbackChainExhaustedError as e:
    print(f"All fallbacks failed: {e}")
    # Handle complete failure
```

## Custom Dispatcher Creation

### Implementing a Custom Dispatcher

```python
from layerzero.dispatch.protocols import BaseDispatcher
from layerzero.dispatch import (
    DispatchMode,
    DispatchResult,
    DispatchTiming,
    DispatchConfig,
)
from layerzero.selection import SelectionContext
from layerzero.registry import KernelSpec
import torch
import time

class CustomDispatcher(BaseDispatcher):
    """Custom dispatcher with specialized logic."""

    def __init__(
        self,
        config: DispatchConfig,
        kernel_registry,
        custom_selector,
    ):
        self._config = config
        self._kernel_registry = kernel_registry
        self._custom_selector = custom_selector
        self._call_count = 0

    @property
    def mode(self) -> DispatchMode:
        return DispatchMode.DYNAMIC

    def dispatch(
        self,
        operation: str,
        inputs: dict[str, torch.Tensor],
        context: SelectionContext | None = None,
        **kwargs,
    ) -> DispatchResult:
        start_time = time.perf_counter_ns()
        self._call_count += 1

        # Build context if not provided
        if context is None:
            context = self._build_context(inputs, **kwargs)

        # Custom kernel selection logic
        selection_start = time.perf_counter_ns()
        kernel_spec = self._custom_selector.select(operation, context)
        selection_end = time.perf_counter_ns()

        # Execute kernel
        execution_start = time.perf_counter_ns()
        output = self._execute_kernel(kernel_spec, inputs, kwargs)
        execution_end = time.perf_counter_ns()

        end_time = time.perf_counter_ns()

        # Build timing info
        timing = DispatchTiming(
            total_ns=end_time - start_time,
            selection_ns=selection_end - selection_start,
            execution_ns=execution_end - execution_start,
        )

        return DispatchResult(
            output=output,
            kernel_id=kernel_spec.kernel_id,
            timing=timing,
            was_fallback=False,
        )

    def get_kernel_for_operation(
        self,
        operation: str,
        context: SelectionContext,
    ) -> KernelSpec:
        return self._custom_selector.select(operation, context)

    def _build_context(self, inputs: dict[str, torch.Tensor], **kwargs) -> SelectionContext:
        # Extract tensor info
        first_tensor = next(iter(inputs.values()))
        return SelectionContext(
            batch_size=first_tensor.shape[0],
            dtype=str(first_tensor.dtype).replace("torch.", ""),
            device=str(first_tensor.device),
            **kwargs,
        )

    def _execute_kernel(self, kernel_spec: KernelSpec, inputs: dict, kwargs: dict):
        # Map arguments and execute
        mapped_kwargs = kernel_spec.map_arguments(inputs, kwargs)
        return kernel_spec.impl(**mapped_kwargs)

# Use custom dispatcher
custom_selector = MyCustomSelector()
dispatcher = CustomDispatcher(config, kernel_registry, custom_selector)
result = dispatcher.dispatch("attention.causal", inputs)
```

### Custom Dispatcher with Policy Integration

```python
from layerzero.dispatch.protocols import BaseDispatcher
from layerzero.policy import PolicyEngine

class PolicyAwareDispatcher(BaseDispatcher):
    """Dispatcher that integrates with policy engine."""

    def __init__(self, config, kernel_registry, policy_engine: PolicyEngine):
        self._config = config
        self._registry = kernel_registry
        self._policy = policy_engine
        self._cache = {}

    def dispatch(
        self,
        operation: str,
        inputs: dict[str, torch.Tensor],
        context: SelectionContext | None = None,
        **kwargs,
    ) -> DispatchResult:
        if context is None:
            context = self._build_context(inputs, **kwargs)

        # Check cache
        cache_key = self._make_cache_key(operation, context)
        if cache_key in self._cache:
            kernel_spec = self._cache[cache_key]
        else:
            # Get candidates from registry
            candidates = self._registry.get_by_operation(operation)

            # Apply policy filtering
            filtered = self._policy.filter_kernels(candidates, context)

            # Apply policy scoring
            scored = self._policy.score_kernels(filtered, context)

            # Select best
            kernel_spec = max(scored, key=lambda x: x[1])[0]

            # Cache result
            self._cache[cache_key] = kernel_spec

        # Execute
        output = kernel_spec.impl(**self._map_args(kernel_spec, inputs, kwargs))

        return DispatchResult(
            output=output,
            kernel_id=kernel_spec.kernel_id,
        )
```

### Registering Custom Dispatcher with Orchestrator

```python
from layerzero.dispatch import (
    DispatchOrchestrator,
    DispatchConfig,
    DispatchMode,
)

# Create custom dispatcher
custom_dispatcher = CustomDispatcher(config, kernel_registry, custom_selector)

# Create orchestrator with custom dispatcher
orchestrator = DispatchOrchestrator(config)

# Register custom dispatcher for a mode
orchestrator.register_dispatcher(DispatchMode.DYNAMIC, custom_dispatcher)

# Use orchestrator
result = orchestrator.dispatch("attention.causal", inputs)
```

## Advanced Examples

### Multi-GPU Dispatch

```python
from layerzero.dispatch import dispatch, DispatchConfig, DispatchMode
from layerzero.selection import SelectionContext
import torch

def multi_gpu_dispatch(inputs_per_device: dict[str, dict[str, torch.Tensor]]):
    """Dispatch across multiple GPUs."""
    results = {}

    for device_id, inputs in inputs_per_device.items():
        # Build device-specific context
        context = SelectionContext(
            device=device_id,
            batch_size=inputs["query"].shape[0],
            seq_len=inputs["query"].shape[1],
            dtype="float16",
            platform="cuda",
        )

        # Dispatch on specific device
        result = dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=context,
        )

        results[device_id] = result

    return results

# Usage
inputs_per_device = {
    "cuda:0": {"query": q0, "key": k0, "value": v0},
    "cuda:1": {"query": q1, "key": k1, "value": v1},
}

results = multi_gpu_dispatch(inputs_per_device)
```

### Batch Processing with Different Kernels

```python
from layerzero.dispatch import get_global_dispatcher
from layerzero.selection import SelectionContext

def process_mixed_batch(batches: list[dict]):
    """Process batches that may require different kernels."""
    dispatcher = get_global_dispatcher()
    results = []

    for batch in batches:
        # Build context based on batch characteristics
        context = SelectionContext(
            batch_size=batch["query"].shape[0],
            seq_len=batch["query"].shape[1],
            dtype=str(batch["query"].dtype).replace("torch.", ""),
            device=str(batch["query"].device),
            is_causal=batch.get("is_causal", True),
        )

        # Dispatcher will select appropriate kernel
        result = dispatcher.dispatch(
            operation="attention.causal",
            inputs={
                "query": batch["query"],
                "key": batch["key"],
                "value": batch["value"],
            },
            context=context,
        )

        results.append(result)

    return results
```

### Inference Server Integration

```python
from layerzero.dispatch import (
    create_orchestrator,
    DispatchConfig,
    DispatchMode,
)
from layerzero.registry import get_global_kernel_registry, get_global_backend_registry
import asyncio

class InferenceDispatcher:
    """Dispatch manager for inference server."""

    def __init__(self):
        self._config = DispatchConfig(
            mode=DispatchMode.DYNAMIC,
            enable_cache=True,
            cache_size=100000,
            circuit_breaker_enabled=True,
            failure_threshold=10,
            recovery_timeout_seconds=60.0,
        )

        self._orchestrator = create_orchestrator(
            config=self._config,
            kernel_registry=get_global_kernel_registry(),
            backend_registry=get_global_backend_registry(),
        )

        self._warmup_complete = False

    async def warmup(self, representative_shapes: list[dict]):
        """Warm up caches with representative workloads."""
        for shape in representative_shapes:
            dummy_inputs = self._create_dummy_inputs(shape)
            try:
                self._orchestrator.dispatch("attention.causal", dummy_inputs)
            except Exception:
                pass  # Ignore warmup errors

        self._warmup_complete = True

    def dispatch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = True,
    ):
        """Dispatch attention computation."""
        if not self._warmup_complete:
            raise RuntimeError("Warmup not complete")

        return self._orchestrator.dispatch(
            operation="attention.causal",
            inputs={"query": query, "key": key, "value": value},
            is_causal=is_causal,
        )

    def get_metrics(self):
        """Get dispatch metrics for monitoring."""
        telemetry = self._orchestrator.get_telemetry()
        return {
            "cache_hit_rate": telemetry.cache_hit_rate,
            "dispatch_count": telemetry.dispatch_count,
            "fallback_count": telemetry.fallback_count,
            "p50_latency_us": telemetry.p50_latency_us,
            "p99_latency_us": telemetry.p99_latency_us,
        }

    def _create_dummy_inputs(self, shape: dict):
        return {
            "query": torch.randn(shape["batch"], shape["seq"], shape["heads"], shape["dim"]),
            "key": torch.randn(shape["batch"], shape["seq"], shape["heads"], shape["dim"]),
            "value": torch.randn(shape["batch"], shape["seq"], shape["heads"], shape["dim"]),
        }

# Usage
dispatcher = InferenceDispatcher()

# Warmup
asyncio.run(dispatcher.warmup([
    {"batch": 1, "seq": 512, "heads": 32, "dim": 128},
    {"batch": 8, "seq": 2048, "heads": 32, "dim": 128},
    {"batch": 32, "seq": 8192, "heads": 32, "dim": 128},
]))

# Process requests
result = dispatcher.dispatch_attention(query, key, value)

# Monitor metrics
metrics = dispatcher.get_metrics()
print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")
```

### A/B Testing with Hot-Reload

```python
from layerzero.dispatch import (
    HotReloadDispatcher,
    DispatchConfig,
    DispatchMode,
)
import random

class ABTestingDispatcher:
    """Dispatcher for A/B testing kernel configurations."""

    def __init__(self, config_a_path: str, config_b_path: str, b_ratio: float = 0.1):
        self._config_a = DispatchConfig(
            mode=DispatchMode.HOT_RELOAD,
            config_path=config_a_path,
            validate_on_reload=True,
        )
        self._config_b = DispatchConfig(
            mode=DispatchMode.HOT_RELOAD,
            config_path=config_b_path,
            validate_on_reload=True,
        )

        self._dispatcher_a = HotReloadDispatcher(self._config_a)
        self._dispatcher_b = HotReloadDispatcher(self._config_b)
        self._b_ratio = b_ratio

        # Start watching both configs
        self._dispatcher_a.watch()
        self._dispatcher_b.watch()

        # Metrics
        self._a_calls = 0
        self._b_calls = 0

    def dispatch(self, operation: str, inputs: dict, **kwargs):
        """Dispatch with A/B traffic split."""
        use_b = random.random() < self._b_ratio

        if use_b:
            self._b_calls += 1
            return self._dispatcher_b.dispatch(operation, inputs, **kwargs)
        else:
            self._a_calls += 1
            return self._dispatcher_a.dispatch(operation, inputs, **kwargs)

    def set_b_ratio(self, ratio: float):
        """Adjust B traffic ratio."""
        self._b_ratio = max(0.0, min(1.0, ratio))

    def get_stats(self):
        """Get A/B test statistics."""
        total = self._a_calls + self._b_calls
        return {
            "a_calls": self._a_calls,
            "b_calls": self._b_calls,
            "a_ratio": self._a_calls / total if total > 0 else 0,
            "b_ratio": self._b_calls / total if total > 0 else 0,
            "config_a_version": self._dispatcher_a.stats.current_version,
            "config_b_version": self._dispatcher_b.stats.current_version,
        }

# Usage
ab_dispatcher = ABTestingDispatcher(
    config_a_path="config/kernels_stable.yaml",
    config_b_path="config/kernels_experimental.yaml",
    b_ratio=0.05,  # 5% traffic to experimental
)

# Process requests
result = ab_dispatcher.dispatch("attention.causal", inputs)

# Gradually increase B traffic
ab_dispatcher.set_b_ratio(0.10)  # 10%
ab_dispatcher.set_b_ratio(0.50)  # 50%

# Check stats
print(ab_dispatcher.get_stats())
```
