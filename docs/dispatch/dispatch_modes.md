# Dispatch Modes

LayerZero supports four dispatch modes, each optimized for different use cases. This document explains when to use each mode, their trade-offs, and configuration options.

## Overview

| Mode | Overhead | Flexibility | Configuration | Use Case |
|------|----------|-------------|---------------|----------|
| STATIC | Zero | Low | Compile-time | Production with known kernels |
| DYNAMIC | ~100-500ns | High | Runtime | General production |
| HOT_RELOAD | ~1-10ms reload | Very High | File-based | Development, A/B testing |
| CONFIG | ~100ns | Medium | YAML-driven | Ops-controlled deployment |
| AUTO | Variable | Adaptive | Automatic | Let system choose |

## Static Dispatch (STATIC)

Static dispatch achieves near-zero overhead through compile-time/import-time kernel resolution.

### Characteristics

- **Zero Runtime Overhead**: Kernel is resolved at initialization
- **O(1) Lookup**: Dict-based lookup from pre-computed mapping
- **Match Statement Dispatch**: Type-safe operation routing (Python 3.10+)
- **Immutable Registry**: Registry is frozen after initialization

### When to Use

- Production deployments with fixed kernel selection
- Latency-critical applications where every nanosecond matters
- Inference servers with known model configurations
- Edge deployments with limited resources

### Trade-offs

| Advantage | Disadvantage |
|-----------|--------------|
| Zero dispatch overhead | No runtime flexibility |
| Predictable behavior | Cannot adapt to context |
| Simple debugging | Manual fallback handling |
| Low memory usage | Must restart to change kernels |

### Configuration

```python
from layerzero.dispatch import (
    StaticDispatcher,
    StaticDispatcherBuilder,
    StaticKernelRegistry,
    DispatchConfig,
    DispatchMode,
)

# Method 1: Using the builder pattern
dispatcher = (
    StaticDispatcherBuilder()
    .with_kernel(flash_attn_spec, operation="attention.causal", default=True)
    .with_kernel(sdpa_spec, operation="attention.causal")
    .with_kernel(rms_norm_spec, operation="rms_norm", default=True)
    .with_config(enable_telemetry=True)
    .build()
)

# Method 2: Using static_kernel_map in config
config = DispatchConfig(
    mode=DispatchMode.STATIC,
    static_kernel_map={
        "attention.causal": "flash_attn_v2_attention",
        "attention.cross": "flash_attn_v2_attention",
        "rms_norm": "liger_rms_norm",
    },
)

# Method 3: From kernel registry
registry = StaticKernelRegistry()
registry.register_from_spec(flash_attn_spec, is_default=True)
registry.register_from_spec(sdpa_spec)
registry.freeze()  # No more modifications allowed

dispatcher = StaticDispatcher(config, registry=registry)
```

### Operation Type Mapping

Static dispatch uses an enum-based operation type system for efficient matching:

```python
from layerzero.dispatch.static import OperationType, get_operation_type

# Map operation strings to enum
op_type = get_operation_type("attention.causal")  # -> OperationType.ATTENTION_CAUSAL

# Supported operation types
OperationType.ATTENTION_CAUSAL
OperationType.ATTENTION_FULL
OperationType.ATTENTION_SLIDING_WINDOW
OperationType.NORM_RMS
OperationType.NORM_LAYER
OperationType.ROPE
OperationType.SWIGLU
# ... and more
```

## Dynamic Dispatch (DYNAMIC)

Dynamic dispatch provides runtime kernel selection with ~100-500ns overhead.

### Characteristics

- **Runtime Selection**: Selects kernel based on current context
- **SelectionEngine Integration**: Uses full selection pipeline
- **Circuit Breaker**: Automatic fault isolation per kernel
- **Fallback Chain**: Graceful degradation on failure
- **MVCC Caching**: Concurrent cache access without locks

### When to Use

- General production deployments
- When kernel selection depends on tensor shapes/dtypes
- Multi-tenant serving with diverse workloads
- When graceful degradation is important

### Trade-offs

| Advantage | Disadvantage |
|-----------|--------------|
| Context-aware selection | Small dispatch overhead |
| Automatic fallback | Higher memory for caching |
| Policy integration | More complex debugging |
| Runtime adaptation | Non-deterministic selection |

### Configuration

```python
from layerzero.dispatch import (
    DynamicDispatcher,
    create_dynamic_dispatcher,
    DispatchConfig,
    DispatchMode,
)

# Using factory function
dispatcher = create_dynamic_dispatcher(
    kernel_registry=kernel_registry,
    backend_registry=backend_registry,
    config=DispatchConfig(
        mode=DispatchMode.DYNAMIC,
        enable_cache=True,
        cache_size=10000,
        cache_ttl_seconds=3600.0,
        circuit_breaker_enabled=True,
        failure_threshold=5,
        recovery_timeout_seconds=30.0,
    ),
    use_mvcc_cache=True,
)

# Access components
engine = dispatcher.selection_engine
circuit_breaker = dispatcher.circuit_breaker
```

### Selection Pipeline

1. Build SelectionContext from inputs
2. Check MVCC cache for cached selection
3. Check circuit breaker for kernel health
4. Use SelectionEngine to select best kernel
5. Execute with fallback support
6. Update circuit breaker and telemetry

## Hot-Reload Dispatch (HOT_RELOAD)

Hot-reload dispatch enables zero-downtime configuration updates for development and A/B testing.

### Characteristics

- **File Watching**: Monitors config files for changes
- **Atomic Config Swap**: Double-buffered configuration
- **Validation Before Apply**: Dry-run validation
- **Automatic Rollback**: Reverts on reload failure
- **Grace Period**: Waits for in-flight requests

### When to Use

- Development and testing environments
- A/B testing different kernel configurations
- Gradual rollout of new kernels
- Debugging production issues without restart

### Trade-offs

| Advantage | Disadvantage |
|-----------|--------------|
| Zero-downtime updates | Higher reload latency |
| Instant config changes | File watching overhead |
| Rollback capability | Memory for two configs |
| Development friendly | More complex state |

### Configuration

```python
from layerzero.dispatch import (
    HotReloadDispatcher,
    create_hot_reload_dispatcher,
    DispatchConfig,
    DispatchMode,
)

# Using factory function
dispatcher = create_hot_reload_dispatcher(
    config_path="config/kernels.yaml",
    validate_on_reload=True,
    watch_interval_seconds=1.0,
    start_watching=True,
    selection_engine=selection_engine,
    kernel_registry=kernel_registry,
)

# Manual configuration
config = DispatchConfig(
    mode=DispatchMode.HOT_RELOAD,
    config_path="config/kernels.yaml",
    watch_interval_seconds=1.0,
    validate_on_reload=True,
)

dispatcher = HotReloadDispatcher(config)
dispatcher.watch()  # Start file watching

# Manual reload
dispatcher.reload(validate=True, grace_period_seconds=1.0)

# Rollback to previous config
dispatcher.rollback()

# Check reload statistics
stats = dispatcher.stats
print(f"Total reloads: {stats.total_reloads}")
print(f"Failed reloads: {stats.failed_reloads}")
print(f"Last reload time: {stats.last_reload_time_ms}ms")
```

### Reload States

```python
from layerzero.dispatch import ReloadState

ReloadState.IDLE        # Normal operation
ReloadState.RELOADING   # Loading new config
ReloadState.VALIDATING  # Validating new config
ReloadState.APPLYING    # Applying new config
ReloadState.ROLLING_BACK # Rolling back on failure
```

### Callbacks

```python
# Register reload callbacks
def on_config_reloaded(version: ConfigVersion):
    print(f"Config reloaded to version {version.version}")
    print(f"Config hash: {version.config_hash}")

def on_reload_error(error: Exception):
    print(f"Reload failed: {error}")

dispatcher.on_reload(on_config_reloaded)
dispatcher.on_error(on_reload_error)
```

## Config-Driven Dispatch (CONFIG)

Config-driven dispatch enables ops-controlled kernel deployment via YAML configuration.

### Characteristics

- **YAML Configuration**: Declarative kernel rules
- **Rule-Based Selection**: Condition-based kernel matching
- **Pattern Matching**: Glob patterns for kernel filtering
- **Priority Override**: Per-rule priority control
- **Compiled Rules**: Pre-compiled for efficient evaluation

### When to Use

- Ops-controlled production deployments
- Complex kernel selection logic
- Multi-environment configurations
- Gradual kernel rollouts

### Trade-offs

| Advantage | Disadvantage |
|-----------|--------------|
| Declarative config | Learning curve for YAML schema |
| Ops-friendly | Rule evaluation overhead |
| Flexible rules | Requires config management |
| Version controlled | Static until reload |

### Configuration

```yaml
# config/kernels.yaml
version: "1.0"

defaults:
  fallback_policy: torch_sdpa
  default_priority: 50

dispatch_rules:
  - operation: "attention.*"
    conditions:
      batch_size_gte: 8
      dtype: [float16, bfloat16]
    kernel: flash_attention_v2
    priority: 100

  - operation: "attention.causal"
    conditions:
      seq_len_gt: 8192
    kernel: xformers_memory_efficient
    priority: 90

kernel_locks:
  attention.causal: flash_attention_v2

kernel_denies:
  - "*_experimental"

kernel_allows:
  - "flash_attn.*"
  - "torch_sdpa"

fallback_chains:
  attention.causal:
    - flash_attention_v2
    - xformers_memory_efficient
    - torch_sdpa
```

```python
from layerzero.dispatch import (
    ConfigDrivenDispatcher,
    create_config_dispatcher,
)

# Using factory function
dispatcher = create_config_dispatcher(
    config_path="config/kernels.yaml",
    kernel_registry=kernel_registry,
    cache_size=10000,
    cache_ttl_seconds=60.0,
)

# Reload configuration
dispatcher.reload_config("config/kernels.yaml")

# Explain selection for debugging
explanation = dispatcher.explain_selection("attention.causal", context)
print(explanation)
```

### Rule Conditions

Supported condition operators:

| Suffix | Operator | Example |
|--------|----------|---------|
| (none) | equals | `dtype: float16` |
| `_eq` | equals | `batch_size_eq: 8` |
| `_ne` | not equals | `dtype_ne: float32` |
| `_gt` | greater than | `seq_len_gt: 1024` |
| `_gte` | greater or equal | `batch_size_gte: 8` |
| `_lt` | less than | `head_dim_lt: 256` |
| `_lte` | less or equal | `num_heads_lte: 32` |
| `_in` | in list | `dtype_in: [float16, bfloat16]` |
| `_not_in` | not in list | `platform_not_in: [cpu]` |
| `_match` | glob pattern | `device_match: "cuda:*"` |
| `_regex` | regex match | `kernel_regex: "flash.*"` |

## AUTO Mode

AUTO mode automatically selects the best dispatch mode based on configuration and context.

### Selection Logic

```python
def _select_best_mode(self, operation: str, context: SelectionContext) -> DispatchMode:
    # If static kernel map has this operation, use static
    if operation in self._config.static_kernel_map:
        return DispatchMode.STATIC

    # If hot-reload is enabled and config path exists, use hot-reload
    if self._config.config_path and self._config.mode == DispatchMode.HOT_RELOAD:
        return DispatchMode.HOT_RELOAD

    # Default to dynamic for flexibility
    return self._default_mode
```

### Configuration

```python
config = DispatchConfig(
    mode=DispatchMode.AUTO,
    # Provide static map for known operations
    static_kernel_map={
        "attention.causal": "flash_attn_v2_attention",
    },
    # Fallback to dynamic for unknown operations
)

orchestrator = DispatchOrchestrator(config, default_mode=DispatchMode.DYNAMIC)
```

## Mode Switching

The orchestrator supports runtime mode switching:

```python
from layerzero.dispatch import get_global_dispatcher, DispatchMode

dispatcher = get_global_dispatcher()

# Change default mode
dispatcher.set_default_mode(DispatchMode.DYNAMIC)

# Dispatch with explicit mode override
result = dispatcher.dispatch(
    "attention.causal",
    inputs,
    mode=DispatchMode.STATIC,  # Override for this call
)
```

## Fallback Between Modes

The orchestrator maintains fallback chains between modes:

```python
# Default fallback chain
_mode_fallback_chain = {
    DispatchMode.STATIC: [DispatchMode.DYNAMIC],
    DispatchMode.DYNAMIC: [DispatchMode.CONFIG],
    DispatchMode.HOT_RELOAD: [DispatchMode.DYNAMIC],
    DispatchMode.CONFIG: [DispatchMode.DYNAMIC],
    DispatchMode.AUTO: [DispatchMode.DYNAMIC, DispatchMode.STATIC],
}
```

When a dispatch fails in one mode, the orchestrator automatically tries the next mode in the chain.

## Best Practices

### Choosing the Right Mode

1. **Production with known kernels**: Use STATIC
2. **Production with diverse workloads**: Use DYNAMIC
3. **Development/Testing**: Use HOT_RELOAD
4. **Ops-controlled deployment**: Use CONFIG
5. **Unsure**: Use AUTO and monitor telemetry

### Performance Optimization

1. **Enable caching** in DYNAMIC and CONFIG modes
2. **Use STATIC** for latency-critical paths
3. **Tune cache TTL** based on workload patterns
4. **Monitor cache hit rates** via telemetry

### Reliability

1. **Enable circuit breakers** in DYNAMIC mode
2. **Configure fallback chains** for critical operations
3. **Use validation** in HOT_RELOAD mode
4. **Test configurations** before production deployment
