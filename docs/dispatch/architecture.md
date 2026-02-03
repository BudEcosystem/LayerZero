# Dispatch System Architecture

This document provides a detailed overview of the LayerZero kernel dispatch system architecture, its components, data flow, and integration points.

## System Architecture

```
                                    Client Request
                                          |
                                          v
    +------------------------------------------------------------------+
    |                      DispatchOrchestrator                         |
    |  +------------------------------------------------------------+  |
    |  |                    Mode Selector                            |  |
    |  |  - AUTO: Select best mode based on context                  |  |
    |  |  - Explicit: Use specified mode                             |  |
    |  +------------------------------------------------------------+  |
    |                              |                                    |
    |         +--------------------+--------------------+               |
    |         |                    |                    |               |
    |         v                    v                    v               |
    |  +-----------+        +-----------+        +-----------+         |
    |  |  STATIC   |        |  DYNAMIC  |        | HOT_RELOAD|         |
    |  | Dispatcher|        | Dispatcher|        | Dispatcher|         |
    |  +-----------+        +-----------+        +-----------+         |
    |         |                    |                    |               |
    |         +--------------------+--------------------+               |
    |                              |                                    |
    |                              v                                    |
    |  +------------------------------------------------------------+  |
    |  |                    KernelExecutorImpl                       |  |
    |  |  - Argument mapping                                         |  |
    |  |  - Layout transformation                                    |  |
    |  |  - Actual kernel invocation                                 |  |
    |  +------------------------------------------------------------+  |
    +------------------------------------------------------------------+
                                   |
                                   v
    +------------------------------------------------------------------+
    |                       Backend Adapters                            |
    |  +------------+  +------------+  +------------+  +------------+  |
    |  | FlashAttn  |  | FlashInfer |  |  xFormers  |  | torch SDPA |  |
    |  +------------+  +------------+  +------------+  +------------+  |
    +------------------------------------------------------------------+
```

## Core Components

### 1. DispatchOrchestrator

The `DispatchOrchestrator` is the unified entry point for all kernel dispatch operations. It coordinates multiple dispatch modes and provides:

- **Mode Selection**: Automatically or explicitly selects the appropriate dispatch mode
- **Fallback Chain**: Falls back between modes when one fails
- **Unified Telemetry**: Aggregates metrics across all modes
- **Configuration Management**: Hot-reloads configuration across dispatchers

```python
class DispatchOrchestrator:
    """Unified dispatch orchestrator."""

    __slots__ = (
        "_config",           # DispatchConfig
        "_dispatchers",      # Dict[DispatchMode, Dispatcher]
        "_executor",         # KernelExecutorImpl
        "_telemetry",        # OrchestratorTelemetry
        "_default_mode",     # Default mode for AUTO
        "_selection_engine", # SelectionEngine reference
    )
```

### 2. Dispatcher Implementations

Each dispatch mode has a dedicated dispatcher implementation:

#### StaticDispatcher
- Zero-overhead dispatch using compile-time resolution
- Pre-computed kernel mapping (operation -> kernel)
- O(1) dict lookup with inline caching
- Match statement dispatch for type safety

#### DynamicDispatcher
- Runtime kernel selection via SelectionEngine
- Circuit breaker for fault isolation
- Fallback chain for kernel failures
- MVCC cache for selection results

#### HotReloadDispatcher
- File watching for config changes (watchdog/polling)
- Atomic config swap with rollback
- Grace period for in-flight requests
- Validation before apply

#### ConfigDrivenDispatcher
- YAML-based rule evaluation
- Condition-based kernel selection
- Allow/deny pattern matching
- Compiled rules for efficiency

### 3. KernelExecutorImpl

The executor bridges kernel selection to actual execution:

```python
class KernelExecutorImpl:
    """Kernel executor implementation."""

    def execute(
        self,
        kernel_spec: KernelSpec,
        inputs: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute a kernel with given inputs."""
        # 1. Map arguments to kernel-specific format
        mapped_kwargs = self._map_arguments(kernel_spec, inputs, kwargs)

        # 2. Execute kernel implementation
        output = kernel_spec.impl(**mapped_kwargs)

        # 3. Record success/failure metrics
        return output
```

### 4. Circuit Breaker

The circuit breaker provides fault tolerance:

```
                  success
        +------------------------+
        |                        |
        v                        |
    +-------+    failure    +------+
    |CLOSED |-------------->| OPEN |
    +-------+   threshold   +------+
        ^                        |
        |    cooldown elapsed    |
        |                        v
        |                  +-----------+
        +------------------| HALF_OPEN |
            success        +-----------+
```

**States:**
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Too many failures, requests blocked
- **HALF_OPEN**: Testing recovery with limited requests

## Data Flow

### Selection to Dispatch Flow

```
1. Client Request
       |
       v
2. Build SelectionContext
   - Device detection
   - Tensor shapes/dtypes
   - Operation parameters
       |
       v
3. Check Cache (if enabled)
   - Cache key = hash(context + policy)
   - Return cached plan if hit
       |
       v (cache miss)
4. Kernel Selection
   - Policy lock check
   - Get candidates by operation
   - Apply allow/deny rules
   - Filter by compatibility
   - Score candidates
   - Select highest score
       |
       v
5. Create ExecutionPlan
   - kernel_spec
   - pre_transforms
   - post_transforms
       |
       v
6. Cache Plan (if enabled)
       |
       v
7. Check Circuit Breaker
   - If open, try fallback
   - If closed, proceed
       |
       v
8. Execute Kernel
   - Map arguments
   - Transform tensors
   - Invoke implementation
       |
       v
9. Build DispatchResult
   - output tensor
   - timing metrics
   - debug info
       |
       v
10. Update Telemetry
    - Record success/failure
    - Update circuit breaker
```

### Configuration Reload Flow

```
1. Config File Change Detected
       |
       v
2. Debounce (avoid rapid reloads)
       |
       v
3. Parse New Configuration
   - YAML/JSON parsing
   - Schema validation
       |
       v
4. Validate Configuration
   - Check kernel references
   - Validate rule syntax
   - Dry-run rules
       |
       v (validation passed)
5. Wait for Grace Period
   - Let in-flight requests complete
       |
       v
6. Atomic Config Swap
   - previous_config = current_config
   - current_config = new_config
       |
       v
7. Invalidate Caches
   - Clear rule evaluation cache
   - Invalidate selection cache
       |
       v
8. Fire Reload Callbacks
   - Notify listeners
   - Log success
```

## Integration Points

### SelectionEngine Integration

The dispatch system integrates with the existing SelectionEngine:

```python
class DynamicDispatcher:
    def _select_kernel(self, context: SelectionContext) -> ExecutionPlan:
        # Use SelectionEngine for kernel selection
        plan = self._selection_engine.select(
            context,
            use_cache=self._config.enable_cache,
            debug=False,
        )
        return plan
```

The SelectionEngine pipeline:
1. Check policy locks
2. Check cache
3. Get candidates
4. Apply policy filters
5. Filter by compatibility
6. Score candidates
7. Select best
8. Cache result

### KernelRegistry Integration

The dispatch system accesses kernels through the KernelRegistry:

```python
# Get all kernels for an operation
candidates = kernel_registry.get_by_operation("attention.causal")

# Get specific kernel by ID
kernel = kernel_registry.get("flash_attn_v2_attention")
```

### BackendRegistry Integration

Backend health is tracked via the BackendRegistry:

```python
# Record successful execution
backend_registry.record_success(kernel_spec.source)

# Record failure
backend_registry.record_failure(kernel_spec.source, error)

# Check backend health
health = backend_registry.get_health(backend_id)
```

## Thread Safety Model

### Immutability

- `DispatchConfig`: Immutable after creation (replace for updates)
- `StaticKernelEntry`: Frozen dataclass
- `CompiledConfig`: Immutable compiled rules
- `ConfigVersion`: Frozen dataclass

### Fine-Grained Locking

- `CircuitBreaker`: Per-circuit RLock
- `RuleEvaluationCache`: Per-cache Lock
- `HotReloadDispatcher`: Separate locks for config, stats, in-flight

### Atomic Operations

- Config reload uses atomic pointer swap
- Circuit breaker state uses atomic operations
- Telemetry counters use atomic increments

### Lock-Free Patterns

- Static dispatch: No locks in hot path
- Cache reads: MVCC for concurrent access
- Inline caching: Thread-local when possible

## Memory Model

### Allocation Strategy

```
Per-Request Memory:
  - SelectionContext: ~200 bytes
  - DispatchResult: ~100 bytes
  - DispatchTiming: ~40 bytes

Cached Data:
  - ExecutionPlan: ~300 bytes per entry
  - CompiledRule: ~200 bytes per rule
  - CircuitStats: ~80 bytes per kernel

Static Data:
  - StaticKernelRegistry: O(n) kernels
  - CompiledConfig: O(r) rules
  - OperationTypeMap: O(1) constant
```

### Cache Memory

```python
# Selection cache sizing
cache_size = config.cache_size  # Default: 10000 entries
max_memory = cache_size * 300   # ~3MB for default

# MVCC cache sharding
num_shards = 256
entries_per_shard = cache_size // num_shards
```

## Extension Points

### Custom Dispatcher

```python
from layerzero.dispatch.protocols import BaseDispatcher

class CustomDispatcher(BaseDispatcher):
    @property
    def mode(self) -> DispatchMode:
        return DispatchMode.DYNAMIC

    def dispatch(
        self,
        operation: str,
        inputs: dict[str, torch.Tensor],
        context: SelectionContext | None = None,
        **kwargs: Any,
    ) -> DispatchResult:
        # Custom dispatch logic
        pass

    def get_kernel_for_operation(
        self,
        operation: str,
        context: SelectionContext,
    ) -> KernelSpec:
        # Custom selection logic
        pass
```

### Custom Config Validator

```python
from layerzero.dispatch import HotReloadDispatcher

def validate_kernel_exists(config: dict[str, Any]) -> list[str]:
    """Custom validator that checks kernel references."""
    errors = []
    for rule in config.get("dispatch_rules", []):
        kernel_id = rule.get("kernel")
        if not kernel_registry.get(kernel_id):
            errors.append(f"Unknown kernel: {kernel_id}")
    return errors

dispatcher = HotReloadDispatcher(config, config_path=path)
dispatcher.add_validator(validate_kernel_exists)
```

### Custom Circuit Breaker Listener

```python
from layerzero.dispatch import CircuitBreaker, CircuitState

def on_state_change(old_state: CircuitState, new_state: CircuitState):
    """Log circuit breaker state changes."""
    logger.warning(f"Circuit state: {old_state.name} -> {new_state.name}")

circuit = CircuitBreaker("my-kernel")
circuit.add_listener(on_state_change)
```

## Performance Considerations

### Hot Path Optimization

The dispatch hot path is optimized for minimal overhead:

1. **Inline Caching**: Static dispatch pre-populates kernel cache
2. **Match Statement**: O(1) operation type dispatch in Python 3.10+
3. **Frozen Dataclasses**: Immutable, hashable, slots-based
4. **Time Functions**: `time.perf_counter_ns()` for nanosecond precision

### Cache Efficiency

- **Two-Level Cache**: Selection cache + rule evaluation cache
- **MVCC**: Concurrent reads without locking
- **TTL Expiration**: Automatic stale entry cleanup
- **Policy Hash**: Cache invalidation on policy change

### Memory Efficiency

- **__slots__**: All core classes use slots
- **Frozenset**: Immutable sets for allowed operations
- **Tuple**: Immutable sequences for rules/conditions
- **Compiled Regex**: Pre-compiled patterns for matching
