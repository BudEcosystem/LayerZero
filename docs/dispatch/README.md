# LayerZero Kernel Dispatch System

The LayerZero kernel dispatch system provides a state-of-the-art, near-zero overhead kernel routing layer that connects the SelectionEngine to actual kernel execution. It supports multiple dispatch modes optimized for different use cases, from production deployments requiring zero overhead to development environments needing hot-reload capabilities.

## Overview

The dispatch system bridges the gap between kernel selection and execution by:

- Selecting the appropriate kernel based on operation type, context, and configuration
- Executing kernels with proper argument mapping and tensor transformations
- Providing fault tolerance through circuit breakers and fallback chains
- Enabling zero-downtime configuration updates via hot-reload
- Offering comprehensive telemetry and monitoring

## Key Features

| Feature | Description |
|---------|-------------|
| **Multiple Dispatch Modes** | Static, Dynamic, Hot-Reload, and Config-Driven |
| **Near-Zero Overhead** | ~100-500ns for dynamic dispatch, zero for static |
| **Circuit Breaker** | Automatic fault isolation and recovery |
| **Fallback Chains** | Graceful degradation when kernels fail |
| **Hot-Reload** | Zero-downtime configuration updates |
| **YAML Configuration** | Ops-controlled kernel deployment |
| **Thread-Safe** | Designed for concurrent access |

## Quick Start

### Basic Dynamic Dispatch

```python
from layerzero.dispatch import dispatch, DispatchMode

# Simple dispatch using the global orchestrator
result = dispatch(
    operation="attention.causal",
    inputs={"query": q, "key": k, "value": v},
    is_causal=True,
)

# Access the output tensor
output = result.output

# Check timing information
print(f"Selection time: {result.timing.selection_us:.2f}us")
print(f"Execution time: {result.timing.execution_us:.2f}us")
```

### Creating a Custom Dispatcher

```python
from layerzero.dispatch import (
    create_orchestrator,
    DispatchConfig,
    DispatchMode,
)
from layerzero.registry import get_global_kernel_registry, get_global_backend_registry

# Create a custom orchestrator
orchestrator = create_orchestrator(
    config=DispatchConfig(
        mode=DispatchMode.DYNAMIC,
        enable_cache=True,
        cache_size=10000,
        circuit_breaker_enabled=True,
        failure_threshold=5,
    ),
    kernel_registry=get_global_kernel_registry(),
    backend_registry=get_global_backend_registry(),
)

# Use the orchestrator
result = orchestrator.dispatch(
    "attention.causal",
    {"query": q, "key": k, "value": v},
    is_causal=True,
)
```

### Using Static Dispatch for Production

```python
from layerzero.dispatch import (
    StaticDispatcherBuilder,
    DispatchMode,
)
from layerzero.registry import get_global_kernel_registry

# Build a static dispatcher with predetermined kernels
registry = get_global_kernel_registry()
flash_attn = registry.get("flash_attn_v2_attention")
sdpa = registry.get("torch_sdpa_attention")

dispatcher = (
    StaticDispatcherBuilder()
    .with_kernel(flash_attn, operation="attention.causal", default=True)
    .with_kernel(sdpa, operation="attention.causal")
    .with_config(enable_telemetry=True)
    .build()
)

# Static dispatch with zero overhead
result = dispatcher.dispatch("attention.causal", inputs)
```

### Config-Driven Dispatch

```python
from layerzero.dispatch import create_config_dispatcher

# Load from YAML configuration
dispatcher = create_config_dispatcher(
    config_path="config/kernels.yaml",
    kernel_registry=kernel_registry,
)

# Dispatch uses rules from configuration
result = dispatcher.dispatch("attention.causal", inputs)
```

## Dispatch Modes Comparison

| Mode | Overhead | Flexibility | Use Case |
|------|----------|-------------|----------|
| **STATIC** | Zero | Low | Production with known kernels |
| **DYNAMIC** | ~100-500ns | High | General production use |
| **HOT_RELOAD** | ~1-10ms reload | Very High | Development, A/B testing |
| **CONFIG** | ~100ns lookup | Medium | Ops-controlled deployment |
| **AUTO** | Variable | Adaptive | Automatic mode selection |

## Documentation

- [Architecture](architecture.md) - System architecture and data flow
- [Dispatch Modes](dispatch_modes.md) - Detailed explanation of each mode
- [Configuration](configuration.md) - Configuration reference and YAML schema
- [API Reference](api_reference.md) - Complete API documentation
- [Performance](performance.md) - Performance tuning guide
- [Examples](examples.md) - Code examples for common use cases

## Module Structure

```
layerzero/dispatch/
    __init__.py          # Public API exports
    types.py             # Core types and exceptions
    protocols.py         # Abstract interfaces
    executor.py          # Kernel execution layer
    static.py            # Static dispatch implementation
    dynamic.py           # Dynamic dispatch implementation
    hot_reload.py        # Hot-reload dispatch implementation
    config_dispatch.py   # Config-driven dispatch implementation
    orchestrator.py      # Unified dispatch orchestrator
    circuit_breaker.py   # Circuit breaker for fault tolerance
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Optional: `watchdog` for file watching in hot-reload mode
- Optional: `pyyaml` for YAML configuration support

## Thread Safety

All dispatch components are designed for thread-safe concurrent access:

- Dispatchers use immutable data structures where possible
- Circuit breakers use thread-safe atomic operations
- Caches use fine-grained locking or MVCC
- Configuration updates use atomic swap patterns

## Performance Targets

| Metric | Target |
|--------|--------|
| Dynamic dispatch overhead | < 500ns p99 |
| Cache lookup | < 100ns p99 |
| Config rule evaluation | < 1us p99 |
| Hot-reload time | < 50ms |

## See Also

- [SelectionEngine](../selection/) - Kernel selection based on context
- [KernelRegistry](../registry/) - Kernel registration and lookup
- [PolicyEngine](../policy/) - Policy-based kernel filtering and boosting
