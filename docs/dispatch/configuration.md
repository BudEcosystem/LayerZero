# Configuration Reference

This document provides a complete reference for configuring the LayerZero dispatch system.

## DispatchConfig Dataclass

The `DispatchConfig` dataclass controls all aspects of dispatch behavior.

```python
@dataclass(slots=True)
class DispatchConfig:
    """Configuration for the dispatch system."""

    # Dispatch mode
    mode: DispatchMode = DispatchMode.DYNAMIC

    # Cache settings
    enable_cache: bool = True
    cache_size: int = 10000
    cache_ttl_seconds: float = 3600.0

    # Fallback settings
    enable_fallback: bool = True
    max_fallback_attempts: int = 3
    fallback_timeout_ms: float = 100.0

    # Hot-reload settings
    config_path: str | None = None
    watch_interval_seconds: float = 1.0
    validate_on_reload: bool = True

    # Execution settings
    enable_transforms: bool = True
    enable_cuda_graphs: bool = False
    sync_after_execution: bool = False

    # Monitoring settings
    enable_telemetry: bool = True
    record_timing: bool = True
    log_fallbacks: bool = True

    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0

    # Static dispatch settings
    static_kernel_map: dict[str, str] = field(default_factory=dict)
```

### Field Reference

#### Dispatch Mode

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `DispatchMode` | `DYNAMIC` | Primary dispatch mode |

Valid modes: `STATIC`, `DYNAMIC`, `HOT_RELOAD`, `CONFIG`, `AUTO`

#### Cache Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_cache` | `bool` | `True` | Enable selection caching |
| `cache_size` | `int` | `10000` | Maximum cached entries |
| `cache_ttl_seconds` | `float` | `3600.0` | Cache entry time-to-live |

#### Fallback Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_fallback` | `bool` | `True` | Enable kernel fallback chain |
| `max_fallback_attempts` | `int` | `3` | Maximum fallback kernels to try |
| `fallback_timeout_ms` | `float` | `100.0` | Timeout for fallback attempts |

#### Hot-Reload Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `config_path` | `str \| None` | `None` | Path to YAML/JSON config file |
| `watch_interval_seconds` | `float` | `1.0` | File watch polling interval |
| `validate_on_reload` | `bool` | `True` | Validate config before applying |

#### Execution Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_transforms` | `bool` | `True` | Enable tensor transformations |
| `enable_cuda_graphs` | `bool` | `False` | Enable CUDA graph capture |
| `sync_after_execution` | `bool` | `False` | Synchronize after kernel execution |

#### Monitoring Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_telemetry` | `bool` | `True` | Collect telemetry data |
| `record_timing` | `bool` | `True` | Record timing metrics |
| `log_fallbacks` | `bool` | `True` | Log fallback events |

#### Circuit Breaker Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `circuit_breaker_enabled` | `bool` | `True` | Enable circuit breaker |
| `failure_threshold` | `int` | `5` | Failures before circuit opens |
| `recovery_timeout_seconds` | `float` | `30.0` | Cooldown before half-open |

#### Static Dispatch Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `static_kernel_map` | `dict[str, str]` | `{}` | Operation to kernel_id mapping |

### Example Configuration

```python
from layerzero.dispatch import DispatchConfig, DispatchMode

# Production configuration
production_config = DispatchConfig(
    mode=DispatchMode.DYNAMIC,
    enable_cache=True,
    cache_size=50000,
    cache_ttl_seconds=7200.0,
    enable_fallback=True,
    max_fallback_attempts=3,
    circuit_breaker_enabled=True,
    failure_threshold=5,
    recovery_timeout_seconds=60.0,
    enable_telemetry=True,
    record_timing=True,
)

# Development configuration
dev_config = DispatchConfig(
    mode=DispatchMode.HOT_RELOAD,
    config_path="config/dev_kernels.yaml",
    watch_interval_seconds=0.5,
    validate_on_reload=True,
    enable_cache=False,  # Disable for immediate changes
    enable_telemetry=True,
    log_fallbacks=True,
)

# Low-latency configuration
low_latency_config = DispatchConfig(
    mode=DispatchMode.STATIC,
    static_kernel_map={
        "attention.causal": "flash_attn_v2_attention",
        "rms_norm": "liger_rms_norm",
    },
    enable_cache=True,
    circuit_breaker_enabled=False,  # Avoid overhead
    enable_telemetry=False,
)
```

## YAML Configuration Schema

For config-driven and hot-reload dispatch modes, configuration is specified in YAML format.

### Schema Version

```yaml
version: "1.0"  # Required, must be "1.0"
```

### Defaults Section

```yaml
defaults:
  fallback_policy: torch_sdpa      # Default fallback kernel
  default_priority: 50             # Default rule priority (0-1000)
```

### Dispatch Rules

Rules are evaluated in priority order (highest first):

```yaml
dispatch_rules:
  - operation: "attention.*"       # Glob pattern for operation matching
    conditions:                    # All conditions must match
      batch_size_gte: 8
      dtype: [float16, bfloat16]   # List = implicit IN operator
    kernel: flash_attention_v2     # Target kernel ID
    priority: 100                  # Rule priority (0-1000)

  - operation: "attention.causal"
    conditions:
      seq_len_gt: 8192
      is_causal: true
    kernel: xformers_memory_efficient
    priority: 90
```

### Condition Fields

| Field | Type | Description |
|-------|------|-------------|
| `batch_size` | `int` | Batch size |
| `seq_len` | `int` | Sequence length |
| `seq_len_q` | `int` | Query sequence length |
| `seq_len_k` | `int` | Key sequence length |
| `head_dim` | `int` | Attention head dimension |
| `num_heads` | `int` | Number of attention heads |
| `num_kv_heads` | `int` | Number of KV heads (for GQA) |
| `dtype` | `string` | Data type |
| `device` | `string` | Device string |
| `platform` | `string` | Platform (cuda, rocm, cpu) |
| `layout` | `string` | Tensor layout |
| `is_causal` | `bool` | Causal attention mask |
| `enable_gqa` | `bool` | Grouped Query Attention |
| `dropout_p` | `float` | Dropout probability |
| `is_cuda_graph_capturing` | `bool` | In CUDA graph capture |
| `requires_deterministic` | `bool` | Requires deterministic ops |
| `tp_size` | `int` | Tensor parallel size |
| `pp_size` | `int` | Pipeline parallel size |

### Condition Operators

```yaml
conditions:
  # Equality (default)
  dtype: float16                  # dtype == float16
  dtype_eq: float16               # Same as above

  # Inequality
  dtype_ne: float32               # dtype != float32

  # Numeric comparisons
  batch_size_gt: 8                # batch_size > 8
  batch_size_gte: 8               # batch_size >= 8
  batch_size_lt: 256              # batch_size < 256
  batch_size_lte: 256             # batch_size <= 256

  # List membership
  dtype_in: [float16, bfloat16]   # dtype IN list
  dtype_not_in: [float32]         # dtype NOT IN list

  # Pattern matching
  device_match: "cuda:*"          # Glob pattern match
  kernel_regex: "flash.*"         # Regex match
```

### Kernel Locks

Force specific kernel for operations:

```yaml
kernel_locks:
  attention.causal: flash_attention_v2
  attention.cross: flash_attention_v2
  "attention.*": flash_attention_v2      # Pattern support
```

### Kernel Allow/Deny Lists

Filter kernels by pattern:

```yaml
# Deny patterns (kernels matching are excluded)
kernel_denies:
  - "*_experimental"
  - "*_debug"

# Allow patterns (if present, only matching kernels allowed)
kernel_allows:
  - "flash_attn.*"
  - "torch_sdpa"
  - "xformers_*"
```

### Fallback Chains

Define fallback order per operation:

```yaml
fallback_chains:
  attention.causal:
    - flash_attention_v2
    - xformers_memory_efficient
    - torch_sdpa

  attention.cross:
    - flash_attention_v2
    - torch_sdpa

  "rms_norm":
    - liger_rms_norm
    - triton_rms_norm
    - torch_rms_norm
```

### Complete Example

```yaml
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

  # Causal attention with dropout
  - operation: "attention.causal"
    conditions:
      dropout_p_gt: 0.0
      is_causal: true
    kernel: flash_attention_v2
    priority: 85

  # RMS normalization
  - operation: "rms_norm"
    conditions:
      platform: cuda
    kernel: liger_rms_norm
    priority: 100

  # RoPE for CUDA
  - operation: "rope.*"
    conditions:
      platform: cuda
    kernel: liger_rope
    priority: 100

kernel_locks:
  # Always use FlashAttention for prefill
  attention.prefill: flash_attention_v2

kernel_denies:
  # Deny experimental kernels in production
  - "*_experimental"
  - "*_debug"
  - "*_test"

kernel_allows:
  # Only allow these kernels
  - "flash_attn_*"
  - "xformers_*"
  - "torch_sdpa"
  - "liger_*"
  - "triton_*"

fallback_chains:
  attention.causal:
    - flash_attention_v2
    - xformers_memory_efficient
    - torch_sdpa

  attention.cross:
    - flash_attention_v2
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

## CircuitBreakerConfig

Configuration for circuit breaker behavior:

```python
@dataclass(slots=True)
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5        # Failures before opening
    success_threshold: int = 2        # Successes to close from half-open
    cooldown_seconds: float = 30.0    # Time before half-open transition
    half_open_max_calls: int = 3      # Max calls in half-open state
    reset_timeout_seconds: float | None = None  # Hard reset timeout
```

### Example

```python
from layerzero.dispatch import CircuitBreakerConfig, CircuitBreaker

config = CircuitBreakerConfig(
    failure_threshold=3,       # Open after 3 failures
    success_threshold=2,       # Close after 2 successes in half-open
    cooldown_seconds=15.0,     # Wait 15s before half-open
    half_open_max_calls=2,     # Allow 2 test calls in half-open
)

circuit = CircuitBreaker("my-kernel", config)
```

## Environment-Based Configuration

Use environment variables for deployment-specific settings:

```python
import os
from layerzero.dispatch import DispatchConfig, DispatchMode

def create_config():
    mode = os.environ.get("LAYERZERO_DISPATCH_MODE", "DYNAMIC")
    cache_size = int(os.environ.get("LAYERZERO_CACHE_SIZE", "10000"))
    config_path = os.environ.get("LAYERZERO_CONFIG_PATH")

    return DispatchConfig(
        mode=DispatchMode[mode],
        cache_size=cache_size,
        config_path=config_path,
        enable_cache=os.environ.get("LAYERZERO_ENABLE_CACHE", "true").lower() == "true",
        circuit_breaker_enabled=os.environ.get("LAYERZERO_CIRCUIT_BREAKER", "true").lower() == "true",
    )
```

## Validation

Configuration is validated on creation:

```python
# DispatchConfig validation
config = DispatchConfig(
    cache_size=-1,  # Raises ValueError: cache_size must be non-negative
)

# YAML schema validation
from layerzero.dispatch import ConfigSchema

schema = ConfigSchema()
errors = schema.validate(yaml_config)
if errors:
    for error in errors:
        print(f"{error.path}: {error.message}")
```

### Validation Errors

Common validation errors:

```python
# Version errors
SchemaError("version", "Missing required field")
SchemaError("version", "Unsupported version, must be one of ['1.0']", "2.0")

# Rule errors
SchemaError("dispatch_rules[0].operation", "Missing required field")
SchemaError("dispatch_rules[0].kernel", "Must be string", 123)

# Condition errors
SchemaError("dispatch_rules[0].conditions.dtype", "Invalid dtype", "fp16")
SchemaError("dispatch_rules[0].conditions.batch_size", "Must be integer", "8")

# Priority errors
SchemaError("dispatch_rules[0].priority", "Must be between 0 and 1000", 2000)
```

## Configuration Best Practices

### Cache Sizing

```python
# Rule of thumb: cache_size = expected_unique_contexts * 2
# For diverse workloads, larger cache helps
cache_size = 50000

# For inference with fixed shapes, smaller cache is fine
cache_size = 1000
```

### TTL Settings

```python
# Long TTL for stable configurations
cache_ttl_seconds = 3600.0  # 1 hour

# Short TTL for dynamic environments
cache_ttl_seconds = 60.0  # 1 minute
```

### Circuit Breaker Tuning

```python
# Aggressive failure detection (faster isolation)
failure_threshold = 3
recovery_timeout_seconds = 15.0

# Conservative (avoid false positives)
failure_threshold = 10
recovery_timeout_seconds = 60.0
```

### Hot-Reload Safety

```yaml
# Always validate before applying
validate_on_reload: true

# Use longer watch interval to reduce overhead
watch_interval_seconds: 2.0
```
