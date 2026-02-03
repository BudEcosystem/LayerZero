# Performance Guide

This document covers performance characteristics, latency targets, optimization strategies, and benchmarking guidelines for the LayerZero dispatch system.

## Latency Targets

### Dispatch Overhead by Mode

| Mode | Target (p50) | Target (p99) | Typical Range |
|------|--------------|--------------|---------------|
| STATIC | < 10ns | < 50ns | 5-20ns |
| DYNAMIC (cached) | < 100ns | < 500ns | 50-200ns |
| DYNAMIC (uncached) | < 10us | < 50us | 5-25us |
| CONFIG | < 100ns | < 500ns | 50-300ns |
| HOT_RELOAD | < 100ns | < 500ns | 50-300ns (normal), 1-10ms (reload) |

### Component Latency Breakdown

| Component | Target (p50) | Target (p99) |
|-----------|--------------|--------------|
| Cache lookup | < 50ns | < 100ns |
| Context building | < 100ns | < 500ns |
| Kernel selection | < 5us | < 20us |
| Circuit breaker check | < 20ns | < 50ns |
| Argument mapping | < 100ns | < 500ns |
| Telemetry recording | < 50ns | < 200ns |

### End-to-End Targets

| Metric | Target |
|--------|--------|
| Dispatch overhead (total) | < 1us p99 |
| Selection + dispatch | < 50us p99 |
| Full pipeline (selection -> dispatch -> execute) | Application-dependent |

## Optimization Strategies

### 1. Enable Caching

Caching provides the most significant performance improvement for repeated operations.

```python
from layerzero.dispatch import DispatchConfig, DispatchMode

# Optimal caching configuration
config = DispatchConfig(
    mode=DispatchMode.DYNAMIC,
    enable_cache=True,
    cache_size=50000,  # Size based on unique context count
    cache_ttl_seconds=3600.0,  # Long TTL for stable workloads
)
```

**Cache Sizing Guidelines:**

| Workload Type | Recommended Size | TTL |
|---------------|------------------|-----|
| Inference (fixed shapes) | 1,000 - 5,000 | 3600s+ |
| Training (variable shapes) | 10,000 - 50,000 | 300-600s |
| Mixed workloads | 20,000 - 100,000 | 600-1800s |

### 2. Use Static Dispatch for Known Operations

Static dispatch eliminates all runtime selection overhead.

```python
from layerzero.dispatch import (
    StaticDispatcherBuilder,
    DispatchConfig,
    DispatchMode,
)

# Pre-resolve kernels at initialization time
dispatcher = (
    StaticDispatcherBuilder()
    .with_kernel(flash_attn, operation="attention.causal", default=True)
    .with_kernel(rms_norm, operation="rms_norm", default=True)
    .with_config(enable_telemetry=False)  # Disable for minimal overhead
    .build()
)

# Or use static_kernel_map in config
config = DispatchConfig(
    mode=DispatchMode.STATIC,
    static_kernel_map={
        "attention.causal": "flash_attn_v2_attention",
        "attention.cross": "flash_attn_v2_attention",
        "rms_norm": "liger_rms_norm",
    },
    enable_telemetry=False,
    circuit_breaker_enabled=False,
)
```

### 3. Minimize Telemetry in Hot Paths

Telemetry adds overhead. Disable for latency-critical paths.

```python
# Low-latency configuration
config = DispatchConfig(
    mode=DispatchMode.STATIC,
    enable_telemetry=False,      # Disable metrics collection
    record_timing=False,         # Disable timing measurement
    log_fallbacks=False,         # Disable fallback logging
    circuit_breaker_enabled=False,  # Disable circuit breaker checks
)
```

### 4. Optimize Circuit Breaker Settings

Balance fault tolerance with overhead.

```python
# Low-overhead circuit breaker
config = DispatchConfig(
    circuit_breaker_enabled=True,
    failure_threshold=10,          # Higher threshold = fewer state transitions
    recovery_timeout_seconds=60.0, # Longer cooldown = fewer recovery attempts
)

# Disable entirely for maximum performance (if fallback not needed)
config = DispatchConfig(
    circuit_breaker_enabled=False,
)
```

### 5. Pre-warm Caches

Warm caches before production traffic.

```python
from layerzero.dispatch import get_global_dispatcher

def warm_cache(representative_inputs: list[dict]):
    """Pre-warm dispatch cache with representative inputs."""
    dispatcher = get_global_dispatcher()

    for inputs in representative_inputs:
        for operation in ["attention.causal", "attention.cross", "rms_norm"]:
            try:
                # Execute to populate cache
                dispatcher.dispatch(operation, inputs)
            except Exception:
                pass  # Ignore errors during warmup

# Call during initialization
warm_cache(get_representative_inputs())
```

### 6. Batch Context Building

Reuse SelectionContext for similar operations.

```python
from layerzero.selection import SelectionContext

# Build context once, reuse for multiple dispatches
context = SelectionContext(
    batch_size=batch_size,
    seq_len=seq_len,
    dtype=dtype,
    device=device,
    is_causal=True,
)

# Reuse context for multiple operations
attention_result = dispatcher.dispatch("attention.causal", attention_inputs, context=context)
norm_result = dispatcher.dispatch("rms_norm", norm_inputs, context=context)
```

### 7. Use MVCC Cache for Concurrent Access

The MVCC cache provides lock-free reads for high-concurrency scenarios.

```python
from layerzero.dispatch import create_dynamic_dispatcher

# Enable MVCC cache for concurrent workloads
dispatcher = create_dynamic_dispatcher(
    kernel_registry=kernel_registry,
    backend_registry=backend_registry,
    config=config,
    use_mvcc_cache=True,  # Lock-free concurrent reads
)
```

## Memory Optimization

### Allocation Budget

| Component | Memory Usage |
|-----------|--------------|
| SelectionContext | ~200 bytes per request |
| DispatchResult | ~100 bytes per request |
| ExecutionPlan (cached) | ~300 bytes per entry |
| CompiledRule | ~200 bytes per rule |
| CircuitStats | ~80 bytes per kernel |

### Cache Memory Calculation

```python
# Estimate cache memory usage
cache_size = 50000  # entries
bytes_per_entry = 300  # ExecutionPlan size
overhead_factor = 1.2  # Hash table overhead

estimated_memory_mb = (cache_size * bytes_per_entry * overhead_factor) / (1024 * 1024)
# ~17 MB for 50,000 entries
```

### Reducing Memory Footprint

```python
# Minimal memory configuration
config = DispatchConfig(
    mode=DispatchMode.STATIC,      # No cache needed
    enable_cache=False,            # Or small cache
    cache_size=1000,
    enable_telemetry=False,        # No metrics storage
    circuit_breaker_enabled=False, # No per-kernel state
)
```

## Profiling and Benchmarking

### Built-in Timing

```python
from layerzero.dispatch import dispatch

result = dispatch("attention.causal", inputs)

# Access timing information
timing = result.timing
print(f"Selection: {timing.selection_us:.2f}us")
print(f"Execution: {timing.execution_us:.2f}us")
print(f"Total: {timing.total_us:.2f}us")

# Detailed phase breakdown
print(f"Context build: {timing.context_build_ns}ns")
print(f"Cache lookup: {timing.cache_lookup_ns}ns")
print(f"Kernel invoke: {timing.kernel_invoke_ns}ns")
```

### Benchmark Template

```python
import time
import statistics
from layerzero.dispatch import dispatch, get_global_dispatcher

def benchmark_dispatch(
    operation: str,
    inputs: dict,
    iterations: int = 1000,
    warmup: int = 100,
) -> dict:
    """Benchmark dispatch latency."""
    dispatcher = get_global_dispatcher()

    # Warmup
    for _ in range(warmup):
        dispatcher.dispatch(operation, inputs)

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        dispatcher.dispatch(operation, inputs)
        end = time.perf_counter_ns()
        latencies.append(end - start)

    latencies.sort()
    return {
        "p50_ns": latencies[len(latencies) // 2],
        "p99_ns": latencies[int(len(latencies) * 0.99)],
        "p999_ns": latencies[int(len(latencies) * 0.999)],
        "mean_ns": statistics.mean(latencies),
        "std_ns": statistics.stdev(latencies),
        "min_ns": min(latencies),
        "max_ns": max(latencies),
    }

# Run benchmark
results = benchmark_dispatch("attention.causal", inputs, iterations=10000)
print(f"p50: {results['p50_ns'] / 1000:.2f}us")
print(f"p99: {results['p99_ns'] / 1000:.2f}us")
print(f"p99.9: {results['p999_ns'] / 1000:.2f}us")
```

### Cache Hit Rate Monitoring

```python
from layerzero.dispatch import get_global_dispatcher

dispatcher = get_global_dispatcher()

# Access telemetry
telemetry = dispatcher.get_telemetry()

# Cache statistics
cache_stats = telemetry.cache_stats
hit_rate = cache_stats.hits / (cache_stats.hits + cache_stats.misses)
print(f"Cache hit rate: {hit_rate * 100:.1f}%")
print(f"Cache size: {cache_stats.size} / {cache_stats.capacity}")
```

### Flame Graph Integration

```python
import cProfile
import pstats

def profile_dispatch():
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(10000):
        dispatch("attention.causal", inputs)

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)

# Or use py-spy for flame graphs
# py-spy record -o profile.svg -- python benchmark.py
```

## Benchmark Results Format

When reporting benchmark results, include:

### Hardware Specification

```yaml
hardware:
  cpu: "AMD EPYC 7763 64-Core"
  cpu_cores: 64
  ram: "512GB DDR4-3200"
  gpu: "NVIDIA A100 80GB"
  gpu_count: 8
  nvlink: true

software:
  os: "Ubuntu 22.04"
  kernel: "5.15.0"
  python: "3.11.4"
  pytorch: "2.1.0"
  cuda: "12.1"
```

### Benchmark Configuration

```yaml
config:
  dispatch_mode: DYNAMIC
  cache_enabled: true
  cache_size: 50000
  circuit_breaker: true
  telemetry: false

workload:
  operation: "attention.causal"
  batch_sizes: [1, 8, 32, 128]
  seq_lengths: [512, 2048, 8192]
  dtype: "float16"
  iterations: 10000
  warmup: 1000
```

### Results Table

```markdown
| Operation | Batch | Seq Len | p50 (us) | p99 (us) | p99.9 (us) | Cache Hit |
|-----------|-------|---------|----------|----------|------------|-----------|
| attention.causal | 1 | 512 | 0.12 | 0.45 | 1.2 | 99.8% |
| attention.causal | 8 | 512 | 0.15 | 0.52 | 1.5 | 99.5% |
| attention.causal | 32 | 2048 | 0.18 | 0.61 | 2.1 | 98.2% |
| attention.causal | 128 | 8192 | 0.25 | 0.89 | 3.5 | 95.1% |
```

## Common Performance Issues

### Issue: High Cache Miss Rate

**Symptoms:** p99 latency much higher than p50

**Diagnosis:**
```python
telemetry = dispatcher.get_telemetry()
hit_rate = telemetry.cache_stats.hits / (telemetry.cache_stats.hits + telemetry.cache_stats.misses)
if hit_rate < 0.9:
    print(f"Low cache hit rate: {hit_rate:.1%}")
```

**Solutions:**
1. Increase cache size
2. Extend cache TTL
3. Reduce context variability
4. Use static dispatch for fixed patterns

### Issue: Circuit Breaker Thrashing

**Symptoms:** Inconsistent latency, frequent fallbacks

**Diagnosis:**
```python
circuit_stats = dispatcher.get_circuit_stats("flash_attn_v2_attention")
print(f"State transitions: {circuit_stats.state_transitions}")
print(f"Current state: {circuit_stats.state}")
```

**Solutions:**
1. Increase failure threshold
2. Extend recovery timeout
3. Fix underlying kernel issues
4. Disable circuit breaker for stable kernels

### Issue: Memory Pressure

**Symptoms:** Increasing memory usage, GC pauses

**Diagnosis:**
```python
import tracemalloc

tracemalloc.start()
# ... dispatch operations ...
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f}MB")
print(f"Peak: {peak / 1024 / 1024:.1f}MB")
```

**Solutions:**
1. Reduce cache size
2. Enable cache TTL eviction
3. Disable telemetry storage
4. Use static dispatch

### Issue: Hot-Reload Latency Spikes

**Symptoms:** Periodic latency spikes during reload

**Diagnosis:**
```python
reload_stats = dispatcher.stats
print(f"Reload time: {reload_stats.last_reload_time_ms}ms")
print(f"Reload count: {reload_stats.total_reloads}")
```

**Solutions:**
1. Increase watch interval
2. Optimize config file size
3. Use grace period for in-flight requests
4. Schedule reloads during low-traffic periods

## Performance Checklist

### Production Deployment

- [ ] Enable caching with appropriate size
- [ ] Set cache TTL based on workload stability
- [ ] Use static dispatch for fixed operations
- [ ] Disable unnecessary telemetry
- [ ] Pre-warm caches before traffic
- [ ] Configure circuit breaker thresholds
- [ ] Monitor cache hit rates
- [ ] Set up latency alerts

### Latency-Critical Applications

- [ ] Use STATIC dispatch mode
- [ ] Disable all telemetry
- [ ] Disable circuit breaker
- [ ] Pre-resolve all kernels
- [ ] Pin threads to CPU cores
- [ ] Minimize context building
- [ ] Avoid dynamic allocation in hot path

### High-Throughput Applications

- [ ] Use MVCC cache for concurrent reads
- [ ] Size cache for expected unique contexts
- [ ] Enable aggressive caching (long TTL)
- [ ] Batch similar operations
- [ ] Monitor memory usage
- [ ] Scale cache with traffic
