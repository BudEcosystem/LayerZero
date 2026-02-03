# LayerZero

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.11%2B-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-Apache%202.0-orange.svg" alt="License">
  <img src="https://img.shields.io/badge/status-alpha-yellow.svg" alt="Status">
</p>

**LayerZero** is a production-grade kernel orchestration and selection framework for PyTorch LLM inference. It provides a unified API for automatic kernel selection across multiple GPU-optimized libraries (FlashAttention, FlashInfer, xFormers, Liger, etc.), solving the critical problem of kernel fragmentation in modern ML inference pipelines.

## Table of Contents

- [Why LayerZero?](#why-layerzero)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Backends](#supported-backends)
- [Dispatch Modes](#dispatch-modes)
- [Policy System](#policy-system)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Why LayerZero?

Modern ML inference relies on optimized kernels from fragmented libraries, but each serving framework independently implements kernel selection logic, causing:

- **Duplicated Effort**: Every framework reimplements the same selection logic
- **Inconsistent Behavior**: Different frameworks make different choices for identical inputs
- **Integration Complexity**: Each new kernel library requires custom integration
- **Debugging Difficulty**: Hard to understand why a specific kernel was selected

LayerZero solves these problems by providing:

```python
import layerzero as lz

# One unified API - selection, dispatch, and adaptation happen automatically
output = lz.attention(q, k, v, is_causal=True)

# Full explainability - understand every selection decision
report = lz.explain(q, k, v, operation="attention.causal")
print(report.selected_kernel)       # "flashinfer.decode"
print(report.rejection_reasons)     # Why other kernels were filtered
```

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Unified API** | Single interface for all kernel operations across backends |
| **Automatic Selection** | Hardware-aware kernel selection using device capabilities, dtype, and shapes |
| **Multi-Backend Support** | FlashAttention, FlashInfer, xFormers, Liger, Torch SDPA, and more |
| **Policy Control** | YAML-based rules for production deployments (lock, allow, deny, boost) |
| **Graceful Degradation** | Always falls back to torch SDPA - never fails silently |
| **Full Explainability** | Every selection decision is traceable and explainable |

### Performance Features

| Feature | Description |
|---------|-------------|
| **Zero Overhead Static Dispatch** | Compile-time kernel selection with 0ns runtime overhead |
| **Fast Dynamic Dispatch** | Runtime selection with ~100-500ns overhead |
| **MVCC Sharded Cache** | 256-shard cache with O(1) invalidation for high throughput |
| **Performance Database** | SQLite-based persistent perf tracking for informed decisions |
| **JIT Warmup** | Prevent first-call latency spikes with bucket-aware warmup |

### Production Features

| Feature | Description |
|---------|-------------|
| **Circuit Breakers** | Automatic failure detection and recovery |
| **Hot-Reload Config** | Change policies without restart for A/B testing |
| **CUDA Graph Safety** | Automatic validation and whitelisting |
| **Telemetry & Metrics** | Prometheus and OpenTelemetry exporters |
| **Real-Time Audio Support** | Pre-allocated ring buffers and aligned allocators |

### Operations Supported

- **Attention**: Causal, full, sliding window, cross, prefill, decode
- **Normalization**: RMS Norm, Layer Norm, Group Norm
- **Position Encodings**: RoPE (standard/interleaved), ALiBi
- **Activations**: SwiGLU, GELU, SiLU
- **Sampling**: Top-k, Top-p, Greedy
- **Quantization**: INT8, FP8, GPTQ, AWQ (format support)

## Installation

### From PyPI (Coming Soon)

```bash
pip install layerzero
```

### From Source

```bash
git clone https://github.com/BudEcosystem/LayerZero.git
cd LayerZero
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### Backend Installation

LayerZero has **zero required dependencies** - all backends are optional plugins:

```bash
# FlashAttention (NVIDIA GPUs, SM >= 80)
pip install flash-attn

# FlashInfer (Paged attention, recommended)
pip install flashinfer

# xFormers (Memory-efficient attention)
pip install xformers

# Liger Kernel (Fused Triton kernels)
pip install liger-kernel

# CPU Optimization (Intel)
pip install intel-extension-for-pytorch
```

## Quick Start

### Basic Usage

```python
import torch
import layerzero as lz

# Create input tensors
batch, seq_len, num_heads, head_dim = 2, 1024, 32, 128
q = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
k = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
v = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)

# Automatic kernel selection and execution
output = lz.attention(q, k, v, is_causal=True)

# Check which kernel was selected
print(lz.which("attention.causal", q, k, v))  # e.g., "flashinfer.prefill"
```

### Explainability

```python
# Get detailed selection report
report = lz.explain(q, k, v, operation="attention.causal")

print(f"Selected: {report.selected_kernel}")
print(f"Score: {report.score}")
print(f"Cached: {report.cached}")

# See why other kernels were rejected
for kernel_id, reasons in report.rejection_reasons.items():
    print(f"  {kernel_id}: {[r.name for r in reasons]}")
```

### Policy Control

```python
# Lock a specific kernel (deterministic selection)
lz.lock("attention.causal", "flashinfer.decode")

# Set backend preferences
lz.prefer("flashinfer")

# Disable slow fallbacks
lz.disabled("torch_sdpa")

# Load YAML policy file
lz.configure("policy.yaml")
```

### Warmup (Prevent Latency Spikes)

```python
# Warmup specific shapes before production
lz.warmup(
    operation="attention.causal",
    shapes=[
        (1, 512, 32, 128),   # Small batch
        (8, 2048, 32, 128),  # Large batch
    ]
)
```

## Supported Backends

### Attention Backends

| Backend | GPU | Features | Min SM |
|---------|-----|----------|--------|
| **FlashAttention v2/v3/v4** | NVIDIA | IO-aware, fast | 80 (A100) |
| **FlashInfer** | NVIDIA | Paged KV-cache, GQA | 80 (A100) |
| **xFormers** | NVIDIA | Sparse patterns | 70 |
| **Torch SDPA** | All | Universal fallback | Any |

### Normalization Backends

| Backend | Features |
|---------|----------|
| **Liger Kernel** | Fused Triton, numerically stable |
| **APEX** | NVIDIA optimized |
| **Torch** | Universal fallback |

### CPU Backends

| Backend | Platform |
|---------|----------|
| **oneDNN** | Intel CPUs |
| **ZenDNN** | AMD Zen CPUs |
| **IPEX** | Intel GPUs |

### Tokenization Backends

| Backend | Features |
|---------|----------|
| **HF Tokenizers** | Rust-based, fast |
| **SentencePiece** | BPE/Unigram |
| **TikToken** | OpenAI models |

## Dispatch Modes

LayerZero supports multiple dispatch modes for different use cases:

### Static Dispatch (Zero Overhead)

```python
from layerzero.dispatch import StaticDispatcherBuilder

# Build static dispatcher at initialization
dispatcher = (
    StaticDispatcherBuilder()
    .with_kernel(flash_attn_spec, operation="attention.causal", default=True)
    .with_kernel(sdpa_spec, operation="attention.causal")
    .build()
)

# Zero overhead at runtime - kernel selected at build time
result = dispatcher.dispatch("attention.causal", inputs)
```

### Dynamic Dispatch (~100-500ns)

```python
from layerzero.dispatch import create_dynamic_dispatcher

# Runtime selection based on context
dispatcher = create_dynamic_dispatcher(
    kernel_registry=registry,
    backend_registry=backends,
)

# Selection happens at runtime with caching
result = dispatcher.dispatch("attention.causal", inputs, context=ctx)
```

### Hot-Reload Dispatch (Development)

```python
from layerzero.dispatch import create_hot_reload_dispatcher

# Config file is watched for changes
dispatcher = create_hot_reload_dispatcher(
    config_path="dispatch_config.yaml",
    watch_interval_seconds=1.0,
)

# Changes to config file take effect without restart
```

### Config-Driven Dispatch (Production)

```yaml
# dispatch_config.yaml
version: "1.0"
rules:
  - operation: "attention.*"
    conditions:
      - field: "batch_size"
        operator: ">"
        value: 32
    kernel: "flashinfer.prefill"

  - operation: "attention.*"
    kernel: "flash_attn"  # Default for smaller batches
```

## Policy System

LayerZero's policy system provides fine-grained control over kernel selection:

### Four-Tier Rules

```yaml
# policy.yaml
version: "1.0"

# Tier 1: Lock Rules (Highest Priority - Force specific kernel)
locks:
  - operation: "attention.causal"
    backend: "flashinfer"
    condition: "sm >= 80"

# Tier 2: Allow Rules (Whitelist)
allows:
  - operation: "attention.*"
    backends: ["flashinfer", "flash_attn", "xformers"]

# Tier 3: Deny Rules (Blacklist)
denies:
  - operation: "attention.*"
    backends: ["torch_sdpa"]
    condition: "batch > 32"

# Tier 4: Boost Rules (Priority Modifiers)
boosts:
  - operation: "attention.*"
    backend: "flashinfer"
    boost_factor: 1.5
    condition: "seq_len > 1024"
```

### Programmatic Control

```python
import layerzero as lz

# Lock kernel for deterministic behavior
lz.lock("attention.causal", "flashinfer.decode")

# Unlock to restore automatic selection
lz.unlock("attention.causal")

# Set preferences (soft constraint)
lz.prefer("flashinfer", priority=100)

# Disable backends
lz.disabled("torch_sdpa")

# Load policy from file
lz.configure("policy.yaml")
```

## API Reference

### Operations Module

```python
import layerzero as lz

# Attention
output = lz.attention(q, k, v, is_causal=True, dropout_p=0.0)
output = lz.paged_attention(q, k_cache, v_cache, page_table)

# Normalization
output = lz.rms_norm(x, weight, eps=1e-6)
output = lz.layer_norm(x, weight, bias, eps=1e-5)

# Position Encodings
output = lz.rope(x, cos, sin, interleaved=False)

# Sampling
tokens = lz.sample_topk(logits, k=50, temperature=1.0)
tokens = lz.sample_topp(logits, p=0.9, temperature=1.0)

# Tokenization
tokens = lz.tokenize(text, tokenizer_id="gpt2")
text = lz.detokenize(tokens, tokenizer_id="gpt2")
```

### Inspection Module

```python
# Get selected kernel without executing
kernel = lz.select(operation="attention.causal", context=ctx)

# Detailed explanation
report = lz.explain(q, k, v, operation="attention.causal")

# Which kernel will be used
kernel_id = lz.which("attention.causal", q, k, v)

# List all available kernels
kernels = lz.list_kernels(operation="attention.*")

# Validate compatibility
is_valid, reasons = lz.validate(kernel_id, context=ctx)
```

### System Module

```python
# Health diagnostics
lz.doctor()

# Pre-deployment validation
ready, issues = lz.readiness_check()

# Offline plan compilation
plan = lz.compile(operation="attention.causal", shapes=shapes)

# Dry run (test selection without execution)
result = lz.dry_run(operation="attention.causal", context=ctx)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Application                            │
│                   import layerzero as lz                            │
│                   lz.attention(q, k, v)                             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         API Layer                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │Operations│  │  Config  │  │Inspection│  │  System  │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Dispatch Orchestrator                          │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌────────────┐            │
│  │ Static  │  │ Dynamic │  │Hot-Reload│  │Config-Driven│           │
│  └─────────┘  └─────────┘  └──────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Selection Engine                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Policy  │→ │ Filter  │→ │  Score  │→ │ Select  │→ │  Cache  │  │
│  │  Check  │  │  Phase  │  │  Phase  │  │  Best   │  │ (MVCC)  │  │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Backend Adapters                               │
│  ┌───────────┐  ┌───────────┐  ┌─────────┐  ┌───────┐  ┌────────┐ │
│  │FlashAttn  │  │ FlashInfer│  │ xFormers│  │ Liger │  │Torch   │ │
│  │  v2/v3/v4 │  │Prefill/Dec│  │         │  │       │  │SDPA    │ │
│  └───────────┘  └───────────┘  └─────────┘  └───────┘  └────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       GPU Kernels                                    │
│              CUDA / ROCm / CPU / Intel XPU                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Selection Pipeline Flow

```
SelectionContext (device, dtype, shapes, masks)
         │
         ▼
┌─────────────────────────────────────────┐
│ 1. Policy Check (LOCK rules)            │
│    └─ Lock matched? → Return immediately │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 2. Cache Check (MVCC, 256 shards)       │
│    └─ Cache hit? → Return cached plan   │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 3. Filter Phase (Hard Constraints)      │
│    ├─ Platform matching                 │
│    ├─ Hardware (SM version)             │
│    ├─ Dtype support                     │
│    ├─ Shape constraints                 │
│    └─ Layout compatibility              │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 4. Apply Policy (ALLOW/DENY)            │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 5. Score Phase                          │
│    ├─ Base kernel priority              │
│    ├─ Policy BOOST rules                │
│    ├─ Historical performance            │
│    └─ Transform cost                    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 6. Select Best & Cache Result           │
└─────────────────────────────────────────┘
         │
         ▼
    ExecutionPlan (kernel + transforms)
```

## Testing

LayerZero has comprehensive test coverage:

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest tests/ --cov=layerzero --cov-report=html

# Run specific test categories
pytest -m "not gpu"      # Skip GPU tests
pytest -m "not slow"     # Skip slow tests
pytest -m "not stress"   # Skip stress tests
```

### Test Categories

| Category | Description | Count |
|----------|-------------|-------|
| Unit | Component isolation tests | ~2,000+ |
| Integration | Multi-component tests | ~200+ |
| Property | Hypothesis-based fuzzing | ~100+ |
| Fuzz | libFuzzer input validation | ~50+ |
| Stress | Concurrent load tests | ~50+ |
| Correctness | Numerical accuracy | ~30+ |

## Deployment

### Basic Deployment

```python
import layerzero as lz

# Pre-deployment validation
ready, issues = lz.readiness_check()
if not ready:
    for issue in issues:
        print(f"Issue: {issue}")
    exit(1)

# Warmup common shapes
lz.warmup(
    operation="attention.causal",
    shapes=[
        (1, 512, 32, 128),
        (1, 1024, 32, 128),
        (1, 2048, 32, 128),
        (8, 512, 32, 128),
    ]
)

# Load production policy
lz.configure("production_policy.yaml")

# Start serving...
```

### Production Policy Example

```yaml
# production_policy.yaml
version: "1.0"

# Force FlashInfer for large batches on A100+
locks:
  - operation: "attention.*"
    backend: "flashinfer"
    condition: "sm >= 80 and batch_size >= 8"

# Allow only proven backends
allows:
  - operation: "attention.*"
    backends: ["flashinfer", "flash_attn"]

# Deny slow fallbacks in production
denies:
  - operation: "attention.*"
    backends: ["torch_sdpa"]
    condition: "batch_size > 1"

# Boost FlashInfer for long sequences
boosts:
  - operation: "attention.*"
    backend: "flashinfer"
    boost_factor: 2.0
    condition: "seq_len > 2048"
```

### Monitoring

```python
from layerzero.telemetry import get_metrics_exporter

# Prometheus export
exporter = get_metrics_exporter("prometheus")
metrics = exporter.export()

# OpenTelemetry export
from layerzero.telemetry import OTelExporter
otel = OTelExporter(endpoint="http://collector:4317")
otel.export_traces()
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install CUDA runtime (for GPU deployments)
# ... cuda installation ...

WORKDIR /app

# Install LayerZero and backends
COPY requirements.txt .
RUN pip install layerzero flash-attn flashinfer

# Copy application
COPY . .

# Pre-warmup at build time (optional)
RUN python -c "import layerzero as lz; lz.warmup(...)"

CMD ["python", "serve.py"]
```

## Performance

### Latency Targets

| Component | Target | Notes |
|-----------|--------|-------|
| Selection (cache hit) | < 100ns | MVCC sharded cache |
| Selection (cache miss) | < 500ns | Full pipeline |
| Static dispatch | 0ns | Compile-time resolution |
| Circuit breaker check | < 50ns | Lock-free read |

### Benchmarks

```bash
# Run dispatch benchmarks
python benchmarks/dispatch/run_benchmarks.py

# Run cache benchmarks
python benchmarks/dispatch/bench_cache_performance.py
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest tests/`)
5. Run linting (`ruff check src/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/BudEcosystem/LayerZero.git
cd LayerZero
pip install -e ".[dev]"
pre-commit install
```

### Code Style

- Python 3.11+ with type hints
- Ruff for linting and formatting
- MyPy for static type checking
- Google-style docstrings

## Roadmap

- [x] Core selection engine
- [x] Policy system (lock/allow/deny/boost)
- [x] MVCC sharded cache
- [x] Circuit breaker pattern
- [x] **Dispatch system (complete)**
  - [x] Static dispatch (zero overhead)
  - [x] Dynamic dispatch (~100-500ns)
  - [x] Hot-reload dispatch (config watching)
  - [x] Config-driven dispatch (YAML rules)
  - [x] Dispatch orchestrator
  - [x] Real-time audio buffers
- [x] Backend adapters (FlashAttention, FlashInfer, xFormers)
- [ ] Speculative decoding integration
- [ ] Automatic performance tuning
- [ ] Distributed inference support
- [ ] Web dashboard for monitoring

## License

LayerZero is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [FlashAttention](https://github.com/Dao-AILab/flash-attention) by Tri Dao
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - MLSys 2025 Best Paper
- [xFormers](https://github.com/facebookresearch/xformers) by Meta
- [Liger Kernel](https://github.com/linkedin/liger-kernel) by LinkedIn
- The PyTorch team for SDPA

---

<p align="center">
  Built with care by <a href="https://bud.ai">Bud Ecosystem</a>
</p>
