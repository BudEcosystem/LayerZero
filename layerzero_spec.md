# LayerZero: A Practical Kernel Orchestration System for PyTorch

**Version:** 1.0 Full-Scope Specification  
**Status:** Implementation-Ready

---

## Abstract

Modern ML inference relies on optimized kernels from multiple libraries (FlashAttention, FlashInfer, xFormers, Triton, oneDNN, ZenDNN, and more), yet each serving framework independently implements kernel selection logic, leading to duplicated effort and inconsistent behavior. We present LayerZero, a practical kernel orchestration system that provides: (1) a unified API for kernel invocation across the full inference stack, (2) automatic multi-factor kernel selection, (3) a policy system for production control, and (4) a persistent performance database for tuning. This specification defines the full-scope architecture and a phased implementation plan that can be delivered incrementally without sacrificing correctness or production readiness.

**Scope:** LayerZero targets the full inference pipeline (tokenization, embeddings, positional ops, attention, MLP/GEMM, normalization, sampling/decoding, and quantization) across NVIDIA CUDA, AMD ROCm, Intel/CPU, and Habana HPU. Training support is phased and optional; the system is designed to degrade safely to reference fallbacks when optimized kernels are unavailable.

---

## 1. Introduction

### 1.1 The Problem

Optimized ML kernels are fragmented across libraries:
- **FlashAttention 2/3**: IO-aware attention for NVIDIA GPUs
- **FlashInfer**: Flexible attention with paged KV-cache support
- **xFormers**: Memory-efficient attention with sparse patterns
- **Liger Kernel**: Fused Triton kernels for norms and MLPs
- **Triton**: Custom GPU kernels in Python

Every serving framework (vLLM, SGLang, TensorRT-LLM, HuggingFace TGI) independently implements:
1. Hardware capability detection
2. Kernel availability checking
3. Shape/dtype/feature constraint validation
4. Heuristic kernel selection
5. Fallback logic

This duplication causes bugs, inconsistent behavior, and makes adding new kernels difficult.

### 1.2 Our Solution

LayerZero provides a single library that frameworks call instead of directly calling kernel libraries:

```python
# Instead of framework-specific selection logic:
if has_flash_attn and sm >= 80 and head_dim <= 256:
    output = flash_attn_func(q, k, v, causal=True)
elif has_xformers:
    output = xformers.ops.memory_efficient_attention(q, k, v)
else:
    output = F.scaled_dot_product_attention(q, k, v)

# Frameworks simply call:
import layerzero as lz
output = lz.attention(q, k, v, causal=True)
# LayerZero handles selection, adaptation, and fallback
```

### 1.3 Design Principles

**Principle 1: Explicit API, Not Magic.**  
LayerZero provides explicit functions (`lz.attention()`, `lz.rms_norm()`). We do not attempt to transparently intercept PyTorch operations—this is too fragile and surprising. Frameworks must opt-in by calling LayerZero APIs.

**Principle 2: Correctness Over Performance.**  
Every operation has a PyTorch fallback. If no optimized kernel is valid, LayerZero uses the fallback and logs a warning. Silent incorrectness is never acceptable.

**Principle 3: Traceable Decisions.**  
Users can always ask "why was this kernel selected?" and get a structured answer. No black-box selection.

**Principle 4: Production Control.**  
Operators can lock kernels, set preferences, and disable specific libraries without code changes.

**Principle 5: Full-Scope by Design, Phased by Delivery.**  
LayerZero is designed for the full inference stack and heterogeneous hardware. Delivery is phased, but the architecture must not hardcode MVP-only constraints.

**Principle 6: Backend Isolation.**  
Backends are optional plugins. Import failures or ABI conflicts must not crash LayerZero. Dynamic loading and graceful disablement are mandatory.

**Principle 7: Predictable Latency.**  
JIT backends must be warmed up before production traffic. Latency spikes are not acceptable in serving environments.

**Principle 8: Plan-Aware Selection.**  
Kernel selection must consider layout/dtype transforms and fused ops across adjacent operations, not just per-op speed.

**Principle 9: Graph Safety by Validation.**  
CUDA graph safety must be provable, not assumed. Optional strict validation should be available before production use.

### 1.4 What LayerZero Is NOT

- **Not a serving framework**: No batching, scheduling, or request routing (kernels only)
- **Not a compiler**: Does not replace torch.compile or TensorRT
- **Not a kernel library**: Does not implement kernels, only orchestrates them
- **Not fully transparent**: Requires explicit API calls (by design)

---

## 2. Architecture Overview

### 2.1 System Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER CODE / FRAMEWORKS                             │
│                     (vLLM, SGLang, HuggingFace, Custom)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ lz.attention(q, k, v, causal=True)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LAYERZERO API                                   │
│                                                                              │
│   lz.attention()     lz.rms_norm()     lz.layer_norm()     lz.configure()   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SELECTION ENGINE                                  │
│                                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│   │   Policy    │ → │   Filter    │ → │    Score    │ → │    Cache    │    │
│   │   Check     │   │  (HW/Dtype) │   │  (Priority) │   │   Result    │    │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BACKEND LOADER / PLUGIN MANAGER                          │
│               (dynamic import, entry_points, capabilities)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            KERNEL REGISTRY                                   │
│                                                                              │
│   flash_attn_v3 │ flash_attn_v2 │ flashinfer │ xformers │ torch_sdpa       │
│   liger_rms     │ apex_rms      │ triton_rms │ torch_rms                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KERNEL IMPLEMENTATIONS                               │
│                        (External Libraries)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
lz.attention(q, k, v, causal=True) called
                │
                ▼
┌───────────────────────────────────────┐
│ 1. BUILD CONTEXT                      │
│    • device: cuda:0                   │
│    • dtype: bfloat16                  │
│    • batch: 32, seq: 4096, heads: 32  │
│    • head_dim: 128, causal: true      │
│    • sm_version: 90 (H100)            │
└───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│ 2. CHECK POLICY                       │
│    • Locked kernel? → Use it          │
│    • Forbidden library? → Exclude     │
│    • Preferred library? → Boost score │
└───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│ 3. CHECK CACHE                        │
│    • Cache key = hash(op, context)    │
│    • Hit? → Return cached kernel      │
└───────────────────────────────────────┘
                │ miss
                ▼
┌───────────────────────────────────────┐
│ 3.5 LOAD BACKENDS (LAZY)              │
│    • Dynamic import via plugin loader │
│    • Read capabilities descriptors    │
└───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│ 4. FILTER CANDIDATES                  │
│    • Hardware: sm90 required? ✓/✗     │
│    • Dtype: bf16 supported? ✓/✗       │
│    • Constraints: head_dim ≤ 256? ✓/✗ │
│    • Available: import works? ✓/✗     │
│                                       │
│    Each failure → structured Reason   │
└───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│ 5. SCORE & SELECT                     │
│    • Base priority                    │
│    • Policy preference bonus          │
│    • PerfDB timing data (if present)  │
│    • Select highest score             │
└───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│ 6. CACHE & DISPATCH                   │
│    • Store in selection cache         │
│    • Adapt arguments to kernel API    │
│    • Call kernel                      │
│    • Return result                    │
└───────────────────────────────────────┘
```

---

## 3. Supported Operations (Full Scope)

### 3.1 Operation Scope

LayerZero orchestrates kernels across the full inference stack. Optimized coverage
is backend-dependent, but every operation has a correct fallback.

| Operation | Description | Fallback |
|-----------|-------------|----------|
| `tokenize.encode` | Text → tokens | Native tokenizer library |
| `tokenize.decode` | Tokens → text | Native tokenizer library |
| `embedding.lookup` | Token embeddings | PyTorch reference |
| `posenc.rope` | Rotary positional encoding | PyTorch reference |
| `posenc.alibi` | ALiBi or bias-based positional encoding | PyTorch reference |
| `attention.causal` | Causal self-attention | `F.scaled_dot_product_attention` |
| `attention.full` | Bidirectional attention | `F.scaled_dot_product_attention` |
| `attention.paged` | Paged KV-cache attention | Reference implementation |
| `mlp.linear` | Linear / GEMM | `torch.matmul` / `torch.nn.functional.linear` |
| `mlp.fused` | Fused MLP + activation | PyTorch reference |
| `norm.rms` | RMSNorm | PyTorch reference |
| `norm.layer` | LayerNorm | `F.layer_norm` |
| `sampling.topk` | Top-k sampling | PyTorch reference |
| `sampling.topp` | Top-p sampling | PyTorch reference |
| `sampling.speculative` | Speculative decoding | PyTorch reference |
| `quantize` | Quantize/dequantize, dtype conversion | PyTorch reference |

### 3.2 Operation Semantic Contract

Each operation has a precise semantic definition. For attention:

**Attention Semantic Contract:**
```
Given:
  Q: (batch, seq_q, num_heads, head_dim) or (batch, num_heads, seq_q, head_dim)
  K: (batch, seq_k, num_kv_heads, head_dim) or (batch, num_kv_heads, seq_k, head_dim)
  V: (batch, seq_k, num_kv_heads, head_dim) or (batch, num_kv_heads, seq_k, head_dim)
  scale: float = 1/sqrt(head_dim)
  causal: bool

Compute:
  1. scores = (Q @ K.transpose(-2, -1)) * scale
  2. if causal: scores = scores.masked_fill(causal_mask, -inf)
  3. attn_weights = softmax(scores, dim=-1)
  4. output = attn_weights @ V

Invariants:
  - NaN in input → NaN in output (propagate, don't sanitize)
  - Supports GQA: num_kv_heads may differ from num_heads
  - Layout: both BSHD and BHSD accepted, output matches input layout
  - If attn_mask is provided, is_causal must be False (PyTorch constraint)

Tolerances (vs PyTorch reference):
  - float16: rtol=1e-3, atol=1e-3
  - bfloat16: rtol=1e-2, atol=1e-2
  - float32: rtol=1e-5, atol=1e-5
```

**Tokenization Semantic Contract (Summary):**
- Deterministic for the same `tokenizer_id` + `vocab_hash` + `merges_hash` + `normalizer_id`
- Include `added_tokens_hash` and `special_tokens_hash` in cache keys
- Offsets returned only when `return_offsets=True`
- Any mismatch in tokenizer metadata must invalidate cache entries

### 3.3 Phased Coverage and Constraints

LayerZero’s architecture is full-scope, but optimized coverage is phased:
- Any op may fall back to a reference implementation when no optimized kernel is valid.
- Some advanced features (block-sparse, sliding window, complex attention bias) may be fallback-only in early phases.
- Training/backward support is optional and may be delivered after inference stabilization.

---

## 4. Kernel Specification

### 4.1 KernelSpec Structure

Each registered kernel has a specification:

```python
@dataclass(frozen=True)
class KernelSpec:
    # Identity
    kernel_id: str              # "flash_attn.v3.causal"
    operation: str              # "attention.causal"
    source: str                 # "flash_attn"
    version: str                # "3.0.0"
    
    # Implementation
    impl: Callable              # The actual function
    
    # Hardware Requirements
    platform: str               # "cuda"
    min_sm: int                 # 90 (for SM90+)
    max_sm: Optional[int]       # None (no upper limit)
    
    # Constraints
    supported_dtypes: frozenset # {torch.float16, torch.bfloat16}
    min_head_dim: int           # 32
    max_head_dim: int           # 256
    head_dim_multiple: int      # 8
    max_seq_len: Optional[int]  # 65536 or None
    supports_gqa: bool          # True
    supports_attn_mask: bool    # True/False
    supported_attn_mask_types: frozenset  # {"none", "bool", "float"}
    supports_dropout: bool
    supports_scale: bool
    requires_last_dim_stride1: bool
    requires_layouts: frozenset # {"BSHD", "BHSD", "NHD", ...}
    produces_layout: Optional[str]
    requires_dtype: Optional[torch.dtype]
    produces_dtype: Optional[torch.dtype]
    supports_kv_cache_layouts: frozenset
    supports_kv_cache_dtypes: frozenset

    # Execution Properties
    is_cuda_graph_safe: bool    # True
    deterministic: bool         # False (FA has non-determinism)
    
    # Priority (0-100, higher = preferred)
    priority: int               # 100

    # Fusion and prepack
    fuses_ops: list[str]        # e.g., ["posenc.rope", "attention.causal"]
    requires_packed_weights: bool
    supports_prepack: bool
    transform_cost_hint: int    # rough relative cost for layout/dtype transforms
    
    # Check function for complex constraints
    def check(self, ctx: SelectionContext) -> list[Reason]:
        """Return empty list if valid, else list of failure reasons."""
        ...
```

### 4.2 Built-in Kernel Registry

LayerZero ships with wrappers for a core set of backends across attention,
normalization, tokenization, and CPU acceleration. Additional kernels are
added via plugins and the HF Kernel Hub.

| Kernel ID | Operation | Platform | Dtypes | Priority |
|-----------|-----------|----------|--------|----------|
| `flash_attn.v3.causal` | attention.causal | cuda | fp16, bf16 | 100 |
| `flash_attn.v2.causal` | attention.causal | cuda | fp16, bf16 | 90 |
| `flashinfer.causal` | attention.causal | cuda | fp16, bf16 | 85 |
| `xformers.causal` | attention.causal | cuda | fp16, bf16, fp32 | 75 |
| `torch.sdpa.causal` | attention.causal | cuda/cpu | fp16, bf16, fp32 | 50 |
| `liger.rms_norm` | norm.rms | cuda/rocm | fp16, bf16 | 100 |
| `apex.rms_norm` | norm.rms | cuda | fp16, bf16, fp32 | 90 |
| `torch.rms_norm` | norm.rms | cpu/cuda | any | 10 |
| `tokenizers.encode` | tokenize.encode | cpu | text → tokens | 60 |
| `tiktoken.encode` | tokenize.encode | cpu | text → tokens | 70 |
| `sentencepiece.encode` | tokenize.encode | cpu | text → tokens | 50 |

### 4.3 Custom Kernel Registration

Users can register custom kernels:

```python
import layerzero as lz

@lz.register_kernel(
    operation="attention.causal",
    kernel_id="my_org.custom_attention",
    platform="cuda",
    min_sm=80,
    supported_dtypes=[torch.float16, torch.bfloat16],
    max_head_dim=128,
    priority=95,
)
def my_custom_attention(q, k, v, scale=None, causal=True):
    # Custom implementation
    ...
```

### 4.4 Plugin Registration (Entry Points)

Third-party backends can register automatically via `entry_points`:

```python
# setup.py
setup(
    ...,
    entry_points={
        "layerzero.backends": [
            "my_backend = my_pkg.layerzero_backend:register_backend",
        ]
    },
)
```

The backend loader discovers these plugins at runtime and treats missing
dependencies as backend-unavailable (never fatal).

### 4.5 Capabilities Descriptors (Data-Driven Constraints)

Backends should expose a capabilities descriptor that lists supported ops,
constraints, and dtypes. This allows LayerZero to update constraints without
code changes and reduces hardcoded checks.

Example:
```json
{
  "backend": "flash_attn",
  "version": "2.5.6",
  "ops": {
    "attention.causal": {
      "min_sm": 80,
      "max_head_dim": 256,
      "dtypes": ["float16", "bfloat16"]
    }
  }
}
```

---

## 5. Selection Engine

### 5.1 Selection Context

```python
@dataclass
class SelectionContext:
    # Device info (detected automatically)
    device: torch.device
    sm_version: int              # 90 for SM90
    
    # Tensor properties (from inputs)
    dtype: torch.dtype
    batch_size: int
    seq_len_q: int
    seq_len_k: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    layout: str                  # "BSHD" or "BHSD"
    stride_last_dim: int
    is_last_dim_contiguous: bool
    attn_mask_type: str          # "none" | "bool" | "float"
    dropout_p: float
    scale: Optional[float]
    enable_gqa: bool
    kv_cache_layout: Optional[str]
    kv_cache_dtype: Optional[torch.dtype]
    packed_weights_id: Optional[str]
    policy_hash: Optional[str]
    tokenizer_id: Optional[str]
    vocab_hash: Optional[str]
    merges_hash: Optional[str]
    added_tokens_hash: Optional[str]
    normalizer_id: Optional[str]
    pretokenizer_id: Optional[str]
    special_tokens_hash: Optional[str]
    return_offsets: bool
    
    # Operation properties (from call)
    is_causal: bool
    
    # Runtime context
    is_cuda_graph_capturing: bool
    requires_deterministic: bool
    
    @classmethod
    def from_tensors(cls, q, k, v, **kwargs) -> "SelectionContext":
        """Build context by inspecting input tensors."""
        ...
```

### 5.2 Selection Pipeline

```python
class SelectionEngine:
    def select(self, operation: str, ctx: SelectionContext) -> KernelSpec:
        # Step 1: Check policy locks
        if locked := self.policy.get_locked_kernel(operation, ctx):
            return self.registry.get(locked)
        
        # Step 2: Check cache
        cache_key = self._make_cache_key(operation, ctx)
        if cached := self.cache.get(cache_key):
            return cached
        
        # Step 3: Get candidates
        candidates = self.registry.get_kernels(operation)
        
        # Step 4: Filter by hard constraints
        valid = []
        failures = {}
        for kernel in candidates:
            reasons = self._check_kernel(kernel, ctx)
            if not reasons:
                valid.append(kernel)
            else:
                failures[kernel.kernel_id] = reasons
        
        # Step 5: Handle no valid kernels
        if not valid:
            fallback = self.registry.get_fallback(operation)
            if fallback:
                self._log_fallback(operation, failures)
                return fallback
            raise NoKernelFoundError(operation, failures)
        
        # Step 6: Score and select
        selected = self._score_and_select(valid, ctx)
        
        # Step 7: Cache result
        self.cache.put(cache_key, selected)
        
        return selected
    
    def _check_kernel(self, kernel: KernelSpec, ctx: SelectionContext) -> list[Reason]:
        """Check all constraints, return failure reasons."""
        reasons = []
        
        # Hardware check
        if kernel.platform != "cuda" or ctx.device.type != "cuda":
            reasons.append(Reason("PLATFORM_MISMATCH", f"requires {kernel.platform}"))
        
        if ctx.sm_version < kernel.min_sm:
            reasons.append(Reason("SM_TOO_OLD", f"requires SM{kernel.min_sm}+, have SM{ctx.sm_version}"))
        
        if kernel.max_sm and ctx.sm_version >= kernel.max_sm:
            reasons.append(Reason("SM_TOO_NEW", f"requires SM<{kernel.max_sm}"))
        
        # Dtype check
        if ctx.dtype not in kernel.supported_dtypes:
            reasons.append(Reason("DTYPE_UNSUPPORTED", f"requires {kernel.supported_dtypes}"))
        
        # Shape checks
        if ctx.head_dim < kernel.min_head_dim or ctx.head_dim > kernel.max_head_dim:
            reasons.append(Reason("HEAD_DIM_INVALID", f"requires {kernel.min_head_dim}-{kernel.max_head_dim}"))
        
        if ctx.head_dim % kernel.head_dim_multiple != 0:
            reasons.append(Reason("HEAD_DIM_ALIGNMENT", f"requires multiple of {kernel.head_dim_multiple}"))
        
        if kernel.max_seq_len and ctx.seq_len_q > kernel.max_seq_len:
            reasons.append(Reason("SEQ_TOO_LONG", f"max {kernel.max_seq_len}"))
        
        # GQA check
        if ctx.num_kv_heads != ctx.num_heads and not kernel.supports_gqa:
            reasons.append(Reason("GQA_UNSUPPORTED", "kernel doesn't support GQA"))
        
        # CUDA graph check
        if ctx.is_cuda_graph_capturing and not kernel.is_cuda_graph_safe:
            reasons.append(Reason("CUDA_GRAPH_UNSAFE", "not safe for graph capture"))
        
        # Determinism check
        if ctx.requires_deterministic and not kernel.deterministic:
            reasons.append(Reason("NON_DETERMINISTIC", "kernel is non-deterministic"))
        
        # Availability check
        if not self._is_available(kernel):
            reasons.append(Reason("NOT_INSTALLED", f"{kernel.source} not installed"))
        
        # Custom check function
        reasons.extend(kernel.check(ctx))
        
        return reasons
    
    def _score_and_select(self, kernels: list[KernelSpec], ctx: SelectionContext) -> KernelSpec:
        """Score kernels and select the best one."""
        def score(k: KernelSpec) -> float:
            s = k.priority
            
            # Policy preferences
            if k.source in self.policy.preferred_sources:
                s += 20
            if k.source in self.policy.avoided_sources:
                s -= 50
            
            # PerfDB lookup (if available)
            if timing := self.perfdb.get_timing(k.kernel_id, ctx):
                # Normalize timing to priority adjustment
                # Faster kernels get bonus, slower get penalty
                s += self._timing_to_score(timing, ctx)
            
            return s
        
        return max(kernels, key=score)
```

### 5.3 Selection Overhead Mitigation
- Hot-path selection must be O(1) with minimal allocations
- Optional C++/Rust fast path if profiling shows >5us overhead per call
- `lz.compile(model)` can bake kernel choices to remove runtime selection

### 5.4 Cache Key Design

The cache key must be **sound**: never return a cached kernel that might be invalid.

```python
def _make_cache_key(self, operation: str, ctx: SelectionContext) -> str:
    """Generate cache key from context.
    
    Key includes ALL fields that affect kernel VALIDITY.
    Performance-only fields (like exact seq_len) use buckets.
    """
    # Validity-critical fields (exact values)
    validity_parts = [
        operation,
        ctx.device.type,
        str(ctx.sm_version),
        str(ctx.dtype),
        str(ctx.head_dim),
        str(ctx.num_heads),
        str(ctx.num_kv_heads),
        str(ctx.layout),
        str(ctx.stride_last_dim),
        str(ctx.attn_mask_type),
        str(ctx.enable_gqa),
        str(ctx.kv_cache_layout),
        str(ctx.kv_cache_dtype),
        str(ctx.packed_weights_id),
        str(ctx.policy_hash),
        str(ctx.tokenizer_id),
        str(ctx.vocab_hash),
        str(ctx.merges_hash),
        str(ctx.added_tokens_hash),
        str(ctx.normalizer_id),
        str(ctx.special_tokens_hash),
        str(ctx.is_causal),
        str(ctx.is_cuda_graph_capturing),
        str(ctx.requires_deterministic),
    ]
    
    # Performance fields (bucketed to improve hit rate)
    seq_bucket = self._bucket(ctx.seq_len_q, [128, 512, 2048, 8192, 32768])
    batch_bucket = self._bucket(ctx.batch_size, [1, 4, 16, 64, 256])
    
    perf_parts = [
        f"seq:{seq_bucket}",
        f"batch:{batch_bucket}",
    ]
    
    return ":".join(validity_parts + perf_parts)

def _bucket(self, value: int, boundaries: list[int]) -> int:
    """Return the smallest boundary >= value."""
    for b in boundaries:
        if value <= b:
            return b
    return boundaries[-1]
```

Notes:
- Cache implementation must be thread-safe under multi-stream workloads.
- Cache must be invalidated when backend capabilities or versions change.

### 5.5 Structured Reasons

```python
@dataclass
class Reason:
    code: str       # Machine-readable code
    message: str    # Human-readable explanation
    
    # Standard codes:
    # PLATFORM_MISMATCH, SM_TOO_OLD, SM_TOO_NEW
    # DTYPE_UNSUPPORTED, HEAD_DIM_INVALID, HEAD_DIM_ALIGNMENT
    # SEQ_TOO_LONG, GQA_UNSUPPORTED, CUDA_GRAPH_UNSAFE
    # ATTN_MASK_UNSUPPORTED, STRIDE_LAST_DIM
    # KV_CACHE_LAYOUT_MISMATCH, KV_CACHE_DTYPE_MISMATCH
    # PACKED_WEIGHTS_REQUIRED, CAPABILITIES_SCHEMA_MISMATCH
    # NON_DETERMINISTIC, NOT_INSTALLED, BACKEND_IMPORT_FAILED, CONSTRAINT_FAILED
```

### 5.6 Transform-Aware Scoring
- Penalize kernels that require layout/dtype transforms when adjacent ops can avoid them
- Favor kernels that fuse upstream/downstream ops (e.g., RoPE + attention)
- Track transform costs in ExecutionPlan for explainability

### 5.7 Plan-Aware Selection
- Optional planner builds a multi-op plan for a model block (attention + norm + MLP)
- Planner chooses kernels jointly to minimize total latency, not per-op latency
- Plans can be cached and baked via `lz.compile(model)`

### 5.8 Baked Plan Shape Bucketing
- Baked plans must be decision trees, not single kernels
- Use bucketed shape ranges to handle dynamic sequence lengths safely
- Fallback to runtime selection when shapes fall outside known buckets

### 5.9 Memory-Aware and Health-Aware Selection
- Consider memory overhead and workspace requirements during scoring
- Exclude kernels that exceed configured headroom
- Use backend health status to avoid recently failing kernels

---

## 6. Policy System

### 6.1 Policy Configuration

Policy is configured via YAML file or environment variables:

```yaml
# layerzero.yaml
version: 1

# Global settings
fallback_enabled: true
log_selections: false
explain_enabled: true
strict_mode: false           # Fail instead of adapting when constraints mismatch
plan_mode: false             # Enable plan-aware selection (multi-op)
allow_adaptation: true       # Allow layout/dtype transforms
graph_strict_mode: false     # Validate CUDA-graph safety via dummy capture
backend_isolation: in_process  # or subprocess

# Library preferences
prefer_sources:
  - flash_attn
  - flashinfer
  
avoid_sources:
  - experimental_lib

# Kernel locks (bypass selection entirely)
locks:
  attention.causal: "flash_attn.v3.causal"   # Always use FA3 for causal attention
  
# Conditional preferences
rules:
  # Prefer FlashAttention v3 on Hopper
  - match:
      sm: ">=90"
      operation: "attention.*"
    prefer: "flash_attn.v3.*"
    
  # Avoid xFormers for long sequences (known perf issue)
  - match:
      seq_len: ">16384"
    avoid_sources:
      - xformers
```

### 6.2 Environment Variable Overrides

```bash
# Disable LayerZero optimization (use fallbacks only)
export LAYERZERO_DISABLED=1

# Lock a specific kernel
export LAYERZERO_LOCK_ATTENTION="flash_attn.v3.causal"

# Prefer a library
export LAYERZERO_PREFER="flashinfer"

# Avoid a library
export LAYERZERO_AVOID="xformers"

# Enable verbose logging
export LAYERZERO_VERBOSE=1

# Require deterministic kernels
export LAYERZERO_DETERMINISTIC=1
```

### 6.3 Programmatic Configuration

```python
import layerzero as lz

# Configure at startup
lz.configure(
    prefer_sources=["flash_attn"],
    avoid_sources=["xformers"],
    fallback_enabled=True,
    log_selections=True,
)

# Lock a kernel
lz.lock("attention.causal", "flash_attn.v3.causal")

# Unlock
lz.unlock("attention.causal")

# Temporary override (context manager)
with lz.prefer("flashinfer"):
    output = lz.attention(q, k, v)
```

---

## 7. Performance Database (PerfDB)

### 7.1 Purpose

PerfDB stores measured kernel performance to improve selection beyond static priorities.

### 7.2 Storage Design

```
~/.cache/layerzero/
├── perfdb.sqlite          # SQLite database (WAL mode for concurrency)
├── config.yaml            # User configuration
└── logs/                  # Debug logs
```

Using SQLite with WAL mode provides:
- Concurrent reads from multiple processes
- Atomic writes
- No partial read issues
- Easy querying and migration

### 7.3 Schema

```sql
CREATE TABLE perf_records (
    id INTEGER PRIMARY KEY,
    
    -- Key fields
    kernel_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    device_id TEXT NOT NULL,        -- GPU UUID or name
    sm_version INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    head_dim INTEGER NOT NULL,
    seq_bucket INTEGER NOT NULL,
    batch_bucket INTEGER NOT NULL,
    
    -- Measurement
    median_us REAL NOT NULL,        -- Median latency in microseconds
    p95_us REAL NOT NULL,           -- P95 latency
    samples INTEGER NOT NULL,       -- Number of measurements
    variance_us REAL,              -- Variance for confidence scoring
    warmup_ms REAL,                -- One-time warmup or JIT cost

    -- Provenance
    kernel_version TEXT,            -- Version of kernel library
    layerzero_version TEXT,         -- LayerZero version
    capabilities_hash TEXT,         -- Backend capabilities signature
    power_state TEXT,               -- Optional: power/clock mode
    measured_at TIMESTAMP,
    
    -- Validity
    valid BOOLEAN DEFAULT TRUE,
    
    UNIQUE(kernel_id, operation, device_id, sm_version, dtype, 
           head_dim, seq_bucket, batch_bucket)
);

CREATE INDEX idx_lookup ON perf_records(
    operation, device_id, sm_version, dtype, head_dim, seq_bucket, batch_bucket
);
```

### 7.4 Measurement Protocol

```python
def benchmark_kernel(kernel: KernelSpec, ctx: SelectionContext, 
                     warmup: int = 5, samples: int = 20) -> PerfRecord:
    """Measure kernel performance with proper methodology."""
    
    # Create test inputs
    q, k, v = create_test_inputs(ctx)
    
    # Warmup
    for _ in range(warmup):
        kernel.impl(q, k, v, scale=None, causal=ctx.is_causal)
        torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        kernel.impl(q, k, v, scale=None, causal=ctx.is_causal)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds
    
    return PerfRecord(
        kernel_id=kernel.kernel_id,
        median_us=statistics.median(times),
        p95_us=statistics.quantiles(times, n=20)[18],  # 95th percentile
        samples=samples,
    )
```

### 7.5 PerfDB Integration in Selection

```python
def _timing_to_score(self, timing: PerfRecord, ctx: SelectionContext) -> float:
    """Convert timing data to priority adjustment.
    
    Faster kernels get positive adjustment, slower get negative.
    Scale factor ensures timing doesn't completely override priority.
    """
    # Get all timings for this context
    all_timings = self.perfdb.get_all_timings(ctx)
    if not all_timings:
        return 0.0
    
    # Find fastest timing
    fastest = min(t.median_us for t in all_timings)
    
    # Score adjustment: 0 for fastest, negative for slower
    # Cap at -30 to not completely override priority
    slowdown_ratio = timing.median_us / fastest
    return max(-30, -10 * (slowdown_ratio - 1))
```

Notes:
- Lower confidence (high variance, few samples) reduces PerfDB influence.
- Warmup/JIT time is tracked separately and should not distort steady-state scores.

### 7.6 Tuning Mode

```python
# Run tuning for a specific workload
lz.tune(
    operation="attention.causal",
    dtype=torch.bfloat16,
    head_dim=128,
    seq_lens=[512, 2048, 8192],
    batch_sizes=[1, 8, 32],
    samples=20,
)

# Or auto-tune during warmup
lz.configure(auto_tune_on_warmup=True, tune_samples=10)
```

---

## 8. API Reference

### 8.1 Core Operations

```python
import layerzero as lz

# Attention
output = lz.attention(
    query,                      # (B, S, H, D) or (B, H, S, D)
    key,                        # (B, S, Hkv, D) or (B, Hkv, S, D)
    value,                      # (B, S, Hkv, D) or (B, Hkv, S, D)
    causal: bool = True,        # Apply causal mask
    scale: float = None,        # Softmax scale, default 1/sqrt(D)
) -> Tensor

# Paged attention (for serving)
output = lz.paged_attention(
    query,                      # (B, 1, H, D) - single token
    key_cache,                  # (num_blocks, block_size, Hkv, D)
    value_cache,                # (num_blocks, block_size, Hkv, D)
    block_tables,               # (B, max_blocks) - block indices
    context_lens,               # (B,) - actual sequence lengths
    scale: float = None,
) -> Tensor

# RMSNorm
output = lz.rms_norm(
    input,                      # (..., D)
    weight,                     # (D,)
    eps: float = 1e-6,
) -> Tensor

# LayerNorm
output = lz.layer_norm(
    input,                      # (..., D)
    normalized_shape,           # (D,) or tuple
    weight = None,              # (D,)
    bias = None,                # (D,)
    eps: float = 1e-5,
) -> Tensor

# Tokenization
tokens = lz.tokenize(
    texts,                      # list[str] or str
    tokenizer_id: str,
    return_offsets: bool = False,
) -> dict

text = lz.detokenize(
    tokens,                     # list[int] or list[list[int]]
    tokenizer_id: str,
) -> str | list[str]

Notes:
- LayerZero must resolve `tokenizer_id` to a concrete tokenizer and compute
  `vocab_hash`, `merges_hash`, and `special_tokens_hash` for cache validity.

# Positional encodings
output = lz.rope(
    input,                      # (..., H, D)
    cos, sin,                   # precomputed tables
    interleaved: bool = False,
) -> Tensor

# Sampling
next_ids = lz.sample_topk(
    logits,                     # (..., vocab)
    k: int,
) -> Tensor

next_ids = lz.sample_topp(
    logits,                     # (..., vocab)
    p: float,
) -> Tensor

# Quantization / conversion
qt = lz.quantize(
    input,                      # Tensor
    dtype: str,                 # "int8", "fp8_e4m3", "mxfp4", ...
    scale: Tensor | None,
) -> Tensor
```

### 8.2 Inspection and Debugging

```python
# Check which kernel would be selected
info = lz.which(
    "attention.causal",
    q, k, v,  # Or provide explicit context
)
# Returns: {"kernel_id": "flash_attn.v3.causal", "score": 120}

# Get detailed explanation
report = lz.explain(
    "attention.causal",
    q, k, v,
)
# Returns: SelectionReport with all candidates, scores, and failure reasons

# List available kernels
kernels = lz.list_kernels("attention.causal")

# Validate a kernel works for given inputs
valid, reasons = lz.validate("flash_attn.v3.causal", q, k, v)

# Run diagnostics
lz.doctor()
# Prints: installed libraries, detected hardware, common issues

# Production readiness check
report = lz.readiness_check()
# Validates dispatch table, caches, and capabilities schemas
```

### 8.3 Configuration

```python
# Load config from file
lz.load_config("layerzero.yaml")

# Programmatic config
lz.configure(
    prefer_sources=["flash_attn"],
    avoid_sources=["xformers"],
    fallback_enabled=True,
    deterministic=False,
    log_selections=False,
    strict_mode=False,
    plan_mode=False,
    allow_adaptation=True,
    graph_strict_mode=False,
    backend_isolation="in_process",  # or "subprocess"
    memory_headroom_mb=1024,
)

# Kernel locking
lz.lock("attention.causal", "flash_attn.v3.causal")
lz.unlock("attention.causal")

# Context managers
with lz.prefer("flashinfer"):
    ...

with lz.disabled():
    # Uses fallback only
    ...
```

### 8.4 Warmup and Baking

```python
# Precompile JIT kernels for known shapes
lz.warmup(
    operation="attention.causal",
    shapes=[(1, 1024, 16, 128), (1, 2048, 16, 128)],
    dtype=torch.float16,
)

# Bake selections for static workloads
plan = lz.compile(model)

# Dry-run to see selections without executing kernels
report = lz.dry_run(model)

# Build a dispatch table for production (bucketed shapes)
dispatch_table = lz.solve(
    model,
    shape_buckets={"seq_len": [128, 512, 2048, 8192]},
    compile_jit=True,
)
```

Notes:
- `lz.warmup` should persist JIT caches to disk for reuse across restarts.
- `LAYERZERO_DEBUG=1` enables per-call `torch.cuda.synchronize()` for debugging.
- `lz.solve` should emit a decision tree, not a single kernel.
- Dispatch tables must include hardware signature + capabilities hash for validity.

---

## 9. Kernel Wrappers

### 9.1 Wrapper Structure

Each kernel library needs a wrapper that:
1. Checks availability
2. Adapts arguments to library's expected format
3. Calls the kernel
4. Adapts output back to standard format
5. Exposes a capabilities descriptor for data-driven constraints
6. Declares fused ops, prepack requirements, and layout/dtype transforms

```python
# layerzero/kernels/flash_attn.py

class FlashAttentionV3Wrapper:
    kernel_id = "flash_attn.v3.causal"
    operation = "attention.causal"
    source = "flash_attn"
    
    # Requirements
    platform = "cuda"
    min_sm = 90
    supported_dtypes = frozenset([torch.float16, torch.bfloat16])
    max_head_dim = 256
    head_dim_multiple = 8
    supports_gqa = True
    is_cuda_graph_safe = True
    deterministic = False
    priority = 100
    
    _available = None
    
    @classmethod
    def is_available(cls) -> bool:
        if cls._available is None:
            try:
                from flash_attn import flash_attn_func
                cls._impl = flash_attn_func
                cls._available = True
            except ImportError:
                cls._available = False
        return cls._available
    
    @classmethod
    def check(cls, ctx: SelectionContext) -> list[Reason]:
        # Additional checks beyond standard ones
        return []
    
    @classmethod
    def __call__(cls, q, k, v, scale=None, causal=True):
        # FlashAttention expects (B, S, H, D)
        # Detect input layout
        if q.shape[1] == q.shape[2]:
            # Ambiguous, assume BSHD
            pass
        elif q.shape[1] > q.shape[2]:
            # Likely BSHD, no change needed
            pass
        else:
            # BHSD -> BSHD
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            needs_transpose_back = True
        
        output = cls._impl(q, k, v, softmax_scale=scale, causal=causal)
        
        if needs_transpose_back:
            output = output.transpose(1, 2)
        
        return output
```

### 9.2 Layout Handling

Different libraries expect different layouts:

| Library | Expected Layout | Notes |
|---------|----------------|-------|
| FlashAttention | BSHD | `(batch, seq, heads, dim)` |
| FlashInfer | BHSD or NHD | Varies by API |
| xFormers | BSHD | `(batch, seq, heads, dim)` |
| PyTorch SDPA | BHSD | `(batch, heads, seq, dim)` |

LayerZero's API accepts both BSHD and BHSD and adapts internally:

```python
def _detect_layout(q: Tensor) -> str:
    """Detect tensor layout from shape."""
    # (B, S, H, D) vs (B, H, S, D)
    # Heuristic: H is typically smaller than S
    if q.ndim != 4:
        raise ValueError("Expected 4D tensor")
    
    # Shape: (dim0, dim1, dim2, dim3)
    # BSHD: dim1=seq (large), dim2=heads (small)
    # BHSD: dim1=heads (small), dim2=seq (large)
    
    if q.shape[1] > q.shape[2]:
        return "BSHD"
    elif q.shape[1] < q.shape[2]:
        return "BHSD"
    else:
        # Ambiguous (S == H), assume BSHD
        return "BSHD"
```

### 9.3 Backend Compatibility and Isolation
- Enforce a single CUDA/ROCm version policy for in-process backends
- Provide reference container images with mutually compatible backend sets
- Allow optional subprocess isolation for incompatible stacks
- Dynamic import alone is insufficient to prevent ABI conflicts

---

## 10. CUDA Graph Integration

### 10.1 Detection

```python
def _is_cuda_graph_capturing() -> bool:
    """Detect if currently inside CUDA graph capture."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except AttributeError:
        # Older PyTorch versions
        return False
```

### 10.2 Selection Behavior

When capturing CUDA graphs:
- Only `is_cuda_graph_safe=True` kernels are considered
- Selection is cached for the entire graph lifetime
- If no graph-safe kernel exists, raise clear error

```python
# In selection engine
if ctx.is_cuda_graph_capturing:
    candidates = [k for k in candidates if k.is_cuda_graph_safe]
    if not candidates:
        raise CudaGraphUnsafeError(
            f"No CUDA-graph-safe kernels available for {operation}. "
            f"Excluded: {[k.kernel_id for k in original_candidates]}"
        )
```

### 10.3 Graph Strict Mode
- When enabled, LayerZero runs a dummy graph capture during validation
- Kernels that allocate or synchronize are rejected
- Use `graph_strict_mode=true` for production graph capture

---

## 11. torch.compile Integration

### 11.1 Custom Op Registration

LayerZero operations are registered as custom ops for torch.compile compatibility:

```python
import torch
from torch.library import Library, impl

# Create library
lz_lib = Library("layerzero", "DEF")

# Define schema
lz_lib.define(
    "attention(Tensor q, Tensor k, Tensor v, float? scale, bool causal) -> Tensor"
)

# CUDA implementation
@impl(lz_lib, "attention", "CUDA")
def attention_cuda(q, k, v, scale, causal):
    return _dispatch_attention(q, k, v, scale, causal)

# Meta implementation for tracing
@impl(lz_lib, "attention", "Meta")  
def attention_meta(q, k, v, scale, causal):
    return torch.empty_like(q)

# Users can call via:
# torch.ops.layerzero.attention(q, k, v, scale, causal)
```

### 11.2 Functional API Uses Custom Ops

```python
def attention(q, k, v, scale=None, causal=True):
    """Public API that routes to custom op."""
    return torch.ops.layerzero.attention(q, k, v, scale, causal)
```

This ensures torch.compile treats LayerZero calls as opaque operations without graph breaks.

---

## 12. Framework Integration

### 12.1 vLLM Integration

```python
# In vLLM's attention layer
import layerzero as lz

class Attention:
    def forward(self, q, k, v, kv_cache, attn_metadata):
        if attn_metadata.is_prefill:
            return lz.attention(q, k, v, causal=True)
        else:
            return lz.paged_attention(
                q, 
                kv_cache.key_cache,
                kv_cache.value_cache,
                attn_metadata.block_tables,
                attn_metadata.context_lens,
            )
```

### 12.2 HuggingFace Transformers Integration

```python
# layerzero/integrations/transformers.py

def register_with_transformers():
    """Register LayerZero as a Transformers attention backend."""
    from transformers.modeling_utils import AttentionInterface
    
    @AttentionInterface.register("layerzero")
    def layerzero_attention(
        query, key, value, 
        attention_mask=None,
        is_causal=False,
        **kwargs
    ):
        # Adapt HF format to LayerZero format
        return lz.attention(query, key, value, causal=is_causal)

# Usage:
# model = AutoModel.from_pretrained(..., attn_implementation="layerzero")
```

### 12.3 Direct Usage

```python
# Custom model using LayerZero directly
import torch.nn as nn
import layerzero as lz

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = lambda x: lz.rms_norm(x, self.norm_weight)
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (B, S, H, D)
        q = q.view(B, S, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim)
        
        attn_out = lz.attention(q, k, v, causal=True)
        
        attn_out = attn_out.view(B, S, -1)
        return residual + self.o_proj(attn_out)
```

---

## 13. Distributed Training Considerations

### 13.1 Selection Consistency

In distributed training, all ranks must select the same kernel:

```python
def ensure_consistent_selection():
    """Ensure all ranks use same kernel selection."""
    if not torch.distributed.is_initialized():
        return
    
    # Rank 0 makes selection, broadcasts to all
    if torch.distributed.get_rank() == 0:
        selection = _local_selection()
        selections = [selection]
    else:
        selections = [None]
    
    torch.distributed.broadcast_object_list(selections, src=0)
    return selections[0]
```

Notes:
- If hardware differs across ranks, use “capability groups” and select per-group.
- In inference-only sharding, per-rank selection may be acceptable when numerics align.

### 13.2 Configuration

```python
lz.configure(
    distributed_selection="rank0_broadcast",  # or "independent"
)
```

---

## 14. Error Handling

### 14.1 Exception Hierarchy

```python
class LayerZeroError(Exception):
    """Base exception for LayerZero errors."""
    pass

class NoKernelFoundError(LayerZeroError):
    """No kernel available for the operation."""
    def __init__(self, operation: str, failures: dict[str, list[Reason]]):
        self.operation = operation
        self.failures = failures
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        lines = [f"No kernel available for '{self.operation}'"]
        lines.append("\nCandidate failures:")
        for kernel_id, reasons in self.failures.items():
            lines.append(f"  {kernel_id}:")
            for r in reasons:
                lines.append(f"    - [{r.code}] {r.message}")
        return "\n".join(lines)

class CudaGraphUnsafeError(LayerZeroError):
    """No CUDA-graph-safe kernel available."""
    pass

class KernelExecutionError(LayerZeroError):
    """Kernel raised an exception during execution."""
    pass
```

### 14.2 Fallback Behavior

```python
def _dispatch_with_fallback(operation, ctx, *args, **kwargs):
    try:
        kernel = selection_engine.select(operation, ctx)
        return kernel(*args, **kwargs)
    except NoKernelFoundError:
        if config.fallback_enabled:
            fallback = registry.get_fallback(operation)
            logger.warning(f"Using fallback for {operation}: {fallback.kernel_id}")
            return fallback(*args, **kwargs)
        raise
    except Exception as e:
        if config.fallback_enabled:
            fallback = registry.get_fallback(operation)
            logger.warning(f"Kernel failed, using fallback: {e}")
            return fallback(*args, **kwargs)
        raise KernelExecutionError(f"Kernel execution failed: {e}") from e
```

Notes:
- In `strict_mode`, LayerZero should avoid implicit layout/dtype adaptation and fail fast.

### 14.3 Observability
- Structured selection logs (kernel_id, reasons, latency)
- Optional OpenTelemetry spans for selection and dispatch

### 14.4 Debug and Async Errors
- `LAYERZERO_DEBUG=1` forces `torch.cuda.synchronize()` after dispatch
- Helps surface CUDA errors at the call site

### 14.5 Backend Circuit Breaker
- Disable a backend after repeated CUDA errors or illegal memory access
- Prevent cascading failures in production

### 14.6 Critical Challenges (gap.md) and Mitigations
- Selection overhead: use baked plans, plan‑aware selection, and optional compiled hot‑path dispatch
- Dependency conflicts: enforce single CUDA/ROCm policy or use subprocess isolation + reference containers
- JIT latency spikes: mandatory warmup and `lz.solve` with JIT compilation + persistent caches
- CUDA graph safety: validate via graph strict mode, reject unsafe kernels
- Dynamic shapes: bucketed dispatch tables with runtime fallback for out‑of‑range shapes

---

## 15. Implementation Roadmap

### 15.1 Phased Rollout (Full Scope)

| Phase | Scope | Notes |
|-------|-------|-------|
| P1: Core Engine | SelectionEngine, cache, policy, backend loader, capabilities, fallback | Foundation for all ops |
| P2: Attention + Norms | FA2/FA3, FlashInfer, xFormers, SDPA, Liger/Apex | Inference-critical |
| P3: Tokenization + PosEnc + Sampling | Tokenizers/tiktoken/SPM, RoPE/ALiBi, top-k/top-p | End-to-end inference |
| P4: Quantization + GEMM/MLP | INT8/FP8/FP4, fused MLP where available | Perf and cost |
| P5: CPU/ROCm/HPU | oneDNN/ZenDNN, ROCm backends, Habana | Hardware breadth |
| P6: Training + Backward | Backward kernels, determinism policy | Optional extension |

**Delivery is phased, but the architecture remains full-scope.**

---

## 16. Testing Strategy

### 16.1 Unit Tests

```python
# Test kernel constraint checking
def test_flash_attn_v3_requires_sm90():
    ctx = SelectionContext(sm_version=80, ...)
    kernel = FlashAttentionV3Wrapper()
    reasons = kernel.check(ctx)
    assert any(r.code == "SM_TOO_OLD" for r in reasons)

# Test selection
def test_selects_best_available():
    ctx = SelectionContext(sm_version=90, dtype=torch.bfloat16, ...)
    kernel = selection_engine.select("attention.causal", ctx)
    assert kernel.kernel_id == "flash_attn.v3.causal"
```

### 16.2 Numerical Correctness Tests

```python
def test_attention_matches_reference():
    q, k, v = random_inputs(...)
    
    # Reference: PyTorch SDPA
    ref = F.scaled_dot_product_attention(
        q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), 
        is_causal=True
    ).transpose(1,2)
    
    # LayerZero
    out = lz.attention(q, k, v, causal=True)
    
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
```

### 16.3 Fuzzing
- Randomized shapes/dtypes/masks vs PyTorch reference
- Run nightly with expanded coverage
- Fail fast on tolerance regressions

### 16.4 Integration Tests

```python
def test_vllm_integration():
    """Test LayerZero works in vLLM-like usage pattern."""
    # Simulate prefill
    q = torch.randn(1, 1024, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
    
    output = lz.attention(q, k, v, causal=True)
    
    assert output.shape == q.shape
    assert not torch.isnan(output).any()
```

---

## 17. Repository Structure

```
layerzero/
├── __init__.py                 # Public API exports
├── api.py                      # User-facing functions
├── config.py                   # Configuration management
├── context.py                  # SelectionContext
├── selection.py                # SelectionEngine
├── registry.py                 # Kernel registry
├── backend_loader.py           # Dynamic backend loading + entry_points
├── capabilities.py             # Capabilities descriptor parsing
├── cache.py                    # Selection cache
├── perfdb.py                   # Performance database
├── warmup.py                   # JIT warmup and cache handling
├── reasons.py                  # Structured reasons
├── exceptions.py               # Custom exceptions
│
├── kernels/                    # Kernel wrappers
│   ├── __init__.py
│   ├── base.py                 # KernelSpec base class
│   ├── flash_attn.py           # FlashAttention wrappers
│   ├── flashinfer.py           # FlashInfer wrappers
│   ├── xformers.py             # xFormers wrappers
│   ├── liger.py                # Liger wrappers
│   ├── apex.py                 # Apex wrappers
│   └── torch_fallback.py       # PyTorch fallbacks
│
├── ops/                        # Custom op registration
│   ├── __init__.py
│   └── attention.py
│
├── integrations/               # Framework integrations
│   ├── __init__.py
│   └── transformers.py
│
└── tests/
    ├── test_selection.py
    ├── test_kernels.py
    ├── test_correctness.py
    └── test_integration.py
```

---

## 18. Conclusion

LayerZero provides a practical, full-scope solution to kernel fragmentation in ML inference. By grounding the design in explicit APIs, data-driven constraints, backend isolation, and traceable selection decisions, we deliver a system that is:

1. **Correct**: Fallbacks ensure safety, strict validation and fuzzing protect accuracy
2. **Controllable**: Policy system and dry-run/warmup support production ops
3. **Extensible**: Plugins and capabilities descriptors allow continuous growth
4. **Performant**: Selection caching, PerfDB, and baked plans minimize overhead
5. **Portable**: Works across CUDA, ROCm, CPU, and HPU with phased adoption

Delivery is phased, but the architecture is full-scope from day one to avoid rewrite-driven churn.

---

## Appendix A: Quick Start

```python
# Install
pip install layerzero

# Basic usage
import layerzero as lz
import torch

q = torch.randn(2, 1024, 32, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(2, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)

# LayerZero automatically selects the best kernel
output = lz.attention(q, k, v, causal=True)

# Check what was selected
print(lz.which("attention.causal", q, k, v))
# {"kernel_id": "flash_attn.v3.causal", "score": 120}
```

## Appendix B: Configuration Reference

```yaml
# layerzero.yaml - Full configuration reference
version: 1

# Enable/disable LayerZero (use fallbacks when disabled)
enabled: true

# Always have a working fallback
fallback_enabled: true

# Log kernel selections (for debugging)
log_selections: false

# Require deterministic kernels
deterministic: false

# Strict mode disables implicit layout/dtype adaptation
strict_mode: false

# Enable plan-aware selection (multi-op planner)
plan_mode: false

# Allow layout/dtype transforms when needed
allow_adaptation: true

# Memory headroom for selection (MB)
memory_headroom_mb: 1024

# Library preferences (in order)
prefer_sources:
  - flash_attn
  - flashinfer
  - xformers

# Libraries to avoid
avoid_sources: []

# Explicit kernel locks
locks: {}
  # attention.causal: flash_attn.v3.causal

# PerfDB settings
perfdb:
  enabled: true
  path: ~/.cache/layerzero/perfdb.sqlite
  auto_tune: false
  tune_samples: 20

# Selection cache settings  
cache:
  enabled: true
  max_size: 10000

# Conditional rules
rules: []
```

## Appendix C: Supported Kernel Libraries

| Library | Operations | Installation |
|---------|------------|--------------|
| flash-attn | attention | `pip install flash-attn` |
| flashinfer | attention, paged_attention | `pip install flashinfer` |
| xformers | attention | `pip install xformers` |
| liger-kernel | rms_norm, layer_norm | `pip install liger-kernel` |
| apex | rms_norm, layer_norm | Build from source |
| triton | custom kernels | `pip install triton` |
| oneDNN | CPU kernels | system or `pip install onednn` (platform-specific) |
| ZenDNN | AMD CPU kernels | system install |
| tokenizers | tokenization | `pip install tokenizers` |
| tiktoken | tokenization | `pip install tiktoken` |
| sentencepiece | tokenization | `pip install sentencepiece` |
| HF kernels | various kernels | `pip install kernels` |

## Appendix D: Capabilities Descriptor Schema (v1)

Capabilities descriptors are JSON files shipped by backends or generated by CI.
They are validated at load time and hashed for cache invalidation.

Required fields:
```json
{
  "schema_version": "1.0",
  "backend": "flash_attn",
  "backend_version": "2.5.6",
  "platform": "cuda",
  "ops": {
    "attention.causal": [
      {
        "kernel_id": "flash_attn.v2.causal",
        "min_sm": 80,
        "max_sm": null,
        "dtypes": ["float16", "bfloat16"],
        "min_head_dim": 32,
        "max_head_dim": 256,
        "head_dim_multiple": 8,
        "max_seq_len": null,
        "supports_gqa": true,
        "supports_attn_mask": false,
        "supports_dropout": false,
        "requires_layouts": ["BSHD"],
        "produces_layout": "BSHD",
        "requires_last_dim_stride1": true,
        "is_cuda_graph_safe": true,
        "deterministic": false,
        "fuses_ops": ["posenc.rope"],
        "transform_cost_hint": 0
      }
    ]
  }
}
```

Validation rules:
- Reject unknown `schema_version` values (fail closed).
- Require `kernel_id` uniqueness per backend.
- Require explicit layout and dtype constraints.
- Missing fields default to “unsupported” rather than permissive values.

## Appendix E: Dispatch Table Format (lz.solve)

The solver emits a decision tree with bucketed shape ranges and precompiled
artifacts. This avoids runtime selection for known workloads.

```json
{
  "schema_version": "1.0",
  "model_id": "llama3-70b",
  "hardware_signature": "cuda_sm90_cu123",
  "shape_buckets": {
    "seq_len": [128, 512, 2048, 8192],
    "batch": [1, 4, 16, 64]
  },
  "plans": [
    {
      "bucket": { "seq_len": 512, "batch": 4 },
      "ops": [
        { "op": "posenc.rope", "kernel_id": "liger.rope" },
        { "op": "attention.causal", "kernel_id": "flash_attn.v2.causal" },
        { "op": "norm.rms", "kernel_id": "liger.rms_norm" }
      ],
      "jit_artifacts": ["flashinfer_cache_key_abc123"]
    }
  ],
  "fallback": { "mode": "runtime_select" }
}
```

## Appendix F: Compatibility Matrix (Initial)

This is illustrative. Real compatibility must be generated by CI probes.

| Backend | Platform | CUDA/ROCm | Notes |
|---------|----------|-----------|-------|
| FlashAttention | CUDA | 12.0+ | SM80+, head_dim <= 256 |
| FlashAttention | ROCm | 6.0+ | CK/Triton backends |
| FlashInfer | CUDA | 12.0+ | JIT and optional prebuilt cubins |
| xFormers | CUDA | varies | build‑dependent |
| Liger/Triton | CUDA/ROCm | varies | JIT compile |
| oneDNN | CPU | n/a | ISA‑dependent |
| ZenDNN | CPU | n/a | AMD EPYC |

## Appendix G: Production Readiness Checklist

- Backends installed with a single CUDA/ROCm version policy
- Capabilities descriptors validated (schema_version supported)
- `lz.solve` dispatch table generated and loaded (for production workloads)
- JIT caches persisted and warmup completed
- Graph strict mode validation (if CUDA graphs are used)
- Memory headroom configured and verified
- Health checks and circuit breakers enabled
- Selection and dispatch telemetry enabled (sampling acceptable)

## Appendix H: Risk Register (Top Risks)

| Risk | Impact | Mitigation |
|------|--------|------------|
| Python selection overhead | latency regressions | baked plans, plan‑aware selection, optional compiled dispatch |
| Backend ABI conflicts | crashes or corruption | single CUDA policy, containers, subprocess isolation |
| JIT spikes | P99 latency | warmup, solver‑triggered JIT, persistent caches |
| Graph unsafe kernels | capture failures | graph_strict_mode validation |
| Capability drift | incorrect selection | schema validation, CI capability matrix |
| Memory pressure | OOM | memory‑aware selection, headroom config |

---

## Appendix I: Production Hardening (v1.1 - Scenarios Analysis)

**Status:** CTO-Approved Updates from Ralph Loop Analysis
**Date:** 2026-01-16

This section documents critical enhancements identified through systematic failure scenario analysis. All items are required for production readiness.

### I.1 Selection Cache Thread-Safety (Problem 1)

**Requirement:** The selection cache must support high-concurrency LLM serving (10K+ QPS) without lock contention.

**Implementation:**
- Use MVCC (Multi-Version Concurrency Control) pattern with copy-on-write semantics
- Shard cache into 256 partitions with per-shard versioning
- Implement selection deduplication to prevent thundering herd on cache miss
- Use bounded LRU cache with configurable max size (default 10K entries) and 60s TTL

```python
@dataclass
class SelectionCacheConfig:
    num_shards: int = 256
    max_entries_per_shard: int = 40  # ~10K total
    entry_ttl_seconds: float = 60.0
    enable_deduplication: bool = True
```

### I.2 CUDA Launch Configuration Validation (Problem 2)

**Requirement:** Validate CUDA kernel launch configurations to prevent block limit crashes.

**Implementation:**
- Add `grid_layout: GridLayoutSpec` field to `KernelSpec` (required for each backend)
- Add `validate_launch_config(ctx: SelectionContext) -> list[Reason]` method
- Query device properties for SM-specific limits
- Prefer selecting different kernels over batch splitting when possible
- Batch splitting is last-resort fallback with performance warning

```python
@dataclass
class GridLayoutSpec:
    """Describes how kernel maps work to CUDA grid."""
    x_formula: str  # "batch * heads" or "batch * heads * seq_tiles"
    y_formula: Optional[str] = None
    z_formula: Optional[str] = None
    blocks_per_batch_head: int = 1
    tiles_sequence: bool = False
    tile_size: int = 128
```

**New Reason Codes:**
- `CUDA_BLOCK_LIMIT_EXCEEDED`
- `CUDA_GRID_DIM_EXCEEDED`

### I.3 JIT Compilation Strategy (Problem 3)

**Requirement:** Prevent JIT compilation latency from causing production timeouts.

**Implementation:**
- **Pre-flight warmup:** Blocking warmup at startup before serving
- **Runtime fallback:** If kernel not in cache, use fallback immediately (no inline JIT)
- **Background compilation:** Queue newly-seen shapes for background JIT
- **Bounded compile queue:** Max size configurable, drop-oldest policy
- **Fallback availability:** Add `has_fallback: bool` field to `OperationSpec`
- **Shape manifest:** Track production shapes, version by model config hash

```python
@dataclass
class JITConfig:
    enable_background_compile: bool = True
    compile_queue_max_size: int = 1000
    manifest_path: str = "~/.cache/layerzero/shape_manifest.json"
    manifest_stale_days: int = 7
    fail_on_no_fallback: bool = True
```

### I.4 CUDA Graph Safety (Problem 4)

**Requirement:** CUDA graph safety must be provable before production use.

**Implementation:**
- **Whitelist approach:** Maintain explicit list of verified graph-safe kernels
- **Default unsafe:** Unknown kernels default to `is_cuda_graph_safe=False`
- **Warmup protocol:** Mandatory CUBLAS/cuDNN warmup on side stream before capture
- **Shape-aware verification:** Track `shape_sensitive_graph_safety` for kernels with shape-dependent behavior
- **Memory threshold:** 1MB delta threshold for allocation detection warning

```python
@dataclass
class GraphSafetyConfig:
    strict_mode: bool = False  # Require verification before graph capture
    memory_delta_warning_mb: float = 1.0
    warmup_iterations: int = 3
    default_graph_safe: bool = False  # Unknown = unsafe
```

### I.5 Backend Compatibility (Problem 5)

**Requirement:** Detect and handle backend ABI conflicts safely.

**Implementation:**
- **Compatibility matrix:** CI-generated matrix from actual tests (not manual curation)
- **Version check at load:** Fail fast if incompatible versions detected
- **IPC limitations:** Document clearly; fallback to serialization when unsupported
- **Container isolation:** Opt-in, reserved for heterogeneous hardware (CUDA+ROCm)

```python
@dataclass
class BackendCompatibilityConfig:
    matrix_path: str = "~/.cache/layerzero/compat_matrix.json"
    check_at_load: bool = True
    isolation_mode: Literal["in_process", "subprocess", "container"] = "in_process"
    fail_on_incompatible: bool = True
```

### I.6 Layout Detection (Problem 6)

**Requirement:** Prevent silent incorrect layout assumptions.

**Implementation:**
- **Explicit layout parameter:** Add `layout: Optional[Literal["BSHD", "BHSD"]]` to attention API
- **Stride-based detection:** Use tensor stride patterns as primary signal
- **Rate-limited warnings:** Once per unique key per 60s when ambiguous
- **Configurable head counts:** Make common head count set configurable

```python
def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    causal: bool = True,
    layout: Optional[Literal["BSHD", "BHSD"]] = None,  # Required when S == H
) -> Tensor:
    ...
```

### I.7 PerfDB Environmental Awareness (Problem 7)

**Requirement:** Performance measurements must account for production conditions.

**Implementation:**
- **Environmental bucketing:** Temperature, memory pressure, power state
- **Optional NVML:** Make GPU monitoring optional with graceful fallback
- **Async timing:** Use non-blocking event queries in background thread
- **Calibrated adjustment:** Per-GPU-model adjustment factors derived empirically
- **Relative rankings:** Prefer ranking stability over absolute timing accuracy

```python
@dataclass
class EnvironmentalConditions:
    gpu_temperature_bucket: Literal["cold", "warm", "hot"]
    memory_pressure_bucket: Literal["low", "medium", "high"]
    power_state: Literal["boost", "nominal", "throttled"]
```

### I.8 Tokenization Cache Safety (Problem 8)

**Requirement:** Prevent tokenization cache key collisions.

**Implementation:**
- **Full hash:** Use 64-char SHA256 (not truncated)
- **Comprehensive config:** Include normalizer params, pretokenizer, special tokens
- **Model namespace:** Isolate cache by `model_id:tokenizer_hash`
- **Adaptive validation:** 10% for first 100 hits, 1% for next 1000, 0.1% thereafter
- **Multi-backend:** Support tiktoken (vocab dict hash), sentencepiece (binary hash)

```python
@dataclass
class TokenizerCacheConfig:
    enable_namespace_isolation: bool = True
    validation_rate_initial: float = 0.10  # First 100 hits
    validation_rate_warmup: float = 0.01   # Next 1000 hits
    validation_rate_stable: float = 0.001  # Thereafter
```

### I.9 Memory Fragmentation Awareness (Problem 9)

**Requirement:** Selection must consider memory fragmentation, not just free memory.

**Implementation:**
- **Cached probe:** Probe largest contiguous block with 60s TTL
- **Workspace method:** Add `workspace_bytes(ctx) -> int` to KernelSpec interface
- **Async defragmentation:** Schedule during idle periods, not synchronously
- **Probe-on-OOM:** Re-probe on allocation failure

```python
class KernelSpec:
    @abstractmethod
    def workspace_bytes(self, ctx: SelectionContext) -> int:
        """Return estimated workspace requirement in bytes."""
        ...
```

### I.10 Distributed Version Consistency (Problem 10)

**Requirement:** Prevent selection divergence during rolling updates.

**Implementation:**
- **Efficient version check:** Use packed int64 with all_reduce MIN/MAX
- **Periodic check:** Every 1000 selections or 60s (not every call)
- **Mode-aware fallback:** "fallback" for inference, "error" for training
- **Rolling update protocol:** Health check endpoints for Kubernetes

```python
@dataclass
class DistributedConfig:
    selection_mode: Literal["broadcast", "independent"] = "broadcast"
    version_check_interval: int = 1000  # selections
    on_version_mismatch: Literal["fallback", "error", "best_effort"] = "error"
    verify_selections: bool = True
```

### I.11 Updated Risk Register

| Risk | Impact | Mitigation (Updated) |
|------|--------|---------------------|
| Selection cache contention | latency spikes at high QPS | MVCC + 256 shards + deduplication |
| CUDA block limit crash | production outage | grid layout validation + batch split fallback |
| JIT compilation timeout | request failures | pre-flight warmup + background compile + fallback |
| CUDA graph capture failure | graph mode disabled | whitelist approach + warmup protocol |
| Backend ABI conflict | segfault or corruption | CI-generated compat matrix + fail-fast |
| Layout misdetection | silent incorrect results | explicit layout param + stride detection |
| PerfDB inaccuracy | suboptimal kernel selection | environmental bucketing + relative ranking |
| Tokenization cache collision | silent model degradation | full hash + namespace isolation + validation |
| Memory fragmentation OOM | allocation failures | contiguous block probe + workspace estimation |
| Distributed version skew | training divergence | packed version check + mode-aware fallback |

---

## Appendix J: Production Hardening v1.2 - Advanced Scenarios (Iteration 4)

**Status:** CTO-Approved Updates from Ralph Loop Iteration 4
**Date:** 2026-01-16

This section documents additional critical enhancements identified through systematic failure scenario analysis in Iteration 4. These address Blackwell architecture support, 4-bit quantization accuracy, tensor parallel determinism, KV cache management, and speculative decoding coordination.

### J.1 Blackwell Architecture Support (Problem 11)

**Requirement:** Support NVIDIA Blackwell (SM100/120) GPUs with generation-aware kernel routing.

**Implementation:**

- Add `GPUGeneration` enum: `TURING`, `AMPERE`, `ADA_LOVELACE`, `HOPPER`, `BLACKWELL`
- Extend `DeviceSpec` with `gpu_generation: GPUGeneration` and `tensor_core_generation: int`
- Add `KernelSpec.supported_generations: frozenset[GPUGeneration]`
- Add `KernelSpec.instruction_set: Optional[str]` for "wgmma" vs "tcgen05.mma"
- Generation mapping configurable via capabilities descriptors

```python
@dataclass
class GPUGenerationConfig:
    """GPU generation detection and routing configuration."""
    sm_to_generation: dict[range, str] = field(default_factory=lambda: {
        range(75, 80): "turing",
        range(80, 89): "ampere",
        range(89, 90): "ada",
        range(90, 100): "hopper",
        range(100, 200): "blackwell",
    })
    generation_to_tc_gen: dict[str, int] = field(default_factory=lambda: {
        "turing": 2, "ampere": 3, "ada": 3, "hopper": 4, "blackwell": 5
    })
```

**New Reason Codes:**
- `GPU_GENERATION_UNSUPPORTED`
- `TENSOR_CORE_GEN_UNSUPPORTED`
- `INSTRUCTION_SET_MISMATCH`

### J.2 4-Bit Quantization Format Selection (Problem 12)

**Requirement:** Select optimal 4-bit quantization format considering accuracy and hardware support.

**Implementation:**

- Add `QuantFormat` enum: `INT4`, `NVFP4`, `MXFP4`, `FP8_E4M3`, `FP8_E5M2`
- Add `QuantFormatSpec` with `accuracy_rank`, `group_size`, `hardware_vendors`
- Accuracy ranking: NVFP4 (10) > INT4 (8) > MXFP4 (5)
- Cross-vendor fallback paths with requantization cost tracking
- Optional accuracy profile override for model-specific tuning

```python
@dataclass
class QuantFormatSelectionConfig:
    """Configuration for quantization format selection."""
    accuracy_priority: Literal["high", "balanced", "performance"] = "balanced"
    allow_format_fallback: bool = True
    max_accuracy_drop_percent: float = 3.0
    requantization_enabled: bool = False
    accuracy_profile: Optional[dict[str, int]] = None  # Model-specific overrides
```

**New Reason Codes:**
- `QUANT_FORMAT_UNSUPPORTED`
- `QUANT_ACCURACY_THRESHOLD_EXCEEDED`
- `REQUANTIZATION_REQUIRED`

### J.3 Tensor Parallel Determinism (Problem 13)

**Requirement:** Guarantee deterministic outputs across tensor parallel sizes for RL and evaluation workloads.

**Implementation:**

- Add `TPInvarianceSpec` with `is_tp_invariant`, `reduction_order`, `uses_tbik`
- Add `DeterministicConfig.require_tp_invariant: bool`
- Synchronized kernel selection via int64 hash broadcast
- Fallback to deterministic reference kernels when optimized TP-invariant kernels unavailable

```python
@dataclass
class TPInvarianceConfig:
    """Configuration for tensor parallel invariance."""
    require_tp_invariant: bool = False
    reduction_algorithm: Literal["ring", "tree", "canonical"] = "tree"
    tp_invariant_fallback_mode: Literal["reference", "error", "best_effort"] = "reference"
    synchronize_selection: bool = True
```

**New Reason Codes:**
- `TP_INVARIANCE_REQUIRED`
- `TP_SIZE_EXCEEDED`
- `REDUCTION_ORDER_MISMATCH`

### J.4 KV Cache Strategy Abstraction (Problem 14)

**Requirement:** Support multiple KV cache strategies (contiguous, paged, vAttention) with CUDA graph compatibility.

**Implementation:**

- Add `KVCacheStrategy` enum: `CONTIGUOUS`, `PAGED`, `VIRTUAL`, `UNIFIED`
- Add `KVCacheManager` abstract interface
- Implement `vAttentionManager` using CUDA VMM APIs (cuMemAddressReserve, cuMemMap)
- Add `KernelSpec.supports_kv_strategies: frozenset[KVCacheStrategy]`
- Driver version validation for vAttention
- Per-device virtual address space management for multi-GPU

```python
@dataclass
class KVCacheConfig:
    """Configuration for KV cache management."""
    strategy: KVCacheStrategy = KVCacheStrategy.VIRTUAL
    virtual_size_gb: float = 64.0
    physical_chunk_size_mb: float = 16.0
    enable_cpu_offload: bool = False
    min_driver_version: str = "520.0"  # CUDA 11.2+ driver required
```

**New Reason Codes:**
- `KV_STRATEGY_UNSUPPORTED`
- `DRIVER_VERSION_UNSUPPORTED`
- `VIRTUAL_MEMORY_EXHAUSTED`

### J.5 Speculative Decoding Coordination (Problem 15)

**Requirement:** Coordinate kernel selection between draft and target models for speculative decoding.

**Implementation:**

- Add `SpeculativeDecodingContext` with draft/target configuration
- Add `SpeculativeKernelMatcher` for compatible kernel pair selection
- Add `AdaptiveSpeculativeSelector` for batch-size-aware speculation
- Pipeline parallelism validation (currently incompatible)
- Algorithm-specific configuration (EAGLE, EAGLE-3, Medusa)
- KV cache sharing mode option

```python
@dataclass
class SpeculativeDecodingConfig:
    """Configuration for speculative decoding kernel coordination."""
    enabled: bool = False
    algorithm: Literal["eagle", "eagle3", "medusa", "generic"] = "generic"
    draft_tp_size: int = 1  # Configurable with validation
    max_draft_tp_size: int = 1  # Current constraint
    num_speculative_tokens: int = 5
    adaptive_speculation: bool = True
    share_kv_cache: bool = False
    cache_kernel_pairs: bool = True
```

**New Reason Codes:**
- `SPEC_DECODE_PP_INCOMPATIBLE`
- `SPEC_DECODE_DRAFT_TP_CONSTRAINT`
- `SPEC_DECODE_KV_INCOMPATIBLE`
- `SPEC_DECODE_ALGORITHM_UNSUPPORTED`

### J.6 Updated Risk Register (v1.2)

| Risk | Impact | Mitigation (v1.2) |
|------|--------|-------------------|
| Blackwell performance degradation | 50%+ slower inference | Generation-aware routing + FA4 integration |
| 4-bit quantization accuracy loss | Model degradation | Accuracy-ranked format selection + fallback paths |
| TP non-determinism | RL training corruption | TP-invariant kernel mode + synchronized selection |
| KV cache block table overhead | CPU bottleneck | vAttention with CUDA VMM + graph compatibility |
| Speculative decoding failures | Draft-target mismatch | Kernel pair matching + algorithm-specific config |
