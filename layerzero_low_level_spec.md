# LayerZero Low-Level Spec (Production-Oriented)

Version: 0.1
Status: Draft for review

## 0. Methodology
I am applying systems design + FMEA (failure mode and effects analysis) + constraints-driven architecture. The system has many interacting parts (kernels, backends, device capabilities, and framework integration), so this combination best exposes operational risks, edge cases, and performance tradeoffs while keeping the design grounded in real-world constraints.

## 1. Scope and Goals

Primary objective: a kernel orchestration layer that selects the right kernel at runtime (dynamic) or honors user-defined selection (static) across the full inference stack (tokenization, embeddings, positional encoding, attention, NN primitives, sampling/decoding, quantization), and integrates natively with PyTorch and Hugging Face (Transformers, diffusers, pipelines).

Full-scope target operations:
- Tokenization and pre/postprocessing (library selection across tokenizers)
- Embedding and positional ops (RoPE, ALiBi, rotary variants)
- Attention (causal, full, paged/paged KV cache)
- NN primitives (GEMM/Linear, activation, elementwise, pooling, convolution)
- RMSNorm / LayerNorm
- Fused MLP and activation kernels where backends expose stable APIs
- Sampling/decoding kernels (top-k, top-p, speculative)
- Quantization/dequantization kernels and format conversions

Hardware targets:
- NVIDIA CUDA GPUs (multi-CUDA versions, SM 7.5+ to Blackwell)
- AMD ROCm GPUs (CDNA/RDNA where kernel libs exist)
- Intel GPU via oneAPI/oneDNN (optional)
- Habana Gaudi (HPU) support path
- CPU (oneDNN, ZenDNN, SIMD portable kernels)

Precision targets (super set; kernels may support subsets):
- FP32, TF32, FP16, BF16
- INT8 (per-tensor and per-channel)
- FP8 (E4M3, E5M2)
- MXFP4, NVFP4
- Other vendor-specific low-precision formats as supported by kernels

Key outcomes:
- Consistent and debuggable kernel selection across frameworks
- Reduced duplicated selection logic in serving stacks
- Production-safe fallbacks and traceability for every decision
- Clear integration path for external kernel repos and HF kernel hub

## 2. Non-Goals (for the first production release)

- Full training support for all ops (forward only is primary)
- Universal kernel interception or graph rewriting
- Replacing torch.compile or TensorRT
- Removing the need to install kernel libraries

## 3. Terminology

- Operation: A logical computation (e.g., attention.causal).
- Kernel: A concrete implementation in a specific library.
- Backend: A family of kernels from a vendor/library (flash-attn, flashinfer, xformers, oneDNN, etc).
- Context: Runtime properties of tensors, device, and config.
- ExecutionPlan: Selected kernel plus required argument transforms (layout, dtype, scaling).

### 3.1 Operation Taxonomy

LayerZero is not limited to attention. It must orchestrate kernels across:
- Tokenization and text preprocessing (CPU-bound, library selection)
- Embeddings and positional encodings (RoPE/ALiBi)
- Attention (prefill/decode, paged KV-cache)
- Linear/GEMM and MLP (including fused activations)
- Normalization (RMSNorm/LayerNorm)
- Sampling and decoding (top-k/top-p/speculative)
- Quantization and dtype conversion (INT8/FP8/FP4 formats)
- Auxiliary kernels (KV-cache compaction, block table updates)

## 4. Architecture Overview

```
User Code / Frameworks
  |
  | lz.attention(...)
  v
LayerZero API
  |
  v
Selection Engine
  |-- Policy (static locks, rules, allow/deny lists)
  |-- Filter (hard constraints)
  |-- Scoring (priority + perfdb + rules)
  |-- Cache (validity + perf buckets)
  v
Backend Loader / Plugin Manager
  |-- Dynamic imports (importlib) + entry_points
  |-- Capabilities descriptors (JSON)
  |-- Backend health tracking
  v
Kernel Registry (KernelSpec + BackendSpec)
  |
  v
Kernel Wrappers (FlashAttention, FlashInfer, xFormers, Liger, oneDNN, ZenDNN, HF kernels)
  |
  v
External Kernel Libraries
```

Notes:
- Backends are optional and isolated. Missing imports disable that backend without crashing.
- Capabilities are data-driven and can be updated without code changes.
- Optional process isolation may be used to avoid ABI conflicts (future extension).

## 5. Data Model (Low-Level)

### 5.1 OperationSpec
Defines the semantic contract and valid input space.

```
OperationSpec:
  op_id: "attention.causal"
  op_kind: "tensor" | "tokenization" | "sampling" | "communication" | "prepost"
  input_kinds: ["tensor"] | ["text"] | ["tokens"]
  output_kind: "tensor" | "tokens" | "text"
  semantics: explicit math definition
  tokenization_metadata_required: ["tokenizer_id", "vocab_hash", "merges_hash", "normalizer_id", "added_tokens_hash", "special_tokens_hash"]
  returns_offsets: bool
  layouts: ["BSHD", "BHSD"]
  supports_gqa: true
  supports_batched: true
  fallback_impl: torch-based reference
  precision_tolerances: rtol/atol by dtype
```

### 5.2 KernelSpec
Describes a single kernel implementation.

```
KernelSpec:
  kernel_id: "flash_attn.v2.causal"
  operation: "attention.causal"
  source: "flash_attn"
  version: "2.x"
  impl: callable

  platform: "cuda" | "rocm" | "cpu" | "hpu"
  min_sm: int
  max_sm: optional int
  supported_dtypes: set[torch.dtype]
  supported_quant_dtypes: set[str]  # "int8", "fp8_e4m3", "fp8_e5m2", "mxfp4", "nvfp4"
  quantization_requires_scales: bool
  quantization_scale_granularity: "per_tensor" | "per_channel" | "per_head" | "blockwise"
  quantization_zero_point: bool
  min_head_dim: int
  max_head_dim: int
  head_dim_multiple: int
  max_seq_len: optional int
  supports_gqa: bool
  supports_bfloat16: bool
  supports_fp8: bool
  supports_int8: bool
  supports_mxfp4: bool
  supports_nvfp4: bool
  supports_attn_mask: bool
  supported_attn_mask_types: set[str]  # "none", "bool", "float"
  supports_dropout: bool
  supports_scale: bool
  requires_layouts: set[str]
  produces_layout: optional str
  requires_dtype: optional torch.dtype
  produces_dtype: optional torch.dtype
  supports_kv_cache_layouts: set[str]
  supports_kv_cache_dtypes: set[torch.dtype]

  is_cuda_graph_safe: bool
  deterministic: bool
  requires_contiguous: bool
  requires_last_dim_stride1: bool
  requires_aligned: bool
  priority: int  # base priority (0-100)
  fuses_ops: list[str]
  requires_packed_weights: bool
  supports_prepack: bool
  transform_cost_hint: int
  tokenizer_ids: optional set[str]
  vocab_hashes: optional set[str]
  supports_offsets: bool
  supports_normalizers: optional set[str]
  supports_pretokenizers: optional set[str]

  check(ctx) -> list[Reason]
  adapt(args, ctx) -> AdaptedArgs
```

### 5.3 BackendSpec
Defines backend availability and compatibility.

```
BackendSpec:
  name: "flashinfer"
  version: "0.5.3"
  platform: "cuda"
  import_name: "flashinfer"
  plugin_origin: "builtin" | "entry_point" | "hub"
  entry_point_group: "layerzero.backends"
  init_cost_ms: optional (for JIT and cold-start)
  requires: ["cuda>=12.0", "torch>=2.2"]
  provides_ops: ["attention.causal", "attention.paged", "norm.rms"]
  supports_graphs: bool
  capabilities: optional dict or json path
  capabilities_hash: optional str
  capabilities_schema_version: optional str
```

### 5.4 DeviceSpec
Captures device capabilities and driver/runtime info.

```
DeviceSpec:
  device_type: "cuda" | "rocm" | "cpu" | "hpu"
  device_name: "RTX 3080"
  device_vendor: "nvidia" | "amd" | "intel" | "habana"
  device_uuid: string
  sm_version: int (for CUDA)
  roc_version: string (for ROCm)
  cuda_version: string
  driver_version: string
  supports_bf16: bool
  supports_fp8: bool
  supports_fp16: bool
  supports_tensor_cores: bool
  supports_graphs: bool
```

### 5.5 SelectionContext
Constructed per call.

```
SelectionContext:
  op_id
  op_metadata: dict
  device_spec
  dtype
  quant_dtype: optional str
  quant_scale_granularity: optional str
  quant_zero_point: optional bool
  batch_size
  seq_len_q
  seq_len_k
  num_heads
  num_kv_heads
  head_dim
  layout
  layout_hint: optional str
  stride_last_dim
  is_contiguous
  is_last_dim_contiguous
  is_causal
  attn_mask_type: "none" | "bool" | "float"
  attn_mask_layout: "broadcast" | "per_head" | "block_sparse"
  dropout_p
  scale
  enable_gqa
  is_training
  requires_grad
  tokenizer_id: optional str
  vocab_hash: optional str
  merges_hash: optional str
  added_tokens_hash: optional str
  normalizer_id: optional str
  pretokenizer_id: optional str
  special_tokens_hash: optional str
  return_offsets: bool
  kv_cache_layout: optional str
  kv_cache_dtype: optional torch.dtype
  packed_weights_id: optional str
  policy_hash: optional str
  memory_headroom_bytes: optional int
  free_memory_bytes: optional int
  is_cuda_graph_capturing
  requires_deterministic
  allow_fallback
  allow_jit_compile
```

### 5.6 ExecutionPlan
Final selected kernel and any required transforms.

```
ExecutionPlan:
  kernel_id
  transforms: [transpose, contiguous, scale_adjustment, cast, dequant, requant]
  fused_ops: list[str]
  transform_cost_total: int
  uses_fallback: bool
  debug_info: SelectionReport
```

### 5.7 Reason Codes (Standardized)
Examples:
- PLATFORM_MISMATCH
- DEVICE_CAPABILITY_UNSUPPORTED
- DTYPE_UNSUPPORTED
- HEAD_DIM_INVALID
- HEAD_DIM_ALIGNMENT
- HEAD_DIM_TOO_LARGE
- SEQ_TOO_LONG
- GQA_UNSUPPORTED
- GQA_HEADS_MISMATCH
- ATTN_MASK_UNSUPPORTED
- ATTN_MASK_INVALID
- CUDA_GRAPH_UNSAFE
- NON_DETERMINISTIC
- NOT_INSTALLED
- BACKEND_IMPORT_FAILED
- JIT_DISABLED
- NOT_CONTIGUOUS
- STRIDE_LAST_DIM
- TOKENIZER_ID_MISMATCH
- VOCAB_HASH_MISMATCH
- NORMALIZER_MISMATCH
- MERGES_HASH_MISMATCH
- ADDED_TOKENS_MISMATCH
- SPECIAL_TOKENS_MISMATCH
- PACKED_WEIGHTS_REQUIRED
- KV_CACHE_LAYOUT_MISMATCH
- KV_CACHE_DTYPE_MISMATCH
- CAPABILITIES_SCHEMA_MISMATCH
- MEMORY_HEADROOM_EXCEEDED
- PLAN_BUCKET_MISS
- BACKEND_ERROR

## 6. Backend Loading and Kernel Provider Interface

### 6.1 Dynamic Backend Loading
Backends are optional and must be lazily imported to avoid dependency conflicts:
- Use `importlib` inside `KernelProvider.is_available()`
- Never import heavy backends at module import time
- Import failures are treated as backend-unavailable, not fatal

### 6.2 Capabilities Descriptors (Data-Driven Constraints)
Each backend should expose a capabilities descriptor (JSON or dict) that declares
supported ops, dtypes, shapes, and device constraints. The selection engine reads
this descriptor to build `KernelSpec` entries.

Requirements:
- Include a `schema_version` field and validate against supported versions
- Reject ambiguous or incomplete constraints (fail closed)
- Hash the descriptor to support cache invalidation

### 6.3 Capabilities Schema (v1)
Minimum required fields:
```
schema_version: "1.0"
backend: "flash_attn"
backend_version: "2.5.6"
platform: "cuda"
ops:
  attention.causal:
    - kernel_id: "flash_attn.v2.causal"
      min_sm: 80
      max_sm: null
      dtypes: ["float16", "bfloat16"]
      min_head_dim: 32
      max_head_dim: 256
      head_dim_multiple: 8
      max_seq_len: null
      supports_gqa: true
      supports_attn_mask: false
      supports_dropout: false
      requires_layouts: ["BSHD"]
      produces_layout: "BSHD"
      requires_last_dim_stride1: true
      is_cuda_graph_safe: true
      deterministic: false
      fuses_ops: ["posenc.rope"]
      transform_cost_hint: 0
```

Validation rules:
- Unknown `schema_version` is a hard failure (backend disabled).
- Missing layout/dtype constraints are invalid.
- Kernel IDs must be unique across a backend.
 
### 6.4 Plugin Entry Points
Support `entry_points` for third-party kernels:
- Entry point group: `layerzero.backends`
- Allows `pip install my-kernel` to auto-register without user code changes

### 6.5 Kernel Provider Interface

All kernel repos integrate through a unified adapter:

```
class KernelProvider:
  def is_available() -> bool
  def get_kernel_specs() -> list[KernelSpec]
  def validate(ctx) -> list[Reason]
  def run(*args, **kwargs) -> Tensor
  def supports(ctx) -> bool
```

Adapter responsibilities:
- Import and version detection
- Layout conversion (BSHD <-> BHSD <-> NHD/HND)
- dtype conversion (fp16/bf16/fp32)
- argument normalization (scale, causal, masks)
- error mapping to LayerZero reason codes

## 7. Selection Engine

### 7.1 Pipeline
1) Build SelectionContext from inputs
2) Apply policy locks and allow/deny rules
3) Cache lookup (validity + perf buckets)
4) Filter by hard constraints (reasons recorded)
5) Score candidates (priority + PerfDB + policy bonuses)
6) Select best kernel or fallback
7) Cache result and dispatch

### 7.2 Scoring
Weighted score example:
```
score = priority
score += policy_bonus
score += perfdb_bonus
score -= jit_penalty_if_first_use
score -= reliability_penalty_if_recent_errors
score -= transform_cost_penalty
score += fusion_bonus_if_fuses_adjacent_ops
```

### 7.3 Caching
Two layers:
- Validity cache: keyed by fields that affect correctness
- Perf cache: bucketed by seq_len and batch_size

Cache key must include:
- op_id, dtype, device_type, sm_version, head_dim, num_heads, num_kv_heads
- layout, stride_last_dim, is_last_dim_contiguous
- seq_len_q, seq_len_k (or buckets) when kernels have max_seq_len constraints
- is_causal, attn_mask_type, dropout_p, scale, enable_gqa
- is_cuda_graph_capturing, requires_deterministic
- tokenizer_id, vocab_hash, merges_hash, added_tokens_hash, normalizer_id, special_tokens_hash (tokenization ops)
- kv_cache_layout, kv_cache_dtype, packed_weights_id (when applicable)
- backend_version or capabilities_hash when cached selection depends on backend constraints
- policy_hash and memory_headroom_bytes when memory-aware selection is enabled

### 7.4 Failure Handling
- If selected kernel fails at runtime, mark as temporarily unhealthy and fallback
- Record failure signature in perfdb/healthdb
- Optional exponential backoff to retry

### 7.5 Selection Overhead Mitigations
- Hot-path cache should be O(1) and minimal allocations
- Optional C++/Rust fast path for selection lookup if Python overhead exceeds budget
- Provide `lz.compile(model)` to bake kernel selections for static workloads
- Provide `lz.dry_run(model)` to show what would be selected without execution

### 7.6 Plan-Aware Selection
- Allow a planner to select kernels jointly across adjacent ops
- Penalize plans with expensive layout/dtype transforms
- Cache multi-op plans by model block signature

Plan execution:
- Baked plans must be decision trees to handle dynamic shapes
- Bucketed dispatch is required for variable sequence lengths

### 7.7 Strict vs Lenient Adaptation
- `strict_mode`: refuse implicit layout/dtype transforms, fail fast
- `allow_adaptation`: permit transforms with explicit cost accounting

### 7.8 Memory-Aware Selection
- Estimate workspace and temporary allocation requirements per kernel
- Reject candidates that exceed configured memory headroom
- Record memory rejections in SelectionReport

### 7.9 Build-Time Solver
- `lz.solve` generates a dispatch table with bucketed shape rules
- Solver triggers JIT compilation for all bucketed shapes
- Dispatch tables include a hardware signature and capabilities hash

## 8. Policy System (Static vs Dynamic)

### 8.1 Static Selection
Use YAML or environment variables to lock kernels or restrict backends.
Example:
```
locks:
  attention.causal: "flashinfer.causal"
avoid_sources:
  - xformers
strict_mode: false
plan_mode: false
allow_adaptation: true
graph_strict_mode: false
backend_isolation: "in_process"  # or "subprocess"
```

### 8.2 Dynamic Selection
Use runtime heuristics and perfdb timings; override with policy rules.

### 8.3 Rule Engine
Rules match on context fields:
```
rules:
  - match:
      sm: ">=90"
      op: "attention.*"
    prefer: "flash_attn.v3.*"
  - match:
      seq_len: ">16384"
    avoid_sources: ["xformers"]
```

## 9. PerfDB

### 9.1 Schema Extensions
Add fields for driver/runtime to keep results valid:
- cuda_version, driver_version, backend_version
- torch_version, layerzero_version
- device_uuid
- capabilities_hash, warmup_ms, variance_us

### 9.2 Sampling Strategy
- Warmup + timed runs
- Median + p95 stored
- Track variance for stability

### 9.3 Invalidations
- Invalidate on driver/toolkit change
- Invalidate on kernel version change
- Optional TTL for nightly builds

### 9.4 Warmup and JIT Cache
- Provide `lz.warmup(shapes)` to compile JIT kernels before serving traffic
- Persist JIT cache directories to disk across restarts
- Track warmup completeness in health/telemetry

## 10. Backend Integration Details

### 10.1 FlashAttention (FA2/FA3)
Key constraints:
- CUDA >= 12.0
- FA2 supports Ampere/Ada/Hopper (SM80+), fp16/bf16, head_dim <= 256
- FA3 is H100/H800 (SM90), CUDA >= 12.3
- Layout: BSHD
- Causal mask alignment differs when seq_q != seq_k (bottom-right aligned)
- ROCm backend exists (CK + Triton); CK targets MI200/MI300, Triton supports CDNA/RDNA with GQA/ALiBi/RoPE
Risks:
- Strict head_dim and dtype constraints
- SM version gating
Mitigation:
- Encode constraints in KernelSpec.check
- Provide fallback to SDPA or xformers

Refs:
- https://github.com/Dao-AILab/flash-attention
- https://arxiv.org/abs/2205.14135
- https://tridao.me/publications/flash2/flash2.pdf
- https://tridao.me/publications/flash3/flash3.pdf

### 10.2 FlashInfer
Key constraints:
- Multiple backends (FlashAttn, cuDNN, CUTLASS, TRT-LLM)
- Supports SM75+ through Blackwell
- Layout for prefill/decode is NHD/HND (no batch) or dedicated batch APIs
- GQA group sizes supported (backend-specific) and custom mask layouts supported
- JIT kernel generation and optional precompiled cubins
- Paged KV-cache APIs have strict layout metadata (page size, block tables)
- KV metadata (block_tables/context_lens) dtype and device placement are strict
Risks:
- JIT overhead on first use
- API shape differences vs SDPA
Mitigation:
- Warmup/tuning phase
- Adapter to map BSHD/BHSD to NHD/HND
- Encode per-backend head_dim constraints (benchmarks show specific head_dim sets)

Refs:
- https://github.com/flashinfer-ai/flashinfer
- https://docs.flashinfer.ai

### 10.3 xFormers
Key constraints:
- memory_efficient_attention expects BSHD
- optional ROCm builds exist
- can use multiple internal backends
- attn_bias must be on the same device and must not broadcast batch/head dims
- Experimental support for MQA/GQA via 5D inputs
Risks:
- Differences in supported dtypes and masks by backend
- Build issues on custom CUDA versions
Mitigation:
- Use xformers.info to detect available kernels
- Record supported masks/dtypes in KernelSpec
- Explicitly expand attn_bias to avoid implicit broadcasting

Refs:
- https://github.com/facebookresearch/xformers

### 10.4 Liger Kernel
Key constraints:
- Triton kernels for RMSNorm, RoPE, SwiGLU, etc
- HF model patching APIs (training oriented but usable in inference)
- Supports CUDA and ROCm via Triton
Risks:
- Triton version compatibility
- Kernel API changes across releases
Mitigation:
- Treat as optional backend with strict version pinning
- Test RMSNorm/LayerNorm outputs against PyTorch

Refs:
- https://github.com/linkedin/Liger-Kernel
- https://arxiv.org/pdf/2410.10989

### 10.5 oneDNN (Intel)
Key constraints:
- CPU and Intel GPU optimized kernels
- JIT ISA dispatch at runtime
Risks:
- Different performance on non-Intel CPUs
Mitigation:
- Use oneDNN only when device vendor is Intel or when it outperforms torch baseline

Refs:
- https://github.com/uxlfoundation/oneDNN

### 10.6 Intel Extension for PyTorch (IPEX)
Key constraints:
- Project is in retirement; upstreaming into PyTorch is ongoing
- CPU and Intel GPU optimizations, device type "xpu"
Risks:
- Uncertain long-term maintenance and binary distribution
Mitigation:
- Prefer upstream PyTorch + oneDNN; use IPEX only when explicitly installed

Refs:
- https://github.com/intel/intel-extension-for-pytorch

### 10.7 ZenDNN (AMD)
Key constraints:
- AMD EPYC optimized CPU kernels
- Requires AOCL-BLAS
Risks:
- External dependency setup complexity
Mitigation:
- Detect ZenDNN plugin presence at runtime
- Fall back to torch/oneDNN if absent

Refs:
- https://github.com/amd/ZenDNN

### 10.8 HF Kernel Hub (kernels)
Key constraints:
- Kernels are loaded dynamically from the Hub
- Portable loading outside PYTHONPATH and multiple versions in one process
- Requires ABI3, manylinux_2_28, and unique torch.ops namespace
Risks:
- Version clashes or incompatible kernels
Mitigation:
- Use kernel lockfiles
- Validate ABI and namespace uniqueness

Refs:
- https://github.com/huggingface/kernels
- https://huggingface.co/kernels-community

### 10.9 Triton
Used as a generic backend for custom kernels and Liger. Triton is portable across
CUDA and ROCm and should be treated as a first-class kernel compiler option
where native kernels are unavailable or too rigid for new data types.
Risks:
- Kernel compilation time
- Driver and CUDA version mismatch
Mitigation:
- Cache compiled kernels, prebuild for common CUDA versions

Refs:
- https://github.com/triton-lang/triton
- https://openai.com/research/triton

### 10.10 Tokenization Backends
Key options:
- HF tokenizers (Rust) with offset mapping and fast batch encode/decode
- tiktoken (fast BPE with plugin mechanism via tiktoken_ext)
- SentencePiece (BPE/unigram, NFKC normalization, deterministic model files)
Risks:
- Normalization differences change token ids (must be part of cache keys)
- CPU-bound; can become a bottleneck for high-throughput serving
Mitigation:
- Include tokenizer_id + vocab_hash + normalizer_id in SelectionContext
- Cache tokenization results for repeated prompts and system prompts
- Include merges_hash, added_tokens_hash, and special_tokens_hash in cache keys

Refs:
- https://github.com/huggingface/tokenizers
- https://github.com/openai/tiktoken
- https://github.com/google/sentencepiece

### 10.11 Quantization and Low-Precision Formats
LayerZero should support a super set of datatypes but only dispatch to kernels
that explicitly declare compatibility. For attention and GEMM-heavy ops, include
support for:
- FP16, BF16, FP32/TF32
- INT8 (per-tensor and per-channel)
- FP8 (E4M3 and E5M2)
- MXFP4, NVFP4 (Blackwell and vendor-specific)

Constraints:
- Many kernels require calibration scales or blockwise scales for FP8/FP4.
- Some kernels require specific scale layouts (per-head or blockwise).
- Quantization support is backend-specific and often hardware-gated.

Design requirements:
- Explicit quantization metadata in KernelSpec and SelectionContext.
- Adapter functions must map quantization metadata to backend-specific arguments.
- PerfDB entries must include dtype and quantization metadata for validity.
- Support prepack/pack operations and cache packed weights for reuse.

### 10.12 Habana (HPU)
Expected integration:
- Use habana_frameworks.torch on HPU
- Provide HPU-specific KernelSpec if supported
Risks:
- Limited third-party kernel availability
Mitigation:
- Rely on native HPU ops + torch fallback first

Refs:
- https://docs.habana.ai

### 10.13 Precision Support Matrix (Initial)

This matrix is illustrative; exact availability must be detected at runtime.

| Backend | FP16/BF16 | INT8 | FP8 (E4M3/E5M2) | MXFP4 | NVFP4 |
|---------|-----------|------|----------------|-------|-------|
| FlashAttention | Yes | No | Partial (Hopper/Blackwell, backend-specific) | No | No |
| FlashInfer | Yes | Yes | Yes (SM90+) | Yes (Blackwell) | Yes (Blackwell) |
| xFormers | Yes | No | No | No | No |
| Triton (custom) | Yes | Yes | Yes (if hardware + kernel support) | Yes (if hardware + kernel support) | Yes (if hardware + kernel support) |
| oneDNN | Yes | Yes | No | No | No |
| ZenDNN | Yes | Yes | No | No | No |

## 11. Cross-Hardware and Multi-Version Support

### 11.1 CUDA Multi-Version
Problems:
- Kernel wheels tied to CUDA minor versions
- JIT compile overhead and cache invalidation
Solutions:
- Artifact Manager that resolves per CUDA version
- Support flashinfer-cubin / flashinfer-jit-cache packages
- Store kernel build metadata in PerfDB
- Provide reference container images with pinned backend versions

### 11.2 ROCm
Problems:
- Backend fragmentation (CK vs Triton)
- Limited kernel coverage vs CUDA
Solutions:
- Separate ROCm capability matrix and KernelSpec set
- Prefer Triton-backed kernels where supported

### 11.3 CPU and SIMD
Problems:
- CPU performance heavily ISA-dependent
Solutions:
- Use oneDNN/ZenDNN when available
- Offer portable SIMD kernels (Highway) for custom ops
- Detect ISA at runtime and select SIMD variants

Refs:
- https://github.com/google/highway

### 11.4 Dependency and ABI Management
Problems:
- Conflicting CUDA/ROCm wheels across backends
- Symbol lookup errors on import
Solutions:
- Dynamic imports with error isolation
- Compatibility matrix and reference containers
- Optional subprocess-backed backend execution for incompatible stacks
- Enforce single CUDA/ROCm version policy for in-process backends

## 12. PyTorch Integration

### 12.1 Custom Ops
Register LayerZero ops via torch.library for torch.compile compatibility.

```
torch.library.Library("layerzero", "DEF")
```

Provide CUDA and CPU implementations plus meta kernels.

### 12.2 Interaction with SDPA
PyTorch has multiple SDPA backends (FLASH, EFFICIENT, MATH, CUDNN).
LayerZero should:
- Use torch.nn.attention.sdpa_kernel (torch.backends.cuda.sdp_kernel is deprecated)
- Respect torch.nn.attention.sdpa_kernel context
- Avoid double-dispatch by disabling SDPA backends when using external kernels
- Use can_use_flash_attention / can_use_efficient_attention to build reasons
- Treat attn_mask + is_causal as invalid (PyTorch errors in math backend)
- Encode backend constraints observed in practice:
  - Flash SDPA does not support non-null attn_mask
  - Memory-efficient requires head_dim % 8 == 0
  - cuDNN SDPA requires head_dim <= 128 and head_dim % 8 == 0
  - All fused kernels require stride(-1) == 1
  - GQA (enable_gqa) works for flash + math; mem_efficient fails

### 12.3 CUDA Graph Strict Mode
- Validate graph safety via dummy capture during kernel validation
- Reject kernels that allocate or synchronize during capture
- Cache graph-safe selections for the capture lifetime

Refs:
- https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://pytorch.org/docs/stable/nn.attention.html

## 13. Hugging Face Integration

### 13.1 Transformers
Transformers uses config.attn_implementation. It already falls back to HF Kernel Hub when flash-attn is missing.
Add support for "layerzero" by:
- Providing a minimal wrapper that maps HF QKV layout to LayerZero
- Exposing a "layerzero" attention implementation path
- Respecting the existing HF kernels fallback behavior

Relevant code paths:
- transformers/modeling_utils.py _check_and_adjust_attn_implementation

### 13.2 Diffusers
Diffusers uses AttentionProcessor classes. Provide a LayerZero processor:
- Replaces self-attention and cross-attention calls
- Uses lz.attention with appropriate masks

### 13.3 Pipelines
Expose a pipeline-level option (env var or config) to pick LayerZero.

### 13.4 Tokenization
Integrate with HF tokenization stack via:
- Wrapper around PreTrainedTokenizerBase with LayerZero selection
- Explicit tokenizer_id/vocab_hash mapping for cache keys
- Fallback to the native tokenizer when LayerZero is disabled

## 14. Operational Concerns

### 14.1 Observability
- Structured selection logs
- SelectionReport for explainability
- Health metrics per kernel (failures, fallbacks)
- Optional OpenTelemetry spans for selection and dispatch
- Plan-level selection summaries (when plan_mode enabled)

### 14.2 Production Safety
- Always provide correct fallback
- Provide deterministic mode flag
- Allow runtime disable of LayerZero
- Circuit breaker on CUDA errors (disable backend for process lifetime)
- Memory-aware selection for high-pressure scenarios (avoid OOM-prone kernels)

### 14.3 Security
- Allowlist kernel sources
- Optional signature verification for kernel hub downloads

### 14.4 Debug Mode and Async Errors
- `LAYERZERO_DEBUG=1` forces `torch.cuda.synchronize()` after dispatch
- Helps pinpoint illegal memory access and async backend failures

### 14.5 Multi-Device and Stream Safety
- All wrappers must respect the current CUDA stream
- Selection cache must be thread-safe (lock-free or shard locks)

### 14.6 Verification and Fuzzing
- Continuous fuzzing against PyTorch reference for shapes/dtypes/masks
- Enforce rtol/atol tolerances in CI and nightly tests

### 14.7 Backend Isolation Modes
- Default: in-process dynamic import with graceful disablement
- Optional: subprocess-backed execution for conflicting CUDA/ABI stacks
- IPC boundary must preserve tensor metadata and error reporting

### 14.8 Critical Challenges (gap.md) Alignment
- Selection overhead: plan/bake decisions and optional compiled dispatch path
- Dependency conflicts: enforce single CUDA/ROCm policy or isolate via subprocess
- JIT latency: mandatory warmup + prebuilt caches; solver triggers JIT for buckets
- CUDA graph safety: validate via graph strict mode before deployment
- Dynamic shapes: dispatch tables must be bucketed decision trees

### 14.9 Production Readiness Checklist
- Capabilities descriptors validated (schema_version supported)
- Dispatch table generated for production shapes (lz.solve)
- Warmup completed and caches persisted
- Graph strict mode validated if using CUDA graphs
- Memory headroom configured and enforced
- Backend isolation policy applied (single CUDA or subprocess)

### 14.10 Risk Register (Top Risks)
- Python selection overhead → baked plans + compiled dispatch path
- ABI conflicts → single CUDA policy + container images
- JIT spikes → solver precompile + persistent caches
- Graph unsafe kernels → dummy capture validation
- Capability drift → CI capability matrix and schema validation

### 14.11 Readiness Check
- `lz.readiness_check()` validates caches, dispatch tables, and backend schemas
- Returns a structured report for deployment gating

## 15. Scenarios and Failure Modes

### 15.1 Happy Path
- Kernel available, constraints satisfied, perfdb hit -> fast selection

### 15.2 Worst Case
- Kernel installed but fails at runtime due to driver mismatch
  - Solution: mark unhealthy, fallback, emit warning
- Backend import fails due to CUDA/ABI conflicts
  - Solution: dynamic import + backend isolation; provide reference container builds
- JIT compile stalls latency-critical inference
  - Solution: precompile or disable JIT for production
- Kernel returns incorrect results due to silent constraint mismatch
  - Solution: strict KernelSpec.check and validation tests
- Selection overhead regresses tiny-kernel latency
  - Solution: fast-path cache and optional baked selection

### 15.3 Outliers
- Ambiguous layout detection (S == H)
  - Solution: require explicit layout param or shape hints
- Very long sequences ( > 32k )
  - Solution: enforce max_seq_len constraints and fallback
- Mixed devices in a batch (CPU + GPU)
  - Solution: per-device selection and explicit error if mixed
- Mixed dtype Q/K/V or kv-cache dtype mismatch
  - Solution: explicit cast or dequant step in ExecutionPlan
- Nested tensors / ragged batches
  - Solution: require explicit nested-tensor support or fallback
- Memory pressure causing fallback or OOM
  - Solution: memory-aware scoring and preflight allocation checks

### 15.4 Observed Edge Cases (Local + Docs)
- Non-null attn_mask disables Flash SDPA; memory-efficient and cuDNN may work
- attn_mask + is_causal causes errors in the math backend (invalid combination)
- GQA (enable_gqa) disables memory-efficient SDPA; flash + math work
- head_dim > 256 disables Flash; cuDNN SDPA requires head_dim <= 128
- head_dim not divisible by 8 disables memory-efficient and cuDNN
- stride(-1) != 1 disables all fused SDPA kernels (flash/mem_efficient/cuDNN)
- xFormers attn_bias must be on-device and cannot broadcast batch/head dims
- FlashInfer layout mismatches can surface as backend errors (strict adapters needed)
- Paged KV metadata dtype mismatch can cause hard failures

### 15.5 Practical Challenges
- API drift (torch SDPA, xFormers, FlashInfer) requires strict version gating
- JIT compilation spikes (FlashInfer/Triton) must be amortized or prebuilt
- Tokenization normalization differences can silently change token ids
- Kernel health needs per-op isolation to avoid cascading failures
- Co-installing conflicting CUDA wheels requires containerization guidance

## 16. Local Experiments (RTX 3080)

Environment:
- GPU: NVIDIA GeForce RTX 3080 (SM 8.6)
- Torch: 2.9.1+cu126
- flashinfer: 0.5.3
- xformers: 0.0.33.post2 (installed in a venv with system torch)

### 16.1 SDPA Baseline (B=1, H=16, S=1024, D=128, fp16, causal)
Results (median ms, p95 ms):
- sdpa_default: 0.247, 0.464
- sdpa_flash: 0.373, 0.529
- sdpa_mem_efficient: 0.230, 0.414
- sdpa_math: 2.353, 2.584
- xformers_mem_eff: 0.877, 0.977
- flashinfer_prefill: 0.381, 0.798
- flashinfer_vs_sdpa max abs diff: 0.001953125

Additional runs:
- S=4096 (fp16): sdpa_default 1.657, sdpa_flash 1.576, sdpa_mem_efficient 2.709, sdpa_math 28.333
- GQA (H=16, Hkv=4): SDPA fused kernels disabled; flashinfer ok; xFormers failed without explicit broadcast
- BF16: flashinfer_vs_sdpa max abs diff: 0.0078125
- Noncontig with stride(-1)=1 works; stride(-1)!=1 disables fused kernels

Takeaways:
- SDPA backend choice matters (math is much slower).
- flashinfer correctness is close to SDPA within fp16/bf16 tolerances.
- xFormers is slower for this shape on this GPU.

### 16.2 SDPA Edge Cases (B=1, H=16, S=512, D=128, fp16)
Results (median ms, p95 ms unless noted):
- baseline: flash 0.075/0.078, mem_efficient 0.093/0.095, math 0.465/0.477, cudnn 0.080/0.083
- mask_bool: flash error (attn_mask unsupported), mem_efficient 0.160/0.168, math 0.581/0.688, cudnn 0.111/0.113
- mask_float: flash error, mem_efficient 0.115/0.119, math 0.488/0.692, cudnn 0.083/0.084
- mask + is_causal: math error; mem_efficient/cudnn ok; flash error
- dropout_p=0.1: flash 0.082/0.083, mem_efficient 0.118/0.122, math 0.575/0.674, cudnn 0.087/0.130
- GQA enable: flash ok, mem_efficient error, math ok, cudnn ok
- head_dim=320: flash error, cudnn error, mem_efficient/math ok
- head_dim=84: mem_efficient error (head_dim % 8), cudnn error, flash/math ok
- stride(-1)=2: flash/mem_efficient/cudnn error; math ok
- CUDA Graph capture: default + non-default stream ok for this case

Notes:
- torch.backends.cuda.sdp_kernel is deprecated; use torch.nn.attention.sdpa_kernel.

### 16.3 Tokenization Micro-Benchmark (CPU)
Synthetic dataset: 2000 short strings, 10x repeated sentence, vocab_size=100.
Results (median ms, p95 ms):
- tiktoken.encode_batch: 59.70, 61.23
- tokenizers.encode_batch: 48.51, 49.94
- sentencepiece.encode: 234.94, 236.20

Notes:
- Results depend on vocab, model, and normalization; use only for relative behavior.

### 16.4 Selection Overhead Micro-Benchmark (CPU)
Results (us/call):
- dict.get (hit): 0.126
- dict.get (miss): 0.245
- make key: 0.031

GPU op baseline:
- gpu op (no sync per call): 8.611 us/call
- gpu op (sync each call): 22.877 us/call

Notes:
- Debug-mode `synchronize` adds ~14 us/call for tiny kernels.
- Selection overhead is small but can matter for microkernels.

### 16.5 FlashInfer JIT Compilation (CUDA)
Results (first call vs second call):
- shape1 B=1 H=16 S=1024 D=128: 8.706 ms vs 0.095 ms
- shape2 B=1 H=16 S=2048 D=128: 0.212 ms vs 0.116 ms
- shape3 B=1 H=16 S=1024 D=192: 12426.725 ms vs 0.225 ms

Notes:
- New shapes (e.g., head_dim=192) can trigger multi-second JIT compile.
- Warmup and persistent cache are mandatory for production.

### 16.6 Context Build Overhead (CUDA tensors)
Results (us/call):
- build_ctx no mask: 2.548
- build_ctx bool mask: 2.678

Notes:
- Context construction is microseconds per call; cache hits are sub-microsecond.
- Baked selection removes context build in static workloads.

## 17. Implementation Plan (Phased)

Phase 1: Core framework
- OperationSpec, KernelSpec, BackendSpec, SelectionContext
- Selection engine, policy, cache, logging

Phase 2: CUDA backends
- FlashAttention (FA2/FA3) adapters
- FlashInfer adapters (prefill, decode, paged)
- xFormers adapter

Phase 3: Norms and CPU
- Liger RMSNorm/LayerNorm
- Torch fallback
- oneDNN/ZenDNN detection

Phase 4: PerfDB + tuning
- SQLite schema and tuning CLI
- Runtime auto-tune (optional)

Phase 5: PyTorch + HF integration
- torch.library custom ops
- HF Transformers and diffusers adapter

Phase 6: Extended hardware
- ROCm kernels (FlashAttention ROCm, Triton)
- Habana HPU adapter
- SIMD portable kernels (Highway)

## 18. Open Questions

- Which kernel hub repositories should be trusted by default?
- How aggressively should LayerZero override PyTorch SDPA backends?
- What is the minimal kernel set required for first production rollout?
- How to handle ABI drift across CUDA minor versions in third-party wheels?

## 19. References

- FlashAttention repo and papers: https://github.com/Dao-AILab/flash-attention
- FlashAttention v1 paper: https://arxiv.org/abs/2205.14135
- FlashAttention v2 paper: https://tridao.me/publications/flash2/flash2.pdf
- FlashAttention v3 paper: https://tridao.me/publications/flash3/flash3.pdf
- FlashInfer: https://github.com/flashinfer-ai/flashinfer
- FlashInfer docs: https://docs.flashinfer.ai
- xFormers: https://github.com/facebookresearch/xformers
- Liger Kernel: https://github.com/linkedin/Liger-Kernel
- Liger Kernel report: https://arxiv.org/pdf/2410.10989
- oneDNN: https://github.com/uxlfoundation/oneDNN
- Intel Extension for PyTorch (IPEX): https://github.com/intel/intel-extension-for-pytorch
- ZenDNN: https://github.com/amd/ZenDNN
- HF kernels hub: https://github.com/huggingface/kernels
- HF kernels community: https://huggingface.co/kernels-community
- HF kernels requirements: https://github.com/huggingface/kernels/blob/main/docs/source/kernel-requirements.md
- Triton: https://github.com/triton-lang/triton
- OpenAI Triton paper: https://openai.com/research/triton
- HF tokenizers: https://github.com/huggingface/tokenizers
- tiktoken: https://github.com/openai/tiktoken
- SentencePiece: https://github.com/google/sentencepiece
- PyTorch SDPA: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch attention utils: https://pytorch.org/docs/stable/nn.attention.html
- Highway SIMD: https://github.com/google/highway
- Habana docs: https://docs.habana.ai
