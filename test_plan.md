# LayerZero End-to-End Test Specification

Version: 1.1
Status: Draft for review (Updated: 2026-01-16)

> **Note:** For detailed TDD test plans for each implementation task, see [tasks.md](./tasks.md) which contains 58 tasks with 400+ test cases.

## 1. Goals and Quality Gates

Primary goals:
- Correctness across ops and backends
- Stable latency and throughput in production-like workloads
- Safe fallback and clear diagnostics on failures
- Portability across hardware/software matrices

Release gates:
- Zero correctness regressions vs reference
- Performance within thresholds or explicit waivers
- No crash, leak, or health breaker regressions in soak tests
- Capabilities and dispatch tables validate against schema

## 2. Scope

In scope:
- Tokenization, positional ops, attention (prefill/decode/paged), MLP/GEMM, norms,
  sampling, quantization, and dtype conversions
- Backends: FlashAttention, FlashInfer, xFormers, Triton/Liger, oneDNN, ZenDNN,
  HF kernels, and PyTorch fallbacks
- CUDA graphs and torch.compile integration paths

Out of scope:
- End-user application correctness beyond kernel orchestration
- Non-deterministic training correctness (forward-only is primary)

## 3. Test Environment Matrix

Hardware (detailed):
- NVIDIA:
  - SM80 (A100), SM86 (A10/RTX 30), SM89 (L4/RTX 40), SM90 (H100), SM100+ (when available)
  - Verify BF16/FP16/TF32/FP8 support per device and driver
- AMD ROCm:
  - MI200 (gfx90a), MI300 (gfx940/gfx942)
- CPU:
  - Intel Ice Lake (AVX-512), Sapphire Rapids (AMX BF16/INT8)
  - AMD EPYC Zen3 (AVX2), Zen4 (AVX-512)
- Habana:
  - Gaudi2/Gaudi3 where supported

Software (detailed):
- OS: Ubuntu 20.04/22.04, Rocky 8/9 (container and bare metal)
- Python: 3.9, 3.10, 3.11, 3.12
- PyTorch: 2.1-2.4 (stable) plus a nightly lane for early warnings
- CUDA: 11.8, 12.1, 12.4; cuDNN 8/9 where applicable
- ROCm: 5.7, 6.0
- Triton: 2.1, 2.2, 2.3
- Backend versions pinned to the compatibility matrix

Backend isolation modes:
- in_process (single CUDA/ROCm policy)
- subprocess (conflicting stacks)

Precision:
- FP32, TF32, FP16, BF16, INT8, FP8 (E4M3/E5M2), FP4 variants

## 4. Reference Implementations

Correctness baseline:
- PyTorch reference ops for attention, norms, GEMM, and sampling
- Tokenizers: HF tokenizers, tiktoken, sentencepiece for their own outputs

All correctness tests compare against reference outputs with tolerances below.

## 5. Tolerance Table

| Op Category | FP16 | BF16 | FP32 | INT8/FP8/FP4 |
|-------------|------|------|------|-------------|
| Attention   | rtol=1e-3, atol=1e-3 | rtol=1e-2, atol=1e-2 | rtol=1e-5, atol=1e-5 | exact or rtol=1e-2 where defined |
| Norms       | rtol=1e-3, atol=1e-3 | rtol=1e-2, atol=1e-2 | rtol=1e-6, atol=1e-6 | exact for int8 where applicable |
| GEMM/MLP    | rtol=1e-3, atol=1e-3 | rtol=1e-2, atol=1e-2 | rtol=1e-6, atol=1e-6 | backend-defined error bounds |
| RoPE/ALiBi  | rtol=1e-4, atol=1e-4 | rtol=1e-3, atol=1e-3 | rtol=1e-6, atol=1e-6 | not applicable |
| Sampling    | deterministic with fixed seed | deterministic with fixed seed | deterministic with fixed seed | deterministic with fixed seed |
| Tokenizers  | exact match | exact match | exact match | exact match |

Notes:
- For quantized backends, specify exact bounds per backend in capabilities.json.
- Sampling is validated by deterministic seed and distribution invariants.

## 6. Core End-to-End Scenarios

### 6.1 LLM Prefill + Decode (Causal)
Inputs:
- Vary seq_len: [128, 512, 2048, 8192, 32768]
- Heads: H=32, Hkv=8 (GQA)
- Dtypes: fp16, bf16

Pass criteria:
- Output matches reference within tolerance
- Kernel selection uses valid backends for each shape
- No fallback for supported shapes when preferred backends available

### 6.2 Paged KV Attention
Inputs:
- Paged KV cache with variable context_lens
- block_tables dtype int32 and int64

Pass criteria:
- Output matches reference within tolerance
- KV metadata dtype mismatches are rejected with clear reason codes

### 6.3 Tokenization to Sampling Pipeline
Steps:
1) Tokenize input texts with tokenizer_id + hashes
2) Run attention + MLP + norm blocks
3) Sample with top-k/top-p

Pass criteria:
- Tokenization ids and offsets match reference exactly
- Sampling deterministic with fixed seed
- All caches and dispatch tables validated

### 6.4 Diffusers Cross-Attention
Inputs:
- Cross-attention masks, long context

Pass criteria:
- Output matches reference within tolerance
- Mask + is_causal is rejected where invalid

### 6.5 Quantized LLM Pipeline
Inputs:
- INT8/FP8/FP4 quantized weights and activations
- Per-channel and per-block scales

Pass criteria:
- Outputs match quantized reference within defined bounds
- Scale layout mismatches produce clear reason codes

### 6.6 RoPE/ALiBi + Attention Fusion
Inputs:
- RoPE interleaved vs non-interleaved layout
- ALiBi bias with long context

Pass criteria:
- Fused kernels match reference within tolerance
- Non-fused fallback matches reference when fusion unavailable

### 6.7 CPU-Only Fallback Pipeline
Inputs:
- Tokenization + attention + MLP + norms on CPU only

Pass criteria:
- Correct fallback selection across all ops
- No GPU-specific code paths executed

### 6.8 Tokenizer Offsets and Special Tokens
Inputs:
- Texts with unicode normalization edge cases
- Added special tokens and custom vocab merges

Pass criteria:
- Exact token ids and offsets vs reference tokenizer
- Hash mismatch invalidates cache entries

### 6.9 Attention Bias and Sparse Fallbacks
Inputs:
- ALiBi or relative bias masks
- Block-sparse masks (if not supported, expect fallback)

Pass criteria:
- Unsupported bias/mask paths fall back safely
- Outputs match reference within tolerance

### 6.10 Multi-Model Isolation
Inputs:
- Two models with different tokenizer_id and vocab/merges hashes
- Interleaved requests across models

Pass criteria:
- No cache contamination across models
- Selection and tokenization results remain model-specific

### 6.11 Mixed Sequence Lengths in Batch
Inputs:
- Ragged batch with variable sequence lengths
- Nested tensors where supported (or explicit fallback)

Pass criteria:
- Correct outputs or explicit fallback where nested inputs are unsupported
- Clear error messages for unsupported ragged formats

### 6.12 KV Cache Dtype and Layout Variations
Inputs:
- kv_cache dtype mismatch (fp16 vs bf16)
- kv_cache layout differences (NHD/HND vs backend expectation)

Pass criteria:
- Explicit dequant/cast when allowed
- KV cache mismatch rejected with reason codes

### 6.13 Encoder-Decoder (T5/BART) Cross-Attention
Inputs:
- Encoder seq_len: [128, 512, 2048]
- Decoder with kv-cache enabled and disabled
- Padding masks with variable lengths

Pass criteria:
- Output matches reference within tolerance
- Cross-attention kernels selected with correct mask handling
- Encoder/decoder cache separation is preserved

### 6.14 Vision Transformer and Multimodal Blocks
Inputs:
- Patch embedding + attention + MLP
- Channels-last and contiguous inputs

Pass criteria:
- Correct outputs for vision encoder blocks
- Layout-sensitive kernels selected only when constraints are met

### 6.15 Non-Attention Kernels (Embedding, Norm, MLP)
Inputs:
- Embedding lookup + RMSNorm/LayerNorm + MLP (GELU/SiLU)
- Mixed dtype activations

Pass criteria:
- Kernel selection covers non-attention ops
- Fused kernels used when available, fallback otherwise

### 6.16 MoE Gating and Expert Dispatch
Inputs:
- Top-k gating with varying expert counts
- Grouped GEMM paths where available

Pass criteria:
- Deterministic expert routing with fixed seed
- Dispatch kernels selected within tolerance and memory bounds

### 6.17 Speculative Decoding
Inputs:
- Draft + target model pair
- Acceptance rate stress (low/high)

Pass criteria:
- Per-model kernel selection remains isolated
- Acceptance logic yields correct token sequences

### 6.18 Sliding Window and Chunked Attention
Inputs:
- Window sizes [256, 1024]
- Long context with sliding window masks

Pass criteria:
- Correct outputs or explicit fallback for unsupported windows
- No illegal mask/bias combinations accepted

### 6.19 Positional Encoding Variants
Inputs:
- RoPE scaling (NTK, YaRN), XPos
- Interleaved vs non-interleaved layouts

Pass criteria:
- Correct outputs within tolerance for each variant
- Unsupported variants trigger clear fallback

## 7. Selection and Planning Tests

### 7.1 Plan-Aware Selection
Pass criteria:
- Fused ops selected when available and beneficial
- Transform penalties steer selection to lower total cost
- SelectionReport includes transform_cost_total and fused_ops

### 7.2 Baked Plan Bucketing
Pass criteria:
- lz.solve emits decision tree for bucketed seq_len/batch
- Bucket miss triggers runtime selection with PLAN_BUCKET_MISS reason

### 7.3 Strict Mode vs Lenient Mode
Pass criteria:
- strict_mode rejects implicit layout/dtype transforms
- allow_adaptation performs transforms with explicit cost accounting

### 7.4 Policy and Locking
Pass criteria:
- prefer/avoid sources honored deterministically
- explicit locks override scoring
- policy_hash changes invalidate cached selections

### 7.5 Cache Invalidation
Pass criteria:
- capabilities_hash changes invalidate cached selections
- backend version changes invalidate PerfDB entries

### 7.6 Plan and Bucket Miss Behavior
Pass criteria:
- PLAN_BUCKET_MISS triggers runtime selection
- Runtime selection honors strict_mode and policy rules

### 7.7 PerfDB Integrity and Outlier Handling
Pass criteria:
- Corrupted or partial PerfDB entries are ignored and rebuilt
- High-variance measurements are down-weighted or excluded

### 7.8 Cached Plan vs Runtime Drift
Pass criteria:
- Hardware signature or driver change invalidates baked plans
- Policy hash mismatch triggers re-solve

## 8. CUDA Graph and torch.compile Tests

### 8.1 Graph Strict Mode
Pass criteria:
- Kernels that allocate or synchronize are rejected
- Dummy capture validation passes for graph-safe kernels

### 8.2 torch.compile Compatibility
Pass criteria:
- No graph breaks in LayerZero ops
- Meta kernels produce correct shapes and dtypes

### 8.3 Multi-Stream Graph Capture
Pass criteria:
- Graph capture succeeds on non-default streams
- Graph-safe selections are cached for capture lifetime

### 8.4 Graph Safety Negative Tests
Pass criteria:
- Kernels that allocate or synchronize are rejected in graph_strict_mode
- Clear error when no graph-safe kernel exists

### 8.5 torch.export and AOT Compatibility
Pass criteria:
- Export graphs include correct meta kernels and shapes
- Runtime selection honors exported constraints or fails clearly

## 9. Dependency and Isolation Tests

### 9.1 Single CUDA Policy
Pass criteria:
- In-process mode rejects incompatible CUDA versions
- Clear diagnostics for ABI conflicts

### 9.2 Subprocess Isolation
Pass criteria:
- Conflicting backends can run concurrently in subprocess mode
- IPC preserves tensor metadata and error reporting

### 9.3 Compatibility Matrix Validation
Pass criteria:
- Reported compatibility matches actual backend availability
- Mismatches are flagged before deployment

### 9.4 Multi-Backend Coexistence (CUDA/ROCm/CPU)
Pass criteria:
- Device-specific backends never cross devices
- Clear diagnostics when incompatible backends are loaded together

## 10. JIT and Cache Tests

### 10.1 Warmup Coverage
Pass criteria:
- lz.warmup compiles all required kernels for specified shapes
- No JIT compilation on first request after warmup

### 10.2 Persistent Cache
Pass criteria:
- JIT cache reused across restarts
- Cache invalidated on backend version changes

### 10.3 Multi-Process Cache Sharing
Pass criteria:
- Multiple worker processes reuse shared cache without recompiling
- Cache contention does not corrupt artifacts

### 10.4 Bucketed Warmup Coverage
Pass criteria:
- lz.solve triggers JIT for all bucketed shapes
- No compilation on first request within a bucket

### 10.5 JIT Compile Failure Fallback
Pass criteria:
- Failed compilation disables the kernel cleanly
- Fallback path executes without repeated failures

## 11. Memory and Stability Tests

### 11.1 Memory Headroom
Pass criteria:
- Selection avoids kernels exceeding headroom
- MEMORY_HEADROOM_EXCEEDED reason recorded

### 11.2 Soak Test (24-72h)
Pass criteria:
- No memory leaks or unbounded cache growth
- Backend health circuit breaker behaves as expected

### 11.3 Memory Fragmentation Stress
Pass criteria:
- Repeated allocations do not degrade performance or cause unexpected OOM
- Selection avoids kernels with large temporary workspaces

### 11.4 OOM Fallback
Pass criteria:
- On OOM, selection falls back to a lower-memory kernel when possible
- Clear error when no safe fallback exists

### 11.5 External Memory Pressure
Pass criteria:
- Selection adapts when other processes consume device memory
- No unbounded retry loops under pressure

## 12. Observability and Diagnostics

Pass criteria:
- Selection logs include kernel_id, reasons, transform_cost_total
- OTel spans for selection and dispatch (if enabled)
- lz.readiness_check reports schema validity and cache status

Additional checks:
- lz.which/lz.explain output consistent with selection results
- lz.list_kernels and lz.doctor report correct backend status
- Logs are redacted for sensitive tokens when configured

## 13. Failure Injection

Scenarios:
- Backend import failure
- CUDA illegal memory access
- Invalid capabilities.json or dispatch_table.json
- attn_mask + is_causal invalid combination
- Graph unsafe kernel selected under graph_strict_mode
- PLAN_BUCKET_MISS on unseen shape
- KV cache metadata dtype/layout mismatch
- Tokenizer hash mismatch (vocab/merges/special tokens)
- PerfDB corruption or missing entries
- Backend hang/timeout and recovery
- Device lost/reset during execution

Pass criteria:
- Backend disabled safely
- Fallback triggered with clear reason codes
- Readiness check fails fast for invalid schemas
- Invalid inputs rejected with explicit reason codes

## 14. Test Artifacts and Outputs

Artifacts:
- capabilities.json and hash
- dispatch_table.json and schema validation report
- perfdb snapshots
- selection logs and telemetry traces

## 15. Security and Supply Chain

Scenarios:
- Kernel hub allowlist enforcement
- Lockfile pinning validation
- Optional signature verification (when supported)
- Path traversal attempts in hub downloads
- Artifact tamper detection (checksums)

Pass criteria:
- Unauthorized kernels are rejected
- Lockfile violations block loading

## 16. Concurrency and Thread Safety

Scenarios:
- Multi-threaded selection cache access under high QPS
- Multi-stream CUDA execution with concurrent ops
- Forked workers after backend initialization

Pass criteria:
- No data races or corrupted cache entries
- Selection remains deterministic under identical inputs

## 17. Distributed and Multi-Device

Scenarios:
- Rank0 broadcast selection for homogeneous hardware
- Capability-group selection for heterogeneous ranks
- Mixed-device inputs to single op (must be rejected)
- Tensor-parallel attention with per-rank shapes

Pass criteria:
- Consistent selection across ranks or capability groups
- Mixed-device inputs fail with clear diagnostics

## 18. Performance Regression and Overhead

Scenarios:
- Selection overhead micro-bench under load
- Baked plan vs runtime selection comparison
- PerfDB timing influence under varying variance
- Cold-start latency with and without warmup

Pass criteria:
- Hot-path selection stays within budget
- Baked plans outperform runtime selection on steady workloads
- PerfDB influence reduced when variance is high

## 19. Policy and Configuration Tests

Scenarios:
- Environment variable overrides
- Runtime config changes (prefer/avoid/locks)
- strict_mode and plan_mode toggles
- Per-request overrides and timeboxed policies

Pass criteria:
- Policy changes invalidate caches
- Selection follows updated policy deterministically
- strict_mode rejects implicit transforms when enabled

## 20. Schema and Compatibility Tests

Scenarios:
- capabilities.json schema mismatch
- dispatch_table.json schema mismatch
- hardware signature mismatch for dispatch tables
- schema fuzzing with random missing/extra fields
- capabilities schema version migration

Pass criteria:
- Fail closed with clear errors
- Readiness check detects invalid schemas

## 21. Plugin and Hub Tests

Scenarios:
- entry_points discovery for custom backends
- HF kernel hub download + namespace uniqueness
- Plugin upgrade/downgrade compatibility

Pass criteria:
- Plugins register without user code changes
- Multiple versions load without namespace clashes

## 22. Mixed Precision and Quantization Tests

Scenarios:
- Mixed dtype Q/K/V and kv-cache dtype mismatch
- Per-channel and per-block scale layouts
- Packed weights cache reuse
- FP4/NF4/INT4 mixed precision paths

Pass criteria:
- Explicit casts or dequant steps applied as required
- Scale layout mismatches rejected with clear reasons
- Packed weights reused without re-pack per call

## 23. Layout and Stride Tests

Scenarios:
- stride(-1) != 1 inputs
- BHSD/BSHD ambiguity cases (S == H)
- attn_mask types (bool vs float) and broadcast rules
- View/transpose inputs with non-contiguous storage

Pass criteria:
- Fused kernels rejected when stride constraints fail
- Ambiguous layouts require explicit hints or fallback
- Mask broadcast violations produce clear errors

## 24. Readiness Gate Tests

Scenarios:
- Missing JIT caches with production config
- Invalid capabilities or dispatch table
- Missing required kernels for configured policy

Pass criteria:
- lz.readiness_check fails fast with actionable report

## 25. Exit Criteria

Release candidate is accepted when:
- All mandatory scenarios pass on at least one target per hardware class
- No correctness regressions beyond tolerance
- Performance within specified thresholds or waived with justification
- Readiness check is green on production configuration
- No open P0 or P1 issues in tracking system
