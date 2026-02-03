# Kernel Dispatch System Test Scenarios

## Overview

This document provides 150+ comprehensive test scenarios for the LayerZero kernel dispatch system, covering:
- Normal operations
- Edge cases
- Failure scenarios
- Concurrency scenarios
- Performance scenarios

Each scenario includes situation description, expected behavior, potential failure modes, and mitigation strategies.

---

## 1. Normal Operations (35 Scenarios)

### 1.1 Single Kernel Execution

#### NO-001: Basic Single Kernel Dispatch
- **Situation**: Single attention kernel dispatch with standard fp16 input on CUDA device with batch_size=4, seq_len=512, head_dim=64.
- **Expected Behavior**: SelectionEngine finds FlashAttention kernel, builds ExecutionPlan with kernel_id, returns plan with cached=False on first call.
- **Potential Failure Modes**:
  - No kernels registered for operation
  - All kernels filtered out by constraints
- **Mitigation Strategies**:
  - Ensure default kernels are registered at startup
  - Provide fallback SDPA kernel with minimal constraints

#### NO-002: Single Kernel with Causal Mask
- **Situation**: Attention operation with is_causal=True, seq_len_q=512, seq_len_k=512.
- **Expected Behavior**: Operation mapped to "attention.causal", causal-optimized kernel selected.
- **Potential Failure Modes**:
  - Causal kernel not available, falls back to masked full attention
- **Mitigation Strategies**:
  - Register causal variants for all attention backends

#### NO-003: Single RMSNorm Kernel
- **Situation**: RMSNorm operation on tensor shape (8, 4096, 4096).
- **Expected Behavior**: norm.rms operation resolved, Triton or CUDA fused norm kernel selected.
- **Potential Failure Modes**:
  - Hidden dimension not power of 2
- **Mitigation Strategies**:
  - Pad or use scalar fallback for non-power-of-2 dims

### 1.2 Batched Kernel Execution

#### NO-004: Batch of Identical Contexts
- **Situation**: 16 identical SelectionContexts submitted via select_batch().
- **Expected Behavior**: First context triggers selection, remaining 15 hit cache. Total selection time < 2x single selection.
- **Potential Failure Modes**:
  - Cache key collision with different context
- **Mitigation Strategies**:
  - Include all relevant fields in cache_key() hash

#### NO-005: Batch of Varying Batch Sizes
- **Situation**: Batch of contexts with batch_size varying 1, 2, 4, 8, 16, 32.
- **Expected Behavior**: Each unique batch_size triggers separate selection. Cache key includes batch_size.
- **Potential Failure Modes**:
  - Same kernel selected inefficiently for all sizes
- **Mitigation Strategies**:
  - Bucketing in dispatch table ensures shape-appropriate selection

#### NO-006: Batch with Mixed Operations
- **Situation**: Batch containing attention.causal, attention.full, norm.rms operations.
- **Expected Behavior**: Each operation type gets independent selection from its candidate pool.
- **Potential Failure Modes**:
  - Cross-operation interference in cache
- **Mitigation Strategies**:
  - Operation is part of cache key

### 1.3 Sequential Kernel Chain

#### NO-007: Attention-Norm-MLP Chain
- **Situation**: Transformer layer: attention -> layernorm -> mlp -> layernorm.
- **Expected Behavior**: Each operation selects best kernel independently. Total latency = sum of individual kernel latencies.
- **Potential Failure Modes**:
  - Layout mismatch between kernels requiring transform
- **Mitigation Strategies**:
  - pre_transforms/post_transforms in ExecutionPlan handle layout conversion

#### NO-008: Cross-Attention Followed by Self-Attention
- **Situation**: Encoder-decoder model with cross attention then self attention.
- **Expected Behavior**: Different operation IDs (attention.cross, attention.causal) route to appropriate kernels.
- **Potential Failure Modes**:
  - Cross attention kernel unavailable
- **Mitigation Strategies**:
  - Fallback to full attention with explicit mask

#### NO-009: Fused MLP Kernel Chain
- **Situation**: SwiGLU activation requiring gate * silu(up) pattern.
- **Expected Behavior**: Fused kernel selected if available, otherwise sequential kernels.
- **Potential Failure Modes**:
  - Fused kernel has different memory layout requirements
- **Mitigation Strategies**:
  - fuses_ops field in KernelSpec indicates fusion capability

### 1.4 Parallel Kernel Execution

#### NO-010: Parallel Attention Heads
- **Situation**: Multi-query attention with num_heads=32, num_kv_heads=8 (GQA 4:1).
- **Expected Behavior**: GQA-aware kernel selected (supports_gqa=True). Head dimension broadcast handled internally.
- **Potential Failure Modes**:
  - Non-GQA kernel selected causing incorrect broadcast
- **Mitigation Strategies**:
  - enable_gqa flag in context triggers GQA filter

#### NO-011: Tensor Parallel Attention
- **Situation**: Attention with tp_size=4, each rank processes 1/4 of heads.
- **Expected Behavior**: Kernel selection accounts for reduced local head count. No cross-rank communication in kernel.
- **Potential Failure Modes**:
  - Kernel assumes full head count
- **Mitigation Strategies**:
  - Pass local head count, not global, to kernel

#### NO-012: Pipeline Parallel Stages
- **Situation**: pp_size=8, rank=3 executing middle pipeline stage.
- **Expected Behavior**: Selection identical to single-GPU case. Pipeline scheduling is external.
- **Potential Failure Modes**:
  - Rank-specific kernel filtering
- **Mitigation Strategies**:
  - Pipeline parallelism is transparent to kernel selection

### 1.5 Different Input Sizes

#### NO-013: Small Sequence (seq_len=16)
- **Situation**: Very short sequence, typical for decode phase.
- **Expected Behavior**: Decode-optimized kernel selected over prefill kernel.
- **Potential Failure Modes**:
  - Kernel has min_seq_len > 16
- **Mitigation Strategies**:
  - Ensure at least one kernel supports min_seq_len=1

#### NO-014: Large Sequence (seq_len=128K)
- **Situation**: Long context inference with 128K tokens.
- **Expected Behavior**: Kernel with max_seq_len >= 128K selected. May require chunked/tiled implementation.
- **Potential Failure Modes**:
  - No kernel supports such length
  - OOM during kernel execution
- **Mitigation Strategies**:
  - FlashInfer supports up to 2M tokens
  - Workspace allocation check before execution

#### NO-015: Large Batch Size (batch=256)
- **Situation**: High-throughput serving with batch_size=256.
- **Expected Behavior**: Batch-optimized kernel selected. Memory for workspace pre-allocated.
- **Potential Failure Modes**:
  - Kernel max_batch_size exceeded
- **Mitigation Strategies**:
  - Split batch and merge results

#### NO-016: Tiny Head Dimension (head_dim=32)
- **Situation**: Model with small head dimension.
- **Expected Behavior**: Kernel supporting min_head_dim <= 32 selected.
- **Potential Failure Modes**:
  - Vectorization inefficient for small dims
- **Mitigation Strategies**:
  - Scalar fallback or padding

#### NO-017: Large Head Dimension (head_dim=256)
- **Situation**: Mamba-style model with large head dimension.
- **Expected Behavior**: Kernel with max_head_dim >= 256 selected.
- **Potential Failure Modes**:
  - Register pressure on GPU
- **Mitigation Strategies**:
  - Tiled implementation for large head dims

### 1.6 Various Data Types

#### NO-018: FP16 Precision
- **Situation**: Standard fp16 inference.
- **Expected Behavior**: fp16 kernel selected, native tensor core utilization.
- **Potential Failure Modes**: None expected for common case.
- **Mitigation Strategies**: Default path, well tested.

#### NO-019: BF16 Precision
- **Situation**: BF16 input on Ampere+ GPU.
- **Expected Behavior**: BF16 kernel selected for better range than fp16.
- **Potential Failure Modes**:
  - Pre-Ampere GPU doesn't support bf16
- **Mitigation Strategies**:
  - SM version check (min_sm >= 8.0)

#### NO-020: FP32 Precision
- **Situation**: Full precision inference for accuracy testing.
- **Expected Behavior**: FP32 kernel selected, likely SDPA fallback.
- **Potential Failure Modes**:
  - Optimized kernels often don't support fp32
- **Mitigation Strategies**:
  - SDPA always supports fp32

#### NO-021: INT8 Quantized Attention
- **Situation**: W8A8 quantized model inference.
- **Expected Behavior**: INT8 kernel selected with quant_format=QuantFormat.INT8.
- **Potential Failure Modes**:
  - Dequantization overhead
- **Mitigation Strategies**:
  - Fused dequant-compute kernels

#### NO-022: FP8 E4M3 Precision
- **Situation**: Hopper GPU with FP8 tensors.
- **Expected Behavior**: FP8 optimized kernel for maximum throughput.
- **Potential Failure Modes**:
  - Pre-Hopper GPU doesn't support FP8
- **Mitigation Strategies**:
  - SM 9.0 check

#### NO-023: Mixed Precision (FP8 compute, FP16 accumulate)
- **Situation**: FP8 weights, FP16 activations.
- **Expected Behavior**: Mixed precision kernel handles conversion.
- **Potential Failure Modes**:
  - Kernel doesn't support mixed input types
- **Mitigation Strategies**:
  - Cast inputs to common type

### 1.7 Different Hardware Targets

#### NO-024: NVIDIA CUDA (Ampere A100)
- **Situation**: A100 GPU, SM 8.0.
- **Expected Behavior**: Ampere-optimized FlashAttention v2 selected.
- **Potential Failure Modes**: None expected.
- **Mitigation Strategies**: Well-supported path.

#### NO-025: NVIDIA CUDA (Hopper H100)
- **Situation**: H100 GPU, SM 9.0.
- **Expected Behavior**: FlashAttention v3 with TMA selected for best performance.
- **Potential Failure Modes**:
  - FA3 not installed
- **Mitigation Strategies**:
  - Fall back to FA2

#### NO-026: NVIDIA CUDA (Ada L40)
- **Situation**: L40 datacenter GPU, SM 8.9.
- **Expected Behavior**: Ada-specific kernel if available, else Ampere fallback.
- **Potential Failure Modes**:
  - SM 8.9 not in supported_generations
- **Mitigation Strategies**:
  - Include Ada in SM range

#### NO-027: AMD ROCm (MI300X)
- **Situation**: AMD MI300X with ROCm.
- **Expected Behavior**: ROCm-specific attention kernel (AOTriton or CK).
- **Potential Failure Modes**:
  - Platform mismatch with CUDA kernels
- **Mitigation Strategies**:
  - platform field filters CUDA-only kernels

#### NO-028: Intel Gaudi (HPU)
- **Situation**: Habana Gaudi accelerator.
- **Expected Behavior**: HPU-specific kernel with Platform.HPU.
- **Potential Failure Modes**:
  - Limited kernel availability
- **Mitigation Strategies**:
  - CPU fallback if no HPU kernel

#### NO-029: CPU Only (x86 AVX-512)
- **Situation**: CPU inference on Xeon with AVX-512.
- **Expected Behavior**: CPU-optimized SIMD kernel selected.
- **Potential Failure Modes**:
  - Performance much slower than GPU
- **Mitigation Strategies**:
  - Expected behavior for CPU

#### NO-030: CPU Only (ARM NEON)
- **Situation**: ARM CPU (AWS Graviton) inference.
- **Expected Behavior**: ARM NEON optimized kernel.
- **Potential Failure Modes**:
  - x86-specific intrinsics
- **Mitigation Strategies**:
  - Cross-platform SIMD abstractions

#### NO-031: Apple Metal
- **Situation**: Apple Silicon M2 GPU.
- **Expected Behavior**: Metal compute shader kernel.
- **Potential Failure Modes**:
  - MPS limitations
- **Mitigation Strategies**:
  - CPU fallback for unsupported ops

### 1.8 Layout Handling

#### NO-032: BSHD Input Layout
- **Situation**: Input tensor in (batch, seq, heads, dim) layout.
- **Expected Behavior**: SDPA-style kernel expecting BSHD selected.
- **Potential Failure Modes**: None for native layout.
- **Mitigation Strategies**: Default layout support.

#### NO-033: BHSD Input Layout
- **Situation**: Input tensor in (batch, heads, seq, dim) layout.
- **Expected Behavior**: FlashAttention-style kernel expecting BHSD selected.
- **Potential Failure Modes**:
  - Kernel requires different layout
- **Mitigation Strategies**:
  - pre_transforms includes layout permutation

#### NO-034: NHD Ragged Batch Layout
- **Situation**: Variable-length sequences in FlashInfer NHD format.
- **Expected Behavior**: FlashInfer ragged batch kernel selected.
- **Potential Failure Modes**:
  - Fixed-batch kernel can't handle ragged
- **Mitigation Strategies**:
  - Layout enum routing

#### NO-035: Layout Transform Required
- **Situation**: BSHD input but only BHSD kernel available.
- **Expected Behavior**: ExecutionPlan includes transpose in pre_transforms.
- **Potential Failure Modes**:
  - Transform overhead
- **Mitigation Strategies**:
  - transform_cost_hint penalizes kernels needing transforms

---

## 2. Edge Cases (35 Scenarios)

### 2.1 Empty and Boundary Inputs

#### EC-001: Empty Input Tensor
- **Situation**: batch_size=0 or seq_len=0 tensor passed.
- **Expected Behavior**: Return empty output tensor without kernel execution. No selection needed.
- **Potential Failure Modes**:
  - Division by zero in kernel
  - Invalid memory access
- **Mitigation Strategies**:
  - Early return check for empty inputs in dispatch layer

#### EC-002: Single Element Batch
- **Situation**: batch_size=1, seq_len=1, single token.
- **Expected Behavior**: Minimal kernel executes correctly. May use scalar path.
- **Potential Failure Modes**:
  - SIMD underutilization
- **Mitigation Strategies**:
  - Scalar fallback for tiny inputs

#### EC-003: Maximum Batch Size Boundary
- **Situation**: batch_size exactly equals max_batch_size of selected kernel.
- **Expected Behavior**: Kernel executes at capacity limit.
- **Potential Failure Modes**:
  - Off-by-one in constraint check
- **Mitigation Strategies**:
  - Boundary condition testing

#### EC-004: Maximum Sequence Length Boundary
- **Situation**: seq_len exactly equals max_seq_len of kernel.
- **Expected Behavior**: Kernel handles maximum supported length.
- **Potential Failure Modes**:
  - Buffer overflow
- **Mitigation Strategies**:
  - Strict bounds checking

### 2.2 Memory Alignment

#### EC-005: Misaligned Input Tensor
- **Situation**: Tensor data pointer not aligned to 16 bytes.
- **Expected Behavior**: Kernel uses unaligned load (LoadU).
- **Potential Failure Modes**:
  - CUDA alignment error
  - Performance degradation
- **Mitigation Strategies**:
  - requires_contiguous=True for aligned-only kernels

#### EC-006: Non-Contiguous Tensor
- **Situation**: Tensor created via slice/view, not contiguous.
- **Expected Behavior**: If kernel requires_contiguous=True, filter out. Else use strided access.
- **Potential Failure Modes**:
  - Incorrect stride handling
- **Mitigation Strategies**:
  - is_contiguous check in SelectionContext

#### EC-007: Stride Last Dim != 1
- **Situation**: Last dimension has stride > 1 due to transpose view.
- **Expected Behavior**: Kernel filtered if requires_last_dim_stride1=True.
- **Potential Failure Modes**:
  - Vectorization assumes stride=1
- **Mitigation Strategies**:
  - stride_last_dim field in context

#### EC-008: Large Stride Tensor
- **Situation**: Tensor with very large strides (sparse view).
- **Expected Behavior**: May require densification before kernel.
- **Potential Failure Modes**:
  - Memory access pattern cache-unfriendly
- **Mitigation Strategies**:
  - Contiguous copy in pre_transforms

### 2.3 Partial and Boundary Batches

#### EC-009: Partial Batch (Not Power of 2)
- **Situation**: batch_size=7 (not power of 2).
- **Expected Behavior**: Kernel handles non-power-of-2 batch correctly.
- **Potential Failure Modes**:
  - Kernel assumes power-of-2
- **Mitigation Strategies**:
  - Pad to next power of 2 if required

#### EC-010: Odd Sequence Length
- **Situation**: seq_len=513 (not divisible by typical tile sizes).
- **Expected Behavior**: Kernel handles remainder tiles.
- **Potential Failure Modes**:
  - Tile size assumption violated
- **Mitigation Strategies**:
  - Remainder handling in kernel implementation

#### EC-011: Head Dim Not Power of 2
- **Situation**: head_dim=96 (not 64 or 128).
- **Expected Behavior**: Kernel with head_dim_multiple=1 or padding.
- **Potential Failure Modes**:
  - SIMD width mismatch
- **Mitigation Strategies**:
  - head_dim_multiple constraint in KernelSpec

#### EC-012: Unequal Q/K Sequence Lengths
- **Situation**: seq_len_q=512, seq_len_k=1024 (cross attention).
- **Expected Behavior**: Cross-attention kernel handles asymmetric lengths.
- **Potential Failure Modes**:
  - Kernel assumes seq_len_q == seq_len_k
- **Mitigation Strategies**:
  - Separate seq_len_q/seq_len_k fields in context

### 2.4 Mixed Precision Edge Cases

#### EC-013: Q/K/V Different Dtypes
- **Situation**: Q is fp16, K/V are bf16 (rare but possible).
- **Expected Behavior**: Type promotion or rejection.
- **Potential Failure Modes**:
  - Undefined behavior with mixed types
- **Mitigation Strategies**:
  - Validate all input dtypes match

#### EC-014: Output Dtype Override
- **Situation**: Request fp32 output from fp16 computation.
- **Expected Behavior**: produces_dtype field in KernelSpec, cast in post_transforms.
- **Potential Failure Modes**:
  - Silent precision loss
- **Mitigation Strategies**:
  - Explicit dtype conversion in plan

#### EC-015: Subnormal Float Inputs
- **Situation**: Input contains subnormal (denormalized) floats.
- **Expected Behavior**: Correct numerical result, possibly slower.
- **Potential Failure Modes**:
  - Flush-to-zero behavior varies
- **Mitigation Strategies**:
  - Document FTZ behavior per kernel

#### EC-016: Inf/NaN in Input
- **Situation**: Input tensor contains Inf or NaN values.
- **Expected Behavior**: NaN propagation, no crash.
- **Potential Failure Modes**:
  - Infinite loop in softmax
  - GPU hang
- **Mitigation Strategies**:
  - Input validation in debug mode

### 2.5 Device and Platform Edge Cases

#### EC-017: GPU Not Available at Selection Time
- **Situation**: CUDA device temporarily unavailable during kernel selection.
- **Expected Behavior**: Fall back to CPU kernel or raise informative error.
- **Potential Failure Modes**:
  - cryptic CUDA error
- **Mitigation Strategies**:
  - Device availability check in SelectionContext.from_tensors()

#### EC-018: Multi-GPU Different Architectures
- **Situation**: System has both A100 (SM 8.0) and H100 (SM 9.0).
- **Expected Behavior**: Selection based on tensor's current device.
- **Potential Failure Modes**:
  - Wrong GPU architecture assumed
- **Mitigation Strategies**:
  - DeviceSpec.detect() per tensor

#### EC-019: CPU with No SIMD Support
- **Situation**: Old CPU without AVX support.
- **Expected Behavior**: Scalar fallback kernel.
- **Potential Failure Modes**:
  - SIGILL from AVX instruction
- **Mitigation Strategies**:
  - Runtime CPU feature detection

#### EC-020: Virtual GPU (vGPU)
- **Situation**: GPU is virtualized (NVIDIA vGPU).
- **Expected Behavior**: Same as physical GPU, may have memory limits.
- **Potential Failure Modes**:
  - Reduced memory reported
- **Mitigation Strategies**:
  - Check actual available memory

### 2.6 Special Operation Modes

#### EC-021: CUDA Graph Capture Mode
- **Situation**: is_cuda_graph_capturing=True during selection.
- **Expected Behavior**: Only CUDA-graph-safe kernels selected.
- **Potential Failure Modes**:
  - Kernel uses dynamic allocation
- **Mitigation Strategies**:
  - is_cuda_graph_safe filter

#### EC-022: Deterministic Mode Required
- **Situation**: requires_deterministic=True for reproducibility.
- **Expected Behavior**: Only deterministic kernels selected (deterministic=True).
- **Potential Failure Modes**:
  - No deterministic kernel available
- **Mitigation Strategies**:
  - SDPA deterministic fallback

#### EC-023: Dropout During Inference
- **Situation**: dropout_p > 0 during inference (training mode).
- **Expected Behavior**: Kernel supporting dropout selected.
- **Potential Failure Modes**:
  - RNG state management
- **Mitigation Strategies**:
  - supports_dropout filter

#### EC-024: Custom Scale Factor
- **Situation**: Explicit scale parameter != 1/sqrt(head_dim).
- **Expected Behavior**: Kernel supporting scale parameter selected.
- **Potential Failure Modes**:
  - Kernel ignores scale
- **Mitigation Strategies**:
  - supports_scale requirement

### 2.7 Registry Edge Cases

#### EC-025: No Kernels Registered
- **Situation**: Empty kernel registry.
- **Expected Behavior**: NoKernelAvailableError with clear message.
- **Potential Failure Modes**:
  - Silent failure
- **Mitigation Strategies**:
  - Explicit error with available operations list

#### EC-026: All Kernels Filtered Out
- **Situation**: All registered kernels fail constraint checks.
- **Expected Behavior**: NoKernelAvailableError with filtered_out reasons.
- **Potential Failure Modes**:
  - No explanation of why
- **Mitigation Strategies**:
  - Include per-kernel rejection reasons

#### EC-027: Duplicate Kernel Registration
- **Situation**: Attempt to register kernel with existing kernel_id.
- **Expected Behavior**: ValueError raised immediately.
- **Potential Failure Modes**:
  - Silent override
- **Mitigation Strategies**:
  - Explicit duplicate check in register()

#### EC-028: Kernel Unregistration During Selection
- **Situation**: Kernel unregistered while select() in progress.
- **Expected Behavior**: RLock ensures atomic operation completion.
- **Potential Failure Modes**:
  - Race condition
- **Mitigation Strategies**:
  - Thread-safe registry with RLock

### 2.8 Cache Edge Cases

#### EC-029: Cache Key Collision
- **Situation**: Two different contexts produce same MD5 hash.
- **Expected Behavior**: Extremely unlikely (1 in 2^128). If occurs, wrong plan returned.
- **Potential Failure Modes**:
  - Incorrect kernel execution
- **Mitigation Strategies**:
  - Consider SHA-256 for cache keys

#### EC-030: Cache at Maximum Capacity
- **Situation**: Cache shard at max_entries_per_shard limit.
- **Expected Behavior**: LRU eviction of oldest entry.
- **Potential Failure Modes**:
  - Eviction thrashing
- **Mitigation Strategies**:
  - Tune max_entries_per_shard

#### EC-031: TTL Expiration During Request
- **Situation**: Cache entry expires between get() and actual use.
- **Expected Behavior**: Still valid for current request, re-selected on next.
- **Potential Failure Modes**:
  - None (entry lifetime > request duration)
- **Mitigation Strategies**:
  - TTL should be >> request latency

#### EC-032: Zero-Entry Cache (Disabled)
- **Situation**: Cache explicitly disabled with max_entries_per_shard=0.
- **Expected Behavior**: ValueError on construction.
- **Potential Failure Modes**:
  - Division by zero
- **Mitigation Strategies**:
  - Validation in constructor

### 2.9 Policy Edge Cases

#### EC-033: Empty Policy
- **Situation**: Policy with no rules (locks, allows, denies, boosts all empty).
- **Expected Behavior**: All kernels allowed, no score modifications.
- **Potential Failure Modes**:
  - None expected
- **Mitigation Strategies**:
  - Default empty policy is valid

#### EC-034: Lock Rule Without Matching Kernel
- **Situation**: Policy locks to kernel_id that doesn't exist.
- **Expected Behavior**: Lock ignored, normal selection proceeds.
- **Potential Failure Modes**:
  - Kernel not found error
- **Mitigation Strategies**:
  - Check kernel exists before returning locked plan

#### EC-035: Conflicting Allow/Deny Rules
- **Situation**: Same kernel matched by both allow and deny rules.
- **Expected Behavior**: Deny takes precedence.
- **Potential Failure Modes**:
  - Inconsistent behavior
- **Mitigation Strategies**:
  - Document deny > allow precedence

---

## 3. Failure Scenarios (35 Scenarios)

### 3.1 Kernel Execution Failures

#### FS-001: Kernel Crash (CUDA Error)
- **Situation**: Kernel throws CUDA error during execution (e.g., illegal memory access).
- **Expected Behavior**: Exception propagated with context. Device remains usable after reset.
- **Potential Failure Modes**:
  - Device hang requiring driver restart
  - Corrupted GPU state
- **Mitigation Strategies**:
  - Catch CUDA errors, call cudaDeviceReset() if needed
  - Workspace memory bounds checking

#### FS-002: Kernel Timeout
- **Situation**: Kernel takes > 30 seconds (TDR timeout on Windows).
- **Expected Behavior**: CUDA driver kills kernel, reports timeout.
- **Potential Failure Modes**:
  - System freeze
- **Mitigation Strategies**:
  - Watchdog timer, chunked execution for long inputs

#### FS-003: Kernel Returns Wrong Shape
- **Situation**: Kernel returns output with unexpected shape.
- **Expected Behavior**: Shape validation raises error immediately.
- **Potential Failure Modes**:
  - Silent incorrect results downstream
- **Mitigation Strategies**:
  - Assert output shape matches expected

#### FS-004: Kernel Silent Numerical Error
- **Situation**: Kernel produces incorrect but valid-shaped output.
- **Expected Behavior**: Not detectable without reference comparison.
- **Potential Failure Modes**:
  - Wrong model outputs
- **Mitigation Strategies**:
  - Periodic correctness validation against reference

### 3.2 GPU and Driver Failures

#### FS-005: GPU Out of Memory
- **Situation**: cudaMalloc fails during kernel workspace allocation.
- **Expected Behavior**: Clear OOM error with memory usage details.
- **Potential Failure Modes**:
  - Cryptic allocation failure
- **Mitigation Strategies**:
  - Pre-check available memory, fall back to smaller kernel

#### FS-006: GPU Driver Crash
- **Situation**: NVIDIA driver crashes mid-inference.
- **Expected Behavior**: CUDA_ERROR_UNKNOWN, requires process restart.
- **Potential Failure Modes**:
  - System instability
- **Mitigation Strategies**:
  - Heartbeat monitoring, automatic restart

#### FS-007: GPU Thermal Shutdown
- **Situation**: GPU overheats and throttles to zero.
- **Expected Behavior**: Performance degrades, eventually errors.
- **Potential Failure Modes**:
  - Silent slowdown
- **Mitigation Strategies**:
  - Temperature monitoring, alert thresholds

#### FS-008: ECC Memory Error
- **Situation**: Uncorrectable ECC error in GPU memory.
- **Expected Behavior**: CUDA_ERROR_ECC_UNCORRECTABLE raised.
- **Potential Failure Modes**:
  - Silent data corruption
- **Mitigation Strategies**:
  - Enable ECC checking, replace faulty GPU

#### FS-009: PCIe Bus Error
- **Situation**: PCIe communication failure between CPU and GPU.
- **Expected Behavior**: CUDA context becomes invalid.
- **Potential Failure Modes**:
  - System crash
- **Mitigation Strategies**:
  - Error-detecting PCIe config, redundant GPUs

### 3.3 Memory Allocation Failures

#### FS-010: Workspace Allocation Failure
- **Situation**: Kernel workspace_bytes exceeds available memory.
- **Expected Behavior**: Try smaller workspace or different kernel.
- **Potential Failure Modes**:
  - Immediate OOM
- **Mitigation Strategies**:
  - Query workspace before selection

#### FS-011: Output Tensor Allocation Failure
- **Situation**: Cannot allocate output tensor.
- **Expected Behavior**: OOM error before kernel launch.
- **Potential Failure Modes**:
  - Allocation during kernel execution
- **Mitigation Strategies**:
  - Pre-allocate all tensors

#### FS-012: Cache Memory Exhaustion
- **Situation**: Selection cache consumes too much memory.
- **Expected Behavior**: LRU eviction keeps memory bounded.
- **Potential Failure Modes**:
  - Memory leak in cache
- **Mitigation Strategies**:
  - max_entries_per_shard limit

#### FS-013: Python GC Stall
- **Situation**: Python garbage collection during kernel dispatch.
- **Expected Behavior**: Brief latency spike.
- **Potential Failure Modes**:
  - Unpredictable latency
- **Mitigation Strategies**:
  - gc.disable() during critical path

### 3.4 Hot-Reload Failures

#### FS-014: Hot-Reload During Kernel Execution
- **Situation**: Kernel code hot-reloaded while kernel is running.
- **Expected Behavior**: Current execution completes with old code.
- **Potential Failure Modes**:
  - Code mismatch, crash
- **Mitigation Strategies**:
  - Version fence, wait for completion

#### FS-015: Backend Module Import Failure
- **Situation**: Backend module (flash_attn) fails to import.
- **Expected Behavior**: Backend marked unavailable, alternatives used.
- **Potential Failure Modes**:
  - Startup failure
- **Mitigation Strategies**:
  - Lazy import with graceful degradation

#### FS-016: Backend Version Mismatch
- **Situation**: flash_attn version incompatible with PyTorch.
- **Expected Behavior**: Version check fails, kernel not registered.
- **Potential Failure Modes**:
  - Runtime error during execution
- **Mitigation Strategies**:
  - Version validation at registration time

#### FS-017: JIT Compilation Failure
- **Situation**: Triton kernel fails JIT compilation.
- **Expected Behavior**: Compilation error caught, fallback kernel used.
- **Potential Failure Modes**:
  - First-request latency spike
- **Mitigation Strategies**:
  - AOT compilation, cache compiled kernels

### 3.5 Configuration Failures

#### FS-018: Policy File Corruption
- **Situation**: Policy YAML file has syntax errors.
- **Expected Behavior**: PolicyLoadError with line number.
- **Potential Failure Modes**:
  - Default policy silently used
- **Mitigation Strategies**:
  - Strict validation, atomic policy updates

#### FS-019: PerfDB Corruption
- **Situation**: SQLite performance database corrupted.
- **Expected Behavior**: Integrity check fails, recreate DB.
- **Potential Failure Modes**:
  - Query errors
- **Mitigation Strategies**:
  - WAL mode, periodic backups

#### FS-020: Dispatch Table File Missing
- **Situation**: Pre-computed dispatch table file doesn't exist.
- **Expected Behavior**: Build table on first request.
- **Potential Failure Modes**:
  - Cold start latency
- **Mitigation Strategies**:
  - Ship default dispatch tables

#### FS-021: Config Version Mismatch
- **Situation**: Saved config from different LayerZero version.
- **Expected Behavior**: Version check fails, use defaults.
- **Potential Failure Modes**:
  - Incompatible config applied
- **Mitigation Strategies**:
  - Version field in all serialized configs

### 3.6 Distributed Failures

#### FS-022: Network Partition in Distributed Setup
- **Situation**: Network failure between ranks in tensor parallel.
- **Expected Behavior**: NCCL timeout, error propagated.
- **Potential Failure Modes**:
  - Deadlock waiting for other ranks
- **Mitigation Strategies**:
  - Timeout handling, failover

#### FS-023: Rank Failure Mid-Computation
- **Situation**: One rank crashes during all-reduce.
- **Expected Behavior**: Error on all ranks, coordinated shutdown.
- **Potential Failure Modes**:
  - Hanging processes
- **Mitigation Strategies**:
  - Heartbeat, process group monitoring

#### FS-024: Inconsistent Selection Across Ranks
- **Situation**: Different ranks select different kernels.
- **Expected Behavior**: Should not happen with same context.
- **Potential Failure Modes**:
  - Different numerical results
- **Mitigation Strategies**:
  - Seed synchronization, deterministic selection

### 3.7 Resource Exhaustion

#### FS-025: File Descriptor Exhaustion
- **Situation**: Too many PerfDB connections open.
- **Expected Behavior**: Connection error.
- **Potential Failure Modes**:
  - Silent failure
- **Mitigation Strategies**:
  - Connection pooling, cleanup

#### FS-026: Thread Pool Exhaustion
- **Situation**: All worker threads busy.
- **Expected Behavior**: Requests queue, backpressure.
- **Potential Failure Modes**:
  - Unbounded queue growth
- **Mitigation Strategies**:
  - Bounded queue, rejection policy

#### FS-027: CUDA Context Limit
- **Situation**: Too many CUDA contexts (> 64 per GPU typically).
- **Expected Behavior**: Context creation fails.
- **Potential Failure Modes**:
  - Hard limit hit
- **Mitigation Strategies**:
  - Context sharing, lazy creation

#### FS-028: GPU Memory Fragmentation
- **Situation**: Memory available but fragmented.
- **Expected Behavior**: Allocation fails despite "available" memory.
- **Potential Failure Modes**:
  - Intermittent OOM
- **Mitigation Strategies**:
  - Memory pools, pre-allocation

### 3.8 Type and Validation Errors

#### FS-029: Invalid Kernel ID Format
- **Situation**: kernel_id contains invalid characters.
- **Expected Behavior**: Validation error at registration.
- **Potential Failure Modes**:
  - Cache key issues
- **Mitigation Strategies**:
  - Regex validation on kernel_id

#### FS-030: Invalid Operation Identifier
- **Situation**: Unknown operation string passed.
- **Expected Behavior**: NoKernelAvailableError with "no candidates found".
- **Potential Failure Modes**:
  - Confusing error
- **Mitigation Strategies**:
  - List valid operations in error

#### FS-031: Negative Shape Dimension
- **Situation**: batch_size=-1 or seq_len=-1.
- **Expected Behavior**: ValueError immediately.
- **Potential Failure Modes**:
  - Memory allocation issues
- **Mitigation Strategies**:
  - Input validation

#### FS-032: Type Coercion Failure
- **Situation**: String passed where int expected.
- **Expected Behavior**: TypeError with clear message.
- **Potential Failure Modes**:
  - Silent cast to wrong value
- **Mitigation Strategies**:
  - Type hints, runtime validation

### 3.9 External Dependency Failures

#### FS-033: PyTorch Version Incompatibility
- **Situation**: PyTorch version too old for backend.
- **Expected Behavior**: Import error with version requirement.
- **Potential Failure Modes**:
  - Cryptic attribute error
- **Mitigation Strategies**:
  - Version check in __init__.py

#### FS-034: CUDA Toolkit Missing
- **Situation**: CUDA not installed on system.
- **Expected Behavior**: Fall back to CPU kernels.
- **Potential Failure Modes**:
  - torch.cuda.is_available() = False not handled
- **Mitigation Strategies**:
  - Check device availability

#### FS-035: Triton Not Installed
- **Situation**: Triton backend not installed.
- **Expected Behavior**: Triton kernels not registered.
- **Potential Failure Modes**:
  - Import error at runtime
- **Mitigation Strategies**:
  - Optional import, log warning

---

## 4. Concurrency Scenarios (35 Scenarios)

### 4.1 Multiple Threads Same Kernel

#### CS-001: Concurrent Selection Same Context
- **Situation**: 100 threads simultaneously call select() with identical context.
- **Expected Behavior**: First computes, others wait via inflight deduplication.
- **Potential Failure Modes**:
  - Thundering herd without deduplication
- **Mitigation Strategies**:
  - get_or_compute() with Event-based wait

#### CS-002: Concurrent Selection Slightly Different Contexts
- **Situation**: 100 threads with contexts differing only in batch_size.
- **Expected Behavior**: Each unique context computed once, cached.
- **Potential Failure Modes**:
  - Cache thrashing
- **Mitigation Strategies**:
  - Bucket similar contexts

#### CS-003: Concurrent Kernel Execution Same GPU
- **Situation**: Multiple threads launch same kernel on same GPU.
- **Expected Behavior**: CUDA streams serialize execution.
- **Potential Failure Modes**:
  - Memory contention
- **Mitigation Strategies**:
  - Per-stream workspace allocation

#### CS-004: High-Frequency Selection Same Operation
- **Situation**: 10,000 selections/sec for same operation type.
- **Expected Behavior**: Cache absorbs load, < 100 actual selections.
- **Potential Failure Modes**:
  - Lock contention on cache
- **Mitigation Strategies**:
  - 256 shards distribute load

### 4.2 Multiple Threads Different Kernels

#### CS-005: Parallel Selections Different Operations
- **Situation**: Thread A selects attention, Thread B selects norm, Thread C selects MLP.
- **Expected Behavior**: No contention, independent selections.
- **Potential Failure Modes**:
  - Global lock bottleneck
- **Mitigation Strategies**:
  - Per-shard locking

#### CS-006: Parallel Registry Reads
- **Situation**: 100 threads reading from kernel registry.
- **Expected Behavior**: Read-heavy workload, minimal contention.
- **Potential Failure Modes**:
  - Writer starvation
- **Mitigation Strategies**:
  - RwLock or copy-on-write

#### CS-007: Concurrent Selections Different Devices
- **Situation**: Thread A on GPU 0, Thread B on GPU 1.
- **Expected Behavior**: Separate cache keys, no interaction.
- **Potential Failure Modes**:
  - Wrong device in context
- **Mitigation Strategies**:
  - Device in cache key

#### CS-008: Mixed Read/Write Registry Access
- **Situation**: 99 readers, 1 writer registering new kernel.
- **Expected Behavior**: Writer briefly blocks readers.
- **Potential Failure Modes**:
  - Stale reads
- **Mitigation Strategies**:
  - RLock ensures consistency

### 4.3 Hot-Reload During Execution

#### CS-009: Policy Hot-Reload During Selection
- **Situation**: Policy updated via update_policy() while select() in progress.
- **Expected Behavior**: Old policy completes, new policy for next select().
- **Potential Failure Modes**:
  - Partial policy application
- **Mitigation Strategies**:
  - Atomic policy swap

#### CS-010: Kernel Registration During Selection
- **Situation**: New kernel registered while select() iterating candidates.
- **Expected Behavior**: Current select() may or may not see new kernel.
- **Potential Failure Modes**:
  - Iterator invalidation
- **Mitigation Strategies**:
  - Copy candidate list under lock

#### CS-011: Cache Invalidation During Lookup
- **Situation**: invalidate_all() called during concurrent get().
- **Expected Behavior**: Version bump makes entry invalid.
- **Potential Failure Modes**:
  - Stale entry returned
- **Mitigation Strategies**:
  - Version check after acquiring entry

#### CS-012: Backend Health Change During Selection
- **Situation**: Backend becomes unhealthy during candidate filtering.
- **Expected Behavior**: Health check at filter time, unhealthy excluded.
- **Potential Failure Modes**:
  - Selected kernel from unhealthy backend
- **Mitigation Strategies**:
  - Re-check health before execution

### 4.4 Config Update During Execution

#### CS-013: Dispatch Table Update During Lookup
- **Situation**: Dispatch table reloaded while lookup() in progress.
- **Expected Behavior**: Old table completes lookup.
- **Potential Failure Modes**:
  - Mixed old/new entries
- **Mitigation Strategies**:
  - Copy-on-write table replacement

#### CS-014: PerfDB Write During Read
- **Situation**: put_record() while get_all_records() iterating.
- **Expected Behavior**: SQLite WAL handles isolation.
- **Potential Failure Modes**:
  - Inconsistent read
- **Mitigation Strategies**:
  - SQLite transaction isolation

#### CS-015: Cache TTL Change During Operation
- **Situation**: ttl_seconds modified (not currently supported).
- **Expected Behavior**: Not supported, TTL fixed at construction.
- **Potential Failure Modes**:
  - Runtime TTL change would be complex
- **Mitigation Strategies**:
  - Document immutable TTL

#### CS-016: Log Level Change During Execution
- **Situation**: Logger verbosity changed mid-request.
- **Expected Behavior**: New log level applies immediately.
- **Potential Failure Modes**:
  - None significant
- **Mitigation Strategies**:
  - Thread-safe logger

### 4.5 Device Migration

#### CS-017: Tensor Moved to Different GPU Mid-Selection
- **Situation**: Tensor .to(device) called after context created.
- **Expected Behavior**: Context reflects original device. Mismatch detected.
- **Potential Failure Modes**:
  - Wrong kernel for new device
- **Mitigation Strategies**:
  - Re-create context after device change

#### CS-018: Model Sharding During Inference
- **Situation**: Model partially offloaded to different device.
- **Expected Behavior**: Each layer selects for its device.
- **Potential Failure Modes**:
  - Device mismatch at layer boundary
- **Mitigation Strategies**:
  - Per-operation device tracking

#### CS-019: Dynamic GPU Allocation
- **Situation**: GPU dynamically allocated from pool.
- **Expected Behavior**: Device detection at selection time.
- **Potential Failure Modes**:
  - Cached selection for wrong GPU
- **Mitigation Strategies**:
  - Device in cache key

#### CS-020: CPU-GPU Memory Swap During Execution
- **Situation**: Memory pressure causes tensor swap to CPU.
- **Expected Behavior**: Kernel fails if tensor moved.
- **Potential Failure Modes**:
  - Illegal memory access
- **Mitigation Strategies**:
  - Pin tensors, check device before launch

### 4.6 Priority and Deadlock

#### CS-021: Priority Inversion in Selection
- **Situation**: High-priority request blocked by low-priority cache computation.
- **Expected Behavior**: All waiters get same priority result.
- **Potential Failure Modes**:
  - Latency for high-priority request
- **Mitigation Strategies**:
  - Priority-aware scheduling not implemented

#### CS-022: Nested Lock Acquisition
- **Situation**: Code path acquires registry lock then cache lock.
- **Expected Behavior**: Consistent lock order prevents deadlock.
- **Potential Failure Modes**:
  - Deadlock if order reversed
- **Mitigation Strategies**:
  - Document lock ordering

#### CS-023: Self-Deadlock Attempt
- **Situation**: Same thread tries to acquire same lock twice.
- **Expected Behavior**: RLock allows re-acquisition.
- **Potential Failure Modes**:
  - Deadlock with regular Lock
- **Mitigation Strategies**:
  - Use RLock throughout

#### CS-024: Cross-Component Deadlock
- **Situation**: Component A waits for B, B waits for A.
- **Expected Behavior**: Should not occur with proper design.
- **Potential Failure Modes**:
  - System freeze
- **Mitigation Strategies**:
  - Lock hierarchy, timeout on lock acquire

### 4.7 Race Conditions

#### CS-025: Time-of-Check to Time-of-Use (TOCTOU)
- **Situation**: Kernel availability checked, then used, but removed in between.
- **Expected Behavior**: Second check or atomic operation.
- **Potential Failure Modes**:
  - KeyError on kernel access
- **Mitigation Strategies**:
  - Hold lock through use

#### CS-026: Double-Free in Cache
- **Situation**: Cache entry evicted twice concurrently.
- **Expected Behavior**: Lock prevents double eviction.
- **Potential Failure Modes**:
  - Corruption of internal state
- **Mitigation Strategies**:
  - Per-shard lock

#### CS-027: Lost Update in Statistics
- **Situation**: Two threads increment hit counter simultaneously.
- **Expected Behavior**: Both increments recorded.
- **Potential Failure Modes**:
  - Counter less than actual hits
- **Mitigation Strategies**:
  - Atomic increment or lock

#### CS-028: Race in In-Flight Deduplication
- **Situation**: Two threads both try to be "first" for same key.
- **Expected Behavior**: Only one creates Event, other waits.
- **Potential Failure Modes**:
  - Both compute, one result lost
- **Mitigation Strategies**:
  - Check inflight under lock

### 4.8 Concurrent Stress Scenarios

#### CS-029: 1000 Concurrent Selections
- **Situation**: 1000 threads all calling select() simultaneously.
- **Expected Behavior**: System remains responsive, no crash.
- **Potential Failure Modes**:
  - Thread starvation
- **Mitigation Strategies**:
  - Bounded thread pool

#### CS-030: Concurrent Selection and Invalidation
- **Situation**: One thread invalidates cache while others select.
- **Expected Behavior**: Selections complete or retry.
- **Potential Failure Modes**:
  - Stale results
- **Mitigation Strategies**:
  - Version check post-lock

#### CS-031: Rapid Policy Updates
- **Situation**: Policy updated 100 times/second.
- **Expected Behavior**: Each update atomic, cache invalidated.
- **Potential Failure Modes**:
  - Cache thrashing
- **Mitigation Strategies**:
  - Rate limit policy updates

#### CS-032: Concurrent Registry Modification
- **Situation**: Multiple threads registering/unregistering kernels.
- **Expected Behavior**: RLock serializes modifications.
- **Potential Failure Modes**:
  - Inconsistent indices
- **Mitigation Strategies**:
  - Atomic batch operations

### 4.9 Async and Await Patterns

#### CS-033: Async Selection With Await
- **Situation**: Multiple async tasks awaiting selection coroutine.
- **Expected Behavior**: Event loop handles concurrency.
- **Potential Failure Modes**:
  - Blocking call in async context
- **Mitigation Strategies**:
  - Use run_in_executor for blocking ops

#### CS-034: Mixed Sync/Async Calls
- **Situation**: Some callers sync, some async.
- **Expected Behavior**: Both work correctly.
- **Potential Failure Modes**:
  - Deadlock mixing async/sync
- **Mitigation Strategies**:
  - Separate sync and async interfaces

#### CS-035: Cancellation During Selection
- **Situation**: Async task cancelled while waiting for selection.
- **Expected Behavior**: Clean cancellation, no resource leak.
- **Potential Failure Modes**:
  - Dangling computation
- **Mitigation Strategies**:
  - Check cancellation token

---

## 5. Performance Scenarios (35 Scenarios)

### 5.1 Latency Scenarios

#### PS-001: Cold Start Selection Latency
- **Situation**: First selection after process start.
- **Expected Behavior**: < 10ms including backend discovery.
- **Potential Failure Modes**:
  - > 100ms due to JIT compilation
- **Mitigation Strategies**:
  - Warm-up on startup, AOT compilation

#### PS-002: Warm Cache Selection Latency
- **Situation**: Repeated selection with cache hit.
- **Expected Behavior**: < 100us per selection.
- **Potential Failure Modes**:
  - Lock contention
- **Mitigation Strategies**:
  - Sharded cache, minimal critical section

#### PS-003: Cache Miss Selection Latency
- **Situation**: Selection with cache miss but kernels registered.
- **Expected Behavior**: < 1ms for full filter/score pipeline.
- **Potential Failure Modes**:
  - Slow constraint checking
- **Mitigation Strategies**:
  - Pre-computed constraint masks

#### PS-004: Policy Evaluation Latency
- **Situation**: Complex policy with 100 rules.
- **Expected Behavior**: < 500us for rule evaluation.
- **Potential Failure Modes**:
  - Linear scan through rules
- **Mitigation Strategies**:
  - Index rules by match criteria

#### PS-005: Registry Lookup Latency
- **Situation**: 1000 kernels registered, lookup by operation.
- **Expected Behavior**: O(1) via _by_operation index.
- **Potential Failure Modes**:
  - Linear scan
- **Mitigation Strategies**:
  - Hash-based indexing

### 5.2 Cache Performance

#### PS-006: Cache Hit Rate Target
- **Situation**: Steady-state inference workload.
- **Expected Behavior**: > 99% cache hit rate.
- **Potential Failure Modes**:
  - Low hit rate due to poor bucketing
- **Mitigation Strategies**:
  - Shape bucketing, larger cache

#### PS-007: Cache Sharding Efficiency
- **Situation**: 256 shards, 16 threads.
- **Expected Behavior**: Near-zero contention.
- **Potential Failure Modes**:
  - Hot shards
- **Mitigation Strategies**:
  - Good hash distribution

#### PS-008: LRU Eviction Performance
- **Situation**: Cache at capacity with new entries.
- **Expected Behavior**: O(1) LRU eviction.
- **Potential Failure Modes**:
  - O(n) eviction
- **Mitigation Strategies**:
  - OrderedDict for O(1) LRU

#### PS-009: Cache Invalidation Performance
- **Situation**: Invalidate all entries (policy change).
- **Expected Behavior**: O(1) via version bump.
- **Potential Failure Modes**:
  - O(n) entry deletion
- **Mitigation Strategies**:
  - MVCC versioning

#### PS-010: Thundering Herd Prevention
- **Situation**: 1000 threads request same uncached key.
- **Expected Behavior**: 1 computes, 999 wait.
- **Potential Failure Modes**:
  - 1000 concurrent computations
- **Mitigation Strategies**:
  - In-flight deduplication with Event

### 5.3 Memory Performance

#### PS-011: Cache Memory Footprint
- **Situation**: 256 shards * 100 entries = 25,600 cached plans.
- **Expected Behavior**: < 100MB total memory.
- **Potential Failure Modes**:
  - Memory leak
- **Mitigation Strategies**:
  - Bounded size, no circular references

#### PS-012: Selection Allocations
- **Situation**: Single selection operation.
- **Expected Behavior**: Minimal allocations (ideally zero in hot path).
- **Potential Failure Modes**:
  - List/dict creation per selection
- **Mitigation Strategies**:
  - Pre-allocated buffers

#### PS-013: PerfDB Memory Usage
- **Situation**: 1M performance records.
- **Expected Behavior**: < 500MB SQLite database.
- **Potential Failure Modes**:
  - Unbounded growth
- **Mitigation Strategies**:
  - Periodic vacuum, record limits

#### PS-014: Kernel Spec Memory
- **Situation**: 100 registered kernels.
- **Expected Behavior**: < 10MB for all specs.
- **Potential Failure Modes**:
  - Large frozen sets
- **Mitigation Strategies**:
  - Efficient frozenset storage

### 5.4 CPU Utilization

#### PS-015: Selection CPU Overhead
- **Situation**: Selection during inference.
- **Expected Behavior**: < 1% CPU overhead.
- **Potential Failure Modes**:
  - Hash computation overhead
- **Mitigation Strategies**:
  - Cache hit fast path

#### PS-016: CPU Throttling Impact
- **Situation**: CPU in power-save mode during selection.
- **Expected Behavior**: Latency increases proportionally.
- **Potential Failure Modes**:
  - 10x latency increase
- **Mitigation Strategies**:
  - Pin CPU frequency for benchmarks

#### PS-017: Background Thread CPU Usage
- **Situation**: MVCC maintenance thread.
- **Expected Behavior**: < 0.1% CPU when idle.
- **Potential Failure Modes**:
  - Busy polling
- **Mitigation Strategies**:
  - Event-driven, no polling

#### PS-018: GIL Contention
- **Situation**: Multi-threaded Python selection.
- **Expected Behavior**: GIL released during actual kernel execution.
- **Potential Failure Modes**:
  - Python code holds GIL
- **Mitigation Strategies**:
  - Minimize Python in hot path

### 5.5 GPU Performance

#### PS-019: GPU Thermal Throttling
- **Situation**: Sustained inference at 100% GPU utilization.
- **Expected Behavior**: Kernel performance degrades gracefully.
- **Potential Failure Modes**:
  - Sudden performance cliff
- **Mitigation Strategies**:
  - Temperature monitoring, throttle requests

#### PS-020: GPU Memory Bandwidth Saturation
- **Situation**: Large batch attention (memory-bound).
- **Expected Behavior**: Achieve > 80% memory bandwidth.
- **Potential Failure Modes**:
  - Poor memory access pattern
- **Mitigation Strategies**:
  - Select bandwidth-optimized kernel

#### PS-021: GPU Compute Saturation
- **Situation**: Small batch decode (compute-bound).
- **Expected Behavior**: Achieve > 70% SM utilization.
- **Potential Failure Modes**:
  - Low occupancy
- **Mitigation Strategies**:
  - Select compute-optimized kernel

#### PS-022: Multi-GPU Load Balancing
- **Situation**: 8 GPUs with tensor parallelism.
- **Expected Behavior**: Equal work distribution.
- **Potential Failure Modes**:
  - Straggler GPU
- **Mitigation Strategies**:
  - Monitor per-GPU latency

### 5.6 Tail Latency

#### PS-023: P99 Latency Target
- **Situation**: 1000 selections.
- **Expected Behavior**: P99 < 2x median.
- **Potential Failure Modes**:
  - Outliers due to contention
- **Mitigation Strategies**:
  - Lock-free reads where possible

#### PS-024: P99.9 Latency Target
- **Situation**: 10000 selections.
- **Expected Behavior**: P99.9 < 5x median.
- **Potential Failure Modes**:
  - GC pauses, lock contention
- **Mitigation Strategies**:
  - Pre-allocate, minimize GC

#### PS-025: Latency Under Load
- **Situation**: 90% sustained throughput.
- **Expected Behavior**: Latency increases < 2x.
- **Potential Failure Modes**:
  - Non-linear latency growth
- **Mitigation Strategies**:
  - Queue management, backpressure

#### PS-026: Latency During Policy Update
- **Situation**: Policy update during high load.
- **Expected Behavior**: Minimal latency spike (< 10ms).
- **Potential Failure Modes**:
  - All requests wait for update
- **Mitigation Strategies**:
  - Atomic swap, per-shard invalidation

### 5.7 Throughput

#### PS-027: Maximum Selection Throughput
- **Situation**: Benchmark selection throughput (cache hits).
- **Expected Behavior**: > 1M selections/sec single-threaded.
- **Potential Failure Modes**:
  - < 100K/sec
- **Mitigation Strategies**:
  - Minimal code in hot path

#### PS-028: Throughput Scaling with Threads
- **Situation**: 1 to 16 threads selection throughput.
- **Expected Behavior**: Near-linear scaling to 8 threads.
- **Potential Failure Modes**:
  - Contention above 4 threads
- **Mitigation Strategies**:
  - Shard count > thread count

#### PS-029: Sustained Throughput
- **Situation**: 1 hour continuous selection.
- **Expected Behavior**: Throughput stable, no degradation.
- **Potential Failure Modes**:
  - Memory growth, GC impact
- **Mitigation Strategies**:
  - Bounded caches, cleanup

#### PS-030: Throughput Under Memory Pressure
- **Situation**: System memory 90% utilized.
- **Expected Behavior**: Graceful degradation.
- **Potential Failure Modes**:
  - Swapping kills performance
- **Mitigation Strategies**:
  - Lock working set, monitor memory

### 5.8 Queue and Buildup

#### PS-031: Request Queue Buildup
- **Situation**: Burst of 1000 requests in 100ms.
- **Expected Behavior**: Queue absorbs burst, drains in < 1s.
- **Potential Failure Modes**:
  - Unbounded queue growth
- **Mitigation Strategies**:
  - Bounded queue with backpressure

#### PS-032: Queue Depth vs Latency
- **Situation**: Queue depth 10, 100, 1000.
- **Expected Behavior**: Latency proportional to queue depth.
- **Potential Failure Modes**:
  - Head-of-line blocking
- **Mitigation Strategies**:
  - Fair scheduling

#### PS-033: Priority Queue Performance
- **Situation**: High-priority requests in queue.
- **Expected Behavior**: High-priority processed first.
- **Potential Failure Modes**:
  - FIFO only, no priority
- **Mitigation Strategies**:
  - Priority scheduling (if implemented)

#### PS-034: Backpressure Mechanism
- **Situation**: Arrival rate > service rate.
- **Expected Behavior**: Reject/delay new requests.
- **Potential Failure Modes**:
  - System overload
- **Mitigation Strategies**:
  - Load shedding, admission control

### 5.9 Profiling and Monitoring

#### PS-035: Profiling Overhead
- **Situation**: Enable detailed timing metrics.
- **Expected Behavior**: < 5% overhead.
- **Potential Failure Modes**:
  - > 20% overhead
- **Mitigation Strategies**:
  - Sampling, conditional logging

---

## Summary Statistics

| Category | Count | Key Focus Areas |
|----------|-------|-----------------|
| Normal Operations | 35 | Single/batch/chain execution, data types, hardware targets |
| Edge Cases | 35 | Boundaries, alignment, special modes, registry/cache edges |
| Failure Scenarios | 35 | Crashes, OOM, corruption, distributed failures |
| Concurrency | 35 | Thread safety, hot-reload, races, deadlocks |
| Performance | 35 | Latency, throughput, memory, tail latency |
| **Total** | **175** | |

---

## Implementation Priority

### P0 - Critical (Block release)
- NO-001 through NO-010: Basic dispatch functionality
- EC-001 through EC-004: Input validation
- FS-001 through FS-010: Error handling
- CS-001 through CS-005: Basic thread safety
- PS-001 through PS-005: Latency requirements

### P1 - High (Required for production)
- NO-011 through NO-025: Extended normal operations
- EC-005 through EC-020: Memory and device edge cases
- FS-011 through FS-025: Resource and config failures
- CS-006 through CS-020: Advanced concurrency
- PS-006 through PS-020: Cache and GPU performance

### P2 - Medium (Required for scale)
- NO-026 through NO-035: Full hardware support
- EC-021 through EC-035: Policy and registry edges
- FS-026 through FS-035: Resource exhaustion
- CS-021 through CS-035: Stress and async
- PS-021 through PS-035: Tail latency and throughput

---

## Test Implementation Notes

1. **Test Framework**: Use pytest with pytest-asyncio for async tests
2. **Fixtures**: Create reusable fixtures for registry, cache, context setup
3. **Mocking**: Mock GPU operations for CPU-only test environments
4. **Benchmarking**: Use pytest-benchmark for performance tests
5. **Coverage**: Target 100% branch coverage for core dispatch logic
6. **CI Integration**: Run full suite on every PR, subset on every commit

## Related Files

- `/home/bud/Desktop/bud_waav/LayerZero/src/layerzero/selection/engine.py` - SelectionEngine
- `/home/bud/Desktop/bud_waav/LayerZero/src/layerzero/selection/mvcc_cache.py` - MVCCShardedCache
- `/home/bud/Desktop/bud_waav/LayerZero/src/layerzero/registry/kernel_registry.py` - KernelRegistry
- `/home/bud/Desktop/bud_waav/LayerZero/src/layerzero/models/kernel_spec.py` - KernelSpec
- `/home/bud/Desktop/bud_waav/LayerZero/src/layerzero/_solve/dispatch_table.py` - DispatchTable
