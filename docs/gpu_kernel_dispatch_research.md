# GPU Kernel Dispatch Systems: Comprehensive Research

## Executive Summary

This document provides an in-depth analysis of GPU kernel dispatch mechanisms across major platforms: NVIDIA CUDA, AMD ROCm/HIP, Vulkan Compute, Apple Metal, and WebGPU. Understanding these dispatch systems is critical for building high-performance inference engines like Bud Waav, where kernel launch overhead can become a significant bottleneck for real-time audio processing.

**Key Findings:**
- CUDA kernel launch overhead: ~10-25 microseconds per launch
- CUDA Graphs can reduce this to ~2.5 microseconds + ~1ns per node
- Persistent kernels eliminate launch overhead entirely for iterative workloads
- Indirect dispatch (Vulkan/Metal) enables GPU-driven command generation
- Kernel fusion remains the most effective overhead reduction technique

---

## Table of Contents

1. [NVIDIA CUDA Dispatch Mechanisms](#1-nvidia-cuda-dispatch-mechanisms)
2. [AMD ROCm/HIP Dispatch](#2-amd-rocmhip-dispatch)
3. [Vulkan Compute Dispatch](#3-vulkan-compute-dispatch)
4. [Apple Metal Compute](#4-apple-metal-compute)
5. [WebGPU Dispatch Patterns](#5-webgpu-dispatch-patterns)
6. [Performance Comparison Matrix](#6-performance-comparison-matrix)
7. [Best Practices and Recommendations](#7-best-practices-and-recommendations)

---

## 1. NVIDIA CUDA Dispatch Mechanisms

### 1.1 cudaLaunchKernel Internals

The CUDA kernel launch process involves several stages:

```
CPU Host Code
      |
      v
cudaLaunchKernel() or <<<>>> syntax
      |
      v
+-------------------+
| CUDA Runtime API  |  (or Driver API for cuLaunchKernel)
+-------------------+
      |
      v
+-------------------+
| CUDA Driver       |  - Command buffer encoding
+-------------------+  - Parameter validation
      |               - Resource binding
      v
+-------------------+
| GPU Command Queue |  - DMA transfer of launch params
+-------------------+  - Grid configuration
      |
      v
+-------------------+
| GPU Scheduler     |  - Block distribution to SMs
+-------------------+  - Warp scheduling
      |
      v
+-------------------+
| SM Execution      |
+-------------------+
```

**Overhead Components:**

| Component | Typical Latency | Notes |
|-----------|-----------------|-------|
| CPU wrapper overhead | 2-5 us | Mutex locks, parameter validation |
| Driver overhead | 5-10 us | Command buffer encoding |
| Launch latency | 10-20 us | Time from API call to kernel start |
| **Total** | **~20 us** | Measured on V100/A100 |

**Sources:**
- [NVIDIA Nsight Systems Overhead Visualization](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)
- [NVIDIA Developer Forums - Kernel Launch Latency](https://forums.developer.nvidia.com/t/kernel-launch-latency/62455)

### 1.2 CUDA Graphs for Kernel Dispatch

CUDA Graphs separate graph definition from execution, enabling significant overhead reduction.

**How CUDA Graphs Work:**

```cpp
// Phase 1: Capture (one-time setup cost)
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
for (int i = 0; i < N; i++) {
    kernel_a<<<grid, block, 0, stream>>>(params_a);
    kernel_b<<<grid, block, 0, stream>>>(params_b);
}
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// Phase 2: Launch (very fast, repeated many times)
for (int iter = 0; iter < iterations; iter++) {
    cudaGraphLaunch(graphExec, stream);  // Single launch for entire graph
}
```

**Performance Characteristics:**

| Metric | Individual Launches | CUDA Graphs | Improvement |
|--------|---------------------|-------------|-------------|
| Per-kernel overhead | ~3.8 us | ~3.4 us | ~10% |
| Effective kernel time | 3.8 us | 2.9 us (execution only) | ~24% |
| CPU overhead (10 nodes) | 100+ us | ~2.5 us | **40x** |
| Scaling (n nodes) | O(n) | O(1) + 1ns/node | **Constant time** |

**CUDA Toolkit 12.6 Improvements:**
- Straight-line graphs achieve nearly constant 2.5us launch time
- 25-40% faster instantiation
- Up to 15% better repeat launch performance

**When to Use CUDA Graphs:**

| Use Case | Benefit | Notes |
|----------|---------|-------|
| Small batch inference | High | CPU overhead dominates |
| Iterative algorithms | High | Same graph repeated |
| Multi-kernel pipelines | Medium | Fixed dependency graph |
| Dynamic workloads | Low | Graph rebuild overhead |

**Sources:**
- [NVIDIA CUDA Graphs Blog](https://developer.nvidia.com/blog/cuda-graphs/)
- [Constant Time Launch for Straight-Line CUDA Graphs](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)
- [ArXiv: Kernel Batching with CUDA Graphs](https://arxiv.org/html/2501.09398v1)

### 1.3 Persistent Kernels Pattern

Persistent kernels eliminate launch overhead entirely by keeping a kernel running continuously.

**Traditional vs Persistent Approach:**

```cpp
// Traditional: Launch N times
for (int step = 0; step < N; step++) {
    process_step<<<grid, block>>>(data, step);  // N launches = N * 20us overhead
    cudaDeviceSynchronize();
}

// Persistent: Launch once
__global__ void persistent_kernel(volatile int* signal, float* data) {
    // Get SM count for grid sizing
    int num_sms;
    cudaGetDeviceAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

    while (true) {
        // Wait for work signal
        if (threadIdx.x == 0) {
            while (*signal == 0) { /* spin */ }
        }
        __syncthreads();

        // Process work unit
        process_work_unit(data);

        // Check for termination
        if (*signal == -1) return;
    }
}
```

**Grid Sizing for Persistent Kernels:**

```cpp
int num_SMs;
cudaGetDeviceAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device_id);
dim3 dimGrid(num_SMs);  // One CTA per SM
```

**PERKS (PERsistent KernelS) Scheme:**

Research has shown that for memory-bound iterative GPU applications:
- Move time loop inside persistent kernel
- Use device-wide barriers for synchronization
- Cache subset of output in registers/shared memory
- **Results:** 2.29x speedup on A100, 1.53x on V100 for small domains
- **Conjugate Gradient:** 2.47x speedup vs Ginkgo library

**Considerations:**
- Requires occupancy-aware grid sizing
- Signal mechanism for work dispatch
- Not universally supported on all hardware/drivers
- May conflict with GPU scheduling policies

**Sources:**
- [NVIDIA Forums - Persistent Kernels Discussion](https://forums.developer.nvidia.com/t/are-persistent-kernels-supported-now-and-in-the-future/288444)
- [CUTLASS Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
- [Concurrent-RT: Improving Real-Time Performance with Persistent Threads](https://concurrent-rt.com/wp-content/uploads/2020/12/Improving-Real-Time-Performance-With-CUDA-Persistent-Threads.pdf)

### 1.4 Dynamic Parallelism

Dynamic parallelism allows kernels to launch child kernels from the device.

```cpp
__global__ void parent_kernel(float* data, int n) {
    // Process local work
    process_local(data);

    // Launch child kernel from device
    if (threadIdx.x == 0 && need_more_work()) {
        child_kernel<<<child_grid, child_block>>>(data + offset, child_n);
        cudaDeviceSynchronize();  // Wait for children
    }
}
```

**Performance Considerations:**

| Aspect | Impact | Notes |
|--------|--------|-------|
| Launch overhead | Same as host (~20 us) | No reduction from device launch |
| Nested depth | Limited to 24 | Kepler architecture limit |
| Small grids | Severe underutilization | Launch many threads to justify overhead |
| Resource contention | High | Many concurrent child launches |

**When Dynamic Parallelism Helps:**
- High nested parallelism (launch overhead << parallel work)
- Irregular/recursive algorithms (tree traversal, adaptive mesh)
- Reducing CPU-GPU round trips

**Optimization: Workload Consolidation**
- Aggregate kernels from multiple threads into single consolidated launch
- Reduces DP overhead
- Increases GPU utilization

**Sources:**
- [NVIDIA Dynamic Parallelism API](https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/)
- [HPCA 2017: Controlled Kernel Launch for Dynamic Parallelism](https://adwaitjog.github.io/docs/pdf/Controlled-DP-HPCA-2017.pdf)

### 1.5 Cooperative Groups and Grid Synchronization

CUDA Cooperative Groups enable grid-wide synchronization without kernel re-launch.

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cooperative_kernel(float* data) {
    cg::grid_group grid = cg::this_grid();

    // Phase 1: All threads process
    process_phase1(data);

    // Grid-wide barrier (instead of kernel re-launch)
    grid.sync();

    // Phase 2: All threads continue
    process_phase2(data);
}

// Launch with cooperative kernel API
void* args[] = {&data};
cudaLaunchCooperativeKernel(
    (void*)cooperative_kernel,
    grid_dim, block_dim, args, 0, stream
);
```

**Requirements:**
- Use `cudaLaunchCooperativeKernel` API
- Grid size <= maximum active blocks on device
- Cannot exceed occupancy limits

**Note:** As of CUDA 13, cooperative groups cannot be used for multi-device synchronization.

**Sources:**
- [NVIDIA Cooperative Groups Blog](https://developer.nvidia.com/blog/cooperative-groups/)
- [CUDA Programming Guide - Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)

### 1.6 Stream-Ordered Memory Allocator (cudaMallocAsync)

CUDA 11.2 introduced stream-ordered allocation to eliminate synchronization overhead.

```cpp
// Traditional (synchronous, blocks device)
float* buffer;
cudaMalloc(&buffer, size);  // Implicit device sync
kernel<<<grid, block>>>(buffer);
cudaFree(buffer);  // Implicit device sync

// Stream-ordered (asynchronous)
float* buffer;
cudaMallocAsync(&buffer, size, stream);  // No sync
kernel<<<grid, block, 0, stream>>>(buffer);
cudaFreeAsync(buffer, stream);  // No sync
```

**Performance Impact:**
- 2-5x improvement for GPU Big Data Benchmark queries
- Eliminates device-wide synchronization
- Enables memory pooling and reuse

**Memory Pool Configuration:**
```cpp
cudaMemPool_t pool;
cudaDeviceGetDefaultMemPool(&pool, device);

// Configure release threshold (bytes to hold before returning to OS)
size_t threshold = 1024 * 1024 * 1024;  // 1 GB
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
```

**Sources:**
- [NVIDIA Stream-Ordered Memory Allocator Part 1](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/)
- [NVIDIA Stream-Ordered Memory Allocator Part 2](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2/)

### 1.7 Kernel Fusion

Kernel fusion combines multiple operations into a single kernel, eliminating inter-kernel overhead.

**Without Fusion:**
```cpp
// 3 kernel launches, 3 global memory round-trips
kernel_a<<<...>>>(input, temp1);   // Write temp1 to global
kernel_b<<<...>>>(temp1, temp2);   // Read temp1, write temp2
kernel_c<<<...>>>(temp2, output);  // Read temp2, write output
// Total: ~60 us launch overhead + memory bandwidth
```

**With Fusion:**
```cpp
// 1 kernel launch, data stays in registers/shared memory
__global__ void fused_kernel(float* input, float* output) {
    float reg = input[idx];       // Load once
    reg = operation_a(reg);       // In-register
    reg = operation_b(reg);       // In-register
    output[idx] = operation_c(reg);  // Store once
}
// Total: ~20 us launch overhead, minimal memory traffic
```

**Fusion Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Vertical fusion | Chain of dependent ops | Element-wise sequences |
| Horizontal fusion | Parallel independent ops | Mixed compute/memory kernels |
| CUDA Graphs batching | Group 50-100 nodes | Iterative algorithms |

**Real-World Examples:**
- **FlashAttention:** Fused attention achieves 20-50% FLOPs/s improvement
- **Compiler fusion:** 1.4x+ improvement from CUDA Graph batching

**Sources:**
- [CUDA Kernel Fusion Strategies](https://www.emergentmind.com/topics/cuda-kernel-fusion)
- [Part VI - Kernel Fusion in CUDA](https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-vi---kernel-fusion-in-cuda)

---

## 2. AMD ROCm/HIP Dispatch

### 2.1 HIP Kernel Launch Mechanisms

HIP provides CUDA-compatible kernel launch syntax with ROCm backend optimizations.

**Launch Syntax:**
```cpp
// CUDA-style syntax (default since ROCm 5.3)
kernel<<<grid, block, shared_mem, stream>>>(args);

// HIP-specific function (still supported)
hipLaunchKernelGGL(kernel, grid, block, shared_mem, stream, args);
```

**Architecture Stack:**

```
HIP API Layer
      |
      v
+-------------------+
| ROCclr Runtime    |  - Abstraction layer
+-------------------+
      |
      v
+-------------------+
| HSA Runtime       |  - Heterogeneous System Architecture
+-------------------+  - Queue management, memory regions
      |
      v
+-------------------+
| Kernel Fusion     |  - /dev/kfd driver interface
| Driver (KFD)      |  - AQL packet submission
+-------------------+
```

### 2.2 Key Differences from CUDA

| Feature | CUDA | HIP/ROCm | Notes |
|---------|------|----------|-------|
| Dynamic Parallelism | Supported | **Not Supported** | Major limitation |
| Warp Size | 32 threads | 64 threads (most AMD GPUs) | Code portability issue |
| Warp Mask Type | 32-bit | 64-bit always | Higher bits unused on 32-thread devices |
| Kernel Binary | PTX/CUBIN | HSACO | Different compilation pipeline |
| Default Warpsize | Constant | Runtime-folded (ROCm 7.0+) | Better loop unrolling |

### 2.3 HIP Graphs

HIP graphs mirror CUDA Graphs API for reduced launch overhead.

```cpp
hipGraph_t graph;
hipGraphExec_t graphExec;

// Capture
hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
kernel_a<<<grid, block, 0, stream>>>(args_a);
kernel_b<<<grid, block, 0, stream>>>(args_b);
hipStreamEndCapture(stream, &graph);
hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// Launch
hipGraphLaunch(graphExec, stream);
```

**ROCm 7.x Optimizations:**

| Feature | Version | Improvement |
|---------|---------|-------------|
| Doorbell ring optimization | 7.1 | Efficient packet batching for graph launches |
| Reduced module-load latency | 7.1 | Lower time-to-first-kernel |
| Back memset optimization | 7.2 | Better memset node processing |
| Optimized AQL batch submission | 7.2 | Dynamic packet copying, staggered patterns |

**Sources:**
- [HIP Graphs Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)
- [ROCm 7.1 Blog](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.1/README.html)

### 2.4 Performance Guidelines

```cpp
// Occupancy-aware launch
int block_size = 256;
int grid_size;
hipOccupancyMaxActiveBlocksPerMultiprocessor(
    &grid_size, kernel, block_size, 0
);

// Coalesced memory access (64 threads per wavefront on AMD)
// Align to 64-thread groups for optimal memory bandwidth
```

**Sources:**
- [HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)

---

## 3. Vulkan Compute Dispatch

### 3.1 Command Buffer Dispatch

Vulkan uses explicit command buffer recording for compute dispatch.

```cpp
// Direct dispatch
vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
vkCmdDispatch(cmdBuffer, groupCountX, groupCountY, groupCountZ);

// Indirect dispatch (GPU-driven)
vkCmdDispatchIndirect(cmdBuffer, indirectBuffer, offset);
```

**Command Buffer Lifecycle:**

```
1. vkAllocateCommandBuffers() - Allocate from pool
2. vkBeginCommandBuffer() - Start recording
3. vkCmdBindPipeline() - Bind compute pipeline
4. vkCmdBindDescriptorSets() - Bind resources
5. vkCmdDispatch() or vkCmdDispatchIndirect() - Record dispatch
6. vkEndCommandBuffer() - Finish recording
7. vkQueueSubmit() - Submit to GPU queue
8. Reuse or reset command buffer
```

### 3.2 Indirect Dispatch Patterns

Indirect dispatch allows GPU to determine workgroup counts.

```cpp
// Indirect dispatch buffer structure
struct VkDispatchIndirectCommand {
    uint32_t x;  // Workgroup count X
    uint32_t y;  // Workgroup count Y
    uint32_t z;  // Workgroup count Z
};

// Create buffer with VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
VkBufferCreateInfo bufferInfo = {
    .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    // ...
};

// Dispatch with GPU-written parameters
vkCmdDispatchIndirect(cmdBuffer, indirectBuffer, 0);
```

**Use Cases:**
- GPU-driven culling (compute shader determines visible objects)
- Dynamic workload generation
- Hierarchical algorithms with variable work

### 3.3 Pipeline Caching

Vulkan pipeline caching amortizes shader compilation cost.

```cpp
// Create pipeline cache
VkPipelineCacheCreateInfo cacheInfo = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    .initialDataSize = cachedDataSize,
    .pInitialData = cachedData,  // From previous run
};
VkPipelineCache pipelineCache;
vkCreatePipelineCache(device, &cacheInfo, nullptr, &pipelineCache);

// Create compute pipeline with cache
VkComputePipelineCreateInfo pipelineInfo = { /* ... */ };
vkCreateComputePipelines(device, pipelineCache, 1, &pipelineInfo,
                         nullptr, &computePipeline);

// Save cache for next run
size_t dataSize;
vkGetPipelineCacheData(device, pipelineCache, &dataSize, nullptr);
void* data = malloc(dataSize);
vkGetPipelineCacheData(device, pipelineCache, &dataSize, data);
// Write to disk
```

**Performance Impact:**
- First pipeline creation: Full shader compilation (ms-seconds)
- Cached creation: Near-instantaneous
- Critical for reducing startup time and eliminating runtime stutters

**VK_KHR_pipeline_binary Extension:**
- Direct binary retrieval per-pipeline
- Async binary creation
- Eliminates micro-stutters during pipeline creation

### 3.4 Synchronization and Barriers

```cpp
// Memory barrier between compute passes
VkMemoryBarrier memoryBarrier = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
};

vkCmdPipelineBarrier(
    cmdBuffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // src stage
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // dst stage
    0, 1, &memoryBarrier, 0, nullptr, 0, nullptr
);
```

**Best Practice:** Group barriers together for maximum performance. Batched barriers allow GPU to ramp up/down execution units once instead of repeatedly.

### 3.5 VComputeBench Findings

Research comparing Vulkan vs CUDA/OpenCL for compute:

| Optimization | Impact |
|--------------|--------|
| Single command buffer for iterations | Reduced launch overhead vs CUDA |
| Memory barriers vs kernel re-launch | Better iteration performance |
| Batch submission | Reduced CPU-GPU communication |

**Key Insight:** Recording all iteration work in one command buffer with memory barriers between iterations can outperform CUDA's multiple kernel launches.

**Sources:**
- [Vulkan Dispatching Commands](https://docs.vulkan.org/spec/latest/chapters/dispatch.html)
- [Vulkan Pipeline Cache Guide](https://docs.vulkan.org/guide/latest/pipeline_cache.html)
- [NVIDIA Vulkan Dos and Don'ts](https://developer.nvidia.com/blog/vulkan-dos-donts/)

---

## 4. Apple Metal Compute

### 4.1 Metal Compute Command Encoding

```objc
// Create compute command encoder
id<MTLComputeCommandEncoder> computeEncoder =
    [commandBuffer computeCommandEncoder];

// Set pipeline and resources
[computeEncoder setComputePipelineState:pipelineState];
[computeEncoder setBuffer:inputBuffer offset:0 atIndex:0];
[computeEncoder setBuffer:outputBuffer offset:0 atIndex:1];

// Dispatch threadgroups
MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
MTLSize threadgroupCount = MTLSizeMake(
    (dataSize + 255) / 256, 1, 1
);
[computeEncoder dispatchThreadgroups:threadgroupCount
               threadsPerThreadgroup:threadgroupSize];

[computeEncoder endEncoding];
[commandBuffer commit];
```

### 4.2 Argument Buffers for Zero-Overhead Dispatch

Argument buffers aggregate resource bindings for efficient GPU access.

```objc
// Define argument buffer structure
typedef struct {
    texture2d<float> inputTexture;
    texture2d<float> outputTexture;
    constant Params* params;
} ArgumentData;

// Create argument encoder
id<MTLArgumentEncoder> argEncoder =
    [function newArgumentEncoderWithBufferIndex:0];
id<MTLBuffer> argBuffer =
    [device newBufferWithLength:argEncoder.encodedLength
                        options:MTLResourceStorageModeShared];

// Encode arguments
[argEncoder setArgumentBuffer:argBuffer offset:0];
[argEncoder setTexture:inputTexture atIndex:0];
[argEncoder setTexture:outputTexture atIndex:1];
[argEncoder setBuffer:paramsBuffer offset:0 atIndex:2];

// Dispatch with argument buffer
[computeEncoder setBuffer:argBuffer offset:0 atIndex:0];
```

**Benefits:**
- Reduced CPU encoding overhead
- GPU can access resources directly
- Enables GPU-driven rendering/compute

### 4.3 Indirect Command Buffers (ICB)

Metal's ICB enables GPU-driven command generation.

```objc
// Create indirect command buffer descriptor
MTLIndirectCommandBufferDescriptor* desc =
    [[MTLIndirectCommandBufferDescriptor alloc] init];
desc.commandTypes = MTLIndirectCommandTypeDispatch;
desc.inheritPipelineState = YES;
desc.inheritBuffers = YES;

// Create ICB
id<MTLIndirectCommandBuffer> icb =
    [device newIndirectCommandBufferWithDescriptor:desc
                                  maxCommandCount:16384
                                          options:0];

// GPU writes commands to ICB in compute shader
// Then execute ICB
[computeEncoder executeCommandsInBuffer:icb
                              withRange:NSMakeRange(0, commandCount)];
```

**Performance Benchmarks (Tellusim):**

| Platform | ICB vs Loop | Notes |
|----------|-------------|-------|
| M1 | **39% faster** | Small primitives benefit most |
| A14 | **44% faster** | Mobile advantage |
| AMD (Mac) | 18% slower | ICB overhead > loop overhead |

**Crossover Point:** ICB performs better when draw calls contain <200 primitives. Above that, traditional loops win.

**Limitations:**
- Maximum 16,384 commands per ICB
- May require multiple ICB executions for large scenes
- GPU-CPU sync required on some configurations

### 4.4 Apple Silicon Optimization

**Cache Architecture:**
- Separate L1 caches for texture reads vs buffer reads
- Moving buffer data to textures can improve performance
- Better utilization of high-performance texture caches

**Dispatch Model:**
- **Serial mode** (default): Commands execute in order
- **Concurrent mode**: For small thread counts, enables parallel command execution

**Sources:**
- [Apple MTLComputeCommandEncoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder)
- [Apple Indirect Command Encoding](https://developer.apple.com/documentation/metal/indirect_command_encoding)
- [Tellusim Metal MDI Benchmarks](https://tellusim.com/metal-mdi/)

---

## 5. WebGPU Dispatch Patterns

### 5.1 Basic Compute Dispatch

```javascript
// Create compute pipeline
const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
        module: shaderModule,
        entryPoint: 'main',
    },
});

// Create bind group
const bindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
    ],
});

// Encode and dispatch
const commandEncoder = device.createCommandEncoder();
const passEncoder = commandEncoder.beginComputePass();
passEncoder.setPipeline(computePipeline);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
passEncoder.end();
device.queue.submit([commandEncoder.finish()]);
```

### 5.2 Workgroup Size Guidelines

```wgsl
// WGSL compute shader
@compute @workgroup_size(64, 1, 1)  // 64 is recommended default
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Computation
}
```

**Recommendations:**
- Default workgroup size: **64** (fast path on most GPUs)
- Multiple threads within workgroup faster than separate dispatches
- Threads often run in lockstep (16 threads ~ same cost as 1)

### 5.3 Performance Optimization Patterns

**Double-Buffering (Ping-Pong):**
```javascript
// Two buffers to avoid read-after-write hazards
const bufferA = device.createBuffer({ /* ... */ });
const bufferB = device.createBuffer({ /* ... */ });

// Frame N: Read A, Write B
// Frame N+1: Read B, Write A
let readBuffer = bufferA, writeBuffer = bufferB;

function frame() {
    // Dispatch with current buffers
    dispatchCompute(readBuffer, writeBuffer);
    // Swap
    [readBuffer, writeBuffer] = [writeBuffer, readBuffer];
}
```

**Minimize GPU-CPU Transfers:**
- Keep data on GPU longer
- Batch aggressively
- Data transfer overhead can eclipse compute gains

### 5.4 WGSL Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| No 64-bit integers | Modular arithmetic complexity | Manual decomposition |
| Runtime-sized arrays only at buffer end | Data structure constraints | Careful buffer layout |
| Limited texture formats | Some operations require emulation | Use compatible formats |

### 5.5 Real-World Performance

| Application | Performance | Notes |
|-------------|-------------|-------|
| Matrix multiplication (M1) | ~1.2 TFLOPs | vs 10.4 TFLOPs theoretical peak |
| ZK proofs (Stwo prover) | 5x speedup | With compute shader integration |
| AI inference | 10% overhead | SafeRace memory safety measures |

**Sources:**
- [WebGPU Fundamentals - Compute Shaders](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html)
- [surma.dev - WebGPU](https://surma.dev/things/webgpu/)
- [Optimizing WebGPU Matmul for 1TFLOP+](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel)

---

## 6. Performance Comparison Matrix

### 6.1 Kernel Launch Overhead

| Platform | Individual Launch | Batched/Graph | Persistent |
|----------|-------------------|---------------|------------|
| CUDA | ~20 us | ~2.5 us (graphs) | 0 us (no re-launch) |
| HIP/ROCm | ~20 us | ~2.5 us (hipGraphs) | Similar to CUDA |
| Vulkan | ~10-15 us | Single submit | N/A |
| Metal | ~5-10 us | ICB batching | N/A |
| WebGPU | ~10-20 us | Command buffer batching | N/A |

### 6.2 Feature Comparison

| Feature | CUDA | ROCm/HIP | Vulkan | Metal | WebGPU |
|---------|------|----------|--------|-------|--------|
| Graph/ICB batching | Yes | Yes | Implicit | Yes (ICB) | Manual |
| Indirect dispatch | Yes | Yes | Yes | Yes | Limited |
| Persistent kernels | Yes | Yes | N/A | N/A | N/A |
| Dynamic parallelism | Yes | **No** | N/A | N/A | N/A |
| Grid sync | Yes (coop groups) | Yes | Barriers | Barriers | Barriers |
| Pipeline caching | Implicit (driver) | Implicit | Explicit | Implicit | Implicit |
| Stream-ordered alloc | Yes | Yes | Manual | Manual | Manual |

### 6.3 Optimization Techniques by Platform

| Technique | CUDA | ROCm | Vulkan | Metal | WebGPU |
|-----------|------|------|--------|-------|--------|
| **Reduce launches** | CUDA Graphs | hipGraphs | Single cmd buffer | ICB | Batch commands |
| **Eliminate launches** | Persistent kernels | Persistent kernels | N/A | N/A | N/A |
| **GPU-driven dispatch** | Dynamic parallelism | N/A | Indirect dispatch | ICB | Limited |
| **Reduce sync** | Streams, async | Streams | Barriers | Fences | Commands |
| **Memory overhead** | cudaMallocAsync | hipMallocAsync | Manual pools | MTLHeap | Manual pools |
| **Shader compilation** | Driver cache | Driver cache | Pipeline cache | Driver cache | Browser cache |

---

## 7. Best Practices and Recommendations

### 7.1 For Bud Waav Inference Engine

Given the real-time audio constraints (< 10ms latency budget), the following dispatch strategies are recommended:

**1. Use CUDA Graphs for STT/TTS Pipelines**
```cpp
// Capture the full inference pipeline
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
whisper_encoder<<<...>>>(audio_features);
attention_layers(stream);  // Multiple attention kernels
whisper_decoder<<<...>>>(tokens);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// Per-inference: single launch
cudaGraphLaunch(graphExec, stream);  // ~2.5 us vs ~200+ us for individual launches
```

**2. Persistent Kernels for Streaming Audio**
```cpp
// Single persistent kernel for real-time audio processing
__global__ void audio_processor(AudioRingBuffer* input,
                                 AudioRingBuffer* output,
                                 volatile int* signal) {
    while (*signal != TERMINATE) {
        if (input->has_data()) {
            process_chunk(input, output);
        }
    }
}
```

**3. Kernel Fusion for Attention Layers**
- Use FlashAttention-style fused kernels
- Eliminate intermediate memory traffic
- Keep data in registers/shared memory

**4. Stream-Ordered Allocation**
```cpp
// Avoid cudaMalloc in hot path
cudaMallocAsync(&workspace, size, inference_stream);
inference_kernel<<<...>>>(workspace);
cudaFreeAsync(workspace, inference_stream);
```

### 7.2 Platform-Specific Recommendations

**NVIDIA CUDA:**
1. Profile with Nsight Systems to identify launch-bound regions
2. Use CUDA Graphs for repetitive kernel sequences (50-100 nodes optimal)
3. Consider persistent kernels for real-time streaming
4. Use cudaMallocAsync to eliminate synchronization
5. Fuse kernels where possible (FlashAttention pattern)

**AMD ROCm:**
1. Use hipGraphs for same benefits as CUDA Graphs
2. Account for 64-thread wavefronts in occupancy calculations
3. Leverage ROCm 7.x doorbell optimizations
4. Note: No dynamic parallelism available

**Vulkan:**
1. Record entire workload in single command buffer
2. Use memory barriers instead of command buffer boundaries
3. Implement pipeline caching for shader compilation
4. Consider indirect dispatch for variable workloads

**Metal:**
1. Use ICBs for GPU-driven command generation
2. Leverage argument buffers for resource binding
3. Use texture cache for better bandwidth utilization
4. Note: ICB has 16,384 command limit

**WebGPU:**
1. Minimize CPU-GPU buffer transfers
2. Use 64-thread workgroup size as default
3. Implement double-buffering for iterative workloads
4. Batch dispatches in single command encoder

### 7.3 Latency Budget Allocation

For Bud Waav's 200ms end-to-end budget:

| Stage | Budget | Dispatch Strategy |
|-------|--------|-------------------|
| Audio capture | <10 ms | DMA, ring buffers |
| Preprocessing | <5 ms | Fused kernel |
| STT inference | <50 ms | CUDA Graph |
| Processing | <10 ms | Fused kernel |
| TTS synthesis | <100 ms | CUDA Graph |
| Audio output | <5 ms | DMA, ring buffers |
| **Overhead/margin** | **20 ms** | |

### 7.4 Profiling and Measurement

**CUDA:**
```bash
# Profile kernel launches
nsys profile --trace=cuda ./inference_app
nsys stats report.nsys-rep

# Detailed kernel analysis
ncu --set full ./inference_app
```

**Key Metrics:**
- Launch latency (time from API call to kernel start)
- Kernel duration
- Memory throughput
- Occupancy
- Warp stalls

---

## References

### NVIDIA CUDA
- [NVIDIA Nsight Systems Overhead Visualization](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)
- [CUDA Graphs Blog](https://developer.nvidia.com/blog/cuda-graphs/)
- [Constant Time Launch for CUDA Graphs](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)
- [Cooperative Groups](https://developer.nvidia.com/blog/cooperative-groups/)
- [Stream-Ordered Memory Allocator](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/)
- [Dynamic Parallelism API](https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/)

### AMD ROCm
- [HIP Graphs Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)
- [HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)
- [ROCm 7.1 Improvements](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.1/README.html)

### Vulkan
- [Vulkan Dispatching Commands](https://docs.vulkan.org/spec/latest/chapters/dispatch.html)
- [Pipeline Cache Guide](https://docs.vulkan.org/guide/latest/pipeline_cache.html)
- [NVIDIA Vulkan Tips](https://developer.nvidia.com/blog/vulkan-dos-donts/)

### Apple Metal
- [MTLComputeCommandEncoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder)
- [Indirect Command Encoding](https://developer.apple.com/documentation/metal/indirect_command_encoding)
- [Tellusim Metal Benchmarks](https://tellusim.com/metal-mdi/)

### WebGPU
- [WebGPU Compute Shaders](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html)
- [WebGPU Overview](https://surma.dev/things/webgpu/)
- [WebGPU Matmul Optimization](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel)

### Research Papers
- [ArXiv: Kernel Batching with CUDA Graphs](https://arxiv.org/html/2501.09398v1)
- [HPCA 2017: Controlled Kernel Launch for Dynamic Parallelism](https://adwaitjog.github.io/docs/pdf/Controlled-DP-HPCA-2017.pdf)
- [PERKS: Persistent Kernels](https://deepai.org/publication/persistent-kernels-for-iterative-memory-bound-gpu-applications)

---

*Document created: 2026-02-04*
*For Bud Waav Inference Engine Development*
