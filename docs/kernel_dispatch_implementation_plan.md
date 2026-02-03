# State-of-the-Art Kernel Dispatch System Implementation Plan

## Executive Summary

This document presents a comprehensive implementation plan for a production-ready, near-zero overhead, scalable, and pluggable kernel dispatch system for LayerZero/Bud Waav. The plan is based on extensive research across academic papers, production implementations (PyTorch, TensorFlow, ONNX Runtime, TensorRT, TVM), and analysis of 175+ operational scenarios.

---

## 1. Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│
│  │   REST API  │  │  WebSocket  │  │    gRPC     │  │  Native (PyO3/CXX)  ││
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘│
└─────────┼────────────────┼────────────────┼────────────────────┼───────────┘
          │                │                │                    │
          └────────────────┴────────────────┴────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RUST GATEWAY (SAYNA)                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Request Router & Load Balancer                     │  │
│  └─────────────────────────────────┬────────────────────────────────────┘  │
│                                    │                                        │
│  ┌─────────────┐  ┌─────────────┐  │  ┌─────────────┐  ┌─────────────────┐│
│  │   Metrics   │  │   Tracing   │  │  │   Circuit   │  │   Rate Limiter  ││
│  │  Collector  │  │   (OTel)    │  │  │   Breaker   │  │   (Backpressure)││
│  └─────────────┘  └─────────────┘  │  └─────────────┘  └─────────────────┘│
│                                    │                                        │
│  ┌─────────────────────────────────▼────────────────────────────────────┐  │
│  │              KERNEL DISPATCH ORCHESTRATOR (Rust)                      │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Dispatch Strategy Selector                    │ │  │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐│ │  │
│  │  │  │ Static  │  │ Dynamic │  │   Hot   │  │   Config-Driven     ││ │  │
│  │  │  │ Dispatch│  │ Dispatch│  │ Reload  │  │   (YAML/Code)       ││ │  │
│  │  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────────┘│ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Resource Manager                              │ │  │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │ │  │
│  │  │  │ Memory Pool │  │  GPU Queue  │  │   Thread Pool (Rayon)   │ │ │  │
│  │  │  │   (Arena)   │  │  Manager    │  │   + Work-Stealing       │ │ │  │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────────┘ │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌─────────────────────────────────▼────────────────────────────────────┐  │
│  │                 SHARED MEMORY IPC BRIDGE                              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │  │
│  │  │ Ring Buffer │  │ Zero-Copy   │  │   Lock-Free Message Queue   │  │  │
│  │  │   (SPSC)    │  │   Tensors   │  │   (crossbeam/atomic_queue)  │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ IPC (Shared Memory)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PYTHON INFERENCE ENGINE                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                 LAYERZERO KERNEL SELECTION ENGINE                     │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Selection Pipeline                            │ │  │
│  │  │  Policy → Cache → Filter → Score → Select → Execute → Cache     │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │  │
│  │  │   Kernel    │  │    MVCC     │  │      Policy Engine          │  │  │
│  │  │  Registry   │  │   Cache     │  │   (Lock/Deny/Allow/Boost)   │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │  │
│  │                                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │  │
│  │  │   PerfDB    │  │  Telemetry  │  │      Health Monitor         │  │  │
│  │  │  (SQLite)   │  │  (Metrics)  │  │   (Circuit Breaker)         │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌─────────────────────────────────▼────────────────────────────────────┐  │
│  │                    KERNEL EXECUTION LAYER                             │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                 Backend Adapters (30+)                           │ │  │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │ │  │
│  │  │  │  Flash  │ │  Flash  │ │xFormers │ │  Liger  │ │ Triton  │  │ │  │
│  │  │  │ Attn v2 │ │ Infer   │ │         │ │         │ │ Custom  │  │ │  │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │ │  │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │ │  │
│  │  │  │  torch  │ │  cuDNN  │ │  oneDNN │ │  HIPBLAS│ │  Metal  │  │ │  │
│  │  │  │  SDPA   │ │ (fused) │ │  (CPU)  │ │  (ROCm) │ │ (macOS) │  │ │  │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Dispatch Modes

| Mode | Use Case | Overhead | Flexibility |
|------|----------|----------|-------------|
| **Static** | Production with known kernels | Zero (compile-time) | Low |
| **Dynamic** | Runtime kernel selection | ~100-500ns | High |
| **Hot-Reload** | Development, A/B testing | ~1-10ms reload | Very High |
| **Config-Driven** | Ops-controlled deployment | ~100ns lookup | Medium |

---

## 2. Core Components

### 2.1 Static Dispatch (Rust - Zero Overhead)

For maximum performance with known kernel types, use compile-time dispatch via enum pattern:

```rust
// src/dispatch/static_dispatch.rs
use enum_dispatch::enum_dispatch;

/// Zero-overhead kernel dispatch trait
#[enum_dispatch]
pub trait KernelExecutor: Send + Sync {
    fn execute(&self, input: &TensorView, output: &mut TensorView) -> Result<(), KernelError>;
    fn name(&self) -> &'static str;
    fn supports_cuda_graph(&self) -> bool;
}

/// Compile-time kernel enum - no vtable, full inlining
#[enum_dispatch(KernelExecutor)]
pub enum BuiltinKernel {
    // Attention kernels
    FlashAttentionV2(FlashAttentionV2Kernel),
    FlashAttentionV3(FlashAttentionV3Kernel),
    FlashInfer(FlashInferKernel),
    XFormersMemEff(XFormersMemEffKernel),
    TorchSDPA(TorchSDPAKernel),

    // Normalization kernels
    RMSNorm(RMSNormKernel),
    LayerNorm(LayerNormKernel),

    // Positional encoding
    RoPE(RoPEKernel),
    ALiBi(ALiBiKernel),

    // Fused operations
    SwiGLU(SwiGLUKernel),
    FusedMLP(FusedMLPKernel),
}

impl BuiltinKernel {
    /// Static dispatch table builder - runs at compile time
    pub const fn dispatch_table() -> &'static [(&'static str, fn() -> Self)] {
        &[
            ("flash_attn_v2", || Self::FlashAttentionV2(FlashAttentionV2Kernel::new())),
            ("flash_attn_v3", || Self::FlashAttentionV3(FlashAttentionV3Kernel::new())),
            // ... other kernels
        ]
    }
}
```

### 2.2 Dynamic Dispatch with Trait Objects

For runtime extensibility (plugins, custom kernels):

```rust
// src/dispatch/dynamic_dispatch.rs
use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;

/// FFI-safe kernel interface for plugins
#[repr(C)]
pub struct KernelFFI {
    pub name: *const c_char,
    pub version: u32,
    pub execute: unsafe extern "C" fn(
        input_ptr: *const f32,
        input_len: usize,
        output_ptr: *mut f32,
        output_len: usize,
        stream: *mut c_void,
    ) -> i32,
    pub destroy: unsafe extern "C" fn(*mut c_void),
    pub user_data: *mut c_void,
}

/// Dynamic kernel registry with versioning
pub struct DynamicKernelRegistry {
    /// Kernel ID -> (Version -> Kernel)
    kernels: DashMap<String, DashMap<u32, Arc<dyn KernelExecutor>>>,
    /// Circuit breakers per kernel
    circuit_breakers: DashMap<String, CircuitBreaker>,
    /// Plugin libraries (keep loaded)
    plugins: RwLock<Vec<libloading::Library>>,
}

impl DynamicKernelRegistry {
    /// Register kernel with hot-reload support
    pub fn register(&self, kernel: Arc<dyn KernelExecutor>, version: u32) {
        let name = kernel.name().to_string();
        self.kernels
            .entry(name.clone())
            .or_default()
            .insert(version, kernel);

        // Initialize circuit breaker if new kernel
        self.circuit_breakers
            .entry(name)
            .or_insert_with(|| CircuitBreaker::new(Default::default()));
    }

    /// Get kernel with fallback chain
    pub fn get_with_fallback(
        &self,
        name: &str,
        version: Option<u32>,
        fallbacks: &[&str],
    ) -> Option<Arc<dyn KernelExecutor>> {
        // Try primary kernel
        if let Some(kernel) = self.get(name, version) {
            if self.is_circuit_closed(name) {
                return Some(kernel);
            }
        }

        // Try fallbacks
        for fallback in fallbacks {
            if let Some(kernel) = self.get(fallback, None) {
                if self.is_circuit_closed(fallback) {
                    return Some(kernel);
                }
            }
        }

        None
    }
}
```

### 2.3 Hot-Reload Mechanism

For development and zero-downtime updates:

```rust
// src/dispatch/hot_reload.rs
use notify::{Watcher, RecursiveMode, Event};
use std::sync::atomic::{AtomicPtr, Ordering};

/// Hot-reloadable kernel manager
pub struct HotReloadManager {
    /// Atomic pointer for lock-free read access
    current_registry: AtomicPtr<DynamicKernelRegistry>,
    /// File watcher for config changes
    watcher: RecommendedWatcher,
    /// Reload in progress flag
    reloading: AtomicBool,
    /// Shutdown signal
    shutdown: CancellationToken,
}

impl HotReloadManager {
    /// Reload kernels from new configuration
    pub async fn reload(&self, config: KernelConfig) -> Result<(), ReloadError> {
        // Prevent concurrent reloads
        if self.reloading.swap(true, Ordering::AcqRel) {
            return Err(ReloadError::AlreadyReloading);
        }

        defer! { self.reloading.store(false, Ordering::Release); }

        // Build new registry
        let new_registry = Box::new(DynamicKernelRegistry::from_config(&config)?);

        // Validate new kernels with warmup
        self.validate_kernels(&new_registry).await?;

        // Atomic swap
        let new_ptr = Box::into_raw(new_registry);
        let old_ptr = self.current_registry.swap(new_ptr, Ordering::AcqRel);

        // Defer cleanup of old registry (grace period for in-flight requests)
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(5)).await;
            unsafe {
                let _ = Box::from_raw(old_ptr);
            }
        });

        Ok(())
    }

    /// Watch for config file changes
    pub fn watch_config(&mut self, path: &Path) -> Result<(), WatchError> {
        let tx = self.reload_tx.clone();

        self.watcher.watch(path, RecursiveMode::NonRecursive)?;

        self.watcher.configure(notify::Config::default()
            .with_poll_interval(Duration::from_secs(1)))?;

        Ok(())
    }
}
```

### 2.4 Config-Driven Dispatch

YAML configuration for ops-controlled kernel selection:

```yaml
# config/kernels.yaml
version: "1.0.0"
schema_version: 2

defaults:
  fallback_policy: torch_sdpa
  autotuning: true
  cache_ttl_seconds: 3600

kernels:
  attention:
    operation: "attention"
    implementations:
      - name: flash_attention_v2
        dispatch_key: cuda_sm80
        priority: 100
        constraints:
          min_batch_size: 1
          max_batch_size: 256
          dtypes: [float16, bfloat16]
          min_head_dim: 32
          max_head_dim: 256
          supports_causal: true
          supports_gqa: true
          cuda_graph_safe: true

      - name: flash_infer
        dispatch_key: cuda_sm80
        priority: 95
        constraints:
          dtypes: [float16, bfloat16]
          supports_paged_kv: true

      - name: xformers_memory_efficient
        dispatch_key: cuda_sm75
        priority: 80
        constraints:
          dtypes: [float16, float32]
          supports_bias: true

      - name: torch_sdpa
        dispatch_key: any
        priority: 1
        constraints:
          dtypes: [float16, bfloat16, float32]

  rms_norm:
    operation: "rms_norm"
    implementations:
      - name: liger_rms_norm
        dispatch_key: cuda
        priority: 100

      - name: triton_rms_norm
        dispatch_key: cuda
        priority: 90

      - name: torch_rms_norm
        dispatch_key: any
        priority: 1

dispatch_rules:
  # Boost flash attention for batch size >= 8
  - condition: "batch_size >= 8 AND target.has_tensor_cores"
    action:
      priority_boost: 50

  # Use memory-efficient attention for long sequences
  - condition: "seq_len > 8192"
    action:
      kernel_override: xformers_memory_efficient

  # Force deterministic kernels when requested
  - condition: "context.deterministic == true"
    action:
      filter_kernels: ["deterministic == true"]

policies:
  production:
    lock_rules:
      - kernel: flash_attention_v2
        conditions:
          - "target.sm_version >= 80"
          - "dtype in [float16, bfloat16]"
    deny_rules:
      - kernel: "*_experimental"

  development:
    allow_experimental: true
    enable_profiling: true
```

Rust parser and loader:

```rust
// src/dispatch/config.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct KernelConfig {
    pub version: String,
    pub schema_version: u32,
    pub defaults: DefaultConfig,
    pub kernels: HashMap<String, KernelGroup>,
    pub dispatch_rules: Vec<DispatchRule>,
    pub policies: HashMap<String, Policy>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct KernelGroup {
    pub operation: String,
    pub implementations: Vec<KernelImplementation>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct KernelImplementation {
    pub name: String,
    pub dispatch_key: String,
    pub priority: u32,
    pub constraints: KernelConstraints,
}

impl KernelConfig {
    /// Load from file with schema validation
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration against schema
    fn validate(&self) -> Result<(), ValidationError> {
        // Check schema version compatibility
        if self.schema_version > CURRENT_SCHEMA_VERSION {
            return Err(ValidationError::IncompatibleSchema {
                found: self.schema_version,
                max_supported: CURRENT_SCHEMA_VERSION,
            });
        }

        // Validate kernel references
        for rule in &self.dispatch_rules {
            if let Some(override_kernel) = &rule.action.kernel_override {
                if !self.kernel_exists(override_kernel) {
                    return Err(ValidationError::UnknownKernel(override_kernel.clone()));
                }
            }
        }

        Ok(())
    }
}
```

---

## 3. Memory Management

### 3.1 Arena Allocator for Per-Request Memory

```rust
// src/memory/arena.rs
use bumpalo::Bump;
use std::cell::RefCell;

thread_local! {
    /// Thread-local arena for per-request allocations
    static KERNEL_ARENA: RefCell<Bump> = RefCell::new(Bump::with_capacity(1024 * 1024)); // 1MB
}

/// RAII guard that resets arena on drop
pub struct ArenaGuard {
    _marker: std::marker::PhantomData<*const ()>,
}

impl ArenaGuard {
    pub fn new() -> Self {
        Self { _marker: std::marker::PhantomData }
    }

    /// Allocate in thread-local arena
    pub fn alloc<T>(&self, value: T) -> &mut T {
        KERNEL_ARENA.with(|arena| {
            let arena = arena.borrow();
            // SAFETY: Arena outlives the guard
            unsafe { &mut *(arena.alloc(value) as *mut T) }
        })
    }

    /// Allocate slice
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &mut [T] {
        KERNEL_ARENA.with(|arena| {
            let arena = arena.borrow();
            unsafe {
                std::slice::from_raw_parts_mut(
                    arena.alloc_slice_copy(slice).as_mut_ptr(),
                    slice.len()
                )
            }
        })
    }
}

impl Drop for ArenaGuard {
    fn drop(&mut self) {
        KERNEL_ARENA.with(|arena| {
            arena.borrow_mut().reset();
        });
    }
}
```

### 3.2 Lock-Free Ring Buffer for Audio

```rust
// src/memory/ring_buffer.rs
use crossbeam_utils::CachePadded;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Wait-free SPSC ring buffer for real-time audio
#[repr(C)]
pub struct AudioRingBuffer<T> {
    buffer: Box<[std::cell::UnsafeCell<std::mem::MaybeUninit<T>>]>,
    capacity: usize,
    mask: usize,
    // Cache-padded to prevent false sharing
    head: CachePadded<AtomicUsize>,
    tail: CachePadded<AtomicUsize>,
}

unsafe impl<T: Send> Send for AudioRingBuffer<T> {}
unsafe impl<T: Send> Sync for AudioRingBuffer<T> {}

impl<T> AudioRingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let buffer: Vec<_> = (0..capacity)
            .map(|_| std::cell::UnsafeCell::new(std::mem::MaybeUninit::uninit()))
            .collect();

        Self {
            buffer: buffer.into_boxed_slice(),
            capacity,
            mask: capacity - 1,
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
        }
    }

    /// Wait-free push (producer only)
    #[inline]
    pub fn push(&self, value: T) -> Result<(), T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail.wrapping_sub(head) >= self.capacity {
            return Err(value);
        }

        let index = tail & self.mask;
        unsafe {
            (*self.buffer[index].get()).write(value);
        }

        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Wait-free pop (consumer only)
    #[inline]
    pub fn pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        if head == tail {
            return None;
        }

        let index = head & self.mask;
        let value = unsafe { (*self.buffer[index].get()).assume_init_read() };

        self.head.store(head.wrapping_add(1), Ordering::Release);
        Some(value)
    }
}
```

### 3.3 GPU Memory Pool

```rust
// src/memory/gpu_pool.rs
use cuda_runtime_sys::*;
use std::collections::HashMap;

/// Stream-ordered GPU memory pool
pub struct GpuMemoryPool {
    pool: cudaMemPool_t,
    device: i32,
    /// Track allocations for debugging
    allocations: parking_lot::Mutex<HashMap<*mut c_void, AllocationInfo>>,
}

#[derive(Debug)]
struct AllocationInfo {
    size: usize,
    stream: cudaStream_t,
    allocated_at: std::time::Instant,
}

impl GpuMemoryPool {
    pub fn new(device: i32) -> Result<Self, CudaError> {
        let mut pool: cudaMemPool_t = std::ptr::null_mut();

        unsafe {
            let props = cudaMemPoolProps {
                allocType: cudaMemAllocationType::cudaMemAllocationTypePinned,
                handleTypes: cudaMemAllocationHandleType::cudaMemHandleTypeNone,
                location: cudaMemLocation {
                    type_: cudaMemLocationType::cudaMemLocationTypeDevice,
                    id: device,
                },
                ..std::mem::zeroed()
            };

            check_cuda!(cudaMemPoolCreate(&mut pool, &props))?;

            // Set release threshold (keep 512MB cached)
            let threshold: u64 = 512 * 1024 * 1024;
            check_cuda!(cudaMemPoolSetAttribute(
                pool,
                cudaMemPoolAttr::cudaMemPoolAttrReleaseThreshold,
                &threshold as *const _ as *mut c_void,
            ))?;
        }

        Ok(Self {
            pool,
            device,
            allocations: parking_lot::Mutex::new(HashMap::new()),
        })
    }

    /// Async allocation in stream
    pub fn alloc_async(&self, size: usize, stream: cudaStream_t) -> Result<GpuBuffer, CudaError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();

        unsafe {
            check_cuda!(cudaMallocFromPoolAsync(&mut ptr, size, self.pool, stream))?;
        }

        // Track allocation
        self.allocations.lock().insert(ptr, AllocationInfo {
            size,
            stream,
            allocated_at: std::time::Instant::now(),
        });

        Ok(GpuBuffer { ptr, size, pool: self })
    }

    /// Async free
    pub fn free_async(&self, ptr: *mut c_void, stream: cudaStream_t) -> Result<(), CudaError> {
        self.allocations.lock().remove(&ptr);

        unsafe {
            check_cuda!(cudaFreeAsync(ptr, stream))?;
        }

        Ok(())
    }
}
```

---

## 4. IPC Bridge (Rust ↔ Python)

### 4.1 Shared Memory Channel

```rust
// src/ipc/shared_memory.rs
use shared_memory::{Shmem, ShmemConf};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Zero-copy tensor sharing between Rust and Python
#[repr(C, align(64))]
pub struct SharedTensorHeader {
    /// Magic number for validation
    magic: u64,
    /// Data format version
    version: u32,
    /// Tensor shape (max 8 dimensions)
    shape: [u64; 8],
    /// Number of dimensions
    ndim: u32,
    /// Data type enum
    dtype: u32,
    /// Byte offset to data
    data_offset: u64,
    /// Data size in bytes
    data_size: u64,
    /// Producer sequence number
    producer_seq: AtomicU64,
    /// Consumer sequence number
    consumer_seq: AtomicU64,
    /// Ready flag
    ready: AtomicBool,
}

pub struct SharedTensorChannel {
    shm: Shmem,
    header: *mut SharedTensorHeader,
    data: *mut u8,
    capacity: usize,
}

impl SharedTensorChannel {
    pub fn create(name: &str, capacity: usize) -> Result<Self, ShmemError> {
        let total_size = std::mem::size_of::<SharedTensorHeader>() + capacity;

        let shm = ShmemConf::new()
            .size(total_size)
            .os_id(name)
            .create()?;

        let header = shm.as_ptr() as *mut SharedTensorHeader;
        let data = unsafe { (header as *mut u8).add(std::mem::size_of::<SharedTensorHeader>()) };

        // Initialize header
        unsafe {
            (*header).magic = 0xDEADBEEF_CAFEBABE;
            (*header).version = 1;
            (*header).producer_seq = AtomicU64::new(0);
            (*header).consumer_seq = AtomicU64::new(0);
            (*header).ready = AtomicBool::new(false);
        }

        Ok(Self { shm, header, data, capacity })
    }

    /// Write tensor (producer side)
    pub fn write_tensor<T: Pod>(&self, data: &[T], shape: &[usize]) -> Result<(), WriteError> {
        let byte_len = data.len() * std::mem::size_of::<T>();

        if byte_len > self.capacity {
            return Err(WriteError::TooLarge);
        }

        unsafe {
            // Wait for consumer to catch up
            let seq = (*self.header).producer_seq.load(Ordering::Acquire);
            while (*self.header).consumer_seq.load(Ordering::Acquire) + 2 < seq {
                std::hint::spin_loop();
            }

            // Write data
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.data,
                byte_len,
            );

            // Update header
            for (i, &dim) in shape.iter().enumerate() {
                (*self.header).shape[i] = dim as u64;
            }
            (*self.header).ndim = shape.len() as u32;
            (*self.header).dtype = T::dtype_id();
            (*self.header).data_size = byte_len as u64;

            // Memory barrier then signal ready
            std::sync::atomic::fence(Ordering::Release);
            (*self.header).ready.store(true, Ordering::Release);
            (*self.header).producer_seq.fetch_add(1, Ordering::Release);
        }

        Ok(())
    }
}
```

### 4.2 Python Bindings (PyO3)

```rust
// src/python/mod.rs
use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayDyn, PyReadonlyArrayDyn};

#[pyclass]
pub struct KernelDispatcher {
    inner: Arc<DispatchOrchestrator>,
}

#[pymethods]
impl KernelDispatcher {
    #[new]
    pub fn new(config_path: Option<&str>) -> PyResult<Self> {
        let config = match config_path {
            Some(path) => KernelConfig::load(Path::new(path))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
            None => KernelConfig::default(),
        };

        Ok(Self {
            inner: Arc::new(DispatchOrchestrator::new(config)),
        })
    }

    /// Execute kernel with zero-copy input
    pub fn execute<'py>(
        &self,
        py: Python<'py>,
        operation: &str,
        input: PyReadonlyArrayDyn<'py, f32>,
        context: Option<PyObject>,
    ) -> PyResult<&'py PyArrayDyn<f32>> {
        // Build selection context
        let ctx = self.build_context(py, input.shape(), context)?;

        // Select kernel
        let kernel = self.inner
            .select(operation, &ctx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Execute with GIL released
        let input_view = input.as_array();
        let output_shape = kernel.output_shape(&input_view.shape());
        let mut output = ndarray::ArrayD::<f32>::zeros(output_shape);

        py.allow_threads(|| {
            kernel.execute(
                input_view.as_slice().unwrap(),
                output.as_slice_mut().unwrap(),
            )
        }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(output.into_pyarray(py))
    }

    /// Reload configuration (hot-reload)
    pub fn reload(&self, config_path: &str) -> PyResult<()> {
        let config = KernelConfig::load(Path::new(config_path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        self.inner.reload(config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

#[pymodule]
fn layerzero_dispatch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<KernelDispatcher>()?;
    Ok(())
}
```

---

## 5. Fault Tolerance

### 5.1 Circuit Breaker

```rust
// src/resilience/circuit_breaker.rs
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CircuitState {
    Closed = 0,
    Open = 1,
    HalfOpen = 2,
}

pub struct CircuitBreaker {
    state: AtomicU8,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    last_failure_time: AtomicU64,
    config: CircuitBreakerConfig,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_ms: u64,
    pub half_open_max_calls: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_ms: 30_000,
            half_open_max_calls: 3,
        }
    }
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: AtomicU8::new(CircuitState::Closed as u8),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            last_failure_time: AtomicU64::new(0),
            config,
        }
    }

    pub fn can_execute(&self) -> bool {
        match self.state() {
            CircuitState::Closed => true,
            CircuitState::Open => {
                let elapsed = self.time_since_last_failure();
                if elapsed >= self.config.timeout_ms {
                    self.transition_to(CircuitState::HalfOpen);
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    pub fn record_success(&self) {
        match self.state() {
            CircuitState::Closed => {
                self.failure_count.store(0, Ordering::Release);
            }
            CircuitState::HalfOpen => {
                let count = self.success_count.fetch_add(1, Ordering::AcqRel) + 1;
                if count >= self.config.success_threshold {
                    self.transition_to(CircuitState::Closed);
                }
            }
            CircuitState::Open => {}
        }
    }

    pub fn record_failure(&self) {
        self.update_last_failure_time();

        match self.state() {
            CircuitState::Closed => {
                let count = self.failure_count.fetch_add(1, Ordering::AcqRel) + 1;
                if count >= self.config.failure_threshold {
                    self.transition_to(CircuitState::Open);
                }
            }
            CircuitState::HalfOpen => {
                self.transition_to(CircuitState::Open);
            }
            CircuitState::Open => {}
        }
    }

    fn state(&self) -> CircuitState {
        match self.state.load(Ordering::Acquire) {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => unreachable!(),
        }
    }

    fn transition_to(&self, new_state: CircuitState) {
        self.state.store(new_state as u8, Ordering::Release);
        self.failure_count.store(0, Ordering::Release);
        self.success_count.store(0, Ordering::Release);
    }
}
```

### 5.2 Graceful Degradation Controller

```rust
// src/resilience/degradation.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DegradationLevel {
    Full = 0,
    Reduced = 1,
    Cached = 2,
    RuleBased = 3,
    Minimal = 4,
}

pub struct DegradationController {
    level: AtomicU8,
    health_score: AtomicU32,
    thresholds: DegradationThresholds,
}

#[derive(Debug, Clone)]
pub struct DegradationThresholds {
    pub reduced_health: u32,    // Below this -> Reduced
    pub cached_health: u32,     // Below this -> Cached
    pub rule_based_health: u32, // Below this -> RuleBased
    pub minimal_health: u32,    // Below this -> Minimal
}

impl DegradationController {
    pub fn should_degrade(&self, request: &Request) -> DegradationDecision {
        let level = self.level();

        match level {
            DegradationLevel::Full => DegradationDecision::Proceed,
            DegradationLevel::Reduced => {
                if request.is_essential() {
                    DegradationDecision::Proceed
                } else {
                    DegradationDecision::Skip { reason: "Non-essential during degradation" }
                }
            }
            DegradationLevel::Cached => {
                if let Some(cached) = self.cache.get(&request.cache_key()) {
                    DegradationDecision::ServeCached(cached)
                } else {
                    DegradationDecision::Proceed
                }
            }
            DegradationLevel::RuleBased => {
                DegradationDecision::UseRuleBased
            }
            DegradationLevel::Minimal => {
                DegradationDecision::Reject { retry_after: Duration::from_secs(30) }
            }
        }
    }

    pub fn update_health(&self, score: u32) {
        self.health_score.store(score, Ordering::Release);

        // Update degradation level based on health
        let new_level = if score >= self.thresholds.reduced_health {
            DegradationLevel::Full
        } else if score >= self.thresholds.cached_health {
            DegradationLevel::Reduced
        } else if score >= self.thresholds.rule_based_health {
            DegradationLevel::Cached
        } else if score >= self.thresholds.minimal_health {
            DegradationLevel::RuleBased
        } else {
            DegradationLevel::Minimal
        };

        self.level.store(new_level as u8, Ordering::Release);
    }
}
```

---

## 6. Performance Targets

### 6.1 Latency Budgets

| Component | Target (p50) | Target (p99) | Max (p99.9) |
|-----------|--------------|--------------|-------------|
| Kernel Selection | 10µs | 50µs | 100µs |
| Cache Lookup | 100ns | 500ns | 1µs |
| IPC Transfer (1KB) | 1µs | 5µs | 10µs |
| Kernel Launch | 10µs | 20µs | 50µs |
| **End-to-End Dispatch** | **50µs** | **100µs** | **200µs** |

### 6.2 Throughput Targets

| Scenario | Target (requests/sec) |
|----------|----------------------|
| Single GPU, small batch | 100,000 |
| Single GPU, large batch | 50,000 |
| Multi-GPU (4x) | 400,000 |
| Distributed (8 nodes) | 3,000,000 |

### 6.3 Memory Efficiency

| Metric | Target |
|--------|--------|
| Dispatch overhead per request | < 1KB |
| Cache memory per kernel | < 100 bytes |
| Arena allocation efficiency | > 95% |
| Zero-copy transfer rate | > 95% |

---

## 7. Testing Strategy

### 7.1 Test Categories (175 Scenarios)

| Category | Count | Priority |
|----------|-------|----------|
| Normal Operations | 35 | P0 |
| Edge Cases | 35 | P1 |
| Failure Scenarios | 35 | P0 |
| Concurrency | 35 | P0 |
| Performance | 35 | P1 |

### 7.2 Benchmark Suite

```rust
// benches/dispatch_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_kernel_selection(c: &mut Criterion) {
    let dispatcher = KernelDispatcher::new(None).unwrap();
    let context = SelectionContext::test_context();

    let mut group = c.benchmark_group("kernel_selection");

    // Warm cache
    group.bench_function("cache_hit", |b| {
        b.iter(|| dispatcher.select("attention", &context))
    });

    // Cold cache
    group.bench_function("cache_miss", |b| {
        b.iter_with_setup(
            || dispatcher.clear_cache(),
            |_| dispatcher.select("attention", &context)
        )
    });

    group.finish();
}

fn bench_dispatch_overhead(c: &mut Criterion) {
    let dispatcher = KernelDispatcher::new(None).unwrap();

    let mut group = c.benchmark_group("dispatch_overhead");

    for size in [64, 256, 1024, 4096].iter() {
        let input = vec![0.0f32; *size];
        let context = SelectionContext::with_size(*size);

        group.bench_with_input(
            BenchmarkId::new("end_to_end", size),
            size,
            |b, _| {
                b.iter(|| dispatcher.execute("attention", &input, &context))
            }
        );
    }

    group.finish();
}

criterion_group!(benches, bench_kernel_selection, bench_dispatch_overhead);
criterion_main!(benches);
```

---

## 8. Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-2)

**Deliverables:**
- [ ] Static dispatch enum with CRTP pattern
- [ ] Arena allocator integration
- [ ] Lock-free ring buffer for IPC
- [ ] Basic YAML config parser
- [ ] Unit tests for all components

**Files to Create/Modify:**
- `src/dispatch/mod.rs` - Module organization
- `src/dispatch/static_dispatch.rs` - Enum dispatch
- `src/dispatch/dynamic_dispatch.rs` - Trait object dispatch
- `src/memory/arena.rs` - Arena allocator
- `src/memory/ring_buffer.rs` - SPSC ring buffer
- `src/config/kernel_config.rs` - YAML config

### Phase 2: Selection Integration (Weeks 3-4)

**Deliverables:**
- [ ] Integrate with existing LayerZero SelectionEngine
- [ ] Wire up actual kernel execution (not just selection)
- [ ] MVCC cache integration
- [ ] Policy engine integration
- [ ] Integration tests

**Files to Modify:**
- `src/layerzero/selection/engine.py` - Add dispatch hooks
- `src/layerzero/registry/kernel_registry.py` - Version indexing
- `src/layerzero/backends/*.py` - Standardize execute() interface

### Phase 3: Hot-Reload & Resilience (Weeks 5-6)

**Deliverables:**
- [ ] Hot-reload manager with atomic swap
- [ ] Circuit breaker per kernel
- [ ] Graceful degradation controller
- [ ] Fallback chain execution
- [ ] Chaos testing infrastructure

**Files to Create:**
- `src/dispatch/hot_reload.rs`
- `src/resilience/circuit_breaker.rs`
- `src/resilience/degradation.rs`
- `tests/chaos/` - Chaos tests

### Phase 4: Python Integration (Weeks 7-8)

**Deliverables:**
- [ ] PyO3 bindings for dispatcher
- [ ] Zero-copy tensor sharing
- [ ] nanobind migration (optional)
- [ ] Python integration tests
- [ ] Performance benchmarks

**Files to Create:**
- `src/python/mod.rs` - PyO3 module
- `python/layerzero_dispatch/` - Python package
- `tests/python/` - Python tests

### Phase 5: Performance Tuning (Weeks 9-10)

**Deliverables:**
- [ ] Benchmark suite
- [ ] Profile-guided optimization
- [ ] Memory pressure callbacks
- [ ] Autotuning integration
- [ ] Documentation

**Metrics to Achieve:**
- Dispatch overhead < 100µs p99
- Cache hit rate > 95%
- Zero-copy rate > 95%

---

## 9. Integration with Existing LayerZero

### 9.1 Backward Compatibility

The new dispatch system integrates with existing LayerZero components:

```python
# src/layerzero/api/operations.py

def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    # Existing parameters...
    _use_new_dispatch: bool = True,  # Feature flag
) -> torch.Tensor:
    if _use_new_dispatch:
        # New Rust-based dispatch
        return _rust_dispatcher.execute(
            "attention",
            {"q": query, "k": key, "v": value},
            _build_context(query, key, value),
        )
    else:
        # Existing Python dispatch
        return _legacy_dispatch(query, key, value)
```

### 9.2 Migration Path

1. **Week 1-2**: Deploy with `_use_new_dispatch=False` (shadow mode)
2. **Week 3-4**: Enable for 1% of requests (canary)
3. **Week 5-6**: Increase to 25% with monitoring
4. **Week 7-8**: Full rollout with legacy fallback
5. **Week 9+**: Remove legacy code path

---

## 10. Observability

### 10.1 Metrics

```rust
// src/telemetry/metrics.rs
use prometheus::{Counter, Histogram, IntGauge, Registry};

pub struct DispatchMetrics {
    // Counters
    pub selection_total: Counter,
    pub execution_total: Counter,
    pub cache_hits: Counter,
    pub cache_misses: Counter,
    pub fallback_used: Counter,
    pub circuit_breaker_opens: Counter,

    // Histograms
    pub selection_latency: Histogram,
    pub execution_latency: Histogram,
    pub ipc_latency: Histogram,

    // Gauges
    pub active_kernels: IntGauge,
    pub cache_size: IntGauge,
    pub degradation_level: IntGauge,
}
```

### 10.2 Distributed Tracing

```rust
// src/telemetry/tracing.rs
use opentelemetry::trace::{Tracer, Span};

pub fn traced_dispatch<F, T>(
    tracer: &dyn Tracer,
    operation: &str,
    f: F,
) -> T
where
    F: FnOnce(&mut Span) -> T,
{
    let mut span = tracer
        .span_builder(format!("kernel.dispatch.{}", operation))
        .with_kind(SpanKind::Internal)
        .start(tracer);

    span.set_attribute(KeyValue::new("kernel.operation", operation.to_string()));

    let result = f(&mut span);

    span.end();
    result
}
```

---

## 11. Security Considerations

### 11.1 Input Validation

- Validate all tensor shapes before dispatch
- Bounds check array indices
- Sanitize configuration file paths
- Rate limit API endpoints

### 11.2 Memory Safety

- Use Rust's ownership model
- No raw pointer arithmetic in hot paths
- Arena allocator prevents use-after-free
- Shared memory protected with magic numbers

### 11.3 Plugin Security

- Verify plugin signatures
- Sandbox plugin execution
- Capability-based permissions
- Audit log for plugin loads

---

## 12. Conclusion

This implementation plan provides a comprehensive roadmap for building a state-of-the-art kernel dispatch system that:

1. **Achieves near-zero overhead** through static dispatch and compile-time optimization
2. **Supports multiple dispatch modes** (static, dynamic, hot-reload, config-driven)
3. **Scales from single-box to hyperscaler** with work-stealing and distributed patterns
4. **Handles 175+ operational scenarios** with robust fault tolerance
5. **Integrates seamlessly** with existing LayerZero infrastructure
6. **Provides full observability** with metrics, tracing, and health monitoring

The phased implementation approach allows incremental delivery while maintaining backward compatibility and enabling gradual migration from the existing system.

---

## References

### Academic Papers
- [Task-Based Tensor Computations on Modern GPUs](https://arxiv.org/html/2504.07004)
- [Seer: Predictive Runtime Kernel Selection](https://arxiv.org/html/2403.17017)
- [MCFuser: Memory-Bound Fusion for GPU Kernels](https://arxiv.org/html/2506.22169)

### Production Implementations
- [PyTorch Dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/reference/high-level-design.html)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
- [Apache TVM Relay Dispatch](https://tvm.apache.org/docs/v0.9.0/arch/relay_op_strategy.html)

### Rust Crates
- [enum_dispatch](https://docs.rs/enum_dispatch)
- [crossbeam](https://docs.rs/crossbeam)
- [parking_lot](https://docs.rs/parking_lot)
- [bumpalo](https://docs.rs/bumpalo)
- [PyO3](https://pyo3.rs)

### Real-Time Audio
- [Ross Bencina: Real-Time Audio Programming 101](http://www.rossbencina.com/code/real-time-audio-programming-101-time-waits-for-nothing)
- [rtrb: Realtime-safe SPSC Ring Buffer](https://github.com/mgeier/rtrb)
