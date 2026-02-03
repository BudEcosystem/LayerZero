# FlashInfer Analysis and LayerZero Architectural Fix

## Executive Summary

LayerZero currently treats attention variants (GQA, MQA, MLA, Sparse) as **capability flags** on a single "attention" operation. This is fundamentally incorrect. Each variant is a **distinct operation** with different:
- Input tensor signatures
- Memory layouts
- Computational patterns
- Output shapes

This document proposes an architectural fix to adopt a **kernel-as-operation** model where each variant is a separate operation with its own schema.

---

## Part 1: FlashInfer Architecture Analysis

### 1.1 Operation Structure

FlashInfer exposes attention variants as **separate wrapper classes**, each with distinct Python APIs:

```
flashinfer/
├── decode.py                    # Standard decode operations
│   ├── single_decode_with_kv_cache()
│   ├── BatchDecodeWithPagedKVCacheWrapper
│   └── CUDAGraphBatchDecodeWithPagedKVCacheWrapper
│
├── prefill.py                   # Standard prefill operations
│   ├── single_prefill_with_kv_cache()
│   ├── BatchPrefillWithPagedKVCacheWrapper
│   └── BatchPrefillWithRaggedKVCacheWrapper
│
├── mla.py                       # MLA (DeepSeek) operations
│   ├── BatchMLAPagedAttentionWrapper
│   └── BatchDecodeMlaWithPagedKVCacheWrapper
│
├── sparse.py                    # Sparse attention operations
│   ├── BlockSparseAttentionWrapper
│   └── VariableBlockSparseAttentionWrapper
│
├── cascade.py                   # Cascade/hierarchical attention
│   ├── MultiLevelCascadeAttentionWrapper
│   └── BatchDecodeWithSharedPrefixPagedKVCacheWrapper
│
└── pod.py                       # POD attention variant
    ├── PODWithPagedKVCacheWrapper
    └── BatchPODWithPagedKVCacheWrapper
```

### 1.2 Input Signature Differences

**Standard Decode (decode.py)**
```python
def run(
    q: torch.Tensor,              # [batch, heads, dim]
    paged_k_cache: torch.Tensor,  # [num_pages, page_size, heads, dim]
    paged_v_cache: torch.Tensor,  # [num_pages, page_size, heads, dim]
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
) -> torch.Tensor:  # [batch, heads, dim]
```

**MLA Decode (mla.py)**
```python
def run(
    q_nope: torch.Tensor,   # [batch*seq, 128, 512] - query without PE
    q_pe: torch.Tensor,     # [batch*seq, 128, 64]  - query PE component
    ckv: torch.Tensor,      # [kv_len, 1, 512]      - compressed KV
    kpe: torch.Tensor,      # [kv_len, 1, 64]       - key PE component
    return_lse: bool = False,
) -> torch.Tensor:  # [batch*seq, 128, 512]
```

**Sparse Attention (sparse.py)**
```python
def run(
    q: torch.Tensor,        # [M, heads, dim]
    k: torch.Tensor,        # [N, kv_heads, dim]
    v: torch.Tensor,        # [N, kv_heads, dim]
    # BSR sparse format
    indptr: torch.Tensor,   # [M/R + 1] block row pointers
    indices: torch.Tensor,  # [nnz] block column indices
) -> torch.Tensor:  # [M, heads, dim]
```

### 1.3 Key Insight: Different Memory Structures

**Standard PagedKV Cache:**
```cpp
// paged_kv_t structure
struct paged_kv_t<dtype> {
    dtype* k_data;        // Key data
    dtype* v_data;        // Value data (separate from K)
    int num_heads;        // Number of KV heads
    int page_size;
    int head_dim;         // Single head dimension
};
```

**MLA PagedKV Cache:**
```cpp
// paged_kv_mla_t structure
struct paged_kv_mla_t<dtype> {
    dtype* ckv_data;      // Compressed KV (combined)
    dtype* kpe_data;      // Positional encoding data (separate)
    int head_dim_ckv;     // 512 for DeepSeek
    int head_dim_kpe;     // 64 for DeepSeek
    // NOTE: No num_heads! MLA uses different structure
};
```

### 1.4 Compile-Time Dispatch (CUDA Level)

FlashInfer uses compile-time specialization for GQA group sizes:

```cpp
// include/flashinfer/utils.cuh
#define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...) \
  if (group_size == 1) {                                     \
    constexpr size_t GROUP_SIZE = 1; __VA_ARGS__             \
  } else if (group_size == 2) {                              \
    constexpr size_t GROUP_SIZE = 2; __VA_ARGS__             \
  } else if (group_size == 4) {                              \
    constexpr size_t GROUP_SIZE = 4; __VA_ARGS__             \
  } else if (group_size == 8) {                              \
    constexpr size_t GROUP_SIZE = 8; __VA_ARGS__             \
  }
```

Different GROUP_SIZE values result in different:
- Tile sizes (8 for MHA, 1-4 for GQA)
- Thread block configurations
- Memory access patterns

---

## Part 2: LayerZero Current Problems

### 2.1 Capability Flags Model (INCORRECT)

Current `KernelSpec` uses boolean flags:
```python
@dataclass(frozen=True, slots=True)
class KernelSpec:
    # ...
    supports_gqa: bool = True    # WRONG: Treats GQA as a feature
    supports_mqa: bool = True    # WRONG: Treats MQA as a feature
    # ...
```

**Problem:** This assumes all attention kernels have the same input signature with GQA/MQA as optional features. Reality: GQA kernels may have completely different tile sizes, memory access patterns, and even input formats.

### 2.2 Fixed Input Schema

`SelectionContext.from_tensors()` assumes standard attention:
```python
def from_tensors(
    cls,
    q: "torch.Tensor",           # Always expects q
    k: "torch.Tensor | None",    # Always expects k
    v: "torch.Tensor | None",    # Always expects v
    # ...
)
```

**Problem:** Cannot represent MLA inputs (`q_nope`, `q_pe`, `ckv`, `kpe`) or sparse inputs (BSR indices).

### 2.3 Flat Operation Names

Current operations: `"attention.causal"`, `"attention.full"`

**Problem:** No distinction between:
- Prefill vs decode
- Standard vs MLA vs Sparse
- Paged vs contiguous KV cache

---

## Part 3: Proposed Architectural Fix

### 3.1 Hierarchical Operation Names

Replace flat operation names with a hierarchical structure:

```
attention.
├── standard.
│   ├── prefill              # Full prefill, contiguous KV
│   ├── prefill.paged        # Prefill with paged KV cache
│   ├── decode               # Single token decode, contiguous KV
│   └── decode.paged         # Decode with paged KV cache
│
├── mla.
│   ├── prefill              # MLA prefill (absorbed matrices)
│   ├── decode               # MLA decode
│   └── decode.paged         # MLA with paged cache
│
├── sparse.
│   ├── block                # Block sparse attention
│   └── variable_block       # Variable block sparse
│
├── gqa.                     # Explicit GQA operations
│   ├── prefill
│   ├── prefill.paged
│   ├── decode
│   └── decode.paged
│
└── cross.                   # Cross attention (encoder-decoder)
    ├── prefill
    └── decode
```

### 3.2 Operation Schema Registry

Add a new `OperationSchema` class that defines the input/output contract:

```python
@dataclass(frozen=True, slots=True)
class TensorSchema:
    """Schema for a single tensor parameter."""
    name: str
    ndim: int | None = None           # Expected ndim (None = any)
    dtype_constraint: frozenset["torch.dtype"] | None = None
    shape_vars: tuple[str, ...] = ()  # e.g., ("batch", "seq", "heads", "dim")
    required: bool = True

@dataclass(frozen=True, slots=True)
class OperationSchema:
    """Schema defining the input/output contract for an operation."""
    operation: str                     # e.g., "attention.mla.decode"
    inputs: tuple[TensorSchema, ...]   # Input tensor schemas
    outputs: tuple[TensorSchema, ...]  # Output tensor schemas
    params: dict[str, type]            # Scalar parameters (e.g., sm_scale: float)

    def validate(self, *args, **kwargs) -> list[str]:
        """Validate inputs match schema, return list of errors."""
        ...
```

**Example Schemas:**

```python
STANDARD_DECODE_SCHEMA = OperationSchema(
    operation="attention.standard.decode.paged",
    inputs=(
        TensorSchema("q", ndim=3, shape_vars=("batch", "heads", "dim")),
        TensorSchema("paged_k_cache", ndim=4),
        TensorSchema("paged_v_cache", ndim=4),
        TensorSchema("kv_indptr", ndim=1),
        TensorSchema("kv_indices", ndim=1),
        TensorSchema("kv_last_page_len", ndim=1),
    ),
    outputs=(
        TensorSchema("output", ndim=3, shape_vars=("batch", "heads", "dim")),
    ),
    params={"sm_scale": float, "window_left": int},
)

MLA_DECODE_SCHEMA = OperationSchema(
    operation="attention.mla.decode",
    inputs=(
        TensorSchema("q_nope", ndim=3, shape_vars=("tokens", "heads", "dim_ckv")),
        TensorSchema("q_pe", ndim=3, shape_vars=("tokens", "heads", "dim_kpe")),
        TensorSchema("ckv", ndim=3, shape_vars=("kv_len", "1", "dim_ckv")),
        TensorSchema("kpe", ndim=3, shape_vars=("kv_len", "1", "dim_kpe")),
    ),
    outputs=(
        TensorSchema("output", ndim=3, shape_vars=("tokens", "heads", "dim_ckv")),
    ),
    params={"sm_scale": float, "return_lse": bool},
)

SPARSE_ATTENTION_SCHEMA = OperationSchema(
    operation="attention.sparse.block",
    inputs=(
        TensorSchema("q", ndim=3, shape_vars=("M", "heads", "dim")),
        TensorSchema("k", ndim=3, shape_vars=("N", "kv_heads", "dim")),
        TensorSchema("v", ndim=3, shape_vars=("N", "kv_heads", "dim")),
        TensorSchema("indptr", ndim=1),  # BSR format
        TensorSchema("indices", ndim=1), # BSR format
    ),
    outputs=(
        TensorSchema("output", ndim=3, shape_vars=("M", "heads", "dim")),
    ),
    params={"R": int, "C": int, "sm_scale": float},
)
```

### 3.3 Schema-Based Context Builder

Replace `SelectionContext.from_tensors()` with schema-aware builders:

```python
class SelectionContextBuilder:
    """Build SelectionContext from operation schema and inputs."""

    def __init__(self, schema: OperationSchema):
        self.schema = schema

    def build(self, *args, device: DeviceSpec | None = None, **kwargs) -> SelectionContext:
        """Build context, validating against schema."""
        # Validate inputs
        errors = self.schema.validate(*args, **kwargs)
        if errors:
            raise ValueError(f"Schema validation failed: {errors}")

        # Extract common fields from validated inputs
        first_tensor = args[0]
        if device is None:
            device = DeviceSpec.detect(str(first_tensor.device))

        return SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation=self.schema.operation,
            dtype=first_tensor.dtype,
            batch_size=first_tensor.shape[0],
            # Schema-specific extraction handled by subclasses
            **self._extract_schema_fields(*args, **kwargs),
        )

    def _extract_schema_fields(self, *args, **kwargs) -> dict:
        """Override in subclasses for schema-specific field extraction."""
        return {}
```

### 3.4 Updated KernelSpec

Remove capability flags, use operation matching:

```python
@dataclass(frozen=True, slots=True)
class KernelSpec:
    """Kernel specification - simplified without capability flags."""

    # Identity
    kernel_id: str
    operation: str                  # Now matches hierarchical schema
    source: str
    version: str

    # Implementation
    impl: Callable | None = None

    # Hardware requirements (unchanged)
    platform: Platform = Platform.CUDA
    min_sm: tuple[int, int] | None = None
    max_sm: tuple[int, int] | None = None

    # Dtype support (unchanged)
    supported_dtypes: frozenset["torch.dtype"] = field(default_factory=frozenset)

    # Shape constraints - now operation-specific
    shape_constraints: dict[str, tuple[int, int]] = field(default_factory=dict)
    # e.g., {"dim_ckv": (512, 512), "dim_kpe": (64, 64)} for MLA

    # REMOVED: supports_gqa, supports_mqa (these are now separate operations)
    # REMOVED: supports_kv_strategies (embedded in operation name)

    # Execution properties (unchanged)
    is_cuda_graph_safe: bool = True
    deterministic: bool = False
    workspace_bytes: int = 0
    priority: int = 50
```

### 3.5 Updated Registry

Kernels register for specific operations:

```python
class KernelRegistry:
    """Registry with operation-first lookup."""

    def __init__(self):
        self._schemas: dict[str, OperationSchema] = {}
        self._kernels: dict[str, list[KernelSpec]] = {}  # op -> kernels

    def register_schema(self, schema: OperationSchema) -> None:
        """Register an operation schema."""
        self._schemas[schema.operation] = schema
        self._kernels.setdefault(schema.operation, [])

    def register_kernel(self, spec: KernelSpec) -> None:
        """Register a kernel for its operation."""
        if spec.operation not in self._schemas:
            raise ValueError(f"Unknown operation: {spec.operation}")
        self._kernels[spec.operation].append(spec)

    def get_candidates(self, operation: str) -> list[KernelSpec]:
        """Get all kernels for an operation."""
        return self._kernels.get(operation, [])

    def get_schema(self, operation: str) -> OperationSchema | None:
        """Get schema for an operation."""
        return self._schemas.get(operation)
```

---

## Part 4: Migration Path

### Phase 1: Add Schema Infrastructure (Non-Breaking)
1. Add `OperationSchema` and `TensorSchema` classes
2. Add schema registry alongside existing kernel registry
3. Define schemas for existing operations

### Phase 2: Hierarchical Operations
1. Add hierarchical operation names
2. Keep backward compatibility aliases (`"attention.causal"` -> `"attention.standard.prefill"`)
3. Deprecate capability flags with warnings

### Phase 3: New Operation Types
1. Add MLA, Sparse, Cross attention schemas
2. Integrate FlashInfer MLA kernels
3. Add schema validation to selection engine

### Phase 4: Cleanup
1. Remove deprecated capability flags
2. Remove backward compatibility aliases
3. Update all backends to use new schema system

---

## Part 5: Example Usage

### Before (Current LayerZero)
```python
# Cannot represent MLA - no way to pass q_nope, q_pe, ckv, kpe
ctx = SelectionContext.from_tensors(q, k, v, is_causal=True)
kernel = registry.select(ctx)
output = kernel.impl(q, k, v)  # Fixed signature
```

### After (Proposed)
```python
# MLA operation with proper schema
schema = registry.get_schema("attention.mla.decode")
ctx = SelectionContextBuilder(schema).build(
    q_nope, q_pe, ckv, kpe,
    sm_scale=0.125
)
kernel = registry.select(ctx)
output = kernel.impl(q_nope, q_pe, ckv, kpe, sm_scale=0.125)

# Standard decode (also explicit)
schema = registry.get_schema("attention.standard.decode.paged")
ctx = SelectionContextBuilder(schema).build(
    q, paged_k, paged_v, kv_indptr, kv_indices, kv_last_page_len,
    sm_scale=0.125
)
kernel = registry.select(ctx)
output = kernel.impl(q, paged_k, paged_v, kv_indptr, kv_indices, kv_last_page_len)
```

---

## Conclusion

The core insight is that **GQA, MLA, Sparse, etc. are not features of a single "attention" operation - they are fundamentally different operations** with different:
- Input tensor counts and shapes
- Memory access patterns
- Computational kernels
- Output shapes

LayerZero must adopt a **kernel-as-operation** model where:
1. Each variant is a distinct operation with its own schema
2. Kernels register for specific operations, not generic "attention" with flags
3. Selection happens within an operation type, not across all attention variants
4. Input validation uses schema definitions, not hardcoded q/k/v assumptions

This matches how FlashInfer (and other production attention libraries) are actually structured.
