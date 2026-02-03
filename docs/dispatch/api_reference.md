# API Reference

This document provides complete API documentation for all public classes and functions in the LayerZero dispatch system.

## Core Types

### DispatchMode

```python
class DispatchMode(Enum):
    """Kernel dispatch mode."""

    STATIC = auto()     # Zero overhead, compile-time kernel resolution
    DYNAMIC = auto()    # Runtime selection, ~100-500ns overhead
    HOT_RELOAD = auto() # Config file watching, ~1-10ms reload
    CONFIG = auto()     # YAML-driven, ~100ns lookup overhead
    AUTO = auto()       # Automatically choose best mode
```

### DispatchPhase

```python
class DispatchPhase(Enum):
    """Phase of dispatch execution for telemetry."""

    SELECTION = auto()      # Kernel selection phase
    PRE_TRANSFORM = auto()  # Pre-execution transforms
    EXECUTION = auto()      # Kernel execution
    POST_TRANSFORM = auto() # Post-execution transforms
    FALLBACK = auto()       # Fallback handling
```

### DispatchTiming

```python
@dataclass(frozen=True, slots=True)
class DispatchTiming:
    """Timing information for dispatch execution (nanoseconds)."""

    selection_ns: int = 0
    pre_transform_ns: int = 0
    execution_ns: int = 0
    post_transform_ns: int = 0
    total_ns: int = 0

    @property
    def selection_us(self) -> float:
        """Selection time in microseconds."""

    @property
    def execution_us(self) -> float:
        """Execution time in microseconds."""

    @property
    def total_us(self) -> float:
        """Total time in microseconds."""

    @property
    def overhead_ns(self) -> int:
        """Dispatch overhead (everything except execution)."""
```

### DispatchResult

```python
@dataclass(frozen=True, slots=True)
class DispatchResult:
    """Result of kernel dispatch and execution."""

    output: torch.Tensor          # Output tensor from kernel
    kernel_id: str                # Selected kernel identifier
    kernel_spec: KernelSpec       # Full kernel specification
    timing: DispatchTiming        # Timing metrics
    mode: DispatchMode            # Dispatch mode used
    cached: bool = False          # Whether selection was cached
    fallback_used: bool = False   # Whether fallback was used
    fallback_reason: str | None = None  # Reason for fallback

    @property
    def overhead_us(self) -> float:
        """Dispatch overhead in microseconds."""
```

### DispatchConfig

```python
@dataclass(slots=True)
class DispatchConfig:
    """Configuration for the dispatch system."""

    mode: DispatchMode = DispatchMode.DYNAMIC
    enable_cache: bool = True
    cache_size: int = 10000
    cache_ttl_seconds: float = 3600.0
    enable_fallback: bool = True
    max_fallback_attempts: int = 3
    fallback_timeout_ms: float = 100.0
    config_path: str | None = None
    watch_interval_seconds: float = 1.0
    validate_on_reload: bool = True
    enable_transforms: bool = True
    enable_cuda_graphs: bool = False
    sync_after_execution: bool = False
    enable_telemetry: bool = True
    record_timing: bool = True
    log_fallbacks: bool = True
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    static_kernel_map: dict[str, str] = field(default_factory=dict)
```

## Exceptions

### DispatchError

```python
class DispatchError(Exception):
    """Base exception for dispatch failures."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        kernel_id: str | None = None,
        phase: DispatchPhase | None = None,
    ) -> None: ...

    operation: str | None
    kernel_id: str | None
    phase: DispatchPhase | None
```

### KernelExecutionError

```python
class KernelExecutionError(DispatchError):
    """Error during kernel execution."""

    def __init__(
        self,
        message: str,
        operation: str,
        kernel_id: str,
        original_error: Exception | None = None,
    ) -> None: ...

    original_error: Exception | None
```

### FallbackChainExhaustedError

```python
class FallbackChainExhaustedError(DispatchError):
    """All fallback options have been exhausted."""

    def __init__(
        self,
        operation: str,
        attempted_kernels: list[str],
        errors: list[Exception],
    ) -> None: ...

    attempted_kernels: list[str]
    errors: list[Exception]
```

### CircuitOpenError

```python
class CircuitOpenError(DispatchError):
    """Circuit breaker is open for a kernel."""

    def __init__(
        self,
        kernel_id: str,
        retry_after_seconds: float,
    ) -> None: ...

    retry_after_seconds: float
```

### HotReloadError

```python
class HotReloadError(DispatchError):
    """Error during hot-reload."""

    def __init__(
        self,
        message: str,
        config_path: str,
        original_error: Exception | None = None,
    ) -> None: ...

    config_path: str
    original_error: Exception | None
```

## Protocols

### Dispatcher Protocol

```python
@runtime_checkable
class Dispatcher(Protocol):
    """Protocol for kernel dispatchers."""

    @property
    def mode(self) -> DispatchMode:
        """Get the dispatch mode."""
        ...

    def dispatch(
        self,
        operation: str,
        inputs: dict[str, torch.Tensor],
        context: SelectionContext | None = None,
        **kwargs: Any,
    ) -> DispatchResult:
        """Dispatch operation to appropriate kernel.

        Args:
            operation: Operation identifier (e.g., "attention.causal").
            inputs: Dictionary of named input tensors.
            context: Optional pre-built selection context.
            **kwargs: Additional operation-specific arguments.

        Returns:
            DispatchResult with output tensor and metadata.

        Raises:
            DispatchError: If dispatch fails.
        """
        ...

    def get_kernel_for_operation(
        self,
        operation: str,
        context: SelectionContext,
    ) -> KernelSpec:
        """Get the kernel that would be used for an operation.

        Args:
            operation: Operation identifier.
            context: Selection context.

        Returns:
            KernelSpec that would be selected.
        """
        ...
```

### KernelExecutor Protocol

```python
@runtime_checkable
class KernelExecutor(Protocol):
    """Protocol for kernel execution."""

    def execute(
        self,
        kernel_spec: KernelSpec,
        inputs: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute a kernel with given inputs.

        Args:
            kernel_spec: Specification of the kernel to execute.
            inputs: Dictionary of named input tensors.
            **kwargs: Additional kernel-specific arguments.

        Returns:
            Output tensor from kernel execution.

        Raises:
            KernelExecutionError: If execution fails.
        """
        ...

    def supports_cuda_graph(self, kernel_spec: KernelSpec) -> bool:
        """Check if kernel can be captured in CUDA graph."""
        ...
```

## Orchestrator

### DispatchOrchestrator

```python
class DispatchOrchestrator:
    """Unified dispatch orchestrator."""

    def __init__(
        self,
        config: DispatchConfig | None = None,
        kernel_registry: KernelRegistry | None = None,
        backend_registry: BackendRegistry | None = None,
        selection_engine: Any = None,
        default_mode: DispatchMode = DispatchMode.DYNAMIC,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config: Dispatch configuration.
            kernel_registry: Kernel registry for dispatch.
            backend_registry: Backend registry for health tracking.
            selection_engine: Selection engine for dynamic dispatch.
            default_mode: Default mode when AUTO is specified.
        """

    @property
    def config(self) -> DispatchConfig:
        """Get current configuration."""

    @property
    def default_mode(self) -> DispatchMode:
        """Get default dispatch mode."""

    def set_default_mode(self, mode: DispatchMode) -> None:
        """Set the default dispatch mode.

        Args:
            mode: New default mode (cannot be AUTO).
        """

    def dispatch(
        self,
        operation: str,
        inputs: dict[str, torch.Tensor],
        context: SelectionContext | None = None,
        mode: DispatchMode | None = None,
        **kwargs: Any,
    ) -> DispatchResult:
        """Dispatch an operation to the appropriate kernel.

        Args:
            operation: Operation identifier (e.g., "attention.causal").
            inputs: Dictionary of named input tensors.
            context: Optional pre-built selection context.
            mode: Optional mode override.
            **kwargs: Additional operation-specific arguments.

        Returns:
            DispatchResult with output tensor and metadata.

        Raises:
            DispatchError: If dispatch fails in all attempted modes.
        """

    def get_kernel_for_operation(
        self,
        operation: str,
        context: SelectionContext,
        mode: DispatchMode | None = None,
    ) -> KernelSpec:
        """Get the kernel that would be used for an operation."""

    def register_dispatcher(
        self,
        mode: DispatchMode,
        dispatcher: Dispatcher,
    ) -> None:
        """Register a custom dispatcher for a mode."""

    def get_dispatcher(self, mode: DispatchMode) -> Dispatcher | None:
        """Get the dispatcher for a mode."""

    def reload_config(self, config: DispatchConfig) -> None:
        """Reload the orchestrator configuration."""

    def get_telemetry(self) -> dict[str, Any]:
        """Get orchestrator telemetry."""

    def reset_telemetry(self) -> None:
        """Reset all telemetry counters."""

    def shutdown(self) -> None:
        """Shutdown the orchestrator and all dispatchers."""
```

### Factory Functions

```python
def get_global_dispatcher() -> DispatchOrchestrator:
    """Get the global dispatch orchestrator."""

def set_global_dispatcher(orchestrator: DispatchOrchestrator | None) -> None:
    """Set the global dispatch orchestrator."""

def create_orchestrator(
    config: DispatchConfig | None = None,
    kernel_registry: KernelRegistry | None = None,
    backend_registry: BackendRegistry | None = None,
    selection_engine: Any = None,
    default_mode: DispatchMode = DispatchMode.DYNAMIC,
    set_as_global: bool = False,
) -> DispatchOrchestrator:
    """Create a new dispatch orchestrator."""

def dispatch(
    operation: str,
    inputs: dict[str, torch.Tensor],
    context: SelectionContext | None = None,
    mode: DispatchMode | None = None,
    **kwargs: Any,
) -> DispatchResult:
    """Dispatch an operation using the global orchestrator."""
```

## Static Dispatch

### StaticDispatcher

```python
class StaticDispatcher(BaseDispatcher):
    """Static dispatch mode implementation."""

    def __init__(
        self,
        config: DispatchConfig,
        registry: StaticKernelRegistry | None = None,
        executor: KernelExecutor | None = None,
    ) -> None:
        """Initialize static dispatcher.

        Args:
            config: Dispatch configuration.
            registry: Static kernel registry.
            executor: Kernel executor.
        """

    @property
    def mode(self) -> DispatchMode:
        """Always returns DispatchMode.STATIC."""

    @property
    def registry(self) -> StaticKernelRegistry:
        """Get the static registry."""

    def dispatch(
        self,
        operation: str,
        inputs: dict[str, torch.Tensor],
        context: SelectionContext | None = None,
        **kwargs: Any,
    ) -> DispatchResult:
        """Dispatch with O(1) lookup."""

    def get_kernel_for_operation(
        self,
        operation: str,
        context: SelectionContext,
    ) -> KernelSpec:
        """Get kernel for operation."""

    def get_telemetry(self) -> dict[str, Any]:
        """Get dispatch statistics."""
```

### StaticKernelRegistry

```python
class StaticKernelRegistry:
    """Import-time kernel mapping with O(1) lookup."""

    def __init__(self) -> None:
        """Initialize empty registry."""

    @property
    def is_frozen(self) -> bool:
        """Check if registry is frozen."""

    @property
    def operation_count(self) -> int:
        """Number of registered operations."""

    @property
    def kernel_count(self) -> int:
        """Number of registered kernels."""

    def register(self, entry: StaticKernelEntry) -> None:
        """Register a kernel entry.

        Raises:
            RegistryFrozenError: If registry is frozen.
            ValueError: If kernel_id is already registered.
        """

    def register_from_spec(
        self,
        spec: KernelSpec,
        *,
        is_default: bool = False,
    ) -> StaticKernelEntry:
        """Register kernel from KernelSpec."""

    def register_many(self, entries: list[StaticKernelEntry]) -> None:
        """Register multiple entries atomically."""

    def freeze(self) -> None:
        """Freeze registry to prevent further modifications."""

    def get_by_kernel_id(self, kernel_id: str) -> StaticKernelEntry | None:
        """Get entry by kernel ID."""

    def get_by_operation(self, operation: str) -> list[StaticKernelEntry]:
        """Get all entries for an operation."""

    def get_default(self, operation: str) -> StaticKernelEntry | None:
        """Get default kernel for an operation."""

    def get_kernel_for_operation(
        self,
        operation: str,
        context: SelectionContext | None = None,
    ) -> KernelSpec:
        """Get kernel spec for operation."""

    def operations(self) -> frozenset[str]:
        """Get all registered operations."""

    def kernel_ids(self) -> frozenset[str]:
        """Get all registered kernel IDs."""

    def to_mapping(self) -> dict[str, str]:
        """Export as operation -> kernel_id mapping."""

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[str, str],
        kernel_specs: dict[str, KernelSpec],
    ) -> StaticKernelRegistry:
        """Create registry from mapping."""
```

### StaticDispatcherBuilder

```python
class StaticDispatcherBuilder:
    """Builder for constructing StaticDispatcher instances."""

    def __init__(self) -> None:
        """Initialize builder."""

    def with_kernel(
        self,
        spec: KernelSpec,
        *,
        operation: str | None = None,
        default: bool = False,
    ) -> StaticDispatcherBuilder:
        """Add kernel to registry."""

    def with_kernels(
        self,
        specs: list[KernelSpec],
        *,
        defaults: dict[str, str] | None = None,
    ) -> StaticDispatcherBuilder:
        """Add multiple kernels to registry."""

    def with_config(self, **kwargs: Any) -> StaticDispatcherBuilder:
        """Update configuration."""

    def with_executor(
        self,
        executor: KernelExecutor,
    ) -> StaticDispatcherBuilder:
        """Set custom kernel executor."""

    def with_registry(
        self,
        registry: StaticKernelRegistry,
    ) -> StaticDispatcherBuilder:
        """Use existing registry."""

    def build(self) -> StaticDispatcher:
        """Build the dispatcher."""
```

### Factory Functions

```python
def create_static_dispatcher(
    kernel_specs: list[KernelSpec],
    defaults: dict[str, str] | None = None,
    config: DispatchConfig | None = None,
    executor: KernelExecutor | None = None,
) -> StaticDispatcher:
    """Create static dispatcher from kernel specs."""

def create_static_dispatcher_from_config(
    config: DispatchConfig,
    kernel_registry: KernelRegistry,
    executor: KernelExecutor | None = None,
) -> StaticDispatcher:
    """Create static dispatcher from config and kernel registry."""
```

## Dynamic Dispatch

### DynamicDispatcher

```python
class DynamicDispatcher(BaseDispatcher):
    """Dynamic kernel dispatcher with runtime selection."""

    def __init__(
        self,
        selection_engine: SelectionEngine,
        backend_registry: BackendRegistry | None = None,
        config: DispatchConfig | None = None,
        executor: KernelExecutorImpl | None = None,
        mvcc_cache: MVCCShardedCache | None = None,
    ) -> None:
        """Initialize dynamic dispatcher."""

    @property
    def mode(self) -> DispatchMode:
        """Always returns DispatchMode.DYNAMIC."""

    @property
    def selection_engine(self) -> SelectionEngine:
        """Get the selection engine."""

    @property
    def circuit_breaker(self) -> CircuitBreakerManager:
        """Get the circuit breaker manager."""

    def dispatch(
        self,
        operation: str,
        inputs: dict[str, torch.Tensor],
        context: SelectionContext | None = None,
        **kwargs: Any,
    ) -> DispatchResult:
        """Dispatch with runtime kernel selection."""

    def get_kernel_for_operation(
        self,
        operation: str,
        context: SelectionContext,
    ) -> KernelSpec:
        """Get kernel without executing."""

    def get_telemetry(self) -> dict[str, Any]:
        """Get telemetry data."""

    def reset_telemetry(self) -> None:
        """Reset telemetry counters."""

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""

    def invalidate_cache(self) -> None:
        """Invalidate all cached selections."""
```

### CircuitBreakerManager

```python
class CircuitBreakerManager:
    """Manages circuit breakers for all kernels."""

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
    ) -> None:
        """Initialize circuit breaker manager."""

    def is_allowed(self, kernel_id: str) -> bool:
        """Check if kernel is allowed (circuit not open)."""

    def get_retry_after(self, kernel_id: str) -> float | None:
        """Get seconds until circuit might allow retry."""

    def record_success(self, kernel_id: str) -> None:
        """Record successful kernel execution."""

    def record_failure(
        self,
        kernel_id: str,
        error: Exception | None = None,
    ) -> None:
        """Record failed kernel execution."""

    def reset(self, kernel_id: str) -> None:
        """Reset circuit for kernel to closed state."""

    def reset_all(self) -> None:
        """Reset all circuits to closed state."""

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all circuits."""
```

### Factory Function

```python
def create_dynamic_dispatcher(
    kernel_registry: KernelRegistry,
    backend_registry: BackendRegistry,
    config: DispatchConfig | None = None,
    use_mvcc_cache: bool = True,
) -> DynamicDispatcher:
    """Create a fully configured dynamic dispatcher."""
```

## Hot-Reload Dispatch

### HotReloadDispatcher

```python
class HotReloadDispatcher(BaseDispatcher):
    """Hot-reload dispatcher with zero-downtime config updates."""

    def __init__(
        self,
        config: DispatchConfig,
        executor: KernelExecutorImpl | None = None,
        config_path: str | Path | None = None,
        selection_engine: Any = None,
        kernel_registry: Any = None,
    ) -> None:
        """Initialize hot-reload dispatcher."""

    @property
    def mode(self) -> DispatchMode:
        """Always returns DispatchMode.HOT_RELOAD."""

    @property
    def current_version(self) -> ConfigVersion | None:
        """Get current configuration version."""

    @property
    def reload_state(self) -> ReloadState:
        """Get current reload state."""

    @property
    def stats(self) -> ReloadStats:
        """Get reload statistics."""

    def dispatch(
        self,
        operation: str,
        inputs: dict[str, torch.Tensor],
        context: SelectionContext | None = None,
        **kwargs: Any,
    ) -> DispatchResult:
        """Dispatch with current configuration."""

    def get_kernel_for_operation(
        self,
        operation: str,
        context: SelectionContext,
    ) -> KernelSpec:
        """Get kernel that would be used."""

    def reload(
        self,
        config_data: dict[str, Any] | None = None,
        validate: bool = True,
        grace_period_seconds: float = 1.0,
    ) -> bool:
        """Manually trigger config reload.

        Returns:
            True if reload succeeded.

        Raises:
            HotReloadError: If reload fails.
        """

    def rollback(self) -> bool:
        """Rollback to previous configuration version."""

    def watch(self) -> None:
        """Start watching config file for changes."""

    def stop_watching(self) -> None:
        """Stop watching config file."""

    def add_validator(
        self,
        validator: Callable[[dict[str, Any]], list[str]],
    ) -> None:
        """Add a custom configuration validator."""

    def on_reload(
        self,
        callback: Callable[[ConfigVersion], None],
    ) -> None:
        """Register callback for successful reload events."""

    def on_error(
        self,
        callback: Callable[[Exception], None],
    ) -> None:
        """Register callback for reload error events."""

    def get_config_history(self) -> list[ConfigVersion]:
        """Get configuration version history."""

    def dry_run(
        self,
        config_data: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Validate config without applying it."""

    def get_telemetry(self) -> dict[str, Any]:
        """Get telemetry including reload stats."""
```

### ConfigVersion

```python
@dataclass(frozen=True, slots=True)
class ConfigVersion:
    """Immutable configuration version snapshot."""

    version: int                    # Monotonic version number
    config_hash: str                # SHA256 hash of config
    timestamp: float                # Unix timestamp
    config_data: dict[str, Any]     # Parsed configuration
    source_path: str | None = None  # Source file path
```

### ReloadStats

```python
@dataclass(slots=True)
class ReloadStats:
    """Statistics for hot-reload operations."""

    total_reloads: int = 0
    failed_reloads: int = 0
    rollbacks: int = 0
    last_reload_time_ms: float = 0.0
    last_reload_timestamp: float = 0.0
    config_version: int = 0

    def record_success(self, reload_time_ms: float, version: int) -> None:
        """Record a successful reload."""

    def record_failure(self) -> None:
        """Record a failed reload."""

    def record_rollback(self) -> None:
        """Record a rollback operation."""
```

### Factory Function

```python
def create_hot_reload_dispatcher(
    config_path: str | Path,
    *,
    validate_on_reload: bool = True,
    watch_interval_seconds: float = 1.0,
    start_watching: bool = True,
    selection_engine: Any = None,
    kernel_registry: Any = None,
) -> HotReloadDispatcher:
    """Factory function to create a hot-reload dispatcher."""
```

## Config-Driven Dispatch

### ConfigDrivenDispatcher

```python
class ConfigDrivenDispatcher(BaseDispatcher):
    """YAML config-driven kernel dispatcher."""

    def __init__(
        self,
        config: DispatchConfig,
        executor: KernelExecutor | None = None,
        kernel_registry: KernelRegistry | None = None,
        yaml_config: dict[str, Any] | None = None,
        config_path: Path | str | None = None,
    ) -> None:
        """Initialize config-driven dispatcher."""

    @property
    def mode(self) -> DispatchMode:
        """Always returns DispatchMode.CONFIG."""

    @property
    def compiled_config(self) -> CompiledConfig:
        """Get compiled configuration."""

    def dispatch(
        self,
        operation: str,
        inputs: dict[str, torch.Tensor],
        context: SelectionContext | None = None,
        **kwargs: Any,
    ) -> DispatchResult:
        """Dispatch using config rules."""

    def get_kernel_for_operation(
        self,
        operation: str,
        context: SelectionContext,
    ) -> KernelSpec:
        """Get kernel that would be selected."""

    def reload_config(
        self,
        config_path: Path | str | None = None,
    ) -> bool:
        """Reload configuration from file."""

    def update_yaml_config(
        self,
        yaml_config: dict[str, Any],
    ) -> None:
        """Update configuration from dict."""

    def explain_selection(
        self,
        operation: str,
        context: SelectionContext,
    ) -> dict[str, Any]:
        """Explain kernel selection for debugging."""

    def get_telemetry(self) -> dict[str, Any]:
        """Get dispatcher telemetry."""
```

### ConfigSchema

```python
class ConfigSchema:
    """Schema validator for dispatch config files."""

    def validate(self, config: dict[str, Any]) -> list[SchemaError]:
        """Validate config against schema.

        Returns:
            List of validation errors (empty if valid).
        """
```

### Factory Function

```python
def create_config_dispatcher(
    config_path: Path | str | None = None,
    yaml_config: dict[str, Any] | None = None,
    kernel_registry: KernelRegistry | None = None,
    cache_size: int = 10000,
    cache_ttl_seconds: float = 60.0,
) -> ConfigDrivenDispatcher:
    """Create a config-driven dispatcher."""
```

## Circuit Breaker

### CircuitBreaker

```python
class CircuitBreaker(Generic[T]):
    """Thread-safe circuit breaker."""

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize circuit breaker."""

    @property
    def name(self) -> str:
        """Get circuit name."""

    @property
    def state(self) -> CircuitState:
        """Get current state."""

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""

    def can_execute(self) -> bool:
        """Check if a call is currently allowed."""

    def record_success(self) -> None:
        """Record a successful call."""

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed call."""

    def force_open(self) -> None:
        """Manually open the circuit."""

    def force_close(self) -> None:
        """Manually close the circuit."""

    def reset(self) -> None:
        """Reset circuit to initial state."""

    def add_listener(
        self,
        listener: Callable[[CircuitState, CircuitState], None],
    ) -> None:
        """Add state change listener."""

    def wrap(self, func: Callable[..., T]) -> Callable[..., T]:
        """Wrap a function with circuit breaker protection."""

    def to_dict(self) -> dict[str, Any]:
        """Get circuit status as dictionary."""
```

### CircuitBreakerRegistry

```python
class CircuitBreakerRegistry:
    """Registry of circuit breakers for multiple resources."""

    def __init__(
        self,
        default_config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize registry."""

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get existing circuit or create new one."""

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit by name."""

    def can_execute(self, name: str) -> bool:
        """Check if circuit allows execution."""

    def record_success(self, name: str) -> None:
        """Record success for circuit."""

    def record_failure(
        self,
        name: str,
        error: Exception | None = None,
    ) -> None:
        """Record failure for circuit."""

    def reset_all(self) -> None:
        """Reset all circuits."""

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all circuits."""

    def get_open_circuits(self) -> list[str]:
        """Get names of all open circuits."""
```

### Factory Functions

```python
def get_global_circuit_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""

def get_circuit(name: str) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
```

## Executor

### KernelExecutorImpl

```python
class KernelExecutorImpl:
    """Kernel executor implementation."""

    def __init__(
        self,
        backend_registry: BackendRegistry | None = None,
    ) -> None:
        """Initialize executor."""

    def execute(
        self,
        kernel_spec: KernelSpec,
        inputs: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute a kernel with given inputs.

        Raises:
            KernelExecutionError: If execution fails.
        """

    def execute_with_timing(
        self,
        kernel_spec: KernelSpec,
        inputs: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, int]:
        """Execute kernel and return execution time in nanoseconds."""

    def supports_cuda_graph(self, kernel_spec: KernelSpec) -> bool:
        """Check if kernel supports CUDA graph capture."""
```

### CUDAGraphExecutor

```python
class CUDAGraphExecutor:
    """Executor that captures and replays CUDA graphs."""

    def __init__(
        self,
        base_executor: KernelExecutorImpl,
        warmup_count: int = 3,
    ) -> None:
        """Initialize CUDA graph executor."""

    def execute(
        self,
        kernel_spec: KernelSpec,
        inputs: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute kernel, using CUDA graph if available."""

    def clear_graphs(self) -> None:
        """Clear all captured graphs."""
```

### Function

```python
def execute_kernel(
    kernel_spec: KernelSpec,
    inputs: dict[str, torch.Tensor],
    backend_registry: BackendRegistry | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Convenience function to execute a kernel."""
```
