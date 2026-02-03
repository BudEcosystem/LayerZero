"""
LayerZero Kernel Dispatch System

State-of-the-art kernel dispatch supporting multiple modes:
- Static dispatch: Zero-overhead, compile-time kernel selection
- Dynamic dispatch: Runtime kernel selection with ~100-500ns overhead
- Hot-reload dispatch: Development/A/B testing with config file watching
- Config-driven dispatch: YAML-based ops-controlled deployment

The dispatch system integrates with LayerZero's SelectionEngine to provide
actual kernel execution after selection.
"""
from __future__ import annotations

from layerzero.dispatch.types import (
    CircuitOpenError,
    DispatchMode,
    DispatchPhase,
    DispatchResult,
    DispatchTiming,
    DispatchError,
    DispatchConfig,
    KernelExecutionError,
    FallbackChainExhaustedError,
)
from layerzero.dispatch.protocols import (
    BaseDispatcher,
    Dispatcher,
    KernelExecutor,
)
from layerzero.dispatch.executor import (
    execute_kernel,
    KernelExecutorImpl,
    CUDAGraphExecutor,
)
from layerzero.dispatch.dynamic import (
    DynamicDispatcher,
    CircuitBreakerManager,
    FallbackChainImpl,
    create_dynamic_dispatcher,
)
from layerzero.dispatch.static import (
    StaticDispatcher,
    StaticDispatcherBuilder,
    StaticKernelRegistry,
    StaticKernelEntry,
    StaticDispatchError,
    KernelNotRegisteredError,
    RegistryFrozenError,
    OperationType,
    get_operation_type,
    get_global_static_registry,
    set_global_static_registry,
    create_static_dispatcher,
    create_static_dispatcher_from_config,
)
from layerzero.dispatch.config_dispatch import (
    ConfigDrivenDispatcher,
    ConfigSchema,
    CompiledConfig,
    CompiledDispatchRule,
    CompiledCondition,
    ConditionOperator,
    RuleEvaluationCache,
    SchemaError,
    compile_config,
    config_to_policy,
    create_config_dispatcher,
    SCHEMA_VERSION,
    SUPPORTED_VERSIONS,
)
from layerzero.dispatch.hot_reload import (
    HotReloadDispatcher,
    ConfigVersion,
    ReloadState,
    ReloadStats,
    create_hot_reload_dispatcher,
)
from layerzero.dispatch.orchestrator import (
    DispatchOrchestrator,
    OrchestratorTelemetry,
    get_global_dispatcher,
    set_global_dispatcher,
    create_orchestrator,
    dispatch,
)
from layerzero.dispatch.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
    CircuitStats,
    get_global_circuit_registry,
    get_circuit,
)
from layerzero.dispatch.buffers import (
    # Constants
    CACHE_LINE_SIZE,
    SIMD_ALIGNMENT,
    PAGE_SIZE,
    MAX_RING_BUFFER_SIZE,
    # Enums
    BufferState,
    # Core classes
    AlignedAllocator,
    RingBuffer,
    AudioRingBuffer,
    BufferPool,
    PooledBuffer,
    TensorBufferView,
    BufferManager,
    # Global management
    get_global_buffer_manager,
    set_global_buffer_manager,
)

__all__ = [
    # Types
    "CircuitOpenError",
    "DispatchMode",
    "DispatchPhase",
    "DispatchResult",
    "DispatchTiming",
    "DispatchError",
    "DispatchConfig",
    "KernelExecutionError",
    "FallbackChainExhaustedError",
    # Protocols
    "BaseDispatcher",
    "Dispatcher",
    "KernelExecutor",
    # Executor
    "execute_kernel",
    "KernelExecutorImpl",
    "CUDAGraphExecutor",
    # Dynamic dispatch
    "DynamicDispatcher",
    "CircuitBreakerManager",
    "FallbackChainImpl",
    "create_dynamic_dispatcher",
    # Static dispatch
    "StaticDispatcher",
    "StaticDispatcherBuilder",
    "StaticKernelRegistry",
    "StaticKernelEntry",
    "StaticDispatchError",
    "KernelNotRegisteredError",
    "RegistryFrozenError",
    "OperationType",
    "get_operation_type",
    "get_global_static_registry",
    "set_global_static_registry",
    "create_static_dispatcher",
    "create_static_dispatcher_from_config",
    # Config-driven dispatch
    "ConfigDrivenDispatcher",
    "ConfigSchema",
    "CompiledConfig",
    "CompiledDispatchRule",
    "CompiledCondition",
    "ConditionOperator",
    "RuleEvaluationCache",
    "SchemaError",
    "compile_config",
    "config_to_policy",
    "create_config_dispatcher",
    "SCHEMA_VERSION",
    "SUPPORTED_VERSIONS",
    # Hot-reload dispatch
    "HotReloadDispatcher",
    "ConfigVersion",
    "ReloadState",
    "ReloadStats",
    "create_hot_reload_dispatcher",
    # Orchestrator
    "DispatchOrchestrator",
    "OrchestratorTelemetry",
    "get_global_dispatcher",
    "set_global_dispatcher",
    "create_orchestrator",
    "dispatch",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerRegistry",
    "CircuitState",
    "CircuitStats",
    "get_global_circuit_registry",
    "get_circuit",
    # Buffer Management
    "CACHE_LINE_SIZE",
    "SIMD_ALIGNMENT",
    "PAGE_SIZE",
    "MAX_RING_BUFFER_SIZE",
    "BufferState",
    "AlignedAllocator",
    "RingBuffer",
    "AudioRingBuffer",
    "BufferPool",
    "PooledBuffer",
    "TensorBufferView",
    "BufferManager",
    "get_global_buffer_manager",
    "set_global_buffer_manager",
]
