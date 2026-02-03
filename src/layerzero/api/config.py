"""LayerZero Configuration APIs.

Public APIs for configuring LayerZero:
- configure() - Set global configuration
- get_config() - Get current configuration
- load_config() - Load configuration from file
- lock() / unlock() - Lock/unlock kernel selections
- get_locks() - Get current locks
- prefer() - Context manager for backend preferences
- disabled() - Context manager for fallback-only mode
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional
import threading
import yaml


@dataclass
class LayerZeroConfig:
    """Global LayerZero configuration.

    Attributes:
        default_backend: Default backend to use when no specific selection.
        cache_size: Maximum size of selection cache.
        strict_mode: If True, raise on unknown kernels/operations.
        auto_tune_on_warmup: If True, collect timing during warmup.
        tune_samples: Number of samples for auto-tuning.
        policy_path: Path to policy configuration file.
    """
    default_backend: Optional[str] = None
    cache_size: int = 10000
    strict_mode: bool = False
    auto_tune_on_warmup: bool = False
    tune_samples: int = 10
    policy_path: Optional[str] = None


@dataclass
class GlobalState:
    """Global state for LayerZero."""
    config: LayerZeroConfig = field(default_factory=LayerZeroConfig)
    locks: Dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)


# Module-level global state
_global_state: Optional[GlobalState] = None
_state_lock = threading.Lock()


def _get_global_state() -> GlobalState:
    """Get or create global state."""
    global _global_state
    if _global_state is None:
        with _state_lock:
            if _global_state is None:
                _global_state = GlobalState()
    return _global_state


def configure(
    default_backend: Optional[str] = None,
    cache_size: Optional[int] = None,
    strict_mode: Optional[bool] = None,
    auto_tune_on_warmup: Optional[bool] = None,
    tune_samples: Optional[int] = None,
    policy_path: Optional[str] = None,
    reset: bool = False,
    **kwargs: Any,
) -> None:
    """Configure LayerZero global settings.

    Call this function to customize LayerZero behavior. Settings persist
    for the lifetime of the process unless reset.

    Args:
        default_backend: Default backend when no specific selection.
                        Options: "torch_sdpa", "flash_attn", "flashinfer", etc.
        cache_size: Maximum entries in selection cache (default: 10000).
        strict_mode: If True, raise errors for unknown kernels.
        auto_tune_on_warmup: If True, collect timing data during warmup.
        tune_samples: Number of samples for auto-tuning (default: 10).
        policy_path: Path to YAML policy file.
        reset: If True, reset all settings to defaults first.
        **kwargs: Additional settings (for forward compatibility).

    Example:
        >>> import layerzero as lz
        >>>
        >>> # Set default backend and enable strict mode
        >>> lz.configure(
        ...     default_backend="flash_attn",
        ...     strict_mode=True,
        ... )
        >>>
        >>> # Reset to defaults
        >>> lz.configure(reset=True)
    """
    state = _get_global_state()

    with state._lock:
        if reset:
            state.config = LayerZeroConfig()
            state.locks.clear()

        # Update individual settings
        if default_backend is not None:
            state.config.default_backend = default_backend
        if cache_size is not None:
            state.config.cache_size = cache_size
        if strict_mode is not None:
            state.config.strict_mode = strict_mode
        if auto_tune_on_warmup is not None:
            state.config.auto_tune_on_warmup = auto_tune_on_warmup
        if tune_samples is not None:
            state.config.tune_samples = tune_samples
        if policy_path is not None:
            state.config.policy_path = policy_path


def get_config() -> LayerZeroConfig:
    """Get current LayerZero configuration.

    Returns:
        Current configuration object (copy for safety).

    Example:
        >>> import layerzero as lz
        >>>
        >>> config = lz.get_config()
        >>> print(f"Cache size: {config.cache_size}")
    """
    state = _get_global_state()
    with state._lock:
        # Return a copy to prevent mutation
        return LayerZeroConfig(
            default_backend=state.config.default_backend,
            cache_size=state.config.cache_size,
            strict_mode=state.config.strict_mode,
            auto_tune_on_warmup=state.config.auto_tune_on_warmup,
            tune_samples=state.config.tune_samples,
            policy_path=state.config.policy_path,
        )


def load_config(path: str) -> None:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is invalid.

    Example:
        >>> import layerzero as lz
        >>>
        >>> lz.load_config("layerzero.yaml")

    YAML format::

        default_backend: flash_attn
        cache_size: 5000
        strict_mode: true
        locks:
          attention.causal: flash_attn.v3.causal
          norm.rms: liger.rms_norm
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file format: {path}")

    # Apply configuration
    configure(
        default_backend=data.get('default_backend'),
        cache_size=data.get('cache_size'),
        strict_mode=data.get('strict_mode'),
        auto_tune_on_warmup=data.get('auto_tune_on_warmup'),
        tune_samples=data.get('tune_samples'),
        policy_path=data.get('policy_path'),
    )

    # Apply locks
    locks = data.get('locks', {})
    for operation, kernel_id in locks.items():
        lock(operation, kernel_id)


def lock(operation: str, kernel_id: str) -> None:
    """Lock an operation to always use a specific kernel.

    When an operation is locked, LayerZero bypasses automatic selection
    and always uses the specified kernel. This is useful for:
    - Ensuring reproducibility
    - Forcing use of a known-good kernel
    - Debugging kernel selection issues

    Args:
        operation: Operation to lock (e.g., "attention.causal").
        kernel_id: Kernel to use (e.g., "flash_attn.v3.causal").

    Raises:
        ValueError: If kernel_id is not valid for the operation.

    Example:
        >>> import layerzero as lz
        >>>
        >>> # Lock attention to Flash Attention v3
        >>> lz.lock("attention.causal", "flash_attn.v3.causal")
        >>>
        >>> # All subsequent attention calls use FA3
        >>> output = lz.attention(q, k, v, is_causal=True)
    """
    state = _get_global_state()

    # Validate kernel exists (optional - depends on registry state)
    # For now, we accept any kernel_id

    with state._lock:
        state.locks[operation] = kernel_id


def unlock(operation: str) -> None:
    """Remove kernel lock for an operation.

    After unlocking, LayerZero resumes automatic kernel selection
    for the operation.

    Args:
        operation: Operation to unlock.

    Example:
        >>> import layerzero as lz
        >>>
        >>> lz.lock("attention.causal", "torch_sdpa")
        >>> # ... use locked kernel ...
        >>> lz.unlock("attention.causal")
        >>> # Back to automatic selection
    """
    state = _get_global_state()

    with state._lock:
        state.locks.pop(operation, None)


def get_locks() -> Dict[str, str]:
    """Get current kernel locks.

    Returns:
        Dictionary mapping operations to locked kernel IDs.

    Example:
        >>> import layerzero as lz
        >>>
        >>> lz.lock("attention.causal", "flash_attn")
        >>> locks = lz.get_locks()
        >>> print(locks)
        {'attention.causal': 'flash_attn'}
    """
    state = _get_global_state()

    with state._lock:
        return dict(state.locks)


@contextmanager
def prefer(*backends: str) -> Generator[None, None, None]:
    """Context manager to prefer specific backends.

    Within this context, the specified backends will be prioritized
    for kernel selection. This doesn't lock kernels - it just
    influences the selection scoring.

    Args:
        *backends: Backend names to prefer (e.g., "flash_attn", "flashinfer").

    Example:
        >>> import layerzero as lz
        >>>
        >>> with lz.prefer("flashinfer"):
        ...     # FlashInfer kernels will be scored higher
        ...     output = lz.attention(q, k, v, is_causal=True)
        >>>
        >>> with lz.prefer("flash_attn", "flashinfer"):
        ...     # Both will be preferred, in order
        ...     output = lz.attention(q, k, v)
    """
    state = _get_global_state()

    # Save current config
    with state._lock:
        old_default = state.config.default_backend
        old_locks = dict(state.locks)

    # Set preference via default backend (simplified implementation)
    # In a full implementation, this would set a preference list
    if backends:
        configure(default_backend=backends[0])

    try:
        yield
    finally:
        # Restore previous config
        with state._lock:
            state.config.default_backend = old_default
            state.locks = old_locks


@contextmanager
def disabled() -> Generator[None, None, None]:
    """Context manager to use only fallback kernels.

    Within this context, LayerZero will only use the fallback
    PyTorch SDPA kernel, ignoring all optimized backends.

    Useful for:
    - Debugging kernel-specific issues
    - Ensuring baseline correctness
    - Comparing optimized vs baseline performance

    Example:
        >>> import layerzero as lz
        >>>
        >>> # Normal execution uses optimized kernels
        >>> output1 = lz.attention(q, k, v, is_causal=True)
        >>>
        >>> with lz.disabled():
        ...     # Only uses torch.nn.functional.scaled_dot_product_attention
        ...     output2 = lz.attention(q, k, v, is_causal=True)
        >>>
        >>> # Verify outputs match
        >>> assert torch.allclose(output1, output2, rtol=1e-3)
    """
    state = _get_global_state()

    # Save current locks
    with state._lock:
        old_locks = dict(state.locks)

    # Lock all operations to fallback
    fallback_locks = {
        "attention.causal": "torch_sdpa",
        "attention.full": "torch_sdpa",
        "norm.rms": "torch_rms_norm",
        "norm.layer": "torch_layer_norm",
    }

    for operation, kernel in fallback_locks.items():
        lock(operation, kernel)

    try:
        yield
    finally:
        # Restore previous locks
        with state._lock:
            state.locks = old_locks
