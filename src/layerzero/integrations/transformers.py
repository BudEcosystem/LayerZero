"""
HuggingFace Transformers Integration

Provides integration with HuggingFace Transformers models
for attention replacement and optimization.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

# Lazy import for optional dependency
_transformers = None
_transformers_available: bool | None = None


def _get_transformers() -> Any:
    """Lazy import of transformers library."""
    global _transformers
    if _transformers is None:
        try:
            import transformers
            _transformers = transformers
        except ImportError:
            _transformers = False
    return _transformers if _transformers else None


def is_transformers_available() -> bool:
    """Check if HuggingFace Transformers is available.

    Returns:
        True if transformers library is installed.
    """
    global _transformers_available
    if _transformers_available is None:
        _transformers_available = _get_transformers() is not None
    return _transformers_available


def get_transformers_version() -> tuple[int, int, int] | None:
    """Get Transformers library version.

    Returns:
        Tuple of (major, minor, patch) version numbers,
        or None if not available.
    """
    transformers = _get_transformers()
    if transformers is None:
        return None

    try:
        version_str = transformers.__version__
        parts = version_str.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        # Handle versions like "4.30.0.dev0"
        patch_str = parts[2].split(".")[0] if len(parts) > 2 else "0"
        patch = int("".join(c for c in patch_str if c.isdigit()) or "0")
        return (major, minor, patch)
    except Exception:
        return None


def patch_model(
    model: "nn.Module",
    attention_impl: str = "auto",
) -> "nn.Module":
    """Patch a model to use LayerZero attention.

    Replaces the attention modules in the model with LayerZero
    optimized implementations.

    Args:
        model: HuggingFace Transformers model to patch.
        attention_impl: Attention implementation to use.
            "auto" selects the best available.

    Returns:
        The patched model (same object, modified in-place).

    Example:
        ```python
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
        patched = lz.patch_model(model)
        output = patched.generate(...)
        ```
    """
    from layerzero.integrations.model_patching import apply_patches

    apply_patches(model, attention_impl=attention_impl)
    return model


def unpatch_model(model: "nn.Module") -> "nn.Module":
    """Restore original attention implementation.

    Undoes the patching applied by patch_model().

    Args:
        model: Previously patched model.

    Returns:
        The unpatched model (same object, modified in-place).
    """
    from layerzero.integrations.model_patching import remove_patches

    remove_patches(model)
    return model


def auto_patch_on_load(enable: bool = True) -> None:
    """Enable/disable automatic patching when loading models.

    When enabled, models loaded via AutoModel.from_pretrained()
    will be automatically patched.

    Args:
        enable: Whether to enable auto-patching.
    """
    # This would hook into transformers' model loading
    # For now, this is a no-op placeholder
    logger.info(f"Auto-patch on load: {enable}")


class LayerZeroAttentionMixin:
    """Mixin class providing LayerZero attention.

    Can be mixed into model classes to provide optimized
    attention computation.
    """

    def _layerzero_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute attention using LayerZero.

        Args:
            query: Query tensor [B, H, L, D]
            key: Key tensor [B, H, S, D]
            value: Value tensor [B, H, S, D]
            attention_mask: Optional attention mask
            **kwargs: Additional arguments

        Returns:
            Attention output tensor.
        """
        return torch.ops.layerzero.attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            **kwargs,
        )
