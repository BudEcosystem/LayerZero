"""
HuggingFace Diffusers Integration

Provides integration with HuggingFace Diffusers models
for attention replacement in UNet and DiT architectures.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

# Lazy import for optional dependency
_diffusers = None
_diffusers_available: bool | None = None


def _get_diffusers() -> Any:
    """Lazy import of diffusers library."""
    global _diffusers
    if _diffusers is None:
        try:
            import diffusers
            _diffusers = diffusers
        except ImportError:
            _diffusers = False
    return _diffusers if _diffusers else None


def is_diffusers_available() -> bool:
    """Check if HuggingFace Diffusers is available.

    Returns:
        True if diffusers library is installed.
    """
    global _diffusers_available
    if _diffusers_available is None:
        _diffusers_available = _get_diffusers() is not None
    return _diffusers_available


def get_diffusers_version() -> tuple[int, int, int] | None:
    """Get Diffusers library version.

    Returns:
        Tuple of (major, minor, patch) version numbers,
        or None if not available.
    """
    diffusers = _get_diffusers()
    if diffusers is None:
        return None

    try:
        version_str = diffusers.__version__
        parts = version_str.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        # Handle versions like "0.25.0.dev0"
        patch_str = parts[2].split(".")[0] if len(parts) > 2 else "0"
        patch = int("".join(c for c in patch_str if c.isdigit()) or "0")
        return (major, minor, patch)
    except Exception:
        return None


# Mapping of diffuser model types to their attention module patterns
DIFFUSER_ATTENTION_PATTERNS: dict[str, list[str]] = {
    "unet": [
        "down_blocks.*.attentions.*.transformer_blocks.*.attn1",
        "down_blocks.*.attentions.*.transformer_blocks.*.attn2",
        "mid_block.attentions.*.transformer_blocks.*.attn1",
        "mid_block.attentions.*.transformer_blocks.*.attn2",
        "up_blocks.*.attentions.*.transformer_blocks.*.attn1",
        "up_blocks.*.attentions.*.transformer_blocks.*.attn2",
    ],
    "dit": [
        "transformer_blocks.*.attn1",
        "transformer_blocks.*.attn2",
    ],
}


def _get_diffuser_model_type(model: "nn.Module") -> str | None:
    """Get the diffuser model type string.

    Args:
        model: PyTorch model.

    Returns:
        Model type string (e.g., "unet", "dit") or None.
    """
    if hasattr(model, "config"):
        config = model.config
        if hasattr(config, "model_type"):
            model_type = config.model_type.lower()
            if model_type in DIFFUSER_ATTENTION_PATTERNS:
                return model_type

    # Check for known model class names
    class_name = type(model).__name__.lower()

    if "unet" in class_name:
        return "unet"
    if "dit" in class_name or "transformer" in class_name:
        return "dit"

    # Check for UNet-like structure
    if hasattr(model, "down_blocks") and hasattr(model, "up_blocks"):
        return "unet"

    # Check for DiT-like structure
    if hasattr(model, "transformer_blocks"):
        return "dit"

    return None


def patch_unet(
    unet: "nn.Module",
    attention_impl: str = "auto",
) -> "nn.Module":
    """Patch UNet to use LayerZero attention.

    Replaces the attention modules in the UNet with LayerZero
    optimized implementations.

    Args:
        unet: HuggingFace Diffusers UNet model to patch.
        attention_impl: Attention implementation to use.
            "auto" selects the best available.

    Returns:
        The patched UNet (same object, modified in-place).

    Example:
        ```python
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2")
        patched = lz.patch_unet(unet)
        output = patched(latents, timestep, encoder_hidden_states)
        ```
    """
    from layerzero.integrations.model_patching import apply_patches, ATTENTION_MODULE_PATTERNS

    # Register UNet patterns if not already
    if "unet" not in ATTENTION_MODULE_PATTERNS:
        ATTENTION_MODULE_PATTERNS["unet"] = DIFFUSER_ATTENTION_PATTERNS["unet"]

    apply_patches(unet, attention_impl=attention_impl)
    return unet


def unpatch_unet(unet: "nn.Module") -> "nn.Module":
    """Restore original UNet attention implementation.

    Undoes the patching applied by patch_unet().

    Args:
        unet: Previously patched UNet.

    Returns:
        The unpatched UNet (same object, modified in-place).
    """
    from layerzero.integrations.model_patching import remove_patches

    remove_patches(unet)
    return unet


def patch_dit(
    dit: "nn.Module",
    attention_impl: str = "auto",
) -> "nn.Module":
    """Patch DiT to use LayerZero attention.

    Replaces the attention modules in the DiT with LayerZero
    optimized implementations.

    Args:
        dit: HuggingFace Diffusers DiT model to patch.
        attention_impl: Attention implementation to use.
            "auto" selects the best available.

    Returns:
        The patched DiT (same object, modified in-place).

    Example:
        ```python
        from diffusers import DiTTransformer2DModel

        dit = DiTTransformer2DModel.from_pretrained("facebook/DiT-XL-2")
        patched = lz.patch_dit(dit)
        output = patched(hidden_states, timestep, class_labels)
        ```
    """
    from layerzero.integrations.model_patching import apply_patches, ATTENTION_MODULE_PATTERNS

    # Register DiT patterns if not already
    if "dit" not in ATTENTION_MODULE_PATTERNS:
        ATTENTION_MODULE_PATTERNS["dit"] = DIFFUSER_ATTENTION_PATTERNS["dit"]

    apply_patches(dit, attention_impl=attention_impl)
    return dit


def unpatch_dit(dit: "nn.Module") -> "nn.Module":
    """Restore original DiT attention implementation.

    Undoes the patching applied by patch_dit().

    Args:
        dit: Previously patched DiT.

    Returns:
        The unpatched DiT (same object, modified in-place).
    """
    from layerzero.integrations.model_patching import remove_patches

    remove_patches(dit)
    return dit


def patch_pipeline(
    pipeline: Any,
    attention_impl: str = "auto",
) -> Any:
    """Patch all attention modules in a diffusion pipeline.

    Patches both UNet and any transformer components.

    Args:
        pipeline: HuggingFace Diffusers pipeline.
        attention_impl: Attention implementation to use.

    Returns:
        The patched pipeline (same object, modified in-place).

    Example:
        ```python
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained("...")
        patched = lz.patch_pipeline(pipe)
        image = patched("a photo of a cat").images[0]
        ```
    """
    # Patch UNet if present
    if hasattr(pipeline, "unet") and pipeline.unet is not None:
        patch_unet(pipeline.unet, attention_impl)
        logger.debug("Patched pipeline UNet")

    # Patch transformer if present (for DiT-based pipelines like Flux)
    if hasattr(pipeline, "transformer") and pipeline.transformer is not None:
        patch_dit(pipeline.transformer, attention_impl)
        logger.debug("Patched pipeline transformer")

    return pipeline


def unpatch_pipeline(pipeline: Any) -> Any:
    """Restore original attention in a diffusion pipeline.

    Undoes the patching applied by patch_pipeline().

    Args:
        pipeline: Previously patched pipeline.

    Returns:
        The unpatched pipeline (same object, modified in-place).
    """
    # Unpatch UNet if present
    if hasattr(pipeline, "unet") and pipeline.unet is not None:
        unpatch_unet(pipeline.unet)
        logger.debug("Unpatched pipeline UNet")

    # Unpatch transformer if present
    if hasattr(pipeline, "transformer") and pipeline.transformer is not None:
        unpatch_dit(pipeline.transformer)
        logger.debug("Unpatched pipeline transformer")

    return pipeline


class DiffusersPatcher:
    """Context manager for Diffusers model patching.

    Applies patches on enter and removes them on exit.

    Example:
        ```python
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained("...")
        with DiffusersPatcher(pipe) as patched:
            image = patched("a photo of a cat").images[0]
        # Original pipeline restored after context
        ```
    """

    def __init__(
        self,
        model_or_pipeline: Any,
        attention_impl: str = "auto",
    ) -> None:
        """Initialize patcher.

        Args:
            model_or_pipeline: Model or pipeline to patch.
            attention_impl: Attention implementation to use.
        """
        self.target = model_or_pipeline
        self.attention_impl = attention_impl
        self._patched = False
        self._is_pipeline = hasattr(model_or_pipeline, "unet") or hasattr(model_or_pipeline, "transformer")

    def __enter__(self) -> Any:
        """Apply patches and return model/pipeline."""
        if self._is_pipeline:
            patch_pipeline(self.target, self.attention_impl)
        else:
            model_type = _get_diffuser_model_type(self.target)
            if model_type == "unet":
                patch_unet(self.target, self.attention_impl)
            elif model_type == "dit":
                patch_dit(self.target, self.attention_impl)
            else:
                # Try generic patching
                from layerzero.integrations.model_patching import apply_patches
                apply_patches(self.target, self.attention_impl)

        self._patched = True
        return self.target

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Remove patches."""
        if self._patched:
            if self._is_pipeline:
                unpatch_pipeline(self.target)
            else:
                model_type = _get_diffuser_model_type(self.target)
                if model_type == "unet":
                    unpatch_unet(self.target)
                elif model_type == "dit":
                    unpatch_dit(self.target)
                else:
                    from layerzero.integrations.model_patching import remove_patches
                    remove_patches(self.target)

            self._patched = False
