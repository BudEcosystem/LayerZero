"""
LayerZero Framework Integrations

Integrations with popular ML frameworks:
- HuggingFace Transformers
- HuggingFace Diffusers
- vLLM
"""
from layerzero.integrations.transformers import (
    is_transformers_available,
    get_transformers_version,
    patch_model,
    unpatch_model,
)
from layerzero.integrations.model_patching import (
    ModelPatcher,
    get_attention_module_names,
)
from layerzero.integrations.diffusers import (
    is_diffusers_available,
    get_diffusers_version,
    patch_unet,
    unpatch_unet,
    patch_dit,
    unpatch_dit,
    patch_pipeline,
    unpatch_pipeline,
    DiffusersPatcher,
)
from layerzero.integrations.tokenization_pipeline import (
    TokenizationPipeline,
    TokenizedBatch,
    TokenizerType,
    auto_select_tokenizer,
    get_tokenizer_for_model,
    create_pipeline_tokenizer,
)

__all__ = [
    # Transformers
    "is_transformers_available",
    "get_transformers_version",
    "patch_model",
    "unpatch_model",
    "ModelPatcher",
    "get_attention_module_names",
    # Diffusers
    "is_diffusers_available",
    "get_diffusers_version",
    "patch_unet",
    "unpatch_unet",
    "patch_dit",
    "unpatch_dit",
    "patch_pipeline",
    "unpatch_pipeline",
    "DiffusersPatcher",
    # Tokenization Pipeline
    "TokenizationPipeline",
    "TokenizedBatch",
    "TokenizerType",
    "auto_select_tokenizer",
    "get_tokenizer_for_model",
    "create_pipeline_tokenizer",
]
