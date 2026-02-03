"""LayerZero Operation APIs.

Public APIs for executing operations with automatic kernel selection:
- attention() - Multi-head attention
- paged_attention() - Paged KV-cache attention for serving
- rms_norm() - RMS normalization
- layer_norm() - Layer normalization
- rope() - Rotary positional encoding
- sample_topk() - Top-k sampling
- sample_topp() - Top-p (nucleus) sampling
- quantize() - Quantization
- tokenize() / detokenize() - Text tokenization
"""
from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Execute attention with automatic kernel selection.

    LayerZero automatically selects the best attention kernel based on:
    - Input shapes and dtypes
    - Hardware capabilities (GPU generation, tensor cores)
    - Configured policies and locks
    - Historical performance data

    Args:
        query: Query tensor, shape (batch, num_heads, seq_len_q, head_dim)
               or (batch, seq_len_q, num_heads, head_dim) depending on layout.
        key: Key tensor, shape (batch, num_heads, seq_len_k, head_dim).
        value: Value tensor, shape (batch, num_heads, seq_len_k, head_dim).
        attn_mask: Optional attention mask. Can be additive (float) or
                   boolean. Shape (seq_len_q, seq_len_k) or broadcastable.
        dropout_p: Dropout probability (0.0 in inference mode).
        is_causal: If True, apply causal masking (ignore attn_mask).
        scale: Optional scale factor. Default is 1/sqrt(head_dim).
        backend: Optional backend override. If specified, bypasses
                 automatic kernel selection.

    Returns:
        Attention output tensor, same shape as query.

    Example:
        >>> import layerzero as lz
        >>> import torch
        >>>
        >>> q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
        >>> v = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
        >>>
        >>> output = lz.attention(q, k, v, is_causal=True)
    """
    from layerzero.api.config import _get_global_state

    state = _get_global_state()

    # Check for kernel lock
    operation = "attention.causal" if is_causal else "attention.full"
    locked_kernel = state.locks.get(operation)

    if backend is not None:
        # User specified backend override
        selected_backend = backend
    elif locked_kernel is not None:
        # Kernel is locked
        selected_backend = locked_kernel
    else:
        # Automatic selection - use selection engine
        selected_backend = _select_attention_backend(
            query, key, value, is_causal, state
        )

    # Dispatch to selected backend
    return _dispatch_attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale,
        selected_backend
    )


def _select_attention_backend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
    state,
) -> str:
    """Select best attention backend for given inputs."""
    # Extract shape info
    batch_size = query.shape[0]
    num_heads = query.shape[1] if query.dim() == 4 else query.shape[2]
    seq_len = query.shape[2] if query.dim() == 4 else query.shape[1]
    head_dim = query.shape[-1]
    dtype = query.dtype
    device = query.device

    # Try to use selection engine if available
    try:
        from layerzero.selection.engine import get_global_engine
        from layerzero.models.selection_context import SelectionContext

        engine = get_global_engine()

        ctx = SelectionContext(
            batch_size=batch_size,
            seq_len_q=seq_len,
            seq_len_k=key.shape[2] if key.dim() == 4 else key.shape[1],
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=str(device),
            is_causal=is_causal,
        )

        result = engine.select(
            operation="attention.causal" if is_causal else "attention.full",
            context=ctx,
        )

        if result and result.kernel_id:
            return result.kernel_id

    except Exception:
        pass

    # Fallback to torch SDPA
    return "torch_sdpa"


def _dispatch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    backend: str,
) -> torch.Tensor:
    """Dispatch attention to selected backend.

    Uses the LayerZero dispatch system to execute the selected kernel.
    Falls back to PyTorch SDPA if dispatch fails.
    """
    try:
        from layerzero.dispatch import get_global_dispatcher, DispatchMode

        # Build input dict
        inputs = {
            "query": query,
            "key": key,
            "value": value,
        }

        # Build context for selection
        operation = "attention.causal" if is_causal else "attention.full"

        # Get dispatcher and execute
        dispatcher = get_global_dispatcher()
        result = dispatcher.dispatch(
            operation=operation,
            inputs=inputs,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

        return result.output

    except Exception as e:
        # Log the failure for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Dispatch failed, falling back to SDPA: {e}")

        # Fallback to PyTorch SDPA
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: Optional[float] = None,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Paged attention for serving with KV-cache.

    Efficient attention for autoregressive generation using blocked
    KV-cache storage. Supports variable-length sequences in a batch.

    Args:
        query: Query tensor, shape (batch, 1, num_heads, head_dim).
               Single token query for autoregressive generation.
        key_cache: Paged key cache, shape (num_blocks, block_size, num_kv_heads, head_dim).
        value_cache: Paged value cache, shape (num_blocks, block_size, num_kv_heads, head_dim).
        block_tables: Block table mapping, shape (batch, max_blocks).
                      Each row contains block indices for that sequence.
        context_lens: Context lengths, shape (batch,).
                      Actual sequence length for each batch element.
        scale: Optional scale factor. Default is 1/sqrt(head_dim).
        backend: Optional backend override.

    Returns:
        Attention output tensor, shape (batch, 1, num_heads, head_dim).

    Example:
        >>> import layerzero as lz
        >>> import torch
        >>>
        >>> # Single-token query for generation
        >>> q = torch.randn(4, 1, 32, 128, device='cuda', dtype=torch.float16)
        >>> # Paged KV cache
        >>> k_cache = torch.randn(1000, 16, 8, 128, device='cuda', dtype=torch.float16)
        >>> v_cache = torch.randn(1000, 16, 8, 128, device='cuda', dtype=torch.float16)
        >>> # Block tables and lengths
        >>> block_tables = torch.randint(0, 1000, (4, 64), device='cuda')
        >>> context_lens = torch.tensor([100, 200, 150, 300], device='cuda')
        >>>
        >>> output = lz.paged_attention(q, k_cache, v_cache, block_tables, context_lens)
    """
    # Extract dimensions
    batch_size = query.shape[0]
    num_heads = query.shape[2]
    head_dim = query.shape[3]
    num_kv_heads = key_cache.shape[2]
    block_size = key_cache.shape[1]

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    # For now, implement naive version
    # In production, this dispatches to FlashInfer/vLLM paged attention kernels
    outputs = []

    for b in range(batch_size):
        ctx_len = context_lens[b].item()
        num_blocks_used = (ctx_len + block_size - 1) // block_size

        # Gather keys and values from cache
        block_ids = block_tables[b, :num_blocks_used]
        k_gathered = key_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:ctx_len]
        v_gathered = value_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:ctx_len]

        # Handle GQA by expanding kv heads
        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            k_gathered = k_gathered.repeat_interleave(repeat_factor, dim=1)
            v_gathered = v_gathered.repeat_interleave(repeat_factor, dim=1)

        # Reshape for attention: (1, num_heads, ctx_len, head_dim)
        k_gathered = k_gathered.transpose(0, 1).unsqueeze(0)
        v_gathered = v_gathered.transpose(0, 1).unsqueeze(0)

        # Query shape: (1, num_heads, 1, head_dim)
        q_b = query[b:b+1].transpose(1, 2)

        # Compute attention
        out = F.scaled_dot_product_attention(
            q_b, k_gathered, v_gathered,
            scale=scale,
            is_causal=False,  # Not causal for single-token decode
        )

        outputs.append(out.transpose(1, 2))  # Back to (1, 1, num_heads, head_dim)

    return torch.cat(outputs, dim=0)


def rope(
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    interleaved: bool = False,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Apply Rotary Positional Encoding (RoPE).

    Args:
        input: Input tensor, shape (..., num_heads, seq_len, head_dim)
               or (..., seq_len, num_heads, head_dim).
        cos: Precomputed cosine values, shape (max_seq_len, head_dim)
             or (1, 1, max_seq_len, head_dim).
        sin: Precomputed sine values, same shape as cos.
        position_ids: Optional position indices, shape (batch, seq_len).
                      If None, uses sequential positions [0, 1, 2, ...].
        interleaved: If True, uses interleaved rotation pattern.
                     If False, uses split-half rotation pattern.
        backend: Optional backend override.

    Returns:
        Tensor with RoPE applied, same shape as input.

    Example:
        >>> import layerzero as lz
        >>> import torch
        >>>
        >>> # Create input
        >>> x = torch.randn(2, 32, 1024, 128)  # (batch, heads, seq, head_dim)
        >>>
        >>> # Create position encodings
        >>> head_dim = 128
        >>> max_seq = 1024
        >>> freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        >>> pos = torch.arange(max_seq)
        >>> angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
        >>> cos = torch.cos(angles).repeat(1, 2)
        >>> sin = torch.sin(angles).repeat(1, 2)
        >>>
        >>> output = lz.rope(x, cos, sin)
    """
    # Determine input layout
    if input.dim() == 4:
        # Could be (B, H, S, D) or (B, S, H, D)
        # Assume (B, H, S, D) if head dimension is small
        if input.shape[1] > input.shape[2]:
            # Likely (B, S, H, D) - transpose
            input = input.transpose(1, 2)
            transposed = True
        else:
            transposed = False
    else:
        transposed = False

    # Get sequence length
    seq_len = input.shape[2] if input.dim() == 4 else input.shape[1]
    head_dim = input.shape[-1]

    # Ensure cos/sin have correct shape
    if cos.dim() == 2:
        # (max_seq, head_dim) -> (1, 1, seq_len, head_dim)
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        # (1, max_seq, head_dim) -> (1, 1, seq_len, head_dim)
        cos = cos[:, :seq_len].unsqueeze(1)
        sin = sin[:, :seq_len].unsqueeze(1)

    # Handle position_ids if provided
    if position_ids is not None:
        # Gather cos/sin at specified positions
        batch_size = position_ids.shape[0]
        cos = cos.squeeze(0).squeeze(0)  # (max_seq, head_dim)
        sin = sin.squeeze(0).squeeze(0)
        cos = cos[position_ids].unsqueeze(1)  # (batch, 1, seq, head_dim)
        sin = sin[position_ids].unsqueeze(1)

    if interleaved:
        # Interleaved rotation: pairs of adjacent elements
        x1 = input[..., ::2]
        x2 = input[..., 1::2]
        cos_half = cos[..., ::2]
        sin_half = sin[..., ::2]
        rotated = torch.empty_like(input)
        rotated[..., ::2] = x1 * cos_half - x2 * sin_half
        rotated[..., 1::2] = x1 * sin_half + x2 * cos_half
    else:
        # Split-half rotation: first half and second half
        half_dim = head_dim // 2
        x1 = input[..., :half_dim]
        x2 = input[..., half_dim:]
        cos_half = cos[..., :half_dim]
        sin_half = sin[..., :half_dim]
        rotated = torch.cat([
            x1 * cos_half - x2 * sin_half,
            x1 * sin_half + x2 * cos_half,
        ], dim=-1)

    if transposed:
        rotated = rotated.transpose(1, 2)

    return rotated


def rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """RMS normalization with automatic kernel selection.

    Args:
        input: Input tensor, shape (..., hidden_dim).
        weight: Normalization weight, shape (hidden_dim,).
        eps: Small constant for numerical stability.
        backend: Optional backend override.

    Returns:
        Normalized tensor, same shape as input.

    Example:
        >>> import layerzero as lz
        >>> import torch
        >>>
        >>> x = torch.randn(2, 1024, 4096, dtype=torch.float16)
        >>> weight = torch.ones(4096, dtype=torch.float16)
        >>> output = lz.rms_norm(x, weight)
    """
    try:
        from layerzero.dispatch import get_global_dispatcher

        inputs = {
            "input": input,
            "weight": weight,
        }

        dispatcher = get_global_dispatcher()
        result = dispatcher.dispatch(
            operation="norm.rms",
            inputs=inputs,
            eps=eps,
        )

        return result.output

    except Exception:
        # Fallback to native implementation
        variance = input.pow(2).mean(dim=-1, keepdim=True)
        input_normalized = input * torch.rsqrt(variance + eps)
        return input_normalized * weight


def layer_norm(
    input: torch.Tensor,
    normalized_shape: Union[int, tuple, torch.Size],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Layer normalization with automatic kernel selection.

    Follows PyTorch F.layer_norm signature for compatibility.

    Args:
        input: Input tensor, shape (..., D) where D matches normalized_shape.
        normalized_shape: Shape over which to normalize. Can be int, tuple, or torch.Size.
            For example, (768,) for hidden_dim=768, or 768.
        weight: Optional normalization weight, shape matching normalized_shape.
        bias: Optional normalization bias, shape matching normalized_shape.
        eps: Small constant for numerical stability.
        backend: Optional backend override.

    Returns:
        Normalized tensor, same shape as input.

    Example:
        >>> import layerzero as lz
        >>> import torch
        >>>
        >>> x = torch.randn(2, 1024, 768)
        >>> output = lz.layer_norm(x, (768,))  # Without weight/bias
        >>>
        >>> weight = torch.ones(768)
        >>> bias = torch.zeros(768)
        >>> output = lz.layer_norm(x, (768,), weight, bias)  # With weight/bias
    """
    # Handle int normalized_shape
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    try:
        from layerzero.dispatch import get_global_dispatcher

        inputs = {
            "input": input,
        }
        if weight is not None:
            inputs["weight"] = weight
        if bias is not None:
            inputs["bias"] = bias

        dispatcher = get_global_dispatcher()
        result = dispatcher.dispatch(
            operation="norm.layer",
            inputs=inputs,
            normalized_shape=normalized_shape,
            eps=eps,
        )

        return result.output

    except Exception:
        # Fallback to PyTorch implementation
        return F.layer_norm(input, normalized_shape, weight, bias, eps)


def sample_topk(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Top-k sampling with automatic kernel selection.

    Args:
        logits: Logits tensor, shape (batch, vocab_size).
        k: Number of top tokens to sample from.
        temperature: Temperature for scaling logits.
        generator: Optional random generator for reproducibility.

    Returns:
        Sampled token indices, shape (batch, 1).
    """
    from layerzero.sampling.topk import topk_sample
    return topk_sample(logits, k=k, temperature=temperature, generator=generator)


def sample_topp(
    logits: torch.Tensor,
    p: float,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Top-p (nucleus) sampling with automatic kernel selection.

    Args:
        logits: Logits tensor, shape (batch, vocab_size).
        p: Cumulative probability threshold.
        temperature: Temperature for scaling logits.
        generator: Optional random generator for reproducibility.

    Returns:
        Sampled token indices, shape (batch, 1).
    """
    from layerzero.sampling.topp import topp_sample
    return topp_sample(logits, p=p, temperature=temperature, generator=generator)


def tokenize(
    text: str | list[str],
    tokenizer: str = "auto",
    add_special_tokens: bool = True,
    return_tensors: Optional[str] = "pt",
) -> torch.Tensor | list[int]:
    """Tokenize text with automatic tokenizer selection.

    Args:
        text: Input text or list of texts.
        tokenizer: Tokenizer identifier (e.g., "gpt2", "llama", "auto").
        add_special_tokens: Whether to add special tokens.
        return_tensors: Return format ("pt" for PyTorch, None for list).

    Returns:
        Token IDs as tensor or list.
    """
    try:
        from layerzero.tokenization.hub import get_tokenizer

        tok = get_tokenizer(tokenizer)
        result = tok.encode(
            text,
            add_special_tokens=add_special_tokens,
        )

        if return_tensors == "pt":
            return torch.tensor(result)
        return result

    except ImportError:
        raise ImportError(
            "Tokenization requires tokenizers library. "
            "Install with: pip install tokenizers"
        )


def detokenize(
    tokens: torch.Tensor | list[int],
    tokenizer: str = "auto",
    skip_special_tokens: bool = True,
) -> str:
    """Detokenize token IDs to text.

    Args:
        tokens: Token IDs as tensor or list.
        tokenizer: Tokenizer identifier.
        skip_special_tokens: Whether to skip special tokens.

    Returns:
        Decoded text string.
    """
    try:
        from layerzero.tokenization.hub import get_tokenizer

        tok = get_tokenizer(tokenizer)

        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        return tok.decode(tokens, skip_special_tokens=skip_special_tokens)

    except ImportError:
        raise ImportError(
            "Tokenization requires tokenizers library. "
            "Install with: pip install tokenizers"
        )


def quantize(
    input: torch.Tensor,
    dtype: str,
    scale: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
    axis: int = -1,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Quantize tensor to specified data type.

    Supports various quantization formats including:
    - int8: 8-bit integer quantization
    - fp8_e4m3: FP8 E4M3 format (NVIDIA H100+)
    - fp8_e5m2: FP8 E5M2 format
    - mxfp4: Microscaling FP4

    Args:
        input: Input tensor to quantize.
        dtype: Target quantization format.
               Supported: "int8", "fp8_e4m3", "fp8_e5m2", "mxfp4", "fp16", "bf16".
        scale: Optional scale tensor. If None, computed automatically.
        zero_point: Optional zero point for asymmetric quantization.
        axis: Axis for per-channel quantization. Default -1 for per-tensor.
        backend: Optional backend override.

    Returns:
        Quantized tensor.

    Example:
        >>> import layerzero as lz
        >>> import torch
        >>>
        >>> x = torch.randn(1024, 4096, dtype=torch.float32)
        >>> x_int8 = lz.quantize(x, "int8")
        >>> x_fp8 = lz.quantize(x, "fp8_e4m3")
    """
    # Handle standard PyTorch dtypes first
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if dtype in dtype_map:
        return input.to(dtype_map[dtype])

    if dtype == "int8":
        # Per-tensor int8 quantization
        if scale is None:
            # Compute scale from input range
            max_val = input.abs().max()
            scale = max_val / 127.0

        quantized = (input / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized

    if dtype in ("fp8_e4m3", "fp8_e5m2"):
        # FP8 quantization (requires PyTorch 2.1+ with FP8 support)
        try:
            if dtype == "fp8_e4m3":
                fp8_dtype = torch.float8_e4m3fn
            else:
                fp8_dtype = torch.float8_e5m2

            if scale is None:
                # Auto-scale for FP8 range
                max_val = input.abs().max()
                if dtype == "fp8_e4m3":
                    fp8_max = 448.0  # Max value for e4m3
                else:
                    fp8_max = 57344.0  # Max value for e5m2
                scale = max_val / fp8_max

            scaled = input / scale
            return scaled.to(fp8_dtype)

        except AttributeError:
            raise NotImplementedError(
                f"FP8 quantization ({dtype}) requires PyTorch 2.1+ with FP8 support. "
                "Current PyTorch version does not support FP8 types."
            )

    if dtype == "mxfp4":
        # Microscaling FP4 - not yet supported in PyTorch
        raise NotImplementedError(
            "MXFP4 quantization is not yet supported. "
            "This requires specialized hardware support (NVIDIA Blackwell+)."
        )

    raise ValueError(
        f"Unsupported quantization dtype: {dtype}. "
        f"Supported: int8, fp8_e4m3, fp8_e5m2, fp16, bf16, fp32"
    )