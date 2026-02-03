"""
Model fixtures for LayerZero tests.

Provides mock model implementations for testing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MockAttentionModule(nn.Module):
    """Mock attention module for testing.

    Simple multi-head attention implementation for test fixtures.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """Initialize mock attention module.

        Args:
            hidden_size: Hidden dimension size.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor [batch, seq_len, hidden_size].
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply mask
        if attention_mask is not None:
            attn = attn + attention_mask

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(out)


class MockTransformerLayer(nn.Module):
    """Mock transformer layer for testing.

    Single transformer layer with attention and FFN.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_heads: int = 4,
        ffn_size: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialize mock transformer layer.

        Args:
            hidden_size: Hidden dimension size.
            num_heads: Number of attention heads.
            ffn_size: FFN intermediate size (default: 4 * hidden_size).
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_size = hidden_size
        ffn_size = ffn_size or 4 * hidden_size

        self.attention = MockAttentionModule(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor [batch, seq_len, hidden_size].
        """
        # Attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class MockTransformerModel(nn.Module):
    """Mock transformer model for testing.

    Stack of transformer layers with embedding.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 512,
    ) -> None:
        """Initialize mock transformer model.

        Args:
            vocab_size: Vocabulary size.
            hidden_size: Hidden dimension size.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            max_seq_len: Maximum sequence length.
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.layers = nn.ModuleList([
            MockTransformerLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # Config for compatibility
        self.config = type("Config", (), {
            "model_type": "mock",
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "vocab_size": vocab_size,
        })()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Optional attention mask.

        Returns:
            Logits tensor [batch, seq_len, vocab_size].
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        x = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        x = x + self.pos_embedding(positions)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Output
        x = self.norm(x)
        return self.lm_head(x)


class MockGQAAttention(nn.Module):
    """Mock GQA (Grouped Query Attention) module.

    For testing GQA implementations.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_q_heads: int = 8,
        num_kv_heads: int = 2,
    ) -> None:
        """Initialize mock GQA module.

        Args:
            hidden_size: Hidden dimension size.
            num_q_heads: Number of query heads.
            num_kv_heads: Number of key/value heads.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_q_heads
        self.num_key_value_groups = num_q_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor [batch, seq_len, hidden_size].
        """
        batch_size, seq_len, _ = x.shape

        # Project
        q = self.q_proj(x).view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KV for GQA
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(out)
