"""1D Rotary Position Embeddings (RoPE) for text sequences."""

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """1D RoPE for transformer attention."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin caches for the given sequence length.

        Returns:
            cos: (seq_len, dim/2)
            sin: (seq_len, dim/2)
        """
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.

    Args:
        x: (..., seq_len, dim) where dim is even
        cos: (seq_len, dim/2)
        sin: (seq_len, dim/2)

    Returns:
        Rotated tensor of same shape as x.
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]

    # Broadcast cos/sin to match x's batch dims
    cos = cos.unsqueeze(0)  # (1, seq_len, dim/2)
    sin = sin.unsqueeze(0)

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)
