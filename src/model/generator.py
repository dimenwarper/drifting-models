"""DiT-style transformer generator with adaLN-Zero, SwiGLU, RMSNorm, QK-Norm, RoPE."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.rope import RotaryPositionEmbedding, apply_rope
from src.model.style_embeddings import StyleEmbeddings


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # Round to nearest multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class QKNormAttention(nn.Module):
    """Multi-head attention with QK-Norm and RoPE."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q = q.transpose(1, 2)  # (B, heads, S, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.wo(out)


class AdaLNZero(nn.Module):
    """Adaptive Layer Norm Zero modulation.

    Produces scale, shift, gate parameters from conditioning vector.
    """

    def __init__(self, dim: int, n_modulations: int = 6):
        super().__init__()
        self.proj = nn.Linear(dim, n_modulations * dim)
        self.n_modulations = n_modulations
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Args:
            c: (B, dim) conditioning vector

        Returns:
            Tuple of n_modulations tensors, each (B, 1, dim)
        """
        modulations = self.proj(c).unsqueeze(1)  # (B, 1, n_mod * dim)
        return modulations.chunk(self.n_modulations, dim=-1)


class DiTBlock(nn.Module):
    """DiT transformer block with adaLN-Zero."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = QKNormAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)
        self.adaln = AdaLNZero(dim, n_modulations=6)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        # adaLN modulations: scale1, shift1, gate1, scale2, shift2, gate2
        s1, sh1, g1, s2, sh2, g2 = self.adaln(c)

        # Attention branch
        h = self.norm1(x)
        h = h * (1 + s1) + sh1
        h = self.attn(h, rope_cos, rope_sin)
        x = x + g1 * h

        # FFN branch
        h = self.norm2(x)
        h = h * (1 + s2) + sh2
        h = self.ffn(h)
        x = x + g2 * h

        return x


class DriftingGenerator(nn.Module):
    """DiT-style generator for text drifting models.

    Maps noise (B, seq_len, hidden_dim) to continuous embeddings in GPT-2's space.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        seq_len: int = 256,
        n_style_tokens: int = 32,
        codebook_size: int = 64,
        vocab_embed_mean: torch.Tensor | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Input projection
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # Style embeddings
        self.style = StyleEmbeddings(n_style_tokens, codebook_size, hidden_dim)

        # Conditioning MLP (processes noise statistics + style)
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # RoPE
        self.rope = RotaryPositionEmbedding(hidden_dim // n_heads, max_seq_len=seq_len)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, n_heads) for _ in range(n_layers)
        ])

        # Final norm + projection
        self.final_norm = RMSNorm(hidden_dim)
        self.final_adaln = AdaLNZero(hidden_dim, n_modulations=2)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Initialize output projection bias near GPT-2 vocab embedding mean
        if vocab_embed_mean is not None and vocab_embed_mean.shape[0] == hidden_dim:
            with torch.no_grad():
                self.output_proj.bias.copy_(vocab_embed_mean)

        self._init_weights()

    def _init_weights(self) -> None:
        # Standard transformer init
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.output_proj:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise: (B, seq_len, hidden_dim) Gaussian noise

        Returns:
            (B, seq_len, hidden_dim) continuous embeddings
        """
        B, S, C = noise.shape

        # Project input
        x = self.input_proj(noise)

        # Build conditioning: global noise stats + style
        noise_mean = noise.mean(dim=1)  # (B, C)
        style = self.style(B, noise.device)  # (B, C)
        cond = self.cond_mlp(noise_mean + style)  # (B, C)

        # Get RoPE
        rope_cos, rope_sin = self.rope(S)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond, rope_cos, rope_sin)

        # Final projection
        s, sh = self.final_adaln(cond)
        x = self.final_norm(x)
        x = x * (1 + s) + sh
        x = self.output_proj(x)

        return x
