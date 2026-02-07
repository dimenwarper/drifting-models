"""Random style embeddings (inspired by StyleGAN).

32 style tokens, each a random index into a 64-entry learnable codebook.
Summed and added to the conditioning vector.
"""

import torch
import torch.nn as nn


class StyleEmbeddings(nn.Module):
    """Random codebook-based style embeddings.

    At each forward pass, randomly sample indices from a learned codebook
    and sum them to produce a style conditioning vector.
    """

    def __init__(
        self,
        n_style_tokens: int = 32,
        codebook_size: int = 64,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.n_style_tokens = n_style_tokens
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, embed_dim)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random style embeddings.

        Args:
            batch_size: number of samples
            device: target device

        Returns:
            (batch_size, embed_dim) style conditioning vector
        """
        # Random indices: (batch_size, n_style_tokens)
        indices = torch.randint(
            0, self.codebook_size, (batch_size, self.n_style_tokens), device=device
        )
        # Look up and sum: (batch_size, n_style_tokens, embed_dim) -> (batch_size, embed_dim)
        embeddings = self.codebook(indices)
        return embeddings.sum(dim=1)
