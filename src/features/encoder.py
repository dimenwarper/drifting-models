"""Frozen GPT-2 feature encoder with multi-layer extraction."""

import torch
import torch.nn as nn
from transformers import GPT2Model


class GPT2FeatureEncoder(nn.Module):
    """Extract multi-scale features from frozen GPT-2.

    Extracts hidden states from layers {3, 6, 9, 12} (1-indexed).
    """

    EXTRACT_LAYERS = (3, 6, 9, 12)

    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.gpt2.eval()
        for p in self.gpt2.parameters():
            p.requires_grad_(False)
        self.hidden_dim = self.gpt2.config.n_embd  # 768

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        """Extract features from specified layers.

        Args:
            input_ids: (B, seq_len) token ids — use for real data
            inputs_embeds: (B, seq_len, 768) continuous embeddings — use for generated data
            attention_mask: (B, seq_len) attention mask

        Returns:
            Dict mapping layer index to hidden states (B, seq_len, 768)
        """
        outputs = self.gpt2(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # hidden_states is a tuple of (n_layers + 1) tensors
        # Index 0 is the embedding layer output, index i is layer i output
        hidden_states = outputs.hidden_states

        features = {}
        for layer_idx in self.EXTRACT_LAYERS:
            features[layer_idx] = hidden_states[layer_idx]

        return features

    def get_vocab_embeddings(self) -> torch.Tensor:
        """Return GPT-2's token embedding matrix (vocab_size, 768)."""
        return self.gpt2.wte.weight.data

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token ids to embeddings using GPT-2's wte.

        Args:
            input_ids: (B, seq_len)

        Returns:
            (B, seq_len, 768) embeddings
        """
        return self.gpt2.wte(input_ids)
