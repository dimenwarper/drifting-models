"""Multi-scale pooling strategies for text features.

Text analogs of the paper's spatial pooling:
- Per-position: each token position's hidden state (subsampled)
- Global: mean + std across positions
- Window-4: mean + std over non-overlapping 4-token windows
- Window-16: mean + std over non-overlapping 16-token windows
"""

import torch
import torch.nn as nn


class TextFeaturePooler(nn.Module):
    """Pool sequence features into multi-scale representations."""

    def __init__(
        self,
        hidden_dim: int = 768,
        n_subsample_positions: int = 32,
        window_sizes: tuple[int, ...] = (4, 16),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_subsample_positions = n_subsample_positions
        self.window_sizes = window_sizes

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        offset: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Pool hidden states into multiple scales.

        Args:
            hidden_states: (B, seq_len, C)
            attention_mask: (B, seq_len), 1 for valid, 0 for padding
            offset: number of leading positions to skip (e.g., prefix length)

        Returns:
            Dict with keys:
                "per_position": (B * n_subsample, C)
                "global": (B, 2*C)
                "window_4": (B * n_windows, 2*C)
                "window_16": (B * n_windows, 2*C)
        """
        if offset > 0:
            hidden_states = hidden_states[:, offset:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, offset:]

        B, S, C = hidden_states.shape
        result = {}

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
        else:
            mask = torch.ones(B, S, 1, device=hidden_states.device)

        masked = hidden_states * mask

        # 1. Per-position (subsampled)
        n_sub = min(self.n_subsample_positions, S)
        idx = torch.randperm(S, device=hidden_states.device)[:n_sub]
        idx = idx.sort().values
        result["per_position"] = hidden_states[:, idx].reshape(B * n_sub, C)

        # 2. Global: mean + std
        lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        global_mean = masked.sum(dim=1) / lengths  # (B, C)
        global_var = ((masked - global_mean.unsqueeze(1)) * mask).pow(2).sum(dim=1) / lengths
        global_std = global_var.clamp(min=1e-8).sqrt()
        result["global"] = torch.cat([global_mean, global_std], dim=-1)  # (B, 2C)

        # 3. Window pooling
        for ws in self.window_sizes:
            n_windows = S // ws
            if n_windows == 0:
                continue
            truncated = hidden_states[:, :n_windows * ws]  # (B, n_windows*ws, C)
            truncated_mask = mask[:, :n_windows * ws]

            windows = truncated.reshape(B, n_windows, ws, C)
            w_mask = truncated_mask.reshape(B, n_windows, ws, 1)

            w_lengths = w_mask.sum(dim=2).clamp(min=1)  # (B, n_windows, 1)
            w_mean = (windows * w_mask).sum(dim=2) / w_lengths  # (B, n_windows, C)
            w_var = ((windows - w_mean.unsqueeze(2)) * w_mask).pow(2).sum(dim=2) / w_lengths
            w_std = w_var.clamp(min=1e-8).sqrt()

            pooled = torch.cat([w_mean, w_std], dim=-1)  # (B, n_windows, 2C)
            result[f"window_{ws}"] = pooled.reshape(B * n_windows, 2 * C)

        return result


def pool_features(
    layer_features: dict[int, torch.Tensor],
    pooler: TextFeaturePooler,
    attention_mask: torch.Tensor | None = None,
    offset: int = 0,
) -> list[tuple[str, torch.Tensor]]:
    """Pool features from all layers into a flat list.

    Args:
        layer_features: dict from layer index to (B, seq_len, C)
        pooler: TextFeaturePooler instance
        attention_mask: (B, seq_len)
        offset: number of leading positions to skip (e.g., prefix length)

    Returns:
        List of (name, tensor) pairs for each (layer, pool_type) combination.
    """
    all_pooled = []
    for layer_idx in sorted(layer_features.keys()):
        pooled = pooler(layer_features[layer_idx], attention_mask, offset=offset)
        for pool_name, tensor in pooled.items():
            all_pooled.append((f"L{layer_idx}_{pool_name}", tensor))
    return all_pooled
