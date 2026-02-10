"""Learned smooth projection for feature space regularization.

Projects frozen encoder features into a lower-dimensional space trained
to be locally smooth — small input perturbations produce small output changes.
This stabilizes the drifting field V in late training when the generator
is near the real manifold.
"""

import torch
import torch.nn as nn


class SmoothProjectionBank(nn.Module):
    """Bank of projection MLPs, one per feature scale.

    Each MLP maps pooled encoder features (768 or 1536-dim) to a
    lower-dimensional smooth space. Trained jointly with the generator
    via the drifting loss, plus an optional Lipschitz smoothness penalty.
    """

    def __init__(self, proj_dim: int = 256, hidden_mult: int = 2):
        super().__init__()
        self.proj_dim = proj_dim
        self.hidden_mult = hidden_mult
        self.projections = nn.ModuleDict()

    def _make_mlp(self, in_dim: int) -> nn.Sequential:
        hidden = self.proj_dim * self.hidden_mult
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.proj_dim),
        )

    def _get_proj(self, name: str, in_dim: int) -> nn.Sequential:
        # Lazily create projections as we see new feature scales
        key = name.replace(".", "_")  # ModuleDict doesn't allow dots
        if key not in self.projections:
            self.projections[key] = self._make_mlp(in_dim)
        return self.projections[key]

    def forward(
        self, features: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        """Project all feature scales.

        Args:
            features: list of (name, tensor) from pool_features

        Returns:
            list of (name, projected_tensor) with dim = proj_dim
        """
        out = []
        for name, feat in features:
            proj = self._get_proj(name, feat.shape[-1])
            proj = proj.to(feat.device)
            out.append((name, proj(feat)))
        return out

    def smoothness_loss(
        self,
        features: list[tuple[str, torch.Tensor]],
        noise_std: float = 0.1,
    ) -> torch.Tensor:
        """Lipschitz smoothness penalty: ||proj(x+e) - proj(x)||² / ||e||².

        Penalizes projections that amplify small perturbations.

        Args:
            features: list of (name, tensor) — clean (unprojected) features
            noise_std: std of Gaussian perturbation

        Returns:
            scalar smoothness loss (mean ratio across all scales)
        """
        ratios = []
        for name, feat in features:
            proj = self._get_proj(name, feat.shape[-1])
            proj = proj.to(feat.device)

            noise = torch.randn_like(feat) * noise_std
            out_clean = proj(feat)
            out_noisy = proj(feat + noise)

            delta_out = (out_clean - out_noisy).pow(2).sum(dim=-1)  # (N,)
            delta_in = noise.pow(2).sum(dim=-1).clamp(min=1e-8)     # (N,)
            ratios.append((delta_out / delta_in).mean())

        return torch.stack(ratios).mean()
