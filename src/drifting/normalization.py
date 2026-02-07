"""Feature normalization and drift normalization for drifting models."""

import torch


class FeatureNormalizer:
    """Normalize features so average pairwise distance equals sqrt(C).

    From the paper: S = (1/sqrt(C)) * E[||phi(x) - phi(y)||]
    Normalized features: phi_tilde = phi / S
    """

    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self.running_scale: float | None = None

    @torch.no_grad()
    def update_scale(self, features: torch.Tensor) -> None:
        """Update running scale estimate from a batch of features.

        Args:
            features: (N, C) feature vectors
        """
        N, C = features.shape
        if N < 2:
            return

        # Compute average pairwise L2 distance (subsample if large)
        if N > 256:
            idx = torch.randperm(N, device=features.device)[:256]
            features = features[idx]
            N = 256

        # Pairwise distances
        diff = features.unsqueeze(0) - features.unsqueeze(1)  # (N, N, C)
        dists = diff.norm(dim=-1)  # (N, N)

        # Exclude diagonal
        mask = ~torch.eye(N, device=features.device, dtype=torch.bool)
        avg_dist = dists[mask].mean().item()

        scale = avg_dist / (C ** 0.5)

        if self.running_scale is None:
            self.running_scale = scale
        else:
            self.running_scale = self.momentum * self.running_scale + (1 - self.momentum) * scale

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features by the running scale.

        Args:
            features: (N, C) or (N, ..., C)

        Returns:
            Normalized features with average pairwise distance ~ sqrt(C)
        """
        if self.running_scale is None or self.running_scale < 1e-8:
            return features
        return features / self.running_scale


class DriftNormalizer:
    """Normalize drift vectors so E[||V||^2 / C] ~ 1.

    From the paper: V_tilde = V / lambda
    where lambda = sqrt(E[||V||^2 / C])
    """

    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self.running_lambda: float | None = None

    @torch.no_grad()
    def update_lambda(self, V: torch.Tensor) -> None:
        """Update running lambda from drift vectors.

        Args:
            V: (N, C) drift vectors
        """
        C = V.shape[-1]
        # E[||V||^2 / C]
        mean_sq_norm = (V * V).sum(dim=-1).mean().item() / C
        lam = mean_sq_norm ** 0.5

        if self.running_lambda is None:
            self.running_lambda = max(lam, 1e-8)
        else:
            self.running_lambda = self.momentum * self.running_lambda + (1 - self.momentum) * max(lam, 1e-8)

    def normalize(self, V: torch.Tensor) -> torch.Tensor:
        """Normalize drift vectors.

        Args:
            V: (N, C) drift vectors

        Returns:
            Normalized drift vectors
        """
        if self.running_lambda is None or self.running_lambda < 1e-8:
            return V
        return V / self.running_lambda
