"""Spectral regularization for generator weight matrices."""

import torch
import torch.nn.functional as F


def spectral_regularization(model: torch.nn.Module, target_norm: float = 1.0) -> torch.Tensor:
    """Soft spectral norm penalty using power iteration.

    Penalizes (sigma_max - target_norm)^2 for each weight matrix,
    encouraging bounded spectral norms to prevent rank collapse.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0
    for param in model.parameters():
        if param.dim() >= 2 and param.shape[0] >= 4:
            # Approximate top singular value via power iteration (3 iters)
            u = torch.randn(param.shape[0], 1, device=param.device)
            for _ in range(3):
                v = F.normalize(param.t() @ u, dim=0)
                u = F.normalize(param @ v, dim=0)
            sigma = (u.t() @ param @ v).squeeze()
            loss = loss + (sigma - target_norm) ** 2
            count += 1
    return loss / max(count, 1)
