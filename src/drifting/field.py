"""Drifting field computation: attraction - repulsion with double softmax."""

import torch
import torch.nn.functional as F


def pairwise_l2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise L2 distances between x and y.

    Args:
        x: (N, C)
        y: (M, C)

    Returns:
        (N, M) distance matrix.
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    xx = (x * x).sum(dim=-1, keepdim=True)  # (N, 1)
    yy = (y * y).sum(dim=-1, keepdim=True)  # (M, 1)
    dist_sq = xx + yy.t() - 2.0 * x @ y.t()  # (N, M)
    dist_sq = dist_sq.clamp(min=0.0)
    return (dist_sq + 1e-8).sqrt()


def compute_V_single_temp(
    phi_x: torch.Tensor,
    phi_pos: torch.Tensor,
    phi_neg: torch.Tensor,
    tau: float,
) -> dict[str, torch.Tensor]:
    """Compute drifting field V for a single temperature.

    Uses double softmax (over positives/negatives, then over generated samples).

    Args:
        phi_x: (N, C) features of generated samples (also used as negatives if phi_neg is same)
        phi_pos: (M, C) features of real (positive) samples
        phi_neg: (K, C) features of negative samples
        tau: temperature

    Returns:
        dict with "V", "V_attract", "V_repel" — each (N, C)
    """
    N = phi_x.shape[0]

    # Pairwise distances
    dist_pos = pairwise_l2(phi_x, phi_pos)  # (N, M)
    dist_neg = pairwise_l2(phi_x, phi_neg)  # (N, K)

    # First softmax: over positives/negatives for each x
    # k(x, y+) = softmax(-||x - y+|| / tau) over y+
    w_pos = F.softmax(-dist_pos / tau, dim=1)  # (N, M)
    w_neg = F.softmax(-dist_neg / tau, dim=1)  # (N, K)

    # Second softmax: over x for each y+ / y-
    # This normalizes across the batch of generated samples
    w_pos = w_pos * F.softmax(-dist_pos / tau, dim=0)  # (N, M)
    w_neg = w_neg * F.softmax(-dist_neg / tau, dim=0)  # (N, K)

    # Re-normalize rows after combining
    w_pos = w_pos / (w_pos.sum(dim=1, keepdim=True) + 1e-8)
    w_neg = w_neg / (w_neg.sum(dim=1, keepdim=True) + 1e-8)

    # Attraction: weighted mean of (y+ - x)
    # V+ = sum_j w_pos[i,j] * (phi_pos[j] - phi_x[i])
    V_attract = w_pos @ phi_pos - (w_pos.sum(dim=1, keepdim=True)) * phi_x  # (N, C)

    # Repulsion: weighted mean of (y- - x)
    # V- = sum_j w_neg[i,j] * (phi_neg[j] - phi_x[i])
    V_repel = w_neg @ phi_neg - (w_neg.sum(dim=1, keepdim=True)) * phi_x  # (N, C)

    # Drifting field: attraction - repulsion
    V = V_attract - V_repel

    return {"V": V, "V_attract": V_attract, "V_repel": V_repel}


def compute_V(
    phi_x: torch.Tensor,
    phi_pos: torch.Tensor,
    phi_neg: torch.Tensor,
    temperatures: tuple[float, ...] = (0.02, 0.05, 0.2),
    self_mask: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute drifting field V aggregated over multiple temperatures.

    Args:
        phi_x: (N, C) features of generated samples
        phi_pos: (M, C) features of positive (real) samples
        phi_neg: (K, C) features of negative samples
        temperatures: tuple of temperature values
        self_mask: if True and phi_neg is phi_x, mask self-distances in repulsion

    Returns:
        dict with "V", "V_attract", "V_repel" — each (N, C)
    """
    V = torch.zeros_like(phi_x)
    V_attract = torch.zeros_like(phi_x)
    V_repel = torch.zeros_like(phi_x)

    for tau in temperatures:
        if self_mask and phi_neg is phi_x:
            result = _compute_V_self_masked(phi_x, phi_pos, tau)
        else:
            result = compute_V_single_temp(phi_x, phi_pos, phi_neg, tau)
        V = V + result["V"]
        V_attract = V_attract + result["V_attract"]
        V_repel = V_repel + result["V_repel"]

    return {"V": V, "V_attract": V_attract, "V_repel": V_repel}


def _compute_V_self_masked(
    phi_x: torch.Tensor,
    phi_pos: torch.Tensor,
    tau: float,
) -> dict[str, torch.Tensor]:
    """Compute V when negatives == generated samples, with self-masking."""
    N, C = phi_x.shape

    # Distances
    dist_pos = pairwise_l2(phi_x, phi_pos)  # (N, M)
    dist_neg = pairwise_l2(phi_x, phi_x)  # (N, N)

    # Mask self-distances with large value before softmax
    mask = torch.eye(N, device=phi_x.device, dtype=torch.bool)
    dist_neg_masked = dist_neg.masked_fill(mask, float("inf"))

    # First softmax
    w_pos = F.softmax(-dist_pos / tau, dim=1)  # (N, M)
    w_neg = F.softmax(-dist_neg_masked / tau, dim=1)  # (N, N)

    # Second softmax (over x dimension)
    w_pos2 = F.softmax(-dist_pos / tau, dim=0)  # (N, M)
    w_neg2 = F.softmax(-dist_neg_masked / tau, dim=0)  # (N, N)

    w_pos = w_pos * w_pos2
    w_neg = w_neg * w_neg2

    # Re-normalize
    w_pos = w_pos / (w_pos.sum(dim=1, keepdim=True) + 1e-8)
    w_neg = w_neg / (w_neg.sum(dim=1, keepdim=True) + 1e-8)

    # Attraction
    V_attract = w_pos @ phi_pos - phi_x

    # Repulsion
    V_repel = w_neg @ phi_x - phi_x

    return {"V": V_attract - V_repel, "V_attract": V_attract, "V_repel": V_repel}
