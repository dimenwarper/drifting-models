"""Tests for drifting field: anti-symmetry, zero at equilibrium, gradient flow."""

import torch
import pytest
from src.drifting.field import compute_V, compute_V_single_temp, pairwise_l2


@pytest.fixture
def seed():
    torch.manual_seed(42)


def test_pairwise_l2_shape(seed):
    x = torch.randn(10, 32)
    y = torch.randn(20, 32)
    d = pairwise_l2(x, y)
    assert d.shape == (10, 20)


def test_pairwise_l2_nonneg(seed):
    x = torch.randn(10, 32)
    y = torch.randn(20, 32)
    d = pairwise_l2(x, y)
    assert (d >= 0).all()


def test_pairwise_l2_self_zero(seed):
    x = torch.randn(10, 32)
    d = pairwise_l2(x, x)
    assert torch.allclose(d.diag(), torch.zeros(10), atol=1e-2)


def test_anti_symmetry(seed):
    """V_{p,q}(x) should be approximately -V_{q,p}(x).

    When we swap positives and negatives, the field should flip sign.
    """
    N, C = 32, 16
    phi_x = torch.randn(N, C)
    phi_a = torch.randn(N, C)
    phi_b = torch.randn(N, C)

    # V with a as positives, b as negatives
    V_ab = compute_V_single_temp(phi_x, phi_a, phi_b, tau=0.1)["V"]

    # V with b as positives, a as negatives
    V_ba = compute_V_single_temp(phi_x, phi_b, phi_a, tau=0.1)["V"]

    # Should be anti-symmetric: V_ab ~ -V_ba
    assert torch.allclose(V_ab, -V_ba, atol=1e-5), \
        f"Anti-symmetry violated. Max diff: {(V_ab + V_ba).abs().max():.6f}"


def test_zero_at_equilibrium(seed):
    """When positives == negatives (same distribution), V should be ~0."""
    N, C = 64, 16
    phi_x = torch.randn(N, C)
    phi_same = torch.randn(N, C)  # same distribution for both pos and neg

    V = compute_V_single_temp(phi_x, phi_same, phi_same, tau=0.1)["V"]

    assert torch.allclose(V, torch.zeros_like(V), atol=1e-5), \
        f"V not zero at equilibrium. Max: {V.abs().max():.6f}"


def test_zero_at_equilibrium_exact(seed):
    """When pos and neg are literally the same tensor, V must be exactly 0."""
    N, C = 32, 16
    phi_x = torch.randn(N, C)
    phi_same = torch.randn(N, C)

    V = compute_V_single_temp(phi_x, phi_same, phi_same, tau=0.05)["V"]

    assert V.abs().max() < 1e-6, \
        f"V not zero when pos==neg. Max: {V.abs().max():.6f}"


def test_multi_temperature(seed):
    """Multi-temperature V should be average of individual temperature Vs."""
    N, C = 32, 16
    phi_x = torch.randn(N, C)
    phi_pos = torch.randn(N, C)
    phi_neg = torch.randn(N, C)

    temps = (0.02, 0.05, 0.2)
    V_multi = compute_V(phi_x, phi_pos, phi_neg, temperatures=temps, self_mask=False)["V"]

    V_sum = torch.zeros_like(phi_x)
    for tau in temps:
        V_sum += compute_V_single_temp(phi_x, phi_pos, phi_neg, tau)["V"]
    V_avg = V_sum / len(temps)

    assert torch.allclose(V_multi, V_avg, atol=1e-5)


def test_gradient_flow(seed):
    """V should be differentiable w.r.t. phi_x."""
    N, C = 16, 8
    phi_x = torch.randn(N, C, requires_grad=True)
    phi_pos = torch.randn(N, C)
    phi_neg = torch.randn(N, C)

    V = compute_V(phi_x, phi_pos, phi_neg, temperatures=(0.1,), self_mask=False)["V"]
    loss = V.pow(2).sum()
    loss.backward()

    assert phi_x.grad is not None
    assert not torch.isnan(phi_x.grad).any()
    assert phi_x.grad.abs().sum() > 0, "Gradients are all zero"


def test_self_mask(seed):
    """Self-masking should prevent self-interactions."""
    N, C = 16, 8
    phi_x = torch.randn(N, C)
    phi_pos = torch.randn(N, C)

    # With self_mask=True and phi_neg is phi_x
    result = compute_V(phi_x, phi_pos, phi_x, temperatures=(0.1,), self_mask=True)
    V = result["V"]

    assert V.shape == (N, C)
    assert not torch.isnan(V).any()
    assert result["V_attract"].shape == (N, C)
    assert result["V_repel"].shape == (N, C)


def test_output_shape(seed):
    """Output shape should match input shape."""
    N, C = 32, 64
    phi_x = torch.randn(N, C)
    phi_pos = torch.randn(48, C)
    phi_neg = torch.randn(24, C)

    result = compute_V(phi_x, phi_pos, phi_neg, self_mask=False)
    assert result["V"].shape == (N, C)
    assert result["V_attract"].shape == (N, C)
    assert result["V_repel"].shape == (N, C)
