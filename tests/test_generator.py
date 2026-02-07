"""Tests for generator: output shapes, gradient flow, param count."""

import torch
import pytest

from src.model.generator import DriftingGenerator


@pytest.fixture
def generator():
    torch.manual_seed(42)
    return DriftingGenerator(
        hidden_dim=768,
        n_layers=12,
        n_heads=12,
        seq_len=256,
    )


@pytest.fixture
def small_generator():
    torch.manual_seed(42)
    return DriftingGenerator(
        hidden_dim=128,
        n_layers=4,
        n_heads=4,
        seq_len=32,
    )


def test_output_shape(small_generator):
    """Output should match input shape."""
    B, S, C = 2, 32, 128
    noise = torch.randn(B, S, C)
    out = small_generator(noise)
    assert out.shape == (B, S, C)


def test_gradient_flow(small_generator):
    """Gradients should flow through the generator."""
    B, S, C = 2, 32, 128
    noise = torch.randn(B, S, C)
    out = small_generator(noise)
    loss = out.pow(2).mean()
    loss.backward()

    has_grad = False
    for p in small_generator.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients flow through generator"


def test_param_count(generator):
    """Base generator should have ~130M parameters."""
    n_params = sum(p.numel() for p in generator.parameters())
    # Allow range 100M - 160M
    assert 100_000_000 < n_params < 160_000_000, f"Param count: {n_params:,}"


def test_no_nan_output(small_generator):
    """Output should not contain NaN."""
    noise = torch.randn(4, 32, 128)
    out = small_generator(noise)
    assert not torch.isnan(out).any()


def test_different_noise_different_output(small_generator):
    """Different noise inputs should produce different outputs."""
    noise1 = torch.randn(2, 32, 128)
    noise2 = torch.randn(2, 32, 128)

    torch.manual_seed(0)
    out1 = small_generator(noise1)
    torch.manual_seed(0)
    out2 = small_generator(noise2)

    assert not torch.allclose(out1, out2, atol=1e-3)


def test_vocab_embed_mean_init():
    """Output bias should be initialized to vocab embedding mean when provided."""
    mean = torch.ones(128) * 0.42
    gen = DriftingGenerator(
        hidden_dim=128, n_layers=2, n_heads=2, seq_len=16,
        vocab_embed_mean=mean,
    )
    assert torch.allclose(gen.output_proj.bias.data, mean)
