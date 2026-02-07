"""Tests for feature encoder and pooling: shape checks, input_ids == inputs_embeds equivalence."""

import torch
import pytest


@pytest.fixture(scope="module")
def encoder():
    from src.features.encoder import GPT2FeatureEncoder
    return GPT2FeatureEncoder("gpt2")


@pytest.fixture(scope="module")
def pooler():
    from src.features.pooling import TextFeaturePooler
    return TextFeaturePooler(hidden_dim=768, n_subsample_positions=32, window_sizes=(4, 16))


def test_encoder_output_layers(encoder):
    """Encoder should return features for layers 3, 6, 9, 12."""
    input_ids = torch.randint(0, 50257, (2, 64))
    features = encoder(input_ids=input_ids)
    assert set(features.keys()) == {3, 6, 9, 12}


def test_encoder_output_shapes(encoder):
    """All layer features should have shape (B, seq_len, 768)."""
    B, S = 2, 64
    input_ids = torch.randint(0, 50257, (B, S))
    features = encoder(input_ids=input_ids)
    for layer_idx, feat in features.items():
        assert feat.shape == (B, S, 768), f"Layer {layer_idx}: {feat.shape}"


def test_input_ids_vs_inputs_embeds_equivalence(encoder):
    """gpt2(input_ids=ids) must equal gpt2(inputs_embeds=gpt2.wte(ids))."""
    input_ids = torch.randint(0, 50257, (2, 32))
    embeds = encoder.embed_tokens(input_ids)

    feat_ids = encoder(input_ids=input_ids)
    feat_embeds = encoder(inputs_embeds=embeds)

    for layer_idx in encoder.EXTRACT_LAYERS:
        assert torch.allclose(feat_ids[layer_idx], feat_embeds[layer_idx], atol=1e-4), \
            f"Layer {layer_idx} mismatch. Max diff: {(feat_ids[layer_idx] - feat_embeds[layer_idx]).abs().max():.6f}"


def test_encoder_frozen(encoder):
    """All GPT-2 parameters should be frozen."""
    for p in encoder.gpt2.parameters():
        assert not p.requires_grad


def test_vocab_embeddings_shape(encoder):
    """Vocab embeddings should be (50257, 768)."""
    emb = encoder.get_vocab_embeddings()
    assert emb.shape == (50257, 768)


def test_pooler_output_keys(pooler):
    """Pooler should return per_position, global, window_4, window_16."""
    hidden = torch.randn(4, 256, 768)
    pooled = pooler(hidden)
    assert "per_position" in pooled
    assert "global" in pooled
    assert "window_4" in pooled
    assert "window_16" in pooled


def test_pooler_shapes(pooler):
    """Check output shapes of each pooling type."""
    B, S, C = 4, 256, 768
    hidden = torch.randn(B, S, C)
    pooled = pooler(hidden)

    # Per-position: B * n_subsample x C
    assert pooled["per_position"].shape == (B * 32, C)

    # Global: B x 2C
    assert pooled["global"].shape == (B, 2 * C)

    # Window-4: B * (256/4) x 2C
    assert pooled["window_4"].shape == (B * 64, 2 * C)

    # Window-16: B * (256/16) x 2C
    assert pooled["window_16"].shape == (B * 16, 2 * C)


def test_pooler_with_mask(pooler):
    """Pooling should handle attention masks correctly."""
    B, S, C = 2, 64, 768
    hidden = torch.randn(B, S, C)
    mask = torch.ones(B, S)
    mask[0, 32:] = 0  # First sample half-padded

    pooled = pooler(hidden, attention_mask=mask)
    assert pooled["global"].shape == (B, 2 * C)
    assert not torch.isnan(pooled["global"]).any()
