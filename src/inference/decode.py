"""Nearest-neighbor decoding from continuous embeddings to tokens."""

import torch
import torch.nn.functional as F


def nearest_neighbor_decode(
    embeddings: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Decode continuous embeddings to token IDs via nearest-neighbor lookup.

    Args:
        embeddings: (B, seq_len, hidden_dim) generated continuous embeddings
        vocab_embeddings: (vocab_size, hidden_dim) GPT-2's token embeddings
        temperature: sampling temperature. <=0 or 1.0 = argmax, >1.0 = softer

    Returns:
        (B, seq_len) token IDs
    """
    B, S, C = embeddings.shape

    # Normalize for cosine similarity (more robust than L2 in high dim)
    emb_norm = F.normalize(embeddings.float(), dim=-1)
    vocab_norm = F.normalize(vocab_embeddings.float(), dim=-1)

    # Cosine similarity: (B, S, vocab_size)
    # Process per-sample to save memory
    token_ids = []
    for b in range(B):
        sim = emb_norm[b] @ vocab_norm.t()  # (S, vocab_size)

        if temperature <= 0 or temperature == 1.0:
            ids = sim.argmax(dim=-1)  # (S,)
        else:
            probs = F.softmax(sim / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

        token_ids.append(ids)

    return torch.stack(token_ids, dim=0)


def decode_to_text(
    embeddings: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    tokenizer,
    temperature: float = 1.0,
) -> list[str]:
    """Decode continuous embeddings to text strings.

    Args:
        embeddings: (B, seq_len, hidden_dim) generated embeddings
        vocab_embeddings: (vocab_size, hidden_dim) GPT-2's token embeddings
        tokenizer: GPT-2 tokenizer
        temperature: decoding temperature

    Returns:
        List of decoded strings
    """
    token_ids = nearest_neighbor_decode(embeddings, vocab_embeddings, temperature)

    texts = []
    for ids in token_ids:
        # Stop at EOS token if present
        eos_id = tokenizer.eos_token_id
        eos_positions = (ids == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            ids = ids[:eos_positions[0]]
        text = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
        texts.append(text)

    return texts


@torch.no_grad()
def compute_vocab_distances(
    embeddings: torch.Tensor,
    vocab_embeddings: torch.Tensor,
) -> dict[str, float]:
    """Compute statistics about distances to nearest vocab embeddings.

    Useful for monitoring whether generator outputs are near the vocab manifold.

    Args:
        embeddings: (B, seq_len, hidden_dim)
        vocab_embeddings: (vocab_size, hidden_dim)

    Returns:
        Dict with distance statistics
    """
    B, S, C = embeddings.shape
    emb_flat = embeddings.reshape(-1, C).float()
    vocab = vocab_embeddings.float()

    # Subsample if large
    if emb_flat.shape[0] > 1024:
        idx = torch.randperm(emb_flat.shape[0])[:1024]
        emb_flat = emb_flat[idx]

    # L2 distance to nearest vocab embedding
    # (N, 1, C) - (1, V, C) -> too large; compute via inner product
    emb_norm_sq = (emb_flat ** 2).sum(dim=-1, keepdim=True)  # (N, 1)
    vocab_norm_sq = (vocab ** 2).sum(dim=-1, keepdim=True).t()  # (1, V)
    dists_sq = emb_norm_sq + vocab_norm_sq - 2 * emb_flat @ vocab.t()
    min_dists = dists_sq.clamp(min=0).sqrt().min(dim=-1).values

    return {
        "vocab_dist/mean": min_dists.mean().item(),
        "vocab_dist/std": min_dists.std().item(),
        "vocab_dist/max": min_dists.max().item(),
        "vocab_dist/median": min_dists.median().item(),
    }
