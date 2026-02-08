"""Batch generation script for trained drifting models."""

import argparse
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from transformers import GPT2Tokenizer

from src.model.generator import DriftingGenerator
from src.features.encoder import GPT2FeatureEncoder
from src.inference.decode import decode_to_text, compute_vocab_distances


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained drifting model")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--n_samples", type=int, default=16, help="Number of samples")
    parser.add_argument("--temperature", type=float, default=1.0, help="Decoding temperature")
    parser.add_argument("--output", default=None, help="Output file (default: stdout)")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load encoder for vocab embeddings
    encoder = GPT2FeatureEncoder(cfg["encoder"]["model_name"]).to(device)
    vocab_embeddings = encoder.get_vocab_embeddings().to(device)
    vocab_mean = vocab_embeddings.mean(dim=0)

    gen_dim = cfg["generator"]["hidden_dim"]
    enc_dim = encoder.hidden_dim

    # Build generator
    generator = DriftingGenerator(
        hidden_dim=gen_dim,
        n_layers=cfg["generator"]["n_layers"],
        n_heads=cfg["generator"]["n_heads"],
        seq_len=cfg["data"]["seq_len"],
        n_style_tokens=cfg["generator"]["n_style_tokens"],
        codebook_size=cfg["generator"]["codebook_size"],
        vocab_embed_mean=vocab_mean,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    key = "ema_generator" if args.use_ema and "ema_generator" in ckpt else "generator"
    generator.load_state_dict(ckpt[key])
    generator.eval()

    # Embed projection (when generator hidden_dim != encoder hidden_dim)
    embed_proj = None
    if gen_dim != enc_dim:
        embed_proj = nn.Linear(gen_dim, enc_dim, bias=True).to(device)
        if "embed_proj" in ckpt:
            embed_proj.load_state_dict(ckpt["embed_proj"])
        embed_proj.eval()

    step = ckpt.get("step", "?")
    print(f"Loaded checkpoint from step {step} (using {key} weights)")

    # Generate
    S = cfg["data"]["seq_len"]
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    noise = torch.randn(args.n_samples, S, gen_dim, device=device)
    with torch.no_grad():
        embeddings = generator(noise)
        if embed_proj is not None:
            embeddings = embed_proj(embeddings)

    # Vocab distance stats
    dist_stats = compute_vocab_distances(embeddings, vocab_embeddings)
    print(f"Vocab distance stats: {dist_stats}")

    # Decode
    texts = decode_to_text(embeddings, vocab_embeddings, tokenizer, temperature=args.temperature)

    # Output
    output_lines = []
    for i, text in enumerate(texts):
        line = f"[{i:3d}] {text}"
        output_lines.append(line)
        print(line)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write("\n".join(output_lines) + "\n")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
