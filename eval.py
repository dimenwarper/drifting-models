"""Evaluation harness: generate from both models and score with multiple metrics."""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.model.generator import DriftingGenerator
from src.features.encoder import GPT2FeatureEncoder
from src.inference.decode import decode_to_text
from src.data.dataset import TinyStoriesDataset
from src.eval.metrics import compute_perplexity, compute_distinct_n, compute_self_bleu, compute_mauve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def generate_drifting_samples(
    checkpoint_path: str,
    cfg: dict,
    n_samples: int,
    device: torch.device,
    prefix: str | None = None,
) -> list[str]:
    """Generate text samples from a trained drifting model."""
    encoder = GPT2FeatureEncoder(cfg["encoder"]["model_name"]).to(device)
    vocab_embeddings = encoder.get_vocab_embeddings().to(device)
    vocab_mean = vocab_embeddings.mean(dim=0)

    gen_dim = cfg["generator"]["hidden_dim"]
    enc_dim = encoder.hidden_dim

    generator = DriftingGenerator(
        hidden_dim=gen_dim,
        n_layers=cfg["generator"]["n_layers"],
        n_heads=cfg["generator"]["n_heads"],
        seq_len=cfg["data"]["seq_len"],
        n_style_tokens=cfg["generator"]["n_style_tokens"],
        codebook_size=cfg["generator"]["codebook_size"],
        vocab_embed_mean=vocab_mean,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    key = "ema_generator" if "ema_generator" in ckpt else "generator"
    generator.load_state_dict(ckpt[key])
    generator.eval()

    embed_proj = None
    if gen_dim != enc_dim:
        embed_proj = nn.Linear(gen_dim, enc_dim, bias=True).to(device)
        if "embed_proj" in ckpt:
            embed_proj.load_state_dict(ckpt["embed_proj"])
        embed_proj.eval()

    prefix_proj = None
    if gen_dim != enc_dim:
        prefix_proj = nn.Linear(enc_dim, gen_dim, bias=True).to(device)
        if "prefix_proj" in ckpt:
            prefix_proj.load_state_dict(ckpt["prefix_proj"])
        prefix_proj.eval()

    S = cfg["data"]["seq_len"]
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    prefix_text = ""
    prefix_embeds_gen = None
    prefix_len = 0
    if prefix:
        prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
        prefix_len = prefix_ids.shape[1]
        if prefix_len >= S:
            prefix_ids = prefix_ids[:, : S - 1]
            prefix_len = prefix_ids.shape[1]
        prefix_text = tokenizer.decode(prefix_ids[0])
        with torch.no_grad():
            prefix_embeds = encoder.embed_tokens(prefix_ids)
            prefix_embeds = prefix_embeds.expand(n_samples, -1, -1)
            if prefix_proj is not None:
                prefix_embeds_gen = prefix_proj(prefix_embeds)
            else:
                prefix_embeds_gen = prefix_embeds

    suffix_len = S - prefix_len
    noise = torch.randn(n_samples, suffix_len, gen_dim, device=device)
    with torch.no_grad():
        gen_out = generator(noise, prefix_embeds=prefix_embeds_gen)
        suffix_out = gen_out[:, prefix_len:]
        if embed_proj is not None:
            suffix_out = embed_proj(suffix_out)

    suffix_texts = decode_to_text(suffix_out, vocab_embeddings, tokenizer, temperature=1.0)

    texts = []
    for suffix in suffix_texts:
        texts.append(f"{prefix_text}{suffix}" if prefix_text else suffix)
    return texts


def generate_gpt2_samples(
    checkpoint_path: str,
    n_samples: int,
    device: torch.device,
    prefix: str | None = None,
    max_length: int = 256,
) -> list[str]:
    """Generate text samples from a (finetuned) GPT-2 model."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if checkpoint_path == "gpt2":
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    else:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        model.load_state_dict(ckpt["model"])
    model.eval()

    if prefix:
        input_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
        input_ids = input_ids.expand(n_samples, -1)
    else:
        input_ids = torch.full((n_samples, 1), tokenizer.eos_token_id, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    texts = []
    for ids in outputs:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        texts.append(text)
    return texts


def load_reference_texts(n_samples: int, seq_len: int = 256) -> list[str]:
    """Load held-out real TinyStories texts."""
    dataset = TinyStoriesDataset(split="validation", seq_len=seq_len, max_samples=n_samples)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    texts = []
    for i in range(len(dataset)):
        item = dataset[i]
        ids = item["input_ids"]
        mask = item["attention_mask"]
        # Decode only non-pad tokens
        length = mask.sum().item()
        text = tokenizer.decode(ids[:length].tolist(), skip_special_tokens=True)
        texts.append(text)
    return texts


def evaluate_texts(texts: list[str], label: str, device: torch.device) -> dict:
    """Compute all metrics for a set of texts."""
    logger.info(f"Evaluating {label} ({len(texts)} samples)...")
    metrics = {}

    # Filter out empty texts
    texts = [t for t in texts if t.strip()]
    if not texts:
        logger.warning(f"  No non-empty texts for {label}")
        return metrics

    # Perplexity (gpt2-medium judge)
    logger.info(f"  Computing perplexity...")
    metrics["perplexity"] = compute_perplexity(texts, model_name="gpt2-medium", device=device)

    # Distinct-n
    for n in [1, 2, 3]:
        metrics[f"distinct_{n}"] = compute_distinct_n(texts, n)
    logger.info(f"  Distinct-1={metrics['distinct_1']:.4f}, -2={metrics['distinct_2']:.4f}, -3={metrics['distinct_3']:.4f}")

    # Self-BLEU
    logger.info(f"  Computing self-BLEU...")
    metrics["self_bleu"] = compute_self_bleu(texts)

    logger.info(f"  {label}: ppl={metrics['perplexity']:.2f}, self-BLEU={metrics['self_bleu']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate drifting model vs finetuned GPT-2")
    parser.add_argument("--drifting_checkpoint", required=True, help="Path to drifting model checkpoint")
    parser.add_argument("--gpt2_checkpoint", default="gpt2", help="Path to finetuned GPT-2 checkpoint, or 'gpt2' for pretrained")
    parser.add_argument("--config", required=True, help="Drifting model config (for architecture)")
    parser.add_argument("--n_samples", type=int, default=256, help="Number of samples to generate")
    parser.add_argument("--prefix", default=None, help="Optional prefix for conditional generation")
    parser.add_argument("--output", default="eval_results.json", help="Output JSON file")
    parser.add_argument("--skip_mauve", action="store_true", help="Skip MAUVE computation (requires mauve-text)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seq_len = cfg["data"]["seq_len"]

    # Generate from drifting model
    logger.info("Generating from drifting model...")
    drifting_texts = generate_drifting_samples(
        args.drifting_checkpoint, cfg, args.n_samples, device, prefix=args.prefix,
    )
    logger.info(f"  Sample: {drifting_texts[0][:200]}")

    # Generate from GPT-2
    logger.info("Generating from GPT-2...")
    gpt2_texts = generate_gpt2_samples(
        args.gpt2_checkpoint, args.n_samples, device, prefix=args.prefix, max_length=seq_len,
    )
    logger.info(f"  Sample: {gpt2_texts[0][:200]}")

    # Load reference texts
    logger.info("Loading reference texts...")
    reference_texts = load_reference_texts(args.n_samples, seq_len=seq_len)
    logger.info(f"  Sample: {reference_texts[0][:200]}")

    # Evaluate all
    results = {}
    results["drifting"] = evaluate_texts(drifting_texts, "drifting", device)
    results["gpt2"] = evaluate_texts(gpt2_texts, "gpt2", device)
    results["reference"] = evaluate_texts(reference_texts, "reference", device)

    # MAUVE (distribution-level, generated vs reference)
    if not args.skip_mauve:
        logger.info("Computing MAUVE scores...")
        try:
            results["drifting"]["mauve"] = compute_mauve(drifting_texts, reference_texts)
            results["gpt2"]["mauve"] = compute_mauve(gpt2_texts, reference_texts)
            logger.info(f"  MAUVE: drifting={results['drifting']['mauve']:.4f}, gpt2={results['gpt2']['mauve']:.4f}")
        except Exception as e:
            logger.warning(f"  MAUVE computation failed: {e}")
    else:
        logger.info("Skipping MAUVE (--skip_mauve)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Metric':<20} {'Drifting':>15} {'GPT-2':>15} {'Reference':>15}")
    print("-" * 70)
    all_metrics = sorted(set().union(*(r.keys() for r in results.values())))
    for metric in all_metrics:
        vals = []
        for model_name in ["drifting", "gpt2", "reference"]:
            v = results[model_name].get(metric)
            vals.append(f"{v:.4f}" if v is not None else "n/a")
        print(f"{metric:<20} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")
    print("=" * 70)


if __name__ == "__main__":
    main()
