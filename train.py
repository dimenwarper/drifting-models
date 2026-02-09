"""Training loop for Drifting Models text generation."""

import os
import copy
import math
import random
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.model.generator import DriftingGenerator
from src.features.encoder import GPT2FeatureEncoder
from src.features.pooling import TextFeaturePooler, pool_features
from src.drifting.loss import DriftingLoss
from src.drifting.field import pairwise_l2
from src.data.dataset import TinyStoriesDataset
from src.data.queue import SampleQueue
from src.inference.decode import decode_to_text, compute_vocab_distances

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_lr(step: int, warmup: int, total: int, min_ratio: float = 0.01) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def generate_samples(
    generator: nn.Module,
    encoder: GPT2FeatureEncoder,
    n_samples: int,
    seq_len: int,
    hidden_dim: int,
    device: torch.device,
    temperature: float = 1.0,
    embed_proj: nn.Module | None = None,
    prefix_proj: nn.Module | None = None,
    prompt: str | None = None,
) -> list[str]:
    """Generate text samples for logging, optionally conditioned on a prompt."""
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    generator.eval()
    with torch.no_grad():
        prefix_len = 0
        prefix_embeds_gen = None
        prefix_text = ""

        if prompt is not None:
            tokens = tokenizer.encode(prompt)
            prefix_ids = torch.tensor([tokens] * n_samples, device=device)
            prefix_len = len(tokens)
            prefix_embeds = encoder.embed_tokens(prefix_ids)  # (n, P, 768)
            if prefix_proj is not None:
                prefix_embeds_gen = prefix_proj(prefix_embeds)
            else:
                prefix_embeds_gen = prefix_embeds
            prefix_text = prompt

        suffix_len = seq_len - prefix_len
        noise = torch.randn(n_samples, suffix_len, hidden_dim, device=device)
        gen_out = generator(noise, prefix_embeds=prefix_embeds_gen)
        suffix_out = gen_out[:, prefix_len:]  # suffix only

        if embed_proj is not None:
            suffix_out = embed_proj(suffix_out)

    vocab_embs = encoder.get_vocab_embeddings()
    texts = decode_to_text(suffix_out, vocab_embs, tokenizer, temperature=temperature)
    texts = [prefix_text + t for t in texts]
    generator.train()
    return texts


def train(config_path: str = "configs/default.yaml"):
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Feature encoder (frozen GPT-2)
    logger.info("Loading GPT-2 feature encoder...")
    encoder = GPT2FeatureEncoder(cfg["encoder"]["model_name"]).to(device)
    vocab_embeddings = encoder.get_vocab_embeddings()
    vocab_mean = vocab_embeddings.mean(dim=0)

    # Generator
    logger.info("Initializing generator...")
    generator = DriftingGenerator(
        hidden_dim=cfg["generator"]["hidden_dim"],
        n_layers=cfg["generator"]["n_layers"],
        n_heads=cfg["generator"]["n_heads"],
        seq_len=cfg["data"]["seq_len"],
        n_style_tokens=cfg["generator"]["n_style_tokens"],
        codebook_size=cfg["generator"]["codebook_size"],
        vocab_embed_mean=vocab_mean,
    ).to(device)

    # Projections between generator hidden_dim and GPT-2 hidden_dim (if different)
    gen_dim = cfg["generator"]["hidden_dim"]
    enc_dim = encoder.hidden_dim
    embed_proj = None   # gen_dim -> enc_dim (for feeding generator output to GPT-2)
    prefix_proj = None  # enc_dim -> gen_dim (for feeding GPT-2 embeddings to generator)
    if gen_dim != enc_dim:
        embed_proj = nn.Linear(gen_dim, enc_dim, bias=True).to(device)
        prefix_proj = nn.Linear(enc_dim, gen_dim, bias=True).to(device)
        logger.info(f"Added embed projection: {gen_dim} <-> {enc_dim}")

    n_params = sum(p.numel() for p in generator.parameters())
    if embed_proj is not None:
        n_params += sum(p.numel() for p in embed_proj.parameters())
    if prefix_proj is not None:
        n_params += sum(p.numel() for p in prefix_proj.parameters())
    logger.info(f"Generator parameters: {n_params:,}")

    # EMA model
    ema_generator = copy.deepcopy(generator)
    ema_generator.eval()
    for p in ema_generator.parameters():
        p.requires_grad_(False)

    # Pooler
    pooler = TextFeaturePooler(
        hidden_dim=enc_dim,
        n_subsample_positions=cfg["pooling"]["n_subsample_positions"],
        window_sizes=tuple(cfg["pooling"]["window_sizes"]),
    )

    # Loss
    drifting_loss = DriftingLoss(
        temperatures=tuple(cfg["drifting"]["temperatures"]),
        momentum=cfg["drifting"]["normalization_momentum"],
        max_V_norm=cfg["drifting"].get("max_V_norm"),
        normalize_drift=cfg["drifting"].get("normalize_drift", True),
        loss_fn=cfg["drifting"].get("loss_fn", "mse"),
        huber_delta=cfg["drifting"].get("huber_delta", 1.0),
    )

    # Dataset and DataLoader
    logger.info("Loading dataset...")
    dataset = TinyStoriesDataset(
        split=cfg["data"]["split"],
        seq_len=cfg["data"]["seq_len"],
        max_samples=cfg["data"].get("max_samples"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["n_pos"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # Sample queue for extra positives
    queue = SampleQueue(
        feature_dim=enc_dim,
        max_size=cfg["training"]["queue_size"],
    )

    # Optimizer
    opt_cfg = cfg["training"]["optimizer"]
    opt_params = list(generator.parameters())
    if embed_proj is not None:
        opt_params += list(embed_proj.parameters())
    if prefix_proj is not None:
        opt_params += list(prefix_proj.parameters())
    optimizer = torch.optim.AdamW(
        opt_params,
        lr=opt_cfg["lr"],
        betas=tuple(opt_cfg["betas"]),
        weight_decay=opt_cfg["weight_decay"],
    )

    # Wandb
    wandb_enabled = cfg["wandb"].get("enabled", False)
    if wandb_enabled:
        import wandb
        wandb.init(project=cfg["wandb"]["project"], config=cfg)

    # Training
    tc = cfg["training"]
    sc = tc["scheduler"]
    use_bf16 = tc["precision"] == "bf16" and device.type == "cuda"

    # Prompt conditioning config
    use_prefix = cfg["data"].get("prompt_conditioning", False)
    min_prefix = cfg["data"].get("min_prefix_len", 16)
    max_prefix = cfg["data"].get("max_prefix_len", 128)
    if use_prefix:
        logger.info(f"Prompt conditioning enabled: prefix_len in [{min_prefix}, {max_prefix}]")

    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_iter = iter(dataloader)
    logger.info("Starting training...")

    for step in range(1, tc["max_steps"] + 1):
        # Get batch of real data
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # LR schedule
        lr_scale = cosine_lr(step, sc["warmup_steps"], tc["max_steps"], sc["min_lr_ratio"])
        for pg in optimizer.param_groups:
            pg["lr"] = opt_cfg["lr"] * lr_scale

        # Generate samples
        B = tc["batch_size"]
        S = cfg["data"]["seq_len"]
        C = cfg["generator"]["hidden_dim"]

        # Prompt conditioning: random prefix split
        prefix_len = 0
        prefix_embeds_gen = None
        if use_prefix:
            prefix_len = random.randint(min_prefix, min(max_prefix, S - 1))
            with torch.no_grad():
                prefix_embeds = encoder.embed_tokens(input_ids[:B, :prefix_len])  # (B, P, 768)
            if prefix_proj is not None:
                prefix_embeds_gen = prefix_proj(prefix_embeds.detach())
            else:
                prefix_embeds_gen = prefix_embeds.detach()

        suffix_len = S - prefix_len
        noise = torch.randn(B, suffix_len, C, device=device)

        gen_out = generator(noise, prefix_embeds=prefix_embeds_gen)  # (B, P+suffix, C)

        # Slice suffix only for feature extraction
        suffix_out = gen_out[:, prefix_len:]  # (B, suffix_len, C)

        # Project to encoder dim if needed
        enc_input = embed_proj(suffix_out) if embed_proj is not None else suffix_out

        # Extract features — no_grad for real data, grad for generated (to backprop to generator)
        suffix_ids = input_ids[:, prefix_len:]
        suffix_mask = attention_mask[:, prefix_len:]
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=use_bf16, dtype=torch.bfloat16):
                real_features = encoder(input_ids=suffix_ids, attention_mask=suffix_mask)
        with torch.amp.autocast(device.type, enabled=use_bf16, dtype=torch.bfloat16):
            gen_features = encoder(inputs_embeds=enc_input, attention_mask=None)

        # Pool features (suffix only — already sliced)
        real_pooled = pool_features(real_features, pooler, suffix_mask)
        gen_pooled = pool_features(gen_features, pooler, None)

        # Compute drifting loss (in fp32)
        loss, metrics = drifting_loss.compute(
            gen_features=gen_pooled,
            pos_features=real_pooled,
            neg_features=None,  # use gen_features as negatives
        )

        # Diversity regularizer — penalize collapse in generator output space
        diversity_weight = tc.get("diversity_weight", 0.0)
        if diversity_weight > 0:
            gen_flat = suffix_out.mean(dim=1)  # (B, C) — per-sample mean embedding
            pw_dist = pairwise_l2(gen_flat, gen_flat)  # (B, B)
            eye_mask = ~torch.eye(B, device=device, dtype=torch.bool)
            avg_dist = pw_dist[eye_mask].mean()
            diversity_loss = 1.0 / (avg_dist + 1e-4)
            loss = loss + diversity_weight * diversity_loss
            metrics["gen_diversity"] = avg_dist.item()
            metrics["diversity_loss"] = diversity_loss.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (all trainable params, including projections)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            opt_params, tc["gradient_clip"]
        )
        metrics["grad_norm"] = grad_norm.item()

        optimizer.step()

        # EMA update
        update_ema(ema_generator, generator, tc["ema_decay"])

        # Push real embeddings to queue
        with torch.no_grad():
            real_embeds = encoder.embed_tokens(input_ids)
            queue.push(real_embeds.mean(dim=1))  # global mean as summary

        # Logging
        if step % tc["log_interval"] == 0:
            lr = optimizer.param_groups[0]["lr"]

            # Aggregate diagnostics across scales
            v_norms = [v for k, v in metrics.items() if k.endswith("/V_norm")]
            v_raw_norms = [v for k, v in metrics.items() if k.endswith("/V_raw_norm")]
            feat_scales = [v for k, v in metrics.items() if k.endswith("/feat_scale")]
            scale_losses = [v for k, v in metrics.items() if k.endswith("/loss") and k != "total_loss"]
            attract_norms = [v for k, v in metrics.items() if k.endswith("/attraction_norm")]
            repel_norms = [v for k, v in metrics.items() if k.endswith("/repulsion_norm")]
            drift_lambdas = [v for k, v in metrics.items() if k.endswith("/drift_lambda")]

            diag_parts = [
                f"loss={metrics['total_loss']:.4f}",
                f"grad_norm={metrics['grad_norm']:.3f}",
                f"lr={lr:.2e}",
            ]
            if v_norms:
                metrics["mean_V_norm"] = sum(v_norms) / len(v_norms)
                diag_parts.append(f"V={metrics['mean_V_norm']:.1f}")
            if v_raw_norms:
                metrics["mean_V_raw_norm"] = sum(v_raw_norms) / len(v_raw_norms)
                diag_parts.append(f"V_raw={metrics['mean_V_raw_norm']:.1f}")
            if drift_lambdas:
                metrics["mean_drift_lambda"] = sum(drift_lambdas) / len(drift_lambdas)
                diag_parts.append(f"λ={metrics['mean_drift_lambda']:.2f}")
            if feat_scales:
                metrics["mean_feat_scale"] = sum(feat_scales) / len(feat_scales)
            if scale_losses:
                metrics["min_scale_loss"] = min(scale_losses)
                metrics["max_scale_loss"] = max(scale_losses)
            if attract_norms:
                metrics["mean_attraction_norm"] = sum(attract_norms) / len(attract_norms)
                diag_parts.append(f"a={metrics['mean_attraction_norm']:.1f}")
            if repel_norms:
                metrics["mean_repulsion_norm"] = sum(repel_norms) / len(repel_norms)
                diag_parts.append(f"r={metrics['mean_repulsion_norm']:.1f}")
            if "gen_diversity" in metrics:
                diag_parts.append(f"div={metrics['gen_diversity']:.1f}")

            logger.info(f"Step {step} | {' | '.join(diag_parts)}")

            if wandb_enabled:
                import wandb
                wandb.log({"step": step, "lr": lr, **metrics})

        # Generate samples periodically
        if step % tc["generate_interval"] == 0:
            gen_prompt = cfg["inference"].get("prompt")
            texts = generate_samples(
                ema_generator, encoder,
                n_samples=tc["generate_n_samples"],
                seq_len=S, hidden_dim=C, device=device,
                temperature=cfg["inference"]["temperature"],
                embed_proj=embed_proj,
                prefix_proj=prefix_proj,
                prompt=gen_prompt,
            )
            logger.info("=== Generated Samples ===")
            for i, t in enumerate(texts):
                logger.info(f"  [{i}] {t[:200]}")

            # Vocab distance stats
            with torch.no_grad():
                test_noise = torch.randn(4, S, C, device=device)
                test_emb = ema_generator(test_noise)
                if embed_proj is not None:
                    test_emb = embed_proj(test_emb)
                dist_stats = compute_vocab_distances(test_emb, vocab_embeddings.to(device))
            logger.info(f"  Vocab distances: {dist_stats}")
            if wandb_enabled:
                import wandb
                wandb.log({**dist_stats, "step": step})

        # Save checkpoint
        if step % tc["save_interval"] == 0:
            ckpt = {
                "step": step,
                "generator": generator.state_dict(),
                "ema_generator": ema_generator.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }
            if embed_proj is not None:
                ckpt["embed_proj"] = embed_proj.state_dict()
            if prefix_proj is not None:
                ckpt["prefix_proj"] = prefix_proj.state_dict()
            path = checkpoint_dir / f"step_{step:07d}.pt"
            torch.save(ckpt, path)
            logger.info(f"Saved checkpoint: {path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
