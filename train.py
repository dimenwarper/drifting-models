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
from src.drifting.smooth_proj import SmoothProjectionBank
from src.drifting.spectral import spectral_regularization
from src.data.dataset import TinyStoriesDataset
from src.data.queue import SampleQueue
from src.drifting.queue import FeatureQueue
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

    # (smooth_proj params counted after initialization, since it's lazily built)

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

    # Smooth projection (learned feature space regularization)
    smooth_cfg = cfg.get("smooth_proj", {})
    smooth_proj = None
    if smooth_cfg.get("enabled", False):
        smooth_proj = SmoothProjectionBank(
            proj_dim=smooth_cfg.get("proj_dim", 256),
            hidden_mult=smooth_cfg.get("hidden_mult", 2),
        ).to(device)
        logger.info(f"Smooth projection enabled: proj_dim={smooth_cfg.get('proj_dim', 256)}")

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

    # MoCo feature queue for generated negatives
    moco_queue = None
    if cfg["training"].get("use_moco_queue", False):
        moco_queue = FeatureQueue(max_size=cfg["training"].get("moco_queue_size", 8192))
        logger.info(f"MoCo queue enabled: size={cfg['training'].get('moco_queue_size', 8192)}, "
                     f"n_neg={cfg['training'].get('moco_n_neg', 128)}, "
                     f"min={cfg['training'].get('moco_min_queue', 256)}")

    # Optimizer
    opt_cfg = cfg["training"]["optimizer"]
    opt_params = list(generator.parameters())
    if embed_proj is not None:
        opt_params += list(embed_proj.parameters())
    if prefix_proj is not None:
        opt_params += list(prefix_proj.parameters())
    if smooth_proj is not None:
        opt_params += list(smooth_proj.parameters())
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
    accum_steps = tc.get("gradient_accumulation", 1)
    logger.info(f"Starting training... (gradient_accumulation={accum_steps})")

    B = tc["batch_size"]
    ema_loss = None  # EMA-smoothed loss for tracking real trend
    ema_V_raw = None
    div_ema_ref = None  # EMA reference diversity for log-barrier
    S = cfg["data"]["seq_len"]
    C = cfg["generator"]["hidden_dim"]

    # Pre-move vocab embeddings to device for vocab anchoring / LM loss / Gumbel snap
    need_vocab = (tc.get("vocab_anchor_weight", 0.0) > 0
                  or tc.get("lm_weight", 0.0) > 0
                  or tc.get("gumbel_tau", 0.0) > 0)
    vocab_embs_device = vocab_embeddings.to(device) if need_vocab else None
    real_ce_ref = None  # Reference CE from real text (for perplexity-matching LM loss)

    for step in range(1, tc["max_steps"] + 1):
        # LR schedule
        lr_scale = cosine_lr(step, sc["warmup_steps"], tc["max_steps"], sc["min_lr_ratio"])
        for pg in optimizer.param_groups:
            pg["lr"] = opt_cfg["lr"] * lr_scale

        optimizer.zero_grad()
        accum_metrics = {}

        for micro in range(accum_steps):
            # Get batch of real data
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Prompt conditioning: random prefix split
            prefix_len = 0
            prefix_embeds_gen = None
            if use_prefix:
                prefix_len = random.randint(min_prefix, min(max_prefix, S - 1))
                with torch.no_grad():
                    prefix_embeds = encoder.embed_tokens(input_ids[:B, :prefix_len])
                if prefix_proj is not None:
                    prefix_embeds_gen = prefix_proj(prefix_embeds.detach())
                else:
                    prefix_embeds_gen = prefix_embeds.detach()

            suffix_len = S - prefix_len
            noise = torch.randn(B, suffix_len, C, device=device)

            gen_out = generator(noise, prefix_embeds=prefix_embeds_gen)
            suffix_out = gen_out[:, prefix_len:]

            # Project to encoder dim if needed
            enc_input = embed_proj(suffix_out) if embed_proj is not None else suffix_out

            # Top-k Gumbel-Softmax snap: force generator to commit to actual vocab tokens
            # Uses only k nearest tokens per position (not full 50k) for healthy gradients
            gumbel_tau = tc.get("gumbel_tau", 0.0)
            if gumbel_tau > 0:
                gumbel_k = tc.get("gumbel_topk", 64)
                flat = enc_input.reshape(-1, enc_input.shape[-1])  # (B*S, D)
                # Find top-k nearest vocab tokens by dot product similarity
                sims = flat @ vocab_embs_device.T  # (B*S, V)
                topk_sims, topk_ids = sims.topk(gumbel_k, dim=-1)  # (B*S, k)
                # Gumbel-Softmax over k tokens — flat enough for real gradients
                weights = nn.functional.gumbel_softmax(topk_sims, tau=gumbel_tau, hard=False)  # (B*S, k)
                # Weighted sum of top-k token embeddings
                topk_embs = vocab_embs_device[topk_ids]  # (B*S, k, D)
                enc_input = (weights.unsqueeze(-1) * topk_embs).sum(dim=1)  # (B*S, D)
                enc_input = enc_input.reshape(B, -1, topk_embs.shape[-1])

            # --- Real features (computed once, reused across refinement steps) ---
            suffix_ids = input_ids[:, prefix_len:]
            suffix_mask = attention_mask[:, prefix_len:]
            with torch.no_grad():
                with torch.amp.autocast(device.type, enabled=use_bf16, dtype=torch.bfloat16):
                    real_features = encoder(input_ids=suffix_ids, attention_mask=suffix_mask)
            real_pooled = pool_features(real_features, pooler, suffix_mask)

            feat_noise_std = 0.0
            if smooth_proj is not None:
                feat_noise_std = smooth_cfg.get("feature_noise_std", 0.1)
                if feat_noise_std > 0:
                    real_pooled = [
                        (name, feat.detach() + torch.randn_like(feat) * feat_noise_std)
                        for name, feat in real_pooled
                    ]
                real_pooled = smooth_proj(real_pooled)

            # Negatives selection: MoCo queue > random neg > self-contrastive
            neg_pooled = None  # None = self-contrastive (default)
            if moco_queue is not None:
                moco_min = tc.get("moco_min_queue", 256)
                if moco_queue.size >= moco_min:
                    moco_n = tc.get("moco_n_neg", 128)
                    neg_pooled = moco_queue.sample(moco_n, device)
            elif tc.get("use_random_neg", False):
                n_rand = tc.get("n_random_neg", B)
                rand_ids = torch.randint(
                    0, vocab_embeddings.shape[0], (n_rand, suffix_len), device=device
                )
                with torch.no_grad():
                    with torch.amp.autocast(device.type, enabled=use_bf16, dtype=torch.bfloat16):
                        rand_features = encoder(input_ids=rand_ids, attention_mask=None)
                neg_pooled = pool_features(rand_features, pooler, None)
                if smooth_proj is not None:
                    if feat_noise_std > 0:
                        neg_pooled = [
                            (name, feat + torch.randn_like(feat) * feat_noise_std)
                            for name, feat in neg_pooled
                        ]
                    with torch.no_grad():
                        neg_pooled = smooth_proj(neg_pooled)

            # --- Iterative refinement loop ---
            n_refine = tc.get("n_refine_steps", 1)  # 1 = no refinement (backward compat)
            refine_lr = tc.get("refine_lr", 0.1)
            refine_lr_decay = tc.get("refine_lr_decay", 0.5)

            # Ensure enc_input requires grad for autograd.grad in refinement
            enc_input = enc_input.requires_grad_(True)

            metrics = {}  # will be populated by refinement loop + final drift_metrics

            # When n_refine > 1, create_graph=True requires double-backward through
            # attention. Flash/efficient attention doesn't support this, so force the
            # math backend which does. For n_refine=1 this is a no-op context.
            _sdpa_ctx = (
                torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                if n_refine > 1 else torch.nn.attention.sdpa_kernel(
                    [torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                     torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                     torch.nn.attention.SDPBackend.MATH])
            )
            with _sdpa_ctx:
              for refine_k in range(n_refine):
                with torch.amp.autocast(device.type, enabled=use_bf16, dtype=torch.bfloat16):
                    gen_features = encoder(inputs_embeds=enc_input, attention_mask=None)
                gen_pooled = pool_features(gen_features, pooler, None)

                if smooth_proj is not None:
                    if feat_noise_std > 0:
                        gen_pooled = [
                            (name, feat + torch.randn_like(feat) * feat_noise_std)
                            for name, feat in gen_pooled
                        ]
                    gen_pooled = smooth_proj(gen_pooled)

                # MoCo queue: push gen features from final refinement step only
                if moco_queue is not None and refine_k == n_refine - 1:
                    moco_queue.push(gen_pooled)

                # Compute drifting loss (in fp32)
                drift_loss, drift_metrics = drifting_loss.compute(
                    gen_features=gen_pooled,
                    pos_features=real_pooled,
                    neg_features=neg_pooled,
                )
                metrics[f"refine_{refine_k}_loss"] = drift_loss.item()

                if refine_k < n_refine - 1:
                    # Gradient-based refinement step
                    eta = refine_lr * (refine_lr_decay ** refine_k)
                    grad = torch.autograd.grad(drift_loss, enc_input, create_graph=True)[0]
                    grad_norm = grad.norm()
                    if grad_norm > 1.0:
                        grad = grad / grad_norm
                    enc_input = enc_input - eta * grad
                    metrics[f"refine_{refine_k}_grad"] = grad_norm.item()

            # Use final step's loss and metrics
            loss = drift_loss
            metrics.update(drift_metrics)
            if n_refine > 1:
                metrics["n_refine"] = n_refine

            # Log mean distance to random negatives (for monitoring)
            if neg_pooled is not None:
                with torch.no_grad():
                    rn_dists = []
                    for (_, phi_g), (_, phi_r) in zip(gen_pooled, neg_pooled):
                        rn_dists.append(pairwise_l2(phi_g, phi_r).mean().item())
                    metrics["rand_neg"] = sum(rn_dists) / len(rn_dists)

            # Smoothness regularizer
            if smooth_proj is not None:
                smooth_weight = smooth_cfg.get("smoothness_weight", 0.01)
                if smooth_weight > 0:
                    # Recompute on clean (un-noised) pooled features
                    clean_pooled = pool_features(real_features, pooler, suffix_mask)
                    s_loss = smooth_proj.smoothness_loss(
                        clean_pooled,
                        noise_std=smooth_cfg.get("lipschitz_noise_std", 0.1),
                    )
                    loss = loss + smooth_weight * s_loss
                    metrics["smooth_loss"] = s_loss.item()

            # Diversity regularizer (log-barrier)
            diversity_weight = tc.get("diversity_weight", 0.0)
            if diversity_weight > 0:
                gen_flat = suffix_out.mean(dim=1)
                pw_dist = pairwise_l2(gen_flat, gen_flat)
                eye_mask = ~torch.eye(B, device=device, dtype=torch.bool)
                avg_dist = pw_dist[eye_mask].mean()

                # Freeze reference diversity after init — no ratchet
                if div_ema_ref is None:
                    div_ema_ref = avg_dist.item()

                # Log-barrier: strong gradient at all diversity levels
                diversity_loss = -torch.log(avg_dist / div_ema_ref + 1e-6)
                diversity_loss = diversity_loss.clamp(min=0.0)  # No reward for exceeding reference
                loss = loss + diversity_weight * diversity_loss
                metrics["gen_diversity"] = avg_dist.item()
                metrics["diversity_loss"] = diversity_loss.item()
                metrics["div_ref"] = div_ema_ref

            # Spectral regularization on generator weights
            spectral_weight = tc.get("spectral_weight", 0.0)
            if spectral_weight > 0:
                s_reg = spectral_regularization(generator)
                loss = loss + spectral_weight * s_reg
                metrics["spectral_reg"] = s_reg.item()

            # Position diversity loss — penalize high pairwise cosine sim within each sequence
            pos_div_weight = tc.get("position_diversity_weight", 0.0)
            if pos_div_weight > 0:
                enc_norm = nn.functional.normalize(enc_input, dim=-1)  # (B, S, D)
                sim_mat = torch.bmm(enc_norm, enc_norm.transpose(1, 2))  # (B, S, S)
                eye = torch.eye(sim_mat.shape[1], device=device).unsqueeze(0)
                off_diag = sim_mat * (1 - eye)
                n_off = sim_mat.shape[1] * (sim_mat.shape[1] - 1)
                pos_div_loss = off_diag.sum(dim=(1, 2)).mean() / n_off
                loss = loss + pos_div_weight * pos_div_loss
                metrics["pos_div"] = pos_div_loss.item()

            # Bigram diversity loss — penalize repeated bigrams within each sequence
            bigram_div_weight = tc.get("bigram_diversity_weight", 0.0)
            if bigram_div_weight > 0:
                bigrams = torch.cat([enc_input[:, :-1], enc_input[:, 1:]], dim=-1)  # (B, S-1, 2D)
                bg_norm = nn.functional.normalize(bigrams, dim=-1)
                bg_sim = torch.bmm(bg_norm, bg_norm.transpose(1, 2))  # (B, S-1, S-1)
                bg_eye = torch.eye(bg_sim.shape[1], device=device).unsqueeze(0)
                bg_off = bg_sim * (1 - bg_eye)
                bg_n_off = bg_sim.shape[1] * (bg_sim.shape[1] - 1)
                bigram_div_loss = bg_off.sum(dim=(1, 2)).mean() / bg_n_off
                loss = loss + bigram_div_weight * bigram_div_loss
                metrics["bigram_div"] = bigram_div_loss.item()

            # Unique token ratio — non-differentiable monitoring
            if pos_div_weight > 0 or bigram_div_weight > 0:
                with torch.no_grad():
                    flat_enc = enc_input.detach().reshape(-1, enc_input.shape[-1])
                    nearest = torch.cdist(flat_enc, vocab_embs_device if vocab_embs_device is not None else vocab_embeddings.to(device)).argmin(dim=-1)
                    nearest = nearest.reshape(B, -1)
                    uniq_ratios = []
                    for b in range(B):
                        n_uniq = nearest[b].unique().numel()
                        uniq_ratios.append(n_uniq / nearest.shape[1])
                    metrics["uniq_ratio"] = sum(uniq_ratios) / len(uniq_ratios)

            if gumbel_tau > 0:
                metrics["gumbel_tau"] = gumbel_tau

            # GPT-2 perplexity-matching loss — target natural entropy, not minimum
            lm_weight = tc.get("lm_weight", 0.0)
            if lm_weight > 0:
                # Compute reference CE from real text (once, frozen)
                if real_ce_ref is None:
                    with torch.no_grad():
                        real_last = real_features[max(real_features.keys())]
                        real_logits = real_last @ vocab_embs_device.T
                        real_ce_ref = nn.functional.cross_entropy(
                            real_logits[:, :-1].reshape(-1, real_logits.shape[-1]),
                            suffix_ids[:, 1:].reshape(-1),
                        ).item()
                    logger.info(f"Real text CE reference: {real_ce_ref:.2f} (ppl={math.exp(real_ce_ref):.1f})")

                # Generated text CE
                last_hidden = gen_features[max(gen_features.keys())]
                lm_logits = last_hidden @ vocab_embs_device.T  # (B, S, V)
                with torch.no_grad():
                    flat_enc = enc_input.detach().reshape(-1, enc_input.shape[-1])
                    target_ids = torch.cdist(flat_enc, vocab_embs_device).argmin(dim=-1)
                    target_ids = target_ids.reshape(B, -1)
                gen_ce = nn.functional.cross_entropy(
                    lm_logits[:, :-1].reshape(-1, lm_logits.shape[-1]),
                    target_ids[:, 1:].reshape(-1),
                )
                # Perplexity matching: penalize deviation from real CE
                lm_loss = (gen_ce - real_ce_ref) ** 2
                loss = loss + lm_weight * lm_loss
                metrics["lm_loss"] = lm_loss.item()
                metrics["gen_ce"] = gen_ce.item()

            # Repetition penalty — penalize consecutive identical nearest-vocab tokens
            rep_weight = tc.get("rep_penalty_weight", 0.0)
            if rep_weight > 0:
                with torch.no_grad():
                    flat_enc = enc_input.detach().reshape(-1, enc_input.shape[-1])
                    nearest_ids = torch.cdist(flat_enc, vocab_embs_device).argmin(dim=-1)
                    nearest_ids = nearest_ids.reshape(B, -1)
                # Cosine similarity between consecutive token embeddings
                tok_embs = vocab_embs_device[nearest_ids]  # (B, S, D)
                cos_sim = nn.functional.cosine_similarity(
                    tok_embs[:, :-1], tok_embs[:, 1:], dim=-1
                )  # (B, S-1)
                # Penalty: encourage low similarity between consecutive positions
                # Detach targets, gradient flows through enc_input → embed_proj → generator
                # But nearest_ids are discrete, so we need a differentiable proxy:
                # penalize enc_input similarity between consecutive positions directly
                enc_cos = nn.functional.cosine_similarity(
                    enc_input[:, :-1], enc_input[:, 1:], dim=-1
                )
                rep_loss = enc_cos.clamp(min=0.5).mean()  # only penalize high similarity
                loss = loss + rep_weight * rep_loss
                metrics["rep_loss"] = rep_loss.item()
                metrics["rep_cos"] = cos_sim.mean().item()

            # Vocab anchoring — keep generator output near valid token embeddings
            vocab_weight = tc.get("vocab_anchor_weight", 0.0)
            if vocab_weight > 0:
                # enc_input: (B, suffix_len, enc_dim) — generator output in encoder space
                flat_emb = enc_input.reshape(-1, enc_input.shape[-1])  # (B*S, enc_dim)
                # Compute distance to nearest vocab token for each position
                # Use chunked computation to avoid OOM on large vocab (50257)
                chunk_size = tc.get("vocab_anchor_chunk", 0)
                if chunk_size > 0:
                    min_dists = []
                    for i in range(0, flat_emb.shape[0], chunk_size):
                        chunk = flat_emb[i:i + chunk_size]
                        dists = torch.cdist(chunk, vocab_embs_device)  # (chunk, V)
                        min_dists.append(dists.min(dim=-1).values)
                    min_vocab_dist = torch.cat(min_dists).mean()
                else:
                    dists = torch.cdist(flat_emb, vocab_embs_device)  # (B*S, V)
                    min_vocab_dist = dists.min(dim=-1).values.mean()
                loss = loss + vocab_weight * min_vocab_dist
                metrics["vocab_anchor"] = min_vocab_dist.item()

            # Scale loss for accumulation and backward
            (loss / accum_steps).backward()

            # Accumulate metrics (average across micro-steps)
            for k, v in metrics.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v / accum_steps

            # Push real embeddings to queue
            with torch.no_grad():
                real_embeds = encoder.embed_tokens(input_ids)
                queue.push(real_embeds.mean(dim=1))

        metrics = accum_metrics

        # Gradient clipping (all trainable params, including projections)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            opt_params, tc["gradient_clip"]
        )
        metrics["grad_norm"] = grad_norm.item()

        optimizer.step()

        # EMA update
        update_ema(ema_generator, generator, tc["ema_decay"])

        # Update smoothed metrics (every step, not just log steps)
        _ema_alpha = 0.05  # smoothing factor — lower = smoother
        cur_loss = metrics["total_loss"]
        ema_loss = cur_loss if ema_loss is None else (1 - _ema_alpha) * ema_loss + _ema_alpha * cur_loss

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

            metrics["ema_loss"] = ema_loss
            diag_parts = [
                f"loss={metrics['total_loss']:.4f}",
                f"ema={ema_loss:.4f}",
                f"grad={metrics['grad_norm']:.1f}",
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
            if "smooth_loss" in metrics:
                diag_parts.append(f"smooth={metrics['smooth_loss']:.3f}")
            if "vocab_anchor" in metrics:
                diag_parts.append(f"vanc={metrics['vocab_anchor']:.1f}")
            if "div_ref" in metrics:
                diag_parts.append(f"dref={metrics['div_ref']:.1f}")
            if "spectral_reg" in metrics:
                diag_parts.append(f"spec={metrics['spectral_reg']:.3f}")
            if "lm_loss" in metrics:
                diag_parts.append(f"lm={metrics['lm_loss']:.2f}")
            if "gen_ce" in metrics:
                diag_parts.append(f"ce={metrics['gen_ce']:.2f}")
            if "pos_div" in metrics:
                diag_parts.append(f"pdiv={metrics['pos_div']:.3f}")
            if "bigram_div" in metrics:
                diag_parts.append(f"bdiv={metrics['bigram_div']:.3f}")
            if "uniq_ratio" in metrics:
                diag_parts.append(f"uniq={metrics['uniq_ratio']:.2f}")
            if "rep_loss" in metrics:
                diag_parts.append(f"rep={metrics['rep_loss']:.2f}")
            if "rep_cos" in metrics:
                diag_parts.append(f"rcos={metrics['rep_cos']:.2f}")
            if "gumbel_tau" in metrics:
                diag_parts.append(f"gtau={metrics['gumbel_tau']:.2f}")
            if "rand_neg" in metrics:
                diag_parts.append(f"rneg={metrics['rand_neg']:.1f}")
            if moco_queue is not None:
                diag_parts.append(f"qsz={moco_queue.size}")
            if "n_refine" in metrics:
                nref = int(metrics["n_refine"])
                diag_parts.append(f"nref={nref}")
                rloss_parts = []
                for k in range(nref):
                    key = f"refine_{k}_loss"
                    if key in metrics:
                        rloss_parts.append(f"{metrics[key]:.3f}")
                if rloss_parts:
                    diag_parts.append(f"rloss={'→'.join(rloss_parts)}")
                rgrad_parts = []
                for k in range(nref - 1):
                    key = f"refine_{k}_grad"
                    if key in metrics:
                        rgrad_parts.append(f"{metrics[key]:.1f}")
                if rgrad_parts:
                    diag_parts.append(f"rgrad={','.join(rgrad_parts)}")

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
            if smooth_proj is not None:
                ckpt["smooth_proj"] = smooth_proj.state_dict()
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
