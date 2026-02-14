"""Text generation evaluation metrics."""

import math
from collections import Counter

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def compute_perplexity(
    texts: list[str],
    model_name: str = "gpt2-medium",
    device: torch.device | None = None,
    batch_size: int = 8,
    max_length: int = 256,
) -> float:
    """Compute perplexity of texts using an independent judge model.

    Args:
        texts: List of text strings to evaluate.
        model_name: HuggingFace model to use as judge.
        device: Torch device. Defaults to cuda if available.
        batch_size: Batch size for evaluation.
        max_length: Max token length per text.

    Returns:
        Perplexity (exp of mean per-token NLL).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

        # HF returns mean loss over non-ignored tokens. Recover total NLL.
        # For per-token weighting with attention mask, compute manually.
        logits = outputs.logits[:, :-1, :]  # (B, S-1, V)
        targets = input_ids[:, 1:]  # (B, S-1)
        mask = attention_mask[:, 1:]  # (B, S-1)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        token_nll = token_nll * mask

        total_nll += token_nll.sum().item()
        total_tokens += mask.sum().item()

    return math.exp(total_nll / max(total_tokens, 1))


def compute_distinct_n(texts: list[str], n: int) -> float:
    """Compute distinct-n: ratio of unique n-grams to total n-grams.

    Args:
        texts: List of text strings.
        n: N-gram order (1, 2, or 3).

    Returns:
        Fraction of unique n-grams (0 to 1).
    """
    total_ngrams = 0
    unique_ngrams: set[tuple[str, ...]] = set()

    for text in texts:
        words = text.split()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i : i + n])
            unique_ngrams.add(ngram)
            total_ngrams += 1

    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams


def compute_self_bleu(texts: list[str], n_gram: int = 4) -> float:
    """Compute self-BLEU: average BLEU of each sample against all others.

    Lower = more diverse (less mode collapse).

    Args:
        texts: List of text strings.
        n_gram: Maximum n-gram order for BLEU.

    Returns:
        Average self-BLEU score.
    """
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    if len(texts) < 2:
        return 0.0

    tokenized = [text.split() for text in texts]
    smooth = SmoothingFunction().method1

    # Use uniform weights up to n_gram
    weights = tuple(1.0 / n_gram for _ in range(n_gram))

    scores = []
    for i, hypothesis in enumerate(tokenized):
        references = [tokenized[j] for j in range(len(tokenized)) if j != i]
        if not hypothesis:
            scores.append(0.0)
            continue
        score = sentence_bleu(references, hypothesis, weights=weights, smoothing_function=smooth)
        scores.append(score)

    return sum(scores) / len(scores)


def compute_mauve(generated_texts: list[str], reference_texts: list[str]) -> float:
    """Compute MAUVE score: distribution-level comparison to real text.

    Higher = closer to real distribution.

    Args:
        generated_texts: List of generated text strings.
        reference_texts: List of reference (real) text strings.

    Returns:
        MAUVE score (0 to 1).
    """
    import mauve

    result = mauve.compute_mauve(p_text=generated_texts, q_text=reference_texts, verbose=False)
    return result.mauve
