"""TinyStories dataset with GPT-2 tokenization."""

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset


class TinyStoriesDataset(Dataset):
    """TinyStories dataset tokenized with GPT-2 tokenizer.

    Each item is padded/truncated to seq_len tokens.
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 256,
        max_samples: int | None = None,
    ):
        self.seq_len = seq_len
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        ds = load_dataset("roneneldan/TinyStories", split=split)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.data = ds

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.data[idx]["text"]
        encoded = self.tokenizer(
            text,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
