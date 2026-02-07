"""MoCo-style sample queue for storing encoded real data features."""

import torch


class SampleQueue:
    """FIFO queue of feature vectors for use as positive samples.

    Stores pre-computed features from real data to increase the effective
    number of positives without recomputing them each step.
    """

    def __init__(self, feature_dim: int, max_size: int = 4096):
        self.max_size = max_size
        self.feature_dim = feature_dim
        self.buffer = torch.zeros(max_size, feature_dim)
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def push(self, features: torch.Tensor) -> None:
        """Add features to the queue.

        Args:
            features: (N, feature_dim) to enqueue
        """
        features = features.detach().cpu()
        N = features.shape[0]

        if N >= self.max_size:
            self.buffer = features[-self.max_size:].clone()
            self.ptr = 0
            self.full = True
            return

        end = self.ptr + N
        if end <= self.max_size:
            self.buffer[self.ptr:end] = features
        else:
            overflow = end - self.max_size
            self.buffer[self.ptr:] = features[:N - overflow]
            self.buffer[:overflow] = features[N - overflow:]
            self.full = True

        self.ptr = end % self.max_size
        if end >= self.max_size:
            self.full = True

    def sample(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        """Sample n features from the queue without replacement.

        Args:
            n: number of samples
            device: target device

        Returns:
            (n, feature_dim) tensor
        """
        size = self.max_size if self.full else self.ptr
        if size == 0:
            raise RuntimeError("Queue is empty")
        n = min(n, size)
        idx = torch.randperm(size)[:n]
        out = self.buffer[idx]
        if device is not None:
            out = out.to(device)
        return out

    @property
    def size(self) -> int:
        return self.max_size if self.full else self.ptr
