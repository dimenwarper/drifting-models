"""Multi-scale FIFO queue of generated features for MoCo-style negatives."""

import torch


class FeatureQueue:
    """FIFO queue storing past generated features per feature scale.

    Each scale (e.g. "layer_3/window_4") gets its own circular buffer.
    Features are pushed after smooth_proj and sampled as negatives for
    the drifting loss, providing collapse-independent repulsion.
    """

    def __init__(self, max_size: int = 8192):
        self.max_size = max_size
        self.buffers: dict[str, torch.Tensor] = {}
        self.ptrs: dict[str, int] = {}
        self.full: dict[str, bool] = {}

    @torch.no_grad()
    def push(self, features: list[tuple[str, torch.Tensor]]) -> None:
        """Push detached features into each scale's circular buffer.

        Args:
            features: list of (name, tensor) where tensor is (N, C).
        """
        for name, feat in features:
            feat = feat.detach().cpu()
            N, C = feat.shape

            # Lazily initialize buffer for this scale
            if name not in self.buffers:
                self.buffers[name] = torch.zeros(self.max_size, C)
                self.ptrs[name] = 0
                self.full[name] = False

            buf = self.buffers[name]
            ptr = self.ptrs[name]

            if N >= self.max_size:
                buf[:] = feat[-self.max_size:]
                self.ptrs[name] = 0
                self.full[name] = True
                continue

            end = ptr + N
            if end <= self.max_size:
                buf[ptr:end] = feat
            else:
                overflow = end - self.max_size
                buf[ptr:] = feat[:N - overflow]
                buf[:overflow] = feat[N - overflow:]
                self.full[name] = True

            self.ptrs[name] = end % self.max_size
            if end >= self.max_size:
                self.full[name] = True

    def sample(self, n: int, device: torch.device) -> list[tuple[str, torch.Tensor]] | None:
        """Random-sample n entries from each scale's buffer.

        Args:
            n: number of negatives to sample per scale.
            device: target device for returned tensors.

        Returns:
            List of (name, tensor) matching the scale structure, or None if
            any queue is empty.
        """
        if not self.buffers:
            return None

        result = []
        for name, buf in self.buffers.items():
            sz = self.max_size if self.full[name] else self.ptrs[name]
            if sz == 0:
                return None
            k = min(n, sz)
            idx = torch.randperm(sz)[:k]
            result.append((name, buf[idx].to(device)))
        return result

    @property
    def size(self) -> int:
        """Current number of entries (min across all scales)."""
        if not self.buffers:
            return 0
        sizes = [
            self.max_size if self.full[name] else self.ptrs[name]
            for name in self.buffers
        ]
        return min(sizes)
