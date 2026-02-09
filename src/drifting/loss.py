"""Full drifting loss combining all components."""

import torch
import torch.nn.functional as F

from src.drifting.field import compute_V
from src.drifting.normalization import FeatureNormalizer, DriftNormalizer


class DriftingLoss:
    """Orchestrates the full drifting loss computation.

    For each (layer, pooling_type) feature pair:
    1. Normalize features (avg pairwise dist = sqrt(C))
    2. Compute drifting field V (attraction - repulsion)
    3. Normalize drift (E[||V||^2/C] ~ 1)
    4. MSE loss: ||phi(x) - sg(phi(x) + V)||^2
    """

    def __init__(
        self,
        temperatures: tuple[float, ...] = (0.02, 0.05, 0.2),
        momentum: float = 0.99,
        max_V_norm: float | None = None,
        normalize_drift: bool = True,
        loss_fn: str = "mse",
        huber_delta: float = 1.0,
    ):
        self.temperatures = temperatures
        self.max_V_norm = max_V_norm
        self.normalize_drift = normalize_drift
        self.loss_fn = loss_fn
        self.huber_delta = huber_delta
        self.feat_normalizers: dict[str, FeatureNormalizer] = {}
        self.drift_normalizers: dict[str, DriftNormalizer] = {}

    def _get_normalizers(self, name: str) -> tuple[FeatureNormalizer, DriftNormalizer]:
        if name not in self.feat_normalizers:
            self.feat_normalizers[name] = FeatureNormalizer()
            self.drift_normalizers[name] = DriftNormalizer()
        return self.feat_normalizers[name], self.drift_normalizers[name]

    def compute(
        self,
        gen_features: list[tuple[str, torch.Tensor]],
        pos_features: list[tuple[str, torch.Tensor]],
        neg_features: list[tuple[str, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total drifting loss across all feature scales.

        Args:
            gen_features: list of (name, tensor) for generated samples
            pos_features: list of (name, tensor) for positive (real) samples
            neg_features: list of (name, tensor) for negative samples.
                          If None, uses gen_features as negatives.

        Returns:
            (total_loss, metrics_dict)
        """
        if neg_features is None:
            neg_features = gen_features

        total_loss = torch.tensor(0.0, device=gen_features[0][1].device)
        metrics = {}

        for (name, phi_gen), (_, phi_pos), (_, phi_neg) in zip(
            gen_features, pos_features, neg_features
        ):
            feat_norm, drift_norm = self._get_normalizers(name)

            # Update and apply feature normalization
            # Use all features together for scale estimation
            with torch.no_grad():
                all_feats = torch.cat([phi_gen.detach(), phi_pos.detach()], dim=0)
                feat_norm.update_scale(all_feats)

            phi_gen_n = feat_norm.normalize(phi_gen)
            phi_pos_n = feat_norm.normalize(phi_pos.detach())
            phi_neg_n = feat_norm.normalize(phi_neg.detach())

            # Compute drifting field
            self_mask = neg_features is gen_features
            V_result = compute_V(
                phi_gen_n,
                phi_pos_n,
                phi_neg_n if not self_mask else phi_gen_n,
                temperatures=self.temperatures,
                self_mask=self_mask,
            )
            V = V_result["V"]

            # Update and optionally apply drift normalization
            with torch.no_grad():
                drift_norm.update_lambda(V.detach())
            if self.normalize_drift:
                V = drift_norm.normalize(V)

            # Clamp V magnitude to prevent positive feedback divergence
            if self.max_V_norm is not None:
                V_norms = V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                clamp_mask = V_norms > self.max_V_norm
                V = torch.where(clamp_mask, V * (self.max_V_norm / V_norms), V)

            # Loss: push phi(x) toward sg(phi(x) + V)
            target = (phi_gen_n + V).detach()  # stop gradient
            if self.loss_fn == "huber":
                loss = F.smooth_l1_loss(phi_gen_n, target, beta=self.huber_delta)
            else:
                loss = F.mse_loss(phi_gen_n, target)

            total_loss = total_loss + loss

            # Metrics
            with torch.no_grad():
                metrics[f"{name}/loss"] = loss.item()
                metrics[f"{name}/V_norm"] = V.norm(dim=-1).mean().item()
                metrics[f"{name}/V_raw_norm"] = V_result["V"].norm(dim=-1).mean().item()
                metrics[f"{name}/attraction_norm"] = V_result["V_attract"].norm(dim=-1).mean().item()
                metrics[f"{name}/repulsion_norm"] = V_result["V_repel"].norm(dim=-1).mean().item()
                if drift_norm.running_lambda is not None:
                    metrics[f"{name}/drift_lambda"] = drift_norm.running_lambda
                if feat_norm.running_scale is not None:
                    metrics[f"{name}/feat_scale"] = feat_norm.running_scale

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics
