"""Continuous VP diffusion training and deterministic DDIM feature sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import FeatStainDiff


@dataclass(frozen=True)
class DiffusionConfig:
    # These SDE endpoints follow the local experimental code, not a value
    # specified by the article.
    beta_min: float = 0.1
    beta_max: float = 20.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.beta_min)
            or not math.isfinite(self.beta_max)
            or self.beta_min <= 0
            or self.beta_max < self.beta_min
        ):
            raise ValueError("require finite 0 < beta_min <= beta_max")
        # DDIM divides by sqrt(alpha_bar); reject schedules that underflow in
        # float32, including subnormal values that devices may flush to zero.
        final_alpha = alpha_bar(torch.tensor(1.0, dtype=torch.float32), self).item()
        if final_alpha < torch.finfo(torch.float32).tiny:
            raise ValueError("diffusion schedule makes alpha_bar(1) underflow in float32; reduce beta values")


def alpha_bar(t: Tensor, config: DiffusionConfig) -> Tensor:
    integrated_beta = config.beta_min * t + 0.5 * (config.beta_max - config.beta_min) * t.square()
    return (-integrated_beta).exp()


def training_losses(
    model: FeatStainDiff,
    he: Tensor,
    ihc: Tensor,
    diffusion: DiffusionConfig,
    *,
    lambda_contrastive: float = 1.0,
    lambda_mmd: float = 0.5,
    lambda_load: float = 0.1,
) -> dict[str, Tensor]:
    """Evaluate all four terms in Eq. (14) on paired feature vectors."""
    if he.shape != ihc.shape:
        raise ValueError("H&E and IHC feature batches must match")
    if any(not math.isfinite(weight) or weight < 0
           for weight in (lambda_contrastive, lambda_mmd, lambda_load)):
        raise ValueError("loss weights must be finite and non-negative")
    t = torch.rand(ihc.shape[0], device=ihc.device, dtype=torch.float32).clamp_(1e-4, 1.0)
    ab = alpha_bar(t, diffusion)[:, None]
    noise = torch.randn_like(ihc)
    noisy_ihc = ab.sqrt() * ihc + (1.0 - ab).sqrt() * noise
    output = model(noisy_ihc, he, t, target_ihc=ihc)
    # The paper specifies an L2 noise objective but does not define its
    # reduction; mean squared error is the DDPM implementation choice here.
    noise_loss = (noise - output["pred_noise"]).square().mean()
    contrastive = output["contrastive"]
    mmd = output["mmd"]
    load = output["load"]
    total = noise_loss + lambda_contrastive * contrastive + lambda_mmd * mmd + lambda_load * load
    return {
        "total": total,
        "noise": noise_loss,
        "contrastive": contrastive,
        "mmd": mmd,
        "load": load,
    }


@torch.inference_mode()
def ddim_sample(
    model: FeatStainDiff,
    he: Tensor,
    diffusion: DiffusionConfig,
    *,
    steps: int = 30,
    ensemble: int = 5,
    seed: int = 1234,
) -> Tensor:
    """Generate IHC feature vectors from pure Gaussian noise (eta=0)."""
    if steps < 1 or ensemble < 1:
        raise ValueError("steps and ensemble must be positive")
    if he.ndim != 2 or he.shape[1] != model.config.feature_dim:
        raise ValueError(f"he must have shape [batch, {model.config.feature_dim}]")
    model.eval()
    times = torch.linspace(1.0, 0.0, steps + 1, device=he.device, dtype=torch.float32)
    result = torch.zeros_like(he)
    condition = model.prepare_condition(he)
    for member in range(ensemble):
        generator = torch.Generator(device=he.device).manual_seed(seed + member)
        x = torch.randn(he.shape, generator=generator, device=he.device, dtype=he.dtype)
        for current, following in zip(times[:-1], times[1:]):
            t = current.expand(he.shape[0])
            pred_noise = model(x, he, t, condition=condition)["pred_noise"]
            ab_t = alpha_bar(current, diffusion)
            ab_next = alpha_bar(following, diffusion)
            estimate_x0 = (x - (1.0 - ab_t).sqrt() * pred_noise) / ab_t.sqrt()
            x = ab_next.sqrt() * estimate_x0 + (1.0 - ab_next).sqrt() * pred_noise
        result = result + x
    return result / ensemble
