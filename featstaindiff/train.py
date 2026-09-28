"""Train FeatStainDiff from registered, paired H&E/IHC feature vectors."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import PairedFeatures
from .diffusion import DiffusionConfig, training_losses
from .model import FeatStainDiff, ModelConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he-features", required=True, type=Path)
    parser.add_argument("--ihc-features", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--assume-aligned-order", action="store_true",
                        help="required for tensors without paired row ids")
    parser.add_argument("--max-steps", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--semantic-depth", type=int, default=3)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--mmd-bandwidth", type=float, default=None,
                        help="Gaussian sigma; default uses a detached median distance")
    parser.add_argument("--lambda-contrastive", type=float, default=1.0)
    parser.add_argument("--lambda-mmd", type=float, default=0.5)
    parser.add_argument("--lambda-load", type=float, default=0.1)
    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=20.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="max gradient norm; zero disables clipping")
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="off")
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def choose_device(name: str) -> torch.device:
    device = torch.device("cuda" if name == "auto" and torch.cuda.is_available()
                          else "cpu" if name == "auto" else name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def save_checkpoint(
    path: Path,
    *,
    model: FeatStainDiff,
    ema: FeatStainDiff,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "ema_state": ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "model_config": asdict(model_config),
        "diffusion_config": asdict(diffusion_config),
        "train_args": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(args: argparse.Namespace) -> Path:
    for name in ("he_features", "ihc_features", "output_dir", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    for name in ("lr", "weight_decay", "grad_clip", "ema_decay",
                 "lambda_contrastive", "lambda_mmd", "lambda_load"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"{name} must be finite")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    checkpoint_path = args.output_dir / "checkpoints" / "last.pt"
    if os.path.lexists(checkpoint_path):
        if args.resume is None or not args.resume.is_file() or not checkpoint_path.is_file():
            raise FileExistsError(f"{checkpoint_path} exists; resume that checkpoint or use a new --output-dir")
        if not args.resume.samefile(checkpoint_path):
            raise FileExistsError(f"{checkpoint_path} belongs to another run; use a new --output-dir")
    if args.max_steps < 1 or args.batch_size < 2 or args.lr <= 0:
        raise ValueError("max-steps, batch-size and lr must be positive; batch-size must be >= 2")
    if args.save_every < 1 or args.log_every < 1:
        raise ValueError("save/log intervals must be positive")
    if args.weight_decay < 0 or args.grad_clip < 0 or not 0 <= args.ema_decay < 1:
        raise ValueError("weight-decay and grad-clip must be non-negative; EMA decay in [0,1)")
    if min(args.lambda_contrastive, args.lambda_mmd, args.lambda_load) < 0:
        raise ValueError("loss weights must be non-negative")
    device = choose_device(args.device)
    if args.amp != "off" and device.type != "cuda":
        raise ValueError("mixed precision is supported only on CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset = PairedFeatures(args.he_features, args.ihc_features,
                             assume_aligned_order=args.assume_aligned_order)
    if dataset.he.shape[1] != args.feature_dim:
        raise ValueError(f"feature matrix has dimension {dataset.he.shape[1]}, "
                         f"but --feature-dim is {args.feature_dim}")
    if args.batch_size > len(dataset):
        raise ValueError("batch-size exceeds number of paired features")
    # A fixed permutation per epoch makes row order independent of restarts.
    # The final incomplete batch is dropped, as in the experimental loader.
    steps_per_epoch = len(dataset) // args.batch_size
    model_config = ModelConfig(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        patch_size=args.patch_size,
        semantic_depth=args.semantic_depth,
        num_experts=args.num_experts,
        contrastive_temperature=args.contrastive_temperature,
        mmd_bandwidth=args.mmd_bandwidth,
    )
    diffusion_config = DiffusionConfig(args.beta_min, args.beta_max)
    model = FeatStainDiff(model_config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["model_config"] != asdict(model_config):
            raise ValueError("resume checkpoint model config differs from command-line config")
        if checkpoint["diffusion_config"] != asdict(diffusion_config):
            raise ValueError("resume checkpoint diffusion config differs from command-line config")
        previous = checkpoint.get("train_args", {})
        fixed_settings = (
            "batch_size", "lr", "weight_decay", "lambda_contrastive", "lambda_mmd",
            "lambda_load", "ema_decay", "grad_clip", "amp", "seed",
        )
        for key in fixed_settings:
            if previous.get(key) != getattr(args, key):
                raise ValueError(f"resume checkpoint {key} differs from command-line setting")
        for key in ("he_features", "ihc_features"):
            if key not in previous or Path(previous[key]).resolve() != getattr(args, key).resolve():
                raise ValueError(f"resume checkpoint {key} differs from command-line path")
        model.load_state_dict(checkpoint["model_state"])
        ema.load_state_dict(checkpoint["ema_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_step = int(checkpoint["step"])
        del checkpoint
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps({"model": asdict(model_config), "diffusion": asdict(diffusion_config),
                    "training": {key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(args).items()}},
                   indent=2) + "\n"
    )
    epoch_index = -1
    permutation = None
    print(f"device={device} pairs={len(dataset)} parameters={sum(p.numel() for p in model.parameters())} "
          f"start_step={start_step}", flush=True)
    if start_step >= args.max_steps:
        return args.resume if args.resume is not None else checkpoint_path
    for step in range(start_step + 1, args.max_steps + 1):
        this_epoch = (step - 1) // steps_per_epoch
        if this_epoch != epoch_index:
            epoch_index = this_epoch
            order_rng = torch.Generator(device="cpu").manual_seed(args.seed + this_epoch)
            permutation = torch.randperm(len(dataset), generator=order_rng)
        batch_index = (step - 1) % steps_per_epoch
        indices = permutation[batch_index * args.batch_size:(batch_index + 1) * args.batch_size]
        he = dataset.he[indices].to(device)
        ihc = dataset.ihc[indices].to(device)
        model.train()
        # Retry fp16 overflows on the same batch/noise. Only completed optimizer
        # updates advance the step counter and EMA.
        for amp_retries in range(16):
            torch.manual_seed(args.seed + 1_000_000 + step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if args.amp == "fp16"
                                else torch.bfloat16, enabled=args.amp != "off"):
                losses = training_losses(
                    model, he, ihc, diffusion_config,
                    lambda_contrastive=args.lambda_contrastive,
                    lambda_mmd=args.lambda_mmd,
                    lambda_load=args.lambda_load,
                )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"non-finite training loss at step {step}")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0 or not scaler.is_enabled():
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf"),
                    error_if_nonfinite=not scaler.is_enabled(),
                )
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                break
        else:
            raise FloatingPointError(f"fp16 gradients remained non-finite after 16 retries at step {step}")
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters()):
                target.lerp_(source, 1.0 - args.ema_decay)
            for target, source in zip(ema.buffers(), model.buffers()):
                target.copy_(source)
        if step % args.log_every == 0 or step == 1 or step == args.max_steps:
            values = {key: round(value.detach().float().item(), 6) for key, value in losses.items()}
            print(json.dumps({"step": step, "amp_retries": amp_retries, **values}), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(checkpoint_path, model=model, ema=ema, optimizer=optimizer,
                            scaler=scaler, step=step, model_config=model_config,
                            diffusion_config=diffusion_config, args=args)
    return checkpoint_path


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = train(args)
    print(f"checkpoint={checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
