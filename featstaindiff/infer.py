"""Generate IHC feature matrices from H&E feature matrices."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import torch

from .data import load_features
from .diffusion import DiffusionConfig, ddim_sample
from .model import FeatStainDiff, ModelConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="one feature file or a directory of feature files")
    parser.add_argument("--output", required=True, type=Path, help=".pt file or output directory")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--ensemble", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--combine-with-input", action="store_true",
                        help="add generated IHC features to H&E, as in paper's MIST-to-TCGA-BRCA setting")
    parser.add_argument("--raw-weights", action="store_true", help="use non-EMA training weights")
    parser.add_argument("--preserve-ids", action="store_true",
                        help="save a features/ids dictionary when the input contains row IDs")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _file_identity(path: Path) -> tuple[int, int] | None:
    """Return the identity used by samefile(), including hard-link aliases."""
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino


def run(args: argparse.Namespace) -> list[Path]:
    if args.batch_size < 1 or args.steps < 1 or args.ensemble < 1:
        raise ValueError("batch-size, steps and ensemble must be positive")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if args.input.is_dir():
        inputs = sorted(path for path in args.input.iterdir()
                        if path.is_file() and path.suffix in (".pt", ".pth", ".npy", ".npz"))
        if not inputs:
            raise ValueError(f"{args.input}: no supported feature files found")
        if os.path.lexists(args.output) and not args.output.is_dir():
            raise ValueError("directory input requires an output directory")
        outputs = [args.output / (path.stem + ".pt") for path in inputs]
    else:
        inputs = [args.input]
        if args.output.suffix != ".pt":
            raise ValueError("single-file input requires a .pt output path")
        outputs = [args.output]
    if len({path.name for path in outputs}) != len(outputs):
        raise ValueError("input files would produce duplicate output basenames")
    protected_paths = [*inputs, args.checkpoint]
    protected_resolved = {path.resolve() for path in protected_paths}
    protected_identities = {_file_identity(path) for path in protected_paths}
    protected_identities.discard(None)
    for destination in outputs:
        if (destination.resolve() in protected_resolved
                or _file_identity(destination) in protected_identities):
            raise ValueError("output paths must not replace input features or the checkpoint")
        if os.path.lexists(destination) and not args.overwrite:
            raise FileExistsError(f"{destination} exists; use --overwrite to replace it")
        if destination.exists() and not destination.is_file():
            raise ValueError(f"{destination}: output must be a regular file")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = FeatStainDiff(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state" if args.raw_weights else "ema_state"])
    model.eval()
    diffusion = DiffusionConfig(**checkpoint["diffusion_config"])
    # Training checkpoints also contain optimizer states and a second set of
    # model weights; inference needs neither after the selected weights load.
    del checkpoint
    for file_index, (source, destination) in enumerate(zip(inputs, outputs)):
        features, ids = load_features(source)
        if features.shape[1] != model.config.feature_dim:
            raise ValueError(f"{source}: expected dimension {model.config.feature_dim}")
        if len(features) == 0:
            raise ValueError(f"{source}: feature matrix contains no rows")
        generated = []
        for chunk_index, start in enumerate(range(0, len(features), args.batch_size)):
            he = features[start:start + args.batch_size].to(device)
            prediction = ddim_sample(model, he, diffusion, steps=args.steps,
                                     ensemble=args.ensemble,
                                     seed=args.seed + file_index * 1_000_000
                                     + chunk_index * args.ensemble)
            if args.combine_with_input:
                prediction = prediction + he
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"{source}: generated features contain NaN or infinity")
            generated.append(prediction.cpu())
        output = torch.cat(generated, dim=0)
        payload = {"features": output, "ids": ids} if args.preserve_ids and ids is not None else output
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(payload, stream)
            if args.overwrite:
                temporary.replace(destination)
            else:
                # Fail if the target appeared after preflight, including a
                # dangling symlink, without replacing that new entry.
                os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"{source} -> {destination}: {tuple(output.shape)}", flush=True)
    return outputs


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
