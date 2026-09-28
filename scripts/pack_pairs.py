#!/usr/bin/env python3
"""Pack matched per-patch PyTorch feature files into two training NPZ files.

Each input file must contain one floating-point Tensor of shape [D] or [1, D].
The H&E and IHC trees must have identical relative ``.pt`` paths. The
extension-free relative path becomes the ID stored in both outputs.

This script does not print IDs or feature values. Run it only on feature files
you are authorized to process, and keep the resulting NPZ files private.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import unicodedata

import numpy as np
import torch


class PackingError(ValueError):
    """Invalid inputs or outputs for a paired feature archive."""


def _index_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise PackingError("An input directory does not exist or is not a directory.")

    paths: dict[str, Path] = {}
    portable_ids: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PackingError("Input trees must not contain symbolic links.")
        if not path.is_file() or path.suffix != ".pt":
            continue

        feature_id = path.relative_to(root).with_suffix("").as_posix()
        portable_id = unicodedata.normalize("NFC", feature_id).casefold()
        if feature_id in paths or portable_id in portable_ids:
            raise PackingError("An input tree contains duplicate feature IDs.")
        paths[feature_id] = path
        portable_ids.add(portable_id)

    if not paths:
        raise PackingError("An input directory contains no .pt feature files.")
    return paths


def _load_vector(path: Path) -> np.ndarray:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PackingError(
            f"A feature file could not be loaded safely ({type(exc).__name__})."
        ) from None

    if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
        raise PackingError("Every .pt file must contain a floating-point Tensor.")
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1 or value.numel() == 0:
        raise PackingError("Feature Tensors must have shape [D] or [1, D], with D > 0.")
    if not bool(torch.isfinite(value).all()):
        raise PackingError("A feature Tensor contains a non-finite value.")

    vector = value.to(dtype=torch.float32).contiguous().numpy()
    if not bool(np.isfinite(vector).all()):
        raise PackingError("A feature Tensor overflows float32 or is non-finite.")
    return vector


def _stage_archive(output: Path, features: np.ndarray, ids: np.ndarray) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, features=features, ids=ids)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _output_exists(path: Path) -> bool:
    # Path.exists() returns False for dangling symlinks, which must also block us.
    return os.path.lexists(path)


def pack_pairs(
    he_dir: Path, ihc_dir: Path, he_output: Path, ihc_output: Path
) -> tuple[int, int]:
    """Validate paired files and publish no-clobber NPZ outputs.

    Returns the number of pairs and feature dimension. Each output is
    published atomically; two separate paths cannot form one atomic commit.
    """
    try:
        he_dir = he_dir.expanduser().resolve(strict=True)
        ihc_dir = ihc_dir.expanduser().resolve(strict=True)
    except OSError:
        raise PackingError("An input directory does not exist.") from None
    # Keep the final path component intact so an existing (including dangling)
    # symlink cannot bypass the no-overwrite check.
    he_output = he_output.expanduser().absolute()
    ihc_output = ihc_output.expanduser().absolute()

    if he_dir == ihc_dir or he_dir in ihc_dir.parents or ihc_dir in he_dir.parents:
        raise PackingError("H&E and IHC input directories must be separate trees.")
    if he_output.resolve(strict=False) == ihc_output.resolve(strict=False):
        raise PackingError("H&E and IHC outputs must be different files.")
    if he_output.suffix != ".npz" or ihc_output.suffix != ".npz":
        raise PackingError("Both output paths must end in .npz.")
    if _output_exists(he_output) or _output_exists(ihc_output):
        raise PackingError("An output already exists; refusing to overwrite it.")

    he_files = _index_files(he_dir)
    ihc_files = _index_files(ihc_dir)
    if he_files.keys() != ihc_files.keys():
        only_he = len(he_files.keys() - ihc_files.keys())
        only_ihc = len(ihc_files.keys() - he_files.keys())
        raise PackingError(
            f"Input trees are not paired: {only_he} H&E-only and "
            f"{only_ihc} IHC-only files."
        )

    ids = sorted(he_files)
    output_ids = np.asarray(ids, dtype=np.str_)
    first_he = _load_vector(he_files[ids[0]])
    first_ihc = _load_vector(ihc_files[ids[0]])
    feature_dim = len(first_he)
    if len(first_ihc) != feature_dim:
        raise PackingError("H&E and IHC feature dimensions differ.")

    he_output.parent.mkdir(parents=True, exist_ok=True)
    ihc_output.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    published: list[Path] = []
    try:
        # Anonymous scratch files bound memory use to one pair plus the maps.
        with tempfile.TemporaryFile(dir=he_output.parent) as he_scratch, \
                tempfile.TemporaryFile(dir=ihc_output.parent) as ihc_scratch:
            shape = (len(ids), feature_dim)
            he_features = np.memmap(he_scratch, dtype=np.float32, mode="w+", shape=shape)
            ihc_features = np.memmap(ihc_scratch, dtype=np.float32, mode="w+", shape=shape)
            he_features[0] = first_he
            ihc_features[0] = first_ihc

            for index, feature_id in enumerate(ids[1:], start=1):
                he_vector = _load_vector(he_files[feature_id])
                ihc_vector = _load_vector(ihc_files[feature_id])
                if len(he_vector) != feature_dim or len(ihc_vector) != feature_dim:
                    raise PackingError("Feature dimensions are inconsistent across files.")
                he_features[index] = he_vector
                ihc_features[index] = ihc_vector

            he_features.flush()
            ihc_features.flush()
            staged.append(_stage_archive(he_output, he_features, output_ids))
            staged.append(_stage_archive(ihc_output, ihc_features, output_ids))

        # A hard link publishes each complete temporary archive atomically and
        # fails if another process created the output since our preflight check.
        for temporary_path, output in zip(staged, (he_output, ihc_output)):
            os.link(temporary_path, output)
            published.append(output)
    except BaseException:
        for output in published:
            output.unlink(missing_ok=True)
        raise
    finally:
        for temporary_path in staged:
            temporary_path.unlink(missing_ok=True)

    return len(ids), feature_dim


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he-dir", type=Path, required=True)
    parser.add_argument("--ihc-dir", type=Path, required=True)
    parser.add_argument("--he-output", type=Path, required=True)
    parser.add_argument("--ihc-output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        count, dimension = pack_pairs(
            args.he_dir, args.ihc_dir, args.he_output, args.ihc_output
        )
    except (PackingError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Packed {count} paired vectors of dimension {dimension}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
