"""Safe loading and pair validation for pre-extracted feature matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _parse_payload(payload: Any, path: Path) -> tuple[Tensor, list[str] | None]:
    ids = None
    if isinstance(payload, dict):
        for key in ("features", "embeddings"):
            if key in payload:
                features = payload[key]
                break
        else:
            raise ValueError(f"{path}: expected a 'features' or 'embeddings' entry")
        if "ids" in payload:
            ids = [str(value) for value in payload["ids"]]
    else:
        features = payload
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    if not isinstance(features, Tensor) or features.ndim != 2:
        raise ValueError(f"{path}: expected a 2D feature tensor")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"{path}: feature matrix must have at least one row and one feature")
    if not torch.is_floating_point(features):
        raise ValueError(f"{path}: feature matrix must have floating-point dtype")
    features = features.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(features).all():
        raise ValueError(f"{path}: feature matrix contains NaN or infinity")
    if ids is not None and len(ids) != len(features):
        raise ValueError(f"{path}: ids length differs from feature rows")
    if ids is not None and len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate ids are not permitted")
    return features, ids


def load_features(path: str | Path) -> tuple[Tensor, list[str] | None]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix in (".pt", ".pth"):
        # weights_only rejects arbitrary pickle objects in untrusted files.
        payload = torch.load(path, map_location="cpu", weights_only=True)
    elif path.suffix == ".npy":
        payload = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "features" not in archive:
                raise ValueError(f"{path}: .npz must contain 'features'")
            payload = {"features": archive["features"]}
            if "ids" in archive:
                payload["ids"] = archive["ids"].astype(str).tolist()
    else:
        raise ValueError(f"{path}: supported feature files are .pt, .pth, .npy and .npz")
    return _parse_payload(payload, path)


class PairedFeatures(Dataset[tuple[Tensor, Tensor]]):
    """Row-matched H&E and IHC vectors from the same extractor and regions."""

    def __init__(
        self,
        he_path: str | Path,
        ihc_path: str | Path,
        *,
        assume_aligned_order: bool = False,
    ) -> None:
        self.he, he_ids = load_features(he_path)
        self.ihc, ihc_ids = load_features(ihc_path)
        if self.he.shape != self.ihc.shape:
            raise ValueError(f"paired feature shapes differ: {tuple(self.he.shape)} vs {tuple(self.ihc.shape)}")
        if len(self.he) < 2:
            raise ValueError("at least two pairs are needed for contrastive training")
        if (he_ids is None) != (ihc_ids is None):
            raise ValueError("both files must provide ids, or neither may provide ids")
        if he_ids is not None and he_ids != ihc_ids:
            raise ValueError("H&E and IHC ids must match in exactly the same order")
        if he_ids is None and not assume_aligned_order:
            raise ValueError(
                "feature files have no ids; verify registration and row order, "
                "then pass --assume-aligned-order"
            )

    def __len__(self) -> int:
        return len(self.he)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.he[index], self.ihc[index]
