"""Regression checks for inference output and file-preservation boundaries."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from featstaindiff.infer import build_parser, run
from featstaindiff.diffusion import DiffusionConfig
from featstaindiff.model import FeatStainDiff, ModelConfig


class InferenceReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config = ModelConfig(
            feature_dim=16, hidden_dim=8, depth=2, heads=2,
            patch_size=8, semantic_depth=1, num_experts=2,
        )
        model = FeatStainDiff(config)
        self.checkpoint = self.root / "checkpoint.pt"
        torch.save({
            "model_config": asdict(config),
            "diffusion_config": asdict(DiffusionConfig()),
            "model_state": model.state_dict(),
            "ema_state": model.state_dict(),
        }, self.checkpoint)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.source = self.inputs / "a.pt"
        self.features = torch.ones(3, 16)
        self.ids = ["patch-a", "patch-b", "patch-c"]
        torch.save({"features": self.features, "ids": self.ids}, self.source)

    def arguments(self, source: Path, output: Path, *flags: str):
        return build_parser().parse_args([
            "--checkpoint", str(self.checkpoint), "--input", str(source),
            "--output", str(output), "--device", "cpu", "--steps", "1",
            "--ensemble", "1", *flags,
        ])

    @staticmethod
    def zeros(model, he, *args, **kwargs):
        return torch.zeros_like(he)

    def test_dotted_directory_and_optional_row_ids(self) -> None:
        with patch("featstaindiff.infer.ddim_sample", side_effect=self.zeros):
            output_dir = self.root / "generated.v1"
            run(self.arguments(self.inputs, output_dir))
            output = torch.load(output_dir / "a.pt", weights_only=True)
            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(output.shape, self.features.shape)

            with_ids = self.root / "with-ids.pt"
            run(self.arguments(self.source, with_ids, "--preserve-ids"))
            output = torch.load(with_ids, weights_only=True)
            self.assertEqual(output["ids"], self.ids)
            torch.testing.assert_close(output["features"], torch.zeros_like(self.features))

            bare = self.root / "bare.pt"
            torch.save(self.features, bare)
            bare_output = self.root / "bare-output.pt"
            run(self.arguments(bare, bare_output, "--preserve-ids"))
            self.assertIsInstance(torch.load(bare_output, weights_only=True), torch.Tensor)

    def test_inputs_and_checkpoint_cannot_be_replaced_through_aliases(self) -> None:
        for protected in (self.source, self.checkpoint):
            original_bytes = protected.read_bytes()
            hardlink = self.root / f"hardlink-{protected.name}"
            symlink = self.root / f"symlink-{protected.name}"
            os.link(protected, hardlink)
            symlink.symlink_to(protected)
            for destination in (protected, hardlink, symlink):
                with self.subTest(protected=protected.name, output=destination.name):
                    with self.assertRaisesRegex(ValueError, "must not replace"):
                        run(self.arguments(self.source, destination, "--overwrite"))
                    self.assertEqual(protected.read_bytes(), original_bytes)
                    self.assertEqual(destination.read_bytes(), original_bytes)

    def test_directory_outputs_cannot_replace_another_input(self) -> None:
        other_source = self.inputs / "b.pt"
        torch.save(torch.full_like(self.features, 2), other_source)
        original_bytes = other_source.read_bytes()
        output_dir = self.root / "generated"
        output_dir.mkdir()
        (output_dir / "a.pt").symlink_to(other_source)
        with self.assertRaisesRegex(ValueError, "must not replace"):
            run(self.arguments(self.inputs, output_dir, "--overwrite"))
        self.assertEqual(other_source.read_bytes(), original_bytes)
        self.assertTrue((output_dir / "a.pt").is_symlink())
        self.assertFalse((output_dir / "b.pt").exists())

    def test_dangling_output_symlink_is_not_overwritten_by_default(self) -> None:
        missing_target = self.root / "missing.pt"
        output = self.root / "dangling.pt"
        output.symlink_to(missing_target)
        with self.assertRaises(FileExistsError):
            run(self.arguments(self.source, output))
        self.assertTrue(output.is_symlink())
        self.assertEqual(output.readlink(), missing_target)
        self.assertFalse(missing_target.exists())

    def test_nonfinite_predictions_are_never_published(self) -> None:
        for index, invalid in enumerate((float("nan"), float("inf"))):
            output = self.root / f"invalid-{index}.pt"
            with self.subTest(value=invalid):
                with patch("featstaindiff.infer.ddim_sample",
                           return_value=torch.full_like(self.features, invalid)):
                    with self.assertRaises(FloatingPointError):
                        run(self.arguments(self.source, output))
                self.assertFalse(output.exists())
        original_bytes = b"previous output must survive a failed replacement"
        output = self.root / "existing.pt"
        output.write_bytes(original_bytes)
        with patch("featstaindiff.infer.ddim_sample",
                   return_value=torch.full_like(self.features, float("nan"))):
            with self.assertRaises(FloatingPointError):
                run(self.arguments(self.source, output, "--overwrite"))
        self.assertEqual(output.read_bytes(), original_bytes)
        self.assertFalse(list(self.root.rglob("*.tmp")))

    def test_residual_overflow_is_not_published(self) -> None:
        maximum = torch.finfo(torch.float32).max
        huge = torch.full_like(self.features, maximum)
        torch.save(huge, self.source)
        output = self.root / "overflow.pt"
        with patch("featstaindiff.infer.ddim_sample", return_value=huge):
            with self.assertRaises(FloatingPointError):
                run(self.arguments(self.source, output, "--combine-with-input"))
        self.assertFalse(output.exists())

    def test_new_file_created_during_sampling_is_not_overwritten(self) -> None:
        output = self.root / "raced.pt"
        original_bytes = b"file created after output preflight"

        def create_output(model, he, *args, **kwargs):
            output.write_bytes(original_bytes)
            return torch.zeros_like(he)

        with patch("featstaindiff.infer.ddim_sample", side_effect=create_output):
            with self.assertRaises(FileExistsError):
                run(self.arguments(self.source, output))
        self.assertEqual(output.read_bytes(), original_bytes)
        self.assertFalse(list(self.root.rglob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
