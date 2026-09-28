"""Small CPU checks for paired data, training signals, and feature sampling."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from featstaindiff.data import PairedFeatures, load_features
from featstaindiff.diffusion import DiffusionConfig, ddim_sample, training_losses
from featstaindiff.model import FeatStainDiff, ModelConfig
from featstaindiff.train import build_parser, train


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_pair_ids_must_match_and_be_present_in_both_files(self) -> None:
        features = torch.arange(128, dtype=torch.float32).reshape(4, 32)
        ids = [f"patch-{index}" for index in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            he_path = root / "he.pt"
            ihc_path = root / "ihc.pt"
            torch.save({"features": features, "ids": ids}, he_path)

            torch.save({"features": features, "ids": ids[::-1]}, ihc_path)
            with self.assertRaisesRegex(ValueError, "exactly the same order"):
                PairedFeatures(he_path, ihc_path)

            torch.save(features, ihc_path)
            with self.assertRaisesRegex(ValueError, "both files must provide ids"):
                PairedFeatures(he_path, ihc_path)

            torch.save({"features": features, "ids": ids}, ihc_path)
            self.assertEqual(len(PairedFeatures(he_path, ihc_path)), 4)

    def test_invalid_features_and_numerical_settings_fail_early(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.pt"
            for features in (torch.empty(0, 16), torch.empty(2, 0),
                             torch.full((2, 16), float("nan"))):
                with self.subTest(shape=tuple(features.shape)):
                    torch.save(features, path)
                    with self.assertRaises(ValueError):
                        load_features(path)
        for options in ({"hidden_dim": 1, "heads": 1},
                        {"contrastive_temperature": float("nan")},
                        {"mmd_bandwidth": float("inf")}):
            with self.subTest(model=options), self.assertRaises(ValueError):
                ModelConfig(**options)
        for options in ({"beta_min": float("nan")},
                        {"beta_max": float("inf")},
                        {"beta_min": 1000.0, "beta_max": 1000.0}):
            with self.subTest(diffusion=options), self.assertRaises(ValueError):
                DiffusionConfig(**options)

    def test_launcher_uses_active_environment_by_default(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            fake_conda = Path(directory) / "conda"
            fake_conda.write_text("#!/bin/sh\necho unexpected-conda-activation >&2\nexit 57\n")
            fake_conda.chmod(0o755)
            env = dict(os.environ)
            env.pop("CONDA_ENV", None)
            env.pop("SKIP_CONDA_ACTIVATE", None)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PATH"] = os.pathsep.join((directory, str(Path(sys.executable).parent), env["PATH"]))
            result = subprocess.run(
                ["bash", str(repository / "train.sh"), "--help"], env=env,
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--he-features", result.stdout)
            self.assertNotIn("unexpected-conda-activation", result.stderr)

    def test_four_losses_reach_their_modules(self) -> None:
        torch.manual_seed(7)
        model = FeatStainDiff(ModelConfig(
            feature_dim=32, hidden_dim=16, depth=2, heads=4,
            patch_size=8, semantic_depth=1, num_experts=4,
        ))
        he = torch.randn(4, 32)
        ihc = 0.6 * he + 0.4 * torch.randn_like(he)
        losses = training_losses(model, he, ihc, DiffusionConfig())
        self.assertEqual(set(losses), {"total", "noise", "contrastive", "mmd", "load"})
        for name, loss in losses.items():
            with self.subTest(loss=name):
                self.assertEqual(loss.ndim, 0)
                self.assertTrue(torch.isfinite(loss).item())

        checks = (
            ("noise", model.decoder.weight, model.semantic_skips[0].weight),
            ("contrastive", model.he_encoder.blocks[0].attn.in_proj_weight,
             model.ihc_encoder.blocks[0].attn.in_proj_weight),
            ("mmd", model.he_encoder.blocks[0].attn.in_proj_weight,
             model.ihc_encoder.blocks[0].attn.in_proj_weight),
            ("load", model.fmoe.time_gate.weight, model.fmoe.exploration),
        )
        for index, (name, *parameters) in enumerate(checks):
            model.zero_grad(set_to_none=True)
            losses[name].backward(retain_graph=index < len(checks) - 1)
            for parameter in parameters:
                with self.subTest(loss=name, parameter=tuple(parameter.shape)):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_time_gate_and_ddim_output(self) -> None:
        torch.manual_seed(13)
        model = FeatStainDiff(ModelConfig(
            feature_dim=32, hidden_dim=16, depth=2, heads=4,
            patch_size=8, semantic_depth=1, num_experts=4,
        ))
        he = torch.randn(3, 32)
        noisy = torch.randn_like(he)
        with torch.no_grad():
            early = model(noisy, he, torch.zeros(3))["gates"]
            late = model(noisy, he, torch.ones(3))["gates"]
        self.assertEqual(tuple(early.shape), (3, 4))
        torch.testing.assert_close(early.sum(dim=-1), torch.ones(3))
        self.assertGreater((early - late).abs().max().item(), 1e-8)

        generated = ddim_sample(model, he, DiffusionConfig(), steps=2, ensemble=2, seed=5)
        self.assertEqual(tuple(generated.shape), tuple(he.shape))
        self.assertTrue(torch.isfinite(generated).all().item())


    @unittest.skipUnless(
        os.environ.get("FEATSTAINDIFF_TEST_CUDA") == "1" and torch.cuda.is_available(),
        "set FEATSTAINDIFF_TEST_CUDA=1 to run optional GPU checks",
    )
    def test_mixed_precision_checkpoint_counts_completed_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = torch.Generator().manual_seed(9)
            ids = [str(index) for index in range(4)]
            for stain in ("he", "ihc"):
                torch.save({"features": torch.randn(4, 32, generator=generator), "ids": ids},
                           root / f"{stain}.pt")
            for amp in ("fp16", "bf16"):
                with self.subTest(amp=amp):
                    if amp == "bf16" and not torch.cuda.is_bf16_supported():
                        continue
                    args = build_parser().parse_args([
                        "--he-features", str(root / "he.pt"), "--ihc-features", str(root / "ihc.pt"),
                        "--output-dir", str(root / amp), "--max-steps", "1", "--batch-size", "2",
                        "--feature-dim", "32", "--hidden-dim", "16", "--depth", "2",
                        "--heads", "4", "--patch-size", "8", "--semantic-depth", "1",
                        "--device", "cuda", "--amp", amp,
                    ])
                    checkpoint = torch.load(train(args), map_location="cpu", weights_only=True)
                    states = checkpoint["optimizer_state"]["state"]
                    self.assertTrue(states, "GradScaler skipped the optimizer update")
                    self.assertTrue(all(int(state["step"]) == 1 for state in states.values()))
                    self.assertTrue(all(torch.isfinite(value).all()
                                        for value in checkpoint["model_state"].values()))

    def test_resumed_training_matches_uninterrupted_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = torch.Generator().manual_seed(19)
            he = torch.randn(4, 16, generator=generator)
            ihc = 0.5 * he + 0.5 * torch.randn(4, 16, generator=generator)
            ids = [f"patch-{index}" for index in range(4)]
            he_path, ihc_path = root / "he.pt", root / "ihc.pt"
            torch.save({"features": he, "ids": ids}, he_path)
            torch.save({"features": ihc, "ids": ids}, ihc_path)

            def args(output: Path, steps: int, resume: Path | None = None):
                arguments = [
                    "--he-features", str(he_path), "--ihc-features", str(ihc_path),
                    "--output-dir", str(output), "--max-steps", str(steps),
                    "--batch-size", "2", "--feature-dim", "16", "--hidden-dim", "8",
                    "--depth", "2", "--heads", "2", "--patch-size", "8",
                    "--semantic-depth", "1", "--num-experts", "2", "--device", "cpu",
                    "--save-every", "1", "--log-every", "2", "--seed", "23",
                ]
                if resume is not None:
                    arguments += ["--resume", str(resume)]
                return build_parser().parse_args(arguments)

            full_checkpoint = train(args(root / "full", 2))
            split_checkpoint = train(args(root / "split", 1))
            resumed_checkpoint = train(args(root / "split", 2, split_checkpoint))
            changed_seed = args(root / "split", 3, split_checkpoint)
            changed_seed.seed += 1
            with self.assertRaisesRegex(ValueError, "seed differs"):
                train(changed_seed)
            protected_checkpoint = full_checkpoint.read_bytes()
            protected_config = (root / "full" / "config.json").read_bytes()
            with self.assertRaisesRegex(FileExistsError, "another run"):
                train(args(root / "full", 3, resumed_checkpoint))
            self.assertEqual(full_checkpoint.read_bytes(), protected_checkpoint)
            self.assertEqual((root / "full" / "config.json").read_bytes(), protected_config)
            full = torch.load(full_checkpoint, map_location="cpu", weights_only=True)
            resumed = torch.load(resumed_checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual((full["step"], resumed["step"]), (2, 2))
            for state_name in ("model_state", "ema_state"):
                self.assertEqual(full[state_name].keys(), resumed[state_name].keys())
                for name in full[state_name]:
                    with self.subTest(state=state_name, parameter=name):
                        torch.testing.assert_close(
                            full[state_name][name], resumed[state_name][name], rtol=0, atol=0
                        )


if __name__ == "__main__":
    unittest.main()
