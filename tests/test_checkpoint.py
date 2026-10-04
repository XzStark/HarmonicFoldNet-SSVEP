import tempfile
import unittest
from pathlib import Path

import torch

from src.checkpoint import export_deploy_checkpoint, load_checkpoint_model
from src.model import LegacyFusionNet


class CheckpointTests(unittest.TestCase):
    def test_training_and_deploy_checkpoints_load(self):
        config = {
            "sample_rate": 250,
            "model": {
                "width": 16, "local_depth": 1, "attention_depth": 1,
                "heads": 4, "dropout": 0.0, "spectral_bins": 32,
                "spectral_band_hz": [6.0, 45.0],
            },
        }
        model = LegacyFusionNet(
            channels=8, classes=4, sample_rate=250,
            class_frequencies=[8.0, 10.0, 12.0, 14.0], **config["model"],
        ).eval()
        probe = torch.randn(2, 8, 500)
        expected = model(probe)
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.pt"
            deploy_path = Path(directory) / "deploy.pt"
            torch.save({"state_dict": model.state_dict(), "config": config}, train_path)
            loaded, _ = load_checkpoint_model(train_path)
            torch.testing.assert_close(loaded(probe), expected)
            export_deploy_checkpoint(train_path, deploy_path)
            deploy, metadata = load_checkpoint_model(deploy_path)
            self.assertTrue(metadata["deploy"])
            torch.testing.assert_close(deploy(probe), expected, rtol=2e-4, atol=2e-5)
