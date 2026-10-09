from __future__ import annotations

import torch

from src.benchmark_deployment import _leaf_module_inventory, _metadata_from_checkpoint
from src.harmonic_fold import HarmonicFoldNet
from src.model import reparameterize_model


def _model() -> HarmonicFoldNet:
    return HarmonicFoldNet(
        channels=2,
        sample_rate=250,
        class_frequencies=[8.0, 10.0, 12.0],
        class_phases=[0.0, 0.5, 1.0],
        width=8,
        heads=2,
        local_depths=(1, 1),
        harmonics=2,
    ).eval()


def test_metadata_can_be_recovered_from_checkpoint_buffers() -> None:
    frequencies, phases = _metadata_from_checkpoint(_model().state_dict(), classes=3)
    assert frequencies == [8.0, 10.0, 12.0]
    assert torch.allclose(torch.tensor(phases), torch.tensor([0.0, 0.5, 1.0]))


def test_leaf_inventory_detects_folding_reduction() -> None:
    training = _leaf_module_inventory(_model())
    deployment = _leaf_module_inventory(reparameterize_model(_model()))
    assert training["leaf_module_count"] > deployment["leaf_module_count"]
    assert training["by_type"].get("BatchNorm1d", 0) > 0
    assert (
        training["by_type"].get("BatchNorm1d", 0)
        > deployment["by_type"].get("BatchNorm1d", 0)
    )
