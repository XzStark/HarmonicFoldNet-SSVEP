from __future__ import annotations

import json

import torch

from src.harmonic_fold import HarmonicFoldNet
from src.subject_adaptation_eval import SpatialScoreAdapter, _checkpoint_index


def _model() -> HarmonicFoldNet:
    return HarmonicFoldNet(
        channels=2, sample_rate=100,
        class_frequencies=[8.0, 10.0], class_phases=[0.0, 0.0],
        width=8, local_depths=(1, 1), heads=2, harmonics=2,
        neighborhood_bins=1, spectral_resolution_hz=1.0,
        spectral_band_hz=(6.0, 30.0), harmonic_bias_width_hz=1.0,
        local_domain="temporal", dropout=0.0,
    ).eval()


def test_adapter_is_exact_identity_at_initialization():
    base = _model()
    adapter = SpatialScoreAdapter(base, 2, tune_score=False).eval()
    x = torch.randn(3, 2, 100)
    with torch.inference_mode():
        expected = base(x)
        actual = adapter(x)
    torch.testing.assert_close(actual, expected)
    assert adapter.participant_parameter_count == 4
    assert [name for name, value in adapter.named_parameters() if value.requires_grad] == [
        "channel_delta",
    ]


def test_score_adapter_only_unfreezes_shared_last_layer():
    adapter = SpatialScoreAdapter(_model(), 2, tune_score=True)
    trainable = {name for name, value in adapter.named_parameters() if value.requires_grad}
    assert trainable == {
        "channel_delta", "base.score_head.4.weight", "base.score_head.4.bias",
    }
    assert adapter.participant_parameter_count == 13


def test_outer_checkpoint_index_rejects_duplicate_subject(tmp_path):
    root = tmp_path / "demo" / "harmonic_fold_v4_1" / "full" / "seed-1"
    for fold in (0, 1):
        directory = root / f"fold-{fold}"
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps({
            "subjects": {"train": ["2"], "test": ["1"]},
        }), encoding="utf-8")
        (directory / "model.pt").write_bytes(b"checkpoint")
    try:
        _checkpoint_index(tmp_path, "demo", 1)
    except RuntimeError as exc:
        assert "multiple outer folds" in str(exc)
    else:
        raise AssertionError("duplicate subject should be rejected")
