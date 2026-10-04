import argparse
import json

import pytest

from src.compare_ablations import run


def _document(dataset: str, mode: str, seed: int, subject: str, value: float) -> dict:
    return {
        "dataset": dataset,
        "architecture": "legacy_fusion",
        "mode": mode,
        "seed": seed,
        "test": {
            "0.4": {
                "per_subject": {
                    subject: {
                        "accuracy": value,
                        "balanced_accuracy": value,
                        "itr_bits_per_minute": value * 10,
                    }
                }
            }
        },
    }


def _write(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def test_compare_ablation_uses_out_of_fold_participants(tmp_path):
    full = [
        _write(tmp_path / "full0.json", _document("demo", "full", 1, "s1", 0.8)),
        _write(tmp_path / "full1.json", _document("demo", "full", 1, "s2", 0.6)),
    ]
    ablation = [
        _write(tmp_path / "abl0.json", _document("demo", "time_only", 1, "s1", 0.5)),
        _write(tmp_path / "abl1.json", _document("demo", "time_only", 1, "s2", 0.4)),
    ]
    output = tmp_path / "comparison.json"
    result = run(argparse.Namespace(
        full_results=full,
        ablation_results=ablation,
        metrics=["balanced_accuracy"],
        seed=7,
        output=str(output),
    ))
    row = result["comparisons"][0]
    assert row["participants"] == 2
    assert row["full"]["mean"] == pytest.approx(0.7)
    assert row["ablation"]["mean"] == pytest.approx(0.45)
    assert row["paired_full_minus_ablation"]["mean_difference"] == pytest.approx(0.25)
    assert json.loads(output.read_text(encoding="utf-8"))["ablation_mode"] == "time_only"


def test_compare_ablation_rejects_mixed_datasets(tmp_path):
    full = _write(tmp_path / "full.json", _document("a", "full", 1, "s1", 0.8))
    ablation = _write(tmp_path / "abl.json", _document("b", "time_only", 1, "s1", 0.5))
    with pytest.raises(ValueError, match="same dataset"):
        run(argparse.Namespace(
            full_results=[full],
            ablation_results=[ablation],
            metrics=["balanced_accuracy"],
            seed=7,
            output=str(tmp_path / "out.json"),
        ))
