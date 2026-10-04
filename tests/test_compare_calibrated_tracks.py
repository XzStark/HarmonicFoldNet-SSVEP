from __future__ import annotations

import argparse
import json

from src.compare_calibrated_tracks import run


def _document(dataset: str, method: str, values: dict[str, float]) -> dict:
    return {
        "dataset": dataset,
        "methods": {
            method: {
                "1": {
                    "0.8": {
                        "per_subject": {
                            subject: {"balanced_accuracy": value}
                            for subject, value in values.items()
                        }
                    }
                }
            }
        },
    }


def test_compare_calibrated_tracks_pairs_the_same_participants(tmp_path):
    neural = tmp_path / "neural.json"
    classical = tmp_path / "classical.json"
    output = tmp_path / "comparison.json"
    neural.write_text(json.dumps(_document("demo", "adapter", {"1": 0.8, "2": 0.7})))
    classical.write_text(json.dumps(_document("demo", "tdca", {"1": 0.6, "2": 0.5})))
    result = run(argparse.Namespace(
        neural=str(neural), classical=str(classical), neural_method="adapter",
        classical_methods=["tdca"], calibration_blocks=["1"], windows=["0.8"],
        metric="balanced_accuracy", seed=1, output=str(output),
    ))
    row = result["comparisons"][0]
    assert row["participants"] == ["1", "2"]
    assert abs(row["paired_model_minus_baseline"]["mean_difference"] - 0.2) < 1e-9
