from __future__ import annotations

import argparse
import json

from src.aggregate_subject_adaptation import run


def _adaptation(seed: int, zero: dict[str, float], adapted: dict[str, float]) -> dict:
    def rows(values: dict[str, float]) -> dict:
        return {
            subject: {"balanced_accuracy": value}
            for subject, value in values.items()
        }

    return {
        "dataset": "demo",
        "seed": seed,
        "methods": {
            "zero_shot": {"0": {"0.8": {"per_subject": rows(zero)}}},
            "spatial_score": {"1": {"0.8": {"per_subject": rows(adapted)}}},
        },
    }


def test_aggregate_subject_adaptation_averages_seeds_before_pairing(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    classical = tmp_path / "classical.json"
    output = tmp_path / "output.json"
    first.write_text(json.dumps(_adaptation(1, {"1": 0.4, "2": 0.6}, {"1": 0.7, "2": 0.8})))
    second.write_text(json.dumps(_adaptation(2, {"1": 0.6, "2": 0.4}, {"1": 0.9, "2": 0.6})))
    classical.write_text(json.dumps({
        "dataset": "demo",
        "methods": {
            "tdca": {
                "1": {
                    "0.8": {
                        "per_subject": {
                            "1": {"balanced_accuracy": 0.6},
                            "2": {"balanced_accuracy": 0.5},
                        }
                    }
                }
            }
        },
    }))
    result = run(argparse.Namespace(
        results=[str(first), str(second)],
        classical=str(classical),
        method="spatial_score",
        classical_methods=["tdca"],
        calibration_blocks=["1"],
        windows=["0.8"],
        metric="balanced_accuracy",
        seed=1,
        output=str(output),
    ))
    zero = result["conditions"][0]
    adapted = result["conditions"][1]
    comparison = result["classical_comparisons"][0]
    assert abs(zero["summary"]["mean"] - 0.5) < 1e-9
    assert abs(adapted["summary"]["mean"] - 0.75) < 1e-9
    assert abs(
        adapted["paired_adapted_minus_zero_shot"]["mean_difference"] - 0.25
    ) < 1e-9
    assert abs(
        comparison["paired_model_minus_baseline"]["mean_difference"] - 0.2
    ) < 1e-9
