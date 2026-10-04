from __future__ import annotations

import argparse
import json

from src.compare_trial_conditions import run


def _result(seed: int, subjects: dict[str, float]) -> dict:
    return {
        "dataset": "demo",
        "seed": seed,
        "test": {
            "0.8": {
                "per_subject": {
                    subject: {"balanced_accuracy": value}
                    for subject, value in subjects.items()
                }
            }
        },
    }


def test_compare_trial_conditions_merges_folds_before_pairing(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "fold0.json").write_text(json.dumps(_result(1, {"1": 0.8})))
    (left / "fold1.json").write_text(json.dumps(_result(1, {"2": 0.6})))
    (right / "fold0.json").write_text(json.dumps(_result(1, {"1": 0.5})))
    (right / "fold1.json").write_text(json.dumps(_result(1, {"2": 0.5})))
    output = tmp_path / "summary.json"
    result = run(argparse.Namespace(
        condition=[f"left={left}/*.json", f"right={right}/*.json"],
        contrast=["gain=left-right"],
        metric="balanced_accuracy", seed=1, output=str(output),
    ))
    comparison = result["contrasts"][0]
    assert abs(comparison["left_summary"]["mean"] - 0.7) < 1e-9
    assert abs(comparison["right_summary"]["mean"] - 0.5) < 1e-9
    assert abs(comparison["paired_left_minus_right"]["mean_difference"] - 0.2) < 1e-9


def test_compare_trial_conditions_accepts_multiple_globs_for_one_condition(tmp_path):
    left_a = tmp_path / "left_a"
    left_b = tmp_path / "left_b"
    right = tmp_path / "right"
    left_a.mkdir(); left_b.mkdir(); right.mkdir()
    (left_a / "seed1.json").write_text(json.dumps(_result(1, {"1": 0.8, "2": 0.6})))
    (left_b / "seed2.json").write_text(json.dumps(_result(2, {"1": 0.6, "2": 0.8})))
    (right / "seed1.json").write_text(json.dumps(_result(1, {"1": 0.5, "2": 0.5})))
    (right / "seed2.json").write_text(json.dumps(_result(2, {"1": 0.5, "2": 0.5})))
    result = run(argparse.Namespace(
        condition=[
            f"left={left_a}/*.json", f"left={left_b}/*.json",
            f"right={right}/*.json",
        ],
        contrast=["gain=left-right"], metric="balanced_accuracy",
        seed=1, output=str(tmp_path / "summary.json"),
    ))
    comparison = result["contrasts"][0]
    assert abs(comparison["left_summary"]["mean"] - 0.7) < 1e-9
    assert abs(comparison["paired_left_minus_right"]["mean_difference"] - 0.2) < 1e-9


def test_compare_trial_conditions_uses_only_windows_shared_by_every_condition(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir(); right.mkdir()
    left_result = _result(1, {"1": 0.8, "2": 0.6})
    left_result["test"]["0.4"] = left_result["test"].pop("0.8")
    left_result["test"]["0.8"] = {
        "per_subject": {
            "1": {"balanced_accuracy": 0.9},
            "2": {"balanced_accuracy": 0.7},
        }
    }
    right_result = _result(1, {"1": 0.5, "2": 0.5})
    right_result["test"]["0.4"] = right_result["test"].pop("0.8")
    (left / "result.json").write_text(json.dumps(left_result))
    (right / "result.json").write_text(json.dumps(right_result))

    result = run(argparse.Namespace(
        condition=[f"left={left}/*.json", f"right={right}/*.json"],
        contrast=["gain=left-right"], metric="balanced_accuracy",
        seed=1, output=str(tmp_path / "summary.json"),
    ))

    assert set(result["conditions"]["left"]) == {"0.4"}
    assert [row["window_seconds"] for row in result["contrasts"]] == [0.4]
