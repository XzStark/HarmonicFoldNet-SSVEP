from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paper_metrics import atomic_write_json
from .paper_statistics import holm_adjust, paired_test, summarize


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _subject_values(
    document: dict,
    *,
    method: str,
    calibration_blocks: str,
    window: str,
    metric: str,
) -> dict[str, float]:
    rows = document["methods"][method][calibration_blocks][window]["per_subject"]
    return {
        str(subject): float(values[metric])
        for subject, values in rows.items()
    }


def _seed_matrix(
    documents: list[dict],
    *,
    method: str,
    calibration_blocks: str,
    window: str,
    metric: str,
) -> tuple[list[str], np.ndarray, list[int]]:
    by_seed: dict[int, dict[str, float]] = {}
    for document in documents:
        seed = int(document["seed"])
        if seed in by_seed:
            raise ValueError(f"duplicate adaptation result for seed {seed}")
        by_seed[seed] = _subject_values(
            document,
            method=method,
            calibration_blocks=calibration_blocks,
            window=window,
            metric=metric,
        )
    seeds = sorted(by_seed)
    subjects = sorted(by_seed[seeds[0]], key=lambda value: (len(value), value))
    expected = set(subjects)
    for seed in seeds:
        if set(by_seed[seed]) != expected:
            raise ValueError(f"participant mismatch in adaptation seed {seed}")
    matrix = np.asarray(
        [[by_seed[seed][subject] for subject in subjects] for seed in seeds],
        dtype=np.float64,
    )
    return subjects, matrix, seeds


def run(args: argparse.Namespace) -> dict[str, object]:
    documents = [_load(path) for path in args.results]
    if not documents:
        raise ValueError("at least one adaptation result is required")
    dataset = str(documents[0]["dataset"])
    if any(str(document["dataset"]) != dataset for document in documents):
        raise ValueError("adaptation results use different datasets")
    classical = _load(args.classical) if args.classical else None
    if classical is not None and str(classical["dataset"]) != dataset:
        raise ValueError("adaptation and classical results use different datasets")

    output: dict[str, object] = {
        "dataset": dataset,
        "metric": args.metric,
        "adaptation_method": args.method,
        "seeds": sorted(int(document["seed"]) for document in documents),
        "aggregation_rule": "average each participant across seeds before paired tests",
        "conditions": [],
        "classical_comparisons": [],
    }
    conditions: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    improvement_tests: list[dict[str, object]] = []
    classical_tests: list[dict[str, object]] = []

    for window in map(str, args.windows):
        zero_subjects, zero_matrix, seeds = _seed_matrix(
            documents,
            method="zero_shot",
            calibration_blocks="0",
            window=window,
            metric=args.metric,
        )
        zero_values = zero_matrix.mean(axis=0)
        conditions.append({
            "calibration_blocks": 0,
            "window_seconds": float(window),
            "participants": zero_subjects,
            "summary": summarize(zero_values, seed=args.seed),
            "seed_means": {
                str(seed): float(zero_matrix[index].mean())
                for index, seed in enumerate(seeds)
            },
            "mean_within_participant_seed_sd": float(zero_matrix.std(axis=0).mean()),
        })

        for calibration_blocks in map(str, args.calibration_blocks):
            subjects, matrix, condition_seeds = _seed_matrix(
                documents,
                method=args.method,
                calibration_blocks=calibration_blocks,
                window=window,
                metric=args.metric,
            )
            if subjects != zero_subjects or condition_seeds != seeds:
                raise ValueError(
                    f"zero-shot/adaptation mismatch for {calibration_blocks}/{window}"
                )
            values = matrix.mean(axis=0)
            improvement = paired_test(values, zero_values)
            condition = {
                "calibration_blocks": int(calibration_blocks),
                "window_seconds": float(window),
                "participants": subjects,
                "summary": summarize(values, seed=args.seed),
                "seed_means": {
                    str(seed): float(matrix[index].mean())
                    for index, seed in enumerate(seeds)
                },
                "mean_within_participant_seed_sd": float(matrix.std(axis=0).mean()),
                "paired_adapted_minus_zero_shot": improvement,
            }
            conditions.append(condition)
            improvement_tests.append(condition)

            if classical is None:
                continue
            for baseline in args.classical_methods:
                if calibration_blocks not in classical["methods"].get(baseline, {}):
                    continue
                baseline_values_by_subject = _subject_values(
                    classical,
                    method=baseline,
                    calibration_blocks=calibration_blocks,
                    window=window,
                    metric=args.metric,
                )
                if set(baseline_values_by_subject) != set(subjects):
                    raise ValueError(
                        f"participant mismatch for {baseline}/{calibration_blocks}/{window}"
                    )
                baseline_values = np.asarray(
                    [baseline_values_by_subject[subject] for subject in subjects],
                    dtype=np.float64,
                )
                comparison = {
                    "calibration_blocks": int(calibration_blocks),
                    "window_seconds": float(window),
                    "baseline": baseline,
                    "model": summarize(values, seed=args.seed),
                    "baseline_summary": summarize(baseline_values, seed=args.seed + 1),
                    "paired_model_minus_baseline": paired_test(values, baseline_values),
                }
                comparisons.append(comparison)
                classical_tests.append(comparison)

    if improvement_tests:
        adjusted = holm_adjust([
            float(row["paired_adapted_minus_zero_shot"]["p_value"])
            for row in improvement_tests
        ])
        for row, p_value in zip(improvement_tests, adjusted, strict=True):
            row["paired_adapted_minus_zero_shot"]["holm_adjusted_p_value"] = p_value
    if classical_tests:
        adjusted = holm_adjust([
            float(row["paired_model_minus_baseline"]["p_value"])
            for row in classical_tests
        ])
        for row, p_value in zip(classical_tests, adjusted, strict=True):
            row["paired_model_minus_baseline"]["holm_adjusted_p_value"] = p_value

    output["conditions"] = conditions
    output["classical_comparisons"] = comparisons
    atomic_write_json(args.output, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--classical")
    parser.add_argument("--method", default="spatial_score")
    parser.add_argument("--classical-methods", nargs="+", default=["trca", "tdca"])
    parser.add_argument("--calibration-blocks", nargs="+", default=["1", "2"])
    parser.add_argument("--windows", nargs="+", default=["0.4", "0.8", "1.2"])
    parser.add_argument("--metric", default="balanced_accuracy")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
