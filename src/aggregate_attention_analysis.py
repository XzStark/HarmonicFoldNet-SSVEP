from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .paper_metrics import atomic_write_json


METRICS = (
    "normalized_entropy",
    "peak_distance_hz",
    "peak_within_bias_width_fraction",
    "peak_within_1hz_fraction",
    "harmonic_neighborhood_mass",
)


def _bootstrap_ci(values: np.ndarray, *, seed: int = 20261001) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate(paths: list[Path], output: Path) -> dict:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    contracts = {(row["dataset"], float(row["window_seconds"])) for row in records}
    if len(contracts) != 1:
        raise ValueError(f"attention outputs mix dataset/window contracts: {contracts}")
    dataset, window = next(iter(contracts))
    subject_rows: dict[str, dict[str, list[float]]] = {}
    for row in records:
        present = row["with_harmonic_bias"]["per_subject"]
        removed = row["same_checkpoint_without_harmonic_bias"]["per_subject"]
        for subject in present:
            destination = subject_rows.setdefault(subject, {
                f"with::{metric}": [] for metric in METRICS
            } | {
                f"without::{metric}": [] for metric in METRICS
            })
            for metric in METRICS:
                destination[f"with::{metric}"].append(float(present[subject][metric]))
                destination[f"without::{metric}"].append(float(removed[subject][metric]))

    per_subject = {}
    for subject, values in subject_rows.items():
        per_subject[subject] = {
            key: float(np.mean(samples)) for key, samples in values.items()
        }
        per_subject[subject]["replicates"] = int(len(values[f"with::{METRICS[0]}"]))

    results = {}
    subjects = sorted(per_subject, key=lambda value: (len(value), value))
    for metric in METRICS:
        present = np.asarray([per_subject[s][f"with::{metric}"] for s in subjects])
        removed = np.asarray([per_subject[s][f"without::{metric}"] for s in subjects])
        differences = present - removed
        p_value = 1.0 if np.allclose(differences, 0.0) else float(
            wilcoxon(present, removed, alternative="two-sided").pvalue
        )
        results[metric] = {
            "with_harmonic_bias_mean": float(present.mean()),
            "without_harmonic_bias_mean": float(removed.mean()),
            "with_minus_without_mean": float(differences.mean()),
            "with_minus_without_bootstrap_95ci": _bootstrap_ci(differences),
            "wilcoxon_two_sided_p": p_value,
        }
    payload = {
        "status": "complete",
        "dataset": dataset,
        "window_seconds": window,
        "checkpoints": len(records),
        "unique_subjects": len(subjects),
        "replicated_seeds_are_averaged_before_inference": True,
        "inputs": [str(path.resolve()) for path in paths],
        "metrics": results,
        "per_subject": per_subject,
    }
    atomic_write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(
        [Path(value) for value in args.inputs], Path(args.output),
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
