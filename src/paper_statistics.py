from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

from .paper_metrics import atomic_write_json


def summarize(values: np.ndarray, *, seed: int, bootstrap_samples: int = 10_000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    return {
        "participants": int(len(values)),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "bootstrap_mean_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "bootstrap_mean_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
    }


def paired_test(model: np.ndarray, baseline: np.ndarray) -> dict[str, float | int | None]:
    differences = np.asarray(model, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    nonzero = differences != 0
    if not nonzero.any():
        return {"participants": len(differences), "p_value": 1.0, "rank_biserial": 0.0}
    test = wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto")
    absolute_ranks = rankdata(np.abs(differences[nonzero]))
    positive = absolute_ranks[differences[nonzero] > 0].sum()
    negative = absolute_ranks[differences[nonzero] < 0].sum()
    denominator = positive + negative
    effect = (positive - negative) / denominator if denominator else 0.0
    return {
        "participants": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "p_value": float(test.pvalue),
        "rank_biserial": float(effect),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_out_of_fold_documents(
    documents: list[dict], *, window: str, metric: str,
) -> tuple[list[str], np.ndarray, list[int]]:
    """Return one out-of-fold value per participant for every training seed."""
    by_seed: dict[int, dict[str, float]] = {}
    for document in documents:
        seed = int(document["seed"])
        destination = by_seed.setdefault(seed, {})
        values = document["test"][window]["per_subject"]
        overlap = set(destination).intersection(values)
        if overlap:
            raise ValueError(
                f"duplicate out-of-fold subjects for seed {seed}: {sorted(overlap)}"
            )
        destination.update(
            {str(subject): float(row[metric]) for subject, row in values.items()}
        )
    seeds = sorted(by_seed)
    subjects = sorted(by_seed[seeds[0]], key=lambda value: (len(value), value))
    expected = set(subjects)
    for seed in seeds:
        if set(by_seed[seed]) != expected:
            raise ValueError(
                f"incomplete or inconsistent out-of-fold subjects for seed {seed}"
            )
    matrix = np.asarray(
        [[by_seed[seed][subject] for subject in subjects] for seed in seeds],
        dtype=np.float64,
    )
    return subjects, matrix, seeds


def run(args: argparse.Namespace) -> dict[str, object]:
    model_documents = [_load(Path(path)) for path in args.model_results]
    baseline_document = _load(Path(args.baseline_result))
    first = model_documents[0]
    windows = list(first["test"])
    metrics = args.metrics
    output: dict[str, object] = {
        "dataset": first["dataset"], "model_runs": len(model_documents),
        "model_seeds": sorted(set(int(document["seed"]) for document in model_documents)),
        "comparisons": [],
    }
    tests: list[dict[str, object]] = []
    for window in windows:
        for metric in metrics:
            subjects, seed_values, _ = merge_out_of_fold_documents(
                model_documents, window=window, metric=metric,
            )
            model_values = seed_values.mean(axis=0)
            for method in args.baseline_methods:
                baseline_subjects = baseline_document["methods"][method][window]["per_subject"]
                common = [subject for subject in subjects if subject in baseline_subjects]
                positions = [subjects.index(subject) for subject in common]
                compared_model = model_values[positions]
                baseline_values = np.asarray(
                    [baseline_subjects[subject][metric] for subject in common], dtype=np.float64,
                )
                comparison = {
                    "window_seconds": float(window), "metric": metric, "baseline": method,
                    "model": summarize(compared_model, seed=args.seed),
                    "baseline_summary": summarize(baseline_values, seed=args.seed + 1),
                    "paired": paired_test(compared_model, baseline_values),
                    "seed_standard_deviation_mean": float(seed_values[:, positions].std(axis=0).mean()),
                }
                tests.append(comparison)
    adjusted = holm_adjust([float(item["paired"]["p_value"]) for item in tests])
    for item, p_adjusted in zip(tests, adjusted, strict=True):
        item["paired"]["holm_adjusted_p_value"] = p_adjusted
    output["comparisons"] = tests
    atomic_write_json(args.output, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-results", nargs="+", required=True)
    parser.add_argument("--baseline-result", required=True)
    parser.add_argument("--baseline-methods", nargs="+", default=["harmonic", "cca", "fbcca"])
    parser.add_argument("--metrics", nargs="+", default=["accuracy", "balanced_accuracy", "itr_bits_per_minute"])
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
