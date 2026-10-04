from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paper_metrics import atomic_write_json
from .paper_statistics import holm_adjust, merge_out_of_fold_documents, paired_test, summarize


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, object]:
    full_documents = [_load(path) for path in args.full_results]
    ablation_documents = [_load(path) for path in args.ablation_results]
    full_first = full_documents[0]
    ablation_first = ablation_documents[0]
    if full_first["dataset"] != ablation_first["dataset"]:
        raise ValueError("full and ablation results must use the same dataset")

    windows = sorted(
        set(full_first["test"]).intersection(ablation_first["test"]),
        key=float,
    )
    comparisons: list[dict[str, object]] = []
    for window in windows:
        for metric in args.metrics:
            full_subjects, full_by_seed, full_seeds = merge_out_of_fold_documents(
                full_documents, window=window, metric=metric,
            )
            ablation_subjects, ablation_by_seed, ablation_seeds = merge_out_of_fold_documents(
                ablation_documents, window=window, metric=metric,
            )
            common = sorted(
                set(full_subjects).intersection(ablation_subjects),
                key=lambda value: (len(value), value),
            )
            if not common:
                raise ValueError(f"no common out-of-fold subjects for window {window}")
            full_index = {subject: index for index, subject in enumerate(full_subjects)}
            ablation_index = {subject: index for index, subject in enumerate(ablation_subjects)}
            full_values = np.asarray(
                [full_by_seed[:, full_index[subject]].mean() for subject in common],
                dtype=np.float64,
            )
            ablation_values = np.asarray(
                [ablation_by_seed[:, ablation_index[subject]].mean() for subject in common],
                dtype=np.float64,
            )
            comparisons.append({
                "window_seconds": float(window),
                "metric": metric,
                "participants": len(common),
                "full": summarize(full_values, seed=args.seed),
                "ablation": summarize(ablation_values, seed=args.seed + 1),
                "paired_full_minus_ablation": paired_test(full_values, ablation_values),
                "full_seeds": full_seeds,
                "ablation_seeds": ablation_seeds,
            })

    adjusted = holm_adjust([
        float(row["paired_full_minus_ablation"]["p_value"])
        for row in comparisons
    ])
    for row, adjusted_p in zip(comparisons, adjusted, strict=True):
        row["paired_full_minus_ablation"]["holm_adjusted_p_value"] = adjusted_p

    output: dict[str, object] = {
        "dataset": full_first["dataset"],
        "full_architecture": full_first["architecture"],
        "full_mode": full_first["mode"],
        "ablation_architecture": ablation_first["architecture"],
        "ablation_mode": ablation_first["mode"],
        "comparisons": comparisons,
    }
    atomic_write_json(args.output, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-results", nargs="+", required=True)
    parser.add_argument("--ablation-results", nargs="+", required=True)
    parser.add_argument(
        "--metrics", nargs="+",
        default=["accuracy", "balanced_accuracy", "itr_bits_per_minute"],
    )
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
