from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from .paper_metrics import atomic_write_json
from .paper_statistics import holm_adjust, merge_out_of_fold_documents, paired_test, summarize


def _assignment(value: str, *, separator: str = "=") -> tuple[str, str]:
    name, found, expression = value.partition(separator)
    if not found or not name.strip() or not expression.strip():
        raise ValueError(f"invalid assignment {value!r}")
    return name.strip(), expression.strip()


def _load_condition(expression: str) -> list[dict]:
    paths = sorted(Path(path) for path in glob.glob(expression))
    if not paths:
        raise FileNotFoundError(f"condition expression matched no files: {expression}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def run(args: argparse.Namespace) -> dict[str, object]:
    condition_expressions: dict[str, list[str]] = {}
    for value in args.condition:
        name, expression = _assignment(value)
        condition_expressions.setdefault(name, []).append(expression)
    contrasts = []
    for value in args.contrast:
        label, expression = _assignment(value)
        left, separator, right = expression.partition("-")
        if not separator or left not in condition_expressions or right not in condition_expressions:
            raise ValueError(f"invalid contrast {value!r}")
        contrasts.append((label, left, right))

    documents = {
        name: [
            document
            for expression in expressions
            for document in _load_condition(expression)
        ]
        for name, expressions in condition_expressions.items()
    }
    datasets = {
        str(document["dataset"])
        for rows in documents.values()
        for document in rows
    }
    if len(datasets) != 1:
        raise ValueError(f"conditions contain different datasets: {sorted(datasets)}")
    window_sets = [
        set(document["test"])
        for rows in documents.values()
        for document in rows
    ]
    windows = sorted(set.intersection(*window_sets), key=float)
    if not windows:
        raise ValueError("conditions have no common evaluation windows")
    output: dict[str, object] = {
        "dataset": datasets.pop(),
        "metric": args.metric,
        "aggregation_rule": "merge out-of-fold participants, then average each participant across seeds",
        "conditions": {},
        "contrasts": [],
    }
    matrices: dict[tuple[str, str], tuple[list[str], np.ndarray, list[int]]] = {}
    for name, rows in documents.items():
        condition_rows = {}
        for window in windows:
            subjects, seed_matrix, seeds = merge_out_of_fold_documents(
                rows, window=window, metric=args.metric,
            )
            matrices[(name, window)] = (subjects, seed_matrix, seeds)
            participant_values = seed_matrix.mean(axis=0)
            condition_rows[window] = {
                "participants": subjects,
                "seeds": seeds,
                "summary": summarize(participant_values, seed=args.seed),
                "seed_means": {
                    str(seed): float(seed_matrix[index].mean())
                    for index, seed in enumerate(seeds)
                },
                "mean_within_participant_seed_sd": float(seed_matrix.std(axis=0).mean()),
            }
        output["conditions"][name] = condition_rows

    tests = []
    for label, left, right in contrasts:
        for window in windows:
            left_subjects, left_matrix, _ = matrices[(left, window)]
            right_subjects, right_matrix, _ = matrices[(right, window)]
            if left_subjects != right_subjects:
                raise ValueError(f"participant mismatch for contrast {label}/{window}")
            left_values = left_matrix.mean(axis=0)
            right_values = right_matrix.mean(axis=0)
            tests.append({
                "label": label,
                "left": left,
                "right": right,
                "window_seconds": float(window),
                "left_summary": summarize(left_values, seed=args.seed),
                "right_summary": summarize(right_values, seed=args.seed + 1),
                "paired_left_minus_right": paired_test(left_values, right_values),
            })
    adjusted = holm_adjust([
        float(row["paired_left_minus_right"]["p_value"]) for row in tests
    ])
    for row, p_value in zip(tests, adjusted, strict=True):
        row["paired_left_minus_right"]["holm_adjusted_p_value"] = p_value
    output["contrasts"] = tests
    atomic_write_json(args.output, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", action="append", required=True,
        help="NAME=glob expression selecting out-of-fold result.json files.",
    )
    parser.add_argument(
        "--contrast", action="append", default=[],
        help="LABEL=LEFT-RIGHT using two declared condition names.",
    )
    parser.add_argument("--metric", default="balanced_accuracy")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
