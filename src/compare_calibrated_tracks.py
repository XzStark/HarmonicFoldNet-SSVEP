from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paper_metrics import atomic_write_json
from .paper_statistics import holm_adjust, paired_test, summarize


def _participant_values(
    document: dict, method: str, count: str, window: str, metric: str,
) -> dict[str, float]:
    return {
        str(subject): float(row[metric])
        for subject, row in document["methods"][method][count][window]["per_subject"].items()
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    neural = json.loads(Path(args.neural).read_text(encoding="utf-8"))
    classical = json.loads(Path(args.classical).read_text(encoding="utf-8"))
    if neural["dataset"] != classical["dataset"]:
        raise ValueError("neural and classical results use different datasets")
    output: dict[str, object] = {
        "dataset": neural["dataset"], "metric": args.metric,
        "neural_method": args.neural_method, "comparisons": [],
    }
    tests: list[dict[str, object]] = []
    for count in map(str, args.calibration_blocks):
        for window in map(str, args.windows):
            model = _participant_values(
                neural, args.neural_method, count, window, args.metric,
            )
            for baseline_method in args.classical_methods:
                if count not in classical["methods"].get(baseline_method, {}):
                    continue
                baseline = _participant_values(
                    classical, baseline_method, count, window, args.metric,
                )
                if set(model) != set(baseline):
                    raise ValueError(
                        f"participant mismatch for {baseline_method}/{count}/{window}"
                    )
                subjects = sorted(model, key=lambda value: (len(value), value))
                model_values = np.asarray([model[subject] for subject in subjects])
                baseline_values = np.asarray([baseline[subject] for subject in subjects])
                tests.append({
                    "calibration_blocks": int(count),
                    "window_seconds": float(window),
                    "baseline": baseline_method,
                    "participants": subjects,
                    "model": summarize(model_values, seed=args.seed),
                    "baseline_summary": summarize(baseline_values, seed=args.seed + 1),
                    "paired_model_minus_baseline": paired_test(model_values, baseline_values),
                })
    adjusted = holm_adjust([
        float(row["paired_model_minus_baseline"]["p_value"]) for row in tests
    ])
    for row, value in zip(tests, adjusted, strict=True):
        row["paired_model_minus_baseline"]["holm_adjusted_p_value"] = value
    output["comparisons"] = tests
    atomic_write_json(args.output, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural", required=True)
    parser.add_argument("--classical", required=True)
    parser.add_argument("--neural-method", required=True)
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
