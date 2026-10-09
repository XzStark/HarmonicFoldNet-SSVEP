from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper_metrics import atomic_write_json
from src.paper_statistics import (
    holm_adjust,
    merge_out_of_fold_documents,
    paired_test,
    summarize,
)


PRINCIPAL_WINDOWS = ("0.4", "0.8", "1.2")


def _load_many(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _participant_values(
    documents: list[dict[str, Any]], window: str, metric: str,
) -> tuple[list[str], np.ndarray, list[int]]:
    subjects, by_seed, seeds = merge_out_of_fold_documents(
        documents, window=window, metric=metric,
    )
    return subjects, by_seed.mean(axis=0), seeds


def _comparison(
    retained: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    window: str,
    label: str,
) -> dict[str, Any]:
    subjects_a, values_a, seeds_a = _participant_values(
        retained, window, "balanced_accuracy",
    )
    subjects_b, values_b, seeds_b = _participant_values(
        candidate, window, "balanced_accuracy",
    )
    if subjects_a != subjects_b or seeds_a != seeds_b:
        raise RuntimeError(f"mismatched paired evidence for {label} at {window}s")
    return {
        "label": label,
        "window_seconds": float(window),
        "participants": len(subjects_a),
        "seeds": seeds_a,
        "retained": summarize(values_a, seed=20260929),
        "candidate": summarize(values_b, seed=20260930),
        "retained_minus_candidate": paired_test(
            values_a, values_b, seed=20260931,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--control-result-pattern", action="append", required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control_paths = sorted(
        {
            Path(path)
            for pattern in args.control_result_pattern
            for path in glob.glob(pattern)
        }
    )
    if len(control_paths) != 15:
        raise RuntimeError(f"expected 15 retained control results, found {len(control_paths)}")
    controls = _load_many(control_paths)
    conditions = {
        name: _load_many(sorted(args.run_root.glob(f"{name}__*/result.json")))
        for name in (
            "token_stride_1", "token_stride_4",
            "no_temporal_candidate", "no_spectral_candidate",
        )
    }
    for name, documents in conditions.items():
        if len(documents) != 15:
            raise RuntimeError(f"expected 15 {name} results, found {len(documents)}")

    stride_rows = []
    for window in PRINCIPAL_WINDOWS:
        stride_rows.append(
            _comparison(
                controls, conditions["token_stride_1"], window=window,
                label="stride_2_minus_stride_1",
            )
        )
        stride_rows.append(
            _comparison(
                controls, conditions["token_stride_4"], window=window,
                label="stride_2_minus_stride_4",
            )
        )
    adjusted = holm_adjust(
        [float(row["retained_minus_candidate"]["p_value"]) for row in stride_rows]
    )
    for row, value in zip(stride_rows, adjusted, strict=True):
        row["retained_minus_candidate"]["holm_adjusted_p_value"] = value

    windows = sorted(controls[0]["test"], key=float)
    path_rows = []
    for window in windows:
        path_rows.append(
            _comparison(
                controls, conditions["no_temporal_candidate"], window=window,
                label="full_minus_spectral_only_temporal_contribution",
            )
        )
        path_rows.append(
            _comparison(
                controls, conditions["no_spectral_candidate"], window=window,
                label="full_minus_temporal_only_spectral_contribution",
            )
        )
    adjusted = holm_adjust(
        [float(row["retained_minus_candidate"]["p_value"]) for row in path_rows]
    )
    for row, value in zip(path_rows, adjusted, strict=True):
        evidence = row["retained_minus_candidate"]
        evidence["holm_adjusted_p_value"] = value
        row["contribution_supported"] = bool(
            float(evidence["mean_difference"]) > 0.0
            and float(evidence["bootstrap_mean_difference_ci95_low"]) > 0.0
            and value < 0.05
        )

    output = {
        "status": "complete",
        "dataset": "beta",
        "independent_unit": "participant",
        "seed_aggregation": "mean within participant before paired inference",
        "control_results": [str(path) for path in control_paths],
        "stride_accuracy_comparisons": stride_rows,
        "candidate_path_comparisons": path_rows,
    }
    atomic_write_json(args.output, output)


if __name__ == "__main__":
    main()
