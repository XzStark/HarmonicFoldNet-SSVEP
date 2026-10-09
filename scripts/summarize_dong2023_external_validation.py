from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper_metrics import atomic_write_json
from src.paper_statistics import holm_adjust, merge_out_of_fold_documents, paired_test, summarize


def _documents(run_root: Path, architecture: str) -> list[dict]:
    paths = sorted(run_root.glob(f"{architecture}__*/result.json"))
    if len(paths) != 15:
        raise RuntimeError(f"expected 15 {architecture} results, found {len(paths)}")
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(document.get("status") != "complete" for document in documents):
        raise RuntimeError(f"incomplete {architecture} result encountered")
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_documents = _documents(args.run_root, "harmonic_fold_v4_1")
    baseline_documents = _documents(args.run_root, "ssvepformer")
    windows = sorted(model_documents[0]["test"], key=float)
    comparisons = []
    for index, window in enumerate(windows):
        model_subjects, model_by_seed, model_seeds = merge_out_of_fold_documents(
            model_documents, window=window, metric="balanced_accuracy"
        )
        baseline_subjects, baseline_by_seed, baseline_seeds = merge_out_of_fold_documents(
            baseline_documents, window=window, metric="balanced_accuracy"
        )
        if model_subjects != baseline_subjects or model_seeds != baseline_seeds:
            raise RuntimeError(f"mismatched paired evidence at {window}s")
        if len(model_subjects) != 59:
            raise RuntimeError(f"expected 59 participants at {window}s, found {len(model_subjects)}")
        model_values = model_by_seed.mean(axis=0)
        baseline_values = baseline_by_seed.mean(axis=0)
        comparisons.append(
            {
                "window_seconds": float(window),
                "participants": len(model_subjects),
                "seeds": model_seeds,
                "harmonicfoldnet": summarize(model_values, seed=20261009 + index),
                "ssvepformer": summarize(baseline_values, seed=20261019 + index),
                "harmonicfoldnet_minus_ssvepformer": paired_test(
                    model_values, baseline_values, seed=20261029 + index
                ),
                "harmonicfoldnet_seed_sd_mean": float(model_by_seed.std(axis=0).mean()),
                "ssvepformer_seed_sd_mean": float(baseline_by_seed.std(axis=0).mean()),
            }
        )
    adjusted = holm_adjust(
        [float(row["harmonicfoldnet_minus_ssvepformer"]["p_value"]) for row in comparisons]
    )
    for row, value in zip(comparisons, adjusted, strict=True):
        row["harmonicfoldnet_minus_ssvepformer"]["holm_adjusted_p_value"] = value

    atomic_write_json(
        args.output,
        {
            "status": "complete",
            "dataset": "dong2023",
            "protocol": "DONG2023_UNTOUCHED_EXTERNAL_VALIDATION_v1",
            "independent_unit": "participant",
            "participants": 59,
            "folds": 5,
            "seeds": [20260929, 20260930, 20260931],
            "metric": "balanced_accuracy",
            "seed_aggregation": "mean within participant before paired inference",
            "multiple_testing": "Holm correction across six windows",
            "comparisons": comparisons,
        },
    )


if __name__ == "__main__":
    main()
