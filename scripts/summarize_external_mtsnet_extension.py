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


SUPPORTED_DATASETS = ("dong2023",)
WINDOWS = ("0.4", "0.8", "1.2")
EXPECTED_PARTICIPANTS = {"dong2023": 59}


def _read(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("status") != "complete":
        raise RuntimeError(f"incomplete result: {path}")
    return document


def _harmonicfoldnet_documents(root: Path) -> list[dict]:
    paths = sorted(root.glob("harmonic_fold_v4_1__*/result.json"))
    if len(paths) != 15:
        raise RuntimeError(f"expected 15 HarmonicFoldNet results in {root}, found {len(paths)}")
    return [_read(path) for path in paths]


def _mtsnet_documents(root: Path, dataset: str, window: str) -> list[dict]:
    paths = sorted((root / dataset / f"{window}s").glob("seed-*/fold-*/result.json"))
    if len(paths) != 15:
        raise RuntimeError(
            f"expected 15 MTSNet results for {dataset} {window}s, found {len(paths)}"
        )
    documents = [_read(path) for path in paths]
    commits = {document.get("upstream_commit") for document in documents}
    if commits != {"890b0a4f93c3affd50741a1f1d036fb74a30a062"}:
        raise RuntimeError(f"unexpected MTSNet provenance for {dataset} {window}s: {commits}")
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--dong2023-harmonicfoldnet-root", type=Path)
    parser.add_argument("--mtsnet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    harmonic_roots: dict[str, Path | None] = {
        "dong2023": args.dong2023_harmonicfoldnet_root,
    }
    rows = []
    for dataset_index, dataset in enumerate(args.datasets):
        harmonic_root = harmonic_roots[dataset]
        if harmonic_root is None:
            parser.error(f"--{dataset}-harmonicfoldnet-root is required for {dataset}")
        harmonic_documents = _harmonicfoldnet_documents(harmonic_root)
        for window_index, window in enumerate(WINDOWS):
            mtsnet_documents = _mtsnet_documents(args.mtsnet_root, dataset, window)
            metric_rows = {}
            for metric_index, metric in enumerate(("accuracy", "balanced_accuracy")):
                harmonic_subjects, harmonic_by_seed, harmonic_seeds = (
                    merge_out_of_fold_documents(
                        harmonic_documents, window=window, metric=metric,
                    )
                )
                mtsnet_subjects, mtsnet_by_seed, mtsnet_seeds = merge_out_of_fold_documents(
                    mtsnet_documents, window=window, metric=metric,
                )
                if harmonic_subjects != mtsnet_subjects:
                    raise RuntimeError(f"participant mismatch for {dataset} {window}s")
                expected = EXPECTED_PARTICIPANTS[dataset]
                if len(harmonic_subjects) != expected:
                    raise RuntimeError(
                        f"expected {expected} participants for {dataset}, "
                        f"found {len(harmonic_subjects)}"
                    )
                harmonic_values = harmonic_by_seed.mean(axis=0)
                mtsnet_values = mtsnet_by_seed.mean(axis=0)
                seed = 20261009 + dataset_index * 100 + window_index * 10 + metric_index
                metric_rows[metric] = {
                    "harmonicfoldnet": summarize(harmonic_values, seed=seed),
                    "mtsnet": summarize(mtsnet_values, seed=seed + 1),
                    "harmonicfoldnet_minus_mtsnet": paired_test(
                        harmonic_values, mtsnet_values, seed=seed + 2,
                    ),
                    "harmonicfoldnet_seed_sd_mean": float(
                        harmonic_by_seed.std(axis=0).mean()
                    ),
                    "mtsnet_seed_sd_mean": float(mtsnet_by_seed.std(axis=0).mean()),
                }
            rows.append(
                {
                    "dataset": dataset,
                    "window_seconds": float(window),
                    "participants": EXPECTED_PARTICIPANTS[dataset],
                    "harmonicfoldnet_seeds": harmonic_seeds,
                    "mtsnet_seeds": mtsnet_seeds,
                    "metrics": metric_rows,
                }
            )

    adjusted = holm_adjust(
        [
            float(row["metrics"]["accuracy"]["harmonicfoldnet_minus_mtsnet"]["p_value"])
            for row in rows
        ]
    )
    for row, value in zip(rows, adjusted, strict=True):
        row["metrics"]["accuracy"]["harmonicfoldnet_minus_mtsnet"][
            "holm_adjusted_p_value"
        ] = value

    atomic_write_json(
        args.output,
        {
            "status": "complete",
            "protocol": "EXTERNAL_BASELINE_EXTENSION_PROTOCOL_v1",
            "analysis_status": "post_result_supplementary",
            "independent_unit": "participant",
            "seed_aggregation": "mean within participant before paired inference",
            "primary_metric": "accuracy",
            "multiple_testing": (
                f"Holm correction across {len(args.datasets)} dataset(s) by three windows"
            ),
            "mtsnet_upstream_commit": "890b0a4f93c3affd50741a1f1d036fb74a30a062",
            "comparisons": rows,
        },
    )


if __name__ == "__main__":
    main()
