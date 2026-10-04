from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.build_submission_evidence import (
    DEFAULT_CONFIG,
    _bootstrap_difference_ci,
    run,
)


def test_paired_bootstrap_interval_is_deterministic() -> None:
    left = np.asarray([0.2, 0.4, 0.6, 0.8])
    right = np.asarray([0.1, 0.3, 0.5, 0.7])
    first = _bootstrap_difference_ci(left, right, seed=17, samples=2_000)
    second = _bootstrap_difference_ci(left, right, seed=17, samples=2_000)
    assert first == second
    assert np.allclose(first, (0.1, 0.1))


def test_submission_evidence_rebuilds_from_preserved_runs(tmp_path: Path) -> None:
    output = tmp_path / "source_data"
    report = tmp_path / "report.md"
    bundle = run(DEFAULT_CONFIG, output, report, update_figure_data=False)

    assert bundle["main"]["audit"]["validation"] == "passed"
    assert bundle["ablations"]["audit"]["validation"] == "passed"
    assert bundle["mtsnet"]["audit"]["tests"] == 6
    assert len(bundle["ablations"]["contrasts"]) == 18
    assert bundle["split_manifest"]["status"] == "passed"
    assert bundle["split_manifest"]["rows"] == 210
    assert len(bundle["split_manifest"]["checks"]) == 4
    assert bundle["deployment"]["status"] == "passed"
    assert bundle["deployment"]["measurements"] == 12
    assert (output / "split_manifest.csv").exists()
    assert (output / "split_manifest_audit.json").exists()
    assert (output / "deployment_measurements.csv").exists()
    assert (output / "deployment_checkpoint_manifest.csv").exists()
    assert report.exists()

    with (output / "main_participant_metrics_seed_averaged.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    beta_itr = [
        row
        for row in rows
        if row["dataset"] == "beta"
        and row["condition"] == "harmonic_fold"
        and row["window_s"] == "1.2"
    ]
    assert len(beta_itr) == 70
    assert all(float(row["itr_bits_per_minute"]) >= 0.0 for row in beta_itr)
