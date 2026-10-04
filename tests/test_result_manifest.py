import json
from pathlib import Path

import pytest

from scripts.build_result_manifest import validate_dataset


def _write_result(path: Path, *, dataset: str, seed: int, fold: int, tests: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "status": "complete",
        "dataset": dataset,
        "seed": seed,
        "split": f"group-2-fold-{fold}",
        "protocol_version": 4,
        "split_seed": 7,
        "subjects": {"test": tests},
    }), encoding="utf-8")


def test_manifest_accepts_fixed_disjoint_outer_folds(tmp_path: Path) -> None:
    for seed in (1, 2):
        _write_result(
            tmp_path / "demo" / "legacy_fusion" / "full" / f"seed-{seed}" / "fold-0" / "result.json",
            dataset="demo", seed=seed, fold=0, tests=["1", "2"],
        )
        _write_result(
            tmp_path / "demo" / "legacy_fusion" / "full" / f"seed-{seed}" / "fold-1" / "result.json",
            dataset="demo", seed=seed, fold=1, tests=["3", "4"],
        )
    result = validate_dataset(tmp_path, "demo", [1, 2], 2)
    assert result["outer_test_subjects_by_fold"] == [["1", "2"], ["3", "4"]]


def test_manifest_rejects_duplicate_outer_test_subject(tmp_path: Path) -> None:
    _write_result(
        tmp_path / "demo" / "legacy_fusion" / "full" / "seed-1" / "fold-0" / "result.json",
        dataset="demo", seed=1, fold=0, tests=["1", "2"],
    )
    _write_result(
        tmp_path / "demo" / "legacy_fusion" / "full" / "seed-1" / "fold-1" / "result.json",
        dataset="demo", seed=1, fold=1, tests=["2", "3"],
    )
    with pytest.raises(ValueError, match="duplicate out-of-fold"):
        validate_dataset(tmp_path, "demo", [1], 2)
