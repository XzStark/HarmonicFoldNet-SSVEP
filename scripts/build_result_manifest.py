from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_complete(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("status") != "complete":
        raise ValueError(f"incomplete result: {path}")
    return document


def validate_dataset(
    root: Path, dataset: str, seeds: list[int], folds: int,
) -> dict[str, object]:
    seed_summaries = []
    reference_fold_tests: list[list[str]] | None = None
    reference_subjects: set[str] | None = None
    protocol_versions: set[int] = set()
    split_seeds: set[int] = set()
    for seed in seeds:
        fold_tests: list[list[str]] = []
        observed: set[str] = set()
        run_paths = []
        for fold in range(folds):
            path = root / dataset / "legacy_fusion" / "full" / f"seed-{seed}" / f"fold-{fold}" / "result.json"
            document = load_complete(path)
            if document["dataset"] != dataset or int(document["seed"]) != seed:
                raise ValueError(f"identity mismatch: {path}")
            expected_split = f"group-{folds}-fold-{fold}"
            if document["split"] != expected_split:
                raise ValueError(f"split mismatch: {path}")
            test_subjects = list(map(str, document["subjects"]["test"]))
            overlap = observed.intersection(test_subjects)
            if overlap:
                raise ValueError(f"duplicate out-of-fold subjects for {dataset}/{seed}: {sorted(overlap)}")
            observed.update(test_subjects)
            fold_tests.append(test_subjects)
            protocol_versions.add(int(document["protocol_version"]))
            split_seeds.add(int(document["split_seed"]))
            run_paths.append(str(path))
        if reference_fold_tests is None:
            reference_fold_tests = fold_tests
            reference_subjects = observed
        elif fold_tests != reference_fold_tests or observed != reference_subjects:
            raise ValueError(f"outer folds changed across training seeds for {dataset}")
        seed_summaries.append({
            "seed": seed,
            "participants": len(observed),
            "result_files": run_paths,
        })
    if len(protocol_versions) != 1 or len(split_seeds) != 1:
        raise ValueError(f"mixed protocol versions or split seeds for {dataset}")
    return {
        "dataset": dataset,
        "protocol_version": next(iter(protocol_versions)),
        "split_seed": next(iter(split_seeds)),
        "folds": folds,
        "outer_test_subjects_by_fold": reference_fold_tests,
        "seeds": seed_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tracked = [
        Path("configs/paper_multidataset.yaml"),
        Path("src/model.py"),
        Path("src/paper_data.py"),
        Path("src/paper_train.py"),
        Path("src/paper_metrics.py"),
        Path("src/paper_statistics.py"),
    ]
    manifest = {
        "status": "complete",
        "result_root": str(args.root),
        "source_sha256": {str(path): sha256(path) for path in tracked},
        "datasets": [
            validate_dataset(args.root, dataset, args.seeds, args.folds)
            for dataset in args.datasets
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": "complete", "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
