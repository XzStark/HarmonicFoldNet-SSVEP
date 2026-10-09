from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen external MTSNet protocol matrix serially."
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--windows", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/paper_multidataset.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--source",
        default=".research_refs/MTSNet/MTSNet.py",
        help="Pinned path to the authors' unmodified MTSNet.py.",
    )
    args = parser.parse_args()

    jobs = [
        (dataset, window, seed, fold)
        for dataset in args.datasets
        for window in args.windows
        for seed in args.seeds
        for fold in range(args.folds)
    ]
    for index, (dataset, window, seed, fold) in enumerate(jobs, start=1):
        window_name = f"{window:g}s"
        run_dir = (
            Path(args.root) / dataset / window_name / f"seed-{seed}" / f"fold-{fold}"
        )
        command = [
            sys.executable,
            "-m",
            "src.external_mtsnet_train",
            "--config",
            args.config,
            "--dataset",
            dataset,
            "--window",
            str(window),
            "--seed",
            str(seed),
            "--fold-index",
            str(fold),
            "--fold-count",
            str(args.folds),
            "--device",
            args.device,
            "--source",
            args.source,
            "--run-dir",
            str(run_dir),
        ]
        print(
            json.dumps(
                {"job": index, "total": len(jobs), "command": command},
                ensure_ascii=False,
            ),
            flush=True,
        )
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
