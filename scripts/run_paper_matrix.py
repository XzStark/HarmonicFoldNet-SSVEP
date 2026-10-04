from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--architectures", nargs="+", default=["legacy_fusion"])
    parser.add_argument("--modes", nargs="+", default=["full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260929, 20260930, 20260931])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--root", type=Path, default=Path("runs/paper_v3/grouped_cv"))
    parser.add_argument("--config", default="configs/paper_multidataset.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--save-checkpoint", action="store_true",
        help="Persist the selected model.pt for downstream adaptation or deployment tests.",
    )
    parser.add_argument("--evidence-mode", choices=("power", "cca_phase", "phase_demod"))
    parser.add_argument("--attention-depth", type=int)
    parser.add_argument(
        "--train-windows", nargs="+", type=float,
        help="Optional repeated training-window schedule forwarded to every job.",
    )
    parser.add_argument(
        "--selection-windows", nargs="+", type=float,
        help="Optional early-stopping validation windows forwarded to every job.",
    )
    parser.add_argument("--train-trial-filter", action="append")
    parser.add_argument("--validation-trial-filter", action="append")
    parser.add_argument("--test-trial-filter", action="append")
    args = parser.parse_args()
    jobs = []
    for dataset in args.datasets:
        for architecture in args.architectures:
            for mode in args.modes:
                if architecture not in {
                    "legacy_fusion", "temporal_fusion", "spectral_fusion",
                    "harmonic_fold_v4",
                    "harmonic_fold_v4_1",
                    "harmonic_fold_v4_2",
                    "harmonic_fold_v4_3",
                } and mode != "full":
                    continue
                for seed in args.seeds:
                    for fold in range(args.folds):
                        run_dir = args.root / dataset / architecture / mode / f"seed-{seed}" / f"fold-{fold}"
                        jobs.append((dataset, architecture, mode, seed, fold, run_dir))
    for index, (dataset, architecture, mode, seed, fold, run_dir) in enumerate(jobs, 1):
        if complete(run_dir / "result.json"):
            print(json.dumps({"job": index, "total": len(jobs), "status": "skipped", "run_dir": str(run_dir)}), flush=True)
            continue
        command = [
            sys.executable, "-m", "src.paper_train", "--config", args.config,
            "--dataset", dataset, "--architecture", architecture, "--mode", mode,
            "--seed", str(seed), "--fold-index", str(fold), "--fold-count", str(args.folds),
            "--run-dir", str(run_dir),
        ]
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        if args.evidence_mode is not None:
            command.extend(["--evidence-mode", args.evidence_mode])
        if args.attention_depth is not None:
            command.extend(["--attention-depth", str(args.attention_depth)])
        if args.train_windows is not None:
            command.append("--train-windows")
            command.extend(map(str, args.train_windows))
        if args.selection_windows is not None:
            command.append("--selection-windows")
            command.extend(map(str, args.selection_windows))
        for destination, values in (
            ("--train-trial-filter", args.train_trial_filter),
            ("--validation-trial-filter", args.validation_trial_filter),
            ("--test-trial-filter", args.test_trial_filter),
        ):
            for value in values or []:
                command.extend([destination, value])
        if args.save_checkpoint:
            command.append("--save-checkpoint")
        print(json.dumps({"job": index, "total": len(jobs), "command": command}), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
