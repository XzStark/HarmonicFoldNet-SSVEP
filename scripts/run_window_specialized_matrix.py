from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matched window-specialized HarmonicFold matrices.",
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--windows", nargs="+", type=float, required=True)
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[20260929, 20260930, 20260931],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--root", type=Path, default=Path("runs/paper_v19/window_specialized"),
    )
    parser.add_argument("--config", default="configs/paper_multidataset.yaml")
    args = parser.parse_args()

    for window in args.windows:
        destination = args.root / f"{window:g}s"
        command = [
            sys.executable,
            "scripts/run_paper_matrix.py",
            "--datasets", *args.datasets,
            "--architectures", "harmonic_fold_v4_1",
            "--modes", "full",
            "--seeds", *map(str, args.seeds),
            "--folds", str(args.folds),
            "--root", str(destination),
            "--config", args.config,
            "--train-windows", str(window),
            "--selection-windows", str(window),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
