from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _complete(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8"))["status"] == "complete"
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def _selection_score(path: Path, method: str) -> float:
    document = json.loads(path.read_text(encoding="utf-8"))
    values = []
    for count in ("1", "2"):
        for window in ("0.8", "1.2"):
            subjects = document["methods"][method][count][window]["per_subject"]
            values.extend(row["balanced_accuracy"] for row in subjects.values())
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_multidataset.yaml")
    parser.add_argument("--dataset", default="kim2025")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=["spatial", "spatial_score"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.003, 0.01])
    parser.add_argument("--epochs", nargs="+", type=int, default=[3, 10, 30])
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--device")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in args.methods:
        for learning_rate in args.learning_rates:
            for epochs in args.epochs:
                slug = f"{method}_lr-{learning_rate:g}_epochs-{epochs}"
                output = args.output_root / f"{slug}.json"
                if not _complete(output):
                    command = [
                        sys.executable, "-m", "src.subject_adaptation_eval",
                        "--config", args.config, "--dataset", args.dataset,
                        "--checkpoint-root", args.checkpoint_root,
                        "--seed", str(args.seed), "--subjects", *args.subjects,
                        "--methods", method, "--calibration-blocks", "1", "2",
                        "--windows", "0.8", "1.2",
                        "--train-windows", "0.4", "0.8", "1.2",
                        "--epochs", str(epochs), "--learning-rate", str(learning_rate),
                        "--identity-penalty", "0.01", "--output", str(output),
                    ]
                    if args.device:
                        command.extend(["--device", args.device])
                    print(json.dumps({"status": "running", "configuration": slug}), flush=True)
                    subprocess.run(command, check=True)
                rows.append({
                    "method": method, "learning_rate": learning_rate,
                    "epochs": epochs, "selection_score": _selection_score(output, method),
                    "output": str(output),
                })
    rows.sort(key=lambda row: (
        -row["selection_score"],
        0 if row["method"] == "spatial" else 1,
        row["learning_rate"], row["epochs"],
    ))
    summary = {
        "status": "complete", "selection_dataset": args.dataset,
        "subjects": args.subjects,
        "selection_metric": (
            "mean participant balanced accuracy over 0.8/1.2 s and "
            "one/two calibration blocks"
        ),
        "winner": rows[0], "configurations": rows,
    }
    destination = args.output_root / "selection.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(summary["winner"], indent=2), flush=True)


if __name__ == "__main__":
    main()
