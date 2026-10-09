from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "paper_multidataset.yaml"
DEFAULT_RUN_ROOT = ROOT / "runs" / "dong2023_external_validation_20261009"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _jobs(
    architectures: list[str], seeds: list[int], fold_count: int,
) -> list[dict[str, Any]]:
    return [
        {
            "architecture": architecture,
            "seed": int(seed),
            "fold_index": fold_index,
            "fold_count": fold_count,
            "job_id": f"{architecture}__seed-{seed}__fold-{fold_index}",
        }
        for architecture in architectures
        for seed in seeds
        for fold_index in range(fold_count)
    ]


def _validate_shards(shard_dir: Path) -> dict[str, Any]:
    expected_channels = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
    expected_frequencies = np.arange(8.0, 16.0, 0.2, dtype=np.float32)
    expected_phases = (np.arange(40, dtype=np.float32) % 4) * (0.5 * np.pi)
    for subject in range(1, 60):
        path = shard_dir / f"sub-{subject}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as shard:
            x = np.asarray(shard["x"])
            y = np.asarray(shard["y"])
            blocks = np.asarray(shard["block"])
            if x.shape != (160, 8, 960):
                raise RuntimeError(f"{path}: unexpected x shape {x.shape}")
            if not np.isfinite(x).all():
                raise RuntimeError(f"{path}: non-finite samples")
            if np.bincount(y, minlength=40).tolist() != [4] * 40:
                raise RuntimeError(f"{path}: class imbalance")
            if sorted(np.unique(blocks).tolist()) != list(range(4)):
                raise RuntimeError(f"{path}: invalid block coverage")
            if [str(value) for value in shard["channels"]] != expected_channels:
                raise RuntimeError(f"{path}: channel contract mismatch")
            frequencies = np.asarray(shard["frequency_hz"])[::4]
            phases = np.asarray(shard["phase_rad"])[::4]
            if not np.allclose(frequencies, expected_frequencies, atol=1e-5):
                raise RuntimeError(f"{path}: frequency mapping mismatch")
            if not np.allclose(phases, expected_phases, atol=1e-5):
                raise RuntimeError(f"{path}: phase mapping mismatch")
    return {
        "participants": 59,
        "trials": 9440,
        "classes": 40,
        "channels": expected_channels,
        "samples_per_trial": 960,
        "finite": True,
        "class_metadata_consistent": True,
        "source_snr_domain": "lower_than_tsinghua_benchmark_per_source_paper",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen Dong2023 external-validation queue without retries."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--architectures", nargs="+",
        default=["harmonic_fold_v4_1", "ssvepformer"],
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[20260929, 20260930, 20260931],
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--device")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    shard_dir = ROOT / config["datasets"]["dong2023"]["shard_dir"]
    jobs = _jobs(args.architectures, args.seeds, args.fold_count)
    existing_shards = len(list(shard_dir.glob("sub-*.npz"))) if shard_dir.exists() else 0
    queue_state: dict[str, Any] = {
        "protocol": "DONG2023_UNTOUCHED_EXTERNAL_VALIDATION_v1",
        "status": "prepared" if args.plan_only else "validating_data",
        "registered_jobs": len(jobs),
        "completed_jobs": 0,
        "existing_shards": existing_shards,
        "required_shards": 59,
        "jobs": jobs,
    }
    _atomic_json(args.run_root / "queue_state.json", queue_state)
    if args.plan_only:
        print(json.dumps(queue_state, ensure_ascii=False, indent=2))
        return

    integrity = _validate_shards(shard_dir)
    _atomic_json(args.run_root / "data_integrity.json", integrity)
    queue_state["status"] = "running"
    completed = 0
    for job in jobs:
        run_dir = args.run_root / job["job_id"]
        result_path = run_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "complete":
                completed += 1
                queue_state["completed_jobs"] = completed
                _atomic_json(args.run_root / "queue_state.json", queue_state)
                continue
        command = [
            sys.executable,
            "-m",
            "src.paper_train",
            "--config",
            str(args.config),
            "--dataset",
            "dong2023",
            "--architecture",
            str(job["architecture"]),
            "--mode",
            "full",
            "--seed",
            str(job["seed"]),
            "--fold-index",
            str(job["fold_index"]),
            "--fold-count",
            str(job["fold_count"]),
            "--run-dir",
            str(run_dir),
            "--save-checkpoint",
        ]
        if args.device:
            command.extend(["--device", args.device])
        run_dir.mkdir(parents=True, exist_ok=True)
        queue_state["current_job"] = job
        _atomic_json(args.run_root / "queue_state.json", queue_state)
        with (run_dir / "worker.log").open("a", encoding="utf-8") as log:
            completed_process = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed_process.returncode != 0:
            queue_state["status"] = "failed"
            queue_state["failed_job"] = job
            queue_state["returncode"] = completed_process.returncode
            _atomic_json(args.run_root / "queue_state.json", queue_state)
            raise SystemExit(completed_process.returncode)
        completed += 1
        queue_state["completed_jobs"] = completed
        _atomic_json(args.run_root / "queue_state.json", queue_state)

    queue_state["status"] = "complete"
    queue_state.pop("current_job", None)
    _atomic_json(args.run_root / "queue_state.json", queue_state)
    print(json.dumps(queue_state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
