from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "paper_multidataset.yaml"
DEFAULT_RUN_ROOT = ROOT / "runs" / "core_component_evidence_20261009"
SEEDS = (20260929, 20260930, 20260931)
FOLDS = range(5)
CONDITIONS = (
    {"condition": "token_stride_1", "mode": "full", "spectral_token_stride": 1},
    {"condition": "token_stride_4", "mode": "full", "spectral_token_stride": 4},
    {"condition": "no_temporal_candidate", "mode": "no_temporal_candidate"},
    {"condition": "no_spectral_candidate", "mode": "no_spectral_candidate"},
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _registered_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            for fold_index in FOLDS:
                job = dict(condition)
                job.update(
                    {
                        "seed": seed,
                        "fold_index": fold_index,
                        "fold_count": 5,
                        "job_id": (
                            f"{condition['condition']}__seed-{seed}__fold-{fold_index}"
                        ),
                    }
                )
                jobs.append(job)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen BETA token-compression and candidate-path ablation jobs. "
            "The existing full stride-2 matrix is reused and is not retrained here."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--beta-shard-dir", type=Path,
        help="Runtime-only BETA shard location; preserved in the run snapshot.",
    )
    parser.add_argument("--device")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.beta_shard_dir is not None:
        config["datasets"]["beta"]["shard_dir"] = str(
            args.beta_shard_dir.resolve()
        )
    args.run_root.mkdir(parents=True, exist_ok=True)
    runtime_config = args.run_root / "resolved_protocol_config.yaml"
    runtime_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8",
    )
    shard_dir = ROOT / config["datasets"]["beta"]["shard_dir"]
    present = len(list(shard_dir.glob("sub-*.npz"))) if shard_dir.exists() else 0
    jobs = _registered_jobs()
    state: dict[str, Any] = {
        "protocol": "CORE_COMPONENT_EVIDENCE_PROTOCOL_v1",
        "status": "prepared" if args.plan_only else "validating_data",
        "registered_jobs": len(jobs),
        "completed_jobs": 0,
        "existing_beta_shards": present,
        "required_beta_shards": 70,
        "reused_control": {
            "condition": "token_stride_2_full",
            "source": "existing retained five-fold three-seed BETA full-model matrix",
        },
        "jobs": jobs,
    }
    _atomic_json(args.run_root / "queue_state.json", state)
    if args.plan_only:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    missing = [
        str(shard_dir / f"sub-{subject}.npz")
        for subject in range(1, 71)
        if not (shard_dir / f"sub-{subject}.npz").exists()
    ]
    if missing:
        state["status"] = "blocked_missing_data"
        state["missing_shards"] = missing
        _atomic_json(args.run_root / "queue_state.json", state)
        raise FileNotFoundError(missing[0])

    state["status"] = "running"
    completed = 0
    for job in jobs:
        run_dir = args.run_root / job["job_id"]
        result_path = run_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "complete":
                completed += 1
                state["completed_jobs"] = completed
                _atomic_json(args.run_root / "queue_state.json", state)
                continue
        command = [
            sys.executable,
            "-m",
            "src.paper_train",
            "--config",
            str(runtime_config),
            "--dataset",
            "beta",
            "--architecture",
            "harmonic_fold_v4_1",
            "--mode",
            str(job["mode"]),
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
        if "spectral_token_stride" in job:
            command.extend(
                ["--spectral-token-stride", str(job["spectral_token_stride"])]
            )
        if args.device:
            command.extend(["--device", args.device])
        run_dir.mkdir(parents=True, exist_ok=True)
        state["current_job"] = job
        _atomic_json(args.run_root / "queue_state.json", state)
        with (run_dir / "worker.log").open("a", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode != 0:
            state["status"] = "failed"
            state["failed_job"] = job
            state["returncode"] = process.returncode
            _atomic_json(args.run_root / "queue_state.json", state)
            raise SystemExit(process.returncode)
        completed += 1
        state["completed_jobs"] = completed
        _atomic_json(args.run_root / "queue_state.json", state)
    state["status"] = "complete"
    state.pop("current_job", None)
    _atomic_json(args.run_root / "queue_state.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
