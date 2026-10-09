from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-state", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    state: dict[str, Any] = {"status": "waiting_for_core_component_evidence"}
    _atomic_json(args.state, state)
    while True:
        if args.core_state.exists():
            core = json.loads(args.core_state.read_text(encoding="utf-8"))
            if core.get("status") == "complete":
                break
            if core.get("status") == "failed":
                state["status"] = "blocked_core_component_failed"
                _atomic_json(args.state, state)
                raise SystemExit(2)
        time.sleep(args.poll_seconds)
    state["status"] = "summarizing_core_component_accuracy"
    _atomic_json(args.state, state)
    summary_command = [
        sys.executable,
        str(ROOT / "scripts" / "summarize_core_component_evidence.py"),
        "--run-root", str(args.core_state.parent),
        "--control-result-pattern",
        str(
            args.research_root
            / "runs/paper_v9/external_grouped_cv_v4_1/beta/harmonic_fold_v4_1/full/seed-20260929/fold-*/result.json"
        ),
        "--control-result-pattern",
        str(
            args.research_root
            / "runs/paper_v12/grouped_cv_v4_1_multiseed/beta/harmonic_fold_v4_1/full/seed-2026093*/fold-*/result.json"
        ),
        "--output", str(args.core_state.parent / "core_accuracy_summary.json"),
    ]
    log_path = args.state.with_name("core_accuracy_summary.log")
    with log_path.open("a", encoding="utf-8") as log:
        summary_process = subprocess.run(
            summary_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            check=False,
        )
    if summary_process.returncode:
        state["status"] = "failed_core_component_summary"
        state["returncode"] = int(summary_process.returncode)
        _atomic_json(args.state, state)
        raise SystemExit(summary_process.returncode)

    state["status"] = "running_stride_efficiency"
    _atomic_json(args.state, state)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_stride_efficiency_audit.py"),
        "--checkpoint", str(args.checkpoint),
        "--run-root", str(args.run_root),
    ]
    log_path = args.state.with_name("stride_efficiency.log")
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    state["status"] = "complete" if process.returncode == 0 else "failed"
    state["returncode"] = int(process.returncode)
    _atomic_json(args.state, state)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
