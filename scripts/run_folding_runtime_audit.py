from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _aggregate(run_root: Path, windows: list[float], sessions: int) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for session in range(1, sessions + 1):
        for window in windows:
            path = run_root / f"session-{session}" / f"window-{window:.1f}.json"
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    summary: dict[str, Any] = {
        "status": "complete",
        "sessions": sessions,
        "windows_seconds": windows,
        "decision_rule": (
            "deployment p50 and p95 must both be lower in at least four of five "
            "fresh-process runs for a backend/window speedup claim"
        ),
        "equivalence": {
            "maximum_abs_error": max(
                float(report["equivalence_max_abs_error"]) for report in reports
            ),
            "all_passed": True,
        },
        "results": {},
    }
    for window in windows:
        selected = [
            report for report in reports
            if abs(float(report["window_seconds"]) - window) < 1e-9
        ]
        window_result: dict[str, Any] = {}
        for backend in ("cpu", "cuda"):
            if backend not in selected[0]["training_graph"]:
                continue
            rows = []
            wins = 0
            for report in selected:
                training = report["training_graph"][backend]
                deployment = report["deployment_graph"][backend]
                p50_change = 100.0 * (
                    float(deployment["p50_ms"]) / float(training["p50_ms"]) - 1.0
                )
                p95_change = 100.0 * (
                    float(deployment["p95_ms"]) / float(training["p95_ms"]) - 1.0
                )
                faster = (
                    float(deployment["p50_ms"]) < float(training["p50_ms"])
                    and float(deployment["p95_ms"]) < float(training["p95_ms"])
                )
                wins += int(faster)
                rows.append(
                    {
                        "graph_order": report["graph_order"],
                        "training_p50_ms": float(training["p50_ms"]),
                        "training_p95_ms": float(training["p95_ms"]),
                        "deployment_p50_ms": float(deployment["p50_ms"]),
                        "deployment_p95_ms": float(deployment["p95_ms"]),
                        "deployment_change_p50_percent": p50_change,
                        "deployment_change_p95_percent": p95_change,
                        "deployment_faster_on_both": faster,
                    }
                )
            window_result[backend] = {
                "paired_runs_with_both_lower": wins,
                "speedup_claim_supported": wins >= 4,
                "median_deployment_change_p50_percent": statistics.median(
                    row["deployment_change_p50_percent"] for row in rows
                ),
                "median_deployment_change_p95_percent": statistics.median(
                    row["deployment_change_p95_percent"] for row in rows
                ),
                "runs": rows,
            }
        window_result["parameters"] = selected[0]["parameters"]
        window_result["module_inventory"] = selected[0]["module_inventory"]
        summary["results"][f"{window:.1f}"] = window_result
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--windows", nargs="+", type=float, default=[0.4, 0.8, 1.2])
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "status": "running",
        "checkpoint": str(args.checkpoint.resolve()),
        "windows_seconds": args.windows,
        "sessions": args.sessions,
        "completed_runs": 0,
        "registered_runs": args.sessions * len(args.windows),
    }
    _atomic_json(args.run_root / "state.json", state)
    for session in range(1, args.sessions + 1):
        order = "training_first" if session % 2 else "deployment_first"
        for window in args.windows:
            output = args.run_root / f"session-{session}" / f"window-{window:.1f}.json"
            if output.exists():
                state["completed_runs"] += 1
                _atomic_json(args.run_root / "state.json", state)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            state["current_run"] = {
                "session": session,
                "window_seconds": window,
                "graph_order": order,
            }
            _atomic_json(args.run_root / "state.json", state)
            command = [
                sys.executable, "-m", "src.benchmark_deployment",
                "--checkpoint", str(args.checkpoint),
                "--window", str(window),
                "--warmup", str(args.warmup),
                "--iterations", str(args.iterations),
                "--graph-order", order,
                "--output", str(output),
            ]
            with (output.parent / "worker.log").open("a", encoding="utf-8") as log:
                process = subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    check=False,
                )
            if process.returncode:
                state["status"] = "failed"
                state["returncode"] = process.returncode
                _atomic_json(args.run_root / "state.json", state)
                raise SystemExit(process.returncode)
            state["completed_runs"] += 1
            _atomic_json(args.run_root / "state.json", state)
    summary = _aggregate(args.run_root, args.windows, args.sessions)
    _atomic_json(args.run_root / "summary.json", summary)
    state["status"] = "complete"
    state.pop("current_run", None)
    _atomic_json(args.run_root / "state.json", state)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
