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
ORDERS = ([1, 2, 4], [2, 4, 1], [4, 1, 2], [1, 4, 2], [4, 2, 1])


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _aggregate(run_root: Path, windows: list[float], sessions: int) -> dict[str, Any]:
    reports: dict[tuple[int, int, float], dict[str, Any]] = {}
    for session in range(1, sessions + 1):
        for stride in (1, 2, 4):
            for window in windows:
                path = (
                    run_root / f"session-{session}" / f"stride-{stride}"
                    / f"window-{window:.1f}.json"
                )
                reports[(session, stride, window)] = json.loads(
                    path.read_text(encoding="utf-8")
                )
    summary: dict[str, Any] = {
        "status": "complete",
        "sessions": sessions,
        "windows_seconds": windows,
        "graph": "folded deployment graph",
        "batch_size": 1,
        "results": {},
        "paired_relative_to_stride_2": {},
    }
    for stride in (1, 2, 4):
        stride_rows: dict[str, Any] = {}
        for window in windows:
            selected = [reports[(session, stride, window)] for session in range(1, sessions + 1)]
            window_row: dict[str, Any] = {
                "spectral_token_count": selected[0]["spectral_token_count"],
                "parameters": selected[0]["parameters"]["deployment_graph"],
                "operation_count": selected[0]["operation_count"]["deployment_graph"],
            }
            for backend in ("cpu", "cuda"):
                values = [row["deployment_graph"][backend] for row in selected]
                window_row[backend] = {
                    "p50_ms_median": statistics.median(float(row["p50_ms"]) for row in values),
                    "p50_ms_range": [
                        min(float(row["p50_ms"]) for row in values),
                        max(float(row["p50_ms"]) for row in values),
                    ],
                    "p95_ms_median": statistics.median(float(row["p95_ms"]) for row in values),
                    "p95_ms_range": [
                        min(float(row["p95_ms"]) for row in values),
                        max(float(row["p95_ms"]) for row in values),
                    ],
                }
            stride_rows[f"{window:.1f}"] = window_row
        summary["results"][str(stride)] = stride_rows

    for stride in (1, 4):
        comparisons: dict[str, Any] = {}
        for window in windows:
            backend_rows: dict[str, Any] = {}
            for backend in ("cpu", "cuda"):
                changes_p50 = []
                changes_p95 = []
                for session in range(1, sessions + 1):
                    candidate = reports[(session, stride, window)]["deployment_graph"][backend]
                    retained = reports[(session, 2, window)]["deployment_graph"][backend]
                    changes_p50.append(
                        100.0 * (float(candidate["p50_ms"]) / float(retained["p50_ms"]) - 1.0)
                    )
                    changes_p95.append(
                        100.0 * (float(candidate["p95_ms"]) / float(retained["p95_ms"]) - 1.0)
                    )
                backend_rows[backend] = {
                    "candidate_minus_stride_2_p50_percent_median": statistics.median(changes_p50),
                    "candidate_minus_stride_2_p95_percent_median": statistics.median(changes_p95),
                    "paired_sessions": sessions,
                }
            comparisons[f"{window:.1f}"] = backend_rows
        summary["paired_relative_to_stride_2"][str(stride)] = comparisons
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
    if args.sessions > len(ORDERS):
        raise ValueError(f"at most {len(ORDERS)} sessions are registered")
    state: dict[str, Any] = {
        "status": "running",
        "registered_runs": args.sessions * 3 * len(args.windows),
        "completed_runs": 0,
    }
    _atomic_json(args.run_root / "state.json", state)
    for session in range(1, args.sessions + 1):
        windows = args.windows if session % 2 else list(reversed(args.windows))
        for stride in ORDERS[session - 1]:
            for window in windows:
                output = (
                    args.run_root / f"session-{session}" / f"stride-{stride}"
                    / f"window-{window:.1f}.json"
                )
                if output.exists():
                    state["completed_runs"] += 1
                    _atomic_json(args.run_root / "state.json", state)
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                state["current_run"] = {
                    "session": session, "stride": stride, "window_seconds": window,
                }
                _atomic_json(args.run_root / "state.json", state)
                command = [
                    sys.executable, "-m", "src.benchmark_deployment",
                    "--checkpoint", str(args.checkpoint),
                    "--window", str(window),
                    "--warmup", str(args.warmup),
                    "--iterations", str(args.iterations),
                    "--spectral-token-stride", str(stride),
                    "--benchmark-graphs", "deployment",
                    "--output", str(output),
                ]
                with (output.parent / "worker.log").open("a", encoding="utf-8") as log:
                    process = subprocess.run(
                        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        check=False,
                    )
                if process.returncode:
                    state["status"] = "failed"
                    state["returncode"] = int(process.returncode)
                    _atomic_json(args.run_root / "state.json", state)
                    raise SystemExit(process.returncode)
                state["completed_runs"] += 1
                _atomic_json(args.run_root / "state.json", state)
    summary = _aggregate(args.run_root, args.windows, args.sessions)
    _atomic_json(args.run_root / "summary.json", summary)
    state["status"] = "complete"
    state.pop("current_run", None)
    _atomic_json(args.run_root / "state.json", state)


if __name__ == "__main__":
    main()
