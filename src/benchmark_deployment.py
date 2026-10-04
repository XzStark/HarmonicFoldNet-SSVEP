from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .harmonic_fold import SpectralFusionDecoder, HarmonicFoldNet, TemporalFusionDecoder
from .model import LegacyFusionNet, parameter_count, reparameterize_model
from .paper_data import class_frequencies, class_phases
from .paper_metrics import atomic_write_json
from .reference_models import SSVEPFormerReference


class _ModeWrapper(torch.nn.Module):
    """Bind an ablation mode without changing the checkpointed model."""

    def __init__(self, model: torch.nn.Module, mode: str) -> None:
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_mode(x, self.mode)


def _summary(milliseconds: list[float]) -> dict[str, float]:
    values = np.asarray(milliseconds, dtype=np.float64)
    return {
        "iterations": len(values), "mean_ms": float(values.mean()),
        "standard_deviation_ms": float(values.std(ddof=1)),
        "p50_ms": float(np.quantile(values, 0.5)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "minimum_ms": float(values.min()), "maximum_ms": float(values.max()),
    }


@torch.inference_mode()
def benchmark_cpu(model: torch.nn.Module, x: torch.Tensor, *, warmup: int, iterations: int) -> dict[str, float]:
    model = copy.deepcopy(model).cpu().eval()
    x = x.cpu()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(warmup):
            model(x)
        timings = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            model(x)
            timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    finally:
        torch.set_num_threads(previous_threads)
    return _summary(timings)


@torch.inference_mode()
def benchmark_cuda(model: torch.nn.Module, x: torch.Tensor, *, warmup: int, iterations: int) -> dict[str, float]:
    model = copy.deepcopy(model).cuda().eval()
    x = x.cuda()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(x)
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    result = _summary(timings)
    result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--mode", choices=(
            "full", "no_evidence", "no_attention", "no_local", "no_harmonic_bias",
        ),
        help="HarmonicFold forward mode; defaults to the checkpoint result mode.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    result = payload["result"]
    config = payload["config"]
    dataset_cfg = config["datasets"][result["dataset"]]
    frequencies = class_frequencies(
        dataset_cfg["shard_dir"], dataset_cfg["subjects"], classes=int(dataset_cfg["classes"]),
    )
    phases = class_phases(
        dataset_cfg["shard_dir"], dataset_cfg["subjects"], classes=int(dataset_cfg["classes"]),
    )
    if result["architecture"] == "temporal_fusion":
        model = TemporalFusionDecoder(
            channels=len(dataset_cfg["channels"]),
            sample_rate=int(config["sample_rate"]),
            class_frequencies=frequencies,
            class_phases=phases,
            **config["harmonic_fold_model"],
        ).eval()
    elif result["architecture"] == "spectral_fusion":
        model = SpectralFusionDecoder(
            channels=len(dataset_cfg["channels"]),
            sample_rate=int(config["sample_rate"]),
            class_frequencies=frequencies,
            class_phases=phases,
            **config["spectral_fusion_model"],
        ).eval()
    elif result["architecture"] in {
        "harmonic_fold_v4", "harmonic_fold_v4_1", "harmonic_fold_v4_2",
        "harmonic_fold_v4_3",
    }:
        model_key = {
            "harmonic_fold_v4": "harmonic_fold_v4_model",
            "harmonic_fold_v4_1": "harmonic_fold_v4_1_model",
            "harmonic_fold_v4_2": "harmonic_fold_v4_2_model",
            "harmonic_fold_v4_3": "harmonic_fold_v4_3_model",
        }[result["architecture"]]
        model = HarmonicFoldNet(
            channels=len(dataset_cfg["channels"]),
            sample_rate=int(config["sample_rate"]),
            class_frequencies=frequencies,
            class_phases=phases,
            **config[model_key],
        ).eval()
    elif result["architecture"] == "legacy_fusion":
        model = LegacyFusionNet(
            channels=len(dataset_cfg["channels"]), classes=int(dataset_cfg["classes"]),
            sample_rate=int(config["sample_rate"]), class_frequencies=frequencies,
            class_phases=phases, **config["model"],
        ).eval()
    elif result["architecture"] == "ssvepformer":
        model = SSVEPFormerReference(
            channels=len(dataset_cfg["channels"]),
            classes=int(dataset_cfg["classes"]),
            sample_rate=int(config["sample_rate"]),
        ).eval()
    else:
        raise ValueError(f"unsupported checkpoint architecture: {result['architecture']}")
    model.load_state_dict(payload["state_dict"])
    deploy = (
        reparameterize_model(model)
        if result["architecture"] in {
            "legacy_fusion", "temporal_fusion", "spectral_fusion", "harmonic_fold_v4",
            "harmonic_fold_v4_1",
            "harmonic_fold_v4_2",
            "harmonic_fold_v4_3",
        }
        else copy.deepcopy(model).eval()
    )
    mode = args.mode or result.get("mode", "full")
    if result["architecture"] in {
        "legacy_fusion", "temporal_fusion", "spectral_fusion", "harmonic_fold_v4",
        "harmonic_fold_v4_1",
        "harmonic_fold_v4_2",
        "harmonic_fold_v4_3",
    }:
        model = _ModeWrapper(model, mode).eval()
        deploy = _ModeWrapper(deploy, mode).eval()
    elif mode != "full":
        raise ValueError(f"{result['architecture']} only supports mode=full")
    samples = int(round(args.window * int(config["sample_rate"])))
    x = torch.randn(1, len(dataset_cfg["channels"]), samples)
    with torch.inference_mode():
        original = model(x)
        deployed = deploy(x)
    maximum_error = float((original - deployed).abs().max())
    if not torch.allclose(original, deployed, rtol=2e-4, atol=2e-5):
        raise RuntimeError(f"deployment graph mismatch: max abs error {maximum_error}")
    report: dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "window_seconds": args.window,
        "architecture": result["architecture"],
        "architecture_revision": result.get("architecture_revision"),
        "mode": mode, "batch_size": 1, "dtype": "float32",
        "equivalence_max_abs_error": maximum_error,
        "parameters": {
            "training_graph": parameter_count(model),
            "deployment_graph": parameter_count(deploy),
        },
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "cpu_threads_timed": 1,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "training_graph": {"cpu": benchmark_cpu(model, x, warmup=args.warmup, iterations=args.iterations)},
        "deployment_graph": {"cpu": benchmark_cpu(deploy, x, warmup=args.warmup, iterations=args.iterations)},
    }
    if torch.cuda.is_available():
        report["training_graph"]["cuda"] = benchmark_cuda(model, x, warmup=args.warmup, iterations=args.iterations)
        report["deployment_graph"]["cuda"] = benchmark_cuda(deploy, x, warmup=args.warmup, iterations=args.iterations)
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
