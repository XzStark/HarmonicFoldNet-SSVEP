from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

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


def _leaf_module_inventory(model: torch.nn.Module) -> dict[str, object]:
    counts = Counter(
        type(module).__name__
        for module in model.modules()
        if module is not model and not any(module.children())
    )
    return {
        "leaf_module_count": int(sum(counts.values())),
        "by_type": dict(sorted(counts.items())),
    }


def _metadata_from_checkpoint(
    state_dict: dict[str, torch.Tensor], classes: int,
) -> tuple[list[float], list[float]]:
    frequency_keys = [
        key for key, value in state_dict.items()
        if key.endswith("class_frequencies") and int(value.numel()) == classes
    ]
    phase_keys = [
        key for key, value in state_dict.items()
        if key.endswith("class_phases") and int(value.numel()) == classes
    ]
    if not frequency_keys or not phase_keys:
        raise RuntimeError(
            "checkpoint does not contain class-frequency and phase metadata"
        )
    frequencies = state_dict[frequency_keys[0]].detach().cpu().flatten()
    phases = state_dict[phase_keys[0]].detach().cpu().flatten()
    for key in frequency_keys[1:]:
        if not torch.allclose(
            frequencies, state_dict[key].detach().cpu().flatten(), atol=1e-6, rtol=0,
        ):
            raise RuntimeError(f"inconsistent frequency metadata in checkpoint: {key}")
    for key in phase_keys[1:]:
        if not torch.allclose(
            phases, state_dict[key].detach().cpu().flatten(), atol=1e-6, rtol=0,
        ):
            raise RuntimeError(f"inconsistent phase metadata in checkpoint: {key}")
    return frequencies.tolist(), phases.tolist()


@torch.inference_mode()
def _count_flops(model: torch.nn.Module, x: torch.Tensor) -> dict[str, object]:
    graph = copy.deepcopy(model).cpu().eval()
    with FlopCounterMode(display=False) as counter:
        graph(x.cpu())
    flops = int(counter.get_total_flops())
    return {
        "torch_counted_flops": flops,
        "mac_equivalent": float(flops / 2.0),
        "convention": "one multiply-add equals two FLOPs",
        "counter": "torch.utils.flop_counter.FlopCounterMode",
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
        "--graph-order", choices=("training_first", "deployment_first"),
        default="training_first",
        help="Alternate across fresh processes to control timing-order effects.",
    )
    parser.add_argument(
        "--spectral-token-stride", type=int,
        help="Structural timing override for HarmonicFold checkpoints.",
    )
    parser.add_argument(
        "--benchmark-graphs", choices=("both", "training", "deployment"),
        default="both",
    )
    parser.add_argument(
        "--mode", choices=(
            "full", "no_evidence", "no_attention", "no_local", "no_harmonic_bias",
            "no_temporal_candidate", "no_spectral_candidate",
        ),
        help="HarmonicFold forward mode; defaults to the checkpoint result mode.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    result = payload["result"]
    config = payload["config"]
    dataset_cfg = config["datasets"][result["dataset"]]
    classes = int(dataset_cfg["classes"])
    try:
        frequencies = class_frequencies(
            dataset_cfg["shard_dir"], dataset_cfg["subjects"], classes=classes,
        )
        phases = class_phases(
            dataset_cfg["shard_dir"], dataset_cfg["subjects"], classes=classes,
        )
        metadata_source = "dataset_shards"
    except (FileNotFoundError, OSError):
        frequencies, phases = _metadata_from_checkpoint(payload["state_dict"], classes)
        metadata_source = "checkpoint_buffers"
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
        model_kwargs = dict(config[model_key])
        if args.spectral_token_stride is not None:
            if args.spectral_token_stride <= 0:
                raise ValueError("spectral token stride must be positive")
            model_kwargs["spectral_token_stride"] = args.spectral_token_stride
        model = HarmonicFoldNet(
            channels=len(dataset_cfg["channels"]),
            sample_rate=int(config["sample_rate"]),
            class_frequencies=frequencies,
            class_phases=phases,
            **model_kwargs,
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
    spectral_token_count = (
        int(model.spectral_position_features.shape[0])
        if hasattr(model, "spectral_position_features") else None
    )
    spectral_token_stride = getattr(model, "spectral_token_stride", None)
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
        "graph_order": args.graph_order, "metadata_source": metadata_source,
        "benchmark_graphs": args.benchmark_graphs,
        "spectral_token_stride": spectral_token_stride,
        "spectral_token_count": spectral_token_count,
        "equivalence_max_abs_error": maximum_error,
        "parameters": {
            "training_graph": parameter_count(model),
            "deployment_graph": parameter_count(deploy),
        },
        "module_inventory": {
            "training_graph": _leaf_module_inventory(model),
            "deployment_graph": _leaf_module_inventory(deploy),
        },
        "operation_count": {
            "training_graph": _count_flops(model, x),
            "deployment_graph": _count_flops(deploy, x),
        },
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "cpu_threads_timed": 1,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "training_graph": {},
        "deployment_graph": {},
    }
    ordered_graphs = (
        (("training_graph", model), ("deployment_graph", deploy))
        if args.graph_order == "training_first"
        else (("deployment_graph", deploy), ("training_graph", model))
    )
    if args.benchmark_graphs != "both":
        selected_name = f"{args.benchmark_graphs}_graph"
        ordered_graphs = tuple(
            (name, graph) for name, graph in ordered_graphs if name == selected_name
        )
    for name, graph in ordered_graphs:
        report[name]["cpu"] = benchmark_cpu(
            graph, x, warmup=args.warmup, iterations=args.iterations,
        )
    if torch.cuda.is_available():
        for name, graph in ordered_graphs:
            report[name]["cuda"] = benchmark_cuda(
                graph, x, warmup=args.warmup, iterations=args.iterations,
            )
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
