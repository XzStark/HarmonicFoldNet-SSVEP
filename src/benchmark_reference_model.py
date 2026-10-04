from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import yaml

from .benchmark_deployment import benchmark_cpu, benchmark_cuda
from .paper_metrics import atomic_write_json
from .reference_models import FBSSVEPFormerReference, SSVEPFormerReference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Protocol-matched batch-one latency for reference decoders."
    )
    parser.add_argument("--config", default="configs/paper_multidataset.yaml")
    parser.add_argument("--dataset", default="beta")
    parser.add_argument(
        "--architecture", required=True,
        choices=("ssvepformer", "fb_ssvepformer"),
    )
    parser.add_argument("--window", type=float, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_cfg = config["datasets"][args.dataset]
    common = {
        "channels": len(dataset_cfg["channels"]),
        "classes": int(dataset_cfg["classes"]),
        "sample_rate": int(config["sample_rate"]),
    }
    model = (
        SSVEPFormerReference(**common)
        if args.architecture == "ssvepformer"
        else FBSSVEPFormerReference(**common)
    ).eval()
    samples = int(round(float(args.window) * int(config["sample_rate"])))
    x = torch.randn(1, len(dataset_cfg["channels"]), samples)
    report: dict[str, object] = {
        "architecture": args.architecture,
        "protocol_status": "protocol_adapted_reference",
        "dataset_shape_reference": args.dataset,
        "window_seconds": float(args.window),
        "batch_size": 1,
        "dtype": "float32",
        "spectral_preprocessing_included": True,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_threads_timed": 1,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "cpu": benchmark_cpu(
            model, x, warmup=int(args.warmup), iterations=int(args.iterations),
        ),
    }
    if torch.cuda.is_available():
        report["cuda"] = benchmark_cuda(
            model, x, warmup=int(args.warmup), iterations=int(args.iterations),
        )
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
