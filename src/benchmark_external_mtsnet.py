from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import yaml

from .benchmark_deployment import benchmark_cpu, benchmark_cuda
from .external_mtsnet import MTSNET_COMMIT, MTSNET_UPSTREAM, ExternalMTSNetAdapter
from .paper_metrics import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the local research-only MTSNet protocol reconstruction."
    )
    parser.add_argument("--config", default="configs/paper_multidataset.yaml")
    parser.add_argument("--dataset", default="benchmark")
    parser.add_argument("--window", type=float, default=0.8)
    parser.add_argument("--source", default=".research_refs/MTSNet/MTSNet.py")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_cfg = config["datasets"][args.dataset]
    samples = int(round(float(args.window) * int(config["sample_rate"])))
    model = ExternalMTSNetAdapter(
        source=args.source,
        time_points=samples,
        channels=len(dataset_cfg["channels"]),
        classes=int(dataset_cfg["classes"]),
        depth_local=2,
        depth_fusion=2,
        kernel_length=31,
        dropout=0.5,
        spectral_normalization="none",
    ).eval()
    x = torch.randn(1, len(dataset_cfg["channels"]), samples)
    report: dict[str, object] = {
        "architecture": "mtsnet_external_protocol_reconstruction",
        "protocol_status": "reconstruction_not_exact_published_reproduction",
        "upstream": MTSNET_UPSTREAM,
        "upstream_commit": MTSNET_COMMIT,
        "source_path": str(Path(args.source).resolve()),
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
