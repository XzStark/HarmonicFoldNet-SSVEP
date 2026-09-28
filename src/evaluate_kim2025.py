from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from .model import FastSSVEPFusionNet, parameter_count, reparameterize_model
from .train_kim2025 import _class_frequencies, _load_subjects


def _fixed_harmonic_predictions(x: np.ndarray, frequencies: list[float], sample_rate: int) -> np.ndarray:
    power = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    hz = np.fft.rfftfreq(x.shape[-1], 1 / sample_rate)
    scores = []
    for frequency in frequencies:
        bins = [np.argmin(np.abs(hz - frequency * harmonic))
                for harmonic in (1, 2) if frequency * harmonic <= hz[-1]]
        scores.append(power[:, :, bins].mean(axis=(1, 2)))
    return np.stack(scores, axis=1).argmax(axis=1)


@torch.inference_mode()
def _model_predictions(model, x: np.ndarray, device: torch.device, batch_size: int = 256):
    logits, evidence = [], []
    for begin in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[begin:begin + batch_size]).to(device)
        logits.append(model(batch).float().cpu().numpy())
        evidence.append(model.candidate_evidence(batch).float().cpu().numpy())
    return np.concatenate(logits).argmax(1), np.concatenate(evidence).argmax(1)


def _benchmark(model, x: torch.Tensor, device: torch.device, iterations: int) -> float:
    model = model.to(device)
    x = x.to(device)
    with torch.inference_mode():
        for _ in range(30):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    shard_dir = Path(cfg["shard_dir"])
    all_subjects = [str(x) for x in cfg["train_subjects"]]
    frequencies = _class_frequencies(shard_dir, all_subjects)
    model = FastSSVEPFusionNet(
        channels=8, classes=len(frequencies), sample_rate=int(cfg["sample_rate"]),
        class_frequencies=frequencies, **cfg["model"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    deploy = reparameterize_model(model)
    probe = torch.randn(1, 8, 1250)
    torch.testing.assert_close(model(probe), deploy(probe), rtol=2e-4, atol=2e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for split, subjects in (
        ("validation", cfg["validation_subjects"]), ("test", cfg["test_subjects"]),
    ):
        for subject in map(str, subjects):
            x, y = _load_subjects(shard_dir, [subject])
            fusion, evidence = _model_predictions(model.to(device), x, device)
            fixed = _fixed_harmonic_predictions(x, frequencies, int(cfg["sample_rate"]))
            rows.append({
                "subject": subject, "split": split,
                "fixed_harmonic_accuracy": float(accuracy_score(y, fixed)),
                "learned_evidence_accuracy": float(accuracy_score(y, evidence)),
                "fusion_accuracy": float(accuracy_score(y, fusion)),
            })
    train_cuda = deploy_cuda = None
    if torch.cuda.is_available():
        train_cuda = _benchmark(model, probe, torch.device("cuda"), 300)
        deploy_cuda = _benchmark(deploy, probe, torch.device("cuda"), 300)
    torch.set_num_threads(1)
    train_cpu = _benchmark(model.cpu(), probe, torch.device("cpu"), 150)
    deploy_cpu = _benchmark(deploy.cpu(), probe, torch.device("cpu"), 150)
    result = {
        "subjects": rows,
        "aggregate": {
            split: {
                key: float(np.mean([row[key] for row in rows if row["split"] == split]))
                for key in ("fixed_harmonic_accuracy", "learned_evidence_accuracy", "fusion_accuracy")
            }
            for split in ("validation", "test")
        },
        "parameters": {"train": parameter_count(model), "deploy": parameter_count(deploy)},
        "latency_ms_batch1": {
            "cuda_train": train_cuda, "cuda_deploy": deploy_cuda,
            "cpu_1thread_train": train_cpu, "cpu_1thread_deploy": deploy_cpu,
        },
        "reparameterization_equivalence": "passed",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
