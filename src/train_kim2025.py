from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import FastSSVEPFusionNet, parameter_count


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_subjects(shard_dir: Path, subjects: list[str]):
    xs, ys = [], []
    for subject in subjects:
        path = shard_dir / f"sub-{subject}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            xs.append(data["x"])
            ys.append(data["y"])
    return np.concatenate(xs), np.concatenate(ys)


def _class_frequencies(shard_dir: Path, subjects: list[str], classes: int = 40) -> list[float]:
    resolved: dict[int, float] = {}
    for subject in subjects:
        with np.load(shard_dir / f"sub-{subject}.npz", allow_pickle=False) as data:
            for label, frequency in zip(data["y"], data["frequency_hz"], strict=True):
                label = int(label)
                frequency = float(frequency)
                previous = resolved.get(label)
                if previous is not None and not np.isclose(previous, frequency, atol=1e-4):
                    raise RuntimeError(f"inconsistent frequency for class {label}: {previous} vs {frequency}")
                resolved[label] = frequency
    if set(resolved) != set(range(classes)):
        raise RuntimeError(f"frequency metadata does not cover {classes} classes")
    return [resolved[index] for index in range(classes)]


@torch.inference_mode()
def _predict(model, x: np.ndarray, device: torch.device, batch_size: int):
    model.eval()
    logits = []
    for begin in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[begin:begin + batch_size]).to(device)
        logits.append(model(batch).float().cpu().numpy())
    return np.concatenate(logits)


def _metrics(y: np.ndarray, logits: np.ndarray) -> dict:
    pred = logits.argmax(1)
    probs = torch.from_numpy(logits).softmax(1).numpy()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "top5_accuracy": float(top_k_accuracy_score(y, probs, k=5, labels=np.arange(40))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kim2025.yaml")
    parser.add_argument("--run-dir", default="runs/kim2025_initial")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    _seed(int(cfg["seed"]))
    shard_dir = Path(cfg["shard_dir"])
    train_x, train_y = _load_subjects(shard_dir, [str(x) for x in cfg["train_subjects"]])
    val_x, val_y = _load_subjects(shard_dir, [str(x) for x in cfg["validation_subjects"]])
    test_x, test_y = _load_subjects(shard_dir, [str(x) for x in cfg["test_subjects"]])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastSSVEPFusionNet(
        channels=train_x.shape[1], classes=40, sample_rate=int(cfg["sample_rate"]),
        class_frequencies=_class_frequencies(shard_dir, [str(x) for x in cfg["train_subjects"]]),
        **cfg["model"],
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(cfg["label_smoothing"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    best_state = None
    history = []
    stale_epochs = 0
    patience = int(cfg.get("early_stopping_patience", 0))
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        losses = []
        started = time.perf_counter()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_logits = _predict(model, val_x, device, int(cfg["eval_batch_size"]))
        val = _metrics(val_y, val_logits)
        row = {
            "epoch": epoch, "loss": float(np.mean(losses)),
            "val_accuracy": val["accuracy"], "val_top5_accuracy": val["top5_accuracy"],
            "seconds": round(time.perf_counter() - started, 3),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["balanced_accuracy"] > best:
            best = val["balanced_accuracy"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if patience and stale_epochs >= patience:
            print(json.dumps({"early_stop": epoch, "best_validation_balanced_accuracy": best}), flush=True)
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_logits = _predict(model, test_x, device, int(cfg["eval_batch_size"]))
    with torch.inference_mode():
        evidence = []
        for begin in range(0, len(test_x), int(cfg["eval_batch_size"])):
            batch = torch.from_numpy(test_x[begin:begin + int(cfg["eval_batch_size"])]).to(device)
            evidence.append(model.candidate_evidence(batch).float().cpu().numpy())
        evidence_logits = np.concatenate(evidence)
    result = {
        "model": {"name": type(model).__name__, "parameters": parameter_count(model)},
        "device": str(device),
        "split": {
            "train_subjects": cfg["train_subjects"], "validation_subjects": cfg["validation_subjects"],
            "test_subjects": cfg["test_subjects"], "train_trials": len(train_y),
            "validation_trials": len(val_y), "test_trials": len(test_y),
        },
        "test": _metrics(test_y, test_logits),
        "spectral_evidence_only_test": _metrics(test_y, evidence_logits),
        "best_validation_balanced_accuracy": best, "history": history,
    }
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": cfg, "result": result}, run_dir / "model.pt")
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
