from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .baselines import cca_predict
from .data import FREQUENCIES, build_ds004745_windows, describe, load_windows, save_windows
from .model import LegacyFusionNet, LegacyLocalAttentionNet, parameter_count


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def metrics(y, pred, artifact):
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    for name, mask in (("clean", ~artifact), ("artifact", artifact)):
        out[f"{name}_accuracy"] = float(accuracy_score(y[mask], pred[mask])) if mask.any() else None
    return out


@torch.inference_mode()
def infer(model, x, device, batch_size=256):
    model.eval(); output=[]
    for begin in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[begin:begin + batch_size]).to(device)
        output.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ds004745.yaml")
    parser.add_argument("--run-dir", default="runs/ds004745_initial")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs is not None: cfg["epochs"] = args.epochs
    seed_all(int(cfg["seed"]))
    cache = Path(cfg["cache_path"])
    if not cache.exists():
        windows = build_ds004745_windows(
            cfg["dataset_root"], target_sfreq=int(cfg["sample_rate"]),
            window_seconds=float(cfg["window_seconds"]),
            stride_seconds=float(cfg["window_stride_seconds"]),
            onset_guard_seconds=float(cfg["onset_guard_seconds"]),
            offset_guard_seconds=float(cfg["offset_guard_seconds"]),
            bandpass_hz=tuple(cfg["bandpass_hz"]),
        )
        save_windows(cache, windows)
    windows = load_windows(cache)
    test_mask = np.isin(windows.subject, cfg["held_out_subjects"])
    val_mask = np.isin(windows.subject, cfg["validation_subjects"])
    train_mask = ~(test_mask | val_mask)
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise RuntimeError("subject-disjoint train/validation/test split is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = dict(cfg["model"])
    architecture = model_cfg.pop("architecture", "time")
    if architecture == "fusion":
        model = LegacyFusionNet(
            channels=windows.x.shape[1], classes=len(FREQUENCIES),
            sample_rate=windows.sample_rate, **model_cfg,
        ).to(device)
    elif architecture == "time":
        model = LegacyLocalAttentionNet(
            channels=windows.x.shape[1], classes=len(FREQUENCIES), **model_cfg,
        ).to(device)
    else:
        raise ValueError(f"unknown model architecture: {architecture}")
    train_ds = TensorDataset(torch.from_numpy(windows.x[train_mask]), torch.from_numpy(windows.y[train_mask]))
    loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True,
                        num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(cfg["label_smoothing"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0; best_state = None; history=[]
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train(); losses=[]; started=time.perf_counter()
        for x, y in loader:
            x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss=loss_fn(model(x), y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))
        val_pred=infer(model, windows.x[val_mask], device)
        val_bal=float(balanced_accuracy_score(windows.y[val_mask], val_pred))
        row={"epoch":epoch,"loss":float(np.mean(losses)),"val_balanced_accuracy":val_bal,
             "seconds":round(time.perf_counter()-started,3)}
        history.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_bal > best:
            best=val_bal; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state); test_pred=infer(model, windows.x[test_mask], device)
    cca_pred=cca_predict(windows.x[test_mask], FREQUENCIES, windows.sample_rate)
    result={
        "dataset":describe(windows), "split":{
            "train_subjects":sorted(set(windows.subject[train_mask].tolist())),
            "validation_subjects":sorted(set(windows.subject[val_mask].tolist())),
            "test_subjects":sorted(set(windows.subject[test_mask].tolist())),
        },
        "model":{"name":type(model).__name__,"parameters":parameter_count(model)},
        "fast_ssvep":metrics(windows.y[test_mask],test_pred,windows.artifact[test_mask]),
        "cca":metrics(windows.y[test_mask],cca_pred,windows.artifact[test_mask]),
        "best_validation_balanced_accuracy":best,
        "history":history,
    }
    run=Path(args.run_dir); run.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":best_state,"config":cfg,"result":result},run/"model.pt")
    (run/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (run/"config.yaml").write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
