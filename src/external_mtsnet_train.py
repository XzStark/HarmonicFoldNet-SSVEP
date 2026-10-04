from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .external_mtsnet import MTSNET_COMMIT, MTSNET_UPSTREAM, ExternalMTSNetAdapter
from .paper_data import grouped_kfold_split, load_subjects_window
from .paper_metrics import atomic_write_json, classification_metrics


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def infer(model: nn.Module, x: np.ndarray, *, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    rows = []
    for begin in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[begin:begin + batch_size]).to(device, non_blocking=True)
        rows.append(model(batch).float().cpu().numpy())
    return np.concatenate(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    config = copy.deepcopy(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    dataset_cfg = config["datasets"][args.dataset]
    seed_all(int(args.seed))
    subjects = list(map(str, dataset_cfg["subjects"]))
    train_subjects, validation_subjects, test_subjects = grouped_kfold_split(
        subjects,
        fold_index=int(args.fold_index),
        fold_count=int(args.fold_count),
        validation_count=int(dataset_cfg["validation_count_loso"]),
        split_seed=int(config.get("split_seed", 20260929)),
    )
    sample_rate = int(config["sample_rate"])
    window = float(args.window)
    samples = int(round(window * sample_rate))
    channels = list(map(str, dataset_cfg["channels"]))
    load_options = dict(
        window_seconds=window,
        sample_rate=sample_rate,
        channels=channels,
        trial_filters=dataset_cfg.get("trial_filters"),
    )
    train_x, train_y, _ = load_subjects_window(
        dataset_cfg["shard_dir"], train_subjects, **load_options,
    )
    validation_x, validation_y, _ = load_subjects_window(
        dataset_cfg["shard_dir"], validation_subjects, **load_options,
    )
    test_x, test_y, test_subject_ids = load_subjects_window(
        dataset_cfg["shard_dir"], test_subjects, **load_options,
    )
    model = ExternalMTSNetAdapter(
        source=args.source,
        time_points=samples,
        channels=len(channels),
        classes=int(dataset_cfg["classes"]),
        depth_local=int(args.depth_local),
        depth_fusion=int(args.depth_fusion),
        kernel_length=int(args.kernel_length),
        dropout=float(args.dropout),
        spectral_normalization=args.spectral_normalization,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=float(args.learning_rate), weight_decay=float(args.weight_decay),
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -np.inf
    best_epoch = 0
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation_logits = infer(
            model, validation_x, batch_size=int(args.eval_batch_size), device=device,
        )
        score = float(balanced_accuracy_score(validation_y, validation_logits.argmax(axis=1)))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "validation_balanced_accuracy": score,
            "seconds": round(time.perf_counter() - started, 3),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("MTSNet reconstruction produced no checkpoint")
    model.load_state_dict(best_state)
    logits = infer(model, test_x, batch_size=int(args.eval_batch_size), device=device)
    classes = int(dataset_cfg["classes"])
    overall = classification_metrics(
        test_y, logits,
        classes=classes,
        window_seconds=window,
        cue_seconds=float(config["cue_seconds_for_itr"]),
    )
    per_subject = {}
    for subject in sorted(set(map(str, test_subject_ids)), key=lambda value: (len(value), value)):
        mask = test_subject_ids.astype(str) == subject
        per_subject[subject] = classification_metrics(
            test_y[mask], logits[mask],
            classes=classes,
            window_seconds=window,
            cue_seconds=float(config["cue_seconds_for_itr"]),
        )
    result = {
        "status": "complete",
        "dataset": args.dataset,
        "architecture": "mtsnet_external_protocol_reconstruction",
        "upstream": MTSNET_UPSTREAM,
        "upstream_commit": MTSNET_COMMIT,
        "source_path": str(Path(args.source).resolve()),
        "protocol_status": "reconstruction_not_exact_published_reproduction",
        "seed": int(args.seed),
        "split_seed": int(config.get("split_seed", 20260929)),
        "split": f"group-{args.fold_count}-fold-{args.fold_index}",
        "subjects": {
            "train": train_subjects,
            "validation": validation_subjects,
            "test": test_subjects,
        },
        "sample_counts": {
            "train": len(train_y), "validation": len(validation_y), "test": len(test_y),
        },
        "channels": channels,
        "window_seconds": window,
        "spectral_representation": "rfft_n560_drop_dc_concat_real_imag",
        "spectral_normalization": args.spectral_normalization,
        "model_config": {
            "depth_local": int(args.depth_local),
            "depth_fusion": int(args.depth_fusion),
            "kernel_length": int(args.kernel_length),
            "dropout": float(args.dropout),
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "label_smoothing": float(args.label_smoothing),
            "batch_size": int(args.batch_size),
        },
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_score,
        "history": history,
        "test": {str(window): {"overall": overall, "per_subject": per_subject}},
    }
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "result.json", result)
    temporary = run_dir / ".predictions.tmp.npz"
    np.savez_compressed(
        temporary, y=test_y, subject=test_subject_ids, logits=logits.astype(np.float32),
    )
    temporary.replace(run_dir / "predictions.npz")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_multidataset.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", default=".research_refs/MTSNet/MTSNet.py")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--window", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--depth-local", type=int, default=2)
    parser.add_argument("--depth-fusion", type=int, default=2)
    parser.add_argument("--kernel-length", type=int, default=31)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--spectral-normalization", choices=("none", "channel_zscore"), default="none",
    )
    parser.add_argument("--device")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result_path = Path(args.run_dir) / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(json.dumps({"status": "skipped", "path": str(result_path)}))
            return
    run(args)


if __name__ == "__main__":
    main()
