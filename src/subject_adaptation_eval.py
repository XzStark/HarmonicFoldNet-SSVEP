from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from .harmonic_fold import HarmonicFoldNet
from .paper_calibrated_eval import _load_subject, _preceding_calibration_blocks
from .paper_data import class_frequencies, class_phases
from .paper_metrics import atomic_write_json, metrics_from_predictions
from .paper_train import crop_tensor, seed_all


class SpatialScoreAdapter(nn.Module):
    """Tiny participant adapter around a frozen calibration-free model.

    The channel mixer starts as an exact identity.  Optionally, only the last
    shared candidate-scoring layer is adapted with it.  The backbone remains
    frozen, so the number of participant-specific parameters is independent of
    the number of EEG classes.
    """

    def __init__(self, base: HarmonicFoldNet, channels: int, *, tune_score: bool):
        super().__init__()
        self.base = base
        self.channel_delta = nn.Parameter(torch.zeros(channels, channels))
        self.tune_score = bool(tune_score)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        if self.tune_score:
            for parameter in self.base.score_head[-1].parameters():
                parameter.requires_grad_(True)

    def adapted_parameters(self) -> list[nn.Parameter]:
        parameters = [self.channel_delta]
        if self.tune_score:
            parameters.extend(self.base.score_head[-1].parameters())
        return parameters

    @property
    def participant_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.adapted_parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(
            self.channel_delta.shape[0], device=x.device, dtype=x.dtype,
        )
        mixed = torch.einsum("ij,bjt->bit", identity + self.channel_delta, x)
        return self.base.forward_mode(mixed, "full")


def _checkpoint_index(root: Path, dataset: str, seed: int) -> dict[str, Path]:
    pattern = (
        root / dataset / "harmonic_fold_v4_1" / "full" /
        f"seed-{seed}"
    )
    index: dict[str, Path] = {}
    for result_path in sorted(pattern.glob("fold-*/result.json")):
        document = json.loads(result_path.read_text(encoding="utf-8"))
        checkpoint = result_path.with_name("model.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        for subject in map(str, document["subjects"]["test"]):
            if subject in index:
                raise RuntimeError(f"subject {subject} appears in multiple outer folds")
            if subject in set(map(str, document["subjects"]["train"])):
                raise RuntimeError(f"subject {subject} leaked into its source training split")
            index[subject] = checkpoint
    return index


def _make_model(
    payload: dict[str, object], *, channels: list[str], frequencies: list[float],
    phases: list[float], sample_rate: int,
) -> HarmonicFoldNet:
    result = payload["result"]
    if result["architecture"] != "harmonic_fold_v4_1":
        raise ValueError(f"unsupported source architecture: {result['architecture']}")
    config = payload["config"]
    model = HarmonicFoldNet(
        channels=len(channels), sample_rate=sample_rate,
        class_frequencies=frequencies, class_phases=phases,
        **config["harmonic_fold_v4_1_model"],
    )
    model.load_state_dict(payload["state_dict"])
    return model


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int(np.random.SeedSequence([seed, int.from_bytes(digest[:8], "little")]).generate_state(1)[0])


def _predict(
    model: nn.Module, x: np.ndarray, *, window: float, sample_rate: int,
    batch_size: int, device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for begin in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[begin:begin + batch_size]).to(device)
            logits = model(crop_tensor(batch, window, sample_rate))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def _adapt(
    base: HarmonicFoldNet, x: np.ndarray, y: np.ndarray, *,
    channels: int, windows: list[float], sample_rate: int, device: torch.device,
    tune_score: bool, epochs: int, learning_rate: float, identity_penalty: float,
    seed: int,
) -> SpatialScoreAdapter:
    seed_all(seed)
    wrapper = SpatialScoreAdapter(copy.deepcopy(base), channels, tune_score=tune_score).to(device)
    # Keep all frozen normalization statistics and dropout states fixed.  The
    # participant-specific update therefore changes only the declared adapter.
    wrapper.eval()
    optimizer = torch.optim.AdamW(
        wrapper.adapted_parameters(), lr=learning_rate, weight_decay=0.0,
    )
    loss_fn = nn.CrossEntropyLoss()
    full_x = torch.from_numpy(x).to(device)
    full_y = torch.from_numpy(y).to(device)
    rng = np.random.default_rng(seed)
    schedule: list[float] = []
    while len(schedule) < epochs:
        schedule.extend(rng.permutation(windows).tolist())
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        batch = crop_tensor(full_x, float(schedule[epoch]), sample_rate)
        loss = loss_fn(wrapper(batch), full_y)
        loss = loss + float(identity_penalty) * wrapper.channel_delta.square().mean()
        loss.backward()
        optimizer.step()
    return wrapper


def run(args: argparse.Namespace) -> dict[str, object]:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_cfg = config["datasets"][args.dataset]
    subjects = list(map(str, args.subjects or dataset_cfg["subjects"]))
    channels = list(map(str, dataset_cfg["channels"]))
    sample_rate = int(config["sample_rate"])
    classes = int(dataset_cfg["classes"])
    shard_dir = Path(dataset_cfg["shard_dir"])
    frequencies = class_frequencies(shard_dir, subjects, classes=classes)
    phases = class_phases(shard_dir, subjects, classes=classes)
    windows = list(map(float, args.windows or dataset_cfg["selection_windows"]))
    train_windows = list(map(float, args.train_windows or windows))
    counts = list(dict.fromkeys(map(int, args.calibration_blocks)))
    methods = list(dict.fromkeys(args.methods))
    index = _checkpoint_index(Path(args.checkpoint_root), args.dataset, args.seed)
    missing = sorted(set(subjects) - set(index), key=lambda value: (len(value), value))
    if missing:
        raise RuntimeError(f"subjects missing leakage-free outer-fold checkpoints: {missing}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cached: dict[Path, tuple[dict[str, object], HarmonicFoldNet]] = {}
    accumulated: dict[tuple[str, int, float], dict[str, object]] = {}

    for subject_index, subject in enumerate(subjects, 1):
        checkpoint = index[subject]
        if checkpoint not in cached:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            cached[checkpoint] = (
                payload,
                _make_model(
                    payload, channels=channels, frequencies=frequencies,
                    phases=phases, sample_rate=sample_rate,
                ),
            )
        payload, base = cached[checkpoint]
        if subject in set(map(str, payload["result"]["subjects"]["train"])):
            raise RuntimeError(f"source checkpoint for {subject} contains the target subject")
        x, y, block_ids = _load_subject(
            shard_dir, subject, channels=channels,
            trial_filters=dataset_cfg.get("trial_filters"),
        )
        blocks = sorted(np.unique(block_ids).tolist(), key=str)
        per_subject_predictions: dict[tuple[str, int, float], list[np.ndarray]] = {}
        per_subject_truth: dict[tuple[str, int, float], list[np.ndarray]] = {}
        for held_out_index, held_out in enumerate(blocks):
            test_mask = block_ids == held_out
            if held_out_index == 0:
                for window in windows:
                    zero_prediction = _predict(
                        base.to(device), x, window=window, sample_rate=sample_rate,
                        batch_size=args.batch_size, device=device,
                    )
                    # The unadapted model is evaluated once over all blocks.
                    key = ("zero_shot", 0, window)
                    per_subject_predictions.setdefault(key, []).append(zero_prediction)
                    per_subject_truth.setdefault(key, []).append(y)
                base.to("cpu")
            for count in counts:
                calibration = _preceding_calibration_blocks(blocks, held_out_index, count)
                train_mask = np.isin(block_ids, calibration)
                for method in methods:
                    tune_score = method == "spatial_score"
                    adapted = _adapt(
                        base, x[train_mask], y[train_mask], channels=len(channels),
                        windows=train_windows, sample_rate=sample_rate, device=device,
                        tune_score=tune_score, epochs=args.epochs,
                        learning_rate=args.learning_rate,
                        identity_penalty=args.identity_penalty,
                        seed=_stable_seed(args.seed, subject, held_out, count, method),
                    )
                    for window in windows:
                        prediction = _predict(
                            adapted, x[test_mask], window=window,
                            sample_rate=sample_rate, batch_size=args.batch_size,
                            device=device,
                        )
                        key = (method, count, window)
                        per_subject_predictions.setdefault(key, []).append(prediction)
                        per_subject_truth.setdefault(key, []).append(y[test_mask])
                    del adapted
        for key, parts in per_subject_predictions.items():
            truth = np.concatenate(per_subject_truth[key])
            prediction = np.concatenate(parts)
            metric = metrics_from_predictions(
                truth, prediction, classes=classes, window_seconds=key[2],
                cue_seconds=float(config["cue_seconds_for_itr"]),
            )
            destination = accumulated.setdefault(
                key, {"truth": [], "prediction": [], "per_subject": {}},
            )
            destination["truth"].append(truth)
            destination["prediction"].append(prediction)
            destination["per_subject"][subject] = metric
        print(json.dumps({
            "subject": subject, "index": subject_index, "total": len(subjects),
            "blocks": len(blocks), "checkpoint": str(checkpoint),
        }), flush=True)

    output: dict[str, object] = {
        "status": "complete", "dataset": args.dataset,
        "supervision_track": "participant_calibrated_neural",
        "outer_fold_contract": (
            "each participant uses a source checkpoint whose train and validation sets "
            "exclude that participant"
        ),
        "calibration_schedule": (
            "for each held-out block, use the requested immediately preceding blocks "
            "cyclically; each held-out block is predicted exactly once"
        ),
        "adaptation": {
            "methods": methods,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "identity_penalty": args.identity_penalty,
            "train_windows": train_windows,
            "spatial_parameters": len(channels) ** 2,
            "spatial_score_parameters": len(channels) ** 2 + 49,
        },
        "seed": args.seed,
        "methods": {},
    }
    for (method, count, window), payload in accumulated.items():
        truth = np.concatenate(payload["truth"])
        prediction = np.concatenate(payload["prediction"])
        output["methods"].setdefault(method, {}).setdefault(str(count), {})[str(window)] = {
            "overall": metrics_from_predictions(
                truth, prediction, classes=classes, window_seconds=window,
                cue_seconds=float(config["cue_seconds_for_itr"]),
            ),
            "per_subject": payload["per_subject"],
        }
    atomic_write_json(args.output, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_multidataset.yaml")
    parser.add_argument("--dataset", required=True, choices=("kim2025", "benchmark", "beta", "wearable"))
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--methods", nargs="+", default=["spatial", "spatial_score"], choices=("spatial", "spatial_score"))
    parser.add_argument("--calibration-blocks", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--windows", nargs="+", type=float)
    parser.add_argument("--train-windows", nargs="+", type=float)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--identity-penalty", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
