from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .calibrated_baselines import EnsembleTRCA, TDCA, TRCA
from .paper_data import class_frequencies, class_phases, crop_and_standardize
from .paper_metrics import atomic_write_json, metrics_from_predictions


def _load_subject(
    shard_dir: Path,
    subject: str,
    *,
    channels: list[str],
    trial_filters: dict[str, list[str]] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = shard_dir / f"sub-{subject}.npz"
    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        block_key = "block" if "block" in data.files else "session"
        blocks = np.asarray(data[block_key]).astype(str)
        mask = np.ones(len(y), dtype=bool)
        for key, accepted in (trial_filters or {}).items():
            mask &= np.isin(np.asarray(data[key]).astype(str), list(map(str, accepted)))
        available = [str(value) for value in np.asarray(data["channels"])]
        lookup = {name.upper(): index for index, name in enumerate(available)}
        missing = [name for name in channels if name.upper() not in lookup]
        if missing:
            raise RuntimeError(f"{path}: missing requested channels {missing}")
        indices = [lookup[name.upper()] for name in channels]
        return x[mask][:, indices], y[mask], blocks[mask]


def _preceding_calibration_blocks(
    blocks: list[str], held_out_index: int, count: int,
) -> list[str]:
    if count < 0:
        return [block for index, block in enumerate(blocks) if index != held_out_index]
    if count == 0 or count >= len(blocks):
        raise ValueError("calibration block count must be -1 or between 1 and B-1")
    return [blocks[(held_out_index - offset) % len(blocks)] for offset in range(1, count + 1)]


def run(args: argparse.Namespace) -> dict[str, object]:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_cfg = config["datasets"][args.dataset]
    shard_dir = Path(dataset_cfg["shard_dir"])
    subjects = list(map(str, args.subjects or dataset_cfg["subjects"]))
    channels = list(map(str, dataset_cfg["channels"]))
    sample_rate = int(config["sample_rate"])
    classes = int(dataset_cfg["classes"])
    frequencies = class_frequencies(shard_dir, subjects, classes=classes)
    phases = class_phases(shard_dir, subjects, classes=classes)
    windows = [
        float(value) for value in (args.windows or dataset_cfg["selection_windows"])
    ]
    calibration_counts = list(dict.fromkeys(map(int, args.calibration_blocks)))
    methods = list(dict.fromkeys(args.methods))
    accumulated: dict[tuple[str, int, float], dict[str, list[np.ndarray] | dict[str, dict]]] = {}
    for subject_index, subject in enumerate(subjects, 1):
        x, y, block_ids = _load_subject(
            shard_dir, subject, channels=channels,
            trial_filters=dataset_cfg.get("trial_filters"),
        )
        blocks = sorted(np.unique(block_ids).tolist(), key=str)
        subject_predictions: dict[tuple[str, int, float], list[np.ndarray]] = {}
        subject_truth: dict[tuple[str, int, float], list[np.ndarray]] = {}
        for held_out_index, held_out in enumerate(blocks):
            test_mask = block_ids == held_out
            for requested_count in calibration_counts:
                calibration = _preceding_calibration_blocks(
                    blocks, held_out_index, requested_count,
                )
                train_mask = np.isin(block_ids, calibration)
                resolved_count = len(calibration)
                for window in windows:
                    samples = int(round(window * sample_rate))
                    if samples + args.tdca_delay_samples > x.shape[-1]:
                        continue
                    if ({"trca", "etrca"} & set(methods)) and resolved_count >= 2:
                        train_x = crop_and_standardize(x[train_mask], window, sample_rate)
                        test_x = crop_and_standardize(x[test_mask], window, sample_rate)
                        for method in ("trca", "etrca"):
                            if method not in methods:
                                continue
                            estimator = TRCA if method == "trca" else EnsembleTRCA
                            prediction = estimator(
                                n_components=args.components,
                            ).fit(train_x, y[train_mask]).predict(test_x)
                            key = (method, requested_count, window)
                            subject_predictions.setdefault(key, []).append(prediction)
                            subject_truth.setdefault(key, []).append(y[test_mask])
                    if "tdca" in methods:
                        extended_seconds = (samples + args.tdca_delay_samples) / sample_rate
                        train_x = crop_and_standardize(
                            x[train_mask], extended_seconds, sample_rate,
                        )
                        test_x = crop_and_standardize(
                            x[test_mask], extended_seconds, sample_rate,
                        )
                        prediction = TDCA(
                            frequencies=frequencies, phases=phases,
                            sample_rate=sample_rate, samples=samples,
                            harmonics=args.harmonics,
                            delay_samples=args.tdca_delay_samples,
                            n_components=args.components,
                        ).fit(train_x, y[train_mask]).predict(test_x)
                        key = ("tdca", requested_count, window)
                        subject_predictions.setdefault(key, []).append(prediction)
                        subject_truth.setdefault(key, []).append(y[test_mask])
        for key, prediction_parts in subject_predictions.items():
            truth = np.concatenate(subject_truth[key])
            prediction = np.concatenate(prediction_parts)
            metric = metrics_from_predictions(
                truth, prediction, classes=classes, window_seconds=key[2],
                cue_seconds=float(config["cue_seconds_for_itr"]),
            )
            metric["trials"] = int(len(truth))
            destination = accumulated.setdefault(
                key, {"truth": [], "prediction": [], "per_subject": {}},
            )
            destination["truth"].append(truth)
            destination["prediction"].append(prediction)
            destination["per_subject"][subject] = metric
        print(json.dumps({
            "subject": subject, "index": subject_index, "total": len(subjects),
            "blocks": len(blocks),
        }), flush=True)

    output: dict[str, object] = {
        "status": "complete", "dataset": args.dataset,
        "supervision_track": "participant_calibrated",
        "calibration_schedule": (
            "for each held-out block, use the requested number of immediately preceding "
            "blocks cyclically; -1 uses every non-test block"
        ),
        "signal_contract": "same causal 6-45 Hz, 250 Hz, eight-channel shards as the main track",
        "methods": {},
    }
    for (method, count, window), payload in accumulated.items():
        truth = np.concatenate(payload["truth"])
        prediction = np.concatenate(payload["prediction"])
        method_result = output["methods"].setdefault(method, {})
        count_result = method_result.setdefault(str(count), {})
        overall = metrics_from_predictions(
                truth, prediction, classes=classes, window_seconds=window,
                cue_seconds=float(config["cue_seconds_for_itr"]),
            )
        overall["trials"] = int(len(truth))
        count_result[str(window)] = {
            "overall": overall,
            "per_subject": payload["per_subject"],
        }
    atomic_write_json(Path(args.output), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_multidataset.yaml")
    parser.add_argument(
        "--dataset", required=True,
        choices=("kim2025", "benchmark", "beta", "wearable"),
    )
    parser.add_argument(
        "--methods", nargs="+", default=["trca", "etrca", "tdca"],
        choices=("trca", "etrca", "tdca"),
    )
    parser.add_argument("--calibration-blocks", nargs="+", type=int, default=[1, 2, -1])
    parser.add_argument("--windows", nargs="+", type=float)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--harmonics", type=int, default=4)
    parser.add_argument("--tdca-delay-samples", type=int, default=5)
    parser.add_argument("--components", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
