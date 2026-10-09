from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, sosfiltfilt

from .paper_data import class_frequencies, crop_and_standardize, load_subjects_window
from .paper_metrics import atomic_write_json, metrics_from_predictions


def reference_bank(
    frequencies: list[float] | np.ndarray, samples: int, sample_rate: int, harmonics: int = 4,
) -> list[np.ndarray]:
    time_axis = np.arange(samples, dtype=np.float64) / float(sample_rate)
    references = []
    for frequency in frequencies:
        columns = []
        for harmonic in range(1, harmonics + 1):
            angle = 2.0 * np.pi * float(frequency) * harmonic * time_axis
            columns.extend((np.sin(angle), np.cos(angle)))
        references.append(np.stack(columns, axis=1))
    return references


def _inverse_sqrt(covariance: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    values, vectors = np.linalg.eigh(covariance)
    floor = epsilon * max(float(values.max()), 1.0)
    return (vectors * (1.0 / np.sqrt(np.maximum(values, floor)))) @ vectors.T


def cca_scores(
    x: np.ndarray, frequencies: list[float] | np.ndarray, sample_rate: int,
    *, harmonics: int = 4,
) -> np.ndarray:
    references = reference_bank(frequencies, x.shape[-1], sample_rate, harmonics)
    reference_terms = []
    for reference in references:
        centered = reference - reference.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        reference_terms.append((centered, _inverse_sqrt(covariance)))
    scores = np.empty((len(x), len(references)), dtype=np.float32)
    for trial_index, trial in enumerate(np.asarray(x, dtype=np.float64)):
        eeg = trial.T
        eeg -= eeg.mean(axis=0, keepdims=True)
        eeg_covariance = eeg.T @ eeg / max(len(eeg) - 1, 1)
        eeg_whitener = _inverse_sqrt(eeg_covariance)
        for class_index, (reference, reference_whitener) in enumerate(reference_terms):
            cross = eeg.T @ reference / max(len(eeg) - 1, 1)
            canonical = eeg_whitener @ cross @ reference_whitener
            scores[trial_index, class_index] = float(np.linalg.svd(canonical, compute_uv=False)[0])
    return scores


def fixed_harmonic_scores(
    x: np.ndarray, frequencies: list[float] | np.ndarray, sample_rate: int,
    *, harmonics: int = 4,
) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64), axis=-1)) ** 2
    power = spectrum.mean(axis=1)
    bins = power.shape[-1]
    frequency_axis = np.fft.rfftfreq(x.shape[-1], d=1.0 / sample_rate)
    scores = np.zeros((len(x), len(frequencies)), dtype=np.float64)
    for class_index, frequency in enumerate(frequencies):
        for harmonic in range(1, harmonics + 1):
            target = float(frequency) * harmonic
            if target >= frequency_axis[-1]:
                continue
            index = int(np.argmin(np.abs(frequency_axis - target)))
            neighbors = [value for value in range(max(0, index - 2), min(bins, index + 3)) if value != index]
            background = np.median(power[:, neighbors], axis=1) if neighbors else 1.0
            scores[:, class_index] += np.log((power[:, index] + 1e-12) / (background + 1e-12))
    return scores.astype(np.float32)


def fbcca_scores(
    x: np.ndarray, frequencies: list[float] | np.ndarray, sample_rate: int,
    *, harmonics: int = 4, subbands: tuple[float, ...] = (6.0, 12.0, 18.0, 24.0, 30.0),
    high_hz: float = 45.0,
) -> np.ndarray:
    combined = np.zeros((len(x), len(frequencies)), dtype=np.float64)
    valid_bands = [low for low in subbands if low < high_hz]
    for band_index, low_hz in enumerate(valid_bands, 1):
        sos = butter(4, (low_hz, high_hz), btype="bandpass", fs=sample_rate, output="sos")
        filtered = sosfiltfilt(sos, x, axis=-1).astype(np.float32)
        correlations = cca_scores(filtered, frequencies, sample_rate, harmonics=harmonics)
        weight = band_index ** -1.25 + 0.25
        combined += weight * correlations.astype(np.float64) ** 2
    return combined.astype(np.float32)


def run(args: argparse.Namespace) -> dict[str, object]:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_cfg = config["datasets"][args.dataset]
    shard_dir = Path(dataset_cfg["shard_dir"])
    subjects = list(map(str, dataset_cfg["subjects"]))
    test_subjects = (
        subjects if args.all_subjects else list(map(str, dataset_cfg["test_subjects"]))
    )
    sample_rate = int(config["sample_rate"])
    windows = [float(value) for value in dataset_cfg["windows"]]
    channels = list(map(str, dataset_cfg["channels"]))
    classes = int(dataset_cfg["classes"])
    frequencies = class_frequencies(shard_dir, subjects, classes=classes)
    x, y, subject_ids = load_subjects_window(
        shard_dir, test_subjects, window_seconds=max(windows), sample_rate=sample_rate,
        channels=channels, trial_filters=dataset_cfg.get("trial_filters"),
    )
    methods = {
        "harmonic": fixed_harmonic_scores,
        "cca": cca_scores,
        "fbcca": fbcca_scores,
    }
    selected_methods = args.methods or list(methods)
    result: dict[str, object] = {
        "status": "complete", "protocol_version": int(config["protocol_version"]),
        "dataset": args.dataset, "subjects": test_subjects, "channels": channels,
        "methods": {},
    }
    for method_name in selected_methods:
        method_result = {}
        for window in windows:
            window_x = crop_and_standardize(x, window, sample_rate)
            scores = methods[method_name](window_x, frequencies, sample_rate)
            predictions = scores.argmax(axis=1)
            overall = metrics_from_predictions(
                y, predictions, classes=classes, window_seconds=window,
                cue_seconds=float(config["cue_seconds_for_itr"]),
            )
            per_subject = {}
            for subject in test_subjects:
                mask = subject_ids.astype(str) == subject
                per_subject[subject] = metrics_from_predictions(
                    y[mask], predictions[mask], classes=classes, window_seconds=window,
                    cue_seconds=float(config["cue_seconds_for_itr"]),
                )
            method_result[str(window)] = {"overall": overall, "per_subject": per_subject}
            print(json.dumps({"method": method_name, "window": window, **overall}), flush=True)
        result["methods"][method_name] = method_result
    atomic_write_json(Path(args.run_dir) / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_multidataset.yaml")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("kim2025", "benchmark", "beta", "wearable", "dong2023"),
    )
    parser.add_argument("--methods", nargs="*", choices=("harmonic", "cca", "fbcca"))
    parser.add_argument(
        "--all-subjects", action="store_true",
        help="Evaluate every participant for paired out-of-fold comparisons.",
    )
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
