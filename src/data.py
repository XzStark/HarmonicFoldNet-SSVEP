from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


FREQUENCIES = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float32)


@dataclass(frozen=True)
class WindowSet:
    x: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    artifact: np.ndarray
    frequency_hz: np.ndarray
    sample_rate: int


def _paired_blocks(events: pd.DataFrame):
    """Yield (start, stop, class_index, artifact) from paired BIDS markers."""
    pending: dict[int, float] = {}
    for row in events.itertuples(index=False):
        value = int(row.value)
        if value == 10 or value < 1 or value > 8:
            continue
        onset = float(row.onset)
        if value not in pending:
            pending[value] = onset
            continue
        start = pending.pop(value)
        if onset <= start:
            continue
        artifact = value >= 5
        class_index = (value - 5) if artifact else (value - 1)
        yield start, onset, class_index, artifact


def _normalize_window(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-5)


def build_ds004745_windows(
    root: str | Path,
    *,
    target_sfreq: int = 250,
    window_seconds: float = 2.0,
    stride_seconds: float = 1.0,
    onset_guard_seconds: float = 1.0,
    offset_guard_seconds: float = 1.0,
    bandpass_hz: tuple[float, float] = (0.5, 45.0),
) -> WindowSet:
    root = Path(root)
    xs, ys, subjects, artifacts, freqs = [], [], [], [], []
    samples = int(round(window_seconds * target_sfreq))
    stride = max(1, int(round(stride_seconds * target_sfreq)))
    sos = butter(4, bandpass_hz, btype="bandpass", fs=target_sfreq, output="sos")

    for set_path in sorted(root.glob("sub-*/eeg/*_eeg.set")):
        subject = set_path.parts[-3].removeprefix("sub-")
        event_path = set_path.with_name(set_path.name.replace("_eeg.set", "_events.tsv"))
        events = pd.read_csv(event_path, sep="\t")
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
        if int(round(raw.info["sfreq"])) != target_sfreq:
            raw.resample(target_sfreq, npad="auto", verbose="ERROR")
        signal = raw.get_data().astype(np.float32, copy=False)
        for start_s, stop_s, label, artifact in _paired_blocks(events):
            first = int(round((start_s + onset_guard_seconds) * target_sfreq))
            last = int(round((stop_s - offset_guard_seconds) * target_sfreq))
            for begin in range(first, last - samples + 1, stride):
                window = signal[:, begin:begin + samples]
                if window.shape[-1] != samples or not np.isfinite(window).all():
                    continue
                window = sosfiltfilt(sos, window, axis=-1).astype(np.float32)
                xs.append(_normalize_window(window))
                ys.append(label)
                subjects.append(subject)
                artifacts.append(artifact)
                freqs.append(FREQUENCIES[label])
    if not xs:
        raise RuntimeError(f"no windows found under {root}")
    return WindowSet(
        x=np.stack(xs), y=np.asarray(ys, dtype=np.int64),
        subject=np.asarray(subjects), artifact=np.asarray(artifacts, dtype=np.bool_),
        frequency_hz=np.asarray(freqs, dtype=np.float32), sample_rate=target_sfreq,
    )


def save_windows(path: str | Path, windows: WindowSet) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, x=windows.x, y=windows.y, subject=windows.subject,
        artifact=windows.artifact, frequency_hz=windows.frequency_hz,
        sample_rate=np.asarray(windows.sample_rate, dtype=np.int32),
    )


def load_windows(path: str | Path) -> WindowSet:
    with np.load(path, allow_pickle=False) as data:
        return WindowSet(
            x=data["x"], y=data["y"], subject=data["subject"],
            artifact=data["artifact"], frequency_hz=data["frequency_hz"],
            sample_rate=int(data["sample_rate"]),
        )


def describe(windows: WindowSet) -> dict:
    return {
        "shape": list(windows.x.shape),
        "sample_rate": windows.sample_rate,
        "subjects": sorted(set(windows.subject.tolist())),
        "class_counts": {str(float(FREQUENCIES[i])): int((windows.y == i).sum()) for i in range(4)},
        "artifact_windows": int(windows.artifact.sum()),
        "clean_windows": int((~windows.artifact).sum()),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_ds004745_windows(args.root)
    save_windows(args.output, result)
    print(json.dumps(describe(result), ensure_ascii=False, indent=2))
