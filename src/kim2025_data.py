from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


DEFAULT_CHANNELS = ("PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2")


def _normalize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def build_subject_shard(
    subject_dir: str | Path,
    output: str | Path,
    *,
    metadata_root: str | Path | None = None,
    channels: tuple[str, ...] = DEFAULT_CHANNELS,
    target_sfreq: int = 250,
    onset_delay_seconds: float = 0.0,
    window_seconds: float = 5.0,
    bandpass_hz: tuple[float, float] = (6.0, 45.0),
) -> dict:
    subject_dir = Path(subject_dir)
    output = Path(output)
    metadata_root = Path(metadata_root) if metadata_root is not None else None
    xs: list[np.ndarray] = []
    ys: list[int] = []
    frequencies: list[float] = []
    sessions: list[str] = []
    samples = int(round(window_seconds * target_sfreq))
    sos = butter(4, bandpass_hz, btype="bandpass", fs=target_sfreq, output="sos")

    for set_path in sorted(subject_dir.glob("ses-*/eeg/*_eeg.set")):
        session = next(part for part in set_path.parts if part.startswith("ses-"))
        events_path = set_path.with_name(set_path.name.replace("_eeg.set", "_events.tsv"))
        if not events_path.exists() and metadata_root is not None:
            relative = set_path.relative_to(subject_dir.parent)
            events_path = (metadata_root / relative).with_name(
                set_path.name.replace("_eeg.set", "_events.tsv")
            )
        if not events_path.exists():
            raise FileNotFoundError(f"event sidecar not found for {set_path}")
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
        missing = [name for name in channels if name not in raw.ch_names]
        if missing:
            raise RuntimeError(f"{set_path}: missing channels {missing}")
        raw.pick(list(channels))
        if int(round(raw.info["sfreq"])) != target_sfreq:
            raw.resample(target_sfreq, npad="auto", verbose="ERROR")
        signal = raw.get_data().astype(np.float32, copy=False)
        signal = sosfiltfilt(sos, signal, axis=-1).astype(np.float32)
        events = pd.read_csv(events_path, sep="\t")
        for row in events.itertuples(index=False):
            label = int(row.value) - 1
            if label < 0 or label >= 40:
                continue
            start = int(round((float(row.onset) + onset_delay_seconds) * target_sfreq))
            window = signal[:, start:start + samples]
            if window.shape != (len(channels), samples) or not np.isfinite(window).all():
                continue
            xs.append(_normalize(window))
            ys.append(label)
            frequencies.append(float(row.trial_type))
            sessions.append(session)

    if not xs:
        raise RuntimeError(f"no usable trials under {subject_dir}")
    stacked = np.stack(xs)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, x=stacked, y=np.asarray(ys, dtype=np.int64),
        frequency_hz=np.asarray(frequencies, dtype=np.float32),
        session=np.asarray(sessions), channels=np.asarray(channels),
        sample_rate=np.asarray(target_sfreq, dtype=np.int32),
        subject=np.asarray(subject_dir.name.removeprefix("sub-")),
    )
    return {
        "subject": subject_dir.name, "trials": len(xs),
        "sessions": sorted(set(sessions)), "shape": list(stacked.shape),
        "output": str(output),
    }


def build_all(
    root: str | Path, output_dir: str | Path, *,
    metadata_root: str | Path | None = None, force: bool = False,
) -> list[dict]:
    root = Path(root)
    output_dir = Path(output_dir)
    subjects = sorted(root.glob("sub-*"), key=lambda path: int(path.name.removeprefix("sub-")))
    if not subjects:
        raise RuntimeError(f"no sub-* directories found under {root}")
    summaries = []
    for subject_dir in subjects:
        shard = output_dir / f"{subject_dir.name}.npz"
        if shard.exists() and not force:
            with np.load(shard, allow_pickle=False) as data:
                summaries.append({
                    "subject": subject_dir.name, "trials": int(len(data["y"])),
                    "shape": list(data["x"].shape), "output": str(shard),
                    "status": "existing",
                })
            continue
        summary = build_subject_shard(subject_dir, shard, metadata_root=metadata_root)
        summary["status"] = "built"
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    manifest = {
        "source_root": str(root.resolve()), "subjects": summaries,
        "total_trials": sum(item["trials"] for item in summaries),
        "channels": list(DEFAULT_CHANNELS), "sample_rate": 250,
        "window_seconds": 5.0, "onset_delay_seconds": 0.0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata-root")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_all(
        args.root, args.output_dir, metadata_root=args.metadata_root, force=args.force,
    )
    print(json.dumps({"subjects": len(result), "trials": sum(x["trials"] for x in result)}, indent=2))
