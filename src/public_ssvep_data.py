from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt


POSTERIOR_8 = ("PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2")
WEARABLE_8 = ("POz", "PO3", "PO4", "PO5", "PO6", "Oz", "O1", "O2")
COMMON_6 = ("PO3", "POz", "PO4", "O1", "Oz", "O2")
WEARABLE_FREQUENCIES = np.asarray(
    [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75],
    dtype=np.float32,
)
WEARABLE_PHASES = np.asarray(
    [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5],
    dtype=np.float32,
) * np.pi
DONG2023_CHANNELS = ("POz", "PO3", "PO4", "PO7", "PO8", "Oz", "O1", "O2")
DONG2023_FREQUENCIES = np.arange(8.0, 16.0, 0.2, dtype=np.float32)
DONG2023_PHASES = (np.arange(40, dtype=np.float32) % 4) * (0.5 * np.pi)


def _canonical(name: str) -> str:
    return str(name).strip().upper()


def _indices(all_channels: list[str] | tuple[str, ...], selected: tuple[str, ...]) -> list[int]:
    lookup = {_canonical(name): index for index, name in enumerate(all_channels)}
    missing = [name for name in selected if _canonical(name) not in lookup]
    if missing:
        raise RuntimeError(f"missing channels: {missing}")
    return [lookup[_canonical(name)] for name in selected]


def _causal_bandpass(epochs: np.ndarray, sample_rate: int = 250) -> np.ndarray:
    sos = butter(4, (6.0, 45.0), btype="bandpass", fs=sample_rate, output="sos")
    return sosfilt(sos, np.asarray(epochs, dtype=np.float32), axis=-1).astype(np.float32)


def _atomic_savez(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    # Windows rejects fsync on a read-only descriptor; r+b keeps the operation
    # durable without modifying the already completed NumPy archive.
    with temporary.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_manifest(output_dir: Path, value: dict[str, object]) -> None:
    temporary = output_dir / ".manifest.json.tmp"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output_dir / "manifest.json")


def _benchmark_channels(loc_path: Path) -> list[str]:
    channels = []
    for line in loc_path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 4:
            channels.append(fields[-1])
    if len(channels) != 64:
        raise RuntimeError(f"expected 64 Benchmark channels, found {len(channels)}")
    return channels


def preprocess_benchmark(root: Path, output_dir: Path, *, force: bool = False) -> dict[str, object]:
    frequencies = np.asarray(loadmat(root / "Freq_Phase.mat", squeeze_me=True)["freqs"], dtype=np.float32)
    phases = np.asarray(loadmat(root / "Freq_Phase.mat", squeeze_me=True)["phases"], dtype=np.float32)
    channels = _benchmark_channels(root / "64-channels.loc")
    channel_indices = _indices(channels, POSTERIOR_8)
    rows = []
    for subject in range(1, 36):
        source = root / "raw" / f"S{subject}.mat"
        destination = output_dir / f"sub-{subject}.npz"
        if destination.exists() and not force:
            rows.append({"subject": str(subject), "status": "existing"})
            continue
        raw = np.asarray(loadmat(source, variable_names=["data"])["data"])
        if raw.shape != (64, 1500, 40, 6):
            raise RuntimeError(f"{source}: unexpected shape {raw.shape}")
        epochs = raw[channel_indices].transpose(2, 3, 0, 1).reshape(240, 8, 1500)
        epochs = _causal_bandpass(epochs)
        epochs = epochs[..., 160:1410]  # 0.5 s pre-stimulus + 0.14 s response latency
        labels = np.repeat(np.arange(40, dtype=np.int64), 6)
        blocks = np.tile(np.arange(6, dtype=np.int16), 40)
        _atomic_savez(
            destination, x=epochs, y=labels, frequency_hz=frequencies[labels],
            phase_rad=phases[labels], block=blocks, channels=np.asarray(POSTERIOR_8),
            sample_rate=np.asarray(250, dtype=np.int32), subject=np.asarray(str(subject)),
            dataset=np.asarray("TsinghuaBenchmark"), filter_mode=np.asarray("causal_sos_order4_6_45Hz"),
        )
        rows.append({"subject": str(subject), "trials": 240, "samples": 1250, "status": "built"})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    manifest = {
        "dataset": "TsinghuaBenchmark", "participants": 35, "classes": 40,
        "trials_per_participant": 240, "channels": list(POSTERIOR_8),
        "common_channels": list(COMMON_6), "sample_rate": 250,
        "response_latency_seconds": 0.14, "filter": "causal SOS order 4, 6-45 Hz",
        "subjects": rows,
    }
    _write_manifest(output_dir, manifest)
    return manifest


def preprocess_beta(root: Path, output_dir: Path, *, force: bool = False) -> dict[str, object]:
    rows = []
    reference_frequencies: np.ndarray | None = None
    reference_phases: np.ndarray | None = None
    for subject in range(1, 71):
        source = root / "raw" / f"S{subject}.mat"
        destination = output_dir / f"sub-{subject}.npz"
        if destination.exists() and not force:
            rows.append({"subject": str(subject), "status": "existing"})
            continue
        container = loadmat(source, squeeze_me=True, struct_as_record=False, variable_names=["data"])["data"]
        raw = np.asarray(container.EEG)
        if raw.ndim != 4 or raw.shape[0] != 64 or raw.shape[2:] != (4, 40):
            raise RuntimeError(f"{source}: unexpected EEG shape {raw.shape}")
        channel_names = [str(row[-1]) for row in np.asarray(container.suppl_info.chan)]
        channel_indices = _indices(channel_names, POSTERIOR_8)
        frequencies = np.asarray(container.suppl_info.freqs, dtype=np.float32)
        phases = np.asarray(container.suppl_info.phases, dtype=np.float32)
        if reference_frequencies is None:
            reference_frequencies, reference_phases = frequencies, phases
        elif not (np.allclose(frequencies, reference_frequencies) and np.allclose(phases, reference_phases)):
            raise RuntimeError(f"{source}: class metadata differs from prior subjects")
        epochs = raw[channel_indices].transpose(3, 2, 0, 1).reshape(160, 8, raw.shape[1])
        epochs = _causal_bandpass(epochs)
        stimulus_samples = 500 if subject <= 15 else 750
        epochs = epochs[..., 160:125 + stimulus_samples]
        labels = np.repeat(np.arange(40, dtype=np.int64), 4)
        blocks = np.tile(np.arange(4, dtype=np.int16), 40)
        _atomic_savez(
            destination, x=epochs, y=labels, frequency_hz=frequencies[labels],
            phase_rad=phases[labels], block=blocks, channels=np.asarray(POSTERIOR_8),
            sample_rate=np.asarray(250, dtype=np.int32), subject=np.asarray(str(subject)),
            dataset=np.asarray("BETA"), filter_mode=np.asarray("causal_sos_order4_6_45Hz"),
        )
        rows.append({
            "subject": str(subject), "trials": 160, "samples": int(epochs.shape[-1]), "status": "built",
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    manifest = {
        "dataset": "BETA", "participants": 70, "classes": 40,
        "trials_per_participant": 160, "channels": list(POSTERIOR_8),
        "common_channels": list(COMMON_6), "sample_rate": 250,
        "response_latency_seconds": 0.14, "filter": "causal SOS order 4, 6-45 Hz",
        "note": "S1-S15 contain 465 post-latency stimulus samples; S16-S70 contain 715.",
        "subjects": rows,
    }
    _write_manifest(output_dir, manifest)
    return manifest


def preprocess_wearable(root: Path, output_dir: Path, *, force: bool = False) -> dict[str, object]:
    rows = []
    for subject in range(1, 103):
        source = root / "raw" / f"S{subject:03d}.mat"
        destination = output_dir / f"sub-{subject}.npz"
        if destination.exists() and not force:
            rows.append({"subject": str(subject), "status": "existing"})
            continue
        raw = np.asarray(loadmat(source, variable_names=["data"])["data"])
        if raw.shape != (8, 710, 2, 10, 12):
            raise RuntimeError(f"{source}: unexpected shape {raw.shape}")
        epochs = raw.transpose(2, 4, 3, 0, 1).reshape(240, 8, 710)
        epochs = _causal_bandpass(epochs)[..., 160:660]
        labels = np.tile(np.repeat(np.arange(12, dtype=np.int64), 10), 2)
        blocks = np.tile(np.arange(10, dtype=np.int16), 24)
        electrode = np.repeat(np.asarray(["dry", "wet"]), 120)
        _atomic_savez(
            destination, x=epochs, y=labels, frequency_hz=WEARABLE_FREQUENCIES[labels],
            phase_rad=WEARABLE_PHASES[labels], block=blocks, electrode=electrode,
            channels=np.asarray(WEARABLE_8), sample_rate=np.asarray(250, dtype=np.int32),
            subject=np.asarray(str(subject)), dataset=np.asarray("WearableSSVEP"),
            filter_mode=np.asarray("causal_sos_order4_6_45Hz"),
        )
        rows.append({"subject": str(subject), "trials": 240, "samples": 500, "status": "built"})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    manifest = {
        "dataset": "WearableSSVEP", "participants": 102, "classes": 12,
        "trials_per_participant": 240, "channels": list(WEARABLE_8),
        "common_channels": list(COMMON_6), "sample_rate": 250,
        "conditions": ["dry", "wet"], "response_latency_seconds": 0.14,
        "filter": "causal SOS order 4, 6-45 Hz", "subjects": rows,
    }
    _write_manifest(output_dir, manifest)
    return manifest


def preprocess_dong2023(
    root: Path,
    output_dir: Path,
    *,
    force: bool = False,
    subjects: list[int] | tuple[int, ...] | range = range(1, 60),
) -> dict[str, object]:
    """Convert Dong2023 MATLAB records to the common posterior-eight contract.

    The public epochs contain 0.5 s before stimulation, 4 s of stimulation and
    0.5 s after stimulation at 250 Hz.  Dong and Tian report a mean visual
    latency of approximately 160 ms and use it for offline analysis.  The model
    shards therefore start 160 ms after stimulus onset and stop at stimulus
    offset; no post-stimulus sample is retained.
    """
    rows: list[dict[str, object]] = []
    expected_shape = (8, 1250, 40, 4)
    response_start = int(round((0.5 + 0.16) * 250))
    response_stop = int(round((0.5 + 4.0) * 250))
    channel_indices = _indices(DONG2023_CHANNELS, POSTERIOR_8)
    for subject in map(int, subjects):
        source = root / f"S{subject}.mat"
        destination = output_dir / f"sub-{subject}.npz"
        if destination.exists() and not force:
            rows.append({"subject": str(subject), "status": "existing"})
            continue
        if not source.exists():
            raise FileNotFoundError(source)
        raw = np.asarray(loadmat(source, variable_names=["eegdata"])["eegdata"])
        if raw.shape != expected_shape:
            raise RuntimeError(f"{source}: unexpected eegdata shape {raw.shape}")
        epochs = raw[channel_indices].transpose(2, 3, 0, 1).reshape(160, 8, 1250)
        epochs = _causal_bandpass(epochs)[..., response_start:response_stop]
        labels = np.repeat(np.arange(40, dtype=np.int64), 4)
        blocks = np.tile(np.arange(4, dtype=np.int16), 40)
        _atomic_savez(
            destination,
            x=epochs,
            y=labels,
            frequency_hz=DONG2023_FREQUENCIES[labels],
            phase_rad=DONG2023_PHASES[labels],
            block=blocks,
            channels=np.asarray(POSTERIOR_8),
            sample_rate=np.asarray(250, dtype=np.int32),
            subject=np.asarray(str(subject)),
            dataset=np.asarray("Dong2023"),
            filter_mode=np.asarray("causal_sos_order4_6_45Hz"),
        )
        rows.append(
            {
                "subject": str(subject),
                "trials": 160,
                "samples": int(epochs.shape[-1]),
                "status": "built",
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    manifest_rows: list[dict[str, object]] = []
    existing_paths = sorted(
        output_dir.glob("sub-*.npz"),
        key=lambda path: int(path.stem.split("-", 1)[1]),
    )
    for path in existing_paths:
        with np.load(path, allow_pickle=False) as shard:
            manifest_rows.append(
                {
                    "subject": str(np.asarray(shard["subject"]).item()),
                    "trials": int(np.asarray(shard["y"]).size),
                    "samples": int(np.asarray(shard["x"]).shape[-1]),
                    "status": "processed",
                }
            )
    manifest = {
        "dataset": "Dong2023",
        "source_doi": "10.26599/BSA.2023.9050020",
        "source_record": "https://zenodo.org/records/18847318",
        "license": "CC BY-NC 4.0; research use only",
        "participants_expected": 59,
        "participants_processed": len(existing_paths),
        "classes": 40,
        "trials_per_participant": 160,
        "channels": list(POSTERIOR_8),
        "sample_rate": 250,
        "response_latency_seconds": 0.16,
        "usable_response_seconds": (response_stop - response_start) / 250,
        "filter": "causal SOS order 4, 6-45 Hz",
        "note": (
            "The source paper reports significantly lower SNR than the Tsinghua "
            "Benchmark; this is a practical low-SNR external domain, not an "
            "SNR-matched laboratory replication."
        ),
        "subjects": manifest_rows,
    }
    _write_manifest(output_dir, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", choices=("benchmark", "beta", "wearable", "dong2023")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--subjects", nargs="+", type=int)
    args = parser.parse_args()
    builders = {
        "benchmark": preprocess_benchmark,
        "beta": preprocess_beta,
        "wearable": preprocess_wearable,
        "dong2023": preprocess_dong2023,
    }
    options: dict[str, object] = {"force": args.force}
    if args.subjects is not None:
        if args.dataset != "dong2023":
            parser.error("--subjects is currently supported only for dong2023")
        options["subjects"] = args.subjects
    result = builders[args.dataset](args.root, args.output_dir, **options)
    print(json.dumps({key: value for key, value in result.items() if key != "subjects"}, indent=2))


if __name__ == "__main__":
    main()
