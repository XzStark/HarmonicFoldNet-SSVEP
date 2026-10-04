from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np


REGISTERED_WINDOWS = (0.4, 0.6, 0.8, 1.0, 1.2, 2.0, 3.0, 5.0)


def crop_and_standardize(
    x: np.ndarray, window_seconds: float, sample_rate: int,
) -> np.ndarray:
    samples = int(round(float(window_seconds) * int(sample_rate)))
    if samples <= 0 or samples > x.shape[-1]:
        raise ValueError(
            f"window {window_seconds}s produces {samples} samples; source has {x.shape[-1]}"
        )
    cropped = np.asarray(x[..., :samples], dtype=np.float32).copy()
    mean = cropped.mean(axis=-1, keepdims=True)
    std = cropped.std(axis=-1, keepdims=True)
    return ((cropped - mean) / np.maximum(std, 1e-6)).astype(np.float32, copy=False)


def load_subjects_window(
    shard_dir: str | Path,
    subjects: list[str] | tuple[str, ...],
    *,
    window_seconds: float,
    sample_rate: int,
    channels: list[str] | tuple[str, ...] | None = None,
    trial_filters: dict[str, list[str] | tuple[str, ...] | set[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shard_dir = Path(shard_dir)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    subject_ids: list[np.ndarray] = []
    for subject in map(str, subjects):
        path = shard_dir / f"sub-{subject}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            shard_rate = int(np.asarray(data["sample_rate"]).item())
            if shard_rate != int(sample_rate):
                raise RuntimeError(f"{path}: sample rate {shard_rate} != {sample_rate}")
            x = np.asarray(data["x"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.int64)
            mask = np.ones(len(y), dtype=bool)
            for key, accepted in (trial_filters or {}).items():
                if key not in data.files:
                    raise RuntimeError(f"{path}: missing trial metadata {key!r}")
                mask &= np.isin(np.asarray(data[key]).astype(str), list(map(str, accepted)))
            if channels is not None:
                if "channels" not in data.files:
                    raise RuntimeError(f"{path}: missing channel metadata")
                available = [str(value) for value in np.asarray(data["channels"])]
                lookup = {name.upper(): index for index, name in enumerate(available)}
                missing = [name for name in channels if name.upper() not in lookup]
                if missing:
                    raise RuntimeError(f"{path}: missing requested channels {missing}")
                indices = [lookup[name.upper()] for name in channels]
                x = x[:, indices]
            x = crop_and_standardize(x[mask], window_seconds, sample_rate)
            y = y[mask]
            if not len(y):
                raise RuntimeError(f"{path}: trial filters selected no examples")
        xs.append(x)
        ys.append(y)
        subject_ids.append(np.full(len(y), subject, dtype=f"<U{max(2, len(subject))}"))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(subject_ids)


def class_frequencies(
    shard_dir: str | Path,
    subjects: list[str] | tuple[str, ...],
    *,
    classes: int = 40,
) -> list[float]:
    shard_dir = Path(shard_dir)
    resolved: dict[int, float] = {}
    for subject in map(str, subjects):
        with np.load(shard_dir / f"sub-{subject}.npz", allow_pickle=False) as data:
            for label, frequency in zip(data["y"], data["frequency_hz"], strict=True):
                label_i = int(label)
                frequency_f = float(frequency)
                previous = resolved.get(label_i)
                if previous is not None and not np.isclose(previous, frequency_f, atol=1e-4):
                    raise RuntimeError(
                        f"inconsistent frequency for class {label_i}: "
                        f"{previous} vs {frequency_f}"
                    )
                resolved[label_i] = frequency_f
    if set(resolved) != set(range(classes)):
        raise RuntimeError(f"frequency metadata does not cover {classes} classes")
    return [resolved[index] for index in range(classes)]


def class_phases(
    shard_dir: str | Path,
    subjects: list[str] | tuple[str, ...],
    *,
    classes: int,
) -> list[float]:
    shard_dir = Path(shard_dir)
    resolved: dict[int, float] = {}
    for subject in map(str, subjects):
        with np.load(shard_dir / f"sub-{subject}.npz", allow_pickle=False) as data:
            if "phase_rad" not in data.files:
                raise RuntimeError(f"{shard_dir / f'sub-{subject}.npz'}: missing phase_rad")
            for label, phase in zip(data["y"], data["phase_rad"], strict=True):
                label_i = int(label)
                phase_f = float(phase) % (2.0 * np.pi)
                previous = resolved.get(label_i)
                if previous is not None and not np.isclose(previous, phase_f, atol=1e-4):
                    raise RuntimeError(f"inconsistent phase for class {label_i}: {previous} vs {phase_f}")
                resolved[label_i] = phase_f
    if set(resolved) != set(range(classes)):
        raise RuntimeError(f"phase metadata does not cover {classes} classes")
    return [resolved[index] for index in range(classes)]


def loso_split(
    subjects: list[str] | tuple[str, ...],
    test_subject: str,
    *,
    validation_count: int,
    split_seed: int,
) -> tuple[list[str], list[str], list[str]]:
    ordered = [str(subject) for subject in subjects]
    test_subject = str(test_subject)
    if test_subject not in ordered:
        raise ValueError(f"unknown test subject: {test_subject}")
    candidates = [subject for subject in ordered if subject != test_subject]
    if validation_count <= 0 or validation_count >= len(candidates):
        raise ValueError("validation_count leaves no training participants")
    digest = hashlib.sha256(test_subject.encode("utf-8")).digest()
    subject_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(np.random.SeedSequence([int(split_seed), subject_seed]))
    shuffled = np.asarray(candidates, dtype=object)
    rng.shuffle(shuffled)
    order = {subject: index for index, subject in enumerate(ordered)}
    validation = sorted(map(str, shuffled[:validation_count]), key=order.__getitem__)
    validation_set = set(validation)
    train = [subject for subject in ordered if subject != test_subject and subject not in validation_set]
    return train, validation, [test_subject]


def grouped_kfold_split(
    subjects: list[str] | tuple[str, ...], *, fold_index: int, fold_count: int,
    validation_count: int, split_seed: int,
) -> tuple[list[str], list[str], list[str]]:
    ordered = [str(subject) for subject in subjects]
    if fold_count < 2 or fold_count > len(ordered):
        raise ValueError("invalid fold_count")
    if fold_index < 0 or fold_index >= fold_count:
        raise ValueError("fold_index is zero-based and must be smaller than fold_count")
    rng = np.random.default_rng(int(split_seed))
    shuffled = np.asarray(ordered, dtype=object)
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, fold_count)
    test_set = set(map(str, folds[fold_index]))
    remaining = [subject for subject in ordered if subject not in test_set]
    if validation_count <= 0 or validation_count >= len(remaining):
        raise ValueError("validation_count leaves no training participants")
    validation_rng = np.random.default_rng(
        np.random.SeedSequence([int(split_seed), int(fold_index), int(fold_count)])
    )
    validation_candidates = np.asarray(remaining, dtype=object)
    validation_rng.shuffle(validation_candidates)
    validation_set = set(map(str, validation_candidates[:validation_count]))
    train = [subject for subject in ordered if subject not in test_set | validation_set]
    validation = [subject for subject in ordered if subject in validation_set]
    test = [subject for subject in ordered if subject in test_set]
    return train, validation, test
