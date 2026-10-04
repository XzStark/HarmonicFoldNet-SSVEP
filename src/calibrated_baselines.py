from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh, qr


def _center(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=-1, keepdims=True)


def _regularized_eigenvectors(
    between: np.ndarray, within: np.ndarray,
) -> np.ndarray:
    between = 0.5 * (between + between.T)
    within = 0.5 * (within + within.T)
    scale = max(float(np.trace(within)) / max(len(within), 1), 1e-8)
    within = within + np.eye(len(within), dtype=np.float64) * (1e-6 * scale)
    values, vectors = eigh(between, within, check_finite=False)
    return vectors[:, np.argsort(values)[::-1]]


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 1e-12 else 0.0


@dataclass
class TRCA:
    """Participant-calibrated class-specific TRCA with a regularized GED.

    This is an original NumPy implementation of the method described by
    Nakanishi et al. (IEEE TBME, 2018), not copied third-party source.
    """

    n_components: int = 1

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TRCA":
        x = _center(x)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        filters = []
        templates = []
        for label in self.classes_:
            trials = x[y == label]
            if len(trials) < 2:
                raise ValueError("TRCA requires at least two calibration trials per class")
            summed = trials.sum(axis=0)
            within = np.einsum("nct,ndt->cd", trials, trials, optimize=True)
            between = summed @ summed.T - within
            filters.append(_regularized_eigenvectors(between, within))
            templates.append(trials.mean(axis=0))
        self.filters_ = np.stack(filters)
        self.templates_ = np.stack(templates)
        return self

    def scores(self, x: np.ndarray) -> np.ndarray:
        x = _center(x)
        result = np.empty((len(x), len(self.classes_)), dtype=np.float64)
        for class_index, template in enumerate(self.templates_):
            filters = self.filters_[class_index, :, : self.n_components]
            projected_x = np.einsum("cf,nct->nft", filters, x, optimize=True)
            projected_template = filters.T @ template
            for trial_index in range(len(x)):
                result[trial_index, class_index] = _correlation(
                    projected_x[trial_index], projected_template,
                )
        return result

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[self.scores(x).argmax(axis=1)]


@dataclass
class EnsembleTRCA(TRCA):
    """Ensemble TRCA (eTRCA) using all class filters for every class score."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "EnsembleTRCA":
        super().fit(x, y)
        self.ensemble_filter_ = np.concatenate(
            [matrix[:, : self.n_components] for matrix in self.filters_], axis=1,
        )
        return self

    def scores(self, x: np.ndarray) -> np.ndarray:
        x = _center(x)
        projected_x = np.einsum("cf,nct->nft", self.ensemble_filter_, x, optimize=True)
        result = np.empty((len(x), len(self.classes_)), dtype=np.float64)
        for class_index, template in enumerate(self.templates_):
            projected_template = self.ensemble_filter_.T @ template
            for trial_index in range(len(x)):
                result[trial_index, class_index] = _correlation(
                    projected_x[trial_index], projected_template,
                )
        return result


def sinusoidal_references(
    frequencies: list[float] | np.ndarray,
    samples: int,
    sample_rate: int,
    *,
    harmonics: int = 4,
    phases: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=np.float64)
    phases_array = np.zeros_like(frequencies) if phases is None else np.asarray(
        phases, dtype=np.float64,
    )
    time_axis = np.arange(samples, dtype=np.float64) / float(sample_rate)
    outputs = []
    for frequency, phase in zip(frequencies, phases_array, strict=True):
        rows = []
        for harmonic in range(1, harmonics + 1):
            angle = 2.0 * np.pi * harmonic * frequency * time_axis + harmonic * phase
            rows.extend((np.sin(angle), np.cos(angle)))
        outputs.append(np.stack(rows))
    return np.stack(outputs)


def _reference_projection(reference: np.ndarray) -> np.ndarray:
    q, _ = qr(reference.T, mode="economic", check_finite=False)
    return q @ q.T


def _delay_augment(
    x: np.ndarray,
    *,
    samples: int,
    delay_samples: int,
    projection: np.ndarray,
    training: bool,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape((-1, *x.shape[-2:]))
    trials, channels, points = x.shape
    if points < samples + delay_samples:
        raise ValueError("TDCA input does not contain the registered delay margin")
    augmented = np.zeros(
        (trials, channels * (delay_samples + 1), samples), dtype=np.float64,
    )
    for delay in range(delay_samples + 1):
        destination = slice(delay * channels, (delay + 1) * channels)
        if training:
            augmented[:, destination] = x[..., delay : delay + samples]
        else:
            augmented[:, destination, : samples - delay] = x[..., delay:samples]
    projected = augmented @ projection
    return np.concatenate((augmented, projected), axis=-1)


def _dsp_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _center(x)
    y = np.asarray(y, dtype=np.int64)
    classes = np.unique(y)
    global_mean = x.mean(axis=0)
    within = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    between = np.zeros_like(within)
    for label in classes:
        trials = x[y == label]
        class_mean = trials.mean(axis=0)
        residual = trials - class_mean
        within += np.einsum("nct,ndt->cd", residual, residual, optimize=True)
        difference = class_mean - global_mean
        between += len(trials) * (difference @ difference.T)
    return _regularized_eigenvectors(between, within), global_mean


@dataclass
class TDCA:
    """Participant-calibrated TDCA with target-reference projection.

    The implementation follows Liu et al.'s time-delay augmentation and
    discriminant spatial projection, with explicit regularization for the
    one-block calibration case.
    """

    frequencies: list[float] | np.ndarray
    sample_rate: int
    samples: int
    harmonics: int = 4
    delay_samples: int = 5
    n_components: int = 1
    phases: list[float] | np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TDCA":
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        references = sinusoidal_references(
            self.frequencies, self.samples, self.sample_rate,
            harmonics=self.harmonics, phases=self.phases,
        )
        self.projections_ = np.stack([
            _reference_projection(references[int(label)]) for label in self.classes_
        ])
        augmented_parts = []
        label_parts = []
        for class_index, label in enumerate(self.classes_):
            trials = x[y == label]
            augmented_parts.append(_delay_augment(
                trials, samples=self.samples, delay_samples=self.delay_samples,
                projection=self.projections_[class_index], training=True,
            ))
            label_parts.append(np.full(len(trials), label, dtype=np.int64))
        augmented = np.concatenate(augmented_parts)
        labels = np.concatenate(label_parts)
        self.spatial_filter_, self.global_mean_ = _dsp_fit(augmented, labels)
        projected = np.einsum(
            "cf,nct->nft", self.spatial_filter_, augmented - self.global_mean_,
            optimize=True,
        )
        self.templates_ = np.stack([
            projected[labels == label].mean(axis=0) for label in self.classes_
        ])
        return self

    def scores(self, x: np.ndarray) -> np.ndarray:
        result = np.empty((len(x), len(self.classes_)), dtype=np.float64)
        filters = self.spatial_filter_[:, : self.n_components]
        for class_index, projection in enumerate(self.projections_):
            augmented = _delay_augment(
                x, samples=self.samples, delay_samples=self.delay_samples,
                projection=projection, training=False,
            )
            features = np.einsum(
                "cf,nct->nft", filters, augmented - self.global_mean_, optimize=True,
            )
            template = self.templates_[class_index, : self.n_components]
            for trial_index in range(len(x)):
                result[trial_index, class_index] = _correlation(
                    features[trial_index], template,
                )
        return result

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[self.scores(x).argmax(axis=1)]
