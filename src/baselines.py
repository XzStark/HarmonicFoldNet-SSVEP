from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import CCA


def reference_bank(frequencies, samples: int, sample_rate: int, harmonics: int = 4):
    t = np.arange(samples, dtype=np.float64) / float(sample_rate)
    bank = []
    for freq in frequencies:
        rows = []
        for harmonic in range(1, harmonics + 1):
            rows.extend([
                np.sin(2 * np.pi * freq * harmonic * t),
                np.cos(2 * np.pi * freq * harmonic * t),
            ])
        bank.append(np.asarray(rows))
    return bank


def cca_predict(x: np.ndarray, frequencies, sample_rate: int, harmonics: int = 4):
    refs = reference_bank(frequencies, x.shape[-1], sample_rate, harmonics)
    predictions = []
    for trial in x:
        scores = []
        for ref in refs:
            cca = CCA(n_components=1, max_iter=500)
            a, b = cca.fit_transform(trial.T, ref.T)
            scores.append(float(np.corrcoef(a[:, 0], b[:, 0])[0, 1]))
        predictions.append(int(np.nanargmax(scores)))
    return np.asarray(predictions, dtype=np.int64)
