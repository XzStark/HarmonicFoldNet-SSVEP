from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score


def classification_metrics(
    y: np.ndarray,
    logits: np.ndarray,
    *,
    classes: int,
    window_seconds: float,
    cue_seconds: float,
) -> dict[str, float]:
    predictions = logits.argmax(axis=1)
    probabilities = _softmax(logits)
    accuracy = float(accuracy_score(y, predictions))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "top5_accuracy": float(
            top_k_accuracy_score(y, probabilities, k=min(5, classes), labels=np.arange(classes))
        ),
        "itr_bits_per_minute": float(
            information_transfer_rate(
                classes, accuracy, float(window_seconds) + float(cue_seconds),
            )
        ),
    }


def metrics_from_predictions(
    y: np.ndarray,
    predictions: np.ndarray,
    *,
    classes: int,
    window_seconds: float,
    cue_seconds: float,
) -> dict[str, float]:
    accuracy = float(accuracy_score(y, predictions))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "itr_bits_per_minute": float(
            information_transfer_rate(
                classes, accuracy, float(window_seconds) + float(cue_seconds),
            )
        ),
    }


def information_transfer_rate(classes: int, accuracy: float, seconds: float) -> float:
    if classes < 2 or seconds <= 0:
        raise ValueError("classes must be >=2 and seconds must be positive")
    chance = 1.0 / classes
    if accuracy <= chance:
        return 0.0
    if accuracy >= 1.0:
        bits = math.log2(classes)
    else:
        bits = (
            math.log2(classes)
            + accuracy * math.log2(accuracy)
            + (1.0 - accuracy) * math.log2((1.0 - accuracy) / (classes - 1))
        )
    return 60.0 * bits / seconds


def atomic_write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)
