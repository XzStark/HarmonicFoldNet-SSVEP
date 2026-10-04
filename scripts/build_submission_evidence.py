from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from src.paper_metrics import atomic_write_json, information_transfer_rate
from src.paper_statistics import (
    holm_adjust,
    merge_out_of_fold_documents,
    paired_test,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "paper" / "SUBMISSION_EVIDENCE_CONFIG.json"
DEFAULT_OUTPUT = ROOT / "paper" / "source_data"
EXPECTED_SEEDS = [20260929, 20260930, 20260931]
METRICS = ("accuracy", "balanced_accuracy", "itr_bits_per_minute")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _expand(patterns: Iterable[str]) -> tuple[list[Path], list[dict]]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(value) for value in glob.glob(str(ROOT / pattern)))
        if not matches:
            raise FileNotFoundError(f"input pattern matched no files: {pattern}")
        paths.extend(matches)
    unique = sorted(set(paths))
    documents = [_load_json(path) for path in unique]
    return unique, documents


def _validate_group(
    documents: list[dict], *, participants: int, windows: list[str], label: str
) -> None:
    seeds = sorted({int(document["seed"]) for document in documents})
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"{label}: expected seeds {EXPECTED_SEEDS}, found {seeds}")
    for window in windows:
        subjects, _, merged_seeds = merge_out_of_fold_documents(
            documents, window=window, metric="balanced_accuracy"
        )
        if len(subjects) != participants:
            raise ValueError(
                f"{label}/{window}: expected {participants} participants, found {len(subjects)}"
            )
        if merged_seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"{label}/{window}: expected seeds {EXPECTED_SEEDS}, found {merged_seeds}"
            )


def _bootstrap_difference_ci(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _condition_matrices(
    documents: list[dict], windows: list[str]
) -> dict[tuple[str, str], tuple[list[str], np.ndarray, list[int]]]:
    matrices = {}
    for window in windows:
        for metric in METRICS:
            matrices[(window, metric)] = merge_out_of_fold_documents(
                documents, window=window, metric=metric
            )
    return matrices


def _validate_against_frozen(
    frozen: dict,
    matrices: dict[str, dict[tuple[str, str], tuple[list[str], np.ndarray, list[int]]]],
    windows: list[str],
    *,
    tolerance: float = 1e-12,
) -> None:
    if frozen.get("metric") != "balanced_accuracy":
        raise ValueError("frozen comparison is not a balanced-accuracy artifact")
    for condition, condition_matrices in matrices.items():
        for window in windows:
            subjects, seed_matrix, seeds = condition_matrices[(window, "balanced_accuracy")]
            frozen_row = frozen["conditions"][condition][window]
            if subjects != [str(value) for value in frozen_row["participants"]]:
                raise ValueError(f"{condition}/{window}: participant order differs from frozen artifact")
            if seeds != [int(value) for value in frozen_row["seeds"]]:
                raise ValueError(f"{condition}/{window}: seeds differ from frozen artifact")
            observed = float(seed_matrix.mean(axis=0).mean())
            expected = float(frozen_row["summary"]["mean"])
            if abs(observed - expected) > tolerance:
                raise ValueError(
                    f"{condition}/{window}: mean differs from frozen artifact "
                    f"({observed} versus {expected})"
                )


def _summary_row(
    *,
    dataset: str,
    condition: str,
    window: str,
    metric: str,
    values: np.ndarray,
    seed: int,
    bootstrap_samples: int,
) -> dict:
    stats = summarize(values, seed=seed, bootstrap_samples=bootstrap_samples)
    return {
        "dataset": dataset,
        "condition": condition,
        "window_s": float(window),
        "metric": metric,
        "participants": stats["participants"],
        "mean": stats["mean"],
        "standard_deviation": stats["standard_deviation"],
        "median": stats["median"],
        "q1": stats["q1"],
        "q3": stats["q3"],
        "bootstrap_mean_ci95_low": stats["bootstrap_mean_ci95_low"],
        "bootstrap_mean_ci95_high": stats["bootstrap_mean_ci95_high"],
    }


def _contrast_row(
    *,
    family: str,
    dataset: str,
    left_name: str,
    right_name: str,
    window: str,
    metric: str,
    left: np.ndarray,
    right: np.ndarray,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict:
    test = paired_test(left, right)
    ci_low, ci_high = _bootstrap_difference_ci(
        left, right, seed=bootstrap_seed, samples=bootstrap_samples
    )
    return {
        "multiplicity_family": family,
        "dataset": dataset,
        "left": left_name,
        "right": right_name,
        "window_s": float(window),
        "metric": metric,
        "participants": int(test["participants"]),
        "mean_difference": float(test["mean_difference"]),
        "median_difference": float(test["median_difference"]),
        "bootstrap_difference_ci95_low": ci_low,
        "bootstrap_difference_ci95_high": ci_high,
        "wilcoxon_p_value": float(test["p_value"]),
        "rank_biserial": float(test["rank_biserial"]),
    }


def _apply_holm_by_family(rows: list[dict]) -> None:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["multiplicity_family"])].append(index)
    for indices in grouped.values():
        adjusted = holm_adjust([float(rows[index]["wilcoxon_p_value"]) for index in indices])
        for index, value in zip(indices, adjusted, strict=True):
            rows[index]["holm_adjusted_p_value"] = float(value)


def _main_evidence(config: dict, output_dir: Path) -> dict:
    raw_rows: list[dict] = []
    average_rows: list[dict] = []
    summary_rows: list[dict] = []
    contrast_rows: list[dict] = []
    audit = {"datasets": {}, "validation": "passed"}
    bootstrap_seed = int(config["bootstrap_seed"])
    bootstrap_samples = int(config["bootstrap_samples"])

    for dataset_index, (dataset_key, dataset_config) in enumerate(config["main"].items()):
        windows = [str(value) for value in dataset_config["windows"]]
        conditions = {}
        input_paths = {}
        for condition, patterns in dataset_config["conditions"].items():
            paths, documents = _expand(patterns)
            _validate_group(
                documents,
                participants=int(dataset_config["participants"]),
                windows=windows,
                label=f"{dataset_key}/{condition}",
            )
            conditions[condition] = _condition_matrices(documents, windows)
            input_paths[condition] = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]

        frozen_path = ROOT / dataset_config["frozen_comparison"]
        frozen = _load_json(frozen_path)
        _validate_against_frozen(frozen, conditions, windows)

        for condition_index, (condition, matrices) in enumerate(conditions.items()):
            for window_index, window in enumerate(windows):
                metric_values = {}
                for metric_index, metric in enumerate(METRICS):
                    subjects, seed_matrix, seeds = matrices[(window, metric)]
                    averaged = seed_matrix.mean(axis=0)
                    metric_values[metric] = (subjects, seed_matrix, seeds, averaged)
                    summary_rows.append(
                        _summary_row(
                            dataset=dataset_key,
                            condition=condition,
                            window=window,
                            metric=metric,
                            values=averaged,
                            seed=bootstrap_seed
                            + dataset_index * 1000
                            + condition_index * 100
                            + window_index * 10
                            + metric_index,
                            bootstrap_samples=bootstrap_samples,
                        )
                    )
                subjects = metric_values["accuracy"][0]
                seeds = metric_values["accuracy"][2]
                for seed_index, seed in enumerate(seeds):
                    for participant_index, participant in enumerate(subjects):
                        raw_rows.append({
                            "dataset": dataset_key,
                            "condition": condition,
                            "window_s": float(window),
                            "participant": participant,
                            "seed": seed,
                            "accuracy": metric_values["accuracy"][1][seed_index, participant_index],
                            "balanced_accuracy": metric_values["balanced_accuracy"][1][seed_index, participant_index],
                            "itr_bits_per_minute": metric_values["itr_bits_per_minute"][1][seed_index, participant_index],
                        })
                for participant_index, participant in enumerate(subjects):
                    average_rows.append({
                        "dataset": dataset_key,
                        "condition": condition,
                        "window_s": float(window),
                        "participant": participant,
                        "seeds_averaged": len(seeds),
                        "accuracy": metric_values["accuracy"][3][participant_index],
                        "balanced_accuracy": metric_values["balanced_accuracy"][3][participant_index],
                        "itr_bits_per_minute": metric_values["itr_bits_per_minute"][3][participant_index],
                    })

        left_name, right_name = "harmonic_fold", "ssvepformer"
        for metric_index, metric in enumerate(METRICS):
            family = f"main:{dataset_key}:{metric}:all_registered_windows"
            for window_index, window in enumerate(windows):
                left_subjects, left_matrix, _ = conditions[left_name][(window, metric)]
                right_subjects, right_matrix, _ = conditions[right_name][(window, metric)]
                if left_subjects != right_subjects:
                    raise ValueError(f"{dataset_key}/{window}/{metric}: participant mismatch")
                contrast_rows.append(
                    _contrast_row(
                        family=family,
                        dataset=dataset_key,
                        left_name=left_name,
                        right_name=right_name,
                        window=window,
                        metric=metric,
                        left=left_matrix.mean(axis=0),
                        right=right_matrix.mean(axis=0),
                        bootstrap_seed=bootstrap_seed
                        + dataset_index * 1000
                        + metric_index * 100
                        + window_index,
                        bootstrap_samples=bootstrap_samples,
                    )
                )
        audit["datasets"][dataset_key] = {
            "participants": int(dataset_config["participants"]),
            "seeds": EXPECTED_SEEDS,
            "windows": windows,
            "frozen_comparison": str(frozen_path.relative_to(ROOT)).replace("\\", "/"),
            "inputs": input_paths,
        }

    _apply_holm_by_family(contrast_rows)
    _write_csv(
        output_dir / "main_participant_metrics_by_seed.csv",
        raw_rows,
        ["dataset", "condition", "window_s", "participant", "seed", *METRICS],
    )
    _write_csv(
        output_dir / "main_participant_metrics_seed_averaged.csv",
        average_rows,
        ["dataset", "condition", "window_s", "participant", "seeds_averaged", *METRICS],
    )
    _write_csv(
        output_dir / "main_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    _write_csv(
        output_dir / "main_contrasts.csv",
        contrast_rows,
        list(contrast_rows[0]),
    )
    return {
        "audit": audit,
        "summaries": summary_rows,
        "contrasts": contrast_rows,
    }


def _ablation_evidence(config: dict, output_dir: Path) -> dict:
    ablation_config = config["ablations"]
    windows = [str(value) for value in ablation_config["windows"]]
    matrices = {}
    input_paths = {}
    for condition, patterns in ablation_config["conditions"].items():
        paths, documents = _expand(patterns)
        _validate_group(
            documents,
            participants=int(ablation_config["participants"]),
            windows=windows,
            label=f"ablation/{condition}",
        )
        matrices[condition] = _condition_matrices(documents, windows)
        input_paths[condition] = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]

    for ablation, relative in ablation_config["frozen_comparisons"].items():
        frozen = _load_json(ROOT / relative)
        _validate_against_frozen(
            frozen,
            {"full": matrices["full"], ablation: matrices[ablation]},
            windows,
        )

    participant_rows = []
    summary_rows = []
    contrast_rows = []
    bootstrap_seed = int(config["bootstrap_seed"]) + 10000
    bootstrap_samples = int(config["bootstrap_samples"])
    for condition_index, (condition, condition_matrices) in enumerate(matrices.items()):
        for window_index, window in enumerate(windows):
            subjects, seed_matrix, seeds = condition_matrices[(window, "balanced_accuracy")]
            averaged = seed_matrix.mean(axis=0)
            summary_rows.append(
                _summary_row(
                    dataset="beta",
                    condition=condition,
                    window=window,
                    metric="balanced_accuracy",
                    values=averaged,
                    seed=bootstrap_seed + condition_index * 100 + window_index,
                    bootstrap_samples=bootstrap_samples,
                )
            )
            for index, participant in enumerate(subjects):
                participant_rows.append({
                    "dataset": "beta",
                    "condition": condition,
                    "window_s": float(window),
                    "participant": participant,
                    "seeds_averaged": len(seeds),
                    "balanced_accuracy": averaged[index],
                })

    for ablation_index, ablation in enumerate(
        ("no_local", "no_attention", "no_harmonic_bias")
    ):
        family = f"ablation:beta:full_minus_{ablation}:registered_windows"
        for window_index, window in enumerate(windows):
            left_subjects, left_matrix, _ = matrices["full"][(window, "balanced_accuracy")]
            right_subjects, right_matrix, _ = matrices[ablation][(window, "balanced_accuracy")]
            if left_subjects != right_subjects:
                raise ValueError(f"ablation/{ablation}/{window}: participant mismatch")
            contrast_rows.append(
                _contrast_row(
                    family=family,
                    dataset="beta",
                    left_name="full",
                    right_name=ablation,
                    window=window,
                    metric="balanced_accuracy",
                    left=left_matrix.mean(axis=0),
                    right=right_matrix.mean(axis=0),
                    bootstrap_seed=bootstrap_seed + ablation_index * 100 + window_index,
                    bootstrap_samples=bootstrap_samples,
                )
            )
    _apply_holm_by_family(contrast_rows)
    _write_csv(
        output_dir / "ablation_participant_balanced_accuracy.csv",
        participant_rows,
        list(participant_rows[0]),
    )
    _write_csv(output_dir / "ablation_summary.csv", summary_rows, list(summary_rows[0]))
    _write_csv(
        output_dir / "ablation_contrasts.csv", contrast_rows, list(contrast_rows[0])
    )
    return {
        "audit": {
            "participants": int(ablation_config["participants"]),
            "seeds": EXPECTED_SEEDS,
            "windows": windows,
            "multiplicity": "Holm correction is applied separately to the six registered windows of each prespecified component ablation.",
            "inputs": input_paths,
            "validation": "passed",
        },
        "summaries": summary_rows,
        "contrasts": contrast_rows,
    }


def _mtsnet_evidence(config: dict, output_dir: Path) -> dict:
    rows = []
    participant_rows: list[dict] = []
    average_rows: list[dict] = []
    summary_rows: list[dict] = []
    bootstrap_seed = int(config["bootstrap_seed"]) + 60000
    bootstrap_samples = int(config["bootstrap_samples"])
    for relative in config["mtsnet"]["comparison_files"]:
        path = ROOT / relative
        document = _load_json(path)
        if len(document.get("contrasts", [])) != 1:
            raise ValueError(f"expected one contrast in {relative}")
        contrast = document["contrasts"][0]
        paired = contrast["paired_left_minus_right"]
        dataset = str(document["dataset"])
        window = str(float(contrast["window_seconds"]))
        hfn_paths, hfn_documents = _expand(
            config["mtsnet"]["harmonic_fold_sources"][dataset]
        )
        mtsnet_pattern = str(path.parent.relative_to(ROOT) / "seed-*" / "fold-*" / "result.json")
        mtsnet_paths, mtsnet_documents = _expand([mtsnet_pattern])
        participants = int(config["main"][dataset]["participants"])
        _validate_group(
            hfn_documents,
            participants=participants,
            windows=[window],
            label=f"mtsnet/{dataset}/harmonic_fold/{window}",
        )
        _validate_group(
            mtsnet_documents,
            participants=participants,
            windows=[window],
            label=f"mtsnet/{dataset}/mtsnet/{window}",
        )
        condition_values = {}
        for condition, documents, source_paths in (
            ("harmonic_fold", hfn_documents, hfn_paths),
            ("mtsnet", mtsnet_documents, mtsnet_paths),
        ):
            subjects, seed_matrix, seeds = merge_out_of_fold_documents(
                documents, window=window, metric="balanced_accuracy"
            )
            averaged = seed_matrix.mean(axis=0)
            condition_values[condition] = (subjects, averaged)
            expected_mean = float(document["conditions"][condition][window]["summary"]["mean"])
            if abs(float(averaged.mean()) - expected_mean) > 1e-12:
                raise ValueError(f"mtsnet/{dataset}/{condition}/{window}: mean differs")
            summary_rows.append(
                _summary_row(
                    dataset=dataset,
                    condition=condition,
                    window=window,
                    metric="balanced_accuracy",
                    values=averaged,
                    seed=bootstrap_seed + len(summary_rows),
                    bootstrap_samples=bootstrap_samples,
                )
            )
            for seed_index, seed in enumerate(seeds):
                for participant_index, participant in enumerate(subjects):
                    participant_rows.append({
                        "dataset": dataset,
                        "condition": condition,
                        "window_s": float(window),
                        "participant": participant,
                        "seed": seed,
                        "balanced_accuracy": seed_matrix[seed_index, participant_index],
                    })
            for participant_index, participant in enumerate(subjects):
                average_rows.append({
                    "dataset": dataset,
                    "condition": condition,
                    "window_s": float(window),
                    "participant": participant,
                    "seeds_averaged": len(seeds),
                    "balanced_accuracy": averaged[participant_index],
                })
        left_subjects, left_values = condition_values["harmonic_fold"]
        right_subjects, right_values = condition_values["mtsnet"]
        if left_subjects != right_subjects:
            raise ValueError(f"mtsnet/{dataset}/{window}: participant mismatch")
        ci_low, ci_high = _bootstrap_difference_ci(
            left_values,
            right_values,
            seed=bootstrap_seed + len(rows),
            samples=bootstrap_samples,
        )
        rows.append({
            "multiplicity_family": "mtsnet:benchmark_and_beta:0.4_0.8_1.2s",
            "dataset": dataset,
            "window_s": float(contrast["window_seconds"]),
            "participants": int(paired["participants"]),
            "harmonic_fold_mean": float(contrast["left_summary"]["mean"]),
            "mtsnet_mean": float(contrast["right_summary"]["mean"]),
            "mean_difference": float(paired["mean_difference"]),
            "median_difference": float(paired["median_difference"]),
            "bootstrap_difference_ci95_low": ci_low,
            "bootstrap_difference_ci95_high": ci_high,
            "wilcoxon_p_value": float(paired["p_value"]),
            "rank_biserial": float(paired["rank_biserial"]),
            "legacy_within_file_adjusted_p": float(paired["holm_adjusted_p_value"]),
            "source": relative,
        })
    rows.sort(key=lambda row: (row["dataset"], row["window_s"]))
    adjusted = holm_adjust([row["wilcoxon_p_value"] for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p_value_six_test_family"] = float(value)
    _write_csv(
        output_dir / "mtsnet_participant_by_seed.csv",
        participant_rows,
        list(participant_rows[0]),
    )
    _write_csv(
        output_dir / "mtsnet_participant_seed_averaged.csv",
        average_rows,
        list(average_rows[0]),
    )
    _write_csv(
        output_dir / "mtsnet_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    _write_csv(output_dir / "mtsnet_contrasts_corrected.csv", rows, list(rows[0]))
    return {
        "audit": {
            "tests": len(rows),
            "multiplicity": "One Holm family across both datasets and all three registered windows (six paired tests).",
            "validation": "passed",
        },
        "summaries": summary_rows,
        "contrasts": rows,
    }


def _attention_mechanism_evidence(config: dict, output_dir: Path) -> dict:
    mechanism_config = config["attention_mechanism"]
    expected_participants = int(mechanism_config["participants"])
    metrics = [str(value) for value in mechanism_config["metrics"]]
    windows = [str(value) for value in mechanism_config["windows"]]
    participant_rows: list[dict] = []
    contrast_rows: list[dict] = []
    inputs: list[str] = []
    bootstrap_seed = int(config["bootstrap_seed"]) + 20000
    bootstrap_samples = int(config["bootstrap_samples"])
    family = "attention_mechanism:beta:four_metrics_x_three_windows"

    for window_index, window in enumerate(windows):
        relative = mechanism_config["summary_files"][window]
        path = ROOT / relative
        document = _load_json(path)
        inputs.append(relative)
        if int(document["unique_subjects"]) != expected_participants:
            raise ValueError(
                f"attention/{window}: expected {expected_participants} participants, "
                f"found {document['unique_subjects']}"
            )
        if not document.get("replicated_seeds_are_averaged_before_inference"):
            raise ValueError(f"attention/{window}: seed repeats were not averaged")
        participants = sorted(document["per_subject"], key=lambda value: int(value))
        if len(participants) != expected_participants:
            raise ValueError(f"attention/{window}: incomplete per-subject record")
        if {int(document["per_subject"][subject]["replicates"]) for subject in participants} != {3}:
            raise ValueError(f"attention/{window}: expected three seed replicates per participant")

        for metric_index, metric in enumerate(metrics):
            if metric not in document["metrics"]:
                raise ValueError(f"attention/{window}: missing metric {metric}")
            with_values = np.asarray(
                [document["per_subject"][subject][f"with::{metric}"] for subject in participants],
                dtype=np.float64,
            )
            without_values = np.asarray(
                [document["per_subject"][subject][f"without::{metric}"] for subject in participants],
                dtype=np.float64,
            )
            frozen = document["metrics"][metric]
            if abs(float(with_values.mean()) - float(frozen["with_harmonic_bias_mean"])) > 1e-12:
                raise ValueError(f"attention/{window}/{metric}: with-bias mean differs")
            if abs(float(without_values.mean()) - float(frozen["without_harmonic_bias_mean"])) > 1e-12:
                raise ValueError(f"attention/{window}/{metric}: masked mean differs")

            for participant, with_value, without_value in zip(
                participants, with_values, without_values, strict=True
            ):
                participant_rows.append({
                    "dataset": "beta",
                    "window_s": float(window),
                    "participant": participant,
                    "seeds_averaged": 3,
                    "metric": metric,
                    "with_harmonic_bias": float(with_value),
                    "bias_masked": float(without_value),
                })

            row = _contrast_row(
                family=family,
                dataset="beta",
                left_name="with_harmonic_bias",
                right_name="bias_masked",
                window=window,
                metric=metric,
                left=with_values,
                right=without_values,
                bootstrap_seed=bootstrap_seed + window_index * 100 + metric_index,
                bootstrap_samples=bootstrap_samples,
            )
            row["with_harmonic_bias_mean"] = float(with_values.mean())
            row["bias_masked_mean"] = float(without_values.mean())
            contrast_rows.append(row)

    _apply_holm_by_family(contrast_rows)
    _write_csv(
        output_dir / "attention_mechanism_participant_metrics.csv",
        participant_rows,
        list(participant_rows[0]),
    )
    _write_csv(
        output_dir / "attention_mechanism_summary.csv",
        contrast_rows,
        list(contrast_rows[0]),
    )
    return {
        "audit": {
            "participants": expected_participants,
            "seeds_averaged": EXPECTED_SEEDS,
            "windows": windows,
            "metrics": metrics,
            "multiplicity": "One Holm family across four mechanism metrics and three windows (12 paired tests).",
            "inputs": inputs,
            "validation": "passed",
        },
        "contrasts": contrast_rows,
    }


def _itr_sensitivity_evidence(config: dict, output_dir: Path) -> dict:
    source = output_dir / "main_participant_metrics_by_seed.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    grouped: dict[tuple[str, str, float, str], list[float]] = defaultdict(list)
    for row in records:
        grouped[
            (
                str(row["dataset"]),
                str(row["condition"]),
                float(row["window_s"]),
                str(row["participant"]),
            )
        ].append(float(row["accuracy"]))
    overheads = (0.0, 0.25, 0.5, 1.0)
    rows: list[dict] = []
    bootstrap_seed = int(config["bootstrap_seed"]) + 25000
    for dataset_index, (dataset, dataset_config) in enumerate(config["main"].items()):
        classes = int(dataset_config["classes"])
        windows = [float(value) for value in dataset_config["display_windows"]]
        for condition_index, condition in enumerate(("harmonic_fold", "ssvepformer")):
            for overhead_index, overhead in enumerate(overheads):
                for window_index, window in enumerate(windows):
                    participants = sorted(
                        [
                            key[3]
                            for key in grouped
                            if key[0] == dataset and key[1] == condition and key[2] == window
                        ],
                        key=lambda value: (len(value), value),
                    )
                    values = np.asarray(
                        [
                            np.mean(
                                [
                                    information_transfer_rate(
                                        classes,
                                        accuracy,
                                        window + overhead,
                                    )
                                    for accuracy in grouped[(dataset, condition, window, participant)]
                                ]
                            )
                            for participant in participants
                        ],
                        dtype=np.float64,
                    )
                    stats = summarize(
                        values,
                        seed=(
                            bootstrap_seed
                            + dataset_index * 1000
                            + condition_index * 200
                            + overhead_index * 20
                            + window_index
                        ),
                    )
                    rows.append({
                        "dataset": dataset,
                        "condition": condition,
                        "overhead_s": overhead,
                        "window_s": window,
                        "participants": stats["participants"],
                        "mean_itr_bits_per_minute": stats["mean"],
                        "bootstrap_mean_ci95_low": stats["bootstrap_mean_ci95_low"],
                        "bootstrap_mean_ci95_high": stats["bootstrap_mean_ci95_high"],
                    })
    _write_csv(output_dir / "itr_overhead_sensitivity.csv", rows, list(rows[0]))
    peak_rows = []
    for dataset in config["main"]:
        for condition in ("harmonic_fold", "ssvepformer"):
            for overhead in overheads:
                candidates = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["condition"] == condition
                    and row["overhead_s"] == overhead
                ]
                peak = max(candidates, key=lambda row: row["mean_itr_bits_per_minute"])
                peak_rows.append({
                    "dataset": dataset,
                    "condition": condition,
                    "overhead_s": overhead,
                    "peak_window_s": peak["window_s"],
                    "peak_mean_itr_bits_per_minute": peak["mean_itr_bits_per_minute"],
                })
    _write_csv(output_dir / "itr_overhead_sensitivity_peaks.csv", peak_rows, list(peak_rows[0]))
    return {
        "audit": {
            "participants": {
                dataset: int(entry["participants"])
                for dataset, entry in config["main"].items()
            },
            "overheads_s": list(overheads),
            "calculation": "ITR is recomputed per participant and seed before averaging seeds within participant.",
            "validation": "passed",
        },
        "summaries": rows,
        "peaks": peak_rows,
    }


def _analytic_baseline_evidence(config: dict, output_dir: Path) -> dict:
    participant_rows: list[dict] = []
    summary_rows: list[dict] = []
    audit: dict[str, object] = {"datasets": {}, "validation": "passed"}
    bootstrap_seed = int(config["bootstrap_seed"]) + 25000
    bootstrap_samples = int(config["bootstrap_samples"])
    for dataset_index, (dataset, section) in enumerate(
        config["analytic_baselines"].items()
    ):
        expected_participants = int(section["participants"])
        dataset_inputs: list[str] = []
        for method_index, (method, method_config) in enumerate(
            section["methods"].items()
        ):
            relative_path = str(method_config["result_file"])
            document = _load_json(ROOT / relative_path)
            result_method = str(method_config["result_method"])
            if result_method not in document["methods"]:
                raise ValueError(
                    f"analytic_baselines/{dataset}/{method}: missing method {result_method}"
                )
            dataset_inputs.append(relative_path)
            for window_index, (window, payload) in enumerate(
                document["methods"][result_method].items()
            ):
                per_subject = payload["per_subject"]
                subjects = sorted(per_subject, key=lambda value: (len(value), value))
                if len(subjects) != expected_participants:
                    raise ValueError(
                        f"analytic_baselines/{dataset}/{method}/{window}: "
                        f"expected {expected_participants} participants, found {len(subjects)}"
                    )
                values = np.asarray(
                    [float(per_subject[subject]["balanced_accuracy"]) for subject in subjects],
                    dtype=np.float64,
                )
                reconstructed = float(values.mean())
                frozen = float(payload["overall"]["balanced_accuracy"])
                if abs(reconstructed - frozen) > 1e-12:
                    raise ValueError(
                        f"analytic_baselines/{dataset}/{method}/{window}: mean differs "
                        f"from preserved result ({reconstructed} versus {frozen})"
                    )
                summary_rows.append(
                    _summary_row(
                        dataset=dataset,
                        condition=method,
                        window=window,
                        metric="balanced_accuracy",
                        values=values,
                        seed=bootstrap_seed + dataset_index * 1000 + method_index * 100 + window_index,
                        bootstrap_samples=bootstrap_samples,
                    )
                )
                for subject, value in zip(subjects, values, strict=True):
                    participant_rows.append({
                        "dataset": dataset,
                        "condition": method,
                        "window_s": float(window),
                        "participant": subject,
                        "balanced_accuracy": value,
                    })
        audit["datasets"][dataset] = {
            "participants": expected_participants,
            "methods": sorted(section["methods"]),
            "inputs": sorted(set(dataset_inputs)),
        }
    _write_csv(
        output_dir / "analytic_baseline_participant.csv",
        participant_rows,
        list(participant_rows[0]),
    )
    _write_csv(
        output_dir / "analytic_baseline_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    return {"audit": audit, "summaries": summary_rows}


def _strong_filter_bank_evidence(config: dict, output_dir: Path) -> dict:
    section = config["strong_filter_bank"]
    windows = [str(value) for value in section["windows"]]
    matrices = {}
    input_paths = {}
    for condition, patterns in section["conditions"].items():
        paths, documents = _expand(patterns)
        _validate_group(
            documents,
            participants=int(section["participants"]),
            windows=windows,
            label=f"strong_filter_bank/{condition}",
        )
        matrices[condition] = _condition_matrices(documents, windows)
        input_paths[condition] = [
            str(path.relative_to(ROOT)).replace("\\", "/") for path in paths
        ]

    frozen_path = ROOT / section["frozen_comparison"]
    frozen = _load_json(frozen_path)
    _validate_against_frozen(frozen, matrices, windows)

    raw_rows: list[dict] = []
    average_rows: list[dict] = []
    summary_rows: list[dict] = []
    contrast_rows: list[dict] = []
    bootstrap_seed = int(config["bootstrap_seed"]) + 30000
    bootstrap_samples = int(config["bootstrap_samples"])
    for condition_index, (condition, condition_matrices) in enumerate(matrices.items()):
        for window_index, window in enumerate(windows):
            subjects, seed_matrix, seeds = condition_matrices[(window, "balanced_accuracy")]
            averaged = seed_matrix.mean(axis=0)
            summary_rows.append(
                _summary_row(
                    dataset="beta",
                    condition=condition,
                    window=window,
                    metric="balanced_accuracy",
                    values=averaged,
                    seed=bootstrap_seed + condition_index * 100 + window_index,
                    bootstrap_samples=bootstrap_samples,
                )
            )
            for seed_index, seed in enumerate(seeds):
                for participant_index, participant in enumerate(subjects):
                    raw_rows.append({
                        "dataset": "beta",
                        "condition": condition,
                        "window_s": float(window),
                        "participant": participant,
                        "seed": seed,
                        "balanced_accuracy": seed_matrix[seed_index, participant_index],
                    })
            for participant_index, participant in enumerate(subjects):
                average_rows.append({
                    "dataset": "beta",
                    "condition": condition,
                    "window_s": float(window),
                    "participant": participant,
                    "seeds_averaged": len(seeds),
                    "balanced_accuracy": averaged[participant_index],
                })

    for window_index, window in enumerate(windows):
        left_subjects, left_matrix, _ = matrices["harmonic_fold"][(window, "balanced_accuracy")]
        right_subjects, right_matrix, _ = matrices["fb_ssvepformer"][(window, "balanced_accuracy")]
        if left_subjects != right_subjects:
            raise ValueError(f"strong_filter_bank/{window}: participant mismatch")
        contrast_rows.append(
            _contrast_row(
                family="strong_filter_bank:beta:all_registered_windows",
                dataset="beta",
                left_name="harmonic_fold",
                right_name="fb_ssvepformer",
                window=window,
                metric="balanced_accuracy",
                left=left_matrix.mean(axis=0),
                right=right_matrix.mean(axis=0),
                bootstrap_seed=bootstrap_seed + 1000 + window_index,
                bootstrap_samples=bootstrap_samples,
            )
        )
    _apply_holm_by_family(contrast_rows)
    frozen_contrasts = {
        float(row["window_seconds"]): row["paired_left_minus_right"]
        for row in frozen["contrasts"]
    }
    for row in contrast_rows:
        expected = frozen_contrasts[row["window_s"]]
        for key, frozen_key in (
            ("mean_difference", "mean_difference"),
            ("wilcoxon_p_value", "p_value"),
            ("holm_adjusted_p_value", "holm_adjusted_p_value"),
        ):
            if abs(float(row[key]) - float(expected[frozen_key])) > 1e-12:
                raise ValueError(
                    f"strong_filter_bank/{row['window_s']}: {key} differs from frozen artifact"
                )

    _write_csv(
        output_dir / "strong_filter_bank_participant_by_seed.csv",
        raw_rows,
        list(raw_rows[0]),
    )
    _write_csv(
        output_dir / "strong_filter_bank_participant_seed_averaged.csv",
        average_rows,
        list(average_rows[0]),
    )
    _write_csv(
        output_dir / "strong_filter_bank_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    _write_csv(
        output_dir / "strong_filter_bank_contrasts.csv",
        contrast_rows,
        list(contrast_rows[0]),
    )
    return {
        "audit": {
            "participants": int(section["participants"]),
            "seeds": EXPECTED_SEEDS,
            "windows": windows,
            "multiplicity": "One Holm family across the six BETA windows.",
            "frozen_comparison": section["frozen_comparison"],
            "inputs": input_paths,
            "validation": "passed",
        },
        "summaries": summary_rows,
        "contrasts": contrast_rows,
    }


def _nested_subject_values(
    document: dict,
    *,
    method: str,
    calibration_blocks: str,
    window: str,
    metric: str = "balanced_accuracy",
) -> dict[str, float]:
    rows = document["methods"][method][calibration_blocks][window]["per_subject"]
    return {str(subject): float(values[metric]) for subject, values in rows.items()}


def _adaptation_evidence(config: dict, output_dir: Path) -> dict:
    section = config["adaptation"]
    windows = [str(value) for value in section["windows"]]
    documents = [_load_json(ROOT / relative) for relative in section["result_files"]]
    seeds = sorted(int(document["seed"]) for document in documents)
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"adaptation: expected seeds {EXPECTED_SEEDS}, found {seeds}")
    classical = _load_json(ROOT / section["classical_results"])
    frozen = _load_json(ROOT / section["frozen_summary"])
    participant_rows: list[dict] = []
    summary_rows: list[dict] = []
    contrast_rows: list[dict] = []
    bootstrap_seed = int(config["bootstrap_seed"]) + 40000
    bootstrap_samples = int(config["bootstrap_samples"])
    averaged_conditions: dict[tuple[str, str], tuple[list[str], np.ndarray]] = {}

    condition_specs = [("zero_shot", "0")]
    condition_specs.extend(
        (str(section["method"]), str(blocks))
        for blocks in section["calibration_blocks"]
    )
    frozen_conditions = {
        (int(row["calibration_blocks"]), float(row["window_seconds"])): row
        for row in frozen["conditions"]
    }
    for condition_index, (method, blocks) in enumerate(condition_specs):
        label = "zero_shot" if blocks == "0" else f"adapter_{blocks}_block" + ("s" if blocks != "1" else "")
        for window_index, window in enumerate(windows):
            seed_values: list[dict[str, float]] = [
                _nested_subject_values(
                    document,
                    method=method,
                    calibration_blocks=blocks,
                    window=window,
                )
                for document in documents
            ]
            subjects = sorted(seed_values[0], key=lambda value: (len(value), value))
            if len(subjects) != int(section["participants"]):
                raise ValueError(f"adaptation/{label}/{window}: incomplete participant set")
            if any(set(values) != set(subjects) for values in seed_values):
                raise ValueError(f"adaptation/{label}/{window}: participant mismatch")
            matrix = np.asarray(
                [[values[subject] for subject in subjects] for values in seed_values],
                dtype=np.float64,
            )
            averaged = matrix.mean(axis=0)
            averaged_conditions[(label, window)] = (subjects, averaged)
            frozen_row = frozen_conditions[(int(blocks), float(window))]
            if abs(float(averaged.mean()) - float(frozen_row["summary"]["mean"])) > 1e-12:
                raise ValueError(f"adaptation/{label}/{window}: mean differs from frozen artifact")
            summary_rows.append({
                **_summary_row(
                    dataset="beta",
                    condition=label,
                    window=window,
                    metric="balanced_accuracy",
                    values=averaged,
                    seed=bootstrap_seed + condition_index * 100 + window_index,
                    bootstrap_samples=bootstrap_samples,
                ),
                "calibration_blocks": int(blocks),
            })
            for seed_index, seed in enumerate(seeds):
                for participant_index, participant in enumerate(subjects):
                    participant_rows.append({
                        "dataset": "beta",
                        "condition": label,
                        "window_s": float(window),
                        "participant": participant,
                        "seed": seed,
                        "balanced_accuracy": matrix[seed_index, participant_index],
                    })

    improvement_rows = []
    for blocks in map(str, section["calibration_blocks"]):
        label = f"adapter_{blocks}_block" + ("s" if blocks != "1" else "")
        for window_index, window in enumerate(windows):
            subjects, adapted = averaged_conditions[(label, window)]
            zero_subjects, zero = averaged_conditions[("zero_shot", window)]
            if subjects != zero_subjects:
                raise ValueError(f"adaptation/{label}/{window}: participant mismatch")
            row = _contrast_row(
                family="adaptation:beta:adapter_minus_zero:two_blocks_x_three_windows",
                dataset="beta",
                left_name=label,
                right_name="zero_shot",
                window=window,
                metric="balanced_accuracy",
                left=adapted,
                right=zero,
                bootstrap_seed=bootstrap_seed + int(blocks) * 100 + window_index,
                bootstrap_samples=bootstrap_samples,
            )
            improvement_rows.append(row)
    _apply_holm_by_family(improvement_rows)
    for row in improvement_rows:
        blocks = 1 if row["left"] == "adapter_1_block" else 2
        expected = frozen_conditions[(blocks, row["window_s"])]["paired_adapted_minus_zero_shot"]
        if abs(float(row["holm_adjusted_p_value"]) - float(expected["holm_adjusted_p_value"])) > 1e-12:
            raise ValueError(f"adaptation/{row['left']}/{row['window_s']}: corrected p differs")
    contrast_rows.extend(improvement_rows)

    frozen_classical = {
        (int(row["calibration_blocks"]), float(row["window_seconds"]), str(row["baseline"])): row
        for row in frozen["classical_comparisons"]
    }
    headline_blocks = str(section["headline_classical_blocks"])
    for method_index, baseline in enumerate(section["classical_methods"]):
        for window_index, window in enumerate(windows):
            values_by_subject = _nested_subject_values(
                classical,
                method=str(baseline),
                calibration_blocks=headline_blocks,
                window=window,
            )
            subjects, adapted = averaged_conditions[(f"adapter_{headline_blocks}_blocks", window)]
            if set(values_by_subject) != set(subjects):
                raise ValueError(f"adaptation/{baseline}/{window}: participant mismatch")
            baseline_values = np.asarray(
                [values_by_subject[subject] for subject in subjects], dtype=np.float64
            )
            stats = _summary_row(
                dataset="beta",
                condition=f"{baseline}_{headline_blocks}_blocks",
                window=window,
                metric="balanced_accuracy",
                values=baseline_values,
                seed=bootstrap_seed + 1000 + method_index * 100 + window_index,
                bootstrap_samples=bootstrap_samples,
            )
            stats["calibration_blocks"] = int(headline_blocks)
            summary_rows.append(stats)
            for participant, value in zip(subjects, baseline_values, strict=True):
                participant_rows.append({
                    "dataset": "beta",
                    "condition": f"{baseline}_{headline_blocks}_blocks",
                    "window_s": float(window),
                    "participant": participant,
                    "seed": "not_applicable",
                    "balanced_accuracy": value,
                })
            row = _contrast_row(
                family="adaptation:beta:adapter_vs_classical:registered_12_test_family",
                dataset="beta",
                left_name=f"adapter_{headline_blocks}_blocks",
                right_name=f"{baseline}_{headline_blocks}_blocks",
                window=window,
                metric="balanced_accuracy",
                left=adapted,
                right=baseline_values,
                bootstrap_seed=bootstrap_seed + 2000 + method_index * 100 + window_index,
                bootstrap_samples=bootstrap_samples,
            )
            expected = frozen_classical[(int(headline_blocks), float(window), str(baseline))]
            frozen_test = expected["paired_model_minus_baseline"]
            for key, frozen_key in (
                ("mean_difference", "mean_difference"),
                ("wilcoxon_p_value", "p_value"),
            ):
                if abs(float(row[key]) - float(frozen_test[frozen_key])) > 1e-12:
                    raise ValueError(f"adaptation/{baseline}/{window}: {key} differs")
            row["holm_adjusted_p_value"] = float(frozen_test["holm_adjusted_p_value"])
            row["multiplicity_family_total_tests"] = 12
            contrast_rows.append(row)

    _write_csv(
        output_dir / "adaptation_participant_balanced_accuracy.csv",
        participant_rows,
        list(participant_rows[0]),
    )
    _write_csv(
        output_dir / "adaptation_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    _write_csv(
        output_dir / "adaptation_contrasts.csv",
        contrast_rows,
        sorted({key for row in contrast_rows for key in row}),
    )
    return {
        "audit": {
            "participants": int(section["participants"]),
            "seeds": EXPECTED_SEEDS,
            "windows": windows,
            "adaptation_parameters": 113,
            "validation": "passed",
            "inputs": [*section["result_files"], section["classical_results"], section["frozen_summary"]],
        },
        "summaries": summary_rows,
        "contrasts": contrast_rows,
    }


def _electrode_transfer_evidence(config: dict, output_dir: Path) -> dict:
    section = config["electrode_transfer"]
    windows = [str(value) for value in section["windows"]]
    matrices = {}
    input_paths = {}
    for condition, patterns in section["conditions"].items():
        paths, documents = _expand(patterns)
        _validate_group(
            documents,
            participants=int(section["participants"]),
            windows=windows,
            label=f"electrode_transfer/{condition}",
        )
        matrices[condition] = _condition_matrices(documents, windows)
        input_paths[condition] = [
            str(path.relative_to(ROOT)).replace("\\", "/") for path in paths
        ]
    frozen_path = ROOT / section["frozen_comparison"]
    frozen = _load_json(frozen_path)
    _validate_against_frozen(frozen, matrices, windows)

    raw_rows: list[dict] = []
    average_rows: list[dict] = []
    summary_rows: list[dict] = []
    contrast_rows: list[dict] = []
    bootstrap_seed = int(config["bootstrap_seed"]) + 50000
    bootstrap_samples = int(config["bootstrap_samples"])
    for condition_index, (condition, condition_matrices) in enumerate(matrices.items()):
        for window_index, window in enumerate(windows):
            subjects, seed_matrix, seeds = condition_matrices[(window, "balanced_accuracy")]
            averaged = seed_matrix.mean(axis=0)
            summary_rows.append(
                _summary_row(
                    dataset="wearable",
                    condition=condition,
                    window=window,
                    metric="balanced_accuracy",
                    values=averaged,
                    seed=bootstrap_seed + condition_index * 100 + window_index,
                    bootstrap_samples=bootstrap_samples,
                )
            )
            for seed_index, seed in enumerate(seeds):
                for participant_index, participant in enumerate(subjects):
                    raw_rows.append({
                        "dataset": "wearable",
                        "condition": condition,
                        "window_s": float(window),
                        "participant": participant,
                        "seed": seed,
                        "balanced_accuracy": seed_matrix[seed_index, participant_index],
                    })
            for participant_index, participant in enumerate(subjects):
                average_rows.append({
                    "dataset": "wearable",
                    "condition": condition,
                    "window_s": float(window),
                    "participant": participant,
                    "seeds_averaged": len(seeds),
                    "balanced_accuracy": averaged[participant_index],
                })

    for contrast_index, frozen_contrast in enumerate(frozen["contrasts"]):
        window = str(float(frozen_contrast["window_seconds"]))
        left_name = str(frozen_contrast["left"])
        right_name = str(frozen_contrast["right"])
        left_subjects, left_matrix, _ = matrices[left_name][(window, "balanced_accuracy")]
        right_subjects, right_matrix, _ = matrices[right_name][(window, "balanced_accuracy")]
        if left_subjects != right_subjects:
            raise ValueError(f"electrode_transfer/{left_name}/{right_name}/{window}: participant mismatch")
        row = _contrast_row(
            family="electrode_transfer:wearable:all_registered_21_tests",
            dataset="wearable",
            left_name=left_name,
            right_name=right_name,
            window=window,
            metric="balanced_accuracy",
            left=left_matrix.mean(axis=0),
            right=right_matrix.mean(axis=0),
            bootstrap_seed=bootstrap_seed + 1000 + contrast_index,
            bootstrap_samples=bootstrap_samples,
        )
        row["contrast_label"] = str(frozen_contrast["label"])
        contrast_rows.append(row)
    _apply_holm_by_family(contrast_rows)
    for row, expected_row in zip(contrast_rows, frozen["contrasts"], strict=True):
        expected = expected_row["paired_left_minus_right"]
        for key, frozen_key in (
            ("mean_difference", "mean_difference"),
            ("wilcoxon_p_value", "p_value"),
            ("holm_adjusted_p_value", "holm_adjusted_p_value"),
        ):
            if abs(float(row[key]) - float(expected[frozen_key])) > 1e-12:
                raise ValueError(
                    f"electrode_transfer/{row['left']}/{row['right']}/{row['window_s']}: {key} differs"
                )

    _write_csv(
        output_dir / "electrode_transfer_participant_by_seed.csv",
        raw_rows,
        list(raw_rows[0]),
    )
    _write_csv(
        output_dir / "electrode_transfer_participant_seed_averaged.csv",
        average_rows,
        list(average_rows[0]),
    )
    _write_csv(
        output_dir / "electrode_transfer_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    _write_csv(
        output_dir / "electrode_transfer_contrasts.csv",
        contrast_rows,
        list(contrast_rows[0]),
    )
    return {
        "audit": {
            "participants": int(section["participants"]),
            "seeds": EXPECTED_SEEDS,
            "windows": windows,
            "multiplicity": "One Holm family across three directional contrasts and seven windows (21 paired tests).",
            "frozen_comparison": section["frozen_comparison"],
            "inputs": input_paths,
            "validation": "passed",
        },
        "summaries": summary_rows,
        "contrasts": contrast_rows,
    }


def _split_manifest_evidence(config: dict, output_dir: Path) -> dict:
    """Export and validate the exact subject partitions used by split-bearing runs."""

    collections: list[tuple[str, str, int, dict[str, list[str]]]] = []
    for dataset, section in config["main"].items():
        collections.append(("main", dataset, int(section["participants"]), section["conditions"]))
    ablations = config["ablations"]
    collections.append((
        "ablations", str(ablations["dataset"]), int(ablations["participants"]),
        ablations["conditions"],
    ))
    filter_bank = config["strong_filter_bank"]
    collections.append((
        "strong_filter_bank", str(filter_bank["dataset"]), int(filter_bank["participants"]),
        filter_bank["conditions"],
    ))
    transfer = config["electrode_transfer"]
    collections.append((
        "electrode_transfer", str(transfer["dataset"]), int(transfer["participants"]),
        transfer["conditions"],
    ))

    rows: list[dict] = []
    reference_splits: dict[tuple[str, str, int, str], tuple[tuple[str, ...], ...]] = {}
    coverage: dict[tuple[str, str, str, int], list[str]] = defaultdict(list)
    for collection, dataset, expected_participants, conditions in collections:
        for condition, patterns in conditions.items():
            paths, documents = _expand(patterns)
            for path, document in zip(paths, documents, strict=True):
                subjects = document.get("subjects")
                if not isinstance(subjects, dict):
                    raise ValueError(f"{path}: missing subject split metadata")
                train = tuple(map(str, subjects.get("train", [])))
                validation = tuple(map(str, subjects.get("validation", [])))
                test = tuple(map(str, subjects.get("test", [])))
                split_sets = [set(train), set(validation), set(test)]
                if any(split_sets[left] & split_sets[right] for left in range(3) for right in range(left + 1, 3)):
                    raise ValueError(f"{path}: train/validation/test participant overlap")
                if len(set().union(*split_sets)) != expected_participants:
                    raise ValueError(
                        f"{path}: expected {expected_participants} partitioned participants, "
                        f"found {len(set().union(*split_sets))}"
                    )
                seed = int(document["seed"])
                split = str(document["split"])
                signature = tuple(tuple(sorted(values, key=lambda value: (len(value), value))) for values in (train, validation, test))
                reference_key = (collection, dataset, seed, split)
                if reference_key in reference_splits and reference_splits[reference_key] != signature:
                    raise ValueError(
                        f"{collection}/{dataset}/{seed}/{split}: comparator split mismatch"
                    )
                reference_splits[reference_key] = signature
                coverage[(collection, dataset, str(condition), seed)].extend(test)
                rows.append({
                    "collection": collection,
                    "dataset": dataset,
                    "condition": condition,
                    "seed": seed,
                    "split": split,
                    "train_participants": ";".join(train),
                    "validation_participants": ";".join(validation),
                    "test_participants": ";".join(test),
                    "train_count": len(train),
                    "validation_count": len(validation),
                    "test_count": len(test),
                    "result_file": path.relative_to(ROOT).as_posix(),
                    "result_sha256": _sha256(path),
                })

    for (collection, dataset, condition, seed), test_subjects in coverage.items():
        expected = next(
            participants for name, data, participants, _ in collections
            if name == collection and data == dataset
        )
        if len(test_subjects) != expected or len(set(test_subjects)) != expected:
            raise ValueError(
                f"{collection}/{dataset}/{condition}/{seed}: test folds do not cover "
                f"each of {expected} participants exactly once"
            )

    rows.sort(key=lambda row: (
        row["collection"], row["dataset"], row["condition"], row["seed"], row["split"]
    ))
    _write_csv(output_dir / "split_manifest.csv", rows, list(rows[0]))
    audit = {
        "status": "passed",
        "rows": len(rows),
        "checks": [
            "train, validation and test participants are disjoint within every result",
            "the partition covers the declared dataset participant count",
            "test folds cover every participant exactly once per seed and condition",
            "all conditions within a comparison collection use identical folds",
        ],
    }
    atomic_write_json(output_dir / "split_manifest_audit.json", audit)
    return audit


def _deployment_evidence(config: dict, output_dir: Path) -> dict:
    section = config["deployment"]
    paths, documents = _expand(section["files"])
    rows: list[dict] = []
    observed: set[tuple[str, float]] = set()
    for path, document in zip(paths, documents, strict=True):
        architecture = str(document["architecture"])
        window = float(document["window_seconds"])
        observed.add((architecture, window))
        graph = document.get("deployment_graph", document)
        parameters = document.get("parameters", document.get("model_parameters"))
        if isinstance(parameters, dict):
            parameters = parameters["deployment_graph"]
        environment = document["environment"]
        rows.append({
            "architecture": architecture,
            "protocol_status": document.get("protocol_status", "proposed_folded_graph"),
            "window_s": window,
            "batch_size": int(document["batch_size"]),
            "dtype": str(document["dtype"]),
            "parameters": int(parameters),
            "cpu_iterations": int(graph["cpu"]["iterations"]),
            "cpu_p50_ms": float(graph["cpu"]["p50_ms"]),
            "cpu_p95_ms": float(graph["cpu"]["p95_ms"]),
            "cuda_iterations": int(graph["cuda"]["iterations"]),
            "cuda_p50_ms": float(graph["cuda"]["p50_ms"]),
            "cuda_p95_ms": float(graph["cuda"]["p95_ms"]),
            "cuda_peak_allocated_bytes": int(graph["cuda"]["peak_allocated_bytes"]),
            "platform": environment["platform"],
            "python": environment["python"],
            "torch": environment["torch"],
            "cpu_threads_timed": int(environment["cpu_threads_timed"]),
            "cuda": environment["cuda"],
            "gpu": environment["gpu"],
            "session_scope": section["session_scope"],
            "source_file": path.relative_to(ROOT).as_posix(),
            "source_sha256": _sha256(path),
        })
    expected = {
        (str(method), float(window))
        for method in section["expected_methods"]
        for window in section["expected_windows"]
    }
    if observed != expected:
        raise ValueError(
            f"deployment method/window grid mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    rows.sort(key=lambda row: (row["architecture"], row["window_s"]))
    _write_csv(output_dir / "deployment_measurements.csv", rows, list(rows[0]))

    checkpoint_rows = []
    for pattern in section["checkpoint_files"]:
        for path in sorted(Path(value) for value in glob.glob(str(ROOT / pattern))):
            checkpoint_rows.append({
                "checkpoint": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    if not checkpoint_rows:
        raise FileNotFoundError("deployment checkpoint manifest is empty")
    _write_csv(
        output_dir / "deployment_checkpoint_manifest.csv",
        checkpoint_rows,
        list(checkpoint_rows[0]),
    )
    return {
        "status": "passed",
        "measurements": len(rows),
        "checkpoints": len(checkpoint_rows),
        "benchmark_command": section["benchmark_command"],
        "session_scope": section["session_scope"],
    }


def _write_figure_source_data(config: dict, bundle: dict) -> None:
    figure_data = ROOT / "paper" / "figures" / "source_data"
    main_summaries = {
        (row["dataset"], row["condition"], row["window_s"], row["metric"]): row
        for row in bundle["main"]["summaries"]
    }
    main_contrasts = {
        (row["dataset"], row["window_s"], row["metric"]): row
        for row in bundle["main"]["contrasts"]
    }
    rows = []
    method_names = {
        "harmonic_fold": "HarmonicFoldNet",
        "ssvepformer": "Spectral Transformer",
    }
    for dataset, dataset_config in config["main"].items():
        display_dataset = "BETA" if dataset == "beta" else "Benchmark"
        for window in dataset_config["display_windows"]:
            window_value = float(window)
            contrast = main_contrasts[(dataset, window_value, "balanced_accuracy")]
            for condition in ("harmonic_fold", "ssvepformer"):
                summary = main_summaries[
                    (dataset, condition, window_value, "balanced_accuracy")
                ]
                is_main = condition == "harmonic_fold"
                rows.append({
                    "dataset": display_dataset,
                    "window_s": window_value,
                    "method": method_names[condition],
                    "balanced_accuracy_pct": 100.0 * summary["mean"],
                    "ci95_low_pct": 100.0 * summary["bootstrap_mean_ci95_low"],
                    "ci95_high_pct": 100.0 * summary["bootstrap_mean_ci95_high"],
                    "difference_hfn_minus_reference_pp": (
                        100.0 * contrast["mean_difference"] if is_main else ""
                    ),
                    "holm_p": contrast["holm_adjusted_p_value"] if is_main else "",
                })
    _write_csv(
        figure_data / "fig2_calibration_free.csv",
        rows,
        list(rows[0]),
    )

    fig3_path = figure_data / "fig3_harmonic_mechanism.csv"
    no_bias = {
        float(row["window_s"]): row
        for row in bundle["ablations"]["contrasts"]
        if row["right"] == "no_harmonic_bias"
    }
    mechanism = {
        (row["metric"], float(row["window_s"])): row
        for row in bundle["attention_mechanism"]["contrasts"]
    }
    updated = []
    for record, metric in (
        ("harmonic_neighbourhood_mass", "harmonic_neighborhood_mass"),
        ("peak_within_target", "peak_within_bias_width_fraction"),
    ):
        for window in (0.4, 0.8, 1.2):
            contrast = mechanism[(metric, window)]
            for condition, key in (
                ("bias", "with_harmonic_bias_mean"),
                ("masked", "bias_masked_mean"),
            ):
                updated.append({
                    "record": record,
                    "window_s": window,
                    "condition": condition,
                    "value_pct": 100.0 * contrast[key],
                    "ci95_low_pct": "",
                    "ci95_high_pct": "",
                    "holm_p": contrast["holm_adjusted_p_value"],
                })
    for window in config["ablations"]["windows"]:
        contrast = no_bias[float(window)]
        updated.append({
            "record": "full_minus_no_bias_accuracy",
            "window_s": float(window),
            "condition": "from_scratch_ablation",
            "value_pct": 100.0 * contrast["mean_difference"],
            "ci95_low_pct": 100.0 * contrast["bootstrap_difference_ci95_low"],
            "ci95_high_pct": 100.0 * contrast["bootstrap_difference_ci95_high"],
            "holm_p": contrast["holm_adjusted_p_value"],
        })
    _write_csv(fig3_path, updated, list(updated[0]))

    adaptation_index = {
        (row["condition"], float(row["window_s"])): row
        for row in bundle["adaptation"]["summaries"]
    }
    adaptation_rows = []
    for condition, block_label in (
        ("zero_shot", "0 blocks"),
        ("adapter_1_block", "1 block"),
        ("adapter_2_blocks", "2 blocks"),
    ):
        for window in (0.4, 0.8, 1.2):
            row = adaptation_index[(condition, window)]
            adaptation_rows.append({
                "record": "calibration_curve",
                "window_s": window,
                "method_or_blocks": block_label,
                "balanced_accuracy_pct": 100.0 * row["mean"],
                "ci95_low_pct": 100.0 * row["bootstrap_mean_ci95_low"],
                "ci95_high_pct": 100.0 * row["bootstrap_mean_ci95_high"],
            })
    for condition, label in (
        ("adapter_2_blocks", "113-parameter adapter"),
        ("trca_2_blocks", "TRCA"),
        ("etrca_2_blocks", "eTRCA"),
        ("tdca_2_blocks", "TDCA"),
    ):
        for window in (0.4, 0.8, 1.2):
            row = adaptation_index[(condition, window)]
            adaptation_rows.append({
                "record": "two_block_comparison",
                "window_s": window,
                "method_or_blocks": label,
                "balanced_accuracy_pct": 100.0 * row["mean"],
                "ci95_low_pct": 100.0 * row["bootstrap_mean_ci95_low"],
                "ci95_high_pct": 100.0 * row["bootstrap_mean_ci95_high"],
            })
    _write_csv(
        figure_data / "fig4_adaptation.csv",
        adaptation_rows,
        list(adaptation_rows[0]),
    )

    deployment_path = figure_data / "fig6_deployment_pareto.csv"
    with deployment_path.open("r", encoding="utf-8-sig", newline="") as handle:
        deployment_rows = list(csv.DictReader(handle))
    fb_index = {
        (row["condition"], float(row["window_s"])): row
        for row in bundle["strong_filter_bank"]["summaries"]
    }
    mts_index = {
        (row["dataset"], row["condition"], float(row["window_s"])): row
        for row in bundle["mtsnet"]["summaries"]
    }
    accuracy_sources = {
        "HarmonicFoldNet": main_summaries[("beta", "harmonic_fold", 1.2, "balanced_accuracy")],
        "Spectral Transformer": main_summaries[("beta", "ssvepformer", 1.2, "balanced_accuracy")],
        "Filter-bank Transformer": fb_index[("fb_ssvepformer", 1.2)],
        "MTSNet reconstruction": mts_index[("beta", "mtsnet", 1.2)],
    }
    for row in deployment_rows:
        source = accuracy_sources[row["method"]]
        row["balanced_accuracy_pct_1_2s"] = 100.0 * float(source["mean"])
        row["accuracy_ci95_low_pct"] = 100.0 * float(source["bootstrap_mean_ci95_low"])
        row["accuracy_ci95_high_pct"] = 100.0 * float(source["bootstrap_mean_ci95_high"])
    _write_csv(deployment_path, deployment_rows, list(deployment_rows[0]))


def _fmt_mean_ci(row: dict, *, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    return (
        f"{scale * row['mean']:.2f} "
        f"[{scale * row['bootstrap_mean_ci95_low']:.2f}, "
        f"{scale * row['bootstrap_mean_ci95_high']:.2f}]"
    )


def _fmt_p(value: float) -> str:
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _markdown_report(config: dict, bundle: dict) -> str:
    main_summaries = bundle["main"]["summaries"]
    main_contrasts = bundle["main"]["contrasts"]
    lines = [
        "# Submission evidence restoration report",
        "",
        "Generated from the preserved per-participant, per-seed run artifacts. The participant is the independent unit; the three training seeds are averaged within participant before inference.",
        "",
        "## Restored evidence",
        "",
        "- Main Benchmark and BETA results: 35 and 70 held-out participants, respectively, across seeds 20260929, 20260930 and 20260931.",
        "- Participant-level ITR: calculated in each original run with 40 targets and selection time equal to the observation window plus 0.5 s, then averaged across seeds within participant.",
        "- Final BETA component ablations: full, no-local, no-attention and no-harmonic-bias conditions for all 70 participants and three seeds.",
        "- Attention mechanism analysis: four participant-level metrics across three windows with one conservative 12-test Holm family.",
        "- MTSNet multiplicity audit: one Holm family across Benchmark/BETA and 0.4/0.8/1.2 s (six tests).",
        "",
        "## Participant-level ITR",
        "",
        "Values are mean [participant-bootstrap 95% CI] in bits/min. ITR is descriptive; the primary inferential endpoint remains participant-level balanced accuracy.",
        "",
        "| Dataset | Window (s) | HarmonicFoldNet | Spectral Transformer | Mean difference [95% CI] |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    summary_index = {
        (row["dataset"], row["condition"], row["window_s"], row["metric"]): row
        for row in main_summaries
    }
    contrast_index = {
        (row["dataset"], row["window_s"], row["metric"]): row
        for row in main_contrasts
    }
    for dataset, dataset_config in config["main"].items():
        for window in dataset_config["display_windows"]:
            window_value = float(window)
            left = summary_index[(dataset, "harmonic_fold", window_value, "itr_bits_per_minute")]
            right = summary_index[(dataset, "ssvepformer", window_value, "itr_bits_per_minute")]
            contrast = contrast_index[(dataset, window_value, "itr_bits_per_minute")]
            lines.append(
                f"| {dataset.upper() if dataset == 'beta' else 'Benchmark'} | {window_value:.1f} | "
                f"{_fmt_mean_ci(left)} | {_fmt_mean_ci(right)} | "
                f"{contrast['mean_difference']:.2f} "
                f"[{contrast['bootstrap_difference_ci95_low']:.2f}, "
                f"{contrast['bootstrap_difference_ci95_high']:.2f}] |"
            )

    lines.extend([
        "",
        "## Final component ablations",
        "",
        "Differences are full minus ablated balanced accuracy in percentage points. Holm correction is applied over the six registered windows separately for each prespecified component ablation.",
        "",
        "| Ablation | Window (s) | Difference [95% CI], pp | Rank-biserial | Holm p |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in bundle["ablations"]["contrasts"]:
        lines.append(
            f"| {row['right']} | {row['window_s']:.1f} | "
            f"{100 * row['mean_difference']:.2f} "
            f"[{100 * row['bootstrap_difference_ci95_low']:.2f}, "
            f"{100 * row['bootstrap_difference_ci95_high']:.2f}] | "
            f"{row['rank_biserial']:.3f} | {_fmt_p(row['holm_adjusted_p_value'])} |"
        )

    lines.extend([
        "",
        "## MTSNet multiplicity correction",
        "",
        "| Dataset | Window (s) | Difference, pp | Raw p | Holm p (six-test family) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in bundle["mtsnet"]["contrasts"]:
        lines.append(
            f"| {str(row['dataset']).upper() if row['dataset'] == 'beta' else 'Benchmark'} | "
            f"{row['window_s']:.1f} | {100 * row['mean_difference']:.2f} | "
            f"{_fmt_p(row['wilcoxon_p_value'])} | "
            f"{_fmt_p(row['holm_adjusted_p_value_six_test_family'])} |"
        )

    lines.extend([
        "",
        "## Integrity result",
        "",
        "All configured groups contained the registered three seeds and the complete participant set. Recomputed seed-averaged balanced-accuracy means matched the frozen comparison artifacts to numerical tolerance. No group-mean accuracy was substituted into the nonlinear ITR formula.",
        "",
    ])
    return "\n".join(lines)


def run(
    config_path: Path,
    output_dir: Path,
    report_path: Path | None = None,
    *,
    update_figure_data: bool = True,
) -> dict:
    config = _load_json(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": 1,
        "config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "main": _main_evidence(config, output_dir),
        "itr_sensitivity": _itr_sensitivity_evidence(config, output_dir),
        "ablations": _ablation_evidence(config, output_dir),
        "attention_mechanism": _attention_mechanism_evidence(config, output_dir),
        "analytic_baselines": _analytic_baseline_evidence(config, output_dir),
        "strong_filter_bank": _strong_filter_bank_evidence(config, output_dir),
        "adaptation": _adaptation_evidence(config, output_dir),
        "electrode_transfer": _electrode_transfer_evidence(config, output_dir),
        "mtsnet": _mtsnet_evidence(config, output_dir),
        "split_manifest": _split_manifest_evidence(config, output_dir),
        "deployment": _deployment_evidence(config, output_dir),
    }
    atomic_write_json(output_dir / "submission_evidence.json", bundle)
    if update_figure_data:
        _write_figure_source_data(config, bundle)
    if report_path is None:
        report_path = ROOT / "paper" / "SUBMISSION_EVIDENCE_REPORT.md"
    _atomic_write_text(report_path, _markdown_report(config, bundle))
    readme = """# Source data\n\nThis directory is generated by `scripts/build_submission_evidence.py` from the preserved run artifacts listed in `paper/SUBMISSION_EVIDENCE_CONFIG.json`.\n\nThe independent statistical unit is the participant. Training-seed repeats are averaged within participant before summaries and paired tests. CSV files ending in `by_seed` retain the seed-level audit trail; files ending in `seed_averaged` contain the participant-level values used for inference. The attention-allocation tables retain participant-level bias-present and bias-masked measurements. Calibration-free analytic references, strong filter-bank and MTSNet comparisons, low-parameter adaptation, and directional electrode-transfer evidence are reconstructed from their original participant records and checked against the frozen aggregate artifacts.\n\n`split_manifest.csv` records the exact train, validation and test participants for every split-bearing comparison result and `split_manifest_audit.json` records the disjointness, coverage and comparator-alignment checks. `deployment_measurements.csv` contains all method-by-window timing records, and `deployment_checkpoint_manifest.csv` pins the benchmarked HarmonicFoldNet checkpoint. `submission_evidence.json` records summaries, paired effects, bootstrap confidence intervals, correction families and input provenance. `folding_equivalence.json` is produced by `scripts/verify_folding_equivalence.py` and records the held-out train-to-deploy graph audit across preserved checkpoints and windows.\n"""
    _atomic_write_text(output_dir / "README.md", readme)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build participant-level source data and submission statistics."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "paper" / "SUBMISSION_EVIDENCE_REPORT.md",
    )
    args = parser.parse_args()
    bundle = run(args.config.resolve(), args.output_dir.resolve(), args.report.resolve())
    print(json.dumps({
        "status": "ok",
        "main_datasets": sorted(bundle["main"]["audit"]["datasets"]),
        "itr_sensitivity_rows": len(bundle["itr_sensitivity"]["summaries"]),
        "ablation_contrasts": len(bundle["ablations"]["contrasts"]),
        "attention_mechanism_tests": len(bundle["attention_mechanism"]["contrasts"]),
        "analytic_baseline_rows": len(bundle["analytic_baselines"]["summaries"]),
        "strong_filter_bank_tests": len(bundle["strong_filter_bank"]["contrasts"]),
        "adaptation_tests": len(bundle["adaptation"]["contrasts"]),
        "electrode_transfer_tests": len(bundle["electrode_transfer"]["contrasts"]),
        "mtsnet_tests": len(bundle["mtsnet"]["contrasts"]),
        "split_manifest_rows": bundle["split_manifest"]["rows"],
        "deployment_measurements": bundle["deployment"]["measurements"],
        "output": str(args.output_dir.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
