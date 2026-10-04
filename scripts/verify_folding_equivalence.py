from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from src.harmonic_fold import HarmonicFoldNet
from src.model import parameter_count, reparameterize_model
from src.paper_data import class_frequencies, class_phases, load_subjects_window
from src.paper_metrics import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]


def _checkpoint_paths(config: dict) -> list[Path]:
    paths: list[Path] = []
    for dataset in ("benchmark", "beta"):
        patterns = config["mtsnet"]["harmonic_fold_sources"][dataset]
        for pattern in patterns:
            for result_path in glob.glob(str(ROOT / pattern)):
                checkpoint = Path(result_path).with_name("model.pt")
                if checkpoint.exists():
                    paths.append(checkpoint)
    unique = sorted(set(paths))
    if len(unique) < 20:
        raise RuntimeError(
            f"expected at least 20 preserved checkpoints for the equivalence audit, found {len(unique)}"
        )
    return unique


def _representative_indices(subject_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    selected: list[int] = []
    for subject in sorted(np.unique(subject_ids), key=lambda value: (len(str(value)), str(value))):
        subject_indices = np.flatnonzero(subject_ids.astype(str) == str(subject))
        subject_labels = labels[subject_indices]
        for label in np.unique(subject_labels):
            selected.append(int(subject_indices[np.flatnonzero(subject_labels == label)[0]]))
    return np.asarray(sorted(set(selected)), dtype=np.int64)


@torch.inference_mode()
def run(config_path: Path, output: Path, *, batch_size: int = 256) -> dict:
    evidence_config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoints = _checkpoint_paths(evidence_config)
    rows: list[dict] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_examples = 0
    total_predictions = 0
    mismatches = 0
    global_max = 0.0
    global_median_errors: list[float] = []
    train_parameters: set[int] = set()
    deployment_parameters: set[int] = set()

    for checkpoint_index, checkpoint in enumerate(checkpoints, 1):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        result = payload["result"]
        if result["architecture"] != "harmonic_fold_v4_1":
            raise ValueError(f"unsupported architecture in {checkpoint}: {result['architecture']}")
        config = payload["config"]
        dataset = str(result["dataset"])
        dataset_config = config["datasets"][dataset]
        sample_rate = int(config["sample_rate"])
        all_subjects = list(map(str, dataset_config["subjects"]))
        test_subjects = list(map(str, result["subjects"]["test"]))
        frequencies = class_frequencies(
            dataset_config["shard_dir"], all_subjects, classes=int(dataset_config["classes"])
        )
        phases = class_phases(
            dataset_config["shard_dir"], all_subjects, classes=int(dataset_config["classes"])
        )
        training = HarmonicFoldNet(
            channels=len(dataset_config["channels"]),
            sample_rate=sample_rate,
            class_frequencies=frequencies,
            class_phases=phases,
            **config["harmonic_fold_v4_1_model"],
        ).eval()
        training.load_state_dict(payload["state_dict"])
        deployment = reparameterize_model(training)
        train_parameters.add(parameter_count(training))
        deployment_parameters.add(parameter_count(deployment))
        training = training.to(device)
        deployment = deployment.to(device)

        windows = [float(value) for value in result["selection_windows"]]
        max_window = max(windows)
        x, y, subject_ids = load_subjects_window(
            dataset_config["shard_dir"],
            test_subjects,
            window_seconds=max_window,
            sample_rate=sample_rate,
            channels=list(map(str, dataset_config["channels"])),
            trial_filters=dataset_config.get("trial_filters"),
        )
        selected = _representative_indices(subject_ids, y)
        x = x[selected]
        y = y[selected]
        selected_subjects = subject_ids[selected]
        if set(map(str, np.unique(selected_subjects))) != set(test_subjects):
            raise RuntimeError(f"{checkpoint}: representative batch omitted a test participant")

        for window in windows:
            samples = int(round(window * sample_rate))
            errors: list[np.ndarray] = []
            window_mismatches = 0
            predictions = 0
            for start in range(0, len(x), batch_size):
                batch = torch.from_numpy(x[start : start + batch_size, :, :samples]).to(device)
                original = training.forward_mode(batch, "full")
                folded = deployment.forward_mode(batch, "full")
                absolute = (original - folded).abs().detach().cpu().numpy()
                errors.append(absolute.reshape(-1))
                left = original.argmax(dim=1)
                right = folded.argmax(dim=1)
                window_mismatches += int((left != right).sum().item())
                predictions += int(len(batch))
            error_values = np.concatenate(errors)
            maximum_error = float(error_values.max())
            median_error = float(np.median(error_values))
            global_max = max(global_max, maximum_error)
            global_median_errors.append(median_error)
            mismatches += window_mismatches
            total_predictions += predictions
            total_examples += len(x)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(result["seed"]),
                    "split": str(result["split"]),
                    "window_seconds": window,
                    "participants": len(test_subjects),
                    "examples": len(x),
                    "logits": int(len(error_values)),
                    "maximum_absolute_logit_error": maximum_error,
                    "median_absolute_logit_error": median_error,
                    "prediction_mismatches": window_mismatches,
                }
            )
        print(
            json.dumps(
                {
                    "checkpoint": checkpoint_index,
                    "total": len(checkpoints),
                    "dataset": dataset,
                    "seed": result["seed"],
                    "split": result["split"],
                    "examples_per_window": len(x),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    report = {
        "status": "complete",
        "device": str(device),
        "datasets": ["benchmark", "beta"],
        "checkpoints": len(checkpoints),
        "windows_per_checkpoint": 5,
        "test_participants_covered_per_fold": True,
        "selection": "first trial of every class for every held-out participant",
        "checkpoint_coverage": (
            "all 15 preserved BETA fold-seed checkpoints and the five preserved "
            "Benchmark seed-20260929 fold checkpoints"
        ),
        "training_graph_parameters": sorted(train_parameters),
        "deployment_graph_parameters": sorted(deployment_parameters),
        "prediction_comparisons": total_predictions,
        "prediction_mismatches": mismatches,
        "maximum_absolute_logit_error": global_max,
        "median_of_row_median_absolute_logit_errors": float(np.median(global_median_errors)),
        "rows": rows,
    }
    if mismatches:
        raise RuntimeError(f"folded graph changed {mismatches} predictions")
    if global_max > 2e-5:
        raise RuntimeError(f"folded graph exceeded absolute tolerance: {global_max}")
    atomic_write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "paper" / "SUBMISSION_EVIDENCE_CONFIG.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper" / "source_data" / "folding_equivalence.json",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    report = run(args.config.resolve(), args.output.resolve(), batch_size=args.batch_size)
    print(json.dumps({key: report[key] for key in (
        "status", "device", "checkpoints", "prediction_comparisons",
        "prediction_mismatches", "maximum_absolute_logit_error",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
