from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

from .harmonic_fold import HarmonicFoldNet
from .paper_data import class_frequencies, class_phases, load_subjects_window
from .paper_metrics import atomic_write_json


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    result = payload["result"]
    config = payload["config"]
    if result["architecture"] != "harmonic_fold_v4_1":
        raise ValueError("attention analysis requires a harmonic_fold_v4_1 checkpoint")
    dataset_cfg = config["datasets"][result["dataset"]]
    subjects = list(map(str, result["subjects"][args.split]))
    frequencies = class_frequencies(
        dataset_cfg["shard_dir"], dataset_cfg["subjects"],
        classes=int(dataset_cfg["classes"]),
    )
    phases = class_phases(
        dataset_cfg["shard_dir"], dataset_cfg["subjects"],
        classes=int(dataset_cfg["classes"]),
    )
    model = HarmonicFoldNet(
        channels=len(dataset_cfg["channels"]),
        sample_rate=int(config["sample_rate"]),
        class_frequencies=frequencies,
        class_phases=phases,
        **config["harmonic_fold_v4_1_model"],
    )
    model.load_state_dict(payload["state_dict"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    x, y, subject_ids = load_subjects_window(
        dataset_cfg["shard_dir"], subjects,
        window_seconds=float(args.window),
        sample_rate=int(config["sample_rate"]),
        channels=list(map(str, dataset_cfg["channels"])),
        trial_filters=dataset_cfg.get("trial_filters"),
    )

    rows: dict[str, list[np.ndarray]] = {
        "normalized_entropy": [],
        "peak_distance_hz": [],
        "harmonic_neighborhood_mass": [],
    }
    no_bias_rows: dict[str, list[np.ndarray]] = {
        "normalized_entropy": [],
        "peak_distance_hz": [],
        "harmonic_neighborhood_mass": [],
    }
    token_hz = model.fusion_attention.spectral_token_hz.detach().cpu().numpy()
    target_hz = model.fusion_attention.harmonic_target_hz.detach().cpu().numpy()
    width = float(model.fusion_attention.bias_width_hz)

    for begin in range(0, len(y), int(args.batch_size)):
        batch_x = torch.from_numpy(x[begin:begin + int(args.batch_size)]).to(device)
        batch_y = y[begin:begin + int(args.batch_size)]
        for use_bias, destination in ((True, rows), (False, no_bias_rows)):
            probabilities = model.harmonic_attention_probabilities(
                batch_x, use_harmonic_bias=use_bias,
            ).float().cpu().numpy()
            chosen = probabilities[np.arange(len(batch_y)), :, batch_y, :]
            entropy = -(chosen * np.log(np.clip(chosen, 1e-12, None))).sum(axis=-1)
            destination["normalized_entropy"].append(entropy / np.log(chosen.shape[-1]))
            peak_hz = token_hz[chosen.argmax(axis=-1)]
            targets = target_hz[batch_y]
            peak_distance = np.abs(peak_hz[..., None] - targets[:, None, :]).min(axis=-1)
            destination["peak_distance_hz"].append(peak_distance)
            neighborhood = (
                np.abs(token_hz[None, :] - targets[:, :, None]).min(axis=1) <= width
            )
            mass = (chosen * neighborhood[:, None, :]).sum(axis=-1)
            destination["harmonic_neighborhood_mass"].append(mass)

    def finalize(source: dict[str, list[np.ndarray]]) -> dict[str, object]:
        entropy_matrix = np.concatenate(source["normalized_entropy"])
        distance_matrix = np.concatenate(source["peak_distance_hz"])
        mass_matrix = np.concatenate(source["harmonic_neighborhood_mass"])
        entropy = entropy_matrix.reshape(-1)
        distance = distance_matrix.reshape(-1)
        mass = mass_matrix.reshape(-1)
        per_subject = {}
        for subject in sorted(set(map(str, subject_ids)), key=lambda value: (len(value), value)):
            mask = np.asarray(subject_ids).astype(str) == subject
            per_subject[subject] = {
                "trials": int(mask.sum()),
                "normalized_entropy": float(entropy_matrix[mask].mean()),
                "peak_distance_hz": float(distance_matrix[mask].mean()),
                "peak_within_bias_width_fraction": float((distance_matrix[mask] <= width).mean()),
                "peak_within_1hz_fraction": float((distance_matrix[mask] <= 1.0).mean()),
                "harmonic_neighborhood_mass": float(mass_matrix[mask].mean()),
            }
        return {
            "normalized_entropy": _distribution_summary(entropy),
            "peak_distance_hz": _distribution_summary(distance),
            "peak_within_bias_width_fraction": float((distance <= width).mean()),
            "peak_within_1hz_fraction": float((distance <= 1.0).mean()),
            "harmonic_neighborhood_mass": _distribution_summary(mass),
            "per_subject": per_subject,
        }

    with_bias = finalize(rows)
    without_bias = finalize(no_bias_rows)
    subject_order = list(with_bias["per_subject"])
    paired = {}
    for metric in (
        "normalized_entropy", "peak_distance_hz", "peak_within_bias_width_fraction",
        "peak_within_1hz_fraction", "harmonic_neighborhood_mass",
    ):
        present = np.asarray([with_bias["per_subject"][s][metric] for s in subject_order])
        removed = np.asarray([without_bias["per_subject"][s][metric] for s in subject_order])
        differences = present - removed
        p_value = 1.0 if np.allclose(differences, 0.0) else float(
            wilcoxon(present, removed, alternative="two-sided").pvalue
        )
        paired[metric] = {
            "with_minus_without_mean": float(differences.mean()),
            "with_minus_without_median": float(np.median(differences)),
            "wilcoxon_two_sided_p": p_value,
            "subjects": int(len(subject_order)),
        }

    output: dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": result["dataset"],
        "split": args.split,
        "subjects": sorted(set(map(str, subject_ids)), key=lambda value: (len(value), value)),
        "trials": int(len(y)),
        "window_seconds": float(args.window),
        "heads": int(model.fusion_attention.heads),
        "spectral_tokens": int(len(token_hz)),
        "harmonic_bias_width_hz": width,
        "true_class_query_only": True,
        "with_harmonic_bias": with_bias,
        "same_checkpoint_without_harmonic_bias": without_bias,
        "paired_subject_analysis": paired,
    }
    atomic_write_json(args.output, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--window", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
