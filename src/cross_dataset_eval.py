from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .harmonic_fold import HarmonicFoldNet, TemporalFusionDecoder
from .model import LegacyFusionNet
from .paper_data import class_frequencies, class_phases, load_subjects_window
from .paper_metrics import atomic_write_json
from .paper_train import infer_logits, split_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_result = payload["result"]
    config = payload["config"]
    source_cfg = config["datasets"][source_result["dataset"]]
    target_cfg = config["datasets"][args.target_dataset]
    candidate_conditioned = source_result["architecture"] in {
        "temporal_fusion", "harmonic_fold_v4_1",
    }
    if not candidate_conditioned and int(source_cfg["classes"]) != int(target_cfg["classes"]):
        raise RuntimeError("direct transfer requires identical class counts")
    source_frequencies = class_frequencies(
        source_cfg["shard_dir"], source_cfg["subjects"], classes=int(source_cfg["classes"]),
    )
    target_frequencies = class_frequencies(
        target_cfg["shard_dir"], target_cfg["subjects"], classes=int(target_cfg["classes"]),
    )
    source_phases = class_phases(
        source_cfg["shard_dir"], source_cfg["subjects"], classes=int(source_cfg["classes"]),
    )
    target_phases = class_phases(
        target_cfg["shard_dir"], target_cfg["subjects"], classes=int(target_cfg["classes"]),
    )
    if not candidate_conditioned:
        if not np.allclose(source_frequencies, target_frequencies, atol=1e-4):
            raise RuntimeError("direct transfer requires identical ordered class frequencies")
        if not np.allclose(source_phases, target_phases, atol=1e-4):
            raise RuntimeError("direct transfer requires identical ordered class phases")
    if list(map(str, source_cfg["channels"])) != list(map(str, target_cfg["channels"])):
        raise RuntimeError("direct transfer requires identical channel order")
    windows = [
        float(window) for window in target_cfg["windows"]
        if float(window) <= max(map(float, source_cfg["windows"]))
    ]
    sample_rate = int(config["sample_rate"])
    x, y, subject_ids = load_subjects_window(
        target_cfg["shard_dir"], target_cfg["subjects"], window_seconds=max(windows),
        sample_rate=sample_rate, channels=target_cfg["channels"],
        trial_filters=target_cfg.get("trial_filters"),
    )
    if source_result["architecture"] == "temporal_fusion":
        model = TemporalFusionDecoder(
            channels=len(target_cfg["channels"]), sample_rate=sample_rate,
            class_frequencies=target_frequencies, class_phases=target_phases,
            **config["harmonic_fold_model"],
        )
    elif source_result["architecture"] == "harmonic_fold_v4_1":
        model = HarmonicFoldNet(
            channels=len(target_cfg["channels"]), sample_rate=sample_rate,
            class_frequencies=target_frequencies, class_phases=target_phases,
            **config["harmonic_fold_v4_1_model"],
        )
    elif source_result["architecture"] == "legacy_fusion":
        model = LegacyFusionNet(
            channels=len(source_cfg["channels"]), classes=int(source_cfg["classes"]),
            sample_rate=sample_rate, class_frequencies=source_frequencies,
            class_phases=source_phases, **config["model"],
        )
    else:
        raise ValueError(f"unsupported checkpoint architecture: {source_result['architecture']}")
    state_dict = payload["state_dict"]
    target_candidate_keys: list[str] = []
    if candidate_conditioned:
        target_candidate_keys = [
            key for key in state_dict
            if key.endswith("class_frequencies") or key.endswith("class_phases")
        ]
        state_dict = {
            key: value for key, value in state_dict.items()
            if key not in target_candidate_keys
        }
    incompatible = model.load_state_dict(state_dict, strict=not candidate_conditioned)
    if candidate_conditioned:
        if set(incompatible.missing_keys) != set(target_candidate_keys):
            raise RuntimeError(
                "unexpected target-conditioned state mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"unexpected source checkpoint keys: {incompatible.unexpected_keys}"
            )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    output = {
        "status": "complete", "source_dataset": source_result["dataset"],
        "target_dataset": args.target_dataset, "checkpoint": str(Path(args.checkpoint).resolve()),
        "adaptation": "none", "subjects": list(map(str, target_cfg["subjects"])),
        "candidate_conditioning": (
            "target frequency/phase metadata only; no target EEG or labels used for fitting"
            if candidate_conditioned else "source candidate order reused"
        ),
        "target_conditioned_state_keys": target_candidate_keys,
        "windows": {},
    }
    for window in windows:
        logits = infer_logits(
            model, x, architecture=source_result["architecture"], mode=source_result["mode"], window=window,
            sample_rate=sample_rate, batch_size=int(config["eval_batch_size"]), device=device,
        )
        overall, per_subject = split_metrics(
            y, subject_ids, logits, classes=int(target_cfg["classes"]), window=window,
            cue_seconds=float(config["cue_seconds_for_itr"]),
        )
        output["windows"][str(window)] = {"overall": overall, "per_subject": per_subject}
        print({"window": window, **overall}, flush=True)
    atomic_write_json(args.output, output)


if __name__ == "__main__":
    main()
