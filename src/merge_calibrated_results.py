from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .paper_metrics import atomic_write_json, information_transfer_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_multidataset.yaml")
    parser.add_argument("--parts", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    documents = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.parts]
    datasets = {document["dataset"] for document in documents}
    if len(datasets) != 1:
        raise ValueError(f"parts contain different datasets: {sorted(datasets)}")
    dataset = datasets.pop()
    for document in documents[1:]:
        for key in ("supervision_track", "calibration_schedule", "signal_contract"):
            if document[key] != documents[0][key]:
                raise ValueError(f"parts contain different {key} contracts")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    classes = int(config["datasets"][dataset]["classes"])
    cue_seconds = float(config["cue_seconds_for_itr"])
    output = {
        key: value for key, value in documents[0].items() if key != "methods"
    }
    output["parts"] = [str(Path(path)) for path in args.parts]
    output["methods"] = {}
    for document in documents:
        for method, counts in document["methods"].items():
            for count, windows in counts.items():
                for window, row in windows.items():
                    destination = output["methods"].setdefault(method, {}).setdefault(
                        count, {},
                    ).setdefault(window, {"per_subject": {}})
                    overlap = set(destination["per_subject"]).intersection(row["per_subject"])
                    if overlap:
                        raise ValueError(f"duplicate subjects in parts: {sorted(overlap)}")
                    destination["per_subject"].update(row["per_subject"])
    for counts in output["methods"].values():
        for windows in counts.values():
            for window, row in windows.items():
                subjects = list(row["per_subject"].values())
                weights = np.asarray([
                    float(value.get("trials", 1)) for value in subjects
                ])
                accuracy = float(np.average(
                    [value["accuracy"] for value in subjects], weights=weights,
                ))
                balanced = float(np.average([
                    value["balanced_accuracy"] for value in subjects
                ], weights=weights))
                row["overall"] = {
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced,
                    "itr_bits_per_minute": information_transfer_rate(
                        classes, accuracy, float(window) + cue_seconds,
                    ),
                    "trials": (
                        int(weights.sum())
                        if all("trials" in value for value in subjects) else None
                    ),
                }
    atomic_write_json(args.output, output)


if __name__ == "__main__":
    main()
