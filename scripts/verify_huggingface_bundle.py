from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from huggingface.load_model import load_harmonic_fold_checkpoint


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "huggingface"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest["checkpoints"]) != 15:
        raise RuntimeError("the release must contain 15 BETA fold checkpoints")

    expected_grid = {(seed, fold) for seed in (20260929, 20260930, 20260931) for fold in range(5)}
    observed_grid = {(int(entry["seed"]), int(entry["fold"])) for entry in manifest["checkpoints"]}
    if observed_grid != expected_grid:
        raise RuntimeError(f"checkpoint seed/fold grid mismatch: {sorted(observed_grid)}")
    all_test_subjects: dict[int, list[str]] = {seed: [] for seed in (20260929, 20260930, 20260931)}

    for entry in manifest["checkpoints"]:
        model_path = BUNDLE / entry["model"]
        result_path = BUNDLE / entry["result"]
        if sha256(model_path) != entry["model_sha256"]:
            raise RuntimeError(f"checkpoint checksum mismatch: {model_path}")
        if sha256(result_path) != entry["result_sha256"]:
            raise RuntimeError(f"result checksum mismatch: {result_path}")
        model, payload = load_harmonic_fold_checkpoint(model_path, folded=True)
        if sum(parameter.numel() for parameter in model.parameters()) != 435043:
            raise RuntimeError(f"unexpected folded parameter count: {model_path}")
        if payload["result"]["seed"] != entry["seed"]:
            raise RuntimeError(f"seed mismatch: {model_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["dataset"] != "beta" or result["architecture"] != "harmonic_fold_v4_1":
            raise RuntimeError(f"dataset or architecture mismatch: {result_path}")
        if int(result["seed"]) != int(entry["seed"]):
            raise RuntimeError(f"result seed mismatch: {result_path}")
        expected_split = f"group-5-fold-{int(entry['fold'])}"
        if result["split"] != expected_split:
            raise RuntimeError(f"result fold mismatch: {result_path}")
        if list(map(str, result["subjects"]["test"])) != list(map(str, entry["test_subjects"])):
            raise RuntimeError(f"test-subject metadata mismatch: {result_path}")
        all_test_subjects[int(entry["seed"])].extend(map(str, entry["test_subjects"]))

    for seed, subjects in all_test_subjects.items():
        if len(subjects) != 70 or len(set(subjects)) != 70:
            raise RuntimeError(f"seed {seed} does not cover all 70 BETA participants exactly once")

    checksum_lines = (BUNDLE / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for line in checksum_lines:
        expected, relative = line.split("  ", maxsplit=1)
        if sha256(BUNDLE / relative).upper() != expected:
            raise RuntimeError(f"bundle checksum mismatch: {relative}")

    with torch.inference_mode():
        first = manifest["checkpoints"][0]
        model, _ = load_harmonic_fold_checkpoint(BUNDLE / first["model"], folded=True)
        logits = model(torch.zeros(1, 8, 250, dtype=torch.float32))
        if logits.shape != (1, 40) or not torch.isfinite(logits).all():
            raise RuntimeError(f"invalid smoke-test output: {tuple(logits.shape)}")
    print(json.dumps({"status": "ok", "checkpoints": 15, "smoke_shape": [1, 40]}))


if __name__ == "__main__":
    main()
