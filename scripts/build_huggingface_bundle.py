from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "huggingface"
SOURCE = (
    ROOT
    / "runs"
    / "paper_v15"
    / "adaptation_sources"
    / "beta"
    / "harmonic_fold_v4_1"
    / "full"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output = OUTPUT.resolve()
    root = ROOT.resolve()
    if output.parent != root or output.name != "huggingface":
        raise RuntimeError(f"unsafe output directory: {output}")
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    checkpoint_output = output / "checkpoints" / "beta"
    if checkpoint_output.exists():
        shutil.rmtree(checkpoint_output)
    checkpoint_output.mkdir(parents=True)
    for cache in output.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache)

    entries: list[dict[str, object]] = []
    models = sorted(SOURCE.glob("seed-*/fold-*/model.pt"))
    if len(models) != 15:
        raise RuntimeError(f"expected 15 BETA fold checkpoints, found {len(models)}")

    for source_model in models:
        seed = int(source_model.parents[1].name.removeprefix("seed-"))
        fold = int(source_model.parent.name.removeprefix("fold-"))
        source_result = source_model.with_name("result.json")
        if not source_result.exists():
            raise FileNotFoundError(source_result)
        result = json.loads(source_result.read_text(encoding="utf-8"))
        if result["architecture"] != "harmonic_fold_v4_1":
            raise RuntimeError(f"unexpected architecture in {source_result}")
        if result["model_parameters"] != 435139:
            raise RuntimeError(f"unexpected parameter count in {source_result}")

        payload = torch.load(source_model, map_location="cpu", weights_only=False)
        if payload["result"]["architecture"] != "harmonic_fold_v4_1":
            raise RuntimeError(f"checkpoint metadata mismatch in {source_model}")

        destination = checkpoint_output / f"seed-{seed}" / f"fold-{fold}"
        destination.mkdir(parents=True)
        model_path = destination / "model.pt"
        result_path = destination / "result.json"
        shutil.copy2(source_model, model_path)
        shutil.copy2(source_result, result_path)
        entries.append(
            {
                "dataset": "beta",
                "seed": seed,
                "fold": fold,
                "test_subjects": result["subjects"]["test"],
                "model": model_path.relative_to(output).as_posix(),
                "model_sha256": sha256(model_path),
                "result": result_path.relative_to(output).as_posix(),
                "result_sha256": sha256(result_path),
            }
        )

    shutil.copy2(ROOT / "configs" / "paper_multidataset.yaml", output / "config.yaml")
    shutil.copy2(ROOT / "MODEL_LICENSE.md", output / "LICENSE.md")
    shutil.copy2(
        ROOT / "docs" / "LICENSE_PROVENANCE_MATRIX.md",
        output / "LICENSE_PROVENANCE_MATRIX.md",
    )
    package_output = output / "harmonicfoldnet"
    if package_output.exists():
        shutil.rmtree(package_output)
    package_output.mkdir(parents=True)
    (package_output / "__init__.py").write_text(
        "from .harmonic_fold import HarmonicFoldNet\n"
        "from .model import reparameterize_model\n",
        encoding="utf-8",
    )
    shutil.copy2(ROOT / "src" / "harmonic_fold.py", package_output / "harmonic_fold.py")
    shutil.copy2(ROOT / "src" / "model.py", package_output / "model.py")
    harmonic_source = (package_output / "harmonic_fold.py").read_text(encoding="utf-8")
    (package_output / "harmonic_fold.py").write_text(
        harmonic_source.replace("from .model import", "from .model import"),
        encoding="utf-8",
    )
    shutil.copy2(
        ROOT / "paper" / "source_data" / "submission_evidence.json",
        output / "evaluation.json",
    )

    manifest = {
        "schema_version": 1,
        "release": "v0.2.0",
        "model": "HarmonicFoldNet",
        "architecture": "harmonic_fold_v4_1",
        "dataset": "BETA",
        "checkpoint_scope": (
            "Fifteen source-decoder checkpoints used by the three-seed, five-fold "
            "BETA evaluation and optional-adaptation analysis."
        ),
        "implementation": "harmonicfoldnet package included in this model snapshot",
        "weight_scope_limit": (
            "The bundle does not include Benchmark, ablation, Wearable, comparator, "
            "or participant-adapter checkpoints. Derived participant-level evidence "
            "for those analyses is released with the source repository."
        ),
        "checkpoints": entries,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    checksum_files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    checksum_text = "".join(
        f"{sha256(path).upper()}  {path.relative_to(output).as_posix()}\n"
        for path in checksum_files
    )
    (output / "SHA256SUMS.txt").write_text(checksum_text, encoding="utf-8")
    print(json.dumps({"status": "ok", "checkpoints": len(entries), "output": str(output)}))


if __name__ == "__main__":
    main()
