from __future__ import annotations

import glob
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper" / "SUBMISSION_EVIDENCE_CONFIG.json"
OUTPUT_DIR = ROOT / "paper" / "release_bundle"
OUTPUT = OUTPUT_DIR / "HarmonicFoldNet_evidence_inputs.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    paths: set[Path] = {CONFIG}
    for candidate in strings(config):
        if not any(token in candidate for token in ("/", "\\", "*")):
            continue
        for match in glob.glob(str(ROOT / candidate)):
            path = Path(match)
            if path.is_file():
                paths.add(path)

    # MTSNet participant records are discovered by the evidence builder from
    # each registered comparison file's directory rather than listed as a
    # separate config pattern. Include those derived inputs explicitly so a
    # clean public checkout can rebuild the comparison instead of relying on a
    # private local runs directory.
    for relative in config.get("mtsnet", {}).get("comparison_files", []):
        comparison = ROOT / relative
        pattern = comparison.parent / "seed-*" / "fold-*" / "result.json"
        for match in glob.glob(str(pattern)):
            path = Path(match)
            if path.is_file():
                paths.add(path)

    # The statistical evidence builder does not recompute folding equivalence,
    # so its precomputed audit remains a separate released source-data file.
    relative_paths = sorted(path.relative_to(ROOT) for path in paths)
    manifest = {
        "schema_version": 1,
        "purpose": (
            "Inputs required by scripts.build_submission_evidence, preserving "
            "their repository-relative paths. Extract at repository root."
        ),
        "files": [
            {
                "path": path.as_posix(),
                "bytes": (ROOT / path).stat().st_size,
                "sha256": sha256(ROOT / path),
            }
            for path in relative_paths
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in relative_paths:
            archive.write(ROOT / path, path.as_posix())
        archive.writestr("REPRODUCTION_INPUT_MANIFEST.json", manifest_bytes)
    print(json.dumps({
        "status": "ok",
        "files": len(relative_paths),
        "bytes": OUTPUT.stat().st_size,
        "sha256": sha256(OUTPUT),
        "output": str(OUTPUT),
    }))


if __name__ == "__main__":
    main()
