from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from scripts import build_submission_evidence as evidence


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    CODE_ROOT / "paper" / "release_bundle" / "HarmonicFoldNet_evidence_inputs.zip"
)
DEFAULT_OUTPUT = CODE_ROOT / "paper" / "source_data"
DEFAULT_REPORT = CODE_ROOT / "paper" / "SUBMISSION_EVIDENCE_REPORT.md"
MANIFEST_NAME = "REPRODUCTION_INPUT_MANIFEST.json"


def _safe_relative_path(value: str) -> Path:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe path in evidence bundle: {value}")
    return Path(*candidate.parts)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def extract_verified_bundle(archive_path: Path, destination: Path) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        try:
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        except KeyError as error:
            raise ValueError(f"evidence bundle is missing {MANIFEST_NAME}") from error

        listed = manifest.get("files")
        if not isinstance(listed, list) or not listed:
            raise ValueError("evidence bundle manifest has no files")

        expected_names: set[str] = set()
        for item in listed:
            relative = _safe_relative_path(str(item["path"]))
            archive_name = relative.as_posix()
            expected_names.add(archive_name)
            try:
                payload = archive.read(archive_name)
            except KeyError as error:
                raise ValueError(f"manifested input is missing: {archive_name}") from error
            if len(payload) != int(item["bytes"]):
                raise ValueError(f"size mismatch for evidence input: {archive_name}")
            if _sha256_bytes(payload) != str(item["sha256"]):
                raise ValueError(f"SHA-256 mismatch for evidence input: {archive_name}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

        payload_names = {
            name for name in archive.namelist() if name != MANIFEST_NAME and not name.endswith("/")
        }
        if payload_names != expected_names:
            extras = sorted(payload_names - expected_names)
            missing = sorted(expected_names - payload_names)
            raise ValueError(
                f"evidence bundle manifest mismatch: extras={extras}, missing={missing}"
            )
    return manifest


def rebuild(
    archive_path: Path,
    output_dir: Path,
    report_path: Path,
) -> dict:
    archive_path = archive_path.resolve()
    output_dir = output_dir.resolve()
    report_path = report_path.resolve()
    with tempfile.TemporaryDirectory(prefix="harmonicfoldnet-evidence-") as temporary:
        input_root = Path(temporary)
        extract_verified_bundle(archive_path, input_root)
        config_path = input_root / "paper" / "SUBMISSION_EVIDENCE_CONFIG.json"
        original_root = evidence.ROOT
        try:
            evidence.ROOT = input_root
            return evidence.run(
                config_path,
                output_dir,
                report_path,
                update_figure_data=False,
            )
        finally:
            evidence.ROOT = original_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the released evidence bundle and rebuild submission tables."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    bundle = rebuild(args.archive, args.output_dir, args.report)
    print(
        json.dumps(
            {
                "status": "ok",
                "archive": str(args.archive.resolve()),
                "main_datasets": sorted(bundle["main"]["audit"]["datasets"]),
                "split_manifest_rows": bundle["split_manifest"]["rows"],
                "deployment_measurements": bundle["deployment"]["measurements"],
                "output": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
