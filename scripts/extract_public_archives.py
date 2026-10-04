from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import py7zr


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def _extract_to_temporary(archive: Path, temporary: Path) -> None:
    suffixes = "".join(archive.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive) as source:
            names = source.namelist()
            if not all(_safe_member(name) for name in names):
                raise RuntimeError(f"unsafe path in {archive}")
            source.extractall(temporary)
    elif suffixes.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            if not all(_safe_member(member.name) for member in members):
                raise RuntimeError(f"unsafe path in {archive}")
            source.extractall(temporary, filter="data")
    elif suffixes.endswith(".7z"):
        with py7zr.SevenZipFile(archive, "r") as source:
            names = source.getnames()
            if not all(_safe_member(name) for name in names):
                raise RuntimeError(f"unsafe path in {archive}")
            source.extractall(temporary)
    else:
        raise ValueError(f"unsupported archive: {archive}")


def extract_archives(source_root: Path, raw_root: Path) -> list[dict[str, object]]:
    archives = sorted(
        path for path in source_root.iterdir()
        if path.is_file() and "".join(path.suffixes).lower().endswith((".zip", ".7z", ".tar.gz"))
    )
    raw_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, archive in enumerate(archives, 1):
        with tempfile.TemporaryDirectory(prefix="extract-", dir=raw_root.parent) as temp_name:
            temporary = Path(temp_name)
            _extract_to_temporary(archive, temporary)
            files = sorted(path for path in temporary.rglob("*") if path.is_file())
            if not files:
                raise RuntimeError(f"no files extracted from {archive}")
            for source in files:
                relative = source.relative_to(temporary)
                destination = raw_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if destination.stat().st_size != source.stat().st_size:
                        raise RuntimeError(f"existing file size mismatch: {destination}")
                    continue
                shutil.move(str(source), str(destination))
        row = {
            "index": index,
            "total": len(archives),
            "archive": str(archive),
            "files": len(files),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("raw_root", type=Path)
    args = parser.parse_args()
    extract_archives(args.source_root, args.raw_root)


if __name__ == "__main__":
    main()
