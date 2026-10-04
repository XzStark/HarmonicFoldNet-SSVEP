from __future__ import annotations

import argparse
import json
import tarfile
import time
import zipfile
from pathlib import Path

import py7zr


def verify(path: Path) -> dict[str, object]:
    started = time.perf_counter()
    suffixes = "".join(path.suffixes).lower()
    error: str | None = None
    members = 0
    try:
        if suffixes.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                members = len(archive.infolist())
                bad = archive.testzip()
                if bad is not None:
                    error = f"CRC failure: {bad}"
        elif suffixes.endswith(".tar.gz"):
            with tarfile.open(path, "r:gz") as archive:
                for member in archive:
                    members += 1
                    if member.isfile():
                        extracted = archive.extractfile(member)
                        if extracted is not None:
                            while extracted.read(1024 * 1024):
                                pass
        elif suffixes.endswith(".7z"):
            with py7zr.SevenZipFile(path, "r") as archive:
                members = len(archive.getnames())
                bad = archive.test()
                if bad is not None:
                    error = str(bad)
        else:
            raise ValueError(f"unsupported archive: {path}")
    except Exception as exc:  # retained in the machine-readable report
        error = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "members": members,
        "ok": error is None,
        "error": error,
        "seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    archives = sorted(
        path
        for root in args.roots
        for path in root.rglob("*")
        if path.is_file() and "".join(path.suffixes).lower().endswith((".zip", ".7z", ".tar.gz"))
    )
    results = []
    for index, path in enumerate(archives, 1):
        row = verify(path)
        results.append(row)
        print(json.dumps({"index": index, "total": len(archives), **row}, ensure_ascii=False), flush=True)
    report = {
        "archives": len(results),
        "valid": sum(bool(row["ok"]) for row in results),
        "invalid": sum(not bool(row["ok"]) for row in results),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.report)
    if report["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
