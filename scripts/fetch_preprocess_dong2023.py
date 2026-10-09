from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.public_ssvep_data import preprocess_dong2023


ZENODO_RECORD_ID = 18847318
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def _download_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "HarmonicFoldNet/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - publisher checksum verification.
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, checksum: str, size: int) -> bool:
    expected = checksum.removeprefix("md5:").lower()
    return (
        path.exists()
        and path.stat().st_size == int(size)
        and _md5(path).lower() == expected
    )


def _download(url: str, destination: Path, checksum: str, size: int) -> None:
    if _verified(destination, checksum, size):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    for attempt in range(1, 9):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"User-Agent": "HarmonicFoldNet/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                status = int(getattr(response, "status", response.getcode()))
                resumable = offset > 0 and status == 206
                mode = "ab" if resumable else "wb"
                if offset and not resumable:
                    offset = 0
                with temporary.open(mode) as out:
                    copied = offset
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        copied += len(chunk)
                    out.flush()
                    os.fsync(out.fileno())
            if temporary.stat().st_size != int(size):
                raise RuntimeError(f"size mismatch for {destination.name}")
            if _md5(temporary).lower() != checksum.removeprefix("md5:").lower():
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"checksum mismatch for {destination.name}")
            os.replace(temporary, destination)
            return
        except Exception as error:
            if attempt >= 8:
                raise
            print(
                json.dumps(
                    {
                        "event": "download_retry",
                        "file": destination.name,
                        "attempt": attempt,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(min(30.0, 2.0 ** (attempt - 1)))


def _subjects(values: list[str]) -> list[int]:
    selected: set[int] = set()
    for value in values:
        begin, separator, end = value.partition("-")
        if separator:
            selected.update(range(int(begin), int(end) + 1))
        else:
            selected.add(int(value))
    if not selected or min(selected) < 1 or max(selected) > 59:
        raise ValueError("subjects must lie in 1..59")
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stream Dong2023 one participant at a time, verify the publisher "
            "checksum, retain the registered processed shard, and remove raw data."
        )
    )
    parser.add_argument("--subjects", nargs="+", default=["1-59"])
    parser.add_argument(
        "--raw-cache-dir", type=Path, default=Path("data/external/dong2023_stage")
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/Dong2023")
    )
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument(
        "--download-workers",
        type=int,
        default=1,
        help=(
            "Concurrent download workers. Values above one download the selected "
            "raw shards first, then preprocess them serially."
        ),
    )
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = _subjects(args.subjects)
    metadata_path = args.raw_cache_dir / "zenodo_record.json"
    if args.preprocess_only:
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        record = _download_json(ZENODO_API)
        _atomic_json(metadata_path, record)
    files = {entry["key"]: entry for entry in record["files"]}

    if args.download_workers < 1:
        raise ValueError("download-workers must be at least one")

    if not args.preprocess_only and args.download_workers > 1:
        pending: list[tuple[int, Path, dict]] = []
        required_bytes = 0
        for subject in selected:
            output = args.processed_dir / f"sub-{subject}.npz"
            if output.exists() and not args.force:
                continue
            name = f"S{subject}.mat"
            entry = files.get(name)
            if entry is None:
                raise RuntimeError(f"Zenodo record does not contain {name}")
            raw_path = args.raw_cache_dir / name
            if not _verified(raw_path, str(entry["checksum"]), int(entry["size"])):
                required_bytes += max(
                    0,
                    int(entry["size"])
                    - (raw_path.with_name(f".{raw_path.name}.part").stat().st_size
                       if raw_path.with_name(f".{raw_path.name}.part").exists()
                       else 0),
                )
            pending.append((subject, raw_path, entry))
        free = shutil.disk_usage(args.raw_cache_dir.resolve().anchor).free
        reserve = int(float(args.min_free_gb) * 1024**3)
        if free - required_bytes < reserve:
            raise RuntimeError(
                "disk reserve would be crossed by concurrent staging: "
                f"free={free}, required={required_bytes}, reserve={reserve}"
            )
        with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            futures = {
                executor.submit(
                    _download,
                    str(entry["links"]["self"]),
                    raw_path,
                    str(entry["checksum"]),
                    int(entry["size"]),
                ): subject
                for subject, raw_path, entry in pending
            }
            for future in as_completed(futures):
                subject = futures[future]
                future.result()
                print(
                    json.dumps(
                        {"subject": subject, "status": "downloaded"},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    for subject in selected:
        output = args.processed_dir / f"sub-{subject}.npz"
        if output.exists() and not args.force:
            print(json.dumps({"subject": subject, "status": "existing"}), flush=True)
            continue
        name = f"S{subject}.mat"
        entry = files.get(name)
        if entry is None:
            raise RuntimeError(f"Zenodo record does not contain {name}")
        free = shutil.disk_usage(args.raw_cache_dir.resolve().anchor).free
        reserve = int(float(args.min_free_gb) * 1024**3)
        if free - int(entry["size"]) < reserve:
            raise RuntimeError(
                f"disk reserve would be crossed before {name}: "
                f"free={free}, file={entry['size']}, reserve={reserve}"
            )
        raw_path = args.raw_cache_dir / name
        if args.preprocess_only:
            if not _verified(raw_path, str(entry["checksum"]), int(entry["size"])):
                raise RuntimeError(f"offline file verification failed: {raw_path}")
        elif args.download_workers == 1:
            _download(
                str(entry["links"]["self"]),
                raw_path,
                str(entry["checksum"]),
                int(entry["size"]),
            )
        elif not _verified(raw_path, str(entry["checksum"]), int(entry["size"])):
            raise RuntimeError(f"concurrent download verification failed: {raw_path}")
        preprocess_dong2023(
            args.raw_cache_dir,
            args.processed_dir,
            force=args.force,
            subjects=[subject],
        )
        if not args.keep_raw:
            raw_path.unlink(missing_ok=True)
        print(json.dumps({"subject": subject, "status": "processed"}), flush=True)


if __name__ == "__main__":
    main()
