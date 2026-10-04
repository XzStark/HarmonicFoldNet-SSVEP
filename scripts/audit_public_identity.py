"""Fail when submission-only identity leaks into public release surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path


TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".html",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "data",
    "paper",
    "runs",
    "tmp",
}


def load_terms(path: Path) -> list[str]:
    terms = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        term = raw.strip()
        if term and not term.startswith("#"):
            terms.append(term.casefold())
    if not terms:
        raise ValueError(f"No privacy terms found in {path}")
    return terms


def public_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.name in {"LICENSE", "NOTICE"} or path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--terms-file", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    terms = load_terms(args.terms_file.resolve())
    hits: list[tuple[Path, int, str]] = []
    for path in public_text_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            folded = line.casefold()
            if any(term in folded for term in terms):
                hits.append((path.relative_to(root), line_number, line.strip()))

    if hits:
        for path, line_number, line in hits:
            print(f"{path}:{line_number}: {line}")
        return 1
    print("Public identity audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
