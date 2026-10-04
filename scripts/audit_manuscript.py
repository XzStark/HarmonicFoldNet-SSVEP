from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


CITATION_RE = re.compile(r"@([A-Za-z0-9_:\-]+)")
BIB_KEY_RE = re.compile(r"^@\w+\{\s*([^,]+),", re.MULTILINE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)*\b")
DEFINITION_RE = re.compile(
    r"(?P<long>[A-Za-z][A-Za-z0-9–—\-/ ]{2,80}?)\s*\((?P<short>[A-Z][A-Z0-9-]{1,9})\)"
)


@dataclass(frozen=True)
class Finding:
    level: str
    code: str
    line: int
    message: str


def strip_nonprose(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[@[^\]]+\]", "", text)
    text = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
    text = re.sub(r"https?://\S+", "", text)
    return text


def first_line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def audit_acronyms(text: str) -> list[Finding]:
    prose = strip_nonprose(text)
    definitions: dict[str, int] = {}
    for match in DEFINITION_RE.finditer(prose):
        definitions.setdefault(match.group("short"), match.start("short"))

    # Universally understood units, hardware abbreviations and reporting labels.
    allowed = {
        "AI",
        "CPU",
        "GPU",
        "Hz",
        "MiB",
        "PDF",
        "DOI",
        "ORCID",
        "CI",
        "CRediT",
        "BETA",
        "NVIDIA",
        "RTX",
        "Pz",
        "PO7",
        "PO5",
        "PO4",
        "PO3",
        "POz",
        "PO8",
        "PO6",
        "O1",
        "Oz",
        "O2",
    }
    findings: list[Finding] = []
    seen: set[str] = set()
    for match in ACRONYM_RE.finditer(prose):
        short = match.group(0)
        if short in allowed or short in seen:
            continue
        seen.add(short)
        line = first_line(prose, match.start())
        # Acronyms in the title may be defined in the abstract; do not flag line 1.
        if line == 1:
            continue
        definition_at = definitions.get(short)
        if definition_at is None:
            findings.append(
                Finding("warning", "undefined-acronym", line, f"{short} has no long-form definition")
            )
        elif match.start() < definition_at:
            findings.append(
                Finding(
                    "warning",
                    "acronym-before-definition",
                    line,
                    f"{short} appears before its long-form definition",
                )
            )
    return findings


def audit_citations(text: str, bib_text: str) -> list[Finding]:
    # Pandoc citations and email addresses both contain ``@``. Remove emails
    # before collecting citation keys so a corresponding-author address is not
    # mistaken for a missing bibliography entry.
    citation_text = EMAIL_RE.sub("", text)
    cited = set(CITATION_RE.findall(citation_text))
    available = set(BIB_KEY_RE.findall(bib_text))
    findings: list[Finding] = []
    for key in sorted(cited - available):
        offset = text.find(f"@{key}")
        findings.append(
            Finding("error", "missing-bibliography-key", first_line(text, offset), key)
        )
    for key in sorted(available - cited):
        findings.append(Finding("info", "uncited-bibliography-entry", 0, key))
    return findings


def audit_text(text: str) -> list[Finding]:
    findings: list[Finding] = []
    prohibited = {
        "fast" + "vit": "legacy architecture-family name",
        "fast" + "vlm": "unrelated legacy model name",
        "fast" + "ssvep": "legacy project name",
    }
    prose = strip_nonprose(text)
    lowered = prose.lower()
    for token, description in prohibited.items():
        start = 0
        while True:
            offset = lowered.find(token, start)
            if offset < 0:
                break
            findings.append(
                Finding("error", "prohibited-name", first_line(prose, offset), f"{description}: {token}")
            )
            start = offset + len(token)

    placeholder_patterns = {
        "author-placeholder": r"\[to be completed\]|\[confirm before submission\]",
        "draft-marker": r"\b(?:TODO|TBD|FIXME)\b",
    }
    for code, pattern in placeholder_patterns.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            findings.append(
                Finding("warning", code, first_line(text, match.start()), match.group(0))
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit manuscript citations, acronyms, and submission markers.")
    parser.add_argument("--manuscript", type=Path, default=Path("paper/MANUSCRIPT_DRAFT_v1.md"))
    parser.add_argument("--bibliography", type=Path, default=Path("paper/references.bib"))
    parser.add_argument("--output", type=Path, default=Path("paper/MANUSCRIPT_AUDIT.json"))
    args = parser.parse_args()

    manuscript = args.manuscript.read_text(encoding="utf-8")
    bibliography = args.bibliography.read_text(encoding="utf-8")
    findings = audit_citations(manuscript, bibliography)
    findings.extend(audit_acronyms(manuscript))
    findings.extend(audit_text(manuscript))
    findings.sort(key=lambda item: ({"error": 0, "warning": 1, "info": 2}[item.level], item.line, item.code))

    payload = {
        "manuscript": args.manuscript.as_posix(),
        "bibliography": args.bibliography.as_posix(),
        "counts": {
            level: sum(item.level == level for item in findings)
            for level in ("error", "warning", "info")
        },
        "findings": [asdict(item) for item in findings],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=False))
    return 1 if payload["counts"]["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
