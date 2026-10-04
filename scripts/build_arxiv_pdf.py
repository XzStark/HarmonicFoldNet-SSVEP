from __future__ import annotations

from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "paper" / "submission"
MANUSCRIPT = SUBMISSION / "HarmonicFoldNet_manuscript_proof.pdf"
SUPPLEMENT = SUBMISSION / "HarmonicFoldNet_supplement_proof.pdf"
OUTPUT = SUBMISSION / "HarmonicFoldNet_arXiv_preprint.pdf"


def main() -> None:
    output = fitz.open()
    manuscript = fitz.open(MANUSCRIPT)
    supplement = fitz.open(SUPPLEMENT)
    output.insert_pdf(manuscript)
    output.insert_pdf(supplement)
    output.set_metadata(
        {
            "title": "HarmonicFoldNet: A Compact Foldable Local-to-Global Decoder for Cross-Subject SSVEP Recognition",
            "author": "HarmonicFoldNet authors",
            "subject": "Preprint with supplementary information",
            "keywords": "SSVEP, BCI, cross-subject decoding, harmonic attention",
            "creator": "HarmonicFoldNet reproducible submission pipeline",
        }
    )
    output.save(OUTPUT, garbage=4, deflate=True)
    pages = output.page_count
    output.close()
    manuscript.close()
    supplement.close()
    print({"status": "ok", "path": str(OUTPUT), "pages": pages, "bytes": OUTPUT.stat().st_size})


if __name__ == "__main__":
    main()
