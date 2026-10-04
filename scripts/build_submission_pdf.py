from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz
import markdown

from scripts.build_submission_docx import (
    cite_order,
    format_reference,
    parse_bibliography,
    render_equation,
)


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUTPUT = PAPER / "submission"
CHROME_CANDIDATES = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)
FOOTER_FONT = Path(r"C:\Windows\Fonts\arial.ttf")


def ascii_hyphens(text: str) -> str:
    return text.translate(
        {
            ord("\u2010"): "-",
            ord("\u2011"): "-",
            ord("\u2012"): "-",
            ord("\u2013"): "-",
            ord("\u2014"): "-",
            ord("\u2212"): "-",
        }
    )


def prepare_markdown(source: Path, bibliography: Path | None) -> str:
    raw = source.read_text(encoding="utf-8")
    citation_keys: list[str] = []
    references: dict[str, dict[str, str]] = {}
    if bibliography is not None:
        raw, citation_keys = cite_order(raw)
        references = parse_bibliography(bibliography)
        raw = raw.split("\n## References\n", maxsplit=1)[0].rstrip()

    # Chromium does not rasterize a PDF used as an HTML image. Use the
    # publication PNGs generated from the same plotting source instead.
    raw = raw.replace("](figures/fig", "](figures/fig").replace(".pdf)", ".png)")

    equation_dir = OUTPUT / "qa_equations"
    lines = raw.splitlines()
    rendered: list[str] = []
    index = 0
    equation_index = 0
    while index < len(lines):
        if lines[index].strip() != r"\[":
            rendered.append(lines[index])
            index += 1
            continue
        equation: list[str] = []
        index += 1
        while index < len(lines) and lines[index].strip() != r"\]":
            equation.append(lines[index].strip())
            index += 1
        equation_index += 1
        equation_path = equation_dir / f"equation-{equation_index}.png"
        render_equation(" ".join(equation), equation_path)
        relative = equation_path.relative_to(PAPER).as_posix()
        rendered.extend(("", f'<div class="equation"><img src="{relative}" alt="Displayed equation {equation_index}"></div>', ""))
        index += 1

    if bibliography is not None:
        rendered.extend(("", "## References", "", '<ol class="references">'))
        for reference_index, key in enumerate(citation_keys, start=1):
            if key not in references:
                raise KeyError(f"missing bibliography entry: {key}")
            item = format_reference(reference_index, references[key])
            # The ordered-list marker supplies the visible reference number.
            prefix = f"{reference_index}. "
            if item.startswith(prefix):
                item = item[len(prefix) :]
            rendered.append(f"<li>{html.escape(item)}</li>")
        rendered.append("</ol>")

    return ascii_hyphens("\n".join(rendered))


def html_document(markdown_source: str, title: str, supplement: bool) -> str:
    body = markdown.markdown(
        markdown_source,
        extensions=("tables", "sane_lists", "attr_list"),
        output_format="html5",
    )
    body = re.sub(
        r"<p><strong>(Authors|ORCID|Affiliation|Affiliations|Corresponding author):</strong>",
        r'<p class="author-meta"><strong>\1:</strong>',
        body,
    )
    table_font = "6.8pt" if supplement else "7.5pt"
    base_uri = PAPER.resolve().as_uri() + "/"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<base href="{html.escape(base_uri)}">
<title>{html.escape(title)}</title>
<style>
  @page {{ size: A4; margin: 17mm 17mm 19mm 17mm; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    color: #111;
    font-family: "Times New Roman", "Noto Serif", serif;
    font-size: 10.2pt;
    line-height: 1.26;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
  h1, h2, h3 {{
    color: #000;
    font-family: Arial, "Noto Sans", sans-serif;
    page-break-after: avoid;
    break-after: avoid-page;
  }}
  h1 {{ font-size: 17.5pt; line-height: 1.12; margin: 0 0 10pt; text-align: center; }}
  h2 {{ font-size: 13.2pt; margin: 13pt 0 5pt; }}
  h3 {{ font-size: 11.2pt; margin: 9pt 0 4pt; }}
  p {{ margin: 0 0 5pt; orphans: 3; widows: 3; text-align: justify; }}
  .author-meta {{ text-align: center; }}
  strong {{ font-weight: 700; }}
  code {{ font: 8.6pt Consolas, monospace; overflow-wrap: anywhere; }}
  img {{ display: block; max-width: 100%; max-height: 225mm; margin: 7pt auto 4pt; }}
  .equation {{ text-align: center; break-inside: avoid; page-break-inside: avoid; }}
  .equation img {{ max-width: 92%; max-height: 18mm; margin: 5pt auto; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 6pt 0 9pt;
    font-size: {table_font};
    line-height: 1.14;
    page-break-inside: auto;
  }}
  thead {{ display: table-header-group; }}
  tr {{ break-inside: avoid; page-break-inside: avoid; }}
  th, td {{ border: 0.4pt solid #b7bec6; padding: 2.4pt 3pt; vertical-align: top; }}
  th {{ background: #263849; color: white; font-family: Arial, sans-serif; font-weight: 700; }}
  tbody tr:nth-child(even) td {{ background: #f3f6f8; }}
  ul, ol {{ margin: 3pt 0 6pt 18pt; padding: 0; }}
  li {{ margin: 0 0 2pt; }}
  .references {{ font-size: 8.1pt; line-height: 1.12; margin-left: 20pt; }}
  .references li {{ padding-left: 2pt; margin-bottom: 2pt; text-align: left; }}
  a {{ color: #111; text-decoration: none; overflow-wrap: anywhere; }}
</style>
</head>
<body>{body}</body>
</html>"""


def chrome_binary() -> Path:
    for candidate in CHROME_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Chrome or Edge was not found")


def add_page_numbers(raw_pdf: Path, destination: Path, label: str) -> int:
    if not FOOTER_FONT.exists():
        raise FileNotFoundError(f"footer font was not found: {FOOTER_FONT}")
    document = fitz.open(raw_pdf)
    for page_index, page in enumerate(document, start=1):
        page.insert_font(fontname="footerfont", fontfile=str(FOOTER_FONT))
        footer = fitz.Rect(34, page.rect.height - 23, page.rect.width - 34, page.rect.height - 8)
        page.insert_textbox(
            footer,
            f"{label}  |  {page_index}",
            fontsize=7,
            fontname="footerfont",
            color=(0.35, 0.35, 0.35),
            align=fitz.TEXT_ALIGN_CENTER,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination, garbage=4, deflate=True)
    pages = document.page_count
    document.close()
    return pages


def build_pdf(source: Path, destination: Path, title: str, bibliography: Path | None = None) -> dict[str, object]:
    supplement = bibliography is None
    prepared = prepare_markdown(source, bibliography)
    html_text = html_document(prepared, title=title, supplement=supplement)
    html_path = destination.with_suffix(".html")
    html_path.write_text(html_text, encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="hfn-pdf-", dir=OUTPUT) as temp_dir:
        temp = Path(temp_dir)
        raw_pdf = temp / "raw.pdf"
        profile = temp / "chrome-profile"
        command = [
            str(chrome_binary()),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--allow-file-access-from-files",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={raw_pdf}",
            html_path.resolve().as_uri(),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if result.returncode != 0 or not raw_pdf.exists():
            raise RuntimeError(
                f"Chromium PDF export failed ({result.returncode}): "
                f"{result.stdout}\n{result.stderr}"
            )
        pages = add_page_numbers(
            raw_pdf,
            destination,
            "HarmonicFoldNet supplement" if supplement else "HarmonicFoldNet manuscript proof",
        )
    return {"path": str(destination), "pages": pages, "bytes": destination.stat().st_size}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manuscript = build_pdf(
        PAPER / "MANUSCRIPT_DRAFT_v1.md",
        OUTPUT / "HarmonicFoldNet_manuscript_proof.pdf",
        "HarmonicFoldNet manuscript proof",
        PAPER / "references.bib",
    )
    supplement = build_pdf(
        PAPER / "SUPPLEMENTARY_INFORMATION.md",
        OUTPUT / "HarmonicFoldNet_supplement_proof.pdf",
        "HarmonicFoldNet supplementary information",
    )
    print({"manuscript": manuscript, "supplement": supplement})


if __name__ == "__main__":
    main()
