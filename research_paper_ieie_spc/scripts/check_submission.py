#!/usr/bin/env python3
"""Fail on mechanical IEIE submission defects in the generated manuscript."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parents[1]
PDF = PAPER_DIR / "main.pdf"
TEX = PAPER_DIR / "main.tex"


def command(*args: str) -> str:
    return subprocess.run(
        args, cwd=PAPER_DIR, check=True, text=True, capture_output=True
    ).stdout


def main() -> None:
    errors: list[str] = []
    if not PDF.exists():
        raise SystemExit("SUBMISSION CHECK FAILED\n- main.pdf is missing")

    tex = TEX.read_text(encoding="utf-8")
    info = command("pdfinfo", str(PDF))
    text = command("pdftotext", str(PDF), "-")
    fonts = command("pdffonts", str(PDF))

    page_match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    pages = int(page_match.group(1)) if page_match else -1
    if not 9 <= pages <= 14:
        errors.append(f"page count must be 9-14, found {pages}")

    size_match = re.search(
        r"^Page size:\s+([0-9.]+) x ([0-9.]+) pts \(A4\)$", info, re.MULTILINE
    )
    if not size_match:
        errors.append("PDF page size is not identified as A4")
    else:
        width, height = map(float, size_match.groups())
        if abs(width - 595.28) > 1 or abs(height - 841.89) > 1:
            errors.append(f"unexpected A4 dimensions: {width} x {height} pt")

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL
    )
    if not abstract_match:
        errors.append("abstract environment is missing")
        abstract_words = -1
    else:
        clean_abstract = re.sub(r"\\[A-Za-z]+|[{}$]", " ", abstract_match.group(1))
        abstract_words = len(re.findall(r"\b[\w.-]+\b", clean_abstract))
        if not 100 <= abstract_words <= 200:
            errors.append(f"abstract must be 100-200 words, found {abstract_words}")

    keyword_match = re.search(
        r"\\begin\{keywords\}(.*?)\\end\{keywords\}", tex, re.DOTALL
    )
    keywords = (
        [value.strip() for value in keyword_match.group(1).split(",")]
        if keyword_match
        else []
    )
    if not 5 <= len(keywords) <= 6:
        errors.append(f"keywords must number 5-6, found {len(keywords)}")

    forbidden = (
        "10.5573/JSTS",
        "VOL. VV",
        "First Author",
        "Second Author",
        "example.edu",
        "[Insert",
        "biography to be completed",
        "Copyrights ©2024",
        "Received: –",
        "Accepted: –",
        "Published: –",
        r"\journalmonth{0}",
        "Korea Big Data Society Journal",
    )
    for marker in forbidden:
        if marker in text or marker in tex:
            errors.append(f"forbidden placeholder/legacy marker remains: {marker}")
    unsupported_provenance = (
        "550 completed runs",
        "13 independent task blocks",
        "intention-to-treat",
        "defensible causal contrast",
        "reproducible manuscript generator",
        "pre-specified 0.98",
    )
    for marker in unsupported_provenance:
        if marker in text or marker in tex:
            errors.append(f"unsupported provenance wording remains: {marker}")
    required_provenance = (
        "configured for 550 runs",
        "aggregate diagnostic rather than direct proof",
        "exact executed snapshot was not preserved",
        "Compact metric-level rows for those runs are released",
        "aggregate-implied total",
        "ieie-spc-v1.0.0",
        "The Korea Journal of BigData",
    )
    for marker in required_provenance:
        if marker not in text and marker not in tex:
            errors.append(f"required provenance caveat is missing: {marker}")
    if "Performance, Efficiency, and Collapse-Aware Stability" not in text:
        errors.append("expected integrated manuscript title is missing")

    if "TimesNewRoman" not in fonts or "Arial" not in fonts:
        errors.append("Times New Roman and Arial are not both embedded")
    font_flag = re.compile(r"\s(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+$")
    for line in fonts.splitlines()[2:]:
        match = font_flag.search(line)
        if match and match.group(1) != "yes":
            errors.append(f"font is not embedded: {line.split()[0]}")

    log_path = PAPER_DIR / "main.log"
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if "undefined references" in log or "Citation `" in log and "undefined" in log:
            errors.append("unresolved citation/reference warning remains")
        if "Overfull \\hbox" in log:
            errors.append("overfull horizontal box remains")

    if errors:
        print("SUBMISSION CHECK FAILED")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    print("SUBMISSION CHECK PASS")
    print(
        f"pages={pages}, paper=A4, abstract_words={abstract_words}, "
        f"keywords={len(keywords)}, fonts=embedded"
    )


if __name__ == "__main__":
    main()
