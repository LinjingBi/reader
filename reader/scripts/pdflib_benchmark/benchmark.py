from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd


# INTRO_HEADING_RE = re.compile(r"^\s*(?:\d+\.?\s*)?introduction\s*$", re.IGNORECASE)
INTRO_HEADING_RE = re.compile(
    r"""^\s*
        (?:
            (?:
                (?P<num>\d{1,3})      # 1, 2, 10...
                \s*[\.\)]?\s*
            )
            |
            (?:
                (?P<roman>M{0,4}(CM|CD|D?C{0,3})
                (XC|XL|L?X{0,3})
                (IX|IV|V?I{0,3}))     # I, II, IV, V, IX, X, ...
                \s*[\.\)]?\s*
            )
        )?
        introduction
        \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Heuristic for "next section" boundary (optional; safe + conservative)
# NEXT_SECTION_RE = re.compile(
#     r"^\s*(?:\d+\.?\s*)?(related work|background|preliminaries|method|methods|approach|model|experiments?|evaluation|results|discussion|conclusion)\s*$",
#     re.IGNORECASE,
# )
ROMAN_OR_NUM_PREFIX = r"""
(?:                                   # optional section numbering prefix
  (?:
    \d{1,3}(?:\.\d{1,3})*              # 2, 2.1, 3.2.1
    |
    M{0,4}(?:CM|CD|D?C{0,3})           # roman numeral
    (?:XC|XL|L?X{0,3})
    (?:IX|IV|V?I{0,3})
  )
  \s*[\.\)]?\s*
)?
"""


NEXT_SECTION_RE = re.compile(
    rf"""^\s*{ROMAN_OR_NUM_PREFIX}
        (?:
            related\s+work
          | background
          | preliminaries
          | problem\s+setup
          | method(?:s|ology)?
          | approach
          | model(?:s)?
          | framework
          | experiments?
          | evaluation
          | results?
          | discussion
          | conclusion
          | limitations?
          | acknowledg(e)?ments?
          | references
          | appendix
        )
        (?:\s*[:\-–—].*)?\s*$          # allow "Method: ..." or "Method - ..."
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class RunResult:
    pdf: str
    extractor: str
    pages: int
    max_chars: int
    seconds: float
    intro_found: bool
    intro_start_line: Optional[int]
    intro_chars: str
    intro_chars_num: int
    error: Optional[str]


@dataclass
class LineRec:
    text: str
    page: int
    y0: float
    size: float
    boldish: bool


def _is_boldish(span: dict) -> bool:
    # PyMuPDF provides font name and flags. Font name usually contains "Bold" when bold.
    font = span.get("font", "") or ""
    if "bold" in font.lower():
        return True
    # flags is bitmask; bold isn't always reliable, but keep as fallback
    # We'll keep it weak and mostly rely on font size + heading-shape.
    return False


def _extract_lines_pymupdf(doc, max_pages: int) -> List[LineRec]:
    lines: List[LineRec] = []
    n = min(max_pages, doc.page_count)

    for p in range(n):
        page = doc.load_page(p)
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type", 0) != 0:
                continue  # skip non-text blocks
            for ln in block.get("lines", []):
                spans = ln.get("spans", [])
                if not spans:
                    continue
                # Reconstruct line text
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue

                # Representative style for the line
                size = max(float(s.get("size", 0.0)) for s in spans)
                boldish = any(_is_boldish(s) for s in spans)

                bbox = ln.get("bbox", None)
                y0 = float(bbox[1]) if bbox else 0.0

                lines.append(LineRec(text=text, page=p, y0=y0, size=size, boldish=boldish))

    return lines


def _heading_like(text: str) -> bool:
    # Conservative "heading shape"
    t = text.strip()
    if len(t) < 3 or len(t) > 90:
        return False
    if t.endswith("."):
        return False

    # Allow prefixes like "2", "2.1", "II", "II."
    t2 = re.sub(r"^\s*(?:\d{1,3}(?:\.\d{1,3})*|[IVXLCDM]+)\s*[\.\)]?\s*", "", t, flags=re.I)

    # If what's left is basically words
    words = t2.split()
    if not (1 <= len(words) <= 12):
        return False

    alpha = sum(ch.isalpha() for ch in t2)
    if alpha / max(1, len(t2)) < 0.55:
        return False

    return True


def _worker(args):
    pdf_path_str, extractor_name, pages, max_chars = args
    return benchmark_one(Path(pdf_path_str), extractor_name, pages, max_chars)


def find_intro_slice(text: str, max_chars: int) -> Tuple[bool, Optional[int], str]:
    """
    Strict policy:
    - Only return intro slice if an 'Introduction' heading line is found.
    - Slice from heading line until next heading (if found) or char cap.
    """
    # Keep line breaks: best for heading detection
    lines = text.splitlines()
    intro_idx = None
    for i, line in enumerate(lines):
        if INTRO_HEADING_RE.match(line.strip()):
            intro_idx = i
            break

    if intro_idx is None:
        return False, None, ""

    # Slice from intro heading onward
    slice_lines: List[str] = []
    slice_lines.append(lines[intro_idx])

    for j in range(intro_idx + 1, len(lines)):
        line = lines[j]
        # Stop at next major section heading (conservative)
        if NEXT_SECTION_RE.match(line.strip()):
            break
        slice_lines.append(line)
        if sum(len(x) + 1 for x in slice_lines) >= max_chars:
            break

    slice_text = "\n".join(slice_lines)
    if len(slice_text) > max_chars:
        slice_text = slice_text[:max_chars]

    return True, intro_idx, slice_text


def extract_pymupdf(pdf_path: Path, pages: int) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    n = min(pages, doc.page_count)
    chunks = []
    for i in range(n):
        page = doc.load_page(i)
        chunks.append(page.get_text("text"))  # preserves line breaks reasonably
    doc.close()
    return "\n".join(chunks)


def extract_pdfplumber(pdf_path: Path, pages: int) -> str:
    import pdfplumber

    chunks = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        n = min(pages, len(pdf.pages))
        for i in range(n):
            # layout=True helps preserve columns/line breaks (often improves heading find)
            chunks.append(pdf.pages[i].extract_text(layout=True) or "")
    return "\n".join(chunks)


def extract_pymupdf4llm(pdf_path: Path, pages: int) -> str:
    """
    pymupdf4llm converts to markdown-ish text.
    We restrict to first N pages by reading only those pages via PyMuPDF and feeding them.
    """
    import fitz
    import pymupdf4llm

    doc = fitz.open(pdf_path)
    n = min(pages, doc.page_count)

    # Build a temporary document with first N pages (cheap: new doc + insert)
    tmp = fitz.open()
    tmp.insert_pdf(doc, from_page=0, to_page=n - 1)
    doc.close()

    md = pymupdf4llm.to_markdown(tmp)
    tmp.close()
    return md


def extract_pymupdf_style(pdf_path: Path, pages: int, max_chars: int = 8000,
                           size_tol: float = 0.35,
                           require_bold_match: bool = False) -> Tuple[bool, Optional[int], str]:
    """
    Style-based extraction that uses font size and bold information to detect section boundaries.
    Returns (found, intro_line_index, intro_text).
    Only returns intro if an Introduction heading line is found.
    Stops at the next heading-like line with similar style signature.
    """
    import fitz

    doc = fitz.open(pdf_path)
    try:
        lines = _extract_lines_pymupdf(doc, max_pages=pages)
    finally:
        doc.close()

    intro_i: Optional[int] = None
    for i, lr in enumerate(lines):
        if INTRO_HEADING_RE.match(lr.text):
            intro_i = i
            break
    if intro_i is None:
        return False, None, ""

    intro_style_size = round(lines[intro_i].size, 1)
    intro_style_bold = lines[intro_i].boldish

    out_lines: List[str] = []
    out_lines.append(lines[intro_i].text)

    chars = len(out_lines[0]) + 1
    # Start collecting after intro heading
    for j in range(intro_i + 1, len(lines)):
        lr = lines[j]

        # Only consider stopping after we have some content
        if chars > 800:  # prevent stopping immediately on "1.1 Contributions" etc.
            if _heading_like(lr.text):
                size_close = abs(round(lr.size, 1) - intro_style_size) <= size_tol
                bold_ok = (lr.boldish == intro_style_bold) if require_bold_match else True
                if size_close and bold_ok:
                    break

        out_lines.append(lr.text)
        chars += len(lr.text) + 1
        if chars >= max_chars:
            break

    intro_text = "\n".join(out_lines)
    if len(intro_text) > max_chars:
        intro_text = intro_text[:max_chars]

    return True, intro_i, intro_text


def _extract_pymupdf_style_wrapper(pdf_path: Path, pages: int, max_chars: int) -> Tuple[bool, Optional[int], str]:
    """Wrapper to match the signature expected by benchmark_one."""
    return extract_pymupdf_style(pdf_path, pages, max_chars)


EXTRACTORS: Dict[str, Callable] = {
    "pymupdf": extract_pymupdf,
    "pdfplumber": extract_pdfplumber,
    "pymupdf4llm": extract_pymupdf4llm,
    "pymupdf-style": _extract_pymupdf_style_wrapper,
}


def truncate_intro(intro: str) -> str:
    """Truncate intro to first 400 and last 400 chars with '...' in between."""
    if len(intro) <= 800:
        return intro
    return intro[:400] + "..." + intro[-400:]


def benchmark_one(pdf_path: Path, extractor_name: str, pages: int, max_chars: int) -> RunResult:
    fn = EXTRACTORS[extractor_name]
    t0 = time.perf_counter()
    try:
        # Try calling with max_chars first (for style-based extractor)
        # If that fails, fall back to 2-parameter call
        try:
            result = fn(pdf_path, pages, max_chars)
        except TypeError:
            # Old extractors only take 2 parameters
            result = fn(pdf_path, pages)
        
        # Check if extractor returns a tuple (found, start_line, intro_text) or just text
        if isinstance(result, tuple) and len(result) == 3:
            found, start_line, intro = result
        else:
            # Assume it's plain text, use find_intro_slice
            text = result
            found, start_line, intro = find_intro_slice(text, max_chars=max_chars)
        dt = time.perf_counter() - t0
        return RunResult(
            pdf=str(pdf_path.name),
            extractor=extractor_name,
            pages=pages,
            max_chars=max_chars,
            seconds=dt,
            intro_found=found,
            intro_start_line=start_line,
            intro_chars=truncate_intro(intro),
            intro_chars_num=len(intro),
            error=None,
        )
    except Exception as e:
        dt = time.perf_counter() - t0
        return RunResult(
            pdf=str(pdf_path.name),
            extractor=extractor_name,
            pages=pages,
            max_chars=max_chars,
            seconds=dt,
            intro_found=False,
            intro_start_line=None,
            intro_chars="",
            intro_chars_num=0,
            error=f"{type(e).__name__}: {e}",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=str, default='./data', help="Folder containing PDFs.")
    ap.add_argument("--pages", type=int, default=3, help="How many first pages to scan.")
    ap.add_argument("--max-chars", type=int, default=7000, help="Max chars to return for intro slice.")
    ap.add_argument("--extractors", type=str, default="pymupdf,pdfplumber,pymupdf4llm,pymupdf-style",
                    help="Comma-separated list of extractors to run.")
    # ap.add_argument("--extractors", type=str, default="pymupdf,pymupdf-style",
    #                 help="Comma-separated list of extractors to run.")
    ap.add_argument("--out-json", type=str, default="results.json")
    ap.add_argument("--out-csv", type=str, default="summary.csv")
    ap.add_argument("--max-workers", type=int, default=6,
                help="Parallel workers (processes). Tune based on CPU/RAM.")

    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")

    print(f"Found {len(pdfs)} PDFs in {pdf_dir}")

    tasks = []
    for pdf in pdfs:
        for ext in args.extractors.split(","):
            tasks.append((str(pdf), ext, args.pages, args.max_chars))

    print(f"Running {len(tasks)} tasks across {len(args.extractors.split(','))} extractors")

    # cap workers; good default = number of CPU cores
    max_workers = min(os.cpu_count() or 4, args.max_workers)
    print(f"Using {max_workers} worker processes")

    results = []
    out_json = Path(args.out_json)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if i % 10 == 0 or i == len(tasks):
                print(f"Completed {i}/{len(tasks)} tasks ({i*100//len(tasks)}%)")
    
    print(f"Processing completed. Writing results...")
    # Write results as a single JSON array
    with out_json.open("w", encoding="utf-8") as f:
        sorted_results = sorted(results, key=lambda r: r.pdf)
        json.dump([asdict(r) for r in sorted_results], f, ensure_ascii=False, indent=2)

    df = pd.DataFrame([asdict(r) for r in results])

    # Aggregated stats per extractor
    summary = (
        df.groupby("extractor")
          .agg(
              n=("pdf", "count"),
              intro_found_rate=("intro_found", "mean"),
              seconds_mean=("seconds", "mean"),
              seconds_p95=("seconds", lambda s: s.quantile(0.95)),
              errors=("error", lambda x: x.notna().sum()),
              intro_chars_mean=("intro_chars_num", "mean"),
          )
          .reset_index()
          .sort_values(["intro_found_rate", "seconds_mean"], ascending=[False, True])
    )
    summary.to_csv(args.out_csv, index=False)

    print("Wrote:", out_json)
    print("Wrote:", args.out_csv)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

