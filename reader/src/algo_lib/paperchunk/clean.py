from __future__ import annotations

import re

# Remove citation-style reference indices like [1], [12], [3,4], [3-5]
_BRACKET_REF = re.compile(r"\[(?:\s*\d+\s*(?:[-–]\s*\d+)?\s*)(?:,\s*\d+\s*(?:[-–]\s*\d+)?\s*)*\]")
_SUPERSCRIPTS = str.maketrans({c: "" for c in "¹²³⁴⁵⁶⁷⁸⁹⁰"})
_WS = re.compile(r"[ \t]+")

def clean_v1(text: str) -> str:
    """MVP cleaner:
    - remove bracketed numeric reference indices like [12], [3,4], [3-5]
    - remove common superscript digits
    - normalize whitespace (preserve newlines)
    """
    if not text:
        return text
    t = _BRACKET_REF.sub("", text)
    t = t.translate(_SUPERSCRIPTS)
    # normalize spaces but keep newlines
    t = "\n".join(_WS.sub(" ", ln).strip() for ln in t.splitlines())
    # drop repeated blank lines
    out_lines = []
    blank = 0
    for ln in t.splitlines():
        if ln.strip() == "":
            blank += 1
            if blank <= 1:
                out_lines.append("")
        else:
            blank = 0
            out_lines.append(ln)
    return "\n".join(out_lines).strip()
