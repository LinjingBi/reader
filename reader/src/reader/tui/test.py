#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Any, Dict, List, Tuple

from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widgets import Header, Footer, MarkdownViewer

from textual import on
from textual.widgets.markdown import Markdown
from textual.css.query import NoMatches


def sorted_topic_keys(d: Dict[str, Any]) -> List[str]:
    """
    Sort pk_hash keys as strings.
    
    Returns sorted list of pk_hash strings. The UI displays these as indices
    (1, 2, 3, ...) but returns the actual pk_hash when Enter is pressed.
    """
    keys = list(d.keys())
    # Sort pk_hash strings alphabetically
    return sorted(keys)


def truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


# Mapping from section title to report key
SECTION_TO_KEY = {
    "What this topic is about": "what_this_topic_is_about",
    "Why it matters": "why_it_matters",
    "Confidence": "confidence",
    "Representative papers": "representative_papers",
    "Reading order": "reading_order",
    "Notes": "notes",
    "Keywords": "keyword_list",
}

SECTIONS = list(SECTION_TO_KEY.keys())


def slug(s: str) -> str:
    # Matches common markdown slug rules well enough
    import re
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s


def toc_md() -> str:
    lines = ["## Table of contents"]
    for sec in SECTIONS:
        lines.append(f"- [{sec}](#{slug(sec)})")
    return "\n".join(lines)


def format_section_content(section_title: str, value: Any, report: Dict[str, Any]) -> str:
    """Format the content for a given section."""
    if section_title == "Confidence":
        conf_rationale = report.get("confidence_rationale", []) or []
        conf_lines = [f"- {x}" for x in conf_rationale]
        rationale_text = chr(10).join(conf_lines) if conf_lines else ""
        return f"**{value}**\n{rationale_text}".strip()
    
    elif section_title == "Representative papers":
        if not value:
            return "_(none)_"
        lines = []
        for p in value:
            pid = p.get("paper_id", "")
            pt = p.get("title", "")
            lines.append(f"- `{pid}` — {pt}")
        return chr(10).join(lines) if lines else "_(none)_"
    
    elif section_title == "Reading order":
        if not value:
            return "_(none)_"
        lines = []
        for i, p in enumerate(value, start=1):
            pid = p.get("paper_id", "")
            why_now = p.get("why_read_now", "")
            lines.append(f"{i}. `{pid}` — {why_now}")
        return chr(10).join(lines) if lines else "_(none)_"
    
    elif section_title == "Notes":
        if not value:
            return "_(none)_"
        lines = [f"- {x}" for x in value]
        return chr(10).join(lines) if lines else "_(none)_"
    
    elif section_title == "Keywords":
        if not value:
            return "_(none)_"
        return ", ".join(value)
    
    else:
        # Simple string sections
        return value if value else "_(none)_"


def report_to_markdown(report: Dict[str, Any]) -> str:
    title = report.get("title", "(untitled)")
    one_liner = report.get("one_liner", "")
    
    # Build sections dynamically
    section_parts = []
    for section_title in SECTIONS:
        key = SECTION_TO_KEY[section_title]
        value = report.get(key, "" if key != "keyword_list" else [])
        content = format_section_content(section_title, value, report)
        section_parts.append(f"## {section_title}\n{content}")
    
    sections_md = "\n\n".join(section_parts)
    
#     return f"""# {title}

# *{one_liner}*

# {toc_md()}

# {sections_md}
# """

    return f"""# {title}

*{one_liner}*

{sections_md}
"""

class TopicBrowser(App[str]):
    CSS = """
    MarkdownViewer {
        padding: 0 1;
    }
    MarkdownTableOfContents {
    width: 40;      /* tune this */
    min-width: 24;
    max-width: 34;
    }
    """

    BINDINGS = [
        ("left", "prev", "Prev topic"),
        ("right", "next", "Next topic"),
        ("enter", "select", "Select topic"),
        ("q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
    ]

    current_index = reactive(0)

    def __init__(self, clusters: Dict[str, Any]):
        super().__init__()
        self.clusters = clusters
        self.topic_keys = sorted_topic_keys(clusters)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield MarkdownViewer(show_table_of_contents=False)
        yield Footer()

    def on_mount(self) -> None:
        self._render_current()

    def watch_current_index(self, _old: int, _new: int) -> None:
        self._render_current()

    def _current_key_and_report(self) -> Tuple[str, Dict[str, Any]]:
        key = self.topic_keys[self.current_index]
        obj = self.clusters[key]
        report = obj.get("cluster_report", obj)  # tolerate either shape
        return key, report

    def _selector_string(self, n: int, idx: int) -> str:
        # Example: Topics: [1] 2 3 4 | ←/→/number Enter
        parts = []
        for i in range(n):
            label = str(i + 1)
            parts.append(f"[{label}]" if i == idx else label)
        return "Topics: " + " ".join(parts) + "   |   ←/→/number, Enter=select"
    
    @on(Markdown.TableOfContentsSelected)
    def _toc_selected(self, event: Markdown.TableOfContentsSelected) -> None:
        viewer = self.query_one(MarkdownViewer)
        md = viewer.document
        block_id = event.block_id
        if not block_id:
            return

        try:
            target = md.query_one(f"#{block_id}")
        except NoMatches:
            # Fallback: if ids don't exist for some reason, try anchor logic
            md.goto_anchor(block_id)
            return

        # Put the heading at the top of the viewport
        viewer.scroll_to_widget(target, top=True, animate=False)

        # Optional: stop any default “just make visible” behavior
        event.stop()

    def _render_current(self) -> None:
        _key, report = self._current_key_and_report()

        # Put selector + current title into Header
        self.title = self._selector_string(len(self.topic_keys), self.current_index)
        self.sub_title = "Current: " + truncate(report.get("title", "(untitled)"), 80)

        viewer = self.query_one(MarkdownViewer)
        viewer.document.update(report_to_markdown(report))
        viewer.scroll_home(animate=False)

    def action_prev(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1

    def action_next(self) -> None:
        if self.current_index < len(self.topic_keys) - 1:
            self.current_index += 1

    def action_select(self) -> None:
        self.exit(self.topic_keys[self.current_index])

    def on_key(self, event) -> None:
        # Number jump: 1..9 etc.
        if event.key.isdigit():
            idx = int(event.key) - 1
            if 0 <= idx < len(self.topic_keys):
                self.current_index = idx
                event.stop()


def load_clusters(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object (dict).")
    return data


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="Path to cluster report JSON file")
    args = ap.parse_args(argv)

    clusters = load_clusters(args.json_path)
    if not clusters:
        print("Empty JSON object: no topics to display.", file=sys.stderr)
        return 2

    selected = TopicBrowser(clusters).run()
    if selected is not None:
        print(selected)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
