#!/usr/bin/env python3
import sys
from typing import Any, Dict, List, Optional, Tuple

from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widgets import Header, Footer, MarkdownViewer

from textual import on
from textual.widgets.markdown import Markdown
from textual.css.query import NoMatches

from reader.adapters.memo import ClusterObservationData
from reader.logging.logging_setup import get_logger

logger = get_logger()


class ClusterObservationError(Exception):
    """Exception raised when cluster observation data is invalid."""
    pass


def sorted_topic_keys(observations: Dict[str, ClusterObservationData]) -> List[str]:
    """
    Sort pk_hash keys by cluster_period_end (most recent first, descending), 
    then observation_created_time (newest first, descending), then alphabetically.
    
    Returns sorted list of pk_hash strings. The UI displays these as indices
    (1, 2, 3, ...) but returns the actual pk_hash when Enter is pressed.
    """
    def sort_key(pk_hash: str) -> Tuple[float, float, str]:
        obs = observations[pk_hash]
        # Convert datetime objects to timestamps
        # Pydantic already validated these are valid datetime objects
        period_end_ts = obs.cluster_period_end.timestamp()
        created_time_ts = obs.observation_created_time.timestamp()
        
        # Return tuple: (-period_end_ts for descending, -created_time_ts for descending, pk_hash)
        # Negative values because we want newest/most recent first (descending order)
        return (-period_end_ts, -created_time_ts, pk_hash)
    
    keys = list(observations.keys())
    return sorted(keys, key=sort_key)


def truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


def key_to_section_title(key: str) -> str:
    """
    Convert a snake_case key to a title case section name.
    
    Example: "what_this_topic_is_about" -> "What This Topic Is About"
    """
    if not key:
        return ""
    # Replace underscores with spaces and title case each word
    return key.replace("_", " ").title()


def get_sections_from_payloads(observations: Dict[str, ClusterObservationData]) -> List[str]:
    """
    Extract all unique keys from json_payload dicts across all observations.
    Returns sorted list of section titles.
    """
    all_keys = set()
    for obs in observations.values():
        if isinstance(obs.json_payload, dict):
            all_keys.update(obs.json_payload.keys())
    
    # Convert keys to section titles and sort
    sections = sorted([key_to_section_title(key) for key in all_keys])
    return sections


def slug(s: str) -> str:
    # Matches common markdown slug rules well enough
    import re
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s


def format_value(value: Any) -> str:
    """
    Recursively format a value based on its type.
    
    - Non-iterable types (str, int, float, bool, None): display as-is
    - List types: iterate recursively, use bullet points ('-') for each item
    - Dict types: iterate recursively, connect string values with " - " separator
    """
    # Handle None
    if value is None:
        return "_(none)_"
    
    # Handle non-iterable types (but str is iterable, so check it separately)
    if isinstance(value, (int, float, bool)):
        return str(value)
    
    # Handle strings (they're iterable but should be treated as atomic)
    if isinstance(value, str):
        return value if value else "_(none)_"
    
    # Handle lists
    if isinstance(value, list):
        if not value:
            return "_(none)_"
        lines = []
        for item in value:
            formatted = format_value(item)
            lines.append(f"- {formatted}")
        return "\n".join(lines)
    
    # Handle dicts
    if isinstance(value, dict):
        if not value:
            return "_(none)_"
        # For dicts, recursively format all values and connect string results with " - "
        parts = []
        for v in value.values():
            formatted_value = format_value(v)
            # Only include non-empty formatted values
            if formatted_value and formatted_value != "_(none)_":
                parts.append(formatted_value)
        return " - ".join(parts) if parts else "_(none)_"
    
    # Fallback for other types
    return str(value)


def report_to_markdown(report: Dict[str, Any]) -> str:
    """
    Convert a report dict to markdown format.
    Uses dynamic section generation based on keys in the report.
    """
    title = report.get("title") or report.get("topic_name")
    one_liner = report.get("one_liner")
    
    # Get all keys from the report (excluding title/one_liner which are handled separately)
    section_keys = [k for k in report.keys() if k not in ("title", "topic_name", "one_liner")]
    
    # Build sections dynamically
    section_parts = []
    # TODO: consider sorting by importance/relevance instead of reverse order
    for key in sorted(section_keys, reverse=True):
        section_title = key_to_section_title(key)
        value = report.get(key)
        content = format_value(value)
        section_parts.append(f"## {section_title}\n{content}")
    
    sections_md = "\n\n".join(section_parts)
    
    # Build markdown parts conditionally
    markdown_parts = []
    if title:
        markdown_parts.append(f"# {title}")
    if one_liner:
        markdown_parts.append(f"*{one_liner}*")
    if sections_md:
        markdown_parts.append(sections_md)
    
    return "\n\n".join(markdown_parts)


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

    def __init__(self, observations: Dict[str, ClusterObservationData]):
        super().__init__()
        self.observations = observations
        self.topic_keys = sorted_topic_keys(observations)

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
        obs = self.observations[key]
        
        # Validate json_payload is a dict
        if not isinstance(obs.json_payload, dict):
            raise ClusterObservationError(
                f"json_payload for pk_hash '{key}' is not a dict. Got type: {type(obs.json_payload).__name__}"
            )
        
        report = obs.json_payload
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

        # Optional: stop any default "just make visible" behavior
        event.stop()

    def _render_current(self) -> None:
        _key, report = self._current_key_and_report()

        # Put selector into Header
        self.title = self._selector_string(len(self.topic_keys), self.current_index)

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


class UserIntentSelector(App[str]):
    """TUI widget for selecting user intent from a list of options."""
    
    CSS = """
    MarkdownViewer {
        padding: 0 1;
    }
    """
    
    BINDINGS = [
        ("up", "prev", "Prev intent"),
        ("down", "next", "Next intent"),
        ("left", "prev", "Prev intent"),
        ("right", "next", "Next intent"),
        ("enter", "select", "Select intent"),
        ("q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
    ]
    
    current_index = reactive(0)
    
    def __init__(self, intent_options: List[str]):
        super().__init__()
        # intent_options is a list of display-ready strings
        self.intent_options = intent_options
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield MarkdownViewer(show_table_of_contents=False)
        yield Footer()
    
    def on_mount(self) -> None:
        self._render_current()
    
    def watch_current_index(self, _old: int, _new: int) -> None:
        self._render_current()
    
    def _selector_string(self, n: int, idx: int) -> str:
        parts = []
        for i in range(n):
            label = str(i + 1)
            parts.append(f"[{label}]" if i == idx else label)
        return "Intent Options: " + " ".join(parts) + "   |   ↑/↓/number, Enter=select"
    
    def _format_intent_markdown(self) -> str:
        """Format intent options as markdown for display."""
        lines = ["# Select Report Intent\n"]
        lines.append("Choose your preferred report style:\n")
        
        for i, intent_text in enumerate(self.intent_options):
            marker = "**→**" if i == self.current_index else "  "
            lines.append(f"{marker} {i + 1}. **{intent_text}**")
            lines.append("")
        
        return "\n".join(lines)
    
    def _render_current(self) -> None:
        self.title = self._selector_string(len(self.intent_options), self.current_index)
        
        viewer = self.query_one(MarkdownViewer)
        viewer.document.update(self._format_intent_markdown())
        viewer.scroll_home(animate=False)
    
    def action_prev(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
    
    def action_next(self) -> None:
        if self.current_index < len(self.intent_options) - 1:
            self.current_index += 1
    
    def action_select(self) -> None:
        # Return the selected intent string directly
        self.exit(self.intent_options[self.current_index])
    
    def on_key(self, event) -> None:
        # Number jump: 1..9 etc.
        if event.key.isdigit():
            idx = int(event.key) - 1
            if 0 <= idx < len(self.intent_options):
                self.current_index = idx
                event.stop()


def display_clusters_observation(observations: Dict[str, ClusterObservationData], user_intent_options: Optional[List[str]] = None) -> Tuple[str, Optional[str]]:
    """
    Main entry point for the TUI browser.
    
    Two-stage selection: first select a topic/cluster, then select a user intent (if options provided).
    
    Args:
        observations: Dictionary mapping pk_hash to ClusterObservationData
        user_intent_options: Optional list of display-ready user intent option strings
        
    Returns:
        Tuple of (selected pk_hash, selected user intent string or None if skipped)
        
    Raises:
        ClusterObservationError: If observations are empty
    """
    if not observations:
        logger.error("Empty observations: no topics to display.")
        raise ClusterObservationError("Empty observations: no topics to display.")
    
    # Stage 1: Select topic/cluster
    selected_pk_hash = TopicBrowser(observations).run()
    if selected_pk_hash is None:
        logger.info("No topic was selected (user quit without selection).")
        raise ClusterObservationError("No topic was selected (user quit without selection).")
    
    # Stage 2: Select user intent (if options provided)
    selected_intent = None
    if not user_intent_options:
        logger.info("No user_intent_options provided, intent selector will be skipped.")
    else:
        selected_intent = UserIntentSelector(user_intent_options).run()
        
        if selected_intent is None:
            logger.info("No user intent was selected (user quit without selection).")
        else:
            logger.info(f"Selected user intent: {selected_intent}")
    
    return (str(selected_pk_hash), selected_intent)
