"""TUI for displaying ObservationReport in scrollable markdown."""

import asyncio

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, MarkdownViewer

from reader.pipelines.report_generation.report import (
    ObservationReport,
    ReportWriterFrontMatterOutput,
    ReportWriterSectionOutput,
)


def _observation_report_to_markdown(report: ObservationReport) -> str:
    """Convert ObservationReport to markdown for display."""
    parts = []
    fm = report.front_matter
    parts.append(f"# {fm.title}")
    parts.append(f"*{fm.summary}*")
    keywords_str = ", ".join(sorted(fm.keywords)) if fm.keywords else ""
    parts.append(f"Keywords: {keywords_str}")
    for section in report.body:
        parts.append(f"## {section.section_name}")
        parts.append(section.section_text)
    return "\n\n".join(parts)


class ReportViewer(App[None]):
    """Textual App for displaying a single ObservationReport in scrollable markdown."""

    CSS = """
    MarkdownViewer {
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("up", "scroll_up", "Scroll up"),
        ("down", "scroll_down", "Scroll down"),
        ("j", "scroll_down", "Scroll down"),
        ("k", "scroll_up", "Scroll up"),
        ("enter", "quit", "Exit"),
        ("q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
    ]

    def __init__(self, report: ObservationReport):
        super().__init__()
        self.report = report

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield MarkdownViewer(show_table_of_contents=False)
        yield Footer()

    def on_mount(self) -> None:
        viewer = self.query_one(MarkdownViewer)
        viewer.document.update(_observation_report_to_markdown(self.report))

    def action_scroll_up(self) -> None:
        viewer = self.query_one(MarkdownViewer)
        viewer.scroll_up()

    def action_scroll_down(self) -> None:
        viewer = self.query_one(MarkdownViewer)
        viewer.scroll_down()

    def action_quit(self) -> None:
        self.exit()


async def display_report(report: ObservationReport) -> None:
    """Run ReportViewer TUI for the given report."""
    await ReportViewer(report).run_async()


# test the tui under reader/src with command `python -m reader.tui.report_viewer`
if __name__ == "__main__":
    _draft_report = ObservationReport(
        cluster_pk_hash="draft-test-hash",
        front_matter=ReportWriterFrontMatterOutput(
            title="Draft Report",
            summary="This is a draft summary for testing the TUI report viewer. It contains at least 40 characters.",
            keywords={"draft", "test", "tui", "report", "viewer"},
        ),
        body=[
            ReportWriterSectionOutput(
                section_name="Overview",
                section_text="This is draft content for the overview section. Use `python report_viewer.py` to test the TUI.",
                confidence=["high"],
            ),
            ReportWriterSectionOutput(
                section_name="Details",
                section_text="Additional draft content for the details section. Press **q** or **Enter** to quit.",
                confidence=["medium"],
            ),
        ],
    )
    asyncio.run(display_report(_draft_report))
