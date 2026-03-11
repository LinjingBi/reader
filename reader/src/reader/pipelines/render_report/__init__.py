"""Render report pipeline: load report from file and display in TUI."""

from reader.pipelines.render_report.blocks import render_report
from reader.pipelines.render_report.config.config import RenderReportConfig, load_config
from reader.pipelines.render_report.model import RenderReportOutput

__all__ = ["render_report", "RenderReportOutput", "load_config", "RenderReportConfig"]
