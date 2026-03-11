"""Output models for render report pipeline"""

from typing import Literal

from pydantic import BaseModel, Field


class RenderReportOutput(BaseModel):
    """Return value for render_report function."""

    status: Literal["done", "error"] = Field(
        ...,
        description="done when report rendered successfully; error for all error cases",
    )
    message: str = Field(
        ...,
        description="Success message when done; error-specific message when error",
    )
