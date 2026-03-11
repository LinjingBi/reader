"""Output models for report signature check pipeline"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CheckReportSignatureOutput(BaseModel):
    """Return value for check_report_signature function."""

    status: Literal["match", "not_match", "error"] = Field(
        ...,
        description="match when signature verified; not_match or error otherwise",
    )
    message: Optional[str] = Field(
        default=None,
        description="Optional message from memo or error details",
    )
