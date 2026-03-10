"""Configuration loader for report render pipeline"""

from pathlib import Path
import yaml
from pydantic import BaseModel, Field

from reader.pipelines.report_generation.config.config import MemoConfig


class CacheConfig(BaseModel):
    """Cache directory configuration for report render."""

    root: str = Field(..., description="Cache root directory (relative or absolute)")

    @property
    def abs_root(self) -> Path:
        """Absolute path of cache root. Resolves relative paths to absolute."""
        return Path(self.root).resolve()

    @property
    def render_report_log_path(self) -> Path:
        """Path to render report log file. Derived from abs_root."""
        return self.abs_root / "logs" / "render_report.log"


class RenderReportConfig(BaseModel):
    """Main report render configuration."""

    log_config_path: str = Field(..., description="Path to logging configuration YAML file")
    memo: MemoConfig = Field(..., description="Memo CLI configuration")
    cache: CacheConfig = Field(..., description="Cache directory configuration")


def load_config(path: str) -> RenderReportConfig:
    """
    Load and validate configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        RenderReportConfig: Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        config = RenderReportConfig(**data)
        return config
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e
