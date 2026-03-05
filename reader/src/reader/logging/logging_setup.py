from __future__ import annotations

import logging
import logging.config
from datetime import datetime
from pathlib import Path


class TzAwareFormatter(logging.Formatter):
    """Formatter that uses local time with timezone offset (ISO 8601). Parse with datetime.fromisoformat() and convert to UTC via .astimezone(timezone.utc)."""

    def formatTime(self, record, datefmt=None):
        return datetime.now().astimezone().isoformat()


def setup_logging(config_path: str | Path, log_file_path: str | Path) -> None:
    """
    Loads logging configuration from YAML/JSON dictConfig.
    
    Args:
        config_path: Path to logging configuration YAML/JSON file
        log_file_path: Path to log file. Will override the filename in the config
                      and create parent directories.
    """
    path = Path(config_path)

    if path.suffix in {".yaml", ".yml"}:
        import yaml  # pip install pyyaml
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif path.suffix == ".json":
        import json
        config = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported logging config format: {path.suffix}")

    # Update file handler path
    log_file_path = Path(log_file_path)
    # Create parent directories if they don't exist
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    # Update the file handler's filename in config
    if "handlers" in config and "file" in config["handlers"]:
        config["handlers"]["file"]["filename"] = str(log_file_path)

    logging.config.dictConfig(config)

def get_logger(name: str = "reader") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (default: "reader")
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

