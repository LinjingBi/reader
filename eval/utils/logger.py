"""Structured logging utility for eval pipeline"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class EvalLogger:
    """Structured logger for eval pipeline with stage-based logging"""
    
    def __init__(self, run_id: str, log_dir: Optional[Path] = None):
        """
        Initialize logger for a run.
        
        Args:
            run_id: Unique identifier for this run
            log_dir: Directory to write log file (if None, only console)
        """
        self.run_id = run_id
        self.log_dir = log_dir
        self.logger = logging.getLogger(f"eval_{run_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Remove any existing handlers
        
        # Custom formatter: [datetime][stage] - content
        # Stage will be included in message via _log method
        formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d]%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (DEBUG level, if log_dir provided)
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "run.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _log(self, stage: str, message: str, level: int = logging.INFO):
        """Internal log method with stage prefix"""
        # Format: [datetime][stage] - content
        formatted_message = f"[{stage}] - {message}"
        self.logger.log(level, formatted_message)
    
    def run_start(self, snapshot_id: str):
        """Log run start"""
        self._log("run.start", f"run_id={self.run_id}, snapshot_id={snapshot_id}")
    
    def config_artifact(self, config: dict):
        """Log configuration artifacts (excluding credentials like api_key)"""
        # Filter out credential-related keys
        filtered_config = {}
        credential_keys = {'api_key', 'api_key_env', 'password', 'secret', 'token'}
        
        def filter_dict(d, parent_key=''):
            result = {}
            for k, v in d.items():
                full_key = f"{parent_key}.{k}" if parent_key else k
                if any(cred_key in k.lower() for cred_key in credential_keys):
                    continue  # Skip credential fields
                if isinstance(v, dict):
                    result[k] = filter_dict(v, full_key)
                else:
                    result[k] = v
            return result
        
        filtered_config = filter_dict(config)
        config_str = json.dumps(filtered_config, sort_keys=True)
        self._log("config.artifact", config_str)
    
    
    def run_end(self, duration: float):
        """Log run completion"""
        msg = f"duration={duration:.3f}s"
        self._log("run.end", msg)

