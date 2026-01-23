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
    
    def packer_end(self, clusters: int, papers: int, warnings: list):
        """Log packer completion"""
        warnings_str = ",".join(warnings) if warnings else "[]"
        self._log("packer.end", f"clusters={clusters}, papers={papers}, warnings=[{warnings_str}]")
    
    def enrich_prompt(self, template_name: str, template_hash: str, preview: str):
        """Log prompt template details"""
        preview_truncated = preview[:200] + "..." if len(preview) > 200 else preview
        self._log("enrich.prompt", f'template={template_name}, hash={template_hash}, preview="{preview_truncated}"')
    
    def enrich_llm(self, model: str, api_uri: str, args: dict):
        """Log LLM call details"""
        # Sanitize API URI (remove API keys if present)
        sanitized_uri = api_uri.split('?')[0] if '?' in api_uri else api_uri
        args_str = ",".join(f"{k}={v}" for k, v in args.items())
        self._log("enrich.llm", f"model={model}, api_uri={sanitized_uri}, args={{{args_str}}}")
    
    def enrich_response_raw(self, length: int, saved_to: Optional[str] = None):
        """Log raw LLM response"""
        msg = f"length={length} chars"
        if saved_to:
            msg += f", saved to {saved_to}"
        self._log("enrich.response_raw", msg, level=logging.DEBUG)
    
    def enrich_response_parsed(self, cluster_cards: int, valid: bool):
        """Log parsed response"""
        self._log("enrich.response_parsed", f"cluster_cards={cluster_cards}, valid={valid}")
    
    def enrich_parse_error(self, error: str):
        """Log parse error"""
        self._log("enrich.parse_error", f'error="{error}"', level=logging.ERROR)
    
    def heuristic_metrics(self, schema_valid: bool, citations_ok: bool, length_ok: bool, name_generic_penalty: float = 0.0):
        """Log heuristic judge metrics"""
        schema_status = "pass" if schema_valid else "fail"
        citations_status = "pass" if citations_ok else "fail"
        length_status = "pass" if length_ok else "fail"
        self._log("heuristic.metrics", f"schema_valid={schema_status}, citations_ok={citations_status}, length_ok={length_status}, name_generic_penalty={name_generic_penalty}")
    
    def run_end(self, duration: float, overall_scores: dict, winner: Optional[str] = None):
        """Log run completion"""
        scores_str = ",".join(f"{k}={v}" for k, v in overall_scores.items())
        msg = f"duration={duration:.3f}s, overall_scores={{{scores_str}}}"
        if winner:
            msg += f", winner={winner}"
        self._log("run.end", msg)

