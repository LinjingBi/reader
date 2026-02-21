"""Configuration loader for HF data pipeline"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml
from pydantic import BaseModel, Field, computed_field


class RunConfig(BaseModel):
    """Run configuration"""
    month_key: str = Field(..., description="Month key in format 'month=YYYY-MM'")
    log_config_path: str = Field(..., description="Path to logging configuration YAML file")
    log_file_path: str = Field(..., description="Path to log file")


class HuggingFaceSourceConfig(BaseModel):
    """HuggingFace source configuration"""
    daily_papers_api: str = Field(..., description="HF daily papers API URL")
    paper_page_base_url: str = Field(..., description="HF paper page base URL")
    output_json: str = Field(default="papers_report.json", description="Output JSON file path")


class SourcesConfig(BaseModel):
    """Sources configuration"""
    hf: HuggingFaceSourceConfig


class EmbeddingConfig(BaseModel):
    """Embedding algorithm configuration"""
    model: str = Field(..., description="Embedding model name")
    modes: List[str] = Field(..., description="Embedding modes to try")
    top_n_keywords: int = Field(..., description="Number of top keywords to use")


class ClusteringConfig(BaseModel):
    """Clustering algorithm configuration"""
    method: str = Field(default="kmeans", description="Clustering method")
    k_candidates: List[int] = Field(..., description="K values to try")
    random_seed: int = Field(..., description="Random seed for reproducibility")

class PaperChunkConfig(BaseModel):
    """Paper chunk configuration"""
    rules_path: str = Field(..., description="Path to paper chunk rules YAML file")

class AlgosConfig(BaseModel):
    """Algorithms configuration"""
    embedding: EmbeddingConfig
    clustering: ClusteringConfig
    paperchunk: PaperChunkConfig


class OutputsConfig(BaseModel):
    """Outputs configuration"""
    best_cluster_text_report_path_template: Optional[str] = Field(
        default=None,
        description="Template for best cluster text report path (use {month_key} placeholder). If not provided, no text report will be created."
    )
    best_cluster_report_path_template: Optional[str] = Field(
        default=None,
        description="Template for best cluster JSON report path (use {month_key} placeholder). If not provided, no JSON report will be created."
    )
    papers_scoring_summary_report: Optional[str] = Field(
        default=None,
        description="Template for paper scoring summary report path (use {month_key} placeholder). If not provided, no summary report will be created."
    )
    paper_scoring_debug_heading_events: Optional[str] = Field(
        default=None,
        description="Template for paper scoring debug heading events JSONL path (use {month_key} placeholder). If not provided, no debug events file will be created."
    )


class MemoConfig(BaseModel):
    """Memo CLI configuration"""
    bin: str = Field(..., description="Path to memo binary")
    db_path: str = Field(default=None, description="Path to memo database")
    db_schema_path: str = Field(default=None, description="Path to memo database schema")
    timeout_sec: int = Field(default=60, description="Timeout for memo CLI calls")


class LLMGeminiConfig(BaseModel):
    """Gemini LLM API configuration"""
    model: str = Field(..., description="Gemini model name")
    temperature: float = Field(..., description="Temperature parameter")
    max_tokens: int = Field(..., description="Maximum tokens to generate")
    api_key_env: str = Field(..., description="Environment variable name for API key")
    gemini_rpm_limit: int = Field(default=15, description="Requests per minute limit")
    gemini_tpm_limit: int = Field(default=250000, description="Tokens per minute limit")


class ClusterSummarizationConfig(BaseModel):
    """Cluster summarization configuration"""
    llm_gemini: LLMGeminiConfig = Field(..., description="LLM Gemini configuration")
    prompt_template: str = Field(..., description="Path to prompt template file (relative to hf_data/prompts directory)")
    
    @computed_field
    @property
    def prompt_template_path(self) -> Path:
        """
        Resolve and validate prompt template path.
        
        Returns:
            Resolved Path to prompt template file
            
        Raises:
            FileNotFoundError: If template file doesn't exist
        """
        # Resolve template path relative to hf_data/prompts/
        # This assumes config.py is at hf_data/config/config.py
        config_file = Path(__file__)
        # Go up from config/ to hf_data/, then into prompts/
        hf_data_dir = config_file.parent.parent
        prompts_dir = hf_data_dir / "prompts"
        template_path = prompts_dir / self.prompt_template
        
        if not template_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {template_path}. "
                f"Expected location: {prompts_dir}/"
            )
        
        return template_path


class HFDataPipeConfig(BaseModel):
    """HF data pipeline configuration"""
    run: RunConfig
    sources: SourcesConfig
    algos: AlgosConfig
    outputs: OutputsConfig
    memo: MemoConfig
    cluster_summarization: ClusterSummarizationConfig


def load_config(path: str) -> HFDataPipeConfig:
    """
    Load and validate configuration from YAML file.
    
    Args:
        path: Path to YAML config file
        
    Returns:
        HFDataPipeConfig: Validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    try:
        config = HFDataPipeConfig(**data)
        return config
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e


def render_best_cluster_text_report_path(cfg: HFDataPipeConfig, month_key: str) -> Optional[str]:
    """
    Render the best cluster text report path template with month_key.
    
    Args:
        cfg: HFDataPipeConfig instance
        month_key: Month key to substitute (e.g., "month=2025-01")
        
    Returns:
        Rendered path string, or None if template is not configured
    """
    if cfg.outputs.best_cluster_text_report_path_template is None:
        return None
    return cfg.outputs.best_cluster_text_report_path_template.format(month_key=month_key)


def render_best_cluster_report_path(cfg: HFDataPipeConfig, month_key: str) -> Optional[str]:
    """
    Render the best cluster JSON report path template with month_key.
    
    Args:
        cfg: HFDataPipeConfig instance
        month_key: Month key to substitute (e.g., "month=2025-01")
        
    Returns:
        Rendered path string, or None if template is not configured
    """
    if cfg.outputs.best_cluster_report_path_template is None:
        return None
    return cfg.outputs.best_cluster_report_path_template.format(month_key=month_key)


def render_papers_scoring_summary_report_path(cfg: HFDataPipeConfig, month_key: str) -> Optional[str]:
    """
    Render the paper scoring summary report path template with month_key.
    
    Args:
        cfg: HFDataPipeConfig instance
        month_key: Month key to substitute (e.g., "month=2025-01")
        
    Returns:
        Rendered path string, or None if template is not configured
    """
    if cfg.outputs.papers_scoring_summary_report is None:
        return None
    return cfg.outputs.papers_scoring_summary_report.format(month_key=month_key)


def render_paper_scoring_debug_heading_events_path(cfg: HFDataPipeConfig, month_key: str) -> Optional[str]:
    """
    Render the paper scoring debug heading events JSONL path template with month_key.
    
    Args:
        cfg: HFDataPipeConfig instance
        month_key: Month key to substitute (e.g., "month=2025-01")
        
    Returns:
        Rendered path string, or None if template is not configured
    """
    if cfg.outputs.paper_scoring_debug_heading_events is None:
        return None
    return cfg.outputs.paper_scoring_debug_heading_events.format(month_key=month_key)

