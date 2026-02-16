"""Configuration loader for reader package"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml
from pydantic import BaseModel, Field, computed_field


class RunConfig(BaseModel):
    """Run configuration"""
    period_start: str = Field(..., description="Period start date in YYYY-MM-DD format")
    period_end: str = Field(..., description="Period end date in YYYY-MM-DD format")


class MemoConfig(BaseModel):
    """Memo CLI configuration"""
    enabled: bool = Field(default=False, description="Enable memo CLI integration")
    bin: str = Field(..., description="Path to memo binary")
    db_path: str = Field(default=None, description="Path to memo database")
    db_schema_path: str = Field(default=None, description="Path to memo database schema")
    timeout_sec: int = Field(default=60, description="Timeout for memo CLI calls")


class LLMGeminiConfig(BaseModel):
    """Gemini LLM API configuration"""
    models: List[str] = Field(..., description="List of Gemini model names")
    temperature: float = Field(..., description="Temperature parameter")
    max_tokens: int = Field(..., description="Maximum tokens to generate")
    api_key_env: str = Field(..., description="Environment variable name for API key")
    gemini_rpm_limit: int = Field(default=15, description="Requests per minute limit")
    gemini_tpm_limit: int = Field(default=250000, description="Tokens per minute limit")


class ClusterSummarizationConfig(BaseModel):
    """Cluster summarization configuration"""
    enable: bool = Field(default=False, description="Whether to enable cluster summarization")
    llm_model: str = Field(..., description="Model name (must exist in llm_gemini.models list)")
    prompt_template: str = Field(..., description="Path to prompt template file (relative to prompts directory)")
    
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
        # Resolve template path relative to reader/src/reader/prompts/
        # This assumes config.py is at reader/src/reader/config.py
        config_file = Path(__file__)
        prompts_dir = config_file.parent / "prompts"
        template_path = prompts_dir / self.prompt_template
        
        if not template_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {template_path}. "
                f"Expected location: {prompts_dir}/"
            )
        
        return template_path


class ReportGenerationConfig(BaseModel):
    """Report generation configuration"""
    enable: bool = Field(default=False, description="Whether to enable report generation")
    topic_resolver_threshold: float = Field(default=0.98, description="Similarity threshold (0-1) for topic resolution. If best similarity >= threshold, merge; otherwise create")


class ReaderConfig(BaseModel):
    """Main reader configuration"""
    run: RunConfig
    memo: MemoConfig
    llm_gemini: LLMGeminiConfig
    report_generation: Optional[ReportGenerationConfig] = Field(default=None, description="Report generation configuration")


def load_config(path: str) -> ReaderConfig:
    """
    Load and validate configuration from YAML file.
    
    Args:
        path: Path to YAML config file
        
    Returns:
        ReaderConfig: Validated configuration object
        
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
        config = ReaderConfig(**data)
        return config
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e


