"""Configuration loader for report generation pipeline"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field, PrivateAttr


class RunConfig(BaseModel):
    """Run configuration"""
    source: str = Field(..., description="Data source identifier (e.g., hf_monthly)")
    period_start: str = Field(..., description="Period start date in YYYY-MM-DD format")
    period_end: str = Field(..., description="Period end date in YYYY-MM-DD format")
    log_config_path: str = Field(..., description="Path to logging configuration YAML file")
    log_file_path: str = Field(..., description="Path to log file")


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
    gemini_call_max_workers: Optional[int] = Field(default=None, description="Number of worker threads for LLM call executor. If set, enables non-blocking async LLM calls.")
    max_retry: int = Field(default=5, description="Max retry attempts for LLM API calls")
    _gemini_call_executor: Optional[ThreadPoolExecutor] = PrivateAttr(default=None)

    def model_post_init(self, __context):
        """Initialize executor if max_workers is specified"""
        if self.gemini_call_max_workers is not None:
            self._gemini_call_executor = ThreadPoolExecutor(max_workers=self.gemini_call_max_workers)
        else:
            self._gemini_call_executor = None

    @property
    def gemini_call_executor(self) -> Optional[ThreadPoolExecutor]:
        """Get the LLM call executor for non-blocking async calls"""
        return self._gemini_call_executor


class ReportGenSectionConfig(BaseModel):
    """Report generation section configuration"""
    topic_resolver_threshold: float = Field(default=0.98, description="Similarity threshold (0-1) for topic resolution. If best similarity >= threshold, merge; otherwise create")
    llm_gemini: LLMGeminiConfig = Field(..., description="LLM Gemini configuration for report generation")
    max_planner_judge_retries: int = Field(default=3, description="Max retries when judge score below threshold")
    planner_judge_retry_threshold: float = Field(default=1.5, description="Accept if overall > threshold (0-2 scale)")
    planner_output_log_path: Optional[str] = Field(default=None, description="If set, append (planner_output, judge_output) to JSONL")
    max_evidence_gaps_threshold: int = Field(default=3, description="Evidence collection: conclude when len(evidence_gaps) < this")
    max_evidence_loop_iterations: int = Field(default=3, description="Evidence collection: conclude when iteration count exceeds this")
    # Writer config
    writer_supplement_prompt_template: str = Field(..., description="Template name in prompts/writer/ for supply step")
    writer_section_writing_prompt_template: str = Field(..., description="Template name in prompts/writer/ for write step")
    max_writer_supplement_judge_retries: int = Field(default=3, description="Max retries for supply judge")
    writer_supplement_judge_retry_threshold: float = Field(default=0.0, description="Accept supply if overall > threshold")
    writer_supplement_output_log_path: Optional[str] = Field(default=None, description="If set, append supply outputs to JSONL")
    max_writer_writing_judge_retries: int = Field(default=3, description="Max retries for writing judge")
    writer_writing_judge_retry_threshold: float = Field(default=0.0, description="Accept writing if overall > threshold")
    writer_writing_output_log_path: Optional[str] = Field(default=None, description="If set, append writing outputs to JSONL")
    # Front matter (summary) config
    writer_summary_prompt_template: str = Field(..., description="Template path for summary/front matter step")
    max_writer_summary_judge_retries: int = Field(default=3, description="Max retries for summary judge")
    writer_summary_judge_retry_threshold: float = Field(default=0.0, description="Accept summary if overall > threshold")
    writer_summary_output_log_path: Optional[str] = Field(default=None, description="If set, append summary outputs to JSONL")


class ReportGenerationConfig(BaseModel):
    """Main report generation configuration"""
    run: RunConfig
    memo: MemoConfig
    report_generation: ReportGenSectionConfig = Field(..., description="Report generation configuration")


def load_config(path: str) -> ReportGenerationConfig:
    """
    Load and validate configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        ReportGenerationConfig: Validated configuration object

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
        config = ReportGenerationConfig(**data)
        return config
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e
