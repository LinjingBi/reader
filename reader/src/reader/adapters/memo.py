"""Memo CLI adapter"""

import json
import subprocess
from typing import List, Optional

from pydantic import BaseModel

from reader.config import ReaderConfig
from reader.pipelines.report import FreshPaperPayload
from reader.logging.logging_setup import get_logger

logger = get_logger()


# Pydantic response models matching Rust CLI contracts


class PaperCard(BaseModel):
    """Paper card in cluster response."""
    paper_id: str
    title: str
    summary: str
    keywords: List[str]
    url: str
    rank_in_cluster: int
    sim_to_centroid: Optional[float] = None


class ClusterCard(BaseModel):
    """Cluster card in best run response."""
    cluster_index: int
    size: int
    cohesion: Optional[float] = None
    papers: List[PaperCard]


class GetBestRunResponse(BaseModel):
    """Response from get-best-run command."""
    source: str
    period_start: str
    period_end: str
    embed_config_id: str
    cluster_config_id: str
    clusters: List[ClusterCard]


def fresh_paper(payload: FreshPaperPayload, config: ReaderConfig) -> None:
    """
    Call memo CLI fresh-paper command to ingest papers and clustering.
    
    Args:
        payload: FreshPaperPayload instance.
        config: ReaderConfig instance
        
    Returns:
        FreshPaperResponse instance, or None if disabled/error
    """
    if not config.memo.enabled:
        return None
    
    try:
        # Convert payload to JSON string
        payload_json = payload.model_dump_json(indent=2, exclude_none=False)
       
        
        # Build command (use '-' to read from stdin)
        cmd = [
            config.memo.bin,
        ]
        if config.memo.db_path:
            cmd.append('--db')
            cmd.append(config.memo.db_path)
        if config.memo.db_schema_path:
            cmd.append('--schema')
            cmd.append(config.memo.db_schema_path)
        cmd.extend(['fresh-paper', '--input', '-'])
        
        # Run memo CLI with stdin input
        subprocess.run(
            cmd,
            input=payload_json,
            capture_output=True,
            text=True,
            timeout=config.memo.timeout_sec,
            check=True,
        )
        
        return
            
    except subprocess.TimeoutExpired:
        logger.warning(f"memo fresh-paper timed out after {config.memo.timeout_sec}s")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling memo fresh-paper: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing memo output: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in memo fresh-paper: {e}", exc_info=True)
        return None


def get_best_clustering(
    source: str,
    period_start: str,
    period_end: str,
    config: ReaderConfig,
    top_n: int = 10,
) -> Optional[GetBestRunResponse]:
    """
    Call memo CLI get-best-run command to retrieve best clustering.
    
    Args:
        source: Snapshot source (e.g., 'hf_monthly')
        period_start: Period start date (YYYY-MM-DD)
        period_end: Period end date (YYYY-MM-DD)
        config: ReaderConfig instance
        top_n: Maximum papers per cluster to include (default: 10)
        
    Returns:
        GetBestRunResponse instance, or None if disabled/error
    """
    if not config.memo.enabled:
        return None
    
    try:
        # Build command
        cmd = [
            config.memo.bin,
        ]
        if config.memo.db_path:
            cmd.append('--db')
            cmd.append(config.memo.db_path)
        if config.memo.db_schema_path:
            cmd.append('--schema')
            cmd.append(config.memo.db_schema_path)
        cmd.extend(['get-best-run', '--source', source, '--period-start', period_start, '--period-end', period_end, '--top-n', str(top_n)])
        
        # Run memo CLI
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.memo.timeout_sec,
            check=True,
        )
        
        # Parse JSON output and create Pydantic model
        return GetBestRunResponse.model_validate_json(result.stdout)
        
    except subprocess.TimeoutExpired:
        logger.warning(f"memo get-best-run timed out after {config.memo.timeout_sec}s")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling memo get-best-run: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"memo get-best-run output: {result.stdout}")
        logger.error(f"Error parsing memo output: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in memo get-best-run: {e}", exc_info=True)
        return None


def fresh_topic(payload: dict, config: ReaderConfig) -> dict:
    """
    Stub for future topic creation.
    
    Args:
        payload: Topic payload dictionary
        config: ReaderConfig instance
        
    Returns:
        Empty dict (not implemented)
    """
    if not config.memo.enabled:
        return {}
    # TODO: Implement when topic functionality is added
    return {}


def get_topic_metadata(topic_id: str, config: ReaderConfig) -> dict:
    """
    Stub for future topic metadata retrieval.
    
    Args:
        topic_id: Topic identifier
        config: ReaderConfig instance
        
    Returns:
        Empty dict (not implemented)
    """
    if not config.memo.enabled:
        return {}
    # TODO: Implement when topic functionality is added
    return {}


def fresh_report(payload: dict, config: ReaderConfig) -> dict:
    """
    Stub for future report creation.
    
    Args:
        payload: Report payload dictionary
        config: ReaderConfig instance
        
    Returns:
        Empty dict (not implemented)
    """
    if not config.memo.enabled:
        return {}
    # TODO: Implement when report functionality is added
    return {}
