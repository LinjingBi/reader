"""Memo CLI adapter"""

import json
import subprocess
from typing import List, Optional

import pydantic
from pydantic import BaseModel

from reader.config import ReaderConfig
from reader.pipelines.report import FreshPaperPayload, InjectClustersObservationInput, InjectClustersObservationResponse
from reader.logging.logging_setup import get_logger

logger = get_logger()


# Exception classes for memo CLI subcommands
class MemoFreshPaperError(Exception):
    """Exception raised when fresh-paper subcommand fails."""
    pass


class MemoGetBestRunError(Exception):
    """Exception raised when get-best-run subcommand fails."""
    pass


class MemoInjectClustersObservationError(Exception):
    """Exception raised when inject-clusters-observation subcommand fails."""
    pass


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
    pk_hash: str


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
        None on success, or None if memo is disabled
        
    Raises:
        MemoFreshPaperError: If the subcommand execution fails
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
            
    except subprocess.TimeoutExpired as e:
        logger.warning(f"memo fresh-paper timed out after {config.memo.timeout_sec}s")
        raise MemoFreshPaperError(f"memo fresh-paper timed out after {config.memo.timeout_sec}s") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling memo fresh-paper: {e.stderr}")
        raise MemoFreshPaperError(f"Error calling memo fresh-paper: {e.stderr}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo fresh-paper: {e}", exc_info=True)
        raise MemoFreshPaperError(f"Unexpected error in memo fresh-paper: {e}") from e


def get_best_clustering(
    source: str,
    period_start: str,
    period_end: str,
    config: ReaderConfig,
    top_n: int = 10,
) -> GetBestRunResponse:
    """
    Call memo CLI get-best-run command to retrieve best clustering.
    
    Args:
        source: Snapshot source (e.g., 'hf_monthly')
        period_start: Period start date (YYYY-MM-DD)
        period_end: Period end date (YYYY-MM-DD)
        config: ReaderConfig instance
        top_n: Maximum papers per cluster to include (default: 10)
        
    Returns:
        GetBestRunResponse instance, or None if memo is disabled
        
    Raises:
        MemoGetBestRunError: If the subcommand execution fails
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
        
    except subprocess.TimeoutExpired as e:
        logger.warning(f"memo get-best-run timed out after {config.memo.timeout_sec}s")
        raise MemoGetBestRunError(f"memo get-best-run timed out after {config.memo.timeout_sec}s") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling memo get-best-run: {e.stderr}")
        raise MemoGetBestRunError(f"Error calling memo get-best-run: {e.stderr}") from e
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoGetBestRunError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo get-best-run: {e}", exc_info=True)
        raise MemoGetBestRunError(f"Unexpected error in memo get-best-run: {e}") from e


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


def inject_clusters_observation(payload: InjectClustersObservationInput, config: ReaderConfig) -> InjectClustersObservationResponse:
    """
    Call memo CLI inject-clusters-observation command to inject cluster observations.
    
    Args:
        payload: InjectClustersObservationInput dict mapping pk_hash to ClusterObservation
        config: ReaderConfig instance
        
    Returns:
        InjectClustersObservationResponse instance, or None if memo is disabled
        
    Raises:
        MemoInjectClustersObservationError: If the subcommand execution fails
    """
    if not config.memo.enabled:
        return None
    
    try:
        # Convert payload dict to JSON string
        # Each value in the dict is a ClusterObservation (Pydantic model)
        payload_dict = {}
        for pk_hash, observation in payload.items():
            payload_dict[pk_hash] = observation.model_dump()
        
        payload_json = json.dumps(payload_dict, indent=2, default=str)
        
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
        cmd.extend(['inject-clusters-observation', '--input', '-'])
        
        # Run memo CLI with stdin input
        result = subprocess.run(
            cmd,
            input=payload_json,
            capture_output=True,
            text=True,
            timeout=config.memo.timeout_sec,
            check=True,
        )
        
        # Parse JSON output and create Pydantic model
        return InjectClustersObservationResponse.model_validate_json(result.stdout)
        
    except subprocess.TimeoutExpired as e:
        logger.warning(f"memo inject-clusters-observation timed out after {config.memo.timeout_sec}s")
        raise MemoInjectClustersObservationError(f"memo inject-clusters-observation timed out after {config.memo.timeout_sec}s") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling memo inject-clusters-observation: {e.stderr}")
        raise MemoInjectClustersObservationError(f"Error calling memo inject-clusters-observation: {e.stderr}") from e
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoInjectClustersObservationError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo inject-clusters-observation: {e}", exc_info=True)
        raise MemoInjectClustersObservationError(f"Unexpected error in memo inject-clusters-observation: {e}") from e
