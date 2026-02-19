"""Memo CLI adapter"""

import json
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

import pydantic
from pydantic import BaseModel, RootModel

from reader.config import MemoConfig
from reader.pipelines.hf_data.report import FreshPaperPayload
from reader.pipelines.report import InjectClustersObservationInput
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


class MemoGetClustersObservationError(Exception):
    """Exception raised when get-clusters-observation subcommand fails."""
    pass


class MemoStartReportJobError(Exception):
    """Exception raised when start-report-job subcommand fails."""
    pass


class MemoGetTopicResolverMetadataError(Exception):
    """Exception raised when get-topic-resolver-metadata subcommand fails."""
    pass


class MemoGetReportPlannerMetadataError(Exception):
    """Exception raised when get-report-planner-metadata subcommand fails."""
    pass


# Pydantic response models matching Rust CLI contracts

# ============================================================================
# get-best-run command models
# ============================================================================

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
    period_start: str # YYYY-MM-DD
    period_end: str # YYYY-MM-DD
    embed_config_id: str
    cluster_config_id: str
    clusters: List[ClusterCard]


# ============================================================================
# get-clusters-observation command models
# ============================================================================

class ClusterObservationData(BaseModel):
    """Observation data for a single cluster."""
    observation_created_time: datetime  # YYYY-MM-DD format, parsed by Pydantic
    json_payload: Any
    cluster_period_start: datetime  # YYYY-MM-DD format, parsed by Pydantic
    cluster_period_end: datetime  # YYYY-MM-DD format, parsed by Pydantic


class GetClusterObservationResponse(RootModel[Dict[str, ClusterObservationData]]):
    """Response from get-clusters-observation command.
    
    Maps pk_hash to ClusterObservationData.
    Access the dictionary via .root attribute.
    """
    pass


# ============================================================================
# start-report-job command models
# ============================================================================

class StartReportJobResponse(BaseModel):
    """Response from start-report-job command."""
    status: str  # 'running' | 'done' | 'error'
    new_job: bool
    report_id: Optional[str] = None  # Only present when status is 'done'
    message: str


# ============================================================================
# get-topic-resolver-metadata command models
# ============================================================================

class TopicCentroid(BaseModel):
    """Topic centroid data matching the Rust TopicCentroid model."""
    id: str  # Topic ID (from topic_id)
    centroid_b64: str  # Topic centroid as base64-encoded float32 bytes
    centroid_weight: float  # Topic centroid weight (must be positive)


class ClusterMetadata(BaseModel):
    """Cluster metadata for topic resolver."""
    centroid: str  # Cluster centroid as base64-encoded float32 bytes
    centroid_weight: float  # Cluster centroid weight (cluster size)


class GetTopicResolverMetadataResponse(BaseModel):
    """Response from get-topic-resolver-metadata command."""
    topics: List[TopicCentroid]  # List of topics with their centroid data
    cluster: ClusterMetadata  # Cluster metadata


# ============================================================================
# get-report-planner-metadata command models
# ============================================================================

class NewObservation(BaseModel):
    """New observation data from cluster observation."""
    name: str  # Cluster observation title
    summary: str  # Cluster observation summary
    keywords: List[str]  # Keywords from cluster observation
    key_paper_keywords: Dict[str, List[str]]  # Keywords from top ≤5 papers, keyed by paper_id


class TopPaper(BaseModel):
    """Top paper data for the cluster."""
    paper_id: str
    title: str
    summary: str
    keywords: List[str]
    rank_in_cluster: int  # 0 = most representative
    sim_to_centroid: Optional[float] = None


class HistoryReport(BaseModel):
    """History report data for a topic."""
    report_id: int  # Report ID
    title: str
    summary: str
    keywords_json: Any  # Report keywords as JSON
    depth_context_json: Any  # Report depth context as JSON


class GetReportPlannerMetadataResponse(BaseModel):
    """Response from get-report-planner-metadata command."""
    new_observation: NewObservation
    top_papers_from_new_observation: Optional[List[TopPaper]] = None  # Optional Top-K papers (K≤5)
    history_reports: Optional[List[HistoryReport]] = None  # Optional top ≤3 reports for the specified topic


# ============================================================================
# fresh-paper command models
# ============================================================================

class PaperOutput(BaseModel):
    """Paper output in details section."""
    paper_id: str
    rank_in_cluster: int
    paper_url: str


class FreshPaperMeta(BaseModel):
    """Metadata without success field."""
    source: str
    period_start: str  # YYYY-MM-DD
    period_end: str  # YYYY-MM-DD
    papers_count: int
    clusters_count: int


class FreshPaperResponseWithDetails(BaseModel):
    """Response from fresh-paper command."""
    success: bool
    meta: FreshPaperMeta
    details: Optional[Dict[str, List[PaperOutput]]] = None  # Optional details mapping pk_hash to papers


def fresh_paper(payload: FreshPaperPayload, config: MemoConfig, no_details: bool = False) -> FreshPaperResponseWithDetails:
    """
    Call memo CLI fresh-paper command to ingest papers and clustering.
    
    Args:
        payload: FreshPaperPayload instance.
        config: MemoConfig instance
        no_details: If True, skip querying paper details (faster, smaller output)
        
    Returns:
        FreshPaperResponseWithDetails instance
        
    Raises:
        MemoFreshPaperError: If the subcommand execution fails
    """
    try:
        # Convert payload to JSON string
        payload_json = payload.model_dump_json(indent=2, exclude_none=False)
       
        
        # Build command (use '-' to read from stdin)
        cmd = [
            config.bin,
        ]
        if config.db_path:
            cmd.append('--db')
            cmd.append(config.db_path)
        if config.db_schema_path:
            cmd.append('--schema')
            cmd.append(config.db_schema_path)
        cmd.extend(['fresh-paper', '--input', '-'])
        if no_details:
            cmd.append('--no-details')
        
        # Run memo CLI with stdin input
        result = subprocess.run(
            cmd,
            input=payload_json,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
            check=True,
        )
        
        # Parse JSON output and create Pydantic model
        response = FreshPaperResponseWithDetails.model_validate_json(result.stdout)
        
        # Validate success field
        if not response.success:
            raise MemoFreshPaperError(f"memo fresh-paper returned success=false: {result.stdout}")
        
        return response
            
    except subprocess.TimeoutExpired as e:
        logger.warning(f"memo fresh-paper timed out after {config.timeout_sec}s")
        raise MemoFreshPaperError(f"memo fresh-paper timed out after {config.timeout_sec}s") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling memo fresh-paper: {e.stderr}")
        raise MemoFreshPaperError(f"Error calling memo fresh-paper: {e.stderr}") from e
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoFreshPaperError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo fresh-paper: {e}", exc_info=True)
        raise MemoFreshPaperError(f"Unexpected error in memo fresh-paper: {e}") from e


def get_best_clustering(
    source: str,
    period_start: str,
    period_end: str,
    config: MemoConfig,
    top_n: int = 10,
) -> GetBestRunResponse:
    """
    Call memo CLI get-best-run command to retrieve best clustering.
    
    Args:
        source: Snapshot source (e.g., 'hf_monthly')
        period_start: Period start date (YYYY-MM-DD)
        period_end: Period end date (YYYY-MM-DD)
        config: MemoConfig instance
        top_n: Maximum papers per cluster to include (default: 10)
        
    Returns:
        GetBestRunResponse instance
        
    Raises:
        MemoGetBestRunError: If the subcommand execution fails
    """
    try:
        # Build command
        cmd = [
            config.bin,
        ]
        if config.db_path:
            cmd.append('--db')
            cmd.append(config.db_path)
        if config.db_schema_path:
            cmd.append('--schema')
            cmd.append(config.db_schema_path)
        cmd.extend(['get-best-run', '--source', source, '--period-start', period_start, '--period-end', period_end, '--top-n', str(top_n)])
        
        # Run memo CLI
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
            check=True,
        )
        
        # Parse JSON output and create Pydantic model
        return GetBestRunResponse.model_validate_json(result.stdout)
        
    except subprocess.TimeoutExpired as e:
        logger.warning(f"memo get-best-run timed out after {config.timeout_sec}s")
        raise MemoGetBestRunError(f"memo get-best-run timed out after {config.timeout_sec}s") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling memo get-best-run: {e.stderr}")
        raise MemoGetBestRunError(f"Error calling memo get-best-run: {e.stderr}") from e
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoGetBestRunError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo get-best-run: {e}", exc_info=True)
        raise MemoGetBestRunError(f"Unexpected error in memo get-best-run: {e}") from e


def inject_clusters_observation(payload: InjectClustersObservationInput, config: MemoConfig) -> None:
    """
    Call memo CLI inject-clusters-observation command to inject cluster observations.
    
    Args:
        payload: InjectClustersObservationInput dict mapping pk_hash to ClusterObservation
        config: MemoConfig instance
        
    Returns: none
        
    Raises:
        MemoInjectClustersObservationError: If the subcommand execution fails
    """
    try:
        # Convert payload dict to JSON string
        # Each value in the dict is a ClusterObservation (Pydantic model)
        payload_dict = {}
        for pk_hash, observation in payload.items():
            payload_dict[pk_hash] = observation.model_dump()
        
        payload_json = json.dumps(payload_dict, indent=2, default=str)
        
        # Build command (use '-' to read from stdin)
        cmd = [
            config.bin,
        ]
        if config.db_path:
            cmd.append('--db')
            cmd.append(config.db_path)
        if config.db_schema_path:
            cmd.append('--schema')
            cmd.append(config.db_schema_path)
        cmd.extend(['inject-clusters-observation', '--input', '-'])
        
        # Run memo CLI with stdin input
        subprocess.run(
            cmd,
            input=payload_json,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
            check=True,
        )
        
        return None
        
    except subprocess.TimeoutExpired as e:
        logger.warning(f"memo inject-clusters-observation timed out after {config.timeout_sec}s")
        raise MemoInjectClustersObservationError(f"memo inject-clusters-observation timed out after {config.timeout_sec}s") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling memo inject-clusters-observation: {e.stderr}")
        raise MemoInjectClustersObservationError(f"Error calling memo inject-clusters-observation: {e.stderr}") from e
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoInjectClustersObservationError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo inject-clusters-observation: {e}", exc_info=True)
        raise MemoInjectClustersObservationError(f"Unexpected error in memo inject-clusters-observation: {e}") from e


def get_clusters_observation(
    source: str,
    period_start: str,
    period_end: str,
    config: MemoConfig,
) -> Dict[str, ClusterObservationData]:
    """
    Call memo CLI get-clusters-observation command to retrieve cluster observations.
    
    Args:
        source: Snapshot source (e.g., 'hf_monthly')
        period_start: Period start date (YYYY-MM-DD)
        period_end: Period end date (YYYY-MM-DD)
        config: MemoConfig instance
        
    Returns:
        Dict mapping pk_hash to ClusterObservationData
        
    Raises:
        MemoGetClustersObservationError: If the subcommand execution fails
    """
    try:
        # Build command
        cmd = [
            config.bin,
        ]
        if config.db_path:
            cmd.append('--db')
            cmd.append(config.db_path)
        if config.db_schema_path:
            cmd.append('--schema')
            cmd.append(config.db_schema_path)
        cmd.extend(['get-clusters-observation', '--source', source, '--period-start', period_start, '--period-end', period_end])
        
        # Run memo CLI
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
            check=True,
        )
        
        # Parse JSON output and create Pydantic model, then return the dict
        return GetClusterObservationResponse.model_validate_json(result.stdout).root
        
    except subprocess.TimeoutExpired as e:
        logger.warning(f"memo get-clusters-observation timed out after {config.timeout_sec}s")
        raise MemoGetClustersObservationError(f"memo get-clusters-observation timed out after {config.timeout_sec}s") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling memo get-clusters-observation: {e.stderr}")
        raise MemoGetClustersObservationError(f"Error calling memo get-clusters-observation: {e.stderr}") from e
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoGetClustersObservationError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo get-clusters-observation: {e}", exc_info=True)
        raise MemoGetClustersObservationError(f"Unexpected error in memo get-clusters-observation: {e}") from e


def start_report_job(
    cluster_pk_hash: str,
    config: MemoConfig,
) -> StartReportJobResponse:
    """
    Call memo CLI start-report-job command to start a report generation job for a cluster.
    
    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        config: MemoConfig instance
        
    Returns:
        StartReportJobResponse instance
        
    Raises:
        MemoStartReportJobError: If the subcommand execution fails
    """
    try:
        # Build command
        cmd = [
            config.bin,
        ]
        if config.db_path:
            cmd.append('--db')
            cmd.append(config.db_path)
        if config.db_schema_path:
            cmd.append('--schema')
            cmd.append(config.db_schema_path)
        cmd.extend(['start-report-job', '--cluster-pk-hash', cluster_pk_hash])
        
        # Run memo CLI
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
            check=True,
        )
        
        # Parse JSON output and create Pydantic model
        return StartReportJobResponse.model_validate_json(result.stdout)
        
    except subprocess.TimeoutExpired as e:
        logger.warning(f"memo start-report-job timed out after {config.timeout_sec}s")
        raise MemoStartReportJobError(f"memo start-report-job timed out after {config.timeout_sec}s") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling memo start-report-job: {e.stderr}")
        raise MemoStartReportJobError(f"Error calling memo start-report-job: {e.stderr}") from e
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoStartReportJobError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo start-report-job: {e}", exc_info=True)
        raise MemoStartReportJobError(f"Unexpected error in memo start-report-job: {e}") from e


def get_topic_resolver_metadata(
    cluster_pk_hash: str,
    config: MemoConfig,
) -> GetTopicResolverMetadataResponse:
    """
    Call memo CLI get-topic-resolver-metadata command to retrieve topic resolver metadata.
    
    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        config: MemoConfig instance
        
    Returns:
        GetTopicResolverMetadataResponse instance
        
    Raises:
        MemoGetTopicResolverMetadataError: If the subcommand execution fails
    """
    try:
        # Build command
        cmd = [
            config.bin,
        ]
        if config.db_path:
            cmd.append('--db')
            cmd.append(config.db_path)
        if config.db_schema_path:
            cmd.append('--schema')
            cmd.append(config.db_schema_path)
        cmd.extend(['get-topic-resolver-metadata', '--cluster-pk-hash', cluster_pk_hash])
        
        # Run memo CLI
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
            check=True,
        )
        
        # Parse JSON output and create Pydantic model
        return GetTopicResolverMetadataResponse.model_validate_json(result.stdout)
        
    except subprocess.TimeoutExpired as e:
        logger.warning(f"memo get-topic-resolver-metadata timed out after {config.timeout_sec}s")
        raise MemoGetTopicResolverMetadataError(f"memo get-topic-resolver-metadata timed out after {config.timeout_sec}s") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling memo get-topic-resolver-metadata: {e.stderr}")
        raise MemoGetTopicResolverMetadataError(f"Error calling memo get-topic-resolver-metadata: {e.stderr}") from e
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoGetTopicResolverMetadataError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo get-topic-resolver-metadata: {e}", exc_info=True)
        raise MemoGetTopicResolverMetadataError(f"Unexpected error in memo get-topic-resolver-metadata: {e}") from e


def get_report_planner_metadata(
    cluster_pk_hash: str,
    config: MemoConfig,
    topic_id: Optional[str] = None,
    add_top_papers: bool = False,
) -> GetReportPlannerMetadataResponse:
    """
    Call memo CLI get-report-planner-metadata command to retrieve report planner metadata.
    
    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        config: MemoConfig instance
        topic_id: Optional topic_id (as string) to include top ≤3 reports for that topic
        add_top_papers: Whether to include Top-K papers (K≤5) for the cluster
        
    Returns:
        GetReportPlannerMetadataResponse instance
        
    Raises:
        MemoGetReportPlannerMetadataError: If the subcommand execution fails
    """
    try:
        # Build command
        cmd = [
            config.bin,
        ]
        if config.db_path:
            cmd.append('--db')
            cmd.append(config.db_path)
        if config.db_schema_path:
            cmd.append('--schema')
            cmd.append(config.db_schema_path)
        cmd.extend(['get-report-planner-metadata', '--cluster-pk-hash', cluster_pk_hash])
        
        if topic_id is not None:
            cmd.extend(['--add-topic-reports', str(topic_id)])
        if add_top_papers:
            cmd.append('--add-top-papers')
        
        # Run memo CLI
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
            check=True,
        )
        
        # Parse JSON output and create Pydantic model
        return GetReportPlannerMetadataResponse.model_validate_json(result.stdout)
        
    except subprocess.TimeoutExpired as e:
        logger.warning(f"memo get-report-planner-metadata timed out after {config.timeout_sec}s")
        raise MemoGetReportPlannerMetadataError(f"memo get-report-planner-metadata timed out after {config.timeout_sec}s") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling memo get-report-planner-metadata: {e.stderr}")
        raise MemoGetReportPlannerMetadataError(f"Error calling memo get-report-planner-metadata: {e.stderr}") from e
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoGetReportPlannerMetadataError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo get-report-planner-metadata: {e}", exc_info=True)
        raise MemoGetReportPlannerMetadataError(f"Unexpected error in memo get-report-planner-metadata: {e}") from e