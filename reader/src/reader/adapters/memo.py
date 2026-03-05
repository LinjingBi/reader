"""Memo CLI adapter"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pydantic
from pydantic import BaseModel, Field, RootModel

from reader.pipelines.report_generation.config.config import MemoConfig
from reader.pipelines.hf_data.report import FreshPaperPayload, InjectPapersChunkPayload, InjectClustersObservationInput
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


class MemoGetReportGenerationMetadataError(Exception):
    """Exception raised when get-report-generation-metadata subcommand fails."""
    pass


class MemoGetReportGenerationSupplyError(Exception):
    """Exception raised when get-report-generation-supply subcommand fails."""
    pass


class MemoInjectPapersChunkError(Exception):
    """Exception raised when inject-papers-chunk subcommand fails."""
    pass


class MemoNewMemoryError(Exception):
    """Exception raised when new-memory subcommand fails."""
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
# new-memory command models
# ============================================================================

class ResolvedTopicInput(BaseModel):
    """Resolved topic from _resolve_report_topic step."""
    action: str  # 'create' | 'merge'
    merge_to_topic: Optional[str] = None
    new_topic_centroid_b64: str
    new_topic_weight: float
    score: float


class FrontMatterInput(BaseModel):
    """Front matter from _generate_report_front_matter step."""
    title: str
    summary: str
    keywords: List[str]


class SaveMemoryInput(BaseModel):
    """Output from _save_report_to_fs step, input for save_report_to_db."""
    report_path: str
    signature: str


class TopicResolverConfigInput(BaseModel):
    """Topic resolver config (EmbedConfig-style)."""
    topic_resolver_config_id: str
    json_payload: Dict[str, Any]


class NewMemoryRequest(BaseModel):
    """Request for new-memory command."""
    cluster_pk_hash: str
    intent_mode: str
    resolved_topic: ResolvedTopicInput
    plan: Dict[str, Any]
    front_matter: FrontMatterInput
    save_output: SaveMemoryInput
    topic_resolver_config: TopicResolverConfigInput


class NewMemoryResponse(BaseModel):
    """Response from new-memory command."""
    report_id: int


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
# get-report-generation-metadata command models
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
    rank: int  # 0 = most representative


class HistoryReport(BaseModel):
    """History report data for a topic."""
    report_id: int  # Report ID
    title: str
    summary: str
    keywords_json: Any  # Report keywords as JSON
    intent_mode: str
    declared_level: str
    depth_mode: str


class GetReportGenerationMetadataResponse(BaseModel):
    """Response from get-report-generation-metadata command."""
    new_observation: NewObservation
    new_observation_key_paper_details: Optional[List[TopPaper]] = None  # Optional Top-K papers (K≤5)
    history_reports: Optional[List[HistoryReport]] = None  # Optional top ≤3 reports for the specified topic


# ============================================================================
# get-report-generation-supply command models
# ============================================================================

class PaperSupplementRequest(BaseModel):
    """Per-paper supplement request."""
    paper_id: str
    selectors: List[str]


class ReportSupplementRequest(BaseModel):
    """Per-report supplement request."""
    report_id: int
    selectors: List[str]


class GetReportGenerationSupplyRequest(BaseModel):
    """Request for get-report-generation-supply command."""
    paper_requests: List[PaperSupplementRequest] = Field(default_factory=list)
    report_requests: List[ReportSupplementRequest] = Field(default_factory=list)


class GetReportGenerationSupplyResponse(BaseModel):
    """Response from get-report-generation-supply command.
    Matches phase2_supplement structure: paper_id/report_id -> selector -> value.
    """
    paper_supplements: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    report_supplements: Dict[str, Dict[str, str]] = Field(default_factory=dict)


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


# ============================================================================
# inject-papers-chunk command models
# ============================================================================

class InjectPapersChunkMeta(BaseModel):
    """Metadata without success field."""
    total_papers_count: int
    total_chunks_count: int


class InjectPapersChunkResponse(BaseModel):
    """Response from inject-papers-chunk command."""
    success: bool
    meta: InjectPapersChunkMeta


async def fresh_paper(payload: FreshPaperPayload, config: MemoConfig, no_details: bool = True) -> FreshPaperResponseWithDetails:
    """
    Call memo CLI fresh-paper command to ingest papers and clustering.

    Args:
        payload: FreshPaperPayload instance.
        config: MemoConfig instance
        no_details: If True (default), skip querying paper details (faster, smaller output).
            Pass False to include paper details in the response.
        
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
        
        # Run memo CLI with stdin input using async subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=payload_json.encode('utf-8')),
                timeout=config.timeout_sec
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning(f"memo fresh-paper timed out after {config.timeout_sec}s")
            raise MemoFreshPaperError(f"memo fresh-paper timed out after {config.timeout_sec}s")
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        
        stdout_text = stdout.decode('utf-8')
        stderr_text = stderr.decode('utf-8')
        
        if process.returncode != 0:
            logger.error(f"Error calling memo fresh-paper: {stderr_text}")
            raise MemoFreshPaperError(f"Error calling memo fresh-paper: {stderr_text}")
        
        # Parse JSON output and create Pydantic model
        response = FreshPaperResponseWithDetails.model_validate_json(stdout_text)
        
        # Validate success field
        if not response.success:
            raise MemoFreshPaperError(f"memo fresh-paper returned success=false: {stdout_text}")
        
        return response
            
    except MemoFreshPaperError:
        raise
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoFreshPaperError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo fresh-paper: {e}", exc_info=True)
        raise MemoFreshPaperError(f"Unexpected error in memo fresh-paper: {e}") from e


async def get_best_clustering(
    source: str,
    period_start: str,
    period_end: str,
    config: MemoConfig,
    top_n: int | None = None,
    empty_cluster_observation_only: bool = False,
) -> GetBestRunResponse:
    """
    Call memo CLI get-best-run command to retrieve best clustering.
    
    Args:
        source: Snapshot source (e.g., 'hf_monthly')
        period_start: Period start date (YYYY-MM-DD)
        period_end: Period end date (YYYY-MM-DD)
        config: MemoConfig instance
        top_n: Maximum papers per cluster to include. If None, returns all papers per cluster.
        empty_cluster_observation_only: If True, only return clusters that have no cluster_observation
            (checking via pk_hash). Default: False (returns all clusters matching period and source)
        
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
        cmd.extend(['get-best-run', '--source', source, '--period-start', period_start, '--period-end', period_end])
        if top_n is not None:
            cmd.extend(['--top-n', str(top_n)])
        if empty_cluster_observation_only:
            cmd.append('--empty-cluster-observation-only')
        
        # Run memo CLI using async subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_sec
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning(f"memo get-best-run timed out after {config.timeout_sec}s")
            raise MemoGetBestRunError(f"memo get-best-run timed out after {config.timeout_sec}s")
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        
        stdout_text = stdout.decode('utf-8')
        stderr_text = stderr.decode('utf-8')
        
        if process.returncode != 0:
            logger.error(f"Error calling memo get-best-run: {stderr_text}")
            raise MemoGetBestRunError(f"Error calling memo get-best-run: {stderr_text}")
        
        # Parse JSON output and create Pydantic model
        return GetBestRunResponse.model_validate_json(stdout_text)
        
    except MemoGetBestRunError:
        raise
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoGetBestRunError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo get-best-run: {e}", exc_info=True)
        raise MemoGetBestRunError(f"Unexpected error in memo get-best-run: {e}") from e


async def inject_clusters_observation(payload: InjectClustersObservationInput, config: MemoConfig) -> None:
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
        
        # Run memo CLI with stdin input using async subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=payload_json.encode('utf-8')),
                timeout=config.timeout_sec
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning(f"memo inject-clusters-observation timed out after {config.timeout_sec}s")
            raise MemoInjectClustersObservationError(f"memo inject-clusters-observation timed out after {config.timeout_sec}s")
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        
        stderr_text = stderr.decode('utf-8')
        
        if process.returncode != 0:
            logger.error(f"Error calling memo inject-clusters-observation: {stderr_text}")
            raise MemoInjectClustersObservationError(f"Error calling memo inject-clusters-observation: {stderr_text}")
        
        return None
        
    except MemoInjectClustersObservationError:
        raise
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoInjectClustersObservationError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo inject-clusters-observation: {e}", exc_info=True)
        raise MemoInjectClustersObservationError(f"Unexpected error in memo inject-clusters-observation: {e}") from e


async def get_clusters_observation(
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

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_sec
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning(f"memo get-clusters-observation timed out after {config.timeout_sec}s")
            raise MemoGetClustersObservationError(f"memo get-clusters-observation timed out after {config.timeout_sec}s")
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        stdout_text = stdout.decode('utf-8')
        stderr_text = stderr.decode('utf-8')

        if process.returncode != 0:
            logger.error(f"Error calling memo get-clusters-observation: {stderr_text}")
            raise MemoGetClustersObservationError(f"Error calling memo get-clusters-observation: {stderr_text}")

        return GetClusterObservationResponse.model_validate_json(stdout_text).root

    except MemoGetClustersObservationError:
        raise
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoGetClustersObservationError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo get-clusters-observation: {e}", exc_info=True)
        raise MemoGetClustersObservationError(f"Unexpected error in memo get-clusters-observation: {e}") from e


async def start_report_job(
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

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_sec
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning(f"memo start-report-job timed out after {config.timeout_sec}s")
            raise MemoStartReportJobError(f"memo start-report-job timed out after {config.timeout_sec}s")
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        stdout_text = stdout.decode('utf-8')
        stderr_text = stderr.decode('utf-8')

        if process.returncode != 0:
            logger.error(f"Error calling memo start-report-job: {stderr_text}")
            raise MemoStartReportJobError(f"Error calling memo start-report-job: {stderr_text}")

        return StartReportJobResponse.model_validate_json(stdout_text)

    except MemoStartReportJobError:
        raise
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoStartReportJobError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo start-report-job: {e}", exc_info=True)
        raise MemoStartReportJobError(f"Unexpected error in memo start-report-job: {e}") from e


async def new_memory(
    payload: NewMemoryRequest,
    config: MemoConfig,
) -> NewMemoryResponse:
    """
    Call memo CLI new-memory command to persist report generation results to the database.

    Args:
        payload: NewMemoryRequest with cluster_pk_hash, resolved_topic, plan, front_matter, save_output, etc.
        config: MemoConfig instance

    Returns:
        NewMemoryResponse with report_id

    Raises:
        MemoNewMemoryError: If the subcommand execution fails
    """
    try:
        payload_json = payload.model_dump_json(exclude_none=True)

        cmd = [
            config.bin,
        ]
        if config.db_path:
            cmd.append('--db')
            cmd.append(config.db_path)
        if config.db_schema_path:
            cmd.append('--schema')
            cmd.append(config.db_schema_path)
        cmd.extend(['new-memory', '--input', '-'])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=payload_json.encode('utf-8')),
                timeout=config.timeout_sec
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning(f"memo new-memory timed out after {config.timeout_sec}s")
            raise MemoNewMemoryError(f"memo new-memory timed out after {config.timeout_sec}s")
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        stdout_text = stdout.decode('utf-8')
        stderr_text = stderr.decode('utf-8')

        if process.returncode != 0:
            logger.error(f"Error calling memo new-memory: {stderr_text}")
            raise MemoNewMemoryError(f"Error calling memo new-memory: {stderr_text}")

        return NewMemoryResponse.model_validate_json(stdout_text)

    except MemoNewMemoryError:
        raise
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoNewMemoryError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo new-memory: {e}", exc_info=True)
        raise MemoNewMemoryError(f"Unexpected error in memo new-memory: {e}") from e


async def get_topic_resolver_metadata(
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

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_sec
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning(f"memo get-topic-resolver-metadata timed out after {config.timeout_sec}s")
            raise MemoGetTopicResolverMetadataError(f"memo get-topic-resolver-metadata timed out after {config.timeout_sec}s")
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        stdout_text = stdout.decode('utf-8')
        stderr_text = stderr.decode('utf-8')

        if process.returncode != 0:
            logger.error(f"Error calling memo get-topic-resolver-metadata: {stderr_text}")
            raise MemoGetTopicResolverMetadataError(f"Error calling memo get-topic-resolver-metadata: {stderr_text}")

        return GetTopicResolverMetadataResponse.model_validate_json(stdout_text)

    except MemoGetTopicResolverMetadataError:
        raise
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoGetTopicResolverMetadataError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo get-topic-resolver-metadata: {e}", exc_info=True)
        raise MemoGetTopicResolverMetadataError(f"Unexpected error in memo get-topic-resolver-metadata: {e}") from e


async def get_report_generation_metadata(
    cluster_pk_hash: str,
    config: MemoConfig,
    topic_id: Optional[str] = None,
    add_top_papers: bool = False,
) -> GetReportGenerationMetadataResponse:
    """
    Call memo CLI get-report-generation-metadata command to retrieve report generation metadata.

    Args:
        cluster_pk_hash: Cluster pk_hash (primary key hash from cluster table)
        config: MemoConfig instance
        topic_id: Optional topic_id (as string) to include top ≤3 reports for that topic
        add_top_papers: Whether to include Top-K papers (K≤5) for the cluster

    Returns:
        GetReportGenerationMetadataResponse instance

    Raises:
        MemoGetReportGenerationMetadataError: If the subcommand execution fails
    """
    try:
        cmd = [
            config.bin,
        ]
        if config.db_path:
            cmd.append('--db')
            cmd.append(config.db_path)
        if config.db_schema_path:
            cmd.append('--schema')
            cmd.append(config.db_schema_path)
        cmd.extend(['get-report-generation-metadata', '--cluster-pk-hash', cluster_pk_hash])

        if topic_id is not None:
            cmd.extend(['--add-topic-reports', str(topic_id)])
        if add_top_papers:
            cmd.append('--add-top-papers')

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_sec
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning(f"memo get-report-generation-metadata timed out after {config.timeout_sec}s")
            raise MemoGetReportGenerationMetadataError(f"memo get-report-generation-metadata timed out after {config.timeout_sec}s")
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        stdout_text = stdout.decode('utf-8')
        stderr_text = stderr.decode('utf-8')

        if process.returncode != 0:
            logger.error(f"Error calling memo get-report-generation-metadata: {stderr_text}")
            raise MemoGetReportGenerationMetadataError(f"Error calling memo get-report-generation-metadata: {stderr_text}")

        return GetReportGenerationMetadataResponse.model_validate_json(stdout_text)

    except MemoGetReportGenerationMetadataError:
        raise
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoGetReportGenerationMetadataError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo get-report-generation-metadata: {e}", exc_info=True)
        raise MemoGetReportGenerationMetadataError(f"Unexpected error in memo get-report-generation-metadata: {e}") from e


async def get_report_generation_supply(
    request: "GetReportGenerationSupplyRequest",
    config: MemoConfig,
) -> "GetReportGenerationSupplyResponse":
    """
    Call memo CLI get-report-generation-supply command to fetch evidence (paper chunks
    and history report fields) for evidence gaps from planner output.

    Args:
        request: GetReportGenerationSupplyRequest instance
        config: MemoConfig instance

    Returns:
        GetReportGenerationSupplyResponse instance

    Raises:
        MemoGetReportGenerationSupplyError: If the subcommand execution fails
    """
    try:
        payload_json = request.model_dump_json(indent=2, exclude_none=False)

        cmd = [
            config.bin,
        ]
        if config.db_path:
            cmd.append('--db')
            cmd.append(config.db_path)
        if config.db_schema_path:
            cmd.append('--schema')
            cmd.append(config.db_schema_path)
        cmd.extend(['get-report-generation-supply', '--input', '-'])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=payload_json.encode('utf-8')),
                timeout=config.timeout_sec
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning(f"memo get-report-generation-supply timed out after {config.timeout_sec}s")
            raise MemoGetReportGenerationSupplyError(
                f"memo get-report-generation-supply timed out after {config.timeout_sec}s"
            )
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        stdout_text = stdout.decode('utf-8')
        stderr_text = stderr.decode('utf-8')

        if process.returncode != 0:
            logger.error(f"Error calling memo get-report-generation-supply: {stderr_text}")
            raise MemoGetReportGenerationSupplyError(
                f"Error calling memo get-report-generation-supply: {stderr_text}"
            )

        return GetReportGenerationSupplyResponse.model_validate_json(stdout_text)

    except MemoGetReportGenerationSupplyError:
        raise
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoGetReportGenerationSupplyError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo get-report-generation-supply: {e}", exc_info=True)
        raise MemoGetReportGenerationSupplyError(
            f"Unexpected error in memo get-report-generation-supply: {e}"
        ) from e


async def inject_papers_chunk(
    payload: InjectPapersChunkPayload,
    config: MemoConfig,
) -> InjectPapersChunkResponse:
    """
    Call memo CLI inject-papers-chunk command to inject paper chunks.
    
    Args:
        payload: InjectPapersChunkPayload instance containing lib_config and papers
        config: MemoConfig instance
        
    Returns:
        InjectPapersChunkResponse instance
        
    Raises:
        MemoInjectPapersChunkError: If the subcommand execution fails
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
        cmd.extend(['inject-papers-chunk', '--input', '-'])
        
        # Run memo CLI with stdin input using async subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=payload_json.encode('utf-8')),
                timeout=config.timeout_sec
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning(f"memo inject-papers-chunk timed out after {config.timeout_sec}s")
            raise MemoInjectPapersChunkError(f"memo inject-papers-chunk timed out after {config.timeout_sec}s")
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        
        stdout_text = stdout.decode('utf-8')
        stderr_text = stderr.decode('utf-8')
        
        if process.returncode != 0:
            logger.error(f"Error calling memo inject-papers-chunk: {stderr_text}")
            raise MemoInjectPapersChunkError(f"Error calling memo inject-papers-chunk: {stderr_text}")
        
        # Parse JSON output and create Pydantic model
        response = InjectPapersChunkResponse.model_validate_json(stdout_text)
        
        # Validate success field
        if not response.success:
            raise MemoInjectPapersChunkError(f"memo inject-papers-chunk returned success=false: {stdout_text}")
        
        return response
            
    except MemoInjectPapersChunkError:
        raise
    except pydantic.ValidationError as e:
        logger.error(f"Error validating memo output: {e}")
        raise MemoInjectPapersChunkError(f"Error validating memo output: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in memo inject-papers-chunk: {e}", exc_info=True)
        raise MemoInjectPapersChunkError(f"Unexpected error in memo inject-papers-chunk: {e}") from e