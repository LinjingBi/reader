"""Pydantic models for HF data pipeline outputs/reports"""

from typing import List
from pydantic import BaseModel, Field, computed_field


class ClusterMemberInput(BaseModel):
    """Cluster member entry with paper ID, rank, and similarity to centroid."""
    paper_id: str = Field(..., description="Formatted paper ID (e.g., 'hf:2501.12948')")
    rank_in_cluster: int = Field(..., description="Rank of paper within cluster group")
    sim_to_centroid: float = Field(..., description="Similarity to cluster centroid")


class PaperInput(BaseModel):
    """Paper entry in the payload."""
    raw_paper_id: str = Field(..., exclude=True, description="Raw paper ID (excluded from serialization)")
    title: str = Field(..., description="Paper title")
    summary: str = Field(..., description="Paper summary/abstract")
    keywords: List[str] = Field(..., description="List of keywords")
    url: str = Field(default="", description="Paper URL")
    published_at: str = Field(default="", description="Publication date")
    
    def __init__(self, raw_paper_id: str, **kwargs):
        """Initialize PaperInput with raw paper ID."""
        super().__init__(raw_paper_id=raw_paper_id, **kwargs)
    
    @computed_field
    @property
    def paper_id(self) -> str:
        """Format paper ID with 'hf:' prefix."""
        return f"hf:{self.raw_paper_id}"


class EmbedConfigPayload(BaseModel):
    """Embedding configuration payload."""
    model_name: str = Field(..., description="Embedding model name")
    mode: str = Field(..., description="Embedding mode")
    top_n_keywords: int = Field(..., description="Number of top keywords used")


class EmbedConfig(BaseModel):
    """Embedding configuration with dynamic config ID."""
    json_payload: EmbedConfigPayload = Field(..., description="Embedding configuration payload")
    
    @computed_field
    @property
    def embed_config_id(self) -> str:
        """Get embed_config_id from algo_lib.embedding version."""
        try:
            from algo_lib.embedding import __version__ as embed_version
            return f"algo_lib.embedding|{embed_version}"
        except ImportError:
            raise ValueError("algo_lib.embedding is not versioned")


class ClusterConfigPayload(BaseModel):
    """Clustering configuration payload."""
    k: int = Field(..., description="Number of clusters")
    seed: int = Field(..., description="Random seed")
    algorithm: str = Field(default="kmeans", description="Clustering algorithm")


class ClusterConfig(BaseModel):
    """Clustering configuration with dynamic config ID."""
    json_payload: ClusterConfigPayload = Field(..., description="Clustering configuration payload")
    
    @computed_field
    @property
    def cluster_config_id(self) -> str:
        """Get cluster_config_id from algo_lib.clustering version."""
        try:
            from algo_lib.clustering import __version__ as cluster_version
            return f"algo_lib.clustering|{cluster_version}"
        except ImportError:
            raise ValueError("algo_lib.clustering is not versioned")


class ClusterInput(BaseModel):
    """Cluster entry in the payload."""
    cluster_index: int = Field(..., description="Cluster index")
    size: int = Field(..., description="Number of papers in cluster")
    cohesion: float = Field(..., description="Cluster cohesion score")
    centroid_b64: str = Field(..., description="Cluster centroid as base64-encoded float32 bytes")
    members: List[ClusterMemberInput] = Field(..., description="List of cluster members")


class FreshPaperPayload(BaseModel):
    """Fresh paper payload report class."""
    source: str = Field(default="hf_monthly", description="Data source")
    period_start: str = Field(..., description="Period start date (YYYY-MM-DD)")
    period_end: str = Field(..., description="Period end date (YYYY-MM-DD)")
    raw_json: str = Field(default="", description="Optional raw JSON string")
    embed_config: EmbedConfig = Field(..., description="Embedding configuration")
    cluster_config: ClusterConfig = Field(..., description="Clustering configuration")
    papers: List[PaperInput] = Field(..., description="List of papers")
    clusters: List[ClusterInput] = Field(..., description="List of clusters")


class PaperChunkLibConfigPayload(BaseModel):
    """Paper chunk library configuration payload."""
    version: int = Field(..., description="rules version number")
    compiled_regex_version: int = Field(..., description="rules compiled regex version")


class PaperChunkLibConfig(BaseModel):
    """Paper chunk library configuration with dynamic config ID."""
    json_payload: PaperChunkLibConfigPayload = Field(..., description="Paper chunk rules config")
    
    @computed_field
    @property
    def lib_config_id(self) -> str:
        """Get lib_config_id from algo_lib.paperchunk version."""
        try:
            from algo_lib.paperchunk import __version__ as paperchunk_version
            return f"algo_lib.paperchunk|{paperchunk_version}"
        except ImportError:
            raise ValueError("algo_lib.paperchunk is not versioned")


class ChunkEntry(BaseModel):
    """Chunk entry with selector, text, and score."""
    selector_id: str = Field(..., description="Selector name (e.g., 'summary', 'method')")
    text_id: str = Field(..., description="Text ID from ScoreOutput.text_table keys")
    text: str = Field(..., description="Text content from ScoreOutput.text_table values")
    score: float = Field(..., description="Score from ScoreOutput.sel2texts_score_table")


class PaperChunkData(BaseModel):
    """Per-paper chunk data."""
    paper_id: str = Field(..., description="Paper ID")
    status: str = Field(..., description="Paper status: 'ok' | 'partial' | 'error'")
    chunks: List[ChunkEntry] = Field(..., description="List of chunks (empty if status == 'error')")


class InjectPapersChunkPayload(BaseModel):
    """Payload for memo inject-papers-chunk command."""
    lib_config: PaperChunkLibConfig = Field(..., description="Library configuration")
    papers: List[PaperChunkData] = Field(..., description="List of papers with chunks")

