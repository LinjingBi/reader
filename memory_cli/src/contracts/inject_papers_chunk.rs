use serde::{Deserialize, Serialize};

/// Input payload for `inject-papers-chunk` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectPapersChunkRequest {
    pub lib_config: LibConfig,
    pub papers: Vec<PaperChunkData>,
}

/// Lib config input matching the chunk_lib_config table structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibConfig {
    pub lib_config_id: String,
    pub json_payload: serde_json::Value,
}

/// Per-paper chunk data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperChunkData {
    pub paper_id: String,
    pub status: String,  // "ok" | "partial" | "error"
    pub chunks: Vec<ChunkEntry>,  // only if status != "error"
}

/// Chunk entry with selector, text, and score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkEntry {
    pub selector_id: String,  // selector name (e.g., "summary", "method")
    pub text_id: String,      // from ScoreOutput.text_table keys
    pub text: String,         // from ScoreOutput.text_table values
    pub score: f64,           // from ScoreOutput.sel2texts_score_table
}

/// Response for `inject-papers-chunk` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectPapersChunkResponse {
    pub success: bool,
    pub meta: InjectPapersChunkMeta,
}

/// Metadata without success field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectPapersChunkMeta {
    pub total_papers_count: usize,
    pub total_chunks_count: usize,
}

