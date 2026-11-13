//! Embedded Knowledge Base & Hybrid RAG Engine for Vira.
//!
//! Integrates Tantivy BM25 keyword search, in-process cosine vector search
//! (LanceDB compatible), FastEmbed embedding adapters, and Reciprocal Rank Fusion (RRF)
//! to deliver high-precision, low-latency contextual retrieval for language tutoring.

#![deny(clippy::all)]

pub mod chunker;
pub mod hybrid_rrf;
pub mod lancedb_vector;
pub mod tantivy_bm25;
pub mod vector_store;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

pub use chunker::{DocumentChunk, TextChunker};
pub use hybrid_rrf::{reciprocal_rank_fusion, reciprocal_rank_fusion_weighted, FusedDoc};
pub use lancedb_vector::{LanceDbTable, LanceDbVectorStore};
pub use tantivy_bm25::{Bm25Hit, TantivyBm25Index, TantivyDocInput, TantivyError};
pub use vector_store::{
    cosine_similarity, EmbeddingAdapter, FastEmbedCompat, LocalVectorStore, VectorHit,
    VectorRecord, VectorStoreError,
};
pub use vira_ports::{KnowledgeChunk, RagPort, RagPortError};

/// Unified error type for the Hybrid RAG engine.
#[derive(thiserror::Error, Debug)]
pub enum RagError {
    #[error("Tantivy error: {0}")]
    Tantivy(#[from] TantivyError),
    #[error("Vector store error: {0}")]
    VectorStore(#[from] VectorStoreError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Lock error: {0}")]
    Lock(String),
    #[error("Document not found: {0}")]
    NotFound(String),
}

/// Configuration options for the `HybridRagEngine`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HybridRagConfig {
    /// Maximum character length per chunk (default: 400).
    pub max_chunk_size: usize,
    /// Overlap between consecutive chunks (default: 50).
    pub overlap: usize,
    /// RRF smoothing constant k (default: 60).
    pub rrf_k: usize,
    /// BM25 weight in reciprocal rank fusion (default: 1.0).
    pub bm25_weight: f64,
    /// Vector similarity weight in reciprocal rank fusion (default: 1.0).
    pub vector_weight: f64,
    /// Embedding dimension (default: 384 for all-MiniLM-L6-v2).
    pub embedding_dimension: usize,
    /// Embedding model name identifier.
    pub model_name: String,
}

impl Default for HybridRagConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 400,
            overlap: 50,
            rrf_k: 60,
            bm25_weight: 1.0,
            vector_weight: 1.0,
            embedding_dimension: 384,
            model_name: "all-MiniLM-L6-v2".to_string(),
        }
    }
}

/// A synthesized search result containing chunk content, score, and citation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RagSearchResult {
    /// Chunk identifier.
    pub chunk_id: String,
    /// Parent document identifier.
    pub document_id: String,
    /// Title of the source document.
    pub title: String,
    /// Text content excerpt from the matched chunk.
    pub text: String,
    /// Reciprocal Rank Fusion combined score.
    pub score: f64,
    /// BM25 rank (1-based), if retrieved via BM25.
    pub bm25_rank: Option<usize>,
    /// Vector similarity rank (1-based), if retrieved via Vector search.
    pub vector_rank: Option<usize>,
    /// Formatted citation attribution for display on HUD / subtitles.
    pub citation: String,
    /// Custom metadata attached to this chunk.
    pub metadata: HashMap<String, String>,
}

/// Main Hybrid RAG Engine integrating chunking, BM25, LanceDB vector storage, and RRF.
pub struct HybridRagEngine {
    config: HybridRagConfig,
    chunker: TextChunker,
    bm25: TantivyBm25Index,
    vector_store: Arc<RwLock<LocalVectorStore>>,
    embedding_adapter: Arc<dyn EmbeddingAdapter>,
    titles: Arc<RwLock<HashMap<String, String>>>,
    #[allow(dead_code)]
    base_dir: Option<PathBuf>,
}

impl HybridRagEngine {
    /// Creates a new in-memory `HybridRagEngine` (ephemeral, zero disk I/O, fast).
    pub fn new_in_memory(config: HybridRagConfig) -> Result<Self, RagError> {
        let chunker = TextChunker::new(config.max_chunk_size, config.overlap);
        let bm25 = TantivyBm25Index::create_in_ram()?;
        let vector_store = Arc::new(RwLock::new(LocalVectorStore::new()));
        let embedding_adapter: Arc<dyn EmbeddingAdapter> = Arc::new(FastEmbedCompat::new(
            &config.model_name,
            config.embedding_dimension,
        ));

        Ok(Self {
            config,
            chunker,
            bm25,
            vector_store,
            embedding_adapter,
            titles: Arc::new(RwLock::new(HashMap::new())),
            base_dir: None,
        })
    }

    /// Creates or opens a persistent `HybridRagEngine` storing index and vectors under `data_dir`.
    pub fn new_persistent(
        data_dir: impl AsRef<Path>,
        config: HybridRagConfig,
    ) -> Result<Self, RagError> {
        let dir = data_dir.as_ref();
        std::fs::create_dir_all(dir)?;

        let bm25_dir = dir.join("tantivy_bm25");
        let vector_file = dir.join("vectors.json");

        let chunker = TextChunker::new(config.max_chunk_size, config.overlap);
        let bm25 = TantivyBm25Index::create_in_dir(bm25_dir)?;
        let vstore_initial = LocalVectorStore::new_with_persistence(vector_file)?;

        // Restore titles from persistent vector records
        let mut titles_map = HashMap::new();
        for rec in vstore_initial.records() {
            if let Some(t) = rec.metadata.get("title") {
                titles_map.insert(rec.doc_id.clone(), t.clone());
            }
        }

        let vector_store = Arc::new(RwLock::new(vstore_initial));
        let embedding_adapter: Arc<dyn EmbeddingAdapter> = Arc::new(FastEmbedCompat::new(
            &config.model_name,
            config.embedding_dimension,
        ));

        Ok(Self {
            config,
            chunker,
            bm25,
            vector_store,
            embedding_adapter,
            titles: Arc::new(RwLock::new(titles_map)),
            base_dir: Some(dir.to_path_buf()),
        })
    }

    /// Overrides the embedding adapter with a custom implementation.
    pub fn with_embedding_adapter(mut self, adapter: Arc<dyn EmbeddingAdapter>) -> Self {
        self.embedding_adapter = adapter;
        self
    }

    /// Ingests a full text document into the knowledge base:
    /// - Chunks text into overlapping semantic sentences.
    /// - Generates vector embeddings for all chunks.
    /// - Indexes chunks into Tantivy BM25.
    /// - Inserts vectors into the embedded vector store.
    pub async fn ingest_document(
        &self,
        doc_id: &str,
        title: &str,
        text: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<Vec<DocumentChunk>, RagError> {
        let chunks = self.chunker.chunk_document(doc_id, text);
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        // Store document title
        if let Ok(mut titles) = self.titles.write() {
            titles.insert(doc_id.to_string(), title.to_string());
        }

        let meta_map = metadata.unwrap_or_default();
        let meta_json = serde_json::to_string(&meta_map).ok();

        // 1. Ingest into Tantivy BM25
        for chunk in &chunks {
            self.bm25.add_chunk(
                &chunk.id,
                doc_id,
                title,
                &chunk.text,
                meta_json.as_deref(),
            )?;
        }
        self.bm25.commit()?;

        // 2. Ingest into Vector Store
        let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = self.embedding_adapter.embed_batch(&chunk_texts).await?;

        let mut vector_records = Vec::with_capacity(chunks.len());
        for (chunk, emb) in chunks.iter().zip(embeddings) {
            let mut rec = VectorRecord::new(&chunk.id, doc_id, &chunk.text, emb);
            rec.metadata = meta_map.clone();
            rec.metadata.insert("title".to_string(), title.to_string());
            vector_records.push(rec);
        }

        let mut vstore = self.vector_store.write().map_err(|_| {
            RagError::Lock("Poisoned lock on vector store during ingest".to_string())
        })?;
        vstore.insert_batch(vector_records);
        vstore.save_to_disk()?;

        Ok(chunks)
    }

    /// Performs Hybrid Search combining BM25 keyword matching and Vector semantic similarity
    /// using Reciprocal Rank Fusion (RRF).
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<RagSearchResult>, RagError> {
        let trimmed = query.trim();
        if trimmed.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        let candidate_pool_size = (limit * 3).max(20);

        // 1. BM25 search
        let bm25_hits = self.bm25.search(trimmed, candidate_pool_size)?;
        let bm25_chunk_ids: Vec<String> = bm25_hits.iter().map(|h| h.chunk_id.clone()).collect();

        // 2. Vector search
        let query_vec = self.embedding_adapter.embed(trimmed).await?;
        let vector_hits = {
            let vstore = self.vector_store.read().map_err(|_| {
                RagError::Lock("Poisoned lock on vector store during search".to_string())
            })?;
            vstore.search(&query_vec, candidate_pool_size)
        };
        let vector_chunk_ids: Vec<String> = vector_hits.iter().map(|h| h.chunk_id.clone()).collect();

        // 3. Reciprocal Rank Fusion
        let fused = reciprocal_rank_fusion_weighted(
            &bm25_chunk_ids,
            &vector_chunk_ids,
            self.config.rrf_k,
            self.config.bm25_weight,
            self.config.vector_weight,
        );

        // 4. Map chunk data and build results
        let mut hit_map_bm25: HashMap<String, &Bm25Hit> = HashMap::new();
        for h in &bm25_hits {
            hit_map_bm25.insert(h.chunk_id.clone(), h);
        }

        let mut hit_map_vec: HashMap<String, &VectorHit> = HashMap::new();
        for h in &vector_hits {
            hit_map_vec.insert(h.chunk_id.clone(), h);
        }

        let titles = self.titles.read().ok();

        let mut results = Vec::new();
        for item in fused.into_iter().take(limit) {
            let chunk_id = &item.doc_id;

            let (doc_id, text, title, metadata) = if let Some(vh) = hit_map_vec.get(chunk_id) {
                let title = vh
                    .metadata
                    .get("title")
                    .cloned()
                    .or_else(|| titles.as_ref().and_then(|t| t.get(&vh.doc_id)).cloned())
                    .unwrap_or_else(|| "Tài liệu học tập".to_string());
                (vh.doc_id.clone(), vh.text.clone(), title, vh.metadata.clone())
            } else if let Some(bh) = hit_map_bm25.get(chunk_id) {
                let title = if !bh.title.is_empty() {
                    bh.title.clone()
                } else {
                    titles
                        .as_ref()
                        .and_then(|t| t.get(&bh.doc_id))
                        .cloned()
                        .unwrap_or_else(|| "Tài liệu học tập".to_string())
                };
                (bh.doc_id.clone(), bh.content.clone(), title, HashMap::new())
            } else {
                continue;
            };

            let citation = format!("[{title}] ({doc_id})");

            results.push(RagSearchResult {
                chunk_id: chunk_id.clone(),
                document_id: doc_id,
                title,
                text,
                score: item.score,
                bm25_rank: item.bm25_rank,
                vector_rank: item.vector_rank,
                citation,
                metadata,
            });
        }

        Ok(results)
    }

    /// Deletes a document and all its chunks from both BM25 and Vector indices.
    pub async fn delete_document(&self, doc_id: &str) -> Result<(), RagError> {
        self.bm25.delete_document(doc_id)?;
        self.bm25.commit()?;

        let mut vstore = self.vector_store.write().map_err(|_| {
            RagError::Lock("Poisoned lock on vector store during delete".to_string())
        })?;
        vstore.delete_by_doc_id(doc_id);
        vstore.save_to_disk()?;

        if let Ok(mut titles) = self.titles.write() {
            titles.remove(doc_id);
        }

        Ok(())
    }

    /// Clears all indexed knowledge base chunks.
    pub async fn clear(&self) -> Result<(), RagError> {
        self.bm25.clear()?;

        let mut vstore = self.vector_store.write().map_err(|_| {
            RagError::Lock("Poisoned lock on vector store during clear".to_string())
        })?;
        vstore.clear();
        vstore.save_to_disk()?;

        if let Ok(mut titles) = self.titles.write() {
            titles.clear();
        }

        Ok(())
    }

    /// Returns the total number of chunks in the vector index.
    pub fn count_chunks(&self) -> usize {
        self.vector_store
            .read()
            .map(|s| s.count())
            .unwrap_or(0)
    }
}

#[async_trait::async_trait]
impl vira_ports::RagPort for HybridRagEngine {
    async fn ingest(
        &self,
        doc_id: &str,
        title: &str,
        text: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<usize, vira_ports::RagPortError> {
        let chunks = self
            .ingest_document(doc_id, title, text, metadata)
            .await
            .map_err(|e| vira_ports::RagPortError::Index(e.to_string()))?;
        Ok(chunks.len())
    }

    async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<vira_ports::KnowledgeChunk>, vira_ports::RagPortError> {
        let hits = self
            .search(query, limit)
            .await
            .map_err(|e| vira_ports::RagPortError::Query(e.to_string()))?;

        Ok(hits
            .into_iter()
            .map(|h| vira_ports::KnowledgeChunk {
                chunk_id: h.chunk_id,
                doc_id: h.document_id,
                title: h.title,
                text: h.text,
                score: h.score,
                citation: h.citation,
                metadata: h.metadata,
            })
            .collect())
    }

    async fn delete_document(&self, doc_id: &str) -> Result<(), vira_ports::RagPortError> {
        self.delete_document(doc_id)
            .await
            .map_err(|e| vira_ports::RagPortError::Storage(e.to_string()))
    }

    async fn clear(&self) -> Result<(), vira_ports::RagPortError> {
        self.clear()
            .await
            .map_err(|e| vira_ports::RagPortError::Storage(e.to_string()))
    }

    fn count(&self) -> usize {
        self.count_chunks()
    }
}
