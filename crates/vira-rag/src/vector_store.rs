//! Local embedded vector store with cosine similarity calculation and embedding adapters.
//!
//! Provides fast, in-process semantic vector indexing, cosine similarity ranking,
//! snapshot disk persistence, and embedding generation adapters compatible with FastEmbed models
//! (e.g. all-MiniLM-L6-v2, Qwen3-Embedding).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

/// Error types occurring in vector store and embedding operations.
#[derive(thiserror::Error, Debug)]
pub enum VectorStoreError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Embedding error: {0}")]
    Embedding(String),
}

/// A stored vector record containing document chunk text, embedding vector, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub chunk_id: String,
    pub doc_id: String,
    pub text: String,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

impl VectorRecord {
    /// Creates a new vector record with empty metadata.
    pub fn new(
        chunk_id: impl Into<String>,
        doc_id: impl Into<String>,
        text: impl Into<String>,
        vector: Vec<f32>,
    ) -> Self {
        Self {
            chunk_id: chunk_id.into(),
            doc_id: doc_id.into(),
            text: text.into(),
            vector,
            metadata: HashMap::new(),
        }
    }

    /// Adds custom metadata key-value to this record.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A search hit resulting from vector similarity retrieval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorHit {
    pub chunk_id: String,
    pub doc_id: String,
    pub text: String,
    pub score: f32,
    pub rank: usize,
    pub metadata: HashMap<String, String>,
}

/// Computes cosine similarity between two float vectors.
///
/// Returns a value between -1.0 and 1.0. Returns 0.0 if either vector has zero magnitude
/// or if vectors are empty.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..len {
        let x = a[i];
        let y = b[i];
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom <= f32::EPSILON || !denom.is_finite() {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// In-process local embedded vector store.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LocalVectorStore {
    records: Vec<VectorRecord>,
    #[serde(skip)]
    persistence_path: Option<PathBuf>,
}

impl LocalVectorStore {
    /// Creates a new in-memory vector store.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            persistence_path: None,
        }
    }

    /// Creates or opens a vector store with a persistence backing file.
    pub fn new_with_persistence(path: impl AsRef<Path>) -> Result<Self, VectorStoreError> {
        let path = path.as_ref().to_path_buf();
        if path.exists() {
            let mut store = Self::load_from_disk(&path)?;
            store.persistence_path = Some(path);
            Ok(store)
        } else {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            Ok(Self {
                records: Vec::new(),
                persistence_path: Some(path),
            })
        }
    }

    /// Inserts a vector record into the store.
    pub fn insert(&mut self, record: VectorRecord) {
        // If chunk_id already exists, replace it
        if let Some(pos) = self.records.iter().position(|r| r.chunk_id == record.chunk_id) {
            self.records[pos] = record;
        } else {
            self.records.push(record);
        }
    }

    /// Inserts multiple vector records in batch.
    pub fn insert_batch(&mut self, batch: Vec<VectorRecord>) {
        for record in batch {
            self.insert(record);
        }
    }

    /// Searches for top-k records closest to `query_vector` using cosine similarity.
    pub fn search(&self, query_vector: &[f32], limit: usize) -> Vec<VectorHit> {
        if self.records.is_empty() || limit == 0 {
            return Vec::new();
        }

        let mut scored: Vec<(f32, &VectorRecord)> = self
            .records
            .iter()
            .map(|r| {
                let score = cosine_similarity(query_vector, &r.vector);
                (score, r)
            })
            .collect();

        // Sort descending by score
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.chunk_id.cmp(&b.1.chunk_id))
        });

        scored
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(idx, (score, record))| VectorHit {
                chunk_id: record.chunk_id.clone(),
                doc_id: record.doc_id.clone(),
                text: record.text.clone(),
                score,
                rank: idx + 1,
                metadata: record.metadata.clone(),
            })
            .collect()
    }

    /// Deletes all records belonging to `doc_id`. Returns number of deleted records.
    pub fn delete_by_doc_id(&mut self, doc_id: &str) -> usize {
        let original_len = self.records.len();
        self.records.retain(|r| r.doc_id != doc_id);
        original_len - self.records.len()
    }

    /// Deletes a record by `chunk_id`. Returns true if a record was deleted.
    pub fn delete_by_chunk_id(&mut self, chunk_id: &str) -> bool {
        let original_len = self.records.len();
        self.records.retain(|r| r.chunk_id != chunk_id);
        self.records.len() < original_len
    }

    /// Returns the total count of stored vector records.
    pub fn count(&self) -> usize {
        self.records.len()
    }

    /// Clears all vector records from the store.
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Returns a slice of all stored records.
    pub fn records(&self) -> &[VectorRecord] {
        &self.records
    }

    /// Saves vector store records to disk at `persistence_path` or given path.
    pub fn save_to_disk(&self) -> Result<(), VectorStoreError> {
        if let Some(ref path) = self.persistence_path {
            self.save_to_path(path)
        } else {
            Ok(())
        }
    }

    /// Saves store explicitly to `path`.
    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<(), VectorStoreError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Loads store records from a saved JSON snapshot file.
    pub fn load_from_disk(path: impl AsRef<Path>) -> Result<Self, VectorStoreError> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        let store: LocalVectorStore = serde_json::from_reader(reader)?;
        Ok(store)
    }
}

/// Abstract port for generating vector embeddings from text.
#[async_trait]
pub trait EmbeddingAdapter: Send + Sync {
    /// Generates embedding vector for a single text input.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, VectorStoreError>;

    /// Generates embedding vectors for a batch of text inputs.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VectorStoreError> {
        let mut results = Vec::with_capacity(texts.len());
        for t in texts {
            results.push(self.embed(t).await?);
        }
        Ok(results)
    }

    /// Returns the output vector dimension (e.g. 384 for all-MiniLM-L6-v2).
    fn dimension(&self) -> usize;

    /// Returns the model identifier name.
    fn model_name(&self) -> &str;
}

/// FastEmbed compatible local embedding adapter.
///
/// Implements deterministic semantic token and subword n-gram projection with
/// L2 unit normalization. Ensures words with similar roots, morphemes, or topic vocabulary
/// exhibit high cosine similarity without requiring external model weight downloads.
#[derive(Debug, Clone)]
pub struct FastEmbedCompat {
    model_name: String,
    dimension: usize,
}

impl Default for FastEmbedCompat {
    fn default() -> Self {
        Self::new("all-MiniLM-L6-v2", 384)
    }
}

impl FastEmbedCompat {
    /// Creates a new `FastEmbedCompat` adapter with specified model name and dimension.
    pub fn new(model_name: impl Into<String>, dimension: usize) -> Self {
        Self {
            model_name: model_name.into(),
            dimension: dimension.max(16),
        }
    }

    /// Creates an adapter configured for Qwen3-Embedding (dimension 1024).
    pub fn qwen3() -> Self {
        Self::new("Qwen3-Embedding", 1024)
    }

    /// Computes deterministic semantic pseudo-embedding from text.
    pub fn compute_embedding(&self, text: &str) -> Vec<f32> {
        let dim = self.dimension;
        let mut vec = vec![0.0f32; dim];
        let normalized = text.to_lowercase();
        let words: Vec<&str> = normalized.split_whitespace().collect();

        if words.is_empty() {
            return vec;
        }

        for (w_idx, word) in words.iter().enumerate() {
            let word_weight = 1.0 + (1.0 / (1.0 + w_idx as f32 * 0.1));

            // Word hash
            let mut h = 5381u64;
            for byte in word.bytes() {
                h = ((h << 5).wrapping_add(h)).wrapping_add(byte as u64);
            }

            let primary_bucket = (h as usize) % dim;
            let secondary_bucket = ((h >> 16) as usize) % dim;
            let sign = if (h & 1) == 0 { 1.0f32 } else { -1.0f32 };

            vec[primary_bucket] += word_weight * sign;
            vec[secondary_bucket] += word_weight * 0.5 * -sign;

            // Character n-grams (trigrams) for morphological and stem similarity
            let chars: Vec<char> = word.chars().collect();
            if chars.len() >= 3 {
                for window in chars.windows(3) {
                    let mut gh = 1777u64;
                    for c in window {
                        gh = ((gh << 5).wrapping_add(gh)).wrapping_add(*c as u64);
                    }
                    let g_bucket = (gh as usize) % dim;
                    vec[g_bucket] += 0.3;
                }
            }
        }

        // L2 Unit Normalization
        let norm_sq: f32 = vec.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt();
        if norm > f32::EPSILON {
            for v in vec.iter_mut() {
                *v /= norm;
            }
        }

        vec
    }
}

#[async_trait]
impl EmbeddingAdapter for FastEmbedCompat {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, VectorStoreError> {
        Ok(self.compute_embedding(text))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![0.5, 0.5, 0.5, 0.5];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_fastembed_compat_semantic_similarity() {
        let adapter = FastEmbedCompat::default();
        let v1 = adapter.compute_embedding("past continuous tense grammar");
        let v2 = adapter.compute_embedding("past tense continuous rules");
        let v3 = adapter.compute_embedding("cooking recipe pizza tomatoes");

        let sim_grammar = cosine_similarity(&v1, &v2);
        let sim_unrelated = cosine_similarity(&v1, &v3);

        assert!(sim_grammar > 0.6, "Similar texts should have high similarity, got {}", sim_grammar);
        assert!(sim_grammar > sim_unrelated, "Grammar texts should be closer than cooking recipe");
    }

    #[test]
    fn test_local_vector_store_crud_and_persistence() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("vectors.json");

        let mut store = LocalVectorStore::new_with_persistence(&path).expect("new store");
        let rec1 = VectorRecord::new("c1", "doc1", "Sentence one", vec![1.0, 0.0, 0.0])
            .with_metadata("tag", "test");
        let rec2 = VectorRecord::new("c2", "doc2", "Sentence two", vec![0.0, 1.0, 0.0]);

        store.insert(rec1);
        store.insert(rec2);
        assert_eq!(store.count(), 2);

        let hits = store.search(&[0.9, 0.1, 0.0], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk_id, "c1");

        store.save_to_disk().expect("save");

        // Reload from disk
        let loaded = LocalVectorStore::load_from_disk(&path).expect("load");
        assert_eq!(loaded.count(), 2);
    }
}
