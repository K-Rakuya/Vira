//! Hexagonal Outbound Port for Embedded Knowledge Base & Hybrid RAG.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An individual retrieved chunk from knowledge retrieval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeChunk {
    pub chunk_id: String,
    pub doc_id: String,
    pub title: String,
    pub text: String,
    pub score: f64,
    pub citation: String,
    pub metadata: HashMap<String, String>,
}

/// Errors occurring in RAG and knowledge retrieval operations.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RagPortError {
    #[error("Index error: {0}")]
    Index(String),
    #[error("Query error: {0}")]
    Query(String),
    #[error("Document not found: {0}")]
    NotFound(String),
    #[error("Storage error: {0}")]
    Storage(String),
}

/// Outbound Port for Knowledge Ingestion, Indexing, and Hybrid Retrieval.
#[async_trait]
pub trait RagPort: Send + Sync {
    /// Ingests a text document into the knowledge base, returning the number of chunks created.
    async fn ingest(
        &self,
        doc_id: &str,
        title: &str,
        text: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<usize, RagPortError>;

    /// Searches the knowledge base for relevant chunks.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<KnowledgeChunk>, RagPortError>;

    /// Deletes a document and its chunks by doc_id.
    async fn delete_document(&self, doc_id: &str) -> Result<(), RagPortError>;

    /// Clears the knowledge base.
    async fn clear(&self) -> Result<(), RagPortError>;

    /// Returns the total chunk count.
    fn count(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_chunk_serde() {
        let mut meta = HashMap::new();
        meta.insert("author".to_string(), "Vira".to_string());
        let chunk = KnowledgeChunk {
            chunk_id: "c1".to_string(),
            doc_id: "d1".to_string(),
            title: "Grammar".to_string(),
            text: "Content".to_string(),
            score: 0.85,
            citation: "[Grammar] (d1)".to_string(),
            metadata: meta,
        };

        let json = serde_json::to_string(&chunk).expect("serialize");
        let decoded: KnowledgeChunk = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(chunk, decoded);
    }
}
