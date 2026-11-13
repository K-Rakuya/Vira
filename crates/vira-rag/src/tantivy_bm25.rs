//! Embedded Tantivy BM25 full-text search engine.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, RwLock};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, STORED, STRING, TEXT, Value};
use tantivy::{Index, IndexReader, IndexWriter, TantivyDocument, Term};

/// Error types for Tantivy BM25 operations.
#[derive(thiserror::Error, Debug)]
pub enum TantivyError {
    #[error("Tantivy error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),
    #[error("Query parse error: {0}")]
    QueryParser(#[from] tantivy::query::QueryParserError),
    #[error("Directory error: {0}")]
    Directory(String),
    #[error("Writer lock poisoned")]
    PoisonedLock,
}

/// Input chunk descriptor for Tantivy indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TantivyDocInput {
    pub chunk_id: String,
    pub doc_id: String,
    pub title: String,
    pub content: String,
    pub metadata: Option<String>,
}

/// A search hit returned from BM25 full-text search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bm25Hit {
    pub chunk_id: String,
    pub doc_id: String,
    pub title: String,
    pub content: String,
    pub score: f32,
    pub rank: usize,
}

/// Schema fields for the Tantivy knowledge base index.
#[derive(Clone)]
pub struct RagSchemaFields {
    pub chunk_id: Field,
    pub doc_id: Field,
    pub title: Field,
    pub content: Field,
    pub metadata: Field,
}

/// Tantivy BM25 Full-text search engine.
pub struct TantivyBm25Index {
    schema: Schema,
    index: Index,
    reader: IndexReader,
    writer: Arc<RwLock<IndexWriter>>,
    fields: RagSchemaFields,
}

impl TantivyBm25Index {
    /// Builds the standard schema for RAG chunk indexing.
    fn build_schema() -> (Schema, RagSchemaFields) {
        let mut builder = Schema::builder();
        let chunk_id = builder.add_text_field("chunk_id", STRING | STORED);
        let doc_id = builder.add_text_field("doc_id", STRING | STORED);
        let title = builder.add_text_field("title", TEXT | STORED);
        let content = builder.add_text_field("content", TEXT | STORED);
        let metadata = builder.add_text_field("metadata", STORED);
        let schema = builder.build();
        (
            schema,
            RagSchemaFields {
                chunk_id,
                doc_id,
                title,
                content,
                metadata,
            },
        )
    }

    /// Creates an in-memory Tantivy BM25 index (RAM storage, ultra-fast, zero disk persistence).
    pub fn create_in_ram() -> Result<Self, TantivyError> {
        let (schema, fields) = Self::build_schema();
        let index = Index::create_in_ram(schema.clone());
        let reader = index.reader()?;
        let writer = index.writer(50_000_000)?; // 50MB buffer
        Ok(Self {
            schema,
            index,
            reader,
            writer: Arc::new(RwLock::new(writer)),
            fields,
        })
    }

    /// Creates or opens a Tantivy BM25 index in a directory path.
    pub fn create_in_dir(path: impl AsRef<Path>) -> Result<Self, TantivyError> {
        let path = path.as_ref();
        std::fs::create_dir_all(path).map_err(|e| tantivy::TantivyError::IoError(Arc::new(e)))?;
        let (schema, fields) = Self::build_schema();

        let index = match Index::open_in_dir(path) {
            Ok(idx) => idx,
            Err(_) => Index::create_in_dir(path, schema.clone())?,
        };

        let reader = index.reader()?;
        let writer = index.writer(50_000_000)?;
        Ok(Self {
            schema,
            index,
            reader,
            writer: Arc::new(RwLock::new(writer)),
            fields,
        })
    }

    /// Adds a single chunk document to the BM25 index.
    pub fn add_chunk(
        &self,
        chunk_id: &str,
        doc_id: &str,
        title: &str,
        content: &str,
        metadata: Option<&str>,
    ) -> Result<(), TantivyError> {
        let mut doc = TantivyDocument::default();
        doc.add_text(self.fields.chunk_id, chunk_id);
        doc.add_text(self.fields.doc_id, doc_id);
        doc.add_text(self.fields.title, title);
        doc.add_text(self.fields.content, content);
        if let Some(m) = metadata {
            doc.add_text(self.fields.metadata, m);
        }

        let writer = self.writer.write().map_err(|_| TantivyError::PoisonedLock)?;
        writer.add_document(doc)?;
        Ok(())
    }

    /// Adds multiple chunk documents in batch.
    pub fn add_chunks_batch(&self, chunks: &[TantivyDocInput]) -> Result<(), TantivyError> {
        let writer = self.writer.write().map_err(|_| TantivyError::PoisonedLock)?;
        for c in chunks {
            let mut doc = TantivyDocument::default();
            doc.add_text(self.fields.chunk_id, &c.chunk_id);
            doc.add_text(self.fields.doc_id, &c.doc_id);
            doc.add_text(self.fields.title, &c.title);
            doc.add_text(self.fields.content, &c.content);
            if let Some(ref m) = c.metadata {
                doc.add_text(self.fields.metadata, m);
            }
            writer.add_document(doc)?;
        }
        Ok(())
    }

    /// Commits pending index operations and reloads searcher reader.
    pub fn commit(&self) -> Result<(), TantivyError> {
        let mut writer = self.writer.write().map_err(|_| TantivyError::PoisonedLock)?;
        writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    /// Searches for documents matching `query_str` with up to `limit` hits.
    pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<Bm25Hit>, TantivyError> {
        let trimmed = query_str.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }

        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(
            &self.index,
            vec![self.fields.title, self.fields.content],
        );

        // Sanitize or parse query; fall back to escaped query if syntax characters cause issues
        let query = match query_parser.parse_query(trimmed) {
            Ok(q) => q,
            Err(_) => {
                let sanitized = trimmed
                    .replace([':', '(', ')', '[', ']', '{', '}', '^', '"', '~', '*', '?', '\\', '/'], " ");
                query_parser.parse_query(&sanitized)?
            }
        };

        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit).order_by_score())?;
        let mut hits = Vec::new();

        for (rank_idx, (score, doc_address)) in top_docs.into_iter().enumerate() {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;

            let chunk_id = retrieved_doc
                .get_first(self.fields.chunk_id)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            let doc_id = retrieved_doc
                .get_first(self.fields.doc_id)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            let title = retrieved_doc
                .get_first(self.fields.title)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            let content = retrieved_doc
                .get_first(self.fields.content)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            hits.push(Bm25Hit {
                chunk_id,
                doc_id,
                title,
                content,
                score,
                rank: rank_idx + 1,
            });
        }

        Ok(hits)
    }

    /// Deletes all indexed chunks matching `doc_id`.
    pub fn delete_document(&self, doc_id: &str) -> Result<(), TantivyError> {
        let writer = self.writer.write().map_err(|_| TantivyError::PoisonedLock)?;
        let term = Term::from_field_text(self.fields.doc_id, doc_id);
        writer.delete_term(term);
        Ok(())
    }

    /// Deletes a specific chunk by `chunk_id`.
    pub fn delete_chunk(&self, chunk_id: &str) -> Result<(), TantivyError> {
        let writer = self.writer.write().map_err(|_| TantivyError::PoisonedLock)?;
        let term = Term::from_field_text(self.fields.chunk_id, chunk_id);
        writer.delete_term(term);
        Ok(())
    }

    /// Clears the entire index.
    pub fn clear(&self) -> Result<(), TantivyError> {
        let mut writer = self.writer.write().map_err(|_| TantivyError::PoisonedLock)?;
        writer.delete_all_documents()?;
        writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    /// Returns the schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tantivy_in_ram_bm25_search() {
        let index = TantivyBm25Index::create_in_ram().expect("create in ram");
        index
            .add_chunk(
                "c1",
                "doc_1",
                "Vietnamese Tones",
                "Tiếng Việt có sáu thanh điệu: ngang, huyền, sắc, hỏi, ngã, nặng.",
                None,
            )
            .expect("add chunk 1");
        index
            .add_chunk(
                "c2",
                "doc_2",
                "English Grammar",
                "Past continuous tense expresses action ongoing at a past time.",
                None,
            )
            .expect("add chunk 2");

        index.commit().expect("commit");

        let hits = index.search("thanh điệu tiếng Việt", 5).expect("search");
        assert!(!hits.is_empty());
        assert_eq!(hits[0].chunk_id, "c1");
        assert_eq!(hits[0].doc_id, "doc_1");

        let hits_en = index.search("past continuous", 5).expect("search en");
        assert!(!hits_en.is_empty());
        assert_eq!(hits_en[0].chunk_id, "c2");
    }

    #[test]
    fn test_tantivy_delete_document() {
        let index = TantivyBm25Index::create_in_ram().expect("create in ram");
        index
            .add_chunk("c1", "doc_del", "Title", "Some unique keywords abc xyz", None)
            .expect("add");
        index.commit().expect("commit");

        let hits_before = index.search("abc xyz", 5).expect("search");
        assert_eq!(hits_before.len(), 1);

        index.delete_document("doc_del").expect("delete");
        index.commit().expect("commit");

        let hits_after = index.search("abc xyz", 5).expect("search");
        assert_eq!(hits_after.len(), 0);
    }
}
