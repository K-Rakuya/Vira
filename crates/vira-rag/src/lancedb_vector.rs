//! LanceDB-compatible embedded vector table adapter.
//!
//! Provides table-oriented vector operations, semantic indexing, and persistence
//! following the LanceDB interface contract.

pub use crate::vector_store::{
    cosine_similarity, EmbeddingAdapter, FastEmbedCompat, LocalVectorStore, VectorHit,
    VectorRecord, VectorStoreError,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// A named vector table in the LanceDB embedded storage structure.
#[derive(Clone)]
pub struct LanceDbTable {
    name: String,
    store: Arc<RwLock<LocalVectorStore>>,
}

impl LanceDbTable {
    /// Returns the table name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Adds a batch of vector records to this table.
    pub fn add(&self, records: Vec<VectorRecord>) -> Result<(), VectorStoreError> {
        let mut store = self.store.write().map_err(|_| {
            VectorStoreError::Embedding("Lock poisoned while writing to table".to_string())
        })?;
        store.insert_batch(records);
        store.save_to_disk()?;
        Ok(())
    }

    /// Searches the table for nearest neighbors using cosine similarity.
    pub fn search(&self, vector: &[f32], limit: usize) -> Result<Vec<VectorHit>, VectorStoreError> {
        let store = self.store.read().map_err(|_| {
            VectorStoreError::Embedding("Lock poisoned while reading from table".to_string())
        })?;
        Ok(store.search(vector, limit))
    }

    /// Deletes records by parent document identifier.
    pub fn delete_by_doc_id(&self, doc_id: &str) -> Result<usize, VectorStoreError> {
        let mut store = self.store.write().map_err(|_| {
            VectorStoreError::Embedding("Lock poisoned while deleting from table".to_string())
        })?;
        let count = store.delete_by_doc_id(doc_id);
        store.save_to_disk()?;
        Ok(count)
    }

    /// Returns the number of vector rows in this table.
    pub fn count(&self) -> usize {
        self.store.read().map(|s| s.count()).unwrap_or(0)
    }
}

/// Embedded LanceDB-compatible vector database manager.
#[derive(Clone)]
pub struct LanceDbVectorStore {
    base_dir: Option<PathBuf>,
    tables: Arc<RwLock<HashMap<String, LanceDbTable>>>,
}

impl LanceDbVectorStore {
    /// Creates a new in-memory LanceDB store.
    pub fn new_in_memory() -> Self {
        Self {
            base_dir: None,
            tables: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Creates or connects to a persistent LanceDB storage directory.
    pub fn connect(path: impl AsRef<Path>) -> Result<Self, VectorStoreError> {
        let base_dir = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_dir)?;
        Ok(Self {
            base_dir: Some(base_dir),
            tables: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Opens or creates a named vector table.
    pub fn open_table(&self, name: &str) -> Result<LanceDbTable, VectorStoreError> {
        let mut tables = self.tables.write().map_err(|_| {
            VectorStoreError::Embedding("Lock poisoned while opening table".to_string())
        })?;

        if let Some(table) = tables.get(name) {
            return Ok(table.clone());
        }

        let store = if let Some(ref base) = self.base_dir {
            let table_file = base.join(format!("{name}.json"));
            LocalVectorStore::new_with_persistence(table_file)?
        } else {
            LocalVectorStore::new()
        };

        let table = LanceDbTable {
            name: name.to_string(),
            store: Arc::new(RwLock::new(store)),
        };

        tables.insert(name.to_string(), table.clone());
        Ok(table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lancedb_table_add_and_search() {
        let db = LanceDbVectorStore::new_in_memory();
        let table = db.open_table("knowledge_chunks").expect("open table");

        let rec = VectorRecord::new("c1", "doc1", "Hello LanceDB", vec![1.0, 0.0, 0.0]);
        table.add(vec![rec]).expect("add");

        assert_eq!(table.count(), 1);
        let hits = table.search(&[1.0, 0.0, 0.0], 5).expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk_id, "c1");
    }
}
