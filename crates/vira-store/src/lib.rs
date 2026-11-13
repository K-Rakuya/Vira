//! Local persistence and storage engine for Vira AI Companion.

pub mod migrations;
pub mod sqlite;

pub use sqlite::SqliteStore;
