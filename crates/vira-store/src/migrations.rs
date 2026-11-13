//! SQLite database schema migrations engine for Vira.
//!
//! Manages incremental database schema versioning, applying initial tables
//! and subsequent migrations atomically within transactions.

use rusqlite::{params, Connection};
use std::time::SystemTime;
use vira_ports::store::StoreError;

/// Individual schema migration entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Migration {
    /// Sequential version number (e.g. 1, 2, ...).
    pub version: i32,
    /// Descriptive migration name.
    pub name: &'static str,
    /// SQL statements executed to apply the migration.
    pub sql: &'static str,
}

/// Static list of all registered schema migrations in chronological order.
pub static MIGRATIONS: &[Migration] = &[Migration {
    version: 1,
    name: "001_initial_schema",
    sql: r#"
-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY NOT NULL,
    title TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    speakable_content TEXT NOT NULL,
    timestamp INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session_time ON messages(session_id, timestamp ASC);

-- Flashcards table for FSRS spaced repetition
CREATE TABLE IF NOT EXISTS flashcards (
    id TEXT PRIMARY KEY NOT NULL,
    word TEXT NOT NULL,
    definition TEXT NOT NULL,
    cefr_level TEXT,
    state INTEGER NOT NULL DEFAULT 0,
    step INTEGER,
    stability REAL NOT NULL DEFAULT 0.0,
    difficulty REAL NOT NULL DEFAULT 0.0,
    reps INTEGER NOT NULL DEFAULT 0,
    lapses INTEGER NOT NULL DEFAULT 0,
    due INTEGER NOT NULL DEFAULT 0,
    last_review INTEGER,
    front TEXT GENERATED ALWAYS AS (word) VIRTUAL,
    back TEXT GENERATED ALWAYS AS (definition) VIRTUAL
);
CREATE INDEX IF NOT EXISTS idx_flashcards_due ON flashcards(due ASC);

-- Learner profiles table
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    cefr_level TEXT NOT NULL,
    native_language TEXT NOT NULL,
    target_language TEXT NOT NULL,
    preferences_json TEXT NOT NULL DEFAULT '{}',
    updated_at INTEGER NOT NULL
);

-- Application key-value settings table
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
"#,
}];

/// Migration manager for SQLite databases.
pub struct DatabaseMigrations;

impl DatabaseMigrations {
    /// Ensures migration tracking table exists, then runs all pending migrations in order.
    pub fn run(conn: &mut Connection) -> Result<(), StoreError> {
        Self::ensure_migration_table(conn)?;

        let current = Self::current_version(conn)?;
        for migration in MIGRATIONS.iter() {
            if migration.version > current {
                Self::apply_migration(conn, migration)?;
            }
        }

        Ok(())
    }

    /// Creates the `schema_migrations` tracking table if it does not exist yet.
    fn ensure_migration_table(conn: &Connection) -> Result<(), StoreError> {
        conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at INTEGER NOT NULL
            );
            "#,
            [],
        )
        .map_err(|e| StoreError::Migration(format!("Failed to create schema_migrations table: {e}")))?;

        Ok(())
    }

    /// Returns the highest applied migration version, or 0 if no migrations have been applied.
    pub fn current_version(conn: &Connection) -> Result<i32, StoreError> {
        Self::ensure_migration_table(conn)?;

        let mut stmt = conn
            .prepare("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
            .map_err(|e| StoreError::Migration(format!("Failed to query current version: {e}")))?;

        let version: i32 = stmt
            .query_row([], |row| row.get(0))
            .map_err(|e| StoreError::Migration(format!("Failed to read migration version: {e}")))?;

        Ok(version)
    }

    /// Returns true if all known migrations have been applied.
    pub fn is_up_to_date(conn: &Connection) -> Result<bool, StoreError> {
        let current = Self::current_version(conn)?;
        let max_known = MIGRATIONS.iter().map(|m| m.version).max().unwrap_or(0);
        Ok(current >= max_known)
    }

    /// Lists all applied migrations (version, name, applied_at timestamp).
    pub fn list_applied(conn: &Connection) -> Result<Vec<(i32, String, u64)>, StoreError> {
        Self::ensure_migration_table(conn)?;

        let mut stmt = conn
            .prepare("SELECT version, name, applied_at FROM schema_migrations ORDER BY version ASC")
            .map_err(|e| StoreError::Migration(format!("Failed to prepare list_applied query: {e}")))?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i32>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)? as u64,
                ))
            })
            .map_err(|e| StoreError::Migration(format!("Failed to query applied migrations: {e}")))?;

        let mut applied = Vec::new();
        for r in rows {
            applied.push(r.map_err(|e| StoreError::Migration(format!("Row error in list_applied: {e}")))?);
        }

        Ok(applied)
    }

    /// Applies an individual migration atomically inside a transaction.
    fn apply_migration(conn: &mut Connection, migration: &Migration) -> Result<(), StoreError> {
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let tx = conn
            .transaction()
            .map_err(|e| StoreError::Migration(format!("Failed to begin transaction for v{}: {e}", migration.version)))?;

        tx.execute_batch(migration.sql).map_err(|e| {
            StoreError::Migration(format!(
                "Failed executing migration v{} ({}): {e}",
                migration.version, migration.name
            ))
        })?;

        tx.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?1, ?2, ?3)",
            params![migration.version, migration.name, now_ms],
        )
        .map_err(|e| {
            StoreError::Migration(format!(
                "Failed recording migration v{} ({}): {e}",
                migration.version, migration.name
            ))
        })?;

        tx.commit().map_err(|e| {
            StoreError::Migration(format!(
                "Failed committing transaction for migration v{}: {e}",
                migration.version
            ))
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migrations_initial_run_and_idempotence() {
        let mut conn = Connection::open_in_memory().expect("open memory db");

        assert_eq!(DatabaseMigrations::current_version(&conn).unwrap(), 0);
        assert!(!DatabaseMigrations::is_up_to_date(&conn).unwrap());

        DatabaseMigrations::run(&mut conn).expect("run migrations");

        assert_eq!(DatabaseMigrations::current_version(&conn).unwrap(), 1);
        assert!(DatabaseMigrations::is_up_to_date(&conn).unwrap());

        let applied = DatabaseMigrations::list_applied(&conn).expect("list applied");
        assert_eq!(applied.len(), 1);
        assert_eq!(applied[0].0, 1);
        assert_eq!(applied[0].1, "001_initial_schema");

        // Idempotent second run does not fail or duplicate
        DatabaseMigrations::run(&mut conn).expect("rerun migrations");
        assert_eq!(DatabaseMigrations::current_version(&conn).unwrap(), 1);
    }

    #[test]
    fn test_schema_tables_and_virtual_columns() {
        let mut conn = Connection::open_in_memory().expect("open memory db");
        DatabaseMigrations::run(&mut conn).expect("run migrations");

        // Test flashcards base columns and virtual columns
        conn.execute(
            r#"
            INSERT INTO flashcards (id, word, definition, cefr_level, state, stability, difficulty, reps, lapses, due)
            VALUES ('c1', 'serendipity', 'sự may mắn tình cờ', 'C1', 1, 2.5, 4.0, 1, 0, 1000)
            "#,
            [],
        )
        .expect("insert flashcard");

        // Query using virtual columns 'front' and 'back'
        let (front, back): (String, String) = conn
            .query_row(
                "SELECT front, back FROM flashcards WHERE id = 'c1'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("query virtual columns");

        assert_eq!(front, "serendipity");
        assert_eq!(back, "sự may mắn tình cờ");
    }
}
