//! SQLite persistence adapter implementing `vira_ports::StorePort`.
//!
//! Provides thread-safe asynchronous database access using Tokio mutex wrapping
//! rusqlite, with WAL mode and NORMAL synchronous settings for high concurrent performance.

use crate::migrations::DatabaseMigrations;
use async_trait::async_trait;
use rusqlite::{params, Connection, Row};
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::Mutex;
use vira_domain::message::{Message, Role};
use vira_ports::store::{Flashcard, SessionSummary, StoreError, StorePort, UserProfile};

/// SQLite-backed persistent storage implementation.
#[derive(Clone)]
pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteStore {
    /// Opens an in-memory SQLite database, configures pragmas, and runs migrations.
    pub fn open_in_memory() -> Result<Self, StoreError> {
        let mut conn = Connection::open_in_memory()
            .map_err(|e| StoreError::Database(format!("Failed to open in-memory database: {e}")))?;

        Self::configure_connection(&mut conn, false)?;
        DatabaseMigrations::run(&mut conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Opens or creates a file-backed SQLite database at `path`, configures WAL mode, and runs migrations.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    StoreError::Database(format!(
                        "Failed to create database directory {}: {e}",
                        parent.display()
                    ))
                })?;
            }
        }

        let mut conn = Connection::open(path).map_err(|e| {
            StoreError::Database(format!(
                "Failed to open database file {}: {e}",
                path.display()
            ))
        })?;

        Self::configure_connection(&mut conn, true)?;
        DatabaseMigrations::run(&mut conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Configures SQLite pragmas (WAL mode, synchronous, foreign keys, busy timeout).
    fn configure_connection(conn: &mut Connection, is_file: bool) -> Result<(), StoreError> {
        if is_file {
            conn.execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA foreign_keys = ON;
                 PRAGMA busy_timeout = 5000;",
            )
            .map_err(|e| StoreError::Database(format!("Failed to set file pragmas: {e}")))?;
        } else {
            conn.execute_batch(
                "PRAGMA synchronous = NORMAL;
                 PRAGMA foreign_keys = ON;
                 PRAGMA busy_timeout = 5000;",
            )
            .map_err(|e| StoreError::Database(format!("Failed to set in-memory pragmas: {e}")))?;
        }

        Ok(())
    }

    /// Synchronously runs pending migrations (useful when initializing testing instances).
    pub fn run_migrations(&self) -> Result<(), StoreError> {
        if let Ok(mut conn) = self.conn.try_lock() {
            return DatabaseMigrations::run(&mut conn);
        }

        match tokio::runtime::Handle::try_current() {
            Err(_) => {
                let mut conn = self.conn.blocking_lock();
                DatabaseMigrations::run(&mut conn)
            }
            Ok(handle) => match handle.runtime_flavor() {
                tokio::runtime::RuntimeFlavor::MultiThread => {
                    tokio::task::block_in_place(|| {
                        let mut conn = handle.block_on(self.conn.lock());
                        DatabaseMigrations::run(&mut conn)
                    })
                }
                _ => {
                    for _ in 0..50 {
                        std::thread::yield_now();
                        if let Ok(mut conn) = self.conn.try_lock() {
                            return DatabaseMigrations::run(&mut conn);
                        }
                    }
                    Err(StoreError::Database(
                        "Database lock held in single-threaded runtime; use run_migrations_async instead".into(),
                    ))
                }
            },
        }
    }

    /// Asynchronously runs pending migrations.
    pub async fn run_migrations_async(&self) -> Result<(), StoreError> {
        let mut conn = self.conn.lock().await;
        DatabaseMigrations::run(&mut conn)
    }

    /// Synchronously locks connection across sync or async runtime contexts.
    fn lock_conn_sync(&self) -> Result<tokio::sync::MutexGuard<'_, Connection>, StoreError> {
        if let Ok(guard) = self.conn.try_lock() {
            return Ok(guard);
        }
        if tokio::runtime::Handle::try_current().is_err() {
            return Ok(self.conn.blocking_lock());
        }
        tokio::task::block_in_place(|| {
            let handle = tokio::runtime::Handle::current();
            Ok(handle.block_on(self.conn.lock()))
        })
    }

    /// Synchronously saves or updates a message in a conversation session.
    pub fn save_message_sync(&self, session_id: &str, message: &Message) -> Result<(), StoreError> {
        let mut conn = self.lock_conn_sync()?;
        let ts = message.timestamp as i64;

        let tx = conn
            .transaction()
            .map_err(|e| StoreError::Database(format!("Failed to begin transaction for save_message: {e}")))?;

        tx.execute(
            r#"
            INSERT INTO sessions (id, title, created_at, updated_at)
            VALUES (?1, NULL, ?2, ?2)
            ON CONFLICT(id) DO UPDATE SET
                created_at = MIN(sessions.created_at, excluded.created_at),
                updated_at = MAX(sessions.updated_at, excluded.updated_at);
            "#,
            params![session_id, ts],
        )
        .map_err(|e| StoreError::Database(format!("Failed to upsert session: {e}")))?;

        tx.execute(
            r#"
            INSERT INTO messages (id, session_id, role, raw_content, speakable_content, timestamp)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT(id) DO UPDATE SET
                session_id = excluded.session_id,
                role = excluded.role,
                raw_content = excluded.raw_content,
                speakable_content = excluded.speakable_content,
                timestamp = excluded.timestamp;
            "#,
            params![
                message.id,
                session_id,
                message.role.as_str(),
                message.raw_content,
                message.speakable_content,
                ts,
            ],
        )
        .map_err(|e| StoreError::Database(format!("Failed to upsert message: {e}")))?;

        tx.commit()
            .map_err(|e| StoreError::Database(format!("Failed to commit save_message: {e}")))?;

        Ok(())
    }

    /// Synchronously retrieves up to `limit` latest messages for a session in chronological order.
    pub fn get_history_sync(&self, session_id: &str, limit: usize) -> Result<Vec<Message>, StoreError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let conn = self.lock_conn_sync()?;
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, role, raw_content, speakable_content, timestamp
                FROM (
                    SELECT id, role, raw_content, speakable_content, timestamp, rowid
                    FROM messages
                    WHERE session_id = ?1
                    ORDER BY timestamp DESC, rowid DESC
                    LIMIT ?2
                )
                ORDER BY timestamp ASC, rowid ASC;
                "#,
            )
            .map_err(|e| StoreError::Database(format!("Failed to prepare get_history query: {e}")))?;

        let rows = stmt
            .query_map(params![session_id, limit as i64], |row| {
                let id: String = row.get(0)?;
                let role_str: String = row.get(1)?;
                let raw_content: String = row.get(2)?;
                let speakable_content: String = row.get(3)?;
                let timestamp: i64 = row.get(4)?;
                let role = Role::from_str(&role_str).unwrap_or(Role::User);
                Ok(Message::with_id(
                    id,
                    role,
                    raw_content,
                    speakable_content,
                    timestamp as u64,
                ))
            })
            .map_err(|e| StoreError::Database(format!("Failed to execute get_history: {e}")))?;

        let mut messages = Vec::new();
        for msg in rows {
            messages.push(
                msg.map_err(|e| StoreError::Database(format!("Failed to read message row: {e}")))?,
            );
        }

        Ok(messages)
    }

    /// Queries the active SQLite journal mode ("wal", "memory", etc.).
    pub async fn journal_mode(&self) -> Result<String, StoreError> {
        let conn = self.conn.lock().await;
        let mode: String = conn
            .query_row("PRAGMA journal_mode;", [], |row| row.get(0))
            .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(mode.to_lowercase())
    }

    // Inherent async helper methods delegating directly to StorePort implementation

    /// Appends or updates a message in a conversation session.
    pub async fn save_message(&self, session_id: &str, message: &Message) -> Result<(), StoreError> {
        StorePort::save_message(self, session_id, message).await
    }

    /// Retrieves up to `limit` latest messages for a session in chronological order.
    pub async fn get_history(&self, session_id: &str, limit: usize) -> Result<Vec<Message>, StoreError> {
        StorePort::get_history(self, session_id, limit).await
    }

    /// Deletes a session and its associated messages.
    pub async fn delete_session(&self, session_id: &str) -> Result<(), StoreError> {
        StorePort::delete_session(self, session_id).await
    }

    /// Lists summaries of all conversation sessions.
    pub async fn list_sessions(&self) -> Result<Vec<SessionSummary>, StoreError> {
        StorePort::list_sessions(self).await
    }

    /// Updates the display title for a conversation session.
    pub async fn update_session_title(&self, session_id: &str, title: &str) -> Result<(), StoreError> {
        StorePort::update_session_title(self, session_id, title).await
    }

    /// Fetches a user profile by ID.
    pub async fn get_user_profile(&self, user_id: &str) -> Result<Option<UserProfile>, StoreError> {
        StorePort::get_user_profile(self, user_id).await
    }

    /// Saves or updates a user profile.
    pub async fn save_user_profile(&self, profile: &UserProfile) -> Result<(), StoreError> {
        StorePort::save_user_profile(self, profile).await
    }

    /// Saves or updates a spaced-repetition flashcard.
    pub async fn save_flashcard(&self, card: &Flashcard) -> Result<(), StoreError> {
        StorePort::save_flashcard(self, card).await
    }

    /// Retrieves a flashcard by ID.
    pub async fn get_flashcard(&self, id: &str) -> Result<Option<Flashcard>, StoreError> {
        StorePort::get_flashcard(self, id).await
    }

    /// Deletes a flashcard by ID.
    pub async fn delete_flashcard(&self, id: &str) -> Result<(), StoreError> {
        StorePort::delete_flashcard(self, id).await
    }

    /// Returns the total count of flashcards in storage.
    pub async fn count_flashcards(&self) -> Result<usize, StoreError> {
        StorePort::count_flashcards(self).await
    }

    /// Retrieves flashcards whose due date is less than or equal to `current_time`.
    pub async fn get_due_flashcards(
        &self,
        current_time: u64,
        limit: usize,
    ) -> Result<Vec<Flashcard>, StoreError> {
        StorePort::get_due_flashcards(self, current_time, limit).await
    }

    /// Lists flashcards with a limit.
    pub async fn list_flashcards(&self, limit: usize) -> Result<Vec<Flashcard>, StoreError> {
        StorePort::list_flashcards(self, limit).await
    }

    /// Retrieves a setting value by key.
    pub async fn get_setting(&self, key: &str) -> Result<Option<String>, StoreError> {
        StorePort::get_setting(self, key).await
    }

    /// Sets or updates a persistent configuration setting.
    pub async fn set_setting(&self, key: &str, value: &str) -> Result<(), StoreError> {
        StorePort::set_setting(self, key, value).await
    }

    /// Deletes a configuration setting by key.
    pub async fn delete_setting(&self, key: &str) -> Result<(), StoreError> {
        StorePort::delete_setting(self, key).await
    }
}

fn row_to_flashcard(row: &Row<'_>) -> rusqlite::Result<Flashcard> {
    let id: String = row.get(0)?;
    let front: String = row.get(1)?;
    let back: String = row.get(2)?;
    let cefr_level: Option<String> = row.get(3)?;
    let state: i64 = row.get(4)?;
    let step: Option<i64> = row.get(5)?;
    let stability: f64 = row.get(6)?;
    let difficulty: f64 = row.get(7)?;
    let reps: i64 = row.get(8)?;
    let lapses: i64 = row.get(9)?;
    let due: i64 = row.get(10)?;
    let last_review: Option<i64> = row.get(11)?;

    let stability_f32 = if stability.is_finite() {
        stability as f32
    } else {
        0.0
    };
    let difficulty_f32 = if difficulty.is_finite() {
        difficulty as f32
    } else {
        0.0
    };

    Ok(Flashcard {
        id,
        front,
        back,
        cefr_level,
        state: state.max(0) as u32,
        step: step.map(|s| s.max(0) as u32),
        stability: stability_f32,
        difficulty: difficulty_f32,
        due: due.max(0) as u64,
        last_review: last_review.map(|lr| lr.max(0) as u64),
        reps: reps.max(0) as u32,
        lapses: lapses.max(0) as u32,
    })
}

#[async_trait]
impl StorePort for SqliteStore {
    async fn save_message(&self, session_id: &str, message: &Message) -> Result<(), StoreError> {
        let mut conn = self.conn.lock().await;
        let ts = message.timestamp as i64;

        let tx = conn
            .transaction()
            .map_err(|e| StoreError::Database(format!("Failed to begin transaction for save_message: {e}")))?;

        // Ensure session exists or update its timestamps atomically
        tx.execute(
            r#"
            INSERT INTO sessions (id, title, created_at, updated_at)
            VALUES (?1, NULL, ?2, ?2)
            ON CONFLICT(id) DO UPDATE SET
                created_at = MIN(sessions.created_at, excluded.created_at),
                updated_at = MAX(sessions.updated_at, excluded.updated_at);
            "#,
            params![session_id, ts],
        )
        .map_err(|e| StoreError::Database(format!("Failed to upsert session: {e}")))?;

        // Upsert message
        tx.execute(
            r#"
            INSERT INTO messages (id, session_id, role, raw_content, speakable_content, timestamp)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT(id) DO UPDATE SET
                session_id = excluded.session_id,
                role = excluded.role,
                raw_content = excluded.raw_content,
                speakable_content = excluded.speakable_content,
                timestamp = excluded.timestamp;
            "#,
            params![
                message.id,
                session_id,
                message.role.as_str(),
                message.raw_content,
                message.speakable_content,
                ts,
            ],
        )
        .map_err(|e| StoreError::Database(format!("Failed to upsert message: {e}")))?;

        tx.commit()
            .map_err(|e| StoreError::Database(format!("Failed to commit save_message: {e}")))?;

        Ok(())
    }

    async fn get_history(&self, session_id: &str, limit: usize) -> Result<Vec<Message>, StoreError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, role, raw_content, speakable_content, timestamp
                FROM (
                    SELECT id, role, raw_content, speakable_content, timestamp, rowid
                    FROM messages
                    WHERE session_id = ?1
                    ORDER BY timestamp DESC, rowid DESC
                    LIMIT ?2
                )
                ORDER BY timestamp ASC, rowid ASC;
                "#,
            )
            .map_err(|e| StoreError::Database(format!("Failed to prepare get_history query: {e}")))?;

        let rows = stmt
            .query_map(params![session_id, limit as i64], |row| {
                let id: String = row.get(0)?;
                let role_str: String = row.get(1)?;
                let raw_content: String = row.get(2)?;
                let speakable_content: String = row.get(3)?;
                let timestamp: i64 = row.get(4)?;
                let role = Role::from_str(&role_str).unwrap_or(Role::User);
                Ok(Message::with_id(
                    id,
                    role,
                    raw_content,
                    speakable_content,
                    timestamp as u64,
                ))
            })
            .map_err(|e| StoreError::Database(format!("Failed to execute get_history: {e}")))?;

        let mut messages = Vec::new();
        for msg in rows {
            messages.push(
                msg.map_err(|e| StoreError::Database(format!("Failed to read message row: {e}")))?,
            );
        }

        Ok(messages)
    }

    async fn delete_session(&self, session_id: &str) -> Result<(), StoreError> {
        let mut conn = self.conn.lock().await;
        let tx = conn
            .transaction()
            .map_err(|e| StoreError::Database(format!("Failed to begin transaction for delete_session: {e}")))?;

        tx.execute("DELETE FROM messages WHERE session_id = ?1", params![session_id])
            .map_err(|e| StoreError::Database(format!("Failed to delete messages for session: {e}")))?;

        tx.execute("DELETE FROM sessions WHERE id = ?1", params![session_id])
            .map_err(|e| StoreError::Database(format!("Failed to delete session: {e}")))?;

        tx.commit()
            .map_err(|e| StoreError::Database(format!("Failed to commit delete_session: {e}")))?;

        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<SessionSummary>, StoreError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                r#"
                SELECT 
                    s.id, 
                    s.title, 
                    s.created_at, 
                    s.updated_at,
                    COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                GROUP BY s.id
                ORDER BY s.updated_at DESC, s.id DESC;
                "#,
            )
            .map_err(|e| StoreError::Database(format!("Failed to prepare list_sessions query: {e}")))?;

        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let title: Option<String> = row.get(1)?;
                let created_at: i64 = row.get(2)?;
                let updated_at: i64 = row.get(3)?;
                let message_count: i64 = row.get(4)?;

                Ok(SessionSummary {
                    id,
                    title,
                    message_count: message_count as usize,
                    created_at: created_at as u64,
                    updated_at: updated_at as u64,
                })
            })
            .map_err(|e| StoreError::Database(format!("Failed to query sessions: {e}")))?;

        let mut summaries = Vec::new();
        for s in rows {
            summaries.push(
                s.map_err(|e| StoreError::Database(format!("Failed to read session row: {e}")))?,
            );
        }

        Ok(summaries)
    }

    async fn update_session_title(&self, session_id: &str, title: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().await;
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let rows = conn
            .execute(
                "UPDATE sessions SET title = ?1, updated_at = MAX(updated_at, ?2) WHERE id = ?3",
                params![title, now_ms, session_id],
            )
            .map_err(|e| StoreError::Database(format!("Failed to update session title: {e}")))?;

        if rows == 0 {
            return Err(StoreError::NotFound(format!("session {session_id}")));
        }

        Ok(())
    }

    async fn get_user_profile(&self, user_id: &str) -> Result<Option<UserProfile>, StoreError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                r#"
                SELECT user_id, display_name, native_language, target_language, cefr_level, preferences_json
                FROM user_profiles
                WHERE user_id = ?1;
                "#,
            )
            .map_err(|e| StoreError::Database(format!("Failed to prepare get_user_profile query: {e}")))?;

        let mut rows = stmt
            .query(params![user_id])
            .map_err(|e| StoreError::Database(format!("Failed to query user profile: {e}")))?;

        if let Some(row) = rows
            .next()
            .map_err(|e| StoreError::Database(format!("Failed to fetch user profile row: {e}")))?
        {
            let user_id: String = row
                .get(0)
                .map_err(|e| StoreError::Database(format!("Failed to read user_id: {e}")))?;
            let display_name: String = row
                .get(1)
                .map_err(|e| StoreError::Database(format!("Failed to read display_name: {e}")))?;
            let native_language: String = row
                .get(2)
                .map_err(|e| StoreError::Database(format!("Failed to read native_language: {e}")))?;
            let target_language: String = row
                .get(3)
                .map_err(|e| StoreError::Database(format!("Failed to read target_language: {e}")))?;
            let cefr_level: String = row
                .get(4)
                .map_err(|e| StoreError::Database(format!("Failed to read cefr_level: {e}")))?;
            let prefs_json: String = row
                .get(5)
                .map_err(|e| StoreError::Database(format!("Failed to read preferences_json: {e}")))?;

            let preferences: HashMap<String, String> = serde_json::from_str(&prefs_json)
                .map_err(|e| {
                    StoreError::Serialization(format!(
                        "Corrupted preferences JSON for user '{user_id}': {e}"
                    ))
                })?;

            Ok(Some(UserProfile {
                user_id,
                display_name,
                native_language,
                target_language,
                cefr_level,
                preferences,
            }))
        } else {
            Ok(None)
        }
    }

    async fn save_user_profile(&self, profile: &UserProfile) -> Result<(), StoreError> {
        let conn = self.conn.lock().await;
        let prefs_json = serde_json::to_string(&profile.preferences)
            .map_err(|e| StoreError::Serialization(e.to_string()))?;
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        conn.execute(
            r#"
            INSERT INTO user_profiles (user_id, display_name, native_language, target_language, cefr_level, preferences_json, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(user_id) DO UPDATE SET
                display_name = excluded.display_name,
                native_language = excluded.native_language,
                target_language = excluded.target_language,
                cefr_level = excluded.cefr_level,
                preferences_json = excluded.preferences_json,
                updated_at = excluded.updated_at;
            "#,
            params![
                profile.user_id,
                profile.display_name,
                profile.native_language,
                profile.target_language,
                profile.cefr_level,
                prefs_json,
                now_ms,
            ],
        )
        .map_err(|e| StoreError::Database(format!("Failed to upsert user profile: {e}")))?;

        Ok(())
    }

    async fn save_flashcard(&self, card: &Flashcard) -> Result<(), StoreError> {
        let conn = self.conn.lock().await;

        let stability = if card.stability.is_finite() {
            card.stability as f64
        } else {
            0.0
        };
        let difficulty = if card.difficulty.is_finite() {
            card.difficulty as f64
        } else {
            0.0
        };

        conn.execute(
            r#"
            INSERT INTO flashcards (id, word, definition, cefr_level, state, step, stability, difficulty, reps, lapses, due, last_review)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            ON CONFLICT(id) DO UPDATE SET
                word = excluded.word,
                definition = excluded.definition,
                cefr_level = excluded.cefr_level,
                state = excluded.state,
                step = excluded.step,
                stability = excluded.stability,
                difficulty = excluded.difficulty,
                reps = excluded.reps,
                lapses = excluded.lapses,
                due = excluded.due,
                last_review = excluded.last_review;
            "#,
            params![
                card.id,
                card.front,
                card.back,
                card.cefr_level,
                card.state as i64,
                card.step.map(|s| s as i64),
                stability,
                difficulty,
                card.reps as i64,
                card.lapses as i64,
                card.due as i64,
                card.last_review.map(|lr| lr as i64),
            ],
        )
        .map_err(|e| StoreError::Database(format!("Failed to upsert flashcard: {e}")))?;

        Ok(())
    }

    async fn get_flashcard(&self, id: &str) -> Result<Option<Flashcard>, StoreError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, word, definition, cefr_level, state, step, stability, difficulty, reps, lapses, due, last_review
                FROM flashcards
                WHERE id = ?1;
                "#,
            )
            .map_err(|e| StoreError::Database(format!("Failed to prepare get_flashcard: {e}")))?;

        let mut rows = stmt
            .query(params![id])
            .map_err(|e| StoreError::Database(format!("Failed to query flashcard: {e}")))?;

        if let Some(row) = rows
            .next()
            .map_err(|e| StoreError::Database(format!("Failed to fetch flashcard row: {e}")))?
        {
            let card = row_to_flashcard(row)
                .map_err(|e| StoreError::Database(format!("Failed to parse flashcard: {e}")))?;
            Ok(Some(card))
        } else {
            Ok(None)
        }
    }

    async fn delete_flashcard(&self, id: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().await;
        let rows = conn
            .execute("DELETE FROM flashcards WHERE id = ?1", params![id])
            .map_err(|e| StoreError::Database(format!("Failed to delete flashcard: {e}")))?;

        if rows == 0 {
            return Err(StoreError::NotFound(format!("flashcard {id}")));
        }

        Ok(())
    }

    async fn count_flashcards(&self) -> Result<usize, StoreError> {
        let conn = self.conn.lock().await;
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM flashcards", [], |row| row.get(0))
            .map_err(|e| StoreError::Database(format!("Failed to count flashcards: {e}")))?;

        Ok(count as usize)
    }

    async fn get_due_flashcards(
        &self,
        current_time: u64,
        limit: usize,
    ) -> Result<Vec<Flashcard>, StoreError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, word, definition, cefr_level, state, step, stability, difficulty, reps, lapses, due, last_review
                FROM flashcards
                WHERE due <= ?1
                ORDER BY due ASC, id ASC
                LIMIT ?2;
                "#,
            )
            .map_err(|e| StoreError::Database(format!("Failed to prepare get_due_flashcards query: {e}")))?;

        let rows = stmt
            .query_map(params![current_time as i64, limit as i64], row_to_flashcard)
            .map_err(|e| StoreError::Database(format!("Failed to execute get_due_flashcards: {e}")))?;

        let mut cards = Vec::new();
        for c in rows {
            cards.push(
                c.map_err(|e| StoreError::Database(format!("Failed to read flashcard row: {e}")))?,
            );
        }

        Ok(cards)
    }

    async fn list_flashcards(&self, limit: usize) -> Result<Vec<Flashcard>, StoreError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, word, definition, cefr_level, state, step, stability, difficulty, reps, lapses, due, last_review
                FROM flashcards
                ORDER BY due ASC, id ASC
                LIMIT ?1;
                "#,
            )
            .map_err(|e| StoreError::Database(format!("Failed to prepare list_flashcards query: {e}")))?;

        let rows = stmt
            .query_map(params![limit as i64], row_to_flashcard)
            .map_err(|e| StoreError::Database(format!("Failed to execute list_flashcards: {e}")))?;

        let mut cards = Vec::new();
        for c in rows {
            cards.push(
                c.map_err(|e| StoreError::Database(format!("Failed to read flashcard row: {e}")))?,
            );
        }

        Ok(cards)
    }

    async fn get_setting(&self, key: &str) -> Result<Option<String>, StoreError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare("SELECT value FROM settings WHERE key = ?1")
            .map_err(|e| StoreError::Database(format!("Failed to prepare get_setting query: {e}")))?;

        let mut rows = stmt
            .query(params![key])
            .map_err(|e| StoreError::Database(format!("Failed to execute get_setting: {e}")))?;

        if let Some(row) = rows
            .next()
            .map_err(|e| StoreError::Database(format!("Failed to read setting row: {e}")))?
        {
            let val: String = row
                .get(0)
                .map_err(|e| StoreError::Database(format!("Failed to read setting value: {e}")))?;
            Ok(Some(val))
        } else {
            Ok(None)
        }
    }

    async fn set_setting(&self, key: &str, value: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().await;
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        conn.execute(
            r#"
            INSERT INTO settings (key, value, updated_at)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at;
            "#,
            params![key, value, now_ms],
        )
        .map_err(|e| StoreError::Database(format!("Failed to set setting: {e}")))?;

        Ok(())
    }

    async fn delete_setting(&self, key: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM settings WHERE key = ?1", params![key])
            .map_err(|e| StoreError::Database(format!("Failed to delete setting: {e}")))?;

        Ok(())
    }
}
