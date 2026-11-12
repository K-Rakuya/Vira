use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use vira_domain::message::Message;

/// Summary overview of a conversation session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSummary {
    /// Unique session identifier.
    pub id: String,
    /// User-friendly title or topic of conversation.
    pub title: Option<String>,
    /// Total count of messages stored in this session.
    pub message_count: usize,
    /// Creation timestamp in milliseconds UTC.
    pub created_at: u64,
    /// Last update timestamp in milliseconds UTC.
    pub updated_at: u64,
}

/// Learner and user profile preferences.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserProfile {
    /// Unique user ID.
    pub user_id: String,
    /// Display name.
    pub display_name: String,
    /// Primary/native language (e.g. "vi").
    pub native_language: String,
    /// Learning language target (e.g. "en").
    pub target_language: String,
    /// CEFR language proficiency level (e.g. "A1", "A2", "B1", "B2", "C1", "C2").
    pub cefr_level: String,
    /// Arbitrary user preference key-value pairs (e.g. theme, audio gain, tutor mode).
    pub preferences: HashMap<String, String>,
}

/// Spaced repetition flashcard representation for vocabulary learning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Flashcard {
    /// Unique card identifier.
    pub id: String,
    /// Front prompt (word, phrase, or sentence).
    pub front: String,
    /// Back explanation, meaning, or translation.
    pub back: String,
    /// Associated CEFR difficulty level if categorized.
    pub cefr_level: Option<String>,
    /// FSRS card state (0 = New, 1 = Learning, 2 = Review, 3 = Relearning).
    pub state: u32,
    /// Current learning step index.
    pub step: Option<u32>,
    /// FSRS memory stability (retrievability half-life in days).
    pub stability: f32,
    /// FSRS item difficulty rating [1.0, 10.0].
    pub difficulty: f32,
    /// Due review epoch timestamp in milliseconds.
    pub due: u64,
    /// Last reviewed timestamp in milliseconds, if any.
    pub last_review: Option<u64>,
    /// Total successful review repetitions.
    pub reps: u32,
    /// Total lapses / forgot count.
    pub lapses: u32,
}

/// Errors occurring during storage and persistence operations.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StoreError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("Entity not found: {0}")]
    NotFound(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Schema migration error: {0}")]
    Migration(String),
    #[error("Other store error: {0}")]
    Other(String),
}

/// Hexagonal Outbound Port for persistent local storage.
///
/// Handles session chat history, user profile settings, and FSRS vocabulary flashcards.
#[async_trait]
pub trait StorePort: Send + Sync {
    /// Appends or updates a message in a conversation session.
    async fn save_message(&self, session_id: &str, message: &Message) -> Result<(), StoreError>;

    /// Retrieves up to `limit` latest messages for a session.
    async fn get_history(&self, session_id: &str, limit: usize) -> Result<Vec<Message>, StoreError>;

    /// Deletes a session and its associated messages.
    async fn delete_session(&self, session_id: &str) -> Result<(), StoreError>;

    /// Lists summaries of all conversation sessions.
    async fn list_sessions(&self) -> Result<Vec<SessionSummary>, StoreError>;

    /// Fetches a user profile by ID.
    async fn get_user_profile(&self, user_id: &str) -> Result<Option<UserProfile>, StoreError>;

    /// Saves or updates a user profile.
    async fn save_user_profile(&self, profile: &UserProfile) -> Result<(), StoreError>;

    /// Saves or updates a spaced-repetition flashcard.
    async fn save_flashcard(&self, card: &Flashcard) -> Result<(), StoreError>;

    /// Retrieves a flashcard by ID.
    async fn get_flashcard(&self, id: &str) -> Result<Option<Flashcard>, StoreError>;

    /// Retrieves flashcards whose due date is less than or equal to `current_time`.
    async fn get_due_flashcards(&self, current_time: u64, limit: usize) -> Result<Vec<Flashcard>, StoreError>;

    /// Lists flashcards with a limit.
    async fn list_flashcards(&self, limit: usize) -> Result<Vec<Flashcard>, StoreError>;

    /// Retrieves a setting value by key.
    async fn get_setting(&self, key: &str) -> Result<Option<String>, StoreError>;

    /// Sets or updates a persistent configuration setting.
    async fn set_setting(&self, key: &str, value: &str) -> Result<(), StoreError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_models_serde() {
        let summary = SessionSummary {
            id: "sess_1".to_string(),
            title: Some("English Practice".to_string()),
            message_count: 5,
            created_at: 1000,
            updated_at: 1200,
        };
        let sum_json = serde_json::to_string(&summary).expect("serialize summary");
        let sum_dec: SessionSummary = serde_json::from_str(&sum_json).expect("deserialize summary");
        assert_eq!(summary, sum_dec);

        let mut prefs = HashMap::new();
        prefs.insert("tutor_mode".to_string(), "strict".to_string());
        let profile = UserProfile {
            user_id: "u1".to_string(),
            display_name: "Alice".to_string(),
            native_language: "vi".to_string(),
            target_language: "en".to_string(),
            cefr_level: "B1".to_string(),
            preferences: prefs,
        };
        let prof_json = serde_json::to_string(&profile).expect("serialize profile");
        let prof_dec: UserProfile = serde_json::from_str(&prof_json).expect("deserialize profile");
        assert_eq!(profile, prof_dec);

        let card = Flashcard {
            id: "c1".to_string(),
            front: "ephemeral".to_string(),
            back: "phù du, chóng tàn".to_string(),
            cefr_level: Some("C2".to_string()),
            state: 1,
            step: Some(2),
            stability: 1.8,
            difficulty: 5.5,
            due: 5000,
            last_review: Some(4000),
            reps: 2,
            lapses: 1,
        };
        let card_json = serde_json::to_string(&card).expect("serialize card");
        let card_dec: Flashcard = serde_json::from_str(&card_json).expect("deserialize card");
        assert_eq!(card, card_dec);
    }

    #[test]
    fn test_store_error_display() {
        let err = StoreError::NotFound("card_99".to_string());
        assert_eq!(err.to_string(), "Entity not found: card_99");
    }
}

