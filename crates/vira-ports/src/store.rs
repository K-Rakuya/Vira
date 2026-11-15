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

impl SessionSummary {
    /// Creates a new session summary with current UTC timestamp and zero messages.
    pub fn new(id: impl Into<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            id: id.into(),
            title: None,
            message_count: 0,
            created_at: now,
            updated_at: now,
        }
    }

    /// Sets the session title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Sets the message count.
    pub fn with_message_count(mut self, count: usize) -> Self {
        self.message_count = count;
        self
    }

    /// Sets creation and update timestamps in milliseconds UTC.
    pub fn with_timestamps(mut self, created_at: u64, updated_at: u64) -> Self {
        self.created_at = created_at;
        self.updated_at = updated_at;
        self
    }
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

impl UserProfile {
    /// Creates a new user profile with empty preferences.
    pub fn new(
        user_id: impl Into<String>,
        display_name: impl Into<String>,
        native_language: impl Into<String>,
        target_language: impl Into<String>,
        cefr_level: impl Into<String>,
    ) -> Self {
        Self {
            user_id: user_id.into(),
            display_name: display_name.into(),
            native_language: native_language.into(),
            target_language: target_language.into(),
            cefr_level: cefr_level.into(),
            preferences: HashMap::new(),
        }
    }

    /// Adds or updates a user preference setting.
    pub fn with_preference(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.preferences.insert(key.into(), value.into());
        self
    }

    /// Retrieves a preference value by key.
    pub fn get_preference(&self, key: &str) -> Option<&str> {
        self.preferences.get(key).map(|s| s.as_str())
    }
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
    /// Optional International Phonetic Alphabet (IPA) pronunciation (e.g. "/prəˌnʌn.siˈeɪ.ʃən/").
    #[serde(default)]
    pub phonetic: Option<String>,
    /// Optional context sentence demonstrating real-world usage.
    #[serde(default)]
    pub context_sentence: Option<String>,
    /// Optional translation of the context sentence.
    #[serde(default)]
    pub context_translation: Option<String>,
    /// Optional part of speech (noun, verb, adjective, phrase, etc.).
    #[serde(default)]
    pub part_of_speech: Option<String>,
    /// Classification and categorization tags.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl Flashcard {
    /// Creates a new flashcard with initial state (state = 0 / New) due immediately.
    pub fn new(id: impl Into<String>, front: impl Into<String>, back: impl Into<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            id: id.into(),
            front: front.into(),
            back: back.into(),
            cefr_level: None,
            state: 0,
            step: None,
            stability: 0.0,
            difficulty: 0.0,
            due: now,
            last_review: None,
            reps: 0,
            lapses: 0,
            phonetic: None,
            context_sentence: None,
            context_translation: None,
            part_of_speech: None,
            tags: Vec::new(),
        }
    }

    /// Sets the CEFR difficulty level.
    pub fn with_cefr_level(mut self, level: impl Into<String>) -> Self {
        self.cefr_level = Some(level.into());
        self
    }

    /// Sets the phonetic pronunciation.
    pub fn with_phonetic(mut self, phonetic: impl Into<String>) -> Self {
        self.phonetic = Some(phonetic.into());
        self
    }

    /// Sets context sentence and its translation.
    pub fn with_context(
        mut self,
        sentence: impl Into<String>,
        translation: impl Into<Option<String>>,
    ) -> Self {
        self.context_sentence = Some(sentence.into());
        self.context_translation = translation.into();
        self
    }

    /// Sets part of speech.
    pub fn with_part_of_speech(mut self, pos: impl Into<String>) -> Self {
        self.part_of_speech = Some(pos.into());
        self
    }

    /// Sets tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Sets FSRS parameters.
    pub fn with_fsrs(
        mut self,
        stability: f32,
        difficulty: f32,
        state: u32,
        reps: u32,
        lapses: u32,
    ) -> Self {
        self.stability = stability;
        self.difficulty = difficulty;
        self.state = state;
        self.reps = reps;
        self.lapses = lapses;
        self
    }

    /// Sets due timestamp in milliseconds.
    pub fn with_due(mut self, due: u64) -> Self {
        self.due = due;
        self
    }

    /// Returns true if this flashcard is currently due for review at `current_time`.
    pub fn is_due(&self, current_time: u64) -> bool {
        self.due <= current_time
    }
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

    /// Retrieves up to `limit` latest messages for a session in chronological order.
    async fn get_history(&self, session_id: &str, limit: usize)
        -> Result<Vec<Message>, StoreError>;

    /// Deletes a session and its associated messages.
    async fn delete_session(&self, session_id: &str) -> Result<(), StoreError>;

    /// Lists summaries of all conversation sessions.
    async fn list_sessions(&self) -> Result<Vec<SessionSummary>, StoreError>;

    /// Updates the display title for a conversation session.
    async fn update_session_title(&self, session_id: &str, title: &str) -> Result<(), StoreError>;

    /// Fetches a user profile by ID.
    async fn get_user_profile(&self, user_id: &str) -> Result<Option<UserProfile>, StoreError>;

    /// Saves or updates a user profile.
    async fn save_user_profile(&self, profile: &UserProfile) -> Result<(), StoreError>;

    /// Saves or updates a spaced-repetition flashcard.
    async fn save_flashcard(&self, card: &Flashcard) -> Result<(), StoreError>;

    /// Retrieves a flashcard by ID.
    async fn get_flashcard(&self, id: &str) -> Result<Option<Flashcard>, StoreError>;

    /// Deletes a flashcard by ID.
    async fn delete_flashcard(&self, id: &str) -> Result<(), StoreError>;

    /// Returns the total count of flashcards in storage.
    async fn count_flashcards(&self) -> Result<usize, StoreError>;

    /// Retrieves flashcards whose due date is less than or equal to `current_time`.
    async fn get_due_flashcards(
        &self,
        current_time: u64,
        limit: usize,
    ) -> Result<Vec<Flashcard>, StoreError>;

    /// Lists flashcards with a limit.
    async fn list_flashcards(&self, limit: usize) -> Result<Vec<Flashcard>, StoreError>;

    /// Retrieves a setting value by key.
    async fn get_setting(&self, key: &str) -> Result<Option<String>, StoreError>;

    /// Sets or updates a persistent configuration setting.
    async fn set_setting(&self, key: &str, value: &str) -> Result<(), StoreError>;

    /// Deletes a configuration setting by key.
    async fn delete_setting(&self, key: &str) -> Result<(), StoreError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_models_constructors_and_serde() {
        let summary = SessionSummary::new("sess_1")
            .with_title("English Practice")
            .with_message_count(5)
            .with_timestamps(1000, 1200);

        assert_eq!(summary.id, "sess_1");
        assert_eq!(summary.title.as_deref(), Some("English Practice"));
        assert_eq!(summary.message_count, 5);

        let sum_json = serde_json::to_string(&summary).expect("serialize summary");
        let sum_dec: SessionSummary = serde_json::from_str(&sum_json).expect("deserialize summary");
        assert_eq!(summary, sum_dec);

        let profile = UserProfile::new("u1", "Alice", "vi", "en", "B1")
            .with_preference("tutor_mode", "strict")
            .with_preference("theme", "cosmic");

        assert_eq!(profile.user_id, "u1");
        assert_eq!(profile.display_name, "Alice");
        assert_eq!(profile.get_preference("tutor_mode"), Some("strict"));
        assert_eq!(profile.get_preference("nonexistent"), None);

        let prof_json = serde_json::to_string(&profile).expect("serialize profile");
        let prof_dec: UserProfile = serde_json::from_str(&prof_json).expect("deserialize profile");
        assert_eq!(profile, prof_dec);

        let card = Flashcard::new("c1", "ephemeral", "phù du, chóng tàn")
            .with_cefr_level("C2")
            .with_fsrs(1.8, 5.5, 1, 2, 1)
            .with_due(5000);

        assert_eq!(card.front, "ephemeral");
        assert_eq!(card.back, "phù du, chóng tàn");
        assert_eq!(card.cefr_level.as_deref(), Some("C2"));
        assert!(card.is_due(5000));
        assert!(card.is_due(6000));
        assert!(!card.is_due(4000));

        let card_json = serde_json::to_string(&card).expect("serialize card");
        let card_dec: Flashcard = serde_json::from_str(&card_json).expect("deserialize card");
        assert_eq!(card, card_dec);
    }

    #[test]
    fn test_store_error_display() {
        let err = StoreError::NotFound("card_99".to_string());
        assert_eq!(err.to_string(), "Entity not found: card_99");

        let db_err = StoreError::Database("disk full".to_string());
        assert_eq!(db_err.to_string(), "Database error: disk full");
    }
}
