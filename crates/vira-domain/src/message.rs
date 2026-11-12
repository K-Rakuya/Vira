use serde::{Deserialize, Serialize};

/// Roles in a conversational turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message entity in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub role: Role,
    pub raw_content: String,
    pub speakable_content: String,
    pub timestamp: u64,
}

impl Message {
    /// Creates a new message with an auto-generated UUID v4 and current UTC timestamp (millis).
    /// Both `raw_content` and `speakable_content` are initialized to `content`.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        let content = content.into();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            role,
            raw_content: content.clone(),
            speakable_content: content,
            timestamp,
        }
    }

    /// Sets explicit speakable content (e.g. stripped of markdown, tags, or SSML).
    pub fn with_speakable_content(mut self, speakable: impl Into<String>) -> Self {
        self.speakable_content = speakable.into();
        self
    }

    /// Constructs a message with all explicit fields (e.g. when loading from database).
    pub fn with_id(
        id: impl Into<String>,
        role: Role,
        raw_content: impl Into<String>,
        speakable_content: impl Into<String>,
        timestamp: u64,
    ) -> Self {
        Self {
            id: id.into(),
            role,
            raw_content: raw_content.into(),
            speakable_content: speakable_content.into(),
            timestamp,
        }
    }
}
