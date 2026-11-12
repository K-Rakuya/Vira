use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;
use vira_domain::message::Message;
use vira_domain::state::CharacterEmotion;

/// Extracts the content of an inline tag formatted as `[<tag_name>:<value>]`.
pub fn extract_tag(text: &str, tag_name: &str) -> Option<String> {
    let prefix = format!("[{tag_name}:");
    let start_idx = text.find(&prefix)?;
    let after_prefix = &text[start_idx + prefix.len()..];
    let end_idx = after_prefix.find(']')?;
    let val = after_prefix[..end_idx].trim();
    if val.is_empty() {
        None
    } else {
        Some(val.to_string())
    }
}

/// Extracts any inline `[emotion:<value>]` tag from the text.
pub fn extract_emotion_tag(text: &str) -> Option<String> {
    extract_tag(text, "emotion")
}

/// Extracts any inline `[gesture:<value>]` tag from the text.
pub fn extract_gesture_tag(text: &str) -> Option<String> {
    extract_tag(text, "gesture")
}

/// Strips all `[emotion:...]` and `[gesture:...]` tags from text, returning clean speakable text.
pub fn strip_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '[' {
            let mut tag_content = String::new();
            let mut closed = false;
            for inner in chars.by_ref() {
                if inner == ']' {
                    closed = true;
                    break;
                }
                tag_content.push(inner);
            }
            if closed
                && (tag_content.starts_with("emotion:") || tag_content.starts_with("gesture:"))
            {
                // Skip tag
                continue;
            } else {
                result.push('[');
                result.push_str(&tag_content);
                if closed {
                    result.push(']');
                }
            }
        } else {
            result.push(ch);
        }
    }
    result.trim().to_string()
}

/// A chunk of streaming text received from the LLM provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmStreamChunk {
    /// Incremental token text or delta.
    pub delta: String,
    /// Indicates whether this chunk marks the end of the generation stream.
    pub is_final: bool,
    /// Optional emotion tag extracted from or associated with this chunk (e.g. "happy", "surprised").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emotion: Option<String>,
    /// Optional avatar gesture tag extracted from or associated with this chunk (e.g. "wave", "nod").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gesture: Option<String>,
}

impl LlmStreamChunk {
    /// Creates a non-final text chunk without tags.
    pub fn new(delta: impl Into<String>) -> Self {
        Self {
            delta: delta.into(),
            is_final: false,
            emotion: None,
            gesture: None,
        }
    }

    /// Creates a terminal chunk indicating the stream has completed.
    pub fn done() -> Self {
        Self {
            delta: String::new(),
            is_final: true,
            emotion: None,
            gesture: None,
        }
    }

    /// Creates a final chunk carrying the last piece of text delta.
    pub fn final_chunk(delta: impl Into<String>) -> Self {
        Self {
            delta: delta.into(),
            is_final: true,
            emotion: None,
            gesture: None,
        }
    }

    /// Attaches an emotion tag to this chunk.
    pub fn with_emotion(mut self, emotion: impl Into<String>) -> Self {
        self.emotion = Some(emotion.into());
        self
    }

    /// Attaches a gesture tag to this chunk.
    pub fn with_gesture(mut self, gesture: impl Into<String>) -> Self {
        self.gesture = Some(gesture.into());
        self
    }

    /// Constructs a chunk by extracting inline `[emotion:...]` and `[gesture:...]` tags
    /// from the raw text delta, populating `emotion` and `gesture` and leaving clean text in `delta`.
    pub fn from_delta_extracting_tags(raw_delta: &str) -> Self {
        let emotion = extract_emotion_tag(raw_delta);
        let gesture = extract_gesture_tag(raw_delta);
        let delta = strip_tags(raw_delta);
        Self {
            delta,
            is_final: false,
            emotion,
            gesture,
        }
    }

    /// Attempts to parse the emotion tag into a domain `CharacterEmotion` enum value.
    pub fn emotion_as_domain(&self) -> Option<CharacterEmotion> {
        self.emotion
            .as_deref()
            .and_then(|e| e.parse::<CharacterEmotion>().ok())
    }

    /// Returns true if the delta text is empty.
    pub fn is_empty(&self) -> bool {
        self.delta.is_empty()
    }
}

/// Errors occurring during LLM inference or communication.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmError {
    #[error("LLM request was cancelled")]
    Cancelled,
    #[error("LLM network or connection error: {0}")]
    Network(String),
    #[error("LLM inference error: {0}")]
    Inference(String),
    #[error("LLM communication channel error: {0}")]
    Channel(String),
    #[error("LLM configuration error: {0}")]
    Config(String),
    #[error("LLM timeout: {0}")]
    Timeout(String),
    #[error("Other LLM error: {0}")]
    Other(String),
}

/// Hexagonal Outbound Port for Large Language Model inference.
///
/// Implementations handle streaming chat generation, remote HTTP SSE adapters
/// (e.g. OpenAI/Ollama), or local in-process engines (mistral.rs/candle).
#[async_trait]
pub trait LlmPort: Send + Sync {
    /// Streams chat completions given the conversation message history.
    ///
    /// Chunks are dispatched to `sender` as they arrive. If `cancel` is triggered,
    /// the generation stops immediately and returns `Err(LlmError::Cancelled)`.
    async fn stream_chat(
        &self,
        messages: &[Message],
        sender: Sender<LlmStreamChunk>,
        cancel: CancellationToken,
    ) -> Result<(), LlmError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_chunk_constructors() {
        let chunk = LlmStreamChunk::new("hello");
        assert_eq!(chunk.delta, "hello");
        assert!(!chunk.is_final);
        assert!(!chunk.is_empty());
        assert_eq!(chunk.emotion, None);
        assert_eq!(chunk.gesture, None);

        let done = LlmStreamChunk::done();
        assert!(done.is_final);
        assert!(done.is_empty());

        let final_chunk = LlmStreamChunk::final_chunk("world");
        assert!(final_chunk.is_final);
        assert_eq!(final_chunk.delta, "world");
    }

    #[test]
    fn test_tag_extraction_and_stripping() {
        let raw = "[emotion:happy][gesture:wave] Xin chào bạn! Tôi là Vira. [emotion:neutral]";
        assert_eq!(extract_emotion_tag(raw), Some("happy".to_string()));
        assert_eq!(extract_gesture_tag(raw), Some("wave".to_string()));

        let clean = strip_tags(raw);
        assert_eq!(clean, "Xin chào bạn! Tôi là Vira.");

        let chunk = LlmStreamChunk::from_delta_extracting_tags(
            "[emotion:surprised][gesture:nod] Thật tuyệt vời!",
        );
        assert_eq!(chunk.delta, "Thật tuyệt vời!");
        assert_eq!(chunk.emotion.as_deref(), Some("surprised"));
        assert_eq!(chunk.gesture.as_deref(), Some("nod"));
        assert_eq!(chunk.emotion_as_domain(), Some(CharacterEmotion::Surprised));

        let non_domain = LlmStreamChunk::new("hi").with_emotion("curious");
        assert_eq!(non_domain.emotion_as_domain(), None);
    }

    #[test]
    fn test_llm_chunk_serde_roundtrip() {
        let chunk = LlmStreamChunk::new("test token")
            .with_emotion("happy")
            .with_gesture("wave");
        let serialized = serde_json::to_string(&chunk).expect("serialize chunk");
        let deserialized: LlmStreamChunk = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(chunk, deserialized);

        // Backward compatibility: JSON without emotion/gesture fields
        let legacy_json = r#"{"delta":"legacy","is_final":false}"#;
        let legacy_chunk: LlmStreamChunk = serde_json::from_str(legacy_json).expect("deserialize");
        assert_eq!(legacy_chunk.delta, "legacy");
        assert!(!legacy_chunk.is_final);
        assert_eq!(legacy_chunk.emotion, None);
        assert_eq!(legacy_chunk.gesture, None);
    }

    #[test]
    fn test_llm_error_display() {
        let err = LlmError::Network("connection refused".to_string());
        assert_eq!(
            err.to_string(),
            "LLM network or connection error: connection refused"
        );
        let timeout = LlmError::Timeout("request exceeded 15s".to_string());
        assert_eq!(timeout.to_string(), "LLM timeout: request exceeded 15s");
    }
}
