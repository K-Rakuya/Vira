use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;
use vira_domain::message::Message;

/// A chunk of streaming text received from the LLM provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmStreamChunk {
    /// Incremental token text or delta.
    pub delta: String,
    /// Indicates whether this chunk marks the end of the generation stream.
    pub is_final: bool,
}

impl LlmStreamChunk {
    /// Creates a non-final text chunk.
    pub fn new(delta: impl Into<String>) -> Self {
        Self {
            delta: delta.into(),
            is_final: false,
        }
    }

    /// Creates a terminal chunk indicating the stream has completed.
    pub fn done() -> Self {
        Self {
            delta: String::new(),
            is_final: true,
        }
    }

    /// Creates a final chunk carrying the last piece of text delta.
    pub fn final_chunk(delta: impl Into<String>) -> Self {
        Self {
            delta: delta.into(),
            is_final: true,
        }
    }

    /// Returns true if the delta is empty.
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

        let done = LlmStreamChunk::done();
        assert!(done.is_final);
        assert!(done.is_empty());

        let final_chunk = LlmStreamChunk::final_chunk("world");
        assert!(final_chunk.is_final);
        assert_eq!(final_chunk.delta, "world");
    }

    #[test]
    fn test_llm_chunk_serde_roundtrip() {
        let chunk = LlmStreamChunk::new("test token");
        let serialized = serde_json::to_string(&chunk).expect("serialize chunk");
        let deserialized: LlmStreamChunk = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(chunk, deserialized);
    }

    #[test]
    fn test_llm_error_display() {
        let err = LlmError::Network("connection refused".to_string());
        assert_eq!(err.to_string(), "LLM network or connection error: connection refused");
    }
}

