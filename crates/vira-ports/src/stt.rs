use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio_util::sync::CancellationToken;

/// Event emitted during speech-to-text recognition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SttEvent {
    /// Recognized transcription text.
    pub text: String,
    /// Whether this transcription is a finalized sentence/turn or interim hypothesis.
    pub is_final: bool,
    /// Confidence score in [0.0, 1.0], if available.
    pub confidence: Option<f32>,
    /// Detected language code (e.g. "vi", "en"), if available.
    pub language: Option<String>,
    /// Duration of audio processed in milliseconds.
    pub duration_ms: Option<u64>,
}

impl SttEvent {
    /// Creates a finalized speech recognition event.
    pub fn final_result(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_final: true,
            confidence: None,
            language: None,
            duration_ms: None,
        }
    }

    /// Creates an interim (partial) speech recognition event.
    pub fn partial(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_final: false,
            confidence: None,
            language: None,
            duration_ms: None,
        }
    }

    /// Sets confidence score.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Sets detected language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets processed duration.
    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }
}

/// Errors occurring during speech transcription.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SttError {
    #[error("Speech recognition was cancelled")]
    Cancelled,
    #[error("Invalid or unsupported audio format: {0}")]
    InvalidAudio(String),
    #[error("STT model inference error: {0}")]
    Inference(String),
    #[error("STT channel error: {0}")]
    Channel(String),
    #[error("Other STT error: {0}")]
    Other(String),
}

/// Hexagonal Outbound Port for Speech-To-Text (ASR) recognition.
///
/// Supports continuous real-time audio chunk transcription as well as one-shot slice inference
/// (e.g. Whisper, SenseVoice, or cloud ASR engines).
#[async_trait]
pub trait SttPort: Send + Sync {
    /// Ingests streaming audio chunks from `audio_receiver` and dispatches transcribed text
    /// events to `event_sender`.
    async fn stream_transcribe(
        &self,
        audio_receiver: Receiver<Vec<f32>>,
        event_sender: Sender<SttEvent>,
        cancel: CancellationToken,
    ) -> Result<(), SttError>;

    /// Performs one-shot transcription on a complete slice of 32-bit float PCM audio samples.
    async fn transcribe_slice(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<SttEvent, SttError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stt_event_builders_and_serde() {
        let event = SttEvent::final_result("xin chào")
            .with_confidence(0.98)
            .with_language("vi")
            .with_duration_ms(1200);

        assert_eq!(event.text, "xin chào");
        assert!(event.is_final);
        assert_eq!(event.confidence, Some(0.98));
        assert_eq!(event.language.as_deref(), Some("vi"));
        assert_eq!(event.duration_ms, Some(1200));

        let json = serde_json::to_string(&event).expect("serialize");
        let decoded: SttEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(event, decoded);

        let partial = SttEvent::partial("đang nghe");
        assert!(!partial.is_final);
    }

    #[test]
    fn test_stt_error_display() {
        let err = SttError::InvalidAudio("unsupported sample rate 8000".to_string());
        assert_eq!(
            err.to_string(),
            "Invalid or unsupported audio format: unsupported sample rate 8000"
        );
    }
}

