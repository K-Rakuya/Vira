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
    /// Confidence score strictly clamped to [0.0, 1.0], if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Detected language code (e.g. "vi", "en"), if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Duration of audio processed in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

impl SttEvent {
    /// Clamps confidence value safely to [0.0, 1.0], converting NaN to 0.0.
    #[inline]
    pub fn clamp_confidence(val: f32) -> f32 {
        if val.is_nan() || val < 0.0 {
            0.0
        } else if val > 1.0 {
            1.0
        } else {
            val
        }
    }

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

    /// Sets confidence score, strictly clamped to [0.0, 1.0].
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(Self::clamp_confidence(confidence));
        self
    }

    /// Sets detected language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets processed duration in milliseconds.
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

    /// Expected input sample rate in Hz (defaults to 16000 Hz for Whisper/neural models).
    fn sample_rate(&self) -> u32 {
        16000
    }
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
    fn test_stt_confidence_clamping() {
        let over = SttEvent::final_result("test").with_confidence(1.5);
        assert_eq!(over.confidence, Some(1.0));

        let under = SttEvent::final_result("test").with_confidence(-0.5);
        assert_eq!(under.confidence, Some(0.0));

        let nan = SttEvent::final_result("test").with_confidence(f32::NAN);
        assert_eq!(nan.confidence, Some(0.0));
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
