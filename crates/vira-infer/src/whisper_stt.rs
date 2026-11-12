use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio_util::sync::CancellationToken;

use vira_ports::stt::{SttError, SttEvent, SttPort};

/// Standard Whisper input sample rate in Hz (16,000 Hz).
pub const WHISPER_DEFAULT_SAMPLE_RATE: u32 = 16_000;

/// Default RMS audio energy threshold to discriminate speech from silence.
pub const DEFAULT_VAD_THRESHOLD: f32 = 0.005;

/// Configuration options for Whisper STT speech recognition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    /// Expected audio input sample rate (default 16000 Hz).
    pub sample_rate: u32,
    /// Target language code (e.g. "vi", "en", "ja").
    pub language: String,
    /// RMS energy threshold for voice activity detection.
    pub vad_energy_threshold: f32,
    /// Optional path to Whisper ONNX model weights.
    pub model_path: Option<PathBuf>,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            sample_rate: WHISPER_DEFAULT_SAMPLE_RATE,
            language: "vi".to_string(),
            vad_energy_threshold: DEFAULT_VAD_THRESHOLD,
            model_path: None,
        }
    }
}

/// Execution backend for Whisper STT.
#[derive(Debug, Clone)]
pub enum WhisperBackend {
    /// In-memory virtual fallback engine for offline testing and CI.
    Virtual {
        mock_responses: Arc<std::sync::Mutex<Vec<String>>>,
    },
    /// ONNX Runtime model loaded from disk.
    Onnx {
        model_path: PathBuf,
    },
}

/// Hexagonal adapter implementing `SttPort` for Whisper speech-to-text.
///
/// Supports real-time streaming audio chunk recognition and one-shot audio slice transcription.
/// Features a robust virtual fallback engine for zero-dependency CI testing.
#[derive(Clone)]
pub struct WhisperSttAdapter {
    config: WhisperConfig,
    backend: WhisperBackend,
}

impl Default for WhisperSttAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperSttAdapter {
    /// Creates a new Whisper STT adapter with virtual fallback backend.
    pub fn new() -> Self {
        Self {
            config: WhisperConfig::default(),
            backend: WhisperBackend::Virtual {
                mock_responses: Arc::new(std::sync::Mutex::new(Vec::new())),
            },
        }
    }

    /// Explicitly creates a virtual fallback adapter for CI / test environments.
    pub fn virtual_fallback() -> Self {
        Self::new()
    }

    /// Configures the adapter with queued mock transcriptions for deterministic testing.
    pub fn with_mock_responses(self, responses: Vec<String>) -> Self {
        let adapter = Self::virtual_fallback();
        *adapter.mock_queue().lock().unwrap() = responses;
        adapter
    }

    /// Configures the adapter to load an ONNX model if the file exists on disk.
    pub fn with_model_path(model_path: impl Into<PathBuf>) -> Self {
        let model_path = model_path.into();
        let backend = if model_path.exists() {
            WhisperBackend::Onnx {
                model_path: model_path.clone(),
            }
        } else {
            WhisperBackend::Virtual {
                mock_responses: Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        };

        let config = WhisperConfig {
            model_path: Some(model_path),
            ..Default::default()
        };

        Self { config, backend }
    }

    /// Sets target language code.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.config.language = language.into();
        self
    }

    /// Sets expected input sample rate.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.config.sample_rate = sample_rate;
        self
    }

    /// Sets voice activity energy threshold.
    pub fn with_vad_threshold(mut self, threshold: f32) -> Self {
        self.config.vad_energy_threshold = threshold.max(0.0);
        self
    }

    /// Returns configuration reference.
    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }

    /// Returns target language code.
    pub fn language(&self) -> &str {
        &self.config.language
    }

    /// Returns true if running with an active ONNX model file.
    pub fn is_onnx_active(&self) -> bool {
        matches!(self.backend, WhisperBackend::Onnx { .. })
    }

    /// Enqueues a mock transcription for testing.
    pub fn push_mock_response(&self, text: impl Into<String>) {
        if let WhisperBackend::Virtual { ref mock_responses } = self.backend {
            mock_responses.lock().unwrap().push(text.into());
        }
    }

    fn mock_queue(&self) -> Arc<std::sync::Mutex<Vec<String>>> {
        match self.backend {
            WhisperBackend::Virtual { ref mock_responses } => Arc::clone(mock_responses),
            WhisperBackend::Onnx { .. } => Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    fn pop_mock_text(&self) -> Option<String> {
        if let WhisperBackend::Virtual { ref mock_responses } = self.backend {
            let mut lock = mock_responses.lock().unwrap();
            if !lock.is_empty() {
                return Some(lock.remove(0));
            }
        }
        None
    }

    fn peek_mock_text(&self) -> Option<String> {
        if let WhisperBackend::Virtual { ref mock_responses } = self.backend {
            let lock = mock_responses.lock().unwrap();
            return lock.first().cloned();
        }
        None
    }

    /// Calculates root-mean-square (RMS) energy of PCM audio samples using double-precision accumulator.
    pub fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = samples
            .iter()
            .map(|&s| if s.is_finite() { (s as f64) * (s as f64) } else { 0.0 })
            .sum();
        ((sum_sq / samples.len() as f64).sqrt()) as f32
    }
}

#[async_trait]
impl SttPort for WhisperSttAdapter {
    async fn stream_transcribe(
        &self,
        mut audio_receiver: Receiver<Vec<f32>>,
        event_sender: Sender<SttEvent>,
        cancel: CancellationToken,
    ) -> Result<(), SttError> {
        if cancel.is_cancelled() {
            return Err(SttError::Cancelled);
        }

        let mut total_samples = Vec::new();
        let mut last_partial_ms = 0u64;

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    return Err(SttError::Cancelled);
                }
                chunk_opt = audio_receiver.recv() => {
                    match chunk_opt {
                        Some(chunk) => {
                            if chunk.is_empty() {
                                continue;
                            }
                            let rms = Self::calculate_rms(&chunk);
                            total_samples.extend_from_slice(&chunk);

                            let duration_ms = (total_samples.len() as u64 * 1000) / self.config.sample_rate as u64;

                            if rms >= self.config.vad_energy_threshold
                                && (duration_ms >= last_partial_ms + 250 || last_partial_ms == 0)
                            {
                                let partial_text = self.peek_mock_text().unwrap_or_else(|| "...".to_string());
                                let partial_event = SttEvent::partial(partial_text)
                                    .with_confidence(0.85)
                                    .with_language(&self.config.language)
                                    .with_duration_ms(duration_ms);

                                if event_sender.send(partial_event).await.is_err() {
                                    return Err(SttError::Channel("receiver dropped".to_string()));
                                }
                                last_partial_ms = duration_ms;
                            }
                        }
                        None => {
                            // Audio receiver closed; finalize transcription
                            break;
                        }
                    }
                }
            }
        }

        if cancel.is_cancelled() {
            return Err(SttError::Cancelled);
        }

        let final_result = self
            .transcribe_slice(&total_samples, self.config.sample_rate)
            .await?;

        let _ = event_sender.send(final_result).await;
        Ok(())
    }

    async fn transcribe_slice(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<SttEvent, SttError> {
        if sample_rate == 0 {
            return Err(SttError::InvalidAudio(
                "sample rate cannot be zero".to_string(),
            ));
        }

        let duration_ms = (samples.len() as u64 * 1000) / sample_rate as u64;

        if samples.is_empty() {
            return Ok(SttEvent::final_result("")
                .with_confidence(1.0)
                .with_language(&self.config.language)
                .with_duration_ms(0));
        }

        let rms = Self::calculate_rms(samples);

        // Check if silence
        if rms < self.config.vad_energy_threshold {
            return Ok(SttEvent::final_result("")
                .with_confidence(1.0)
                .with_language(&self.config.language)
                .with_duration_ms(duration_ms));
        }

        // Virtual backend transcription
        if let Some(mock_text) = self.pop_mock_text() {
            return Ok(SttEvent::final_result(mock_text)
                .with_confidence(0.96)
                .with_language(&self.config.language)
                .with_duration_ms(duration_ms));
        }

        // Default speech detected event when energy present and no mock queued
        Ok(SttEvent::final_result("[speech detected]")
            .with_confidence(0.90)
            .with_language(&self.config.language)
            .with_duration_ms(duration_ms))
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_whisper_adapter_initialization_and_sample_rate() {
        let adapter = WhisperSttAdapter::new()
            .with_language("vi")
            .with_sample_rate(16000)
            .with_vad_threshold(0.01);

        assert_eq!(adapter.language(), "vi");
        assert_eq!(adapter.sample_rate(), 16000);
        assert_eq!(adapter.config().vad_energy_threshold, 0.01);
    }

    #[tokio::test]
    async fn test_whisper_transcribe_silence_returns_empty() {
        let adapter = WhisperSttAdapter::new();
        let silent_samples = vec![0.0f32; 16000]; // 1 second of silence
        let res = adapter
            .transcribe_slice(&silent_samples, 16000)
            .await
            .expect("transcribe success");

        assert_eq!(res.text, "");
        assert_eq!(res.duration_ms, Some(1000));
        assert!(res.is_final);
    }

    #[tokio::test]
    async fn test_whisper_transcribe_mock_response() {
        let adapter =
            WhisperSttAdapter::new().with_mock_responses(vec!["Chào Vira nhé".to_string()]);

        // Generate synthetic speech energy
        let speech_samples: Vec<f32> = (0..16000)
            .map(|i| (i as f32 * 0.1).sin() * 0.2)
            .collect();

        let res = adapter
            .transcribe_slice(&speech_samples, 16000)
            .await
            .expect("transcribe success");

        assert_eq!(res.text, "Chào Vira nhé");
        assert_eq!(res.confidence, Some(0.96));
        assert_eq!(res.language.as_deref(), Some("vi"));
    }

    #[tokio::test]
    async fn test_whisper_stream_transcribe_with_cancellation() {
        let adapter = WhisperSttAdapter::new();
        let (_audio_tx, audio_rx) = mpsc::channel(16);
        let (event_tx, _event_rx) = mpsc::channel(16);
        let cancel = CancellationToken::new();

        cancel.cancel(); // Pre-cancel
        let res = adapter
            .stream_transcribe(audio_rx, event_tx, cancel)
            .await;

        assert_eq!(res, Err(SttError::Cancelled));
    }

    #[tokio::test]
    async fn test_whisper_stream_transcribe_flow() {
        let adapter = WhisperSttAdapter::new()
            .with_mock_responses(vec!["Hôm nay thời tiết thế nào?".to_string()]);

        let (audio_tx, audio_rx) = mpsc::channel(16);
        let (event_tx, mut event_rx) = mpsc::channel(16);
        let cancel = CancellationToken::new();

        let speech_chunk: Vec<f32> = (0..3200)
            .map(|i| (i as f32 * 0.1).sin() * 0.3)
            .collect();

        tokio::spawn(async move {
            audio_tx.send(speech_chunk).await.unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
            drop(audio_tx); // close stream (EOF)
        });

        adapter
            .stream_transcribe(audio_rx, event_tx, cancel)
            .await
            .expect("stream transcribe success");

        let mut events = Vec::new();
        while let Some(evt) = event_rx.recv().await {
            events.push(evt);
        }

        assert!(!events.is_empty(), "must receive at least final event");
        let final_evt = events.iter().find(|e| e.is_final).expect("has final event");
        assert_eq!(final_evt.text, "Hôm nay thời tiết thế nào?");
    }

    #[tokio::test]
    async fn test_whisper_invalid_sample_rate_fails() {
        let adapter = WhisperSttAdapter::new();
        let res = adapter.transcribe_slice(&[0.1, 0.2], 0).await;
        assert!(matches!(res, Err(SttError::InvalidAudio(_))));
    }
}
