//! Sherpa-ONNX Streaming Speech-to-Text (STT) Adapter for Vira.
//!
//! Implements CPU-only streaming and slice speech recognition using `sherpa-onnx`
//! Zipformer / Moonshine models with endpointing, voice activity detection,
//! and barge-in support. Zero GPU VRAM footprint.

use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio_util::sync::CancellationToken;

use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OnlineRecognizer, OnlineRecognizerConfig};

use vira_ports::stt::{SttError, SttEvent, SttPort};
use crate::offline_models::OfflineModelPaths;

/// Default input sample rate for Zipformer ASR (16,000 Hz).
pub const SHERPA_STT_DEFAULT_SAMPLE_RATE: u32 = 16_000;

/// Default RMS energy threshold for voice activity detection.
pub const DEFAULT_VAD_THRESHOLD: f32 = 0.005;

/// Configuration options for `SherpaSttAdapter`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SherpaSttConfig {
    /// Offline neural model file paths.
    pub model_paths: OfflineModelPaths,
    /// Target language code (e.g. "vi", "en", "ja").
    pub language: String,
    /// Expected audio input sample rate in Hz (default 16000).
    pub sample_rate: u32,
    /// RMS energy threshold for silence gating.
    pub vad_energy_threshold: f32,
    /// Force virtual fallback for testing and CI without ONNX models.
    pub force_fallback: bool,
}

impl Default for SherpaSttConfig {
    fn default() -> Self {
        Self {
            model_paths: OfflineModelPaths::default(),
            language: "vi".to_string(),
            sample_rate: SHERPA_STT_DEFAULT_SAMPLE_RATE,
            vad_energy_threshold: DEFAULT_VAD_THRESHOLD,
            force_fallback: false,
        }
    }
}

/// Internal recognizer engine supporting either streaming (Online) or buffered (Offline) models.
#[derive(Clone)]
pub enum RecognizerEngine {
    Online(Arc<Mutex<OnlineRecognizer>>),
    Offline(Arc<Mutex<OfflineRecognizer>>),
}

/// Hexagonal adapter implementing `SttPort` using `sherpa-onnx` models on CPU.
#[derive(Clone)]
pub struct SherpaSttAdapter {
    config: SherpaSttConfig,
    recognizer: Option<RecognizerEngine>,
    mock_responses: Arc<Mutex<Vec<String>>>,
}

impl Default for SherpaSttAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl SherpaSttAdapter {
    /// Creates a new `SherpaSttAdapter` resolving model paths from the default directory.
    pub fn new() -> Self {
        Self::from_config(SherpaSttConfig::default())
    }

    /// Explicitly creates a virtual fallback adapter (CI safe).
    pub fn virtual_fallback() -> Self {
        let mut config = SherpaSttConfig::default();
        config.force_fallback = true;
        Self::from_config(config)
    }

    /// Configures the adapter with queued mock responses for deterministic testing.
    pub fn with_mock_responses(self, responses: Vec<String>) -> Self {
        let adapter = Self::virtual_fallback();
        *adapter.mock_responses.lock().unwrap() = responses;
        adapter
    }

    /// Creates an adapter from a full configuration struct.
    pub fn from_config(config: SherpaSttConfig) -> Self {
        let mock_responses = Arc::new(Mutex::new(Vec::new()));

        if config.force_fallback || !config.model_paths.is_stt_available() {
            return Self {
                config,
                recognizer: None,
                mock_responses,
            };
        }

        let recognizer = Self::try_init_recognizer(&config.model_paths, config.sample_rate);

        Self {
            config,
            recognizer,
            mock_responses,
        }
    }

    /// Sets target language code.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.config.language = lang.into();
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
    pub fn config(&self) -> &SherpaSttConfig {
        &self.config
    }

    /// Returns target language code.
    pub fn language(&self) -> &str {
        &self.config.language
    }

    /// Returns true if running with an active neural recognizer.
    pub fn is_neural_active(&self) -> bool {
        self.recognizer.is_some()
    }

    /// Enqueues a mock transcription for testing.
    pub fn push_mock_response(&self, text: impl Into<String>) {
        self.mock_responses.lock().unwrap().push(text.into());
    }

    fn pop_mock_text(&self) -> Option<String> {
        let mut lock = self.mock_responses.lock().unwrap();
        if !lock.is_empty() {
            Some(lock.remove(0))
        } else {
            None
        }
    }

    fn peek_mock_text(&self) -> Option<String> {
        let lock = self.mock_responses.lock().unwrap();
        lock.first().cloned()
    }

    fn try_init_recognizer(
        paths: &OfflineModelPaths,
        sample_rate: u32,
    ) -> Option<RecognizerEngine> {
        // Try OfflineRecognizer first (Zipformer-30M / Moonshine)
        let mut offline_config = OfflineRecognizerConfig::default();
        offline_config.model_config.transducer.encoder = Some(paths.stt_encoder.to_string_lossy().to_string());
        offline_config.model_config.transducer.decoder = Some(paths.stt_decoder.to_string_lossy().to_string());
        offline_config.model_config.transducer.joiner = Some(paths.stt_joiner.to_string_lossy().to_string());
        offline_config.model_config.tokens = Some(paths.stt_tokens.to_string_lossy().to_string());
        offline_config.model_config.provider = Some("cpu".to_string());
        offline_config.model_config.num_threads = 2;
        offline_config.feat_config.sample_rate = sample_rate as i32;
        offline_config.decoding_method = Some("greedy_search".to_string());

        if let Some(r) = OfflineRecognizer::create(&offline_config) {
            return Some(RecognizerEngine::Offline(Arc::new(Mutex::new(r))));
        }

        // Otherwise attempt OnlineRecognizer
        let mut online_config = OnlineRecognizerConfig::default();
        online_config.model_config.transducer.encoder = Some(paths.stt_encoder.to_string_lossy().to_string());
        online_config.model_config.transducer.decoder = Some(paths.stt_decoder.to_string_lossy().to_string());
        online_config.model_config.transducer.joiner = Some(paths.stt_joiner.to_string_lossy().to_string());
        online_config.model_config.tokens = Some(paths.stt_tokens.to_string_lossy().to_string());
        online_config.model_config.provider = Some("cpu".to_string());
        online_config.model_config.num_threads = 2;
        online_config.feat_config.sample_rate = sample_rate as i32;
        online_config.enable_endpoint = true;
        online_config.decoding_method = Some("greedy_search".to_string());

        if let Some(r) = OnlineRecognizer::create(&online_config) {
            return Some(RecognizerEngine::Online(Arc::new(Mutex::new(r))));
        }

        None
    }

    /// Calculates root-mean-square (RMS) energy of PCM audio samples.
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
impl SttPort for SherpaSttAdapter {
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

        if let Some(ref engine) = self.recognizer {
            match engine {
                RecognizerEngine::Online(recognizer_arc) => {
                    let stream = {
                        let recognizer = recognizer_arc
                            .lock()
                            .map_err(|e| SttError::Other(format!("recognizer lock poisoned: {e}")))?;
                        recognizer.create_stream()
                    };

                    let mut last_text = String::new();

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
                                        total_samples.extend_from_slice(&chunk);
                                        stream.accept_waveform(self.config.sample_rate as i32, &chunk);

                                        let duration_ms = (total_samples.len() as u64 * 1000) / self.config.sample_rate as u64;

                                        // Decode within a scoped lock so MutexGuard is dropped before any .await
                                        let (partial_to_send, endpoint_to_send) = {
                                            let recognizer = recognizer_arc
                                                .lock()
                                                .map_err(|e| SttError::Other(format!("recognizer lock poisoned: {e}")))?;

                                            while recognizer.is_ready(&stream) {
                                                recognizer.decode(&stream);
                                            }

                                            let mut p_evt = None;
                                            if let Some(res) = recognizer.get_result(&stream) {
                                                let current_text = res.text.trim().to_string();
                                                if !current_text.is_empty() && current_text != last_text {
                                                    p_evt = Some(
                                                        SttEvent::partial(&current_text)
                                                            .with_confidence(0.90)
                                                            .with_language(&self.config.language)
                                                            .with_duration_ms(duration_ms),
                                                    );
                                                    last_text = current_text;
                                                }
                                            }

                                            let mut ep_evt = None;
                                            if recognizer.is_endpoint(&stream) {
                                                if let Some(res) = recognizer.get_result(&stream) {
                                                    let final_text = res.text.trim().to_string();
                                                    if !final_text.is_empty() {
                                                        ep_evt = Some(
                                                            SttEvent::final_result(&final_text)
                                                                .with_confidence(0.95)
                                                                .with_language(&self.config.language)
                                                                .with_duration_ms(duration_ms),
                                                        );
                                                    }
                                                }
                                                recognizer.reset(&stream);
                                                last_text.clear();
                                            }

                                            (p_evt, ep_evt)
                                        };

                                        if let Some(partial_event) = partial_to_send {
                                            if event_sender.send(partial_event).await.is_err() {
                                                return Err(SttError::Channel("receiver dropped".to_string()));
                                            }
                                        }

                                        if let Some(endpoint_event) = endpoint_to_send {
                                            if event_sender.send(endpoint_event).await.is_err() {
                                                return Err(SttError::Channel("receiver dropped".to_string()));
                                            }
                                        }
                                    }
                                    None => break,
                                }
                            }
                        }
                    }

                    if cancel.is_cancelled() {
                        return Err(SttError::Cancelled);
                    }

                    stream.input_finished();

                    let final_event = {
                        let recognizer = recognizer_arc
                            .lock()
                            .map_err(|e| SttError::Other(format!("recognizer lock poisoned: {e}")))?;

                        while recognizer.is_ready(&stream) {
                            recognizer.decode(&stream);
                        }

                        let duration_ms = (total_samples.len() as u64 * 1000) / self.config.sample_rate as u64;
                        let final_text = recognizer
                            .get_result(&stream)
                            .map(|r| r.text.trim().to_string())
                            .unwrap_or_default();

                        SttEvent::final_result(final_text)
                            .with_confidence(0.95)
                            .with_language(&self.config.language)
                            .with_duration_ms(duration_ms)
                    };

                    let _ = event_sender.send(final_event).await;
                    return Ok(());
                }
                RecognizerEngine::Offline(_) => {
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
                                            && (duration_ms >= last_partial_ms + 300 || last_partial_ms == 0)
                                        {
                                            let partial_event = SttEvent::partial("...")
                                                .with_confidence(0.85)
                                                .with_language(&self.config.language)
                                                .with_duration_ms(duration_ms);
                                            let _ = event_sender.send(partial_event).await;
                                            last_partial_ms = duration_ms;
                                        }
                                    }
                                    None => break,
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
                    return Ok(());
                }
            }
        }

        // Virtual Fallback Engine
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
                        None => break,
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

        if rms < self.config.vad_energy_threshold {
            return Ok(SttEvent::final_result("")
                .with_confidence(1.0)
                .with_language(&self.config.language)
                .with_duration_ms(duration_ms));
        }

        // Check mock queue first
        if let Some(mock_text) = self.pop_mock_text() {
            return Ok(SttEvent::final_result(mock_text)
                .with_confidence(0.96)
                .with_language(&self.config.language)
                .with_duration_ms(duration_ms));
        }

        // Active neural recognizer inference
        if let Some(ref engine) = self.recognizer {
            match engine {
                RecognizerEngine::Offline(recognizer_arc) => {
                    let recognizer = recognizer_arc
                        .lock()
                        .map_err(|e| SttError::Other(format!("recognizer lock poisoned: {e}")))?;

                    let stream = recognizer.create_stream();
                    stream.accept_waveform(sample_rate as i32, samples);
                    recognizer.decode(&stream);

                    if let Some(res) = stream.get_result() {
                        let trimmed = res.text.trim().to_string();
                        return Ok(SttEvent::final_result(trimmed)
                            .with_confidence(0.95)
                            .with_language(&self.config.language)
                            .with_duration_ms(duration_ms));
                    }
                }
                RecognizerEngine::Online(recognizer_arc) => {
                    let recognizer = recognizer_arc
                        .lock()
                        .map_err(|e| SttError::Other(format!("recognizer lock poisoned: {e}")))?;

                    let stream = recognizer.create_stream();
                    stream.accept_waveform(sample_rate as i32, samples);
                    stream.input_finished();

                    while recognizer.is_ready(&stream) {
                        recognizer.decode(&stream);
                    }

                    if let Some(res) = recognizer.get_result(&stream) {
                        let trimmed = res.text.trim().to_string();
                        return Ok(SttEvent::final_result(trimmed)
                            .with_confidence(0.95)
                            .with_language(&self.config.language)
                            .with_duration_ms(duration_ms));
                    }
                }
            }
        }

        // Default virtual fallback when energy detected and no mock queued
        Ok(SttEvent::final_result("[speech detected]")
            .with_confidence(0.90)
            .with_language(&self.config.language)
            .with_duration_ms(duration_ms))
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}
