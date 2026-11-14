//! Sherpa-ONNX Offline Neural TTS Adapter for Vira.
//!
//! Provides CPU-only speech synthesis using Piper VITS (Vietnamese) and Kokoro
//! (English/Japanese) with synchronized 60Hz 3D VRM blendshape lip-sync animation
//! and instant barge-in cancellation. Zero GPU VRAM footprint.

use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig,
    OfflineTtsModelConfig, OfflineTtsVitsModelConfig,
};

use vira_ports::llm::strip_tags;
use vira_ports::tts::{
    TimedViseme, TtsAudioChunk, TtsError, TtsOutput, TtsPort, TtsStreamEvent,
};
use vira_voice::phoneme_aligner::{G2pConverter, PhonemeAligner, PhonemeLanguage};

use crate::offline_models::OfflineModelPaths;

/// Default sample rate for neural speech output (24,000 Hz).
pub const SHERPA_TTS_DEFAULT_SAMPLE_RATE: u32 = 24_000;

/// Viseme tick interval in milliseconds (~60 Hz).
pub const VISEME_TICK_INTERVAL_MS: u64 = 16;

/// Configuration for `SherpaTtsAdapter`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SherpaTtsConfig {
    /// Offline neural model file paths.
    pub model_paths: OfflineModelPaths,
    /// Target sample rate in Hz (default 24000).
    pub sample_rate: u32,
    /// Speaking speed multiplier (0.5 to 2.0, default 1.0).
    pub speed: f32,
    /// Chunk duration in milliseconds for streaming audio events (default 50ms).
    pub chunk_ms: u64,
    /// Speaker ID for multi-speaker models (default 0).
    pub speaker_id: i32,
    /// Preferred voice name or identifier.
    pub voice: String,
    /// Whether to force virtual fallback even if model files exist.
    pub force_fallback: bool,
}

impl Default for SherpaTtsConfig {
    fn default() -> Self {
        Self {
            model_paths: OfflineModelPaths::default(),
            sample_rate: SHERPA_TTS_DEFAULT_SAMPLE_RATE,
            speed: 1.0,
            chunk_ms: 50,
            speaker_id: 0,
            voice: "vi_VN-vais1000".to_string(),
            force_fallback: false,
        }
    }
}

/// Hexagonal adapter implementing `TtsPort` using `sherpa-onnx` offline models on CPU.
#[derive(Clone)]
pub struct SherpaTtsAdapter {
    config: SherpaTtsConfig,
    aligner: PhonemeAligner,
    piper_tts: Option<Arc<Mutex<OfflineTts>>>,
    kokoro_tts: Option<Arc<Mutex<OfflineTts>>>,
}

impl Default for SherpaTtsAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl SherpaTtsAdapter {
    /// Creates a new `SherpaTtsAdapter` resolving model paths from the default directory.
    pub fn new() -> Self {
        Self::from_config(SherpaTtsConfig::default())
    }

    /// Explicitly creates a virtual fallback adapter (CI / zero-dependency safe).
    pub fn virtual_fallback() -> Self {
        let mut config = SherpaTtsConfig::default();
        config.force_fallback = true;
        Self::from_config(config)
    }

    /// Creates an adapter with specific model paths.
    pub fn with_model_paths(paths: OfflineModelPaths) -> Self {
        let mut config = SherpaTtsConfig::default();
        config.model_paths = paths;
        Self::from_config(config)
    }

    /// Creates an adapter from a full configuration struct.
    pub fn from_config(config: SherpaTtsConfig) -> Self {
        let aligner = PhonemeAligner::new();

        if config.force_fallback {
            return Self {
                config,
                aligner,
                piper_tts: None,
                kokoro_tts: None,
            };
        }

        let piper_tts = Self::try_init_piper(&config.model_paths, config.speed);
        let kokoro_tts = Self::try_init_kokoro(&config.model_paths, config.speed);

        Self {
            config,
            aligner,
            piper_tts,
            kokoro_tts,
        }
    }

    /// Sets playback speed multiplier.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.config.speed = speed.clamp(0.5, 2.0);
        self
    }

    /// Sets target sample rate in Hz.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.config.sample_rate = sample_rate;
        self
    }

    /// Sets preferred voice name.
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.config.voice = voice.into();
        self
    }

    /// Sets speaker ID.
    pub fn with_speaker_id(mut self, speaker_id: i32) -> Self {
        self.config.speaker_id = speaker_id;
        self
    }

    /// Sets streaming chunk interval in milliseconds.
    pub fn with_chunk_ms(mut self, chunk_ms: u64) -> Self {
        self.config.chunk_ms = chunk_ms.max(20);
        self
    }

    /// Returns configuration reference.
    pub fn config(&self) -> &SherpaTtsConfig {
        &self.config
    }

    /// Returns reference to phoneme aligner.
    pub fn aligner(&self) -> &PhonemeAligner {
        &self.aligner
    }

    /// Returns true if Piper VITS Vietnamese neural model is initialized.
    pub fn is_piper_active(&self) -> bool {
        self.piper_tts.is_some()
    }

    /// Returns true if Kokoro multi-language neural model is initialized.
    pub fn is_kokoro_active(&self) -> bool {
        self.kokoro_tts.is_some()
    }

    /// Returns true if any offline neural TTS model is actively loaded.
    pub fn is_neural_active(&self) -> bool {
        self.is_piper_active() || self.is_kokoro_active()
    }

    fn try_init_piper(paths: &OfflineModelPaths, speed: f32) -> Option<Arc<Mutex<OfflineTts>>> {
        let model_path = paths.resolve_piper_model()?;
        let tokens_path = paths.resolve_piper_tokens()?;

        let length_scale = if speed > 0.0 { 1.0 / speed } else { 1.0 };
        let data_dir = if paths.piper_data_dir.is_dir() {
            Some(paths.piper_data_dir.to_string_lossy().to_string())
        } else {
            None
        };

        let config = OfflineTtsConfig {
            model: OfflineTtsModelConfig {
                vits: OfflineTtsVitsModelConfig {
                    model: Some(model_path.to_string_lossy().to_string()),
                    tokens: Some(tokens_path.to_string_lossy().to_string()),
                    data_dir,
                    noise_scale: 0.667,
                    noise_scale_w: 0.8,
                    length_scale,
                    dict_dir: None,
                    lexicon: None,
                },
                provider: Some("cpu".to_string()),
                num_threads: 2,
                debug: false,
                ..Default::default()
            },
            ..Default::default()
        };

        OfflineTts::create(&config).map(|engine| Arc::new(Mutex::new(engine)))
    }

    fn try_init_kokoro(paths: &OfflineModelPaths, speed: f32) -> Option<Arc<Mutex<OfflineTts>>> {
        let model_path = paths.resolve_kokoro_model()?;
        let voices_path = paths.resolve_kokoro_voices()?;
        let tokens_path = paths.resolve_kokoro_tokens()?;

        let length_scale = if speed > 0.0 { 1.0 / speed } else { 1.0 };
        let data_dir = if paths.kokoro_data_dir.is_dir() {
            Some(paths.kokoro_data_dir.to_string_lossy().to_string())
        } else {
            None
        };

        let config = OfflineTtsConfig {
            model: OfflineTtsModelConfig {
                kokoro: OfflineTtsKokoroModelConfig {
                    model: Some(model_path.to_string_lossy().to_string()),
                    voices: Some(voices_path.to_string_lossy().to_string()),
                    tokens: Some(tokens_path.to_string_lossy().to_string()),
                    data_dir,
                    length_scale,
                    dict_dir: None,
                    lexicon: None,
                    lang: None,
                },
                provider: Some("cpu".to_string()),
                num_threads: 2,
                debug: false,
                ..Default::default()
            },
            ..Default::default()
        };

        OfflineTts::create(&config).map(|engine| Arc::new(Mutex::new(engine)))
    }

    /// Synthesizes audio samples using active offline neural models if available,
    /// or gracefully produces harmonic acoustic fallback samples.
    pub fn synthesize_samples(
        &self,
        clean_text: &str,
        cancel: &CancellationToken,
    ) -> Result<(Vec<f32>, u32, PhonemeLanguage), TtsError> {
        if cancel.is_cancelled() {
            return Err(TtsError::Cancelled);
        }

        let lang = G2pConverter::detect_language(clean_text);

        // Select neural engine by language preference
        let selected_engine = match lang {
            PhonemeLanguage::Vietnamese => {
                self.piper_tts.as_ref().or(self.kokoro_tts.as_ref())
            }
            _ => {
                self.kokoro_tts.as_ref().or(self.piper_tts.as_ref())
            }
        };

        if let Some(engine_lock) = selected_engine {
            let gen_config = GenerationConfig {
                speed: self.config.speed,
                sid: self.config.speaker_id,
                silence_scale: 0.2,
                num_steps: 5,
                ..Default::default()
            };

            let cancel_clone = cancel.clone();
            let progress_cb = move |_samples: &[f32], _progress: f32| -> bool {
                !cancel_clone.is_cancelled()
            };

            let engine = engine_lock
                .lock()
                .map_err(|e| TtsError::Other(format!("engine lock poisoned: {e}")))?;

            if let Some(generated) = engine.generate_with_config(clean_text, &gen_config, Some(progress_cb)) {
                if cancel.is_cancelled() {
                    return Err(TtsError::Cancelled);
                }
                let samples = generated.samples().to_vec();
                let sample_rate = generated.sample_rate() as u32;
                return Ok((samples, sample_rate, lang));
            }
        }

        // Virtual acoustic fallback
        let (samples, sample_rate) = self.generate_virtual_fallback_samples(clean_text);
        Ok((samples, sample_rate, lang))
    }

    /// Virtual fallback acoustic wave generator with harmonic overtones.
    fn generate_virtual_fallback_samples(&self, clean_text: &str) -> (Vec<f32>, u32) {
        let sample_rate = self.config.sample_rate;
        let speed = (self.config.speed as f64).clamp(0.2, 3.0);
        let chars: Vec<char> = clean_text.chars().collect();
        let num_chars = chars.len().max(1);

        let duration_ms = ((num_chars as f64 * 75.0) / speed) as u64;
        let total_samples = ((duration_ms as f64 / 1000.0) * sample_rate as f64) as usize;

        let f0 = 195.0; // Warm vocal pitch
        let dt = 1.0 / sample_rate as f64;
        let mut samples = Vec::with_capacity(total_samples);

        for i in 0..total_samples {
            let t_sec = i as f64 * dt;
            let h1 = (2.0 * std::f64::consts::PI * f0 * t_sec).sin();
            let h2 = 0.5 * (4.0 * std::f64::consts::PI * f0 * t_sec).sin();
            let h3 = 0.25 * (6.0 * std::f64::consts::PI * f0 * t_sec).sin();

            let fade_samples = (0.015 * sample_rate as f64) as usize;
            let envelope = if i < fade_samples {
                i as f32 / fade_samples as f32
            } else if i + fade_samples >= total_samples {
                (total_samples.saturating_sub(i)) as f32 / fade_samples as f32
            } else {
                1.0
            };

            let s = ((h1 + h2 + h3) as f32 * 0.3 * envelope).clamp(-1.0, 1.0);
            samples.push(s);
        }

        (samples, sample_rate)
    }
}

#[async_trait]
impl TtsPort for SherpaTtsAdapter {
    async fn stream_synthesize(
        &self,
        text: &str,
        sender: Sender<TtsStreamEvent>,
        cancel: CancellationToken,
    ) -> Result<(), TtsError> {
        if cancel.is_cancelled() {
            return Err(TtsError::Cancelled);
        }

        let clean = strip_tags(text);
        let trimmed = clean.trim();
        if trimmed.is_empty() {
            let _ = sender.send(TtsStreamEvent::Done).await;
            return Ok(());
        }

        let (samples, sample_rate, lang) = self.synthesize_samples(trimmed, &cancel)?;

        if samples.is_empty() {
            let _ = sender.send(TtsStreamEvent::Done).await;
            return Ok(());
        }

        let duration_ms = (samples.len() as u64 * 1000) / sample_rate.max(1) as u64;
        let visemes = self.aligner.align_text_to_visemes(trimmed, duration_ms, lang);

        let chunk_size = ((self.config.chunk_ms as f64 / 1000.0) * sample_rate as f64) as usize;
        let chunk_size = chunk_size.max(480); // At least 20ms

        let mut offset = 0;
        let mut viseme_idx = 0;

        while offset < samples.len() {
            if cancel.is_cancelled() {
                return Err(TtsError::Cancelled);
            }

            let end = (offset + chunk_size).min(samples.len());
            let chunk_samples = samples[offset..end].to_vec();
            let chunk_end_ms = (end as u64 * 1000) / sample_rate as u64;

            // Collect visemes aligned within this audio chunk window
            let mut chunk_visemes = Vec::new();
            while viseme_idx < visemes.len() && visemes[viseme_idx].timestamp_ms <= chunk_end_ms {
                chunk_visemes.push(visemes[viseme_idx].clone());
                viseme_idx += 1;
            }

            if !chunk_visemes.is_empty()
                && sender
                    .send(TtsStreamEvent::Visemes(chunk_visemes))
                    .await
                    .is_err()
            {
                return Err(TtsError::Channel("receiver dropped".to_string()));
            }

            let audio_chunk = TtsAudioChunk::new(chunk_samples, sample_rate);
            if sender.send(TtsStreamEvent::Audio(audio_chunk)).await.is_err() {
                return Err(TtsError::Channel("receiver dropped".to_string()));
            }

            offset = end;
            tokio::task::yield_now().await;
        }

        // Flush remaining visemes
        if viseme_idx < visemes.len() {
            let remainder = visemes[viseme_idx..].to_vec();
            let _ = sender.send(TtsStreamEvent::Visemes(remainder)).await;
        }

        let _ = sender.send(TtsStreamEvent::Done).await;
        Ok(())
    }

    async fn synthesize(&self, text: &str) -> Result<TtsOutput, TtsError> {
        let clean = strip_tags(text);
        let trimmed = clean.trim();
        if trimmed.is_empty() {
            return Ok(TtsOutput::new(
                Vec::new(),
                self.config.sample_rate,
                vec![TimedViseme::zero(0)],
            ));
        }

        let cancel = CancellationToken::new();
        let (samples, sample_rate, lang) = self.synthesize_samples(trimmed, &cancel)?;

        let duration_ms = (samples.len() as u64 * 1000) / sample_rate.max(1) as u64;
        let visemes = self.aligner.align_text_to_visemes(trimmed, duration_ms, lang);

        Ok(TtsOutput::new(samples, sample_rate, visemes))
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}
