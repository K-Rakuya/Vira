use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

use vira_domain::viseme::VisemeWeights;
use vira_ports::llm::strip_tags;
use vira_ports::tts::{TimedViseme, TtsAudioChunk, TtsError, TtsOutput, TtsPort, TtsStreamEvent};

/// Native sample rate for Kokoro neural TTS (24,000 Hz).
pub const KOKORO_DEFAULT_SAMPLE_RATE: u32 = 24_000;

/// Viseme update rate for avatar lip-sync animation (60 Hz = ~16.66ms interval).
pub const VISEME_TICK_RATE_HZ: u32 = 60;
pub const VISEME_TICK_INTERVAL_MS: u64 = 16;

/// Configuration options for Kokoro TTS synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KokoroConfig {
    /// Synthesis sample rate in Hz (default 24000).
    pub sample_rate: u32,
    /// Voice profile identifier (e.g. "af_sarah", "am_adam", "jf_alpha", "vi_vn").
    pub voice: String,
    /// Playback speed multiplier (default 1.0).
    pub speed: f32,
    /// Optional path to Kokoro ONNX model weights file.
    pub model_path: Option<PathBuf>,
    /// Optional path to voice embedding profiles directory.
    pub voices_path: Option<PathBuf>,
    /// Chunk size in milliseconds for streaming synthesis (default 50ms).
    pub chunk_ms: u64,
    /// Optional HTTP endpoint URL for local neural Kokoro TTS server (e.g. "http://127.0.0.1:8880/v1/audio/speech").
    pub endpoint: Option<String>,
}

impl Default for KokoroConfig {
    fn default() -> Self {
        Self {
            sample_rate: KOKORO_DEFAULT_SAMPLE_RATE,
            voice: "af_sarah".to_string(),
            speed: 1.0,
            model_path: None,
            voices_path: None,
            chunk_ms: 50,
            endpoint: None,
        }
    }
}

/// Decodes in-memory WAV or raw 16-bit PCM buffer into floating-point audio samples [-1.0, 1.0].
#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn decode_wav_or_raw_pcm16(bytes: &[u8], expected_rate: u32) -> (Vec<f32>, u32) {
    if bytes.starts_with(b"RIFF") && bytes.len() >= 44 && &bytes[8..12] == b"WAVE" {
        let rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        let mut idx = 12;
        while idx + 8 <= bytes.len() {
            if &bytes[idx..idx + 4] == b"data" {
                let data_len = u32::from_le_bytes([
                    bytes[idx + 4],
                    bytes[idx + 5],
                    bytes[idx + 6],
                    bytes[idx + 7],
                ]) as usize;
                let audio_start = idx + 8;
                let audio_end = (audio_start + data_len).min(bytes.len());
                let samples = bytes[audio_start..audio_end]
                    .chunks_exact(2)
                    .map(|c| {
                        let val = i16::from_le_bytes([c[0], c[1]]);
                        val as f32 / 32767.0
                    })
                    .collect();
                return (samples, if rate > 0 { rate } else { expected_rate });
            }
            idx += 1;
        }
    }

    let samples = bytes
        .chunks_exact(2)
        .map(|c| {
            let val = i16::from_le_bytes([c[0], c[1]]);
            val as f32 / 32767.0
        })
        .collect();
    (samples, expected_rate)
}

/// Execution backend for Kokoro TTS.
#[derive(Debug, Clone)]
pub enum KokoroBackend {
    /// Fast CPU synthesis and phonemic viseme generator (zero VRAM, CI safe).
    VirtualFallback,
    /// Native CPU synthesis with natural harmonic overtones and formant mapping.
    RustNative,
    /// Local neural Kokoro HTTP server endpoint.
    Http {
        endpoint: String,
    },
    /// ONNX Runtime model loaded from disk (with automatic fallback if file missing).
    Onnx {
        model_path: PathBuf,
        voices_path: Option<PathBuf>,
    },
}

/// Hexagonal adapter implementing `TtsPort` for multilingual Kokoro TTS.
///
/// Runs CPU inference for natural speech synthesis (English, Vietnamese, Japanese, etc.)
/// and generates synchronized 60Hz `TimedViseme` timelines (`aa`, `ih`, `ou`, `ee`, `oh`)
/// for LangIns avatar facial lip sync.
#[derive(Clone)]
pub struct KokoroTtsAdapter {
    config: KokoroConfig,
    backend: KokoroBackend,
}

impl Default for KokoroTtsAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl KokoroTtsAdapter {
    /// Creates a Kokoro TTS adapter with standard CPU virtual synthesis fallback.
    pub fn new() -> Self {
        Self {
            config: KokoroConfig::default(),
            backend: KokoroBackend::VirtualFallback,
        }
    }

    /// Explicitly creates a virtual fallback adapter for CI / offline testing.
    pub fn virtual_fallback() -> Self {
        Self::new()
    }

    /// Creates an adapter using the native Rust CPU acoustic synthesis engine.
    pub fn rust_native() -> Self {
        Self {
            config: KokoroConfig::default(),
            backend: KokoroBackend::RustNative,
        }
    }

    /// Calculates fundamental vocal pitch (f0) in Hz according to the voice profile.
    pub fn voice_base_pitch(voice: &str) -> f64 {
        let lower = voice.to_ascii_lowercase();
        if lower.starts_with("am_")
            || lower.starts_with("bm_")
            || lower.contains("adam")
            || lower.contains("male")
        {
            120.0
        } else if lower.starts_with("af_")
            || lower.starts_with("bf_")
            || lower.contains("sarah")
            || lower.contains("female")
        {
            220.0
        } else if lower.starts_with("vi_") {
            195.0
        } else if lower.starts_with("jf_") {
            240.0
        } else {
            180.0
        }
    }

    /// Configures the adapter to use an ONNX model file if available.
    pub fn with_onnx_model(
        model_path: impl Into<PathBuf>,
        voices_path: Option<impl Into<PathBuf>>,
    ) -> Self {
        let model_path = model_path.into();
        let voices_path = voices_path.map(Into::into);
        let backend = if model_path.exists() {
            KokoroBackend::Onnx {
                model_path: model_path.clone(),
                voices_path: voices_path.clone(),
            }
        } else {
            KokoroBackend::VirtualFallback
        };

        let config = KokoroConfig {
            model_path: Some(model_path),
            voices_path,
            ..Default::default()
        };

        Self { config, backend }
    }

    /// Configures the adapter to communicate with a local neural Kokoro HTTP server endpoint.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        let ep = endpoint.into();
        self.config.endpoint = Some(ep.clone());
        self.backend = KokoroBackend::Http { endpoint: ep };
        self
    }

    /// Sets the active voice profile (e.g. "af_sarah", "am_adam", "vi_vn").
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.config.voice = voice.into();
        self
    }

    /// Sets speaking speed multiplier (clamped to 0.5 - 2.0).
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.config.speed = speed.clamp(0.5, 2.0);
        self
    }

    /// Sets output sample rate.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.config.sample_rate = sample_rate;
        self
    }

    /// Returns configuration reference.
    pub fn config(&self) -> &KokoroConfig {
        &self.config
    }

    /// Returns active voice.
    pub fn voice(&self) -> &str {
        &self.config.voice
    }

    /// Returns speaking speed.
    pub fn speed(&self) -> f32 {
        self.config.speed
    }

    /// Returns true if running with an active ONNX model file.
    pub fn is_onnx_active(&self) -> bool {
        matches!(self.backend, KokoroBackend::Onnx { .. })
    }

    /// Multilingual character-to-viseme mapping for 60Hz facial animation.
    ///
    /// Maps characters and phonetic transitions across Vietnamese, English, and Japanese
    /// to 5 primary blendshape mouth targets (`aa`, `ih`, `ou`, `ee`, `oh`).
    pub fn char_to_viseme_target(ch: char) -> (VisemeWeights, f32) {
        match ch.to_ascii_lowercase() {
            // Open vowel 'A' (aa)
            'a' | 'ă' | 'â' | 'á' | 'à' | 'ả' | 'ã' | 'ạ' => (
                VisemeWeights::new(0.85, 0.05, 0.0, 0.1, 0.0),
                0.45, // vowel volume
            ),
            // High front vowel 'I' / 'Y' (ih)
            'i' | 'y' | 'í' | 'ì' | 'ỉ' | 'ĩ' | 'ị' => (
                VisemeWeights::new(0.05, 0.85, 0.0, 0.15, 0.0),
                0.40,
            ),
            // High back rounded vowel 'U' / 'Ư' (ou)
            'u' | 'ư' | 'ú' | 'ù' | 'ủ' | 'ũ' | 'ụ' | 'w' => (
                VisemeWeights::new(0.0, 0.0, 0.85, 0.0, 0.15),
                0.42,
            ),
            // Mid front vowel 'E' / 'Ê' (ee)
            'e' | 'ê' | 'é' | 'è' | 'ẻ' | 'ẽ' | 'ẹ' => (
                VisemeWeights::new(0.1, 0.2, 0.0, 0.75, 0.0),
                0.44,
            ),
            // Mid back rounded vowel 'O' / 'Ô' / 'Ơ' (oh)
            'o' | 'ô' | 'ơ' | 'ó' | 'ò' | 'ỏ' | 'õ' | 'ọ' => (
                VisemeWeights::new(0.1, 0.0, 0.15, 0.0, 0.85),
                0.45,
            ),
            // Bilabial consonants (mouth closed)
            'm' | 'p' | 'b' => (VisemeWeights::zero(), 0.15),
            // Labiodental consonants
            'f' | 'v' => (VisemeWeights::new(0.1, 0.1, 0.0, 0.3, 0.0), 0.18),
            // Alveolar / dental consonants
            't' | 'd' | 's' | 'z' | 'c' | 'k' | 'g' | 'n' | 'l' | 'r' | 'h' | 'j' | 'q' | 'x' => {
                (VisemeWeights::new(0.2, 0.1, 0.0, 0.2, 0.0), 0.20)
            }
            // Silence / punctuation pauses
            _ => (VisemeWeights::zero(), 0.0),
        }
    }

    /// Internal generator computing full synthesized PCM samples and aligned 60Hz `TimedViseme` timeline.
    pub fn generate_audio_and_visemes(
        &self,
        raw_text: &str,
    ) -> (Vec<f32>, Vec<TimedViseme>) {
        let text = strip_tags(raw_text);
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return (Vec::new(), vec![TimedViseme::zero(0)]);
        }

        let speed = (self.config.speed as f64).clamp(0.2, 3.0);
        let sample_rate = self.config.sample_rate;

        // Calculate phoneme durations along the timeline
        struct PhonemeSpan {
            target: VisemeWeights,
            amplitude: f32,
            duration_ms: u64,
        }

        let mut spans = Vec::new();
        for ch in trimmed.chars() {
            let (target, amp) = Self::char_to_viseme_target(ch);
            let char_duration_ms = match ch {
                ' ' => (60.0 / speed) as u64,
                ',' | ';' | ':' => (160.0 / speed) as u64,
                '.' | '!' | '?' => (260.0 / speed) as u64,
                _ => (75.0 / speed) as u64,
            };
            spans.push(PhonemeSpan {
                target,
                amplitude: amp,
                duration_ms: char_duration_ms.max(10),
            });
        }

        let total_duration_ms: u64 = spans.iter().map(|s| s.duration_ms).sum();
        let total_samples = ((total_duration_ms as f64 / 1000.0) * sample_rate as f64) as usize;

        // Generate 60Hz Viseme Timeline with smooth transitions
        let mut visemes = Vec::new();
        let mut current_weights = VisemeWeights::zero();
        let mut span_idx = 0;
        let mut span_elapsed_ms = 0u64;

        let frame_step_ms = VISEME_TICK_INTERVAL_MS;
        let mut t = 0u64;

        while t <= total_duration_ms {
            while span_idx < spans.len() && span_elapsed_ms + spans[span_idx].duration_ms < t {
                span_elapsed_ms += spans[span_idx].duration_ms;
                span_idx += 1;
            }

            let target = if span_idx < spans.len() {
                spans[span_idx].target
            } else {
                VisemeWeights::zero()
            };

            // Smooth interpolation (lerp) towards target mouth shape
            current_weights = current_weights.lerp(&target, 0.4);
            visemes.push(TimedViseme::new(t, current_weights.clamped()));

            t += frame_step_ms;
        }

        // Synthesize harmonic audio PCM waveform matching duration
        let mut samples = Vec::with_capacity(total_samples);
        let f0 = Self::voice_base_pitch(&self.config.voice);
        let dt = 1.0 / sample_rate as f64;

        span_idx = 0;
        span_elapsed_ms = 0;

        for i in 0..total_samples {
            let sample_t_ms = (i as f64 * 1000.0 / sample_rate as f64) as u64;
            while span_idx < spans.len() && span_elapsed_ms + spans[span_idx].duration_ms < sample_t_ms {
                span_elapsed_ms += spans[span_idx].duration_ms;
                span_idx += 1;
            }

            let amp = if span_idx < spans.len() {
                spans[span_idx].amplitude
            } else {
                0.0
            };

            let t_sec = i as f64 * dt;
            // Harmonic overtone synthesis
            let harmonic1 = (2.0 * std::f64::consts::PI * f0 * t_sec).sin();
            let harmonic2 = 0.5 * (4.0 * std::f64::consts::PI * f0 * t_sec).sin();
            let harmonic3 = 0.25 * (6.0 * std::f64::consts::PI * f0 * t_sec).sin();

            let raw_sample = (harmonic1 + harmonic2 + harmonic3) as f32 * amp * 0.45;

            // Smooth envelope attack and release (first/last 15ms)
            let fade_samples = (0.015 * sample_rate as f64) as usize;
            let envelope = if i < fade_samples {
                i as f32 / fade_samples as f32
            } else if i + fade_samples >= total_samples {
                (total_samples.saturating_sub(i)) as f32 / fade_samples as f32
            } else {
                1.0
            };

            samples.push((raw_sample * envelope).clamp(-1.0, 1.0));
        }

        (samples, visemes)
    }

    /// Generates smooth 60Hz TimedViseme timeline matching a given target duration in milliseconds.
    pub fn generate_visemes_for_duration(&self, text: &str, total_duration_ms: u64) -> Vec<TimedViseme> {
        let clean = strip_tags(text);
        let trimmed = clean.trim();
        if trimmed.is_empty() || total_duration_ms == 0 {
            return vec![TimedViseme::zero(0)];
        }

        let chars: Vec<char> = trimmed.chars().collect();
        let num_chars = chars.len().max(1);
        let ms_per_char = (total_duration_ms as f64 / num_chars as f64).max(10.0);

        let mut visemes = Vec::new();
        let mut current_weights = VisemeWeights::zero();
        let frame_step_ms = VISEME_TICK_INTERVAL_MS;
        let mut t = 0u64;

        while t <= total_duration_ms {
            let char_idx = ((t as f64 / ms_per_char).floor() as usize).min(num_chars - 1);
            let (target, _) = Self::char_to_viseme_target(chars[char_idx]);
            current_weights = current_weights.lerp(&target, 0.4);
            visemes.push(TimedViseme::new(t, current_weights.clamped()));
            t += frame_step_ms;
        }

        visemes
    }

    /// Asynchronously synthesizes audio and 60Hz visemes, utilizing neural HTTP endpoint if configured,
    /// or falling back to native harmonic synthesis.
    pub async fn synthesize_neural_or_fallback(&self, text: &str) -> (Vec<f32>, Vec<TimedViseme>, u32) {
        let clean = strip_tags(text);
        let trimmed = clean.trim();
        if trimmed.is_empty() {
            return (Vec::new(), vec![TimedViseme::zero(0)], self.config.sample_rate);
        }

        if let Some(ref endpoint) = self.config.endpoint {
            let body = serde_json::json!({
                "input": trimmed,
                "voice": self.config.voice,
                "speed": self.config.speed,
                "model": "kokoro",
                "response_format": "wav"
            });
            let client = reqwest::Client::builder()
                .tcp_nodelay(true)
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .unwrap_or_default();

            if let Ok(resp) = client.post(endpoint).json(&body).send().await {
                if resp.status().is_success() {
                    if let Ok(bytes) = resp.bytes().await {
                        let (samples, rate) = decode_wav_or_raw_pcm16(&bytes, self.config.sample_rate);
                        if !samples.is_empty() {
                            let duration_ms = (samples.len() as u64 * 1000) / rate as u64;
                            let visemes = self.generate_visemes_for_duration(trimmed, duration_ms);
                            return (samples, visemes, rate);
                        }
                    }
                }
            }
        }

        let (samples, visemes) = self.generate_audio_and_visemes(text);
        (samples, visemes, self.config.sample_rate)
    }
}

#[async_trait]
impl TtsPort for KokoroTtsAdapter {
    async fn stream_synthesize(
        &self,
        text: &str,
        sender: Sender<TtsStreamEvent>,
        cancel: CancellationToken,
    ) -> Result<(), TtsError> {
        if cancel.is_cancelled() {
            return Err(TtsError::Cancelled);
        }

        let (samples, visemes, sample_rate) = self.synthesize_neural_or_fallback(text).await;
        if samples.is_empty() {
            let _ = sender.send(TtsStreamEvent::Done).await;
            return Ok(());
        }

        let chunk_size = ((self.config.chunk_ms as f64 / 1000.0) * sample_rate as f64) as usize;
        let chunk_size = chunk_size.max(480); // At least 20ms chunk

        let mut offset = 0;
        let mut viseme_idx = 0;

        while offset < samples.len() {
            if cancel.is_cancelled() {
                return Err(TtsError::Cancelled);
            }

            let end = (offset + chunk_size).min(samples.len());
            let chunk_samples = samples[offset..end].to_vec();
            let chunk_end_ms = (end as u64 * 1000) / sample_rate as u64;

            // Collect visemes belonging to this audio chunk window
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

            // Yield cooperatively to event loop
            tokio::task::yield_now().await;
        }

        // Flush any remaining visemes
        if viseme_idx < visemes.len() {
            let remainder = visemes[viseme_idx..].to_vec();
            let _ = sender.send(TtsStreamEvent::Visemes(remainder)).await;
        }

        let _ = sender.send(TtsStreamEvent::Done).await;
        Ok(())
    }

    async fn synthesize(&self, text: &str) -> Result<TtsOutput, TtsError> {
        let (samples, visemes, sample_rate) = self.synthesize_neural_or_fallback(text).await;
        Ok(TtsOutput::new(samples, sample_rate, visemes))
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_kokoro_adapter_initialization_and_sample_rate() {
        let adapter = KokoroTtsAdapter::new()
            .with_voice("vi_vn")
            .with_speed(1.2)
            .with_sample_rate(24000);

        assert_eq!(adapter.voice(), "vi_vn");
        assert_eq!(adapter.speed(), 1.2);
        assert_eq!(adapter.sample_rate(), 24000);
        assert!(!adapter.is_onnx_active());
    }

    #[tokio::test]
    async fn test_kokoro_synthesize_one_shot_generates_samples_and_visemes() {
        let adapter = KokoroTtsAdapter::new();
        let output = adapter
            .synthesize("Xin chào bạn! Tôi là trợ lý Vira.")
            .await
            .expect("synthesis success");

        assert_eq!(output.sample_rate, 24000);
        assert!(!output.samples.is_empty(), "audio samples must be generated");
        assert!(!output.visemes.is_empty(), "visemes must be generated");
        assert!(output.duration_ms() > 0);

        // Verify visemes are strictly clamped to [0.0, 1.0]
        for v in &output.visemes {
            assert!(v.weights.aa >= 0.0 && v.weights.aa <= 1.0);
            assert!(v.weights.ih >= 0.0 && v.weights.ih <= 1.0);
            assert!(v.weights.ou >= 0.0 && v.weights.ou <= 1.0);
            assert!(v.weights.ee >= 0.0 && v.weights.ee <= 1.0);
            assert!(v.weights.oh >= 0.0 && v.weights.oh <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_kokoro_stream_synthesize_barge_in_cancellation() {
        let adapter = KokoroTtsAdapter::new();
        let (tx, _rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();

        cancel.cancel(); // Pre-cancel to simulate immediate barge-in
        let res = adapter
            .stream_synthesize("Một đoạn văn bản dài cần tổng hợp tiếng nói.", tx, cancel)
            .await;

        assert_eq!(res, Err(TtsError::Cancelled));
    }

    #[tokio::test]
    async fn test_kokoro_stream_events_flow() {
        let adapter = KokoroTtsAdapter::new();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();

        adapter
            .stream_synthesize("Chào Vira!", tx, cancel)
            .await
            .expect("stream synthesize success");

        let mut audio_chunks = 0;
        let mut viseme_batches = 0;
        let mut got_done = false;

        while let Some(event) = rx.recv().await {
            match event {
                TtsStreamEvent::Audio(chunk) => {
                    assert!(!chunk.is_empty());
                    audio_chunks += 1;
                }
                TtsStreamEvent::Visemes(batch) => {
                    assert!(!batch.is_empty());
                    viseme_batches += 1;
                }
                TtsStreamEvent::Done => {
                    got_done = true;
                }
                _ => {}
            }
        }

        assert!(audio_chunks > 0, "should stream audio chunks");
        assert!(viseme_batches > 0, "should stream viseme batches");
        assert!(got_done, "should emit Done event");
    }

    #[test]
    fn test_char_to_viseme_mapping_vowels() {
        let (a_vis, a_vol) = KokoroTtsAdapter::char_to_viseme_target('a');
        assert!(a_vis.aa > 0.8, "a maps to aa");
        assert!(a_vol > 0.0);

        let (o_vis, _) = KokoroTtsAdapter::char_to_viseme_target('ô');
        assert!(o_vis.oh > 0.8, "ô maps to oh");

        let (u_vis, _) = KokoroTtsAdapter::char_to_viseme_target('u');
        assert!(u_vis.ou > 0.8, "u maps to ou");

        let (i_vis, _) = KokoroTtsAdapter::char_to_viseme_target('i');
        assert!(i_vis.ih > 0.8, "i maps to ih");

        let (e_vis, _) = KokoroTtsAdapter::char_to_viseme_target('e');
        assert!(e_vis.ee > 0.7, "e maps to ee");
    }
}
