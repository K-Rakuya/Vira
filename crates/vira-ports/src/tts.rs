use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;
use vira_domain::viseme::VisemeWeights;

/// A viseme expression target aligned to a relative playback timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimedViseme {
    /// Milliseconds offset from the beginning of the synthesized audio track.
    pub timestamp_ms: u64,
    /// Blendshape weights (A, I, U, E, O) strictly clamped to [0.0, 1.0].
    pub weights: VisemeWeights,
}

impl TimedViseme {
    /// Creates a new timed viseme snapshot with clamped weights.
    pub fn new(timestamp_ms: u64, weights: VisemeWeights) -> Self {
        Self {
            timestamp_ms,
            weights: weights.clamped(),
        }
    }

    /// Creates a silent / closed mouth viseme at the given timestamp.
    pub fn zero(timestamp_ms: u64) -> Self {
        Self {
            timestamp_ms,
            weights: VisemeWeights::zero(),
        }
    }
}

/// A chunk of raw PCM f32 audio produced during speech synthesis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtsAudioChunk {
    /// Interleaved PCM 32-bit float samples normalized in [-1.0, 1.0].
    pub samples: Vec<f32>,
    /// Sample rate in Hz (e.g. 24000).
    pub sample_rate: u32,
}

impl TtsAudioChunk {
    /// Creates a new TTS audio chunk.
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// Returns the approximate duration in milliseconds for this chunk.
    pub fn duration_ms(&self) -> u64 {
        if self.sample_rate == 0 {
            0
        } else {
            (self.samples.len() as u64 * 1000) / self.sample_rate as u64
        }
    }

    /// Returns true if this chunk contains no audio samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Returns the number of PCM samples in this chunk.
    pub fn len(&self) -> usize {
        self.samples.len()
    }
}

/// Events emitted during streaming TTS generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TtsStreamEvent {
    /// PCM audio samples to be enqueued for playback.
    Audio(TtsAudioChunk),
    /// Single timed viseme mouth shape event.
    Viseme(TimedViseme),
    /// Batch of timed viseme frames.
    Visemes(Vec<TimedViseme>),
    /// Marks the end of synthesis for the requested text.
    Done,
}

/// Complete one-shot speech synthesis output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtsOutput {
    /// Complete PCM audio samples.
    pub samples: Vec<f32>,
    /// Sample rate of the synthesized audio in Hz.
    pub sample_rate: u32,
    /// Full viseme timeline for avatar lip-sync animation.
    pub visemes: Vec<TimedViseme>,
}

impl TtsOutput {
    /// Creates a new TTS output struct.
    pub fn new(samples: Vec<f32>, sample_rate: u32, visemes: Vec<TimedViseme>) -> Self {
        Self {
            samples,
            sample_rate,
            visemes,
        }
    }

    /// Returns duration of the full synthesis in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        if self.sample_rate == 0 {
            0
        } else {
            (self.samples.len() as u64 * 1000) / self.sample_rate as u64
        }
    }

    /// Returns true if this output contains no audio samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Returns the number of PCM samples in this output.
    pub fn len(&self) -> usize {
        self.samples.len()
    }
}

/// Errors occurring during speech synthesis.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TtsError {
    #[error("TTS synthesis was cancelled")]
    Cancelled,
    #[error("TTS model synthesis error: {0}")]
    Synthesis(String),
    #[error("TTS channel error: {0}")]
    Channel(String),
    #[error("Other TTS error: {0}")]
    Other(String),
}

/// Hexagonal Outbound Port for Text-To-Speech (TTS) synthesis and Viseme generation.
///
/// Implemented by local CPU/GPU neural models (e.g. Kokoro ONNX) or remote TTS providers.
#[async_trait]
pub trait TtsPort: Send + Sync {
    /// Streams synthesized audio PCM chunks and viseme alignment frames to `sender`.
    ///
    /// Respects `cancel` for instant cutoff during barge-in.
    async fn stream_synthesize(
        &self,
        text: &str,
        sender: Sender<TtsStreamEvent>,
        cancel: CancellationToken,
    ) -> Result<(), TtsError>;

    /// Synthesizes complete phrase in one-shot mode.
    async fn synthesize(&self, text: &str) -> Result<TtsOutput, TtsError>;

    /// Native output sample rate of the synthesizer in Hz (e.g. 24000).
    fn sample_rate(&self) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timed_viseme_and_audio_duration() {
        let weights = VisemeWeights::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let timed = TimedViseme::new(150, weights);
        assert_eq!(timed.timestamp_ms, 150);
        assert_eq!(timed.weights.aa, 1.0);

        let zero_viseme = TimedViseme::zero(200);
        assert_eq!(zero_viseme.timestamp_ms, 200);
        assert!(zero_viseme.weights.is_silent());

        let audio = TtsAudioChunk::new(vec![0.0; 48000], 48000);
        assert_eq!(audio.duration_ms(), 1000);
        assert_eq!(audio.len(), 48000);
        assert!(!audio.is_empty());

        let audio_zero_rate = TtsAudioChunk::new(vec![0.0; 100], 0);
        assert_eq!(audio_zero_rate.duration_ms(), 0);

        let output = TtsOutput {
            samples: vec![0.0; 24000],
            sample_rate: 24000,
            visemes: vec![timed.clone()],
        };
        assert_eq!(output.duration_ms(), 1000);
        assert_eq!(output.len(), 24000);
        assert_eq!(output.visemes.len(), 1);

        let output_zero_rate = TtsOutput {
            samples: vec![0.0; 100],
            sample_rate: 0,
            visemes: vec![],
        };
        assert_eq!(output_zero_rate.duration_ms(), 0);
    }

    #[test]
    fn test_tts_serde_roundtrip() {
        let timed = TimedViseme::new(200, VisemeWeights::new(0.5, 0.2, 0.0, 0.0, 0.0));
        let json = serde_json::to_string(&timed).expect("serialize timed");
        let decoded: TimedViseme = serde_json::from_str(&json).expect("deserialize timed");
        assert_eq!(timed, decoded);

        let audio = TtsAudioChunk::new(vec![0.1, -0.2, 0.3], 24000);
        let audio_json = serde_json::to_string(&audio).expect("serialize audio");
        let audio_dec: TtsAudioChunk = serde_json::from_str(&audio_json).expect("deserialize audio");
        assert_eq!(audio, audio_dec);

        let event = TtsStreamEvent::Audio(audio.clone());
        let event_json = serde_json::to_string(&event).expect("serialize event");
        let event_dec: TtsStreamEvent = serde_json::from_str(&event_json).expect("deserialize event");
        assert_eq!(event, event_dec);

        let output = TtsOutput::new(vec![0.1, 0.2], 24000, vec![timed]);
        let output_json = serde_json::to_string(&output).expect("serialize output");
        let output_dec: TtsOutput = serde_json::from_str(&output_json).expect("deserialize output");
        assert_eq!(output, output_dec);
    }
}
