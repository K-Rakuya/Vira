//! Hexagonal Ports Definition for Vira AI Companion.
//!
//! Provides abstract async trait contracts (outbound & inbound ports) defining
//! the boundaries between the core domain, AI inference engines, audio I/O hardware,
//! and persistent storage.

#![deny(clippy::all)]

pub mod audio;
pub mod llm;
pub mod store;
pub mod stt;
pub mod tts;

pub use audio::{AudioCapturePort, AudioError, AudioPlaybackPort};
pub use llm::{LlmError, LlmPort, LlmStreamChunk};
pub use store::{Flashcard, SessionSummary, StoreError, StorePort, UserProfile};
pub use stt::{SttError, SttEvent, SttPort};
pub use tts::{TimedViseme, TtsAudioChunk, TtsError, TtsOutput, TtsPort, TtsStreamEvent};
