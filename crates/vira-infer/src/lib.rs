pub mod kokoro_tts;
pub mod openai_compat;
pub mod whisper_stt;

pub use kokoro_tts::{
    KokoroBackend, KokoroConfig, KokoroTtsAdapter, KOKORO_DEFAULT_SAMPLE_RATE,
    VISEME_TICK_INTERVAL_MS, VISEME_TICK_RATE_HZ,
};
pub use openai_compat::{
    resolve_chat_url, HttpTransport, MockLlmTransport, NativeHttpTransport, OpenAiCompatConfig,
    OpenAiCompatLlmAdapter, ReqwestHttpTransport, StreamingMode, StreamingTagExtractor,
};
pub use whisper_stt::{
    WhisperBackend, WhisperConfig, WhisperSttAdapter, DEFAULT_VAD_THRESHOLD,
    WHISPER_DEFAULT_SAMPLE_RATE,
};
