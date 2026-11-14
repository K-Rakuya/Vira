pub mod barge_in;
pub mod capture;
pub mod chunker;
pub mod clock;
pub mod gop;
pub mod phoneme_aligner;
pub mod playback;
pub mod ring_buffer;
pub mod stream_parser;

pub use barge_in::{BargeInController, BargeInReport};
pub use capture::{CaptureConfig, CaptureEngine, VirtualAudioFeed};
pub use chunker::{ChunkerConfig, SentenceChunker};
pub use clock::MasterAudioClock;
pub use gop::{
    phoneme_to_viseme, phoneme_to_vrm_preset, split_phoneme_base_and_tag, AccuracyLevel,
    GopConfig, GopEvaluator, PhonemeResult, PhonemeScore, PronunciationAssessment,
    PronunciationScore,
};
pub use phoneme_aligner::{
    AlignmentError, AlignmentMatch, CtcAligner, DtwAligner, DtwResult, G2pConverter, MatchStatus,
    PhonemeAligner, PhonemeFrameAlignment, PhonemeLanguage,
};
pub use playback::{PlaybackConfig, PlaybackEngine};
pub use ring_buffer::{AudioConsumer, AudioProducer, AudioRingBuffer};
pub use stream_parser::{ParsedStreamEvent, StreamParser};
