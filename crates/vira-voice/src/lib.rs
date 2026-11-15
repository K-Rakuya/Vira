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
pub use capture::{
    resample_linear, CaptureConfig, CaptureEngine, LinearResampler, VirtualAudioFeed,
};
pub use chunker::{ChunkerConfig, SentenceChunker};
pub use clock::MasterAudioClock;
pub use gop::{
    dtw_distance_rolling, phoneme_to_viseme, phoneme_to_vrm_preset, score_to_cefr_level,
    split_phoneme_base_and_tag, AccuracyLevel, DtwMatrix, FluencyMetrics, GopConfig, GopEvaluator,
    PhonemeResult, PhonemeScore, PronunciationAssessment, PronunciationReport, PronunciationScore,
    WordAssessment,
};
pub use phoneme_aligner::{
    AlignmentError, AlignmentMatch, CtcAligner, DtwAligner, DtwResult, G2pConverter, MatchStatus,
    PhonemeAligner, PhonemeFrameAlignment, PhonemeLanguage,
};
pub use playback::{PlaybackConfig, PlaybackEngine};
pub use ring_buffer::{AlignedRingState, AudioConsumer, AudioProducer, AudioRingBuffer, CachePadded};
pub use stream_parser::{ParsedStreamEvent, StreamParser};
