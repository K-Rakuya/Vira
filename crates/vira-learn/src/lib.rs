//! Spaced Repetition (FSRS) and Learner Profile crate for Vira AI Companion.
//!
//! Features:
//! - FSRS (Free Spaced Repetition Scheduler) v4/v5/v7 memory modeling.
//! - CEFR language proficiency tracking and dynamic pedagogical prompt shaping.
//! - Conversational recast directives (correcting learner mistakes naturally without breaking immersion).
//! - Hexagonal persistence integration bridging FSRS to `StorePort::Flashcard`.

#![deny(clippy::all)]

pub mod fsrs_scheduler;
pub mod service;
pub mod tutor_policies;

pub use fsrs_scheduler::{
    CardRating, CardRecord, CardReviewState, FsrsScheduler, PreviewIntervals, RatingParseError,
    StateParseError,
};
pub use service::{LearnError, SpacedRepetitionPort, SpacedRepetitionService};
pub use tutor_policies::{
    CefrLevel, CefrParseError, GrammarIssue, LearnerEvaluation, RecastMode, TutorPolicyManager,
};
