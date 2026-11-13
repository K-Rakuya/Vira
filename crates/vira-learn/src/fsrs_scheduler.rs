//! Free Spaced Repetition Scheduler (FSRS) implementation for Vira Companion.
//!
//! Provides memory stability and difficulty modeling, interval calculation,
//! and flashcard review transitions using the modern FSRS algorithm.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use vira_ports::store::Flashcard;

/// Card rating given by the learner during a review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CardRating {
    /// Complete memory lapse; failed to recall.
    Again = 1,
    /// Recalled with significant difficulty or hesitation.
    Hard = 2,
    /// Recalled successfully with standard effort.
    Good = 3,
    /// Instant and effortless recall.
    Easy = 4,
}

impl CardRating {
    /// Returns integer rating (1..=4).
    pub fn as_u32(&self) -> u32 {
        *self as u32
    }

    /// Converts integer (1..=4) to CardRating.
    pub fn from_u32(val: u32) -> Result<Self, RatingParseError> {
        match val {
            1 => Ok(CardRating::Again),
            2 => Ok(CardRating::Hard),
            3 => Ok(CardRating::Good),
            4 => Ok(CardRating::Easy),
            other => Err(RatingParseError(format!("invalid rating value: {other}"))),
        }
    }

    /// Returns lowercase string identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            CardRating::Again => "again",
            CardRating::Hard => "hard",
            CardRating::Good => "good",
            CardRating::Easy => "easy",
        }
    }
}

impl fmt::Display for CardRating {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Error returned when parsing an invalid rating.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[error("Rating parse error: {0}")]
pub struct RatingParseError(pub String);

impl FromStr for CardRating {
    type Err = RatingParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "1" | "again" => Ok(CardRating::Again),
            "2" | "hard" => Ok(CardRating::Hard),
            "3" | "good" => Ok(CardRating::Good),
            "4" | "easy" => Ok(CardRating::Easy),
            other => Err(RatingParseError(other.to_string())),
        }
    }
}

impl<'de> Deserialize<'de> for CardRating {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct CardRatingVisitor;

        impl<'de> serde::de::Visitor<'de> for CardRatingVisitor {
            type Value = CardRating;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str(
                    "a rating string (\"again\", \"hard\", \"good\", \"easy\") or integer (1..=4)",
                )
            }

            fn visit_u64<E>(self, v: u64) -> Result<CardRating, E>
            where
                E: serde::de::Error,
            {
                CardRating::from_u32(v as u32).map_err(|e| E::custom(e.0))
            }

            fn visit_i64<E>(self, v: i64) -> Result<CardRating, E>
            where
                E: serde::de::Error,
            {
                if v <= 0 {
                    return Err(E::custom(format!("invalid rating value: {v}")));
                }
                CardRating::from_u32(v as u32).map_err(|e| E::custom(e.0))
            }

            fn visit_str<E>(self, v: &str) -> Result<CardRating, E>
            where
                E: serde::de::Error,
            {
                v.parse::<CardRating>().map_err(|e| E::custom(e.0))
            }
        }

        deserializer.deserialize_any(CardRatingVisitor)
    }
}

impl Serialize for CardRating {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

/// Error returned when parsing an invalid review state.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[error("State parse error: {0}")]
pub struct StateParseError(pub String);

/// Review state of a card in the learning lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CardReviewState {
    /// Brand new card not yet reviewed.
    New = 0,
    /// Currently in learning acquisition phase.
    Learning = 1,
    /// Mastered card in standard spaced repetition review.
    Review = 2,
    /// Forgotten card undergoing relearning.
    Relearning = 3,
}

impl CardReviewState {
    /// Converts numeric code to state.
    pub fn from_u32(code: u32) -> Self {
        match code {
            0 => CardReviewState::New,
            1 => CardReviewState::Learning,
            2 => CardReviewState::Review,
            3 => CardReviewState::Relearning,
            _ => CardReviewState::Review,
        }
    }

    /// Converts state to numeric code.
    pub fn as_u32(&self) -> u32 {
        *self as u32
    }

    /// Returns lowercase string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            CardReviewState::New => "new",
            CardReviewState::Learning => "learning",
            CardReviewState::Review => "review",
            CardReviewState::Relearning => "relearning",
        }
    }
}

impl fmt::Display for CardReviewState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for CardReviewState {
    type Err = StateParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "0" | "new" => Ok(CardReviewState::New),
            "1" | "learning" => Ok(CardReviewState::Learning),
            "2" | "review" => Ok(CardReviewState::Review),
            "3" | "relearning" => Ok(CardReviewState::Relearning),
            other => Err(StateParseError(format!("invalid review state: {other}"))),
        }
    }
}

impl<'de> Deserialize<'de> for CardReviewState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct StateVisitor;

        impl<'de> serde::de::Visitor<'de> for StateVisitor {
            type Value = CardReviewState;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str(
                    "a state string (\"new\", \"learning\", \"review\", \"relearning\") or integer (0..=3)",
                )
            }

            fn visit_u64<E>(self, v: u64) -> Result<CardReviewState, E>
            where
                E: serde::de::Error,
            {
                Ok(CardReviewState::from_u32(v as u32))
            }

            fn visit_i64<E>(self, v: i64) -> Result<CardReviewState, E>
            where
                E: serde::de::Error,
            {
                if v < 0 {
                    return Err(E::custom(format!("invalid review state: {v}")));
                }
                Ok(CardReviewState::from_u32(v as u32))
            }

            fn visit_str<E>(self, v: &str) -> Result<CardReviewState, E>
            where
                E: serde::de::Error,
            {
                v.parse::<CardReviewState>().map_err(|e| E::custom(e.0))
            }
        }

        deserializer.deserialize_any(StateVisitor)
    }
}

impl Serialize for CardReviewState {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

/// Full record of a spaced repetition card with FSRS memory parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CardRecord {
    /// Unique card identifier.
    pub id: String,
    /// Front prompt (word, phrase, or sentence).
    pub front: String,
    /// Back explanation, meaning, or translation.
    pub back: String,
    /// Optional CEFR difficulty level (e.g. "B1", "C1").
    pub cefr_level: Option<String>,
    /// FSRS memory stability (retrievability half-life in days).
    pub stability: f32,
    /// FSRS difficulty rating [1.0, 10.0].
    pub difficulty: f32,
    /// Total successful review repetitions.
    pub reps: u32,
    /// Total lapses / forgot count.
    pub lapses: u32,
    /// Current review state in lifecycle.
    pub state: CardReviewState,
    /// Due epoch timestamp in milliseconds.
    pub due_timestamp: u64,
    /// Epoch timestamp of last review in milliseconds, if any.
    pub last_review: Option<u64>,
    /// Scheduled interval in days resulting from the most recent review.
    pub scheduled_days: u32,
}

impl CardRecord {
    /// Creates a new card record due immediately.
    pub fn new(front: impl Into<String>, back: impl Into<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            front: front.into(),
            back: back.into(),
            cefr_level: None,
            stability: 0.0,
            difficulty: 0.0,
            reps: 0,
            lapses: 0,
            state: CardReviewState::New,
            due_timestamp: now,
            last_review: None,
            scheduled_days: 0,
        }
    }

    /// Sets explicit card ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Sets CEFR difficulty level.
    pub fn with_cefr(mut self, cefr: impl Into<String>) -> Self {
        self.cefr_level = Some(cefr.into());
        self
    }

    /// Returns true if this card is currently due at `current_time_ms`.
    pub fn is_due(&self, current_time_ms: u64) -> bool {
        self.due_timestamp <= current_time_ms
    }

    /// Converts this record to the domain `Flashcard` representation.
    pub fn to_flashcard(&self) -> Flashcard {
        Flashcard {
            id: self.id.clone(),
            front: self.front.clone(),
            back: self.back.clone(),
            cefr_level: self.cefr_level.clone(),
            state: self.state.as_u32(),
            step: None,
            stability: self.stability,
            difficulty: self.difficulty,
            due: self.due_timestamp,
            last_review: self.last_review,
            reps: self.reps,
            lapses: self.lapses,
        }
    }

    /// Creates a `CardRecord` from a domain `Flashcard`.
    pub fn from_flashcard(card: &Flashcard) -> Self {
        let scheduled_days = match card.last_review {
            Some(last) if card.due > last => {
                let diff_ms = card.due - last;
                ((diff_ms as f64 + 43_200_000.0) / 86_400_000.0) as u32
            }
            _ => 0,
        };

        Self {
            id: card.id.clone(),
            front: card.front.clone(),
            back: card.back.clone(),
            cefr_level: card.cefr_level.clone(),
            stability: card.stability,
            difficulty: card.difficulty,
            reps: card.reps,
            lapses: card.lapses,
            state: CardReviewState::from_u32(card.state),
            due_timestamp: card.due,
            last_review: card.last_review,
            scheduled_days,
        }
    }
}

/// Preview of scheduled intervals (in days) for all possible ratings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreviewIntervals {
    /// Interval in days if user chooses Again.
    pub again: u32,
    /// Interval in days if user chooses Hard.
    pub hard: u32,
    /// Interval in days if user chooses Good.
    pub good: u32,
    /// Interval in days if user chooses Easy.
    pub easy: u32,
}

impl PreviewIntervals {
    /// Returns interval in days for a specific rating.
    pub fn for_rating(&self, rating: CardRating) -> u32 {
        match rating {
            CardRating::Again => self.again,
            CardRating::Hard => self.hard,
            CardRating::Good => self.good,
            CardRating::Easy => self.easy,
        }
    }
}

/// FSRS spaced repetition scheduler engine.
#[derive(Debug, Clone)]
pub struct FsrsScheduler {
    fsrs: fsrs::FSRS,
    desired_retention: f32,
}

impl Default for FsrsScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl FsrsScheduler {
    /// Creates a scheduler with standard retention rate (90%).
    pub fn new() -> Self {
        Self::with_retention(0.9)
    }

    /// Creates a scheduler with a custom desired retention rate between 0.70 and 0.98.
    pub fn with_retention(retention: f32) -> Self {
        let desired_retention = retention.clamp(0.70, 0.98);
        Self {
            fsrs: fsrs::FSRS::default(),
            desired_retention,
        }
    }

    /// Desired retention target [0.70, 0.98].
    pub fn desired_retention(&self) -> f32 {
        self.desired_retention
    }

    /// Creates a new card initialized with default state.
    pub fn new_card(
        &self,
        front: impl Into<String>,
        back: impl Into<String>,
    ) -> CardRecord {
        CardRecord::new(front, back)
    }

    /// Calculates preview intervals for all ratings using current system time.
    pub fn preview_intervals(&self, card: &CardRecord) -> PreviewIntervals {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.preview_intervals_at(card, now_ms)
    }

    /// Calculates preview intervals for all ratings at an explicit reference timestamp.
    pub fn preview_intervals_at(&self, card: &CardRecord, now_ms: u64) -> PreviewIntervals {
        let (again_item, hard_item, good_item, easy_item) = self.compute_next_states(card, now_ms);

        PreviewIntervals {
            again: again_item.interval.round().max(0.0) as u32,
            hard: hard_item.interval.round().max(1.0) as u32,
            good: good_item.interval.round().max(1.0) as u32,
            easy: easy_item.interval.round().max(1.0) as u32,
        }
    }

    /// Reviews a card using current system time.
    pub fn review(&self, card: &CardRecord, rating: CardRating) -> CardRecord {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.review_at(card, rating, now_ms)
    }

    /// Reviews a card at an explicit reference timestamp (in milliseconds).
    pub fn review_at(&self, card: &CardRecord, rating: CardRating, now_ms: u64) -> CardRecord {
        let (again_item, hard_item, good_item, easy_item) = self.compute_next_states(card, now_ms);

        let chosen_item = match rating {
            CardRating::Again => again_item,
            CardRating::Hard => hard_item,
            CardRating::Good => good_item,
            CardRating::Easy => easy_item,
        };

        let scheduled_days = if rating == CardRating::Again {
            // For lapses, interval can be 0 or 1 day
            chosen_item.interval.round().max(0.0) as u32
        } else {
            // Successful recall guarantees at least 1 day progression
            chosen_item.interval.round().max(1.0) as u32
        };

        // Milliseconds in a day = 86_400_000
        let due_timestamp = if scheduled_days == 0 {
            // 10 minutes immediate relearning window
            now_ms.saturating_add(10 * 60 * 1000)
        } else {
            now_ms.saturating_add((scheduled_days as u64).saturating_mul(86_400_000))
        };

        let (new_state, reps, lapses) = match rating {
            CardRating::Again => (CardReviewState::Relearning, card.reps, card.lapses + 1),
            CardRating::Hard => {
                let state = if card.state == CardReviewState::New {
                    CardReviewState::Learning
                } else {
                    CardReviewState::Review
                };
                (state, card.reps + 1, card.lapses)
            }
            CardRating::Good | CardRating::Easy => {
                (CardReviewState::Review, card.reps + 1, card.lapses)
            }
        };

        CardRecord {
            id: card.id.clone(),
            front: card.front.clone(),
            back: card.back.clone(),
            cefr_level: card.cefr_level.clone(),
            stability: chosen_item.memory.stability,
            difficulty: chosen_item.memory.difficulty,
            reps,
            lapses,
            state: new_state,
            due_timestamp,
            last_review: Some(now_ms),
            scheduled_days,
        }
    }

    fn compute_next_states(
        &self,
        card: &CardRecord,
        now_ms: u64,
    ) -> (fsrs::ItemState, fsrs::ItemState, fsrs::ItemState, fsrs::ItemState) {
        let (prev_state, elapsed_days) = if card.state == CardReviewState::New
            || card.last_review.is_none()
            || !card.stability.is_finite()
            || card.stability <= 0.0
        {
            (None, 0.0f32)
        } else {
            let last = card.last_review.unwrap_or(card.due_timestamp);
            let elapsed_ms = now_ms.saturating_sub(last);
            let elapsed = (elapsed_ms as f32) / (86_400_000.0);
            let diff = if card.difficulty.is_finite() {
                card.difficulty.clamp(1.0, 10.0)
            } else {
                5.0
            };
            let stab = card.stability.clamp(0.01, 36500.0);
            (
                Some(fsrs::MemoryState {
                    stability: stab,
                    difficulty: diff,
                    stability_fast: stab,
                }),
                elapsed.max(0.0),
            )
        };

        let next = self
            .fsrs
            .next_states_with_elapsed_days(prev_state, self.desired_retention, elapsed_days)
            .unwrap_or_else(|_| {
                // Fallback to initial state if parameters caused numerical issue
                self.fsrs
                    .next_states(None, self.desired_retention, 0)
                    .expect("default initial states are always valid")
            });

        (next.again, next.hard, next.good, next.easy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_retention_clamping() {
        let low = FsrsScheduler::with_retention(0.50);
        assert!((low.desired_retention() - 0.70).abs() < f32::EPSILON);

        let high = FsrsScheduler::with_retention(0.999);
        assert!((high.desired_retention() - 0.98).abs() < f32::EPSILON);

        let default_sched = FsrsScheduler::default();
        assert!((default_sched.desired_retention() - 0.90).abs() < f32::EPSILON);
    }

    #[test]
    fn test_card_record_is_due_logic() {
        let mut card = CardRecord::new("hello", "xin chào");
        card.due_timestamp = 1000;

        assert!(card.is_due(1000));
        assert!(card.is_due(1001));
        assert!(!card.is_due(999));
    }

    #[test]
    fn test_card_review_state_code_mapping() {
        assert_eq!(CardReviewState::from_u32(0), CardReviewState::New);
        assert_eq!(CardReviewState::from_u32(1), CardReviewState::Learning);
        assert_eq!(CardReviewState::from_u32(2), CardReviewState::Review);
        assert_eq!(CardReviewState::from_u32(3), CardReviewState::Relearning);
        assert_eq!(CardReviewState::from_u32(99), CardReviewState::Review);
    }

    #[test]
    fn test_immediate_relearning_window_for_again() {
        let scheduler = FsrsScheduler::new();
        let card = scheduler.new_card("test", "thử");
        let now = 1_000_000;
        let reviewed = scheduler.review_at(&card, CardRating::Again, now);

        assert_eq!(reviewed.lapses, 1);
        assert_eq!(reviewed.state, CardReviewState::Relearning);
        // If scheduled_days is 0, due_timestamp is set to now + 10 mins (600_000 ms)
        if reviewed.scheduled_days == 0 {
            assert_eq!(reviewed.due_timestamp, now + 600_000);
        } else {
            assert!(reviewed.due_timestamp > now);
        }
    }
}

