use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Conversation states of the Vira assistant loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ConversationState {
    #[default]
    Idle,
    Listening,
    Thinking,
    Speaking,
    Interrupted,
}

impl ConversationState {
    /// Returns true if the system can accept/process incoming user speech or audio in this state.
    /// Active during Listening (recording turn), Idle (wake word / start), Speaking (barge-in),
    /// Thinking (barge-in / cancel), and Interrupted (resuming).
    pub fn can_listen(&self) -> bool {
        matches!(
            self,
            ConversationState::Idle
                | ConversationState::Listening
                | ConversationState::Speaking
                | ConversationState::Interrupted
                | ConversationState::Thinking
        )
    }

    /// Returns true if user speech should trigger a barge-in interruption.
    pub fn can_barge_in(&self) -> bool {
        matches!(
            self,
            ConversationState::Speaking | ConversationState::Thinking
        )
    }

    /// Returns true if the system is completely idle.
    pub fn is_idle(&self) -> bool {
        matches!(self, ConversationState::Idle)
    }

    /// Returns true if the system is currently listening to user speech.
    pub fn is_listening(&self) -> bool {
        matches!(self, ConversationState::Listening)
    }

    /// Returns true if the LLM/TTS pipeline is currently thinking/processing.
    pub fn is_thinking(&self) -> bool {
        matches!(self, ConversationState::Thinking)
    }

    /// Returns true if the assistant is actively speaking / rendering audio.
    pub fn is_speaking(&self) -> bool {
        matches!(self, ConversationState::Speaking)
    }

    /// Returns true if playback was interrupted (e.g. by barge-in).
    pub fn is_interrupted(&self) -> bool {
        matches!(self, ConversationState::Interrupted)
    }

    /// Returns the static snake_case string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            ConversationState::Idle => "idle",
            ConversationState::Listening => "listening",
            ConversationState::Thinking => "thinking",
            ConversationState::Speaking => "speaking",
            ConversationState::Interrupted => "interrupted",
        }
    }

    /// Validates and performs state transition to the target state.
    pub fn transition_to(&self, target: ConversationState) -> Result<ConversationState, StateTransitionError> {
        let is_valid = match (self, target) {
            // From Idle: can start listening (voice) or start thinking (typed text chat from HUD)
            (ConversationState::Idle, ConversationState::Listening)
            | (ConversationState::Idle, ConversationState::Thinking) => true,

            // From Listening: can move to thinking (turn completed), idle (silence timeout), or interrupted (user cancelled)
            (ConversationState::Listening, ConversationState::Thinking)
            | (ConversationState::Listening, ConversationState::Idle)
            | (ConversationState::Listening, ConversationState::Interrupted) => true,

            // From Thinking: can start speaking (LLM/TTS generated), be interrupted, return to idle (empty/error), or barge-in directly to listening
            (ConversationState::Thinking, ConversationState::Speaking)
            | (ConversationState::Thinking, ConversationState::Interrupted)
            | (ConversationState::Thinking, ConversationState::Listening)
            | (ConversationState::Thinking, ConversationState::Idle) => true,

            // From Speaking: can complete to idle, be interrupted (barge-in), or directly re-enter listening
            (ConversationState::Speaking, ConversationState::Idle)
            | (ConversationState::Speaking, ConversationState::Interrupted)
            | (ConversationState::Speaking, ConversationState::Listening) => true,

            // From Interrupted: can resume listening, reset to idle, or proceed to thinking
            (ConversationState::Interrupted, ConversationState::Listening)
            | (ConversationState::Interrupted, ConversationState::Idle)
            | (ConversationState::Interrupted, ConversationState::Thinking) => true,

            _ => false,
        };

        if is_valid {
            Ok(target)
        } else {
            Err(StateTransitionError::InvalidTransition {
                from: *self,
                to: target,
            })
        }
    }

    pub fn transition_to_listening(&self) -> Result<ConversationState, StateTransitionError> {
        self.transition_to(ConversationState::Listening)
    }

    pub fn transition_to_thinking(&self) -> Result<ConversationState, StateTransitionError> {
        self.transition_to(ConversationState::Thinking)
    }

    pub fn transition_to_speaking(&self) -> Result<ConversationState, StateTransitionError> {
        self.transition_to(ConversationState::Speaking)
    }

    pub fn transition_to_interrupted(&self) -> Result<ConversationState, StateTransitionError> {
        self.transition_to(ConversationState::Interrupted)
    }

    pub fn transition_to_idle(&self) -> Result<ConversationState, StateTransitionError> {
        self.transition_to(ConversationState::Idle)
    }
}

impl std::fmt::Display for ConversationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Error returned when parsing a conversation state string fails.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[error("invalid conversation state: '{0}'")]
pub struct StateParseError(pub String);

impl FromStr for ConversationState {
    type Err = StateParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "idle" => Ok(ConversationState::Idle),
            "listening" => Ok(ConversationState::Listening),
            "thinking" => Ok(ConversationState::Thinking),
            "speaking" => Ok(ConversationState::Speaking),
            "interrupted" => Ok(ConversationState::Interrupted),
            other => Err(StateParseError(other.to_string())),
        }
    }
}

/// Errors occurring during illegal state transitions.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateTransitionError {
    #[error("invalid conversation state transition from {from:?} to {to:?}")]
    InvalidTransition {
        from: ConversationState,
        to: ConversationState,
    },
}

/// Primary emotional states for avatar expressions (aligned with VRM expression presets).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CharacterEmotion {
    #[default]
    Neutral,
    Happy,
    Sad,
    Angry,
    Surprised,
    Relaxed,
}

impl CharacterEmotion {
    /// Returns the static snake_case string representation matching standard VRM 1.0 presets.
    pub fn as_str(&self) -> &'static str {
        match self {
            CharacterEmotion::Neutral => "neutral",
            CharacterEmotion::Happy => "happy",
            CharacterEmotion::Sad => "sad",
            CharacterEmotion::Angry => "angry",
            CharacterEmotion::Surprised => "surprised",
            CharacterEmotion::Relaxed => "relaxed",
        }
    }
}

impl std::fmt::Display for CharacterEmotion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Error returned when parsing an emotion string fails.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[error("invalid character emotion: '{0}'")]
pub struct EmotionParseError(pub String);

impl FromStr for CharacterEmotion {
    type Err = EmotionParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "neutral" => Ok(CharacterEmotion::Neutral),
            "happy" => Ok(CharacterEmotion::Happy),
            "sad" => Ok(CharacterEmotion::Sad),
            "angry" => Ok(CharacterEmotion::Angry),
            "surprised" => Ok(CharacterEmotion::Surprised),
            "relaxed" => Ok(CharacterEmotion::Relaxed),
            other => Err(EmotionParseError(other.to_string())),
        }
    }
}

/// Event emitted whenever conversation state changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateTransitionEvent {
    pub from: ConversationState,
    pub to: ConversationState,
    pub timestamp: u64,
}

impl StateTransitionEvent {
    pub fn new(from: ConversationState, to: ConversationState) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self { from, to, timestamp }
    }

    pub fn with_timestamp(from: ConversationState, to: ConversationState, timestamp: u64) -> Self {
        Self { from, to, timestamp }
    }
}
