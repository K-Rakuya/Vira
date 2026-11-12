use serde::{Deserialize, Serialize};

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
    /// Returns true if the system can accept/process incoming user speech in this state.
    /// Barge-in is supported during Speaking and Interrupted states.
    pub fn can_listen(&self) -> bool {
        matches!(
            self,
            ConversationState::Idle
                | ConversationState::Speaking
                | ConversationState::Interrupted
                | ConversationState::Thinking
        )
    }

    /// Returns true if the system is completely idle.
    pub fn is_idle(&self) -> bool {
        matches!(self, ConversationState::Idle)
    }

    /// Validates and performs state transition to the target state.
    pub fn transition_to(&self, target: ConversationState) -> Result<ConversationState, StateTransitionError> {
        let is_valid = match (self, target) {
            // From Idle: can start listening
            (ConversationState::Idle, ConversationState::Listening) => true,

            // From Listening: can move to thinking (turn completed) or idle (timeout/silence) or interrupted (cancelled)
            (ConversationState::Listening, ConversationState::Thinking)
            | (ConversationState::Listening, ConversationState::Idle)
            | (ConversationState::Listening, ConversationState::Interrupted) => true,

            // From Thinking: can start speaking (LLM/TTS generated) or be interrupted or return to idle
            (ConversationState::Thinking, ConversationState::Speaking)
            | (ConversationState::Thinking, ConversationState::Interrupted)
            | (ConversationState::Thinking, ConversationState::Idle) => true,

            // From Speaking: can complete and become idle, or be interrupted (barge-in), or directly re-enter listening
            (ConversationState::Speaking, ConversationState::Idle)
            | (ConversationState::Speaking, ConversationState::Interrupted)
            | (ConversationState::Speaking, ConversationState::Listening) => true,

            // From Interrupted: can resume listening or reset to idle or proceed to thinking
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

/// Errors occurring during illegal state transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateTransitionError {
    InvalidTransition {
        from: ConversationState,
        to: ConversationState,
    },
}

impl std::fmt::Display for StateTransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid conversation state transition from {from:?} to {to:?}")
            }
        }
    }
}

impl std::error::Error for StateTransitionError {}

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
}
