//! Hexagonal Domain Core for Vira AI Companion.
//!
//! Pure business logic and domain entities with zero external UI or audio I/O dependencies.

#![deny(clippy::all)]

pub mod message;
pub mod state;
pub mod viseme;

pub use message::{Message, Role};
pub use state::{
    CharacterEmotion, ConversationState, StateTransitionError, StateTransitionEvent,
};
pub use viseme::VisemeWeights;
