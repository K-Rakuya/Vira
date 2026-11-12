use vira_domain::message::{Message, Role};
use vira_domain::state::{CharacterEmotion, ConversationState, StateTransitionError, StateTransitionEvent};
use vira_domain::viseme::VisemeWeights;

#[test]
fn test_conversation_state_lifecycle() {
    let mut state = ConversationState::Idle;
    assert!(state.can_listen());
    state = state.transition_to_listening().expect("valid transition");
    assert_eq!(state, ConversationState::Listening);
    state = state.transition_to_thinking().expect("valid transition");
    assert_eq!(state, ConversationState::Thinking);
    state = state.transition_to_speaking().expect("valid transition");
    assert_eq!(state, ConversationState::Speaking);
}

#[test]
fn test_viseme_normalization() {
    // Parameters: (aa, ih, ou, ee, oh)
    let viseme = VisemeWeights::new(1.2, 0.0, 0.0, 0.5, -0.2);
    assert_eq!(viseme.aa, 1.0);
    assert_eq!(viseme.oh, 0.0);
    assert_eq!(viseme.ee, 0.5);
    assert_eq!(viseme.ou, 0.0);
    assert_eq!(viseme.ih, 0.0);
}

#[test]
fn test_barge_in_transition() {
    let mut state = ConversationState::Speaking;
    assert!(!state.is_idle());
    state = state.transition_to_interrupted().expect("valid barge-in transition");
    assert_eq!(state, ConversationState::Interrupted);
    assert!(state.can_listen());
    state = state.transition_to_listening().expect("resume listening");
    assert_eq!(state, ConversationState::Listening);
}

#[test]
fn test_invalid_state_transition() {
    let state = ConversationState::Idle;
    let err = state.transition_to_speaking().unwrap_err();
    assert_eq!(
        err,
        StateTransitionError::InvalidTransition {
            from: ConversationState::Idle,
            to: ConversationState::Speaking,
        }
    );
}

#[test]
fn test_message_creation_and_defaults() {
    let msg = Message::new(Role::User, "Hello Vira");
    assert_eq!(msg.role, Role::User);
    assert_eq!(msg.raw_content, "Hello Vira");
    assert_eq!(msg.speakable_content, "Hello Vira");
    assert!(!msg.id.is_empty());
    assert!(msg.timestamp > 0);

    let assistant_msg = Message::new(Role::Assistant, "Hello! [wave]")
        .with_speakable_content("Hello!");
    assert_eq!(assistant_msg.raw_content, "Hello! [wave]");
    assert_eq!(assistant_msg.speakable_content, "Hello!");
}

#[test]
fn test_message_serde() {
    let msg = Message::new(Role::User, "Hello Vira");
    let json = serde_json::to_string(&msg).expect("serialize message");
    let decoded: Message = serde_json::from_str(&json).expect("deserialize message");
    assert_eq!(decoded.id, msg.id);
    assert_eq!(decoded.role, msg.role);
    assert_eq!(decoded.raw_content, msg.raw_content);
    assert_eq!(decoded.speakable_content, msg.speakable_content);
    assert_eq!(decoded.timestamp, msg.timestamp);
}

#[test]
fn test_character_emotion_roundtrip() {
    let emotions = [
        CharacterEmotion::Neutral,
        CharacterEmotion::Happy,
        CharacterEmotion::Sad,
        CharacterEmotion::Angry,
        CharacterEmotion::Surprised,
        CharacterEmotion::Relaxed,
    ];

    for emotion in emotions {
        let json = serde_json::to_string(&emotion).expect("serialize emotion");
        let decoded: CharacterEmotion = serde_json::from_str(&json).expect("deserialize emotion");
        assert_eq!(decoded, emotion);
    }
}

#[test]
fn test_viseme_edge_cases() {
    let zero = VisemeWeights::zero();
    assert_eq!(zero.aa, 0.0);
    assert_eq!(zero.ih, 0.0);
    assert_eq!(zero.ou, 0.0);
    assert_eq!(zero.ee, 0.0);
    assert_eq!(zero.oh, 0.0);

    // Test clamping of excessive and negative values
    let clamped = VisemeWeights::new(5.0, -10.0, 1.0, 0.5, 999.0);
    assert_eq!(clamped.aa, 1.0);
    assert_eq!(clamped.ih, 0.0);
    assert_eq!(clamped.ou, 1.0);
    assert_eq!(clamped.ee, 0.5);
    assert_eq!(clamped.oh, 1.0);
}

#[test]
fn test_state_transition_event() {
    let event = StateTransitionEvent::new(ConversationState::Idle, ConversationState::Listening);
    assert_eq!(event.from, ConversationState::Idle);
    assert_eq!(event.to, ConversationState::Listening);
    assert!(event.timestamp > 0);
}
