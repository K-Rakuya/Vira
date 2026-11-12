use std::collections::HashSet;
use std::thread;
use vira_domain::message::{Message, Role, RoleParseError};
use vira_domain::state::{
    CharacterEmotion, ConversationState, EmotionParseError, StateParseError, StateTransitionError,
    StateTransitionEvent,
};
use vira_domain::viseme::VisemeWeights;

#[test]
fn test_conversation_state_lifecycle() {
    let mut state = ConversationState::Idle;
    assert!(state.can_listen());
    assert!(state.is_idle());
    assert!(!state.is_listening());
    assert!(!state.can_barge_in());

    // Transition to listening
    state = state.transition_to_listening().expect("valid transition to listening");
    assert_eq!(state, ConversationState::Listening);
    assert!(state.is_listening());
    // CRITICAL: Listening state must be able to accept/process incoming speech
    assert!(state.can_listen(), "ConversationState::Listening must return true for can_listen");

    // Transition to thinking
    state = state.transition_to_thinking().expect("valid transition to thinking");
    assert_eq!(state, ConversationState::Thinking);
    assert!(state.is_thinking());
    assert!(state.can_listen());
    assert!(state.can_barge_in());

    // Transition to speaking
    state = state.transition_to_speaking().expect("valid transition to speaking");
    assert_eq!(state, ConversationState::Speaking);
    assert!(state.is_speaking());
    assert!(state.can_listen());
    assert!(state.can_barge_in());

    // Completion back to idle
    state = state.transition_to_idle().expect("valid completion to idle");
    assert_eq!(state, ConversationState::Idle);
    assert!(state.is_idle());
}

#[test]
fn test_text_chat_state_lifecycle() {
    // When user types a prompt from Cosmic HUD (no audio recording), state moves Idle -> Thinking directly
    let mut state = ConversationState::Idle;
    state = state.transition_to_thinking().expect("valid text chat turn transition: Idle -> Thinking");
    assert_eq!(state, ConversationState::Thinking);

    // Thinking -> Speaking (assistant streams response)
    state = state.transition_to_speaking().expect("Thinking -> Speaking");
    assert_eq!(state, ConversationState::Speaking);

    // Speaking -> Idle (assistant finishes speaking)
    state = state.transition_to_idle().expect("Speaking -> Idle");
    assert_eq!(state, ConversationState::Idle);
}

#[test]
fn test_barge_in_and_thinking_interrupt() {
    // 1. Barge-in during Speaking via Interrupted
    let mut state = ConversationState::Speaking;
    assert!(!state.is_idle());
    state = state.transition_to_interrupted().expect("valid barge-in transition");
    assert_eq!(state, ConversationState::Interrupted);
    assert!(state.is_interrupted());
    assert!(state.can_listen());
    state = state.transition_to_listening().expect("resume listening");
    assert_eq!(state, ConversationState::Listening);

    // 2. Direct barge-in from Speaking directly to Listening
    let speaking = ConversationState::Speaking;
    let direct_listening = speaking.transition_to_listening().expect("direct barge-in Speaking -> Listening");
    assert_eq!(direct_listening, ConversationState::Listening);

    // 3. User barge-in during Thinking directly to Listening
    let thinking = ConversationState::Thinking;
    let thinking_barge_in = thinking.transition_to_listening().expect("barge-in Thinking -> Listening");
    assert_eq!(thinking_barge_in, ConversationState::Listening);

    // 4. Cancellation during Thinking
    let cancelled = thinking.transition_to_interrupted().expect("Thinking -> Interrupted");
    assert_eq!(cancelled, ConversationState::Interrupted);

    // 5. Network disconnect / empty response during Thinking aborts to Idle
    let aborted = thinking.transition_to_idle().expect("Thinking -> Idle");
    assert_eq!(aborted, ConversationState::Idle);
}

#[test]
fn test_invalid_state_transitions() {
    // Idle cannot directly speak
    let state = ConversationState::Idle;
    let err = state.transition_to_speaking().unwrap_err();
    assert_eq!(
        err,
        StateTransitionError::InvalidTransition {
            from: ConversationState::Idle,
            to: ConversationState::Speaking,
        }
    );

    // Idle cannot be interrupted
    assert!(matches!(
        state.transition_to_interrupted().unwrap_err(),
        StateTransitionError::InvalidTransition {
            from: ConversationState::Idle,
            to: ConversationState::Interrupted
        }
    ));

    // Listening cannot jump directly to speaking
    let listening = ConversationState::Listening;
    assert!(matches!(
        listening.transition_to_speaking().unwrap_err(),
        StateTransitionError::InvalidTransition {
            from: ConversationState::Listening,
            to: ConversationState::Speaking
        }
    ));

    // Self transitions are invalid
    assert!(matches!(
        state.transition_to(ConversationState::Idle).unwrap_err(),
        StateTransitionError::InvalidTransition {
            from: ConversationState::Idle,
            to: ConversationState::Idle
        }
    ));
}

#[test]
fn test_conversation_state_display_and_parse() {
    let states = [
        (ConversationState::Idle, "idle"),
        (ConversationState::Listening, "listening"),
        (ConversationState::Thinking, "thinking"),
        (ConversationState::Speaking, "speaking"),
        (ConversationState::Interrupted, "interrupted"),
    ];

    for (state, s) in states {
        assert_eq!(state.as_str(), s);
        assert_eq!(state.to_string(), s);
        assert_eq!(s.parse::<ConversationState>().unwrap(), state);
        assert_eq!(s.to_uppercase().parse::<ConversationState>().unwrap(), state);

        let json = serde_json::to_string(&state).unwrap();
        let decoded: ConversationState = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, state);
    }

    assert!(matches!(
        "unknown_state".parse::<ConversationState>(),
        Err(StateParseError(_))
    ));
}

#[test]
fn test_character_emotion_roundtrip_and_parse() {
    let emotions = [
        (CharacterEmotion::Neutral, "neutral"),
        (CharacterEmotion::Happy, "happy"),
        (CharacterEmotion::Sad, "sad"),
        (CharacterEmotion::Angry, "angry"),
        (CharacterEmotion::Surprised, "surprised"),
        (CharacterEmotion::Relaxed, "relaxed"),
    ];

    for (emotion, s) in emotions {
        assert_eq!(emotion.as_str(), s);
        assert_eq!(emotion.to_string(), s);
        assert_eq!(s.parse::<CharacterEmotion>().unwrap(), emotion);
        assert_eq!(s.to_uppercase().parse::<CharacterEmotion>().unwrap(), emotion);

        let json = serde_json::to_string(&emotion).expect("serialize emotion");
        let decoded: CharacterEmotion = serde_json::from_str(&json).expect("deserialize emotion");
        assert_eq!(decoded, emotion);
    }

    assert!(matches!(
        "disgusted".parse::<CharacterEmotion>(),
        Err(EmotionParseError(_))
    ));
}

#[test]
fn test_role_display_and_parse() {
    let roles = [
        (Role::System, "system"),
        (Role::User, "user"),
        (Role::Assistant, "assistant"),
        (Role::Tool, "tool"),
    ];

    for (role, s) in roles {
        assert_eq!(role.as_str(), s);
        assert_eq!(role.to_string(), s);
        assert_eq!(s.parse::<Role>().unwrap(), role);
        assert_eq!(s.to_uppercase().parse::<Role>().unwrap(), role);

        let json = serde_json::to_string(&role).unwrap();
        let decoded: Role = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, role);
    }

    assert!(matches!("admin".parse::<Role>(), Err(RoleParseError(_))));
}

#[test]
fn test_viseme_normalization() {
    // Parameters: (aa, ih, ou, ee, oh)
    let viseme = VisemeWeights::new(1.2, 0.0, 0.0, 0.5, -0.2);
    assert_eq!(viseme.aa, 1.0);
    assert_eq!(viseme.ih, 0.0);
    assert_eq!(viseme.ou, 0.0);
    assert_eq!(viseme.ee, 0.5);
    assert_eq!(viseme.oh, 0.0);

    // Using from_vowels alias (A, I, U, E, O)
    let vowels = VisemeWeights::from_vowels(1.2, 0.0, 0.0, 0.5, -0.2);
    assert_eq!(vowels, viseme);

    // Using fluent builder methods
    let fluent = VisemeWeights::zero()
        .with_aa(1.2)
        .with_ee(0.5)
        .with_oh(-0.2);
    assert_eq!(fluent.aa, 1.0);
    assert_eq!(fluent.ee, 0.5);
    assert_eq!(fluent.oh, 0.0);
    assert_eq!(fluent.ih, 0.0);
    assert_eq!(fluent.ou, 0.0);
}

#[test]
fn test_viseme_edge_cases() {
    let zero = VisemeWeights::zero();
    assert_eq!(zero.aa, 0.0);
    assert_eq!(zero.ih, 0.0);
    assert_eq!(zero.ou, 0.0);
    assert_eq!(zero.ee, 0.0);
    assert_eq!(zero.oh, 0.0);
    assert!(zero.is_silent());

    // Test clamping of excessive and negative values and NaN / Inf
    let clamped = VisemeWeights::new(5.0, -10.0, 1.0, 0.5, 999.0);
    assert_eq!(clamped.aa, 1.0);
    assert_eq!(clamped.ih, 0.0);
    assert_eq!(clamped.ou, 1.0);
    assert_eq!(clamped.ee, 0.5);
    assert_eq!(clamped.oh, 1.0);
    assert!(!clamped.is_silent());

    let nan_viseme = VisemeWeights::new(f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.3, -1.0);
    assert_eq!(nan_viseme.aa, 0.0);
    assert_eq!(nan_viseme.ih, 1.0);
    assert_eq!(nan_viseme.ou, 0.0);
    assert_eq!(nan_viseme.ee, 0.3);
    assert_eq!(nan_viseme.oh, 0.0);

    // Array conversion
    let arr = [0.1, 0.2, 0.3, 0.4, 0.5];
    let from_arr = VisemeWeights::from_array(arr);
    assert_eq!(from_arr.as_array(), arr);

    // Named weights mapping
    let named = from_arr.to_named_weights();
    assert_eq!(named[0], ("aa", 0.1));
    assert_eq!(named[1], ("ih", 0.2));
    assert_eq!(named[2], ("ou", 0.3));
    assert_eq!(named[3], ("ee", 0.4));
    assert_eq!(named[4], ("oh", 0.5));
}

#[test]
fn test_viseme_lerp_and_approx_eq() {
    let a = VisemeWeights::zero();
    let b = VisemeWeights::new(1.0, 1.0, 1.0, 1.0, 1.0);

    // factor = 0.5
    let mid = a.lerp(&b, 0.5);
    assert!(mid.approx_eq(&VisemeWeights::new(0.5, 0.5, 0.5, 0.5, 0.5), 1e-6));

    // factor = 0.0 -> a
    let source = a.lerp(&b, 0.0);
    assert_eq!(source, a);

    // factor = 1.0 -> b
    let target = a.lerp(&b, 1.0);
    assert_eq!(target, b);

    // factor > 1.0 clamped to 1.0
    let over = a.lerp(&b, 2.0);
    assert_eq!(over, b);
}

#[test]
fn test_viseme_serde_clamping_enforcement() {
    // Unchecked raw JSON with out-of-range weights should be automatically clamped by custom deserializer
    let json = r#"{"aa": 3.5, "ih": -2.0, "ou": 0.3, "ee": 99.0, "oh": 0.8}"#;
    let decoded: VisemeWeights = serde_json::from_str(json).expect("deserialize viseme");
    assert_eq!(decoded.aa, 1.0);
    assert_eq!(decoded.ih, 0.0);
    assert_eq!(decoded.ou, 0.3);
    assert_eq!(decoded.ee, 1.0);
    assert_eq!(decoded.oh, 0.8);
}

#[test]
fn test_message_creation_and_defaults() {
    let msg = Message::new(Role::User, "Hello Vira");
    assert_eq!(msg.role, Role::User);
    assert_eq!(msg.raw_content, "Hello Vira");
    assert_eq!(msg.speakable_content, "Hello Vira");
    assert!(!msg.id.is_empty());
    assert!(msg.timestamp > 0);
    assert!(msg.is_user());
    assert!(!msg.is_assistant());
    assert!(!msg.is_empty());

    let assistant_msg = Message::assistant("Hello! [wave]")
        .with_speakable_content("Hello!")
        .with_timestamp(1700000000000);
    assert_eq!(assistant_msg.raw_content, "Hello! [wave]");
    assert_eq!(assistant_msg.speakable_content, "Hello!");
    assert_eq!(assistant_msg.timestamp, 1700000000000);
    assert!(assistant_msg.is_assistant());

    let sys_msg = Message::system("System prompt");
    assert!(sys_msg.is_system());

    let tool_msg = Message::tool("result");
    assert!(tool_msg.is_tool());

    let empty_msg = Message::user("");
    assert!(empty_msg.is_empty());

    let display_str = format!("{assistant_msg}");
    assert_eq!(display_str, "[assistant] Hello! [wave]");
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
fn test_message_concurrency_stress() {
    // Attack edge case from prior report: concurrent message construction across threads
    let thread_count = 50;
    let mut handles = Vec::new();

    for i in 0..thread_count {
        let handle = thread::spawn(move || {
            let msg = Message::user(format!("Thread message {i}"));
            (msg.id, msg.timestamp)
        });
        handles.push(handle);
    }

    let mut id_set = HashSet::new();
    for handle in handles {
        let (id, ts) = handle.join().expect("thread finish");
        assert!(ts > 0);
        assert!(id_set.insert(id), "UUID collision detected in concurrent Message creation");
    }
    assert_eq!(id_set.len(), thread_count);
}

#[test]
fn test_state_transition_event() {
    let event = StateTransitionEvent::new(ConversationState::Idle, ConversationState::Listening);
    assert_eq!(event.from, ConversationState::Idle);
    assert_eq!(event.to, ConversationState::Listening);
    assert!(event.timestamp > 0);

    let explicit = StateTransitionEvent::with_timestamp(
        ConversationState::Thinking,
        ConversationState::Speaking,
        987654321,
    );
    assert_eq!(explicit.timestamp, 987654321);

    let json = serde_json::to_string(&explicit).unwrap();
    let decoded: StateTransitionEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, explicit);
}

#[test]
fn test_state_transition_error_serde() {
    let err = StateTransitionError::InvalidTransition {
        from: ConversationState::Idle,
        to: ConversationState::Speaking,
    };
    let json = serde_json::to_string(&err).expect("serialize error");
    let decoded: StateTransitionError = serde_json::from_str(&json).expect("deserialize error");
    assert_eq!(decoded, err);
}
