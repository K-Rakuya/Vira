use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use vira_domain::state::ConversationState;
use vira_ports::tts::TimedViseme;
use vira_tauri::commands::{
    handle_get_app_state, handle_save_settings, handle_send_message, handle_set_emotion,
    handle_set_tutor_mode, handle_toggle_mic, handle_trigger_gesture, SendMessagePayload,
};
use vira_tauri::events::{
    AudioStatePayload, EmotionPayload, GesturePayload, MockEventEmitter, SubtitleUpdatePayload,
    VisemeTickPayload, EVENT_AUDIO_STATE, EVENT_EMOTION, EVENT_GESTURE, EVENT_SUBTITLE_UPDATE,
    EVENT_VISEME_TICK,
};
use vira_tauri::spawn_viseme_tick_loop;
use vira_tauri::state::{AppSettingsPayload, AppState, TutorMode};

#[test]
fn test_send_message_payload_validation() {
    let valid = SendMessagePayload {
        text: "Hello Aura".to_string(),
    };
    assert!(!valid.text.trim().is_empty());
}

#[tokio::test]
async fn test_send_message_empty_text_rejected() {
    let state = AppState::new_headless();

    let empty = SendMessagePayload {
        text: "".to_string(),
    };
    let err = handle_send_message(empty, &state).await;
    assert!(err.is_err());
    assert_eq!(err.unwrap_err(), "Message text cannot be empty");

    let whitespace = SendMessagePayload {
        text: "   \n\t  ".to_string(),
    };
    let err_ws = handle_send_message(whitespace, &state).await;
    assert!(err_ws.is_err());
}

#[tokio::test]
async fn test_send_message_end_to_end_conversational_cycle() {
    let emitter = Arc::new(MockEventEmitter::new());
    let state = AppState::new_headless_with_emitter(emitter.clone());

    let payload = SendMessagePayload {
        text: "Xin chào bạn!".to_string(),
    };

    let res = handle_send_message(payload, &state).await.expect("turn succeeded");
    assert_eq!(res.status, "completed");

    // History verification: user + assistant messages recorded
    let history = state.history.read().await;
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].role, vira_domain::message::Role::User);
    assert_eq!(history[0].raw_content, "Xin chào bạn!");
    assert_eq!(history[1].role, vira_domain::message::Role::Assistant);
    assert!(history[1].raw_content.contains("Hello! I am Vira."));

    // Event verification: emotion and gesture extracted from stream
    let emotions = emitter.find_events(EVENT_EMOTION);
    assert!(!emotions.is_empty(), "emotion event should be emitted");
    let parsed_emo: EmotionPayload = serde_json::from_value(emotions[0].clone()).unwrap();
    assert_eq!(parsed_emo.name, "happy");

    let gestures = emitter.find_events(EVENT_GESTURE);
    assert!(!gestures.is_empty(), "gesture event should be emitted");
    let parsed_gest: GesturePayload = serde_json::from_value(gestures[0].clone()).unwrap();
    assert_eq!(parsed_gest.name, "wave");

    // Subtitle verification
    let subtitles = emitter.find_events(EVENT_SUBTITLE_UPDATE);
    assert!(!subtitles.is_empty(), "subtitles should be emitted");
    let has_spoken_subtitle = subtitles.iter().any(|v| {
        let sub: SubtitleUpdatePayload = serde_json::from_value(v.clone()).unwrap();
        sub.state == ConversationState::Speaking && (sub.text == "Hello!" || sub.text == "I am Vira.")
    });
    assert!(has_spoken_subtitle, "should contain speaking subtitle");

    // State is speaking while audio is enqueued in playback buffer
    assert_eq!(state.get_conversation_state(), ConversationState::Speaking);
    assert!(state.audio_playback.is_playing());

    // Flush playback and reset to idle
    state.audio_playback.flush().unwrap();
    state.force_state(ConversationState::Idle);
    state.emit_subtitle("", ConversationState::Idle);
    assert_eq!(state.get_conversation_state(), ConversationState::Idle);
}

#[tokio::test]
async fn test_trigger_gesture_command_validation_and_event() {
    let emitter = Arc::new(MockEventEmitter::new());
    let state = AppState::new_headless_with_emitter(emitter.clone());

    // Valid gesture
    let res = handle_trigger_gesture("bow".to_string(), 1.5, &state).await;
    assert!(res.is_ok());

    let gestures = emitter.find_events(EVENT_GESTURE);
    assert_eq!(gestures.len(), 1);
    let payload: GesturePayload = serde_json::from_value(gestures[0].clone()).unwrap();
    assert_eq!(payload.name, "bow");
    assert!((payload.duration - 1.5).abs() < 1e-4);

    // Empty gesture fails
    let err = handle_trigger_gesture("   ".to_string(), 1.0, &state).await;
    assert!(err.is_err());
    assert_eq!(err.unwrap_err(), "Gesture name cannot be empty");
}

#[tokio::test]
async fn test_set_emotion_command_validation_and_state_update() {
    let emitter = Arc::new(MockEventEmitter::new());
    let state = AppState::new_headless_with_emitter(emitter.clone());

    // Valid emotion
    let res = handle_set_emotion("surprised".to_string(), 0.75, &state).await;
    assert!(res.is_ok());

    assert_eq!(state.emotion.read().unwrap().0, "surprised");
    assert!((state.emotion.read().unwrap().1 - 0.75).abs() < 1e-4);

    let emotions = emitter.find_events(EVENT_EMOTION);
    assert_eq!(emotions.len(), 1);
    let payload: EmotionPayload = serde_json::from_value(emotions[0].clone()).unwrap();
    assert_eq!(payload.name, "surprised");
    assert!((payload.weight - 0.75).abs() < 1e-4);

    // Weight clamping
    let res_clamp = handle_set_emotion("happy".to_string(), 2.5, &state).await;
    assert!(res_clamp.is_ok());
    assert!((state.emotion.read().unwrap().1 - 1.0).abs() < 1e-4);

    // Empty emotion fails
    let err = handle_set_emotion("".to_string(), 0.5, &state).await;
    assert!(err.is_err());
    assert_eq!(err.unwrap_err(), "Emotion name cannot be empty");
}

#[tokio::test]
async fn test_set_tutor_mode_command() {
    let state = AppState::new_headless();

    assert_eq!(
        handle_set_tutor_mode("english".to_string(), &state).await.unwrap(),
        "english"
    );
    assert_eq!(*state.tutor_mode.read().unwrap(), TutorMode::English);

    assert_eq!(
        handle_set_tutor_mode("ja".to_string(), &state).await.unwrap(),
        "japanese"
    );
    assert_eq!(*state.tutor_mode.read().unwrap(), TutorMode::Japanese);

    assert_eq!(
        handle_set_tutor_mode("vietnamese".to_string(), &state).await.unwrap(),
        "vietnamese"
    );
    assert_eq!(*state.tutor_mode.read().unwrap(), TutorMode::Vietnamese);

    assert_eq!(
        handle_set_tutor_mode("general_assistant".to_string(), &state).await.unwrap(),
        "general_assistant"
    );
    assert_eq!(*state.tutor_mode.read().unwrap(), TutorMode::GeneralAssistant);

    assert!(handle_set_tutor_mode("nonexistent_mode".to_string(), &state).await.is_err());
}

#[tokio::test]
async fn test_toggle_mic_and_barge_in_handling() {
    let emitter = Arc::new(MockEventEmitter::new());
    let state = AppState::new_headless_with_emitter(emitter.clone());

    // 1. Toggle mic ON from Idle
    let res = handle_toggle_mic(true, &state).await.expect("toggle mic on");
    assert!(res);
    assert!(state.mic_active.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(state.get_conversation_state(), ConversationState::Listening);

    // 2. Toggle mic OFF
    let res_off = handle_toggle_mic(false, &state).await.expect("toggle mic off");
    assert!(!res_off);
    assert!(!state.mic_active.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(state.get_conversation_state(), ConversationState::Idle);

    // 3. Barge-in interruption during Speaking
    state.force_state(ConversationState::Speaking);
    // Enqueue some audio to simulate active sound
    let _ = state.audio_playback.enqueue_samples(&[0.5; 1024]);
    assert!(state.audio_playback.is_playing());

    // User speaks -> mic toggled on during playback
    let barge = handle_toggle_mic(true, &state).await.expect("barge-in toggle");
    assert!(barge);

    // Playback buffer should be flushed instantly (<10ms)
    assert_eq!(state.audio_playback.buffered_samples(), 0);
    assert_eq!(state.get_conversation_state(), ConversationState::Listening);

    // Audio state event emitted with barge_in_triggered = true
    let audio_events = emitter.find_events(EVENT_AUDIO_STATE);
    assert!(!audio_events.is_empty());
    let last_audio: AudioStatePayload =
        serde_json::from_value(audio_events.last().unwrap().clone()).unwrap();
    assert!(last_audio.barge_in_triggered);
}

#[tokio::test]
async fn test_save_settings_and_get_app_state() {
    let state = AppState::new_headless();

    let settings = AppSettingsPayload {
        llm_endpoint: Some("http://192.168.1.100:11434".to_string()),
        llm_model: Some("qwen2.5:7b".to_string()),
        llm_api_key: None,
        tts_voice: Some("af_bella".to_string()),
        tts_speed: Some(1.1),
        stt_model: Some("whisper-base".to_string()),
        tutor_mode: Some("japanese".to_string()),
    };

    let save_res = handle_save_settings(settings.clone(), &state).await;
    assert!(save_res.is_ok());

    let snapshot = handle_get_app_state(&state).await.expect("snapshot retrieved");
    assert_eq!(snapshot.tutor_mode, TutorMode::Japanese);
    assert_eq!(state.settings.read().unwrap().llm_model, Some("qwen2.5:7b".to_string()));
}

#[tokio::test]
async fn test_viseme_tick_loop_high_frequency_broadcast() {
    let emitter = Arc::new(MockEventEmitter::new());
    let state = AppState::new_headless_with_emitter(emitter.clone());

    // Inject timed viseme timeline
    let timed_visemes = vec![
        TimedViseme::new(0, vira_domain::viseme::VisemeWeights::new(0.8, 0.0, 0.0, 0.0, 0.0)),
        TimedViseme::new(100, vira_domain::viseme::VisemeWeights::new(0.0, 0.9, 0.0, 0.0, 0.0)),
    ];
    *state.active_visemes.write().unwrap() = timed_visemes;

    // Enqueue audio so is_playing returns true
    let _ = state.audio_playback.enqueue_samples(&[0.1; 4800]);

    let cancel = CancellationToken::new();
    let tick_handle = spawn_viseme_tick_loop(state.clone(), cancel.clone());

    // Let the 60Hz loop run for 40ms (~2-3 ticks)
    tokio::time::sleep(std::time::Duration::from_millis(40)).await;

    // Cancel loop
    cancel.cancel();
    let _ = tick_handle.await;

    let ticks = emitter.find_events(EVENT_VISEME_TICK);
    assert!(!ticks.is_empty(), "viseme ticks should be emitted at 60Hz");

    let first_tick: VisemeTickPayload = serde_json::from_value(ticks[0].clone()).unwrap();
    assert!(first_tick.is_speaking);
    assert!(first_tick.weights.aa > 0.0 || first_tick.weights.ih >= 0.0);
}
