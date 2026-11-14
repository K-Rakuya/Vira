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
        ..Default::default()
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

#[tokio::test]
async fn test_multi_chunk_viseme_monotonicity() {
    use vira_infer::openai_compat::{MockLlmTransport, OpenAiCompatLlmAdapter};

    let emitter = Arc::new(MockEventEmitter::new());
    let mut state = AppState::new_headless_with_emitter(emitter.clone());

    // Mock LLM delivering 2 distinct sentences
    let transport = Arc::new(MockLlmTransport::from_deltas(&[
        "This is the first sentence. ",
        "And this is the second sentence.",
    ]));
    let llm = Arc::new(
        OpenAiCompatLlmAdapter::new(
            "http://127.0.0.1:8080/v1".to_string(),
            "default".to_string(),
            None,
        )
        .with_transport(transport),
    );
    state.llm = llm;

    let res = handle_send_message(
        SendMessagePayload {
            text: "Tell me two things".to_string(),
        },
        &state,
    )
    .await
    .expect("turn succeeded");
    assert_eq!(res.status, "completed");

    let visemes = state.active_visemes.read().unwrap().clone();
    assert!(!visemes.is_empty(), "active visemes should not be empty");

    // Timestamps MUST be strictly non-decreasing across all chunks
    for i in 1..visemes.len() {
        assert!(
            visemes[i].timestamp_ms >= visemes[i - 1].timestamp_ms,
            "Viseme timestamps must be monotonic! Found at index {}: {} < {}",
            i,
            visemes[i].timestamp_ms,
            visemes[i - 1].timestamp_ms
        );
    }
}

#[tokio::test]
async fn test_clock_reset_on_subsequent_turn() {
    let state = AppState::new_headless();

    // Advance clock simulating turn 1 audio playback
    state.clock.advance_samples(48000); // 1000ms
    assert_eq!(state.clock.current_time_ms(), 1000);

    // Cancel / start turn 2
    state.cancel_active_speech();

    // Clock must be reset to 0 for next turn
    assert_eq!(
        state.clock.current_time_ms(),
        0,
        "MasterAudioClock must be reset to 0 after cancelling active speech"
    );
}

#[tokio::test]
async fn test_llm_failure_propagates_error() {
    use vira_infer::openai_compat::{MockLlmTransport, OpenAiCompatLlmAdapter};
    use vira_ports::llm::LlmError;

    let mut state = AppState::new_headless();
    // Failing transport
    let transport = Arc::new(
        MockLlmTransport::new(vec![]).with_error(LlmError::Network("Simulated connection refused".into())),
    );
    let llm = Arc::new(
        OpenAiCompatLlmAdapter::new(
            "http://127.0.0.1:8080/v1".to_string(),
            "default".to_string(),
            None,
        )
        .with_transport(transport),
    );
    state.llm = llm;

    let res = handle_send_message(
        SendMessagePayload {
            text: "Hello".to_string(),
        },
        &state,
    )
    .await;

    assert!(res.is_err(), "LLM failure must return Err, not Ok");
    assert_eq!(state.get_conversation_state(), ConversationState::Idle);
}

#[tokio::test]
async fn test_barge_in_during_synthesis_does_not_enqueue_stale_audio() {
    use vira_infer::openai_compat::{MockLlmTransport, OpenAiCompatLlmAdapter};

    let emitter = Arc::new(MockEventEmitter::new());
    let mut state = AppState::new_headless_with_emitter(emitter.clone());

    // Transport with slight delay
    let transport = Arc::new(
        MockLlmTransport::from_deltas(&["This is a long sentence that takes time to speak."])
            .with_delay_ms(30),
    );
    let llm = Arc::new(
        OpenAiCompatLlmAdapter::new(
            "http://127.0.0.1:8080/v1".to_string(),
            "default".to_string(),
            None,
        )
        .with_transport(transport),
    );
    state.llm = llm;

    let state_clone = state.clone();
    let handle = tokio::spawn(async move {
        handle_send_message(
            SendMessagePayload {
                text: "Hello".to_string(),
            },
            &state_clone,
        )
        .await
    });

    // Wait until generation is in progress (Thinking or Speaking)
    tokio::time::sleep(std::time::Duration::from_millis(15)).await;

    // User triggers barge-in by toggling mic on
    handle_toggle_mic(true, &state).await.expect("barge-in toggle");
    assert_eq!(state.get_conversation_state(), ConversationState::Listening);

    // Wait for the background send_message to complete/interrupt
    let res = handle.await.unwrap();
    // Status must be interrupted, or cancelled
    if let Ok(resp) = res {
        assert_eq!(resp.status, "interrupted");
    }

    // Audio playback must be completely empty (no stale audio enqueued post-interruption)
    assert_eq!(state.audio_playback.buffered_samples(), 0);
    // State must NOT have been overwritten back to Speaking
    assert_eq!(state.get_conversation_state(), ConversationState::Listening);
}

#[tokio::test]
async fn test_viseme_tick_loop_transitions_to_idle_on_playback_completion() {
    let emitter = Arc::new(MockEventEmitter::new());
    let state = AppState::new_headless_with_emitter(emitter.clone());

    // Simulate speaking state with a small amount of audio in playback buffer
    state.force_state(ConversationState::Speaking);
    state.emit_subtitle("Speaking now...", ConversationState::Speaking);
    let _ = state.audio_playback.enqueue_samples(&[0.1; 960]); // 20ms at 48kHz
    assert!(state.audio_playback.is_playing());

    let cancel = CancellationToken::new();
    let tick_handle = spawn_viseme_tick_loop(state.clone(), cancel.clone());

    // Drain the playback buffer simulating sound card DAC consumption
    state.audio_playback.flush().unwrap();
    assert!(!state.audio_playback.is_playing());

    // Give tick loop a few frames (~30ms) to detect playback completion
    tokio::time::sleep(std::time::Duration::from_millis(35)).await;

    cancel.cancel();
    let _ = tick_handle.await;

    // State should automatically transition to Idle
    assert_eq!(state.get_conversation_state(), ConversationState::Idle);
    assert_eq!(*state.current_subtitle.read().unwrap(), "");
}

#[test]
fn test_resample_linear() {
    use vira_tauri::commands::resample_linear;

    // Empty or zero rate
    assert!(resample_linear(&[], 24000, 48000).is_empty());
    assert_eq!(resample_linear(&[1.0, 2.0], 0, 48000), vec![1.0, 2.0]);
    assert_eq!(resample_linear(&[1.0, 2.0], 48000, 48000), vec![1.0, 2.0]);

    // Upsample 2x: 24kHz -> 48kHz
    let input = vec![0.0f32, 1.0];
    let upsampled = resample_linear(&input, 24000, 48000);
    assert_eq!(upsampled.len(), 4);
    assert_eq!(upsampled[0], 0.0);

    // Downsample 2x: 48kHz -> 24kHz
    let down_in = vec![0.0f32, 0.5, 1.0, 0.5];
    let downsampled = resample_linear(&down_in, 48000, 24000);
    assert_eq!(downsampled.len(), 2);
}

#[tokio::test]
async fn test_concurrent_send_message_serialization() {
    use vira_infer::openai_compat::{MockLlmTransport, OpenAiCompatLlmAdapter};

    let emitter = Arc::new(MockEventEmitter::new());
    let mut state = AppState::new_headless_with_emitter(emitter.clone());

    let transport = Arc::new(
        MockLlmTransport::from_deltas(&["First stream delayed reply."]).with_delay_ms(30),
    );
    let llm = Arc::new(
        OpenAiCompatLlmAdapter::new(
            "http://127.0.0.1:8080/v1".to_string(),
            "default".to_string(),
            None,
        )
        .with_transport(transport),
    );
    state.llm = llm;

    let s1 = state.clone();
    let t1 = tokio::spawn(async move {
        handle_send_message(
            SendMessagePayload {
                text: "Message 1".to_string(),
            },
            &s1,
        )
        .await
    });

    // Small delay then fire Message 2
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    let s2 = state.clone();
    let t2 = tokio::spawn(async move {
        handle_send_message(
            SendMessagePayload {
                text: "Message 2".to_string(),
            },
            &s2,
        )
        .await
    });

    let res1 = t1.await.unwrap();
    let res2 = t2.await.unwrap();

    // Both should complete without deadlock or panic
    assert!(res1.is_ok());
    assert!(res2.is_ok());

    let resp1 = res1.unwrap();
    let resp2 = res2.unwrap();

    // Either resp1 was interrupted by resp2, or completed in order
    assert!(resp1.status == "interrupted" || resp1.status == "completed");
    assert_eq!(resp2.status, "completed");
}

#[tokio::test]
async fn test_set_emotion_canonical_and_custom() {
    let emitter = Arc::new(MockEventEmitter::new());
    let state = AppState::new_headless_with_emitter(emitter.clone());

    // Canonical domain emotion "Happy" should be normalized to "happy"
    assert!(handle_set_emotion("Happy".to_string(), 0.8, &state).await.is_ok());
    assert_eq!(state.emotion.read().unwrap().0, "happy");

    // Custom non-domain blendshape emotion preserved as trimmed string
    assert!(handle_set_emotion("custom_sparkle_eyes".to_string(), 0.9, &state).await.is_ok());
    assert_eq!(state.emotion.read().unwrap().0, "custom_sparkle_eyes");
}

#[tokio::test]
async fn test_rag_ingest_and_context_search() {
    let state = AppState::new_headless();

    // Ingest a Japanese grammar note
    let doc_id = "test_doc_jp";
    let title = "Japanese Particles";
    let content = "The particle 'wa' marks the topic of a sentence. The particle 'ga' marks the subject.";
    let chunks = state
        .rag
        .ingest_document(doc_id, title, content, None)
        .await
        .expect("ingest doc");
    assert!(!chunks.is_empty());

    // Search query
    let hits = state
        .rag
        .search("topic marker wa", 3)
        .await
        .expect("search hits");
    assert!(!hits.is_empty());
    assert_eq!(hits[0].title, "Japanese Particles");
}

#[tokio::test]
async fn test_flashcard_lifecycle_fsrs() {
    use vira_learn::SpacedRepetitionPort;
    let state = AppState::new_headless();

    // Create flashcard
    let card = state
        .learn
        .create_card("hello", "xin chào", Some("A1".to_string()))
        .await
        .expect("create card");
    assert_eq!(card.front, "hello");
    assert_eq!(card.back, "xin chào");
    assert_eq!(card.reps, 0);

    // List flashcards
    let cards = state.learn.list_cards(10).await.expect("list cards");
    assert_eq!(cards.len(), 1);

    // Review card with Good rating
    let reviewed = state
        .learn
        .review_card(&card.id, vira_learn::fsrs_scheduler::CardRating::Good)
        .await
        .expect("review card");
    assert_eq!(reviewed.reps, 1);
    assert!(reviewed.stability > 0.0);

    // Preview intervals
    let intervals = state
        .learn
        .preview_intervals(&card.id)
        .await
        .expect("preview intervals");
    assert!(intervals.good >= 1);
}

#[tokio::test]
async fn test_pronunciation_assessment_gop() {
    let _state = AppState::new_headless();

    let target = "xin chào";
    let converter = vira_voice::phoneme_aligner::G2pConverter::new();
    let phonemes = converter.text_to_phonemes(target, vira_voice::phoneme_aligner::PhonemeLanguage::Auto);
    assert!(!phonemes.is_empty());

    let evaluator = vira_voice::gop::GopEvaluator::new(vira_voice::gop::GopConfig::default());
    let mut items = Vec::new();
    for (i, p) in phonemes.iter().enumerate() {
        let frames = vec![-0.05, -0.08];
        items.push((p.as_str(), frames, (i * 100) as u32, ((i + 1) * 100) as u32));
    }

    let ref_items: Vec<(&str, &[f32], u32, u32)> = items
        .iter()
        .map(|(p, f, s, e)| (*p, f.as_slice(), *s, *e))
        .collect();

    let assessment = evaluator.evaluate_phonemes(&ref_items);
    assert!(assessment.overall_score >= 80.0);
    assert!(assessment.is_passing(70.0));
    assert_eq!(assessment.accuracy, vira_voice::gop::AccuracyLevel::Correct);
}

#[tokio::test]
async fn test_agent_tool_executor_builtin_math() {
    use vira_ports::AgentPort;
    let state = AppState::new_headless();

    let req = vira_ports::agent::ToolCallRequest::new(
        "math_1",
        "calculator",
        serde_json::json!({ "expression": "24 * 5 + 10" }),
    );

    let res = state.agent.execute_tool(req).await.expect("execute tool");
    assert!(!res.is_error);
    assert!(res.content.contains("130"));
}



