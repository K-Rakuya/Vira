use std::sync::atomic::Ordering;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use vira_domain::message::Message;
use vira_domain::state::ConversationState;
use vira_voice::stream_parser::{ParsedStreamEvent, StreamParser};

use crate::events::{EmotionPayload, GesturePayload, EVENT_EMOTION, EVENT_GESTURE};
use crate::state::{AppSettingsPayload, AppState, GetAppStateResponse, TutorMode};

/// Payload sent from frontend webview to send a user message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SendMessagePayload {
    pub text: String,
}

/// Acknowledgment response for `send_message`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SendMessageResponse {
    pub message_id: String,
    pub status: String,
}

/// Executes a complete conversational turn:
/// LLM stream -> tag extraction (emotion/gesture) -> sentence chunker -> TTS synthesis -> audio playback.
pub async fn execute_conversation_turn(
    text: &str,
    state: &AppState,
) -> Result<SendMessageResponse, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("Message text cannot be empty".to_string());
    }

    // 1. Cancel previous speech turn (if any) and reset audio/visemes
    state.cancel_active_speech();

    // 2. Set new cancellation token
    let cancel = CancellationToken::new();
    *state.active_cancellation.lock().unwrap() = Some(cancel.clone());

    // 3. Move state to Thinking
    state.force_state(ConversationState::Thinking);

    // 4. Record user message in history
    let user_msg = Message::user(trimmed);
    let message_id = user_msg.id.clone();
    {
        let mut history = state.history.write().await;
        history.push(user_msg);
    }

    // 5. Build conversation history with tutor system prompt if not present
    let mut history_snapshot = state.history.read().await.clone();
    let tutor_mode = *state.tutor_mode.read().unwrap();
    let has_system_msg = history_snapshot.first().is_some_and(|m| m.is_system());
    if !has_system_msg {
        history_snapshot.insert(0, Message::system(tutor_mode.system_prompt()));
    }

    // 6. Launch LLM stream
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let llm = state.llm.clone();
    let cancel_llm = cancel.clone();
    tokio::spawn(async move {
        let _ = llm.stream_chat(&history_snapshot, tx, cancel_llm).await;
    });

    // 7. Stream parser accumulates text and extracts inline emotion/gesture tags
    let mut parser = StreamParser::new();
    let mut full_assistant_text = String::new();
    let mut chunks_synthesized = 0usize;

    while let Some(chunk) = rx.recv().await {
        if cancel.is_cancelled() {
            break;
        }

        // Direct tags attached to chunk
        if let Some(ref emo) = chunk.emotion {
            let weight = 1.0f32;
            *state.emotion.write().unwrap() = (emo.clone(), weight);
            let payload = EmotionPayload {
                name: emo.clone(),
                weight,
            };
            if let Ok(v) = serde_json::to_value(payload) {
                let _ = state.emitter.emit(EVENT_EMOTION, v);
            }
        }

        if let Some(ref gest) = chunk.gesture {
            let payload = GesturePayload {
                name: gest.clone(),
                duration: 2.0,
            };
            if let Ok(v) = serde_json::to_value(payload) {
                let _ = state.emitter.emit(EVENT_GESTURE, v);
            }
        }

        if !chunk.delta.is_empty() {
            let events = parser.feed(&chunk.delta);
            for event in events {
                if cancel.is_cancelled() {
                    break;
                }
                match event {
                    ParsedStreamEvent::Emotion(emo) => {
                        let weight = 1.0f32;
                        *state.emotion.write().unwrap() = (emo.clone(), weight);
                        let payload = EmotionPayload { name: emo, weight };
                        if let Ok(v) = serde_json::to_value(payload) {
                            let _ = state.emitter.emit(EVENT_EMOTION, v);
                        }
                    }
                    ParsedStreamEvent::Gesture(gest) => {
                        let payload = GesturePayload {
                            name: gest,
                            duration: 2.0,
                        };
                        if let Ok(v) = serde_json::to_value(payload) {
                            let _ = state.emitter.emit(EVENT_GESTURE, v);
                        }
                    }
                    ParsedStreamEvent::SpeakableText(sentence) => {
                        let clean = sentence.trim();
                        if clean.is_empty() {
                            continue;
                        }

                        if !full_assistant_text.is_empty() {
                            full_assistant_text.push(' ');
                        }
                        full_assistant_text.push_str(clean);

                        state.force_state(ConversationState::Speaking);
                        state.emit_subtitle(clean, ConversationState::Speaking);

                        if let Ok(tts_out) = state.tts.synthesize(clean).await {
                            if !tts_out.samples.is_empty() {
                                let _ = state.audio_playback.enqueue_samples(&tts_out.samples);
                                state.active_visemes.write().unwrap().extend(tts_out.visemes);
                                chunks_synthesized += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // Flush any remaining text in the parser
    if !cancel.is_cancelled() {
        let flushed = parser.finish();
        for event in flushed {
            if cancel.is_cancelled() {
                break;
            }
            match event {
                ParsedStreamEvent::Emotion(emo) => {
                    let weight = 1.0f32;
                    *state.emotion.write().unwrap() = (emo.clone(), weight);
                    let payload = EmotionPayload { name: emo, weight };
                    if let Ok(v) = serde_json::to_value(payload) {
                        let _ = state.emitter.emit(EVENT_EMOTION, v);
                    }
                }
                ParsedStreamEvent::Gesture(gest) => {
                    let payload = GesturePayload {
                        name: gest,
                        duration: 2.0,
                    };
                    if let Ok(v) = serde_json::to_value(payload) {
                        let _ = state.emitter.emit(EVENT_GESTURE, v);
                    }
                }
                ParsedStreamEvent::SpeakableText(sentence) => {
                    let clean = sentence.trim();
                    if !clean.is_empty() {
                        if !full_assistant_text.is_empty() {
                            full_assistant_text.push(' ');
                        }
                        full_assistant_text.push_str(clean);

                        state.force_state(ConversationState::Speaking);
                        state.emit_subtitle(clean, ConversationState::Speaking);

                        if let Ok(tts_out) = state.tts.synthesize(clean).await {
                            if !tts_out.samples.is_empty() {
                                let _ = state.audio_playback.enqueue_samples(&tts_out.samples);
                                state.active_visemes.write().unwrap().extend(tts_out.visemes);
                                chunks_synthesized += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // Check cancellation
    if cancel.is_cancelled() {
        return Ok(SendMessageResponse {
            message_id,
            status: "interrupted".to_string(),
        });
    }

    // Record completed assistant turn to conversation history
    if !full_assistant_text.is_empty() {
        let assistant_msg = Message::assistant(full_assistant_text);
        state.history.write().await.push(assistant_msg);
    }

    if chunks_synthesized == 0 && !cancel.is_cancelled() {
        state.force_state(ConversationState::Idle);
        state.emit_subtitle("", ConversationState::Idle);
    }

    Ok(SendMessageResponse {
        message_id,
        status: "completed".to_string(),
    })
}

/// Handler for `send_message`.
pub async fn handle_send_message(
    payload: SendMessagePayload,
    state: &AppState,
) -> Result<SendMessageResponse, String> {
    execute_conversation_turn(&payload.text, state).await
}

/// Tauri command wrapper for `send_message`.
#[tauri::command]
pub async fn send_message(
    payload: SendMessagePayload,
    state: tauri::State<'_, AppState>,
) -> Result<SendMessageResponse, String> {
    handle_send_message(payload, state.inner()).await
}

/// Handler for `trigger_gesture`.
pub async fn handle_trigger_gesture(
    name: String,
    duration: f32,
    state: &AppState,
) -> Result<(), String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("Gesture name cannot be empty".to_string());
    }
    let duration = if duration <= 0.0 { 2.0 } else { duration.min(60.0) };
    let payload = GesturePayload {
        name: trimmed.to_string(),
        duration,
    };
    state
        .emitter
        .emit(EVENT_GESTURE, serde_json::to_value(payload).map_err(|e| e.to_string())?)
}

/// Tauri command wrapper for `trigger_gesture`.
#[tauri::command]
pub async fn trigger_gesture(
    name: String,
    duration: f32,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    handle_trigger_gesture(name, duration, state.inner()).await
}

/// Handler for `set_emotion`.
pub async fn handle_set_emotion(
    name: String,
    weight: f32,
    state: &AppState,
) -> Result<(), String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("Emotion name cannot be empty".to_string());
    }
    let weight = weight.clamp(0.0, 1.0);
    *state.emotion.write().unwrap() = (trimmed.to_string(), weight);
    let payload = EmotionPayload {
        name: trimmed.to_string(),
        weight,
    };
    state
        .emitter
        .emit(EVENT_EMOTION, serde_json::to_value(payload).map_err(|e| e.to_string())?)
}

/// Tauri command wrapper for `set_emotion`.
#[tauri::command]
pub async fn set_emotion(
    name: String,
    weight: f32,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    handle_set_emotion(name, weight, state.inner()).await
}

/// Handler for `set_tutor_mode`.
pub async fn handle_set_tutor_mode(
    mode: String,
    state: &AppState,
) -> Result<String, String> {
    let parsed = mode.parse::<TutorMode>()?;
    *state.tutor_mode.write().unwrap() = parsed;
    Ok(parsed.as_str().to_string())
}

/// Tauri command wrapper for `set_tutor_mode`.
#[tauri::command]
pub async fn set_tutor_mode(
    mode: String,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    handle_set_tutor_mode(mode, state.inner()).await
}

/// Handler for `toggle_mic`.
pub async fn handle_toggle_mic(
    active: bool,
    state: &AppState,
) -> Result<bool, String> {
    if active {
        let can_barge_in = state.get_conversation_state().can_barge_in();
        if can_barge_in {
            state.cancel_active_speech();
            state.force_state(ConversationState::Listening);
            state.emit_audio_state(0.0, true);
        } else {
            state.force_state(ConversationState::Listening);
        }
        state
            .audio_capture
            .start_capture()
            .map_err(|e| e.to_string())?;
        state.mic_active.store(true, Ordering::Release);
    } else {
        state
            .audio_capture
            .stop_capture()
            .map_err(|e| e.to_string())?;
        state.mic_active.store(false, Ordering::Release);
        if state.get_conversation_state() == ConversationState::Listening {
            state.force_state(ConversationState::Idle);
        }
    }
    Ok(active)
}

/// Tauri command wrapper for `toggle_mic`.
#[tauri::command]
pub async fn toggle_mic(
    active: bool,
    state: tauri::State<'_, AppState>,
) -> Result<bool, String> {
    handle_toggle_mic(active, state.inner()).await
}

/// Handler for `save_settings`.
pub async fn handle_save_settings(
    settings: AppSettingsPayload,
    state: &AppState,
) -> Result<(), String> {
    if let Some(ref mode_str) = settings.tutor_mode {
        if let Ok(m) = mode_str.parse::<TutorMode>() {
            *state.tutor_mode.write().unwrap() = m;
        }
    }
    *state.settings.write().unwrap() = settings;
    Ok(())
}

/// Tauri command wrapper for `save_settings`.
#[tauri::command]
pub async fn save_settings(
    settings: AppSettingsPayload,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    handle_save_settings(settings, state.inner()).await
}

/// Handler for `get_app_state`.
pub async fn handle_get_app_state(
    state: &AppState,
) -> Result<GetAppStateResponse, String> {
    Ok(state.get_snapshot())
}

/// Tauri command wrapper for `get_app_state`.
#[tauri::command]
pub async fn get_app_state(
    state: tauri::State<'_, AppState>,
) -> Result<GetAppStateResponse, String> {
    handle_get_app_state(state.inner()).await
}
