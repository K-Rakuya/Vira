use std::sync::atomic::Ordering;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use vira_domain::message::Message;
use vira_domain::state::{CharacterEmotion, ConversationState};
use vira_ports::tts::TimedViseme;
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

/// Performs linear interpolation resampling between audio sample rates.
pub fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() || from_rate == 0 || to_rate == 0 {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = (samples.len() as f64 * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_idx = i as f64 / ratio;
        let idx0 = src_idx.floor() as usize;
        let frac = (src_idx - idx0 as f64) as f32;
        let s0 = samples.get(idx0).copied().unwrap_or(0.0);
        let s1 = samples.get(idx0 + 1).copied().unwrap_or(s0);
        out.push(s0 + frac * (s1 - s0));
    }
    out
}

/// Helper to process, synthesize, and enqueue a single speakable sentence.
async fn process_speakable_sentence(
    clean: &str,
    state: &AppState,
    cancel: &CancellationToken,
    full_assistant_text: &mut String,
    cumulative_duration_ms: &mut u64,
    chunks_synthesized: &mut usize,
) -> bool {
    if cancel.is_cancelled() {
        return false;
    }

    if !full_assistant_text.is_empty() {
        full_assistant_text.push(' ');
    }
    full_assistant_text.push_str(clean);

    let tts_out = match state.tts.synthesize(clean).await {
        Ok(out) => out,
        Err(_) => return true,
    };

    if cancel.is_cancelled() {
        return false;
    }

    if !tts_out.samples.is_empty() {
        let pb_rate = state.audio_playback.sample_rate();
        let samples = if tts_out.sample_rate != pb_rate && tts_out.sample_rate > 0 && pb_rate > 0 {
            resample_linear(&tts_out.samples, tts_out.sample_rate, pb_rate)
        } else {
            tts_out.samples
        };

        if cancel.is_cancelled() {
            return false;
        }

        let offset = *cumulative_duration_ms;
        let offset_visemes: Vec<TimedViseme> = tts_out
            .visemes
            .into_iter()
            .map(|v| TimedViseme::new(v.timestamp_ms + offset, v.weights))
            .collect();

        let chunk_dur_from_samples = if pb_rate > 0 {
            (samples.len() as u64 * 1000) / pb_rate as u64
        } else {
            0
        };
        let chunk_dur_from_visemes = offset_visemes
            .last()
            .map(|v| v.timestamp_ms.saturating_sub(offset))
            .unwrap_or(0);
        let chunk_dur = chunk_dur_from_samples.max(chunk_dur_from_visemes);

        state.force_state(ConversationState::Speaking);
        state.emit_subtitle(clean, ConversationState::Speaking);

        let _ = state.audio_playback.enqueue_samples(&samples);
        state.active_visemes.write().unwrap().extend(offset_visemes);
        *cumulative_duration_ms += chunk_dur;
        *chunks_synthesized += 1;
    }

    true
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

    // Acquire turn lock to serialize turn execution and prevent interleaved synthesis
    let _turn_guard = state.turn_lock.lock().await;

    // 1. Cancel previous speech turn (if any), flush audio, and reset clock
    state.cancel_active_speech();

    // 2. Set new cancellation token
    let cancel = CancellationToken::new();
    *state.active_cancellation.lock().unwrap() = Some(cancel.clone());
    state.set_generating(true);

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

    // 6. Launch LLM stream and capture join handle to catch errors
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let llm = state.llm.clone();
    let cancel_llm = cancel.clone();
    let llm_handle = tokio::spawn(async move {
        llm.stream_chat(&history_snapshot, tx, cancel_llm).await
    });

    // 7. Stream parser accumulates text and extracts inline emotion/gesture tags
    let mut parser = StreamParser::new();
    let mut full_assistant_text = String::new();
    let mut chunks_synthesized = 0usize;
    let mut cumulative_duration_ms = 0u64;

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
                        if !process_speakable_sentence(
                            clean,
                            state,
                            &cancel,
                            &mut full_assistant_text,
                            &mut cumulative_duration_ms,
                            &mut chunks_synthesized,
                        )
                        .await
                        {
                            break;
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
                    if clean.is_empty() {
                        continue;
                    }
                    if !process_speakable_sentence(
                        clean,
                        state,
                        &cancel,
                        &mut full_assistant_text,
                        &mut cumulative_duration_ms,
                        &mut chunks_synthesized,
                    )
                    .await
                    {
                        break;
                    }
                }
            }
        }
    }

    // Inspect LLM task completion
    let llm_res = llm_handle.await;

    // Generation is done
    state.set_generating(false);

    // Check cancellation
    if cancel.is_cancelled() {
        return Ok(SendMessageResponse {
            message_id,
            status: "interrupted".to_string(),
        });
    }

    // If no text was produced, check if LLM stream failed with an error
    if chunks_synthesized == 0 && full_assistant_text.is_empty() {
        state.force_state(ConversationState::Idle);
        state.emit_subtitle("", ConversationState::Idle);

        match llm_res {
            Ok(Err(llm_err)) => {
                return Err(format!("LLM inference failed: {llm_err}"));
            }
            Err(join_err) => {
                return Err(format!("LLM task panicked: {join_err}"));
            }
            _ => {
                return Ok(SendMessageResponse {
                    message_id,
                    status: "empty".to_string(),
                });
            }
        }
    }

    // Record completed assistant turn to conversation history
    if !full_assistant_text.is_empty() {
        let assistant_msg = Message::assistant(full_assistant_text);
        state.history.write().await.push(assistant_msg);
    }

    if chunks_synthesized == 0 {
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

/// Tauri command wrapper for `play_gesture` (alias for `trigger_gesture`).
#[tauri::command]
pub async fn play_gesture(
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
    let canonical_name = match trimmed.parse::<CharacterEmotion>() {
        Ok(domain_emotion) => domain_emotion.as_str().to_string(),
        Err(_) => trimmed.to_string(),
    };
    let weight = weight.clamp(0.0, 1.0);
    *state.emotion.write().unwrap() = (canonical_name.clone(), weight);
    let payload = EmotionPayload {
        name: canonical_name,
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
        state
            .audio_capture
            .start_capture()
            .map_err(|e| e.to_string())?;
        state.mic_active.store(true, Ordering::Release);

        let can_barge_in = state.get_conversation_state().can_barge_in();
        if can_barge_in {
            state.cancel_active_speech();
            state.force_state(ConversationState::Listening);
            state.emit_audio_state(0.0, true);
        } else {
            state.force_state(ConversationState::Listening);
        }
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
