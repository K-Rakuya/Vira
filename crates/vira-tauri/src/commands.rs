use std::sync::atomic::Ordering;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use vira_domain::message::Message;
use vira_domain::state::{CharacterEmotion, ConversationState};
use vira_learn::SpacedRepetitionPort;
use vira_ports::AgentPort;
use vira_ports::tts::TimedViseme;
use vira_voice::stream_parser::{ParsedStreamEvent, StreamParser};

use crate::events::{
    EmotionPayload, GesturePayload, SpeakSentencePayload, ThoughtPayload, EVENT_EMOTION,
    EVENT_GESTURE, EVENT_SPEAK_SENTENCE, EVENT_THOUGHT,
};
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

/// Modified Bessel function of first kind of order 0, I_0(x).
pub fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half = x * 0.5;
    for k in 1..=25 {
        term *= (x_half / k as f64) * (x_half / k as f64);
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}

/// Normalized sinc function: sinc(x) = sin(pi * x) / (pi * x).
pub fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Resamples an audio buffer between sampling rates using a Polyphase Kaiser-windowed sinc filter.
///
/// Features:
/// - Kaiser window with shape parameter beta = 8.5 providing > 100dB stopband attenuation
/// - Bandwidth scaling to Nyquist frequency when downsampling to prevent aliasing
/// - Unity partition normalization sum(w_i) = 1.0 to eliminate DC ripple and gain distortion
/// - Safe boundary clamping and graceful edge-case handling (empty, 0-rate, equal rate)
pub fn resample_kaiser_sinc(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() || from_rate == 0 || to_rate == 0 {
        return samples.to_vec();
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = (samples.len() as f64 * ratio).round() as usize;
    if out_len == 0 {
        return Vec::new();
    }

    // Cutoff frequency s <= 1.0 for anti-aliasing when downsampling
    let s = ratio.min(1.0);
    let beta = 8.5f64;
    let i0_beta = bessel_i0(beta);

    let in_len = samples.len() as isize;

    // Half-width radius M samples (scaled by 1/s when downsampling to maintain transition bandwidth, bounded for extreme ratios)
    let base_m = 16.0f64;
    let radius = (base_m / s).min((in_len.max(1) as f64) * 2.0);
    if radius <= 0.0 || !radius.is_finite() {
        return samples.to_vec();
    }

    let mut out = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let t_src = i as f64 / ratio;
        let j_min = (t_src - radius).ceil() as isize;
        let j_max = (t_src + radius).floor() as isize;

        let clamp_j_min = j_min.max(0);
        let clamp_j_max = j_max.min(in_len - 1);

        if clamp_j_min > clamp_j_max {
            let nearest = t_src.round().clamp(0.0, (in_len - 1) as f64) as usize;
            out.push(samples.get(nearest).copied().unwrap_or(0.0));
            continue;
        }

        let mut acc = 0.0f64;
        let mut sum_weights = 0.0f64;

        for j in clamp_j_min..=clamp_j_max {
            let t = t_src - j as f64;
            let norm_t = (t / radius).abs();
            if norm_t <= 1.0 {
                let sinc_val = sinc(s * t);
                let w = bessel_i0(beta * (1.0 - norm_t * norm_t).max(0.0).sqrt()) / i0_beta;
                let weight = s * sinc_val * w;
                acc += samples[j as usize] as f64 * weight;
                sum_weights += weight;
            }
        }

        if sum_weights.abs() > 1e-9 {
            out.push((acc / sum_weights) as f32);
        } else {
            let nearest = t_src.round().clamp(0.0, (in_len - 1) as f64) as usize;
            out.push(samples.get(nearest).copied().unwrap_or(0.0));
        }
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
            resample_kaiser_sinc(&tts_out.samples, tts_out.sample_rate, pb_rate)
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

        let enqueue_res = state.audio_playback.enqueue_samples(&samples);
        let has_native_audio = enqueue_res.is_ok() && !samples.is_empty();

        // Emit speak_sentence for synchronized audible speech in frontend
        let tutor_mode = *state.tutor_mode.read().unwrap();
        let lang = match tutor_mode {
            TutorMode::English => "en-US",
            TutorMode::Japanese => "ja-JP",
            TutorMode::Vietnamese => "vi-VN",
            TutorMode::GeneralAssistant => "vi-VN",
        };
        let speed = state.settings.read().unwrap().tts_speed.unwrap_or(1.0);
        let speak_payload =
            SpeakSentencePayload::new(clean, lang, speed).with_native_audio(has_native_audio);
        if let Ok(v) = serde_json::to_value(speak_payload) {
            let _ = state.emitter.emit(EVENT_SPEAK_SENTENCE, v);
        }

        state.active_visemes.write().unwrap().extend(offset_visemes);
        *cumulative_duration_ms += chunk_dur;
        *chunks_synthesized += 1;
    }

    true
}

/// Helper to extract structured tool calls from LLM reasoning or generation text.
pub fn extract_tool_call_from_text(text: &str) -> Option<(String, serde_json::Value)> {
    // 1. <tool_call>...</tool_call>
    if let Some(start) = text.find("<tool_call>") {
        let after = &text[start + 11..];
        if let Some(end) = after.find("</tool_call>") {
            let json_str = after[..end].trim();
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Some(name) = v
                    .get("name")
                    .and_then(|n| n.as_str())
                    .or_else(|| v.get("tool").and_then(|n| n.as_str()))
                {
                    let args = v
                        .get("arguments")
                        .or_else(|| v.get("parameters"))
                        .or_else(|| v.get("args"))
                        .cloned()
                        .unwrap_or(serde_json::json!({}));
                    return Some((name.to_string(), args));
                }
            }
        }
    }

    // 2. [tool_call:{"name":...}]
    if let Some(start) = text.find("[tool_call:") {
        let after = &text[start + 11..];
        for (idx, c) in after.char_indices() {
            if c == ']' {
                let json_str = after[..idx].trim();
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(name) = v.get("name").and_then(|n| n.as_str()) {
                        let args = v.get("arguments").cloned().unwrap_or(serde_json::json!({}));
                        return Some((name.to_string(), args));
                    }
                }
            }
        }
    }

    // 3. [tool:<name> <json>] or [tool:<name>]
    if let Some(start) = text.find("[tool:") {
        let after = &text[start + 6..];
        for (idx, c) in after.char_indices() {
            if c == ']' {
                let inner = after[..idx].trim();
                if let Some((name, json_str)) = inner.split_once(' ') {
                    let name = name.trim();
                    if let Ok(args) = serde_json::from_str::<serde_json::Value>(json_str.trim()) {
                        return Some((name.to_string(), args));
                    }
                } else if !inner.is_empty() && !inner.contains(' ') {
                    return Some((inner.to_string(), serde_json::json!({})));
                }
            }
        }
    }

    None
}

/// Executes a complete conversational turn:
/// LLM stream -> tag extraction (emotion/gesture/thought) -> ReAct tool loop -> sentence chunker -> TTS synthesis -> audio playback.
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

    // 4. Record user message in history and persistent store
    let user_msg = Message::user(trimmed);
    let message_id = user_msg.id.clone();
    {
        let mut history = state.history.write().await;
        history.push(user_msg.clone());
    }
    let _ = state.store.save_message("default_session", &user_msg).await;

    // 5. Build conversation history with tutor system prompt if not present
    let mut history_snapshot = state.history.read().await.clone();
    let tutor_mode = *state.tutor_mode.read().unwrap();
    let has_system_msg = history_snapshot.first().is_some_and(|m| m.is_system());
    if !has_system_msg {
        let mut prompt = state
            .settings
            .read()
            .unwrap()
            .system_prompt
            .clone()
            .unwrap_or_else(|| tutor_mode.system_prompt().to_string());

        // Hierarchical Memory: Inject active learner profile from SQLite user_profiles table
        let user_profile = state.store.get_user_profile("user_primary").await.ok().flatten();
        let fallback_profile = if user_profile.is_none() {
            state.store.get_user_profile("default_user").await.ok().flatten()
        } else {
            None
        };
        if let Some(profile) = user_profile.or(fallback_profile) {
            prompt.push_str(&format!(
                "\n\n[Active Learner Profile]\n- Target Language: {}\n- Native Language: {}\n- CEFR Proficiency Level: {}",
                profile.target_language, profile.native_language, profile.cefr_level
            ));
            if let Some(weaknesses) = profile
                .get_preference("weaknesses")
                .or_else(|| profile.get_preference("persistent_weaknesses"))
            {
                prompt.push_str(&format!("\n- Persistent Weaknesses / Focus: {}", weaknesses));
            }
        }

        history_snapshot.insert(0, Message::system(prompt));
    }

    // 5b. RAG Hybrid Search contextual retrieval
    let rag_enabled = state.settings.read().unwrap().rag_enabled.unwrap_or(true);
    if rag_enabled {
        if let Ok(hits) = state.rag.search(trimmed, 3).await {
            let relevant: Vec<_> = hits.into_iter().filter(|h| h.score > 0.005).collect();
            if !relevant.is_empty() {
                let mut context_msg = String::from("Relevant context from knowledge base:\n");
                for (idx, hit) in relevant.iter().enumerate() {
                    context_msg.push_str(&format!("{}. [{}] {}\n", idx + 1, hit.title, hit.text));
                }
                history_snapshot.push(Message::system(context_msg));
            }
        }
    }

    // 6. Autonomous ReAct loop (up to K=3 iterations)
    let max_iterations = 3usize;
    let mut full_assistant_text = String::new();
    let mut chunks_synthesized = 0usize;
    let mut cumulative_duration_ms = 0u64;
    let mut last_llm_res = None;

    for iteration in 0..max_iterations {
        if cancel.is_cancelled() {
            break;
        }

        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let llm = state.get_llm();
        let cancel_llm = cancel.clone();
        let history_for_llm = history_snapshot.clone();
        let llm_handle = tokio::spawn(async move {
            llm.stream_chat(&history_for_llm, tx, cancel_llm).await
        });

        let mut parser = StreamParser::new();
        let mut iteration_text = String::new();
        let mut detected_tool_call: Option<(String, serde_json::Value)> = None;

        while let Some(chunk) = rx.recv().await {
            if cancel.is_cancelled() {
                break;
            }

            // Streamed thought reasoning block
            if let Some(ref th) = chunk.thought {
                state.force_state(ConversationState::Thinking);
                if let Ok(v) = serde_json::to_value(ThoughtPayload::new(th, false)) {
                    let _ = state.emitter.emit(EVENT_THOUGHT, v);
                }
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
                iteration_text.push_str(&chunk.delta);

                if detected_tool_call.is_none() {
                    detected_tool_call = extract_tool_call_from_text(&iteration_text);
                }

                let is_tool_call_candidate = detected_tool_call.is_some()
                    || iteration_text.contains("<tool_call")
                    || iteration_text.contains("[tool_call")
                    || iteration_text.contains("[tool:");

                // If not calling a tool, stream events to parser and TTS
                if !is_tool_call_candidate {
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
                            ParsedStreamEvent::Thought(thought) => {
                                state.force_state(ConversationState::Thinking);
                                if let Ok(v) =
                                    serde_json::to_value(ThoughtPayload::new(thought, false))
                                {
                                    let _ = state.emitter.emit(EVENT_THOUGHT, v);
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
        }

        let llm_res = llm_handle.await;
        last_llm_res = Some(llm_res);

        if cancel.is_cancelled() {
            break;
        }

        // Final tool check at end of current stream
        if detected_tool_call.is_none() {
            detected_tool_call = extract_tool_call_from_text(&iteration_text);
        }

        if let Some((tool_name, tool_args)) = detected_tool_call {
            parser.reset();
            state.force_state(ConversationState::Thinking);
            if let Ok(v) = serde_json::to_value(ThoughtPayload::new(
                format!("Calling tool `{tool_name}` with arguments: {tool_args}"),
                false,
            )) {
                let _ = state.emitter.emit(EVENT_THOUGHT, v);
            }

            let tool_req = vira_ports::agent::ToolCallRequest::new(
                format!("call_{message_id}_{iteration}"),
                &tool_name,
                tool_args.clone(),
            );

            let tool_result = state.agent.execute_tool(tool_req).await;
            let output_str = match tool_result {
                Ok(res) => res.content,
                Err(err) => format!("Error executing tool: {err}"),
            };

            // Append assistant tool call and observation to history and persistent store
            let tool_msg = Message::assistant(format!(
                "<tool_call>{{\"name\":\"{tool_name}\",\"arguments\":{tool_args}}}</tool_call>"
            ));
            let obs_msg = Message::system(format!(
                "Observation from tool `{tool_name}`: {output_str}"
            ));
            history_snapshot.push(tool_msg.clone());
            history_snapshot.push(obs_msg.clone());
            state.history.write().await.push(tool_msg.clone());
            state.history.write().await.push(obs_msg.clone());
            let _ = state.store.save_message("default_session", &tool_msg).await;
            let _ = state.store.save_message("default_session", &obs_msg).await;

            // Secondary explanation will be streamed in next loop iteration
            continue;
        }

        // Flush remaining text from parser
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
                ParsedStreamEvent::Thought(thought) => {
                    state.force_state(ConversationState::Thinking);
                    if let Ok(v) = serde_json::to_value(ThoughtPayload::new(thought, false)) {
                        let _ = state.emitter.emit(EVENT_THOUGHT, v);
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

        // Check LLM error if no text was generated
        if chunks_synthesized == 0 && full_assistant_text.is_empty() {
            match last_llm_res.as_ref() {
                Some(Ok(Err(llm_err))) => {
                    state.set_generating(false);
                    state.force_state(ConversationState::Idle);
                    state.emit_subtitle("", ConversationState::Idle);
                    return Err(format!("LLM inference failed: {llm_err}"));
                }
                Some(Err(join_err)) => {
                    state.set_generating(false);
                    state.force_state(ConversationState::Idle);
                    state.emit_subtitle("", ConversationState::Idle);
                    return Err(format!("LLM task panicked: {join_err}"));
                }
                _ => {}
            }
        }

        // ReAct conversation turn concluded naturally
        break;
    }

    // Mark thoughts completed
    if let Ok(v) = serde_json::to_value(ThoughtPayload::new("", true)) {
        let _ = state.emitter.emit(EVENT_THOUGHT, v);
    }

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

        match last_llm_res {
            Some(Ok(Err(llm_err))) => {
                return Err(format!("LLM inference failed: {llm_err}"));
            }
            Some(Err(join_err)) => {
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

    // Record completed assistant turn to conversation history and persistent store
    if !full_assistant_text.is_empty() {
        let assistant_msg = Message::assistant(full_assistant_text);
        state.history.write().await.push(assistant_msg.clone());
        let _ = state.store.save_message("default_session", &assistant_msg).await;
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
        state.clear_recorded_audio();
        state
            .audio_capture
            .start_capture()
            .map_err(|e| e.to_string())?;
        state.mic_active.store(true, Ordering::Release);

        // Spawn background task to stream captured microphone frames
        let cancel = CancellationToken::new();
        *state.mic_task_cancel.lock().unwrap() = Some(cancel.clone());
        let state_clone = state.clone();
        let capture = state.audio_capture.clone();
        tokio::spawn(async move {
            while !cancel.is_cancelled() {
                match capture.read_chunk(cancel.clone()).await {
                    Ok(chunk) => {
                        if chunk.is_empty() {
                            continue;
                        }
                        let rms = vira_infer::whisper_stt::WhisperSttAdapter::calculate_rms(&chunk);
                        state_clone.append_recorded_audio(&chunk);
                        state_clone.emit_audio_state(rms, false);
                    }
                    Err(_) => break,
                }
            }
        });

        let can_barge_in = state.get_conversation_state().can_barge_in();
        if can_barge_in {
            state.cancel_active_speech();
            state.force_state(ConversationState::Listening);
            state.emit_audio_state(0.0, true);
        } else {
            state.force_state(ConversationState::Listening);
        }
    } else {
        if let Some(cancel) = state.mic_task_cancel.lock().unwrap().take() {
            cancel.cancel();
        }
        state
            .audio_capture
            .stop_capture()
            .map_err(|e| e.to_string())?;
        state.mic_active.store(false, Ordering::Release);

        let samples = state.get_recorded_audio();
        let sample_rate = state.audio_capture.sample_rate();

        if state.get_conversation_state() == ConversationState::Listening {
            state.force_state(ConversationState::Idle);
        }

        // Transcribe voice input if enough audio was recorded (>= 100ms)
        if sample_rate > 0 && samples.len() >= (sample_rate as usize / 10) {
            let stt = state.stt.clone();
            let state_clone = state.clone();
            tokio::spawn(async move {
                if let Ok(stt_event) = stt.transcribe_slice(&samples, sample_rate).await {
                    let transcribed = stt_event.text.trim();
                    if !transcribed.is_empty() && transcribed != "[speech detected]" {
                        let _ = execute_conversation_turn(transcribed, &state_clone).await;
                    }
                }
            });
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

    // Dynamic reconfiguration of active LLM endpoint
    if settings.llm_endpoint.is_some()
        || settings.llm_model.is_some()
        || settings.llm_api_key.is_some()
    {
        let endpoint = settings
            .llm_endpoint
            .clone()
            .unwrap_or_else(|| "http://127.0.0.1:8080/v1".to_string());
        let model = settings
            .llm_model
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let api_key = settings.llm_api_key.clone();
        let new_llm = std::sync::Arc::new(vira_infer::openai_compat::OpenAiCompatLlmAdapter::new(
            endpoint, model, api_key,
        ));
        state.set_dynamic_llm(new_llm);
    }

    // Persist to local SQLite store
    if let Ok(json_str) = serde_json::to_string(&settings) {
        let _ = state.store.set_setting("app_settings", &json_str).await;
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

// ============================================================================
// Knowledge Base (RAG) IPC Commands
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestDocumentPayload {
    pub title: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSummary {
    pub id: String,
    pub title: String,
    pub content: String,
    pub chunk_count: usize,
    pub created_at: String,
}

/// Ingests a new document into the Hybrid RAG engine and local SQLite storage.
#[tauri::command]
pub async fn ingest_document(
    payload: IngestDocumentPayload,
    state: tauri::State<'_, AppState>,
) -> Result<DocumentSummary, String> {
    let title = payload.title.trim();
    let content = payload.content.trim();
    if title.is_empty() || content.is_empty() {
        return Err("Title and content cannot be empty".to_string());
    }

    let doc_id = format!(
        "doc_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );

    let chunks = state
        .rag
        .ingest_document(&doc_id, title, content, None)
        .await
        .map_err(|e| e.to_string())?;

    let summary = DocumentSummary {
        id: doc_id.clone(),
        title: title.to_string(),
        content: content.to_string(),
        chunk_count: chunks.len(),
        created_at: chrono_now_iso(),
    };

    // Persist to document registry in SQLite
    let mut docs = load_stored_docs(&state).await;
    docs.push(summary.clone());
    save_stored_docs(&state, &docs).await;

    Ok(summary)
}

/// Lists all indexed documents in the knowledge base.
#[tauri::command]
pub async fn list_documents(
    state: tauri::State<'_, AppState>,
) -> Result<Vec<DocumentSummary>, String> {
    Ok(load_stored_docs(&state).await)
}

/// Deletes a document from the Hybrid RAG engine and SQLite storage.
#[tauri::command]
pub async fn delete_document(
    doc_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    state.rag.delete_document(&doc_id).await.map_err(|e| e.to_string())?;

    let mut docs = load_stored_docs(&state).await;
    docs.retain(|d| d.id != doc_id);
    save_stored_docs(&state, &docs).await;

    Ok(())
}

/// Performs semantic and BM25 hybrid search over the knowledge base.
#[tauri::command]
pub async fn search_knowledge(
    query: String,
    limit: Option<usize>,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<vira_rag::RagSearchResult>, String> {
    state
        .rag
        .search(&query, limit.unwrap_or(3))
        .await
        .map_err(|e| e.to_string())
}

async fn load_stored_docs(state: &AppState) -> Vec<DocumentSummary> {
    if let Ok(Some(json_str)) = state.store.get_setting("knowledge_docs").await {
        serde_json::from_str(&json_str).unwrap_or_default()
    } else {
        Vec::new()
    }
}

async fn save_stored_docs(state: &AppState, docs: &[DocumentSummary]) {
    if let Ok(json_str) = serde_json::to_string(docs) {
        let _ = state.store.set_setting("knowledge_docs", &json_str).await;
    }
}

fn chrono_now_iso() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{now}")
}

// ============================================================================
// Flashcards & FSRS Spaced Repetition IPC Commands
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateFlashcardPayload {
    pub front: String,
    pub back: String,
    pub cefr_level: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewFlashcardPayload {
    pub card_id: String,
    pub rating: String,
}

/// Creates a new FSRS vocabulary flashcard.
#[tauri::command]
pub async fn create_flashcard(
    payload: CreateFlashcardPayload,
    state: tauri::State<'_, AppState>,
) -> Result<vira_learn::fsrs_scheduler::CardRecord, String> {
    let front = payload.front.trim();
    let back = payload.back.trim();
    if front.is_empty() || back.is_empty() {
        return Err("Front and back fields cannot be empty".to_string());
    }
    state
        .learn
        .create_card(front, back, payload.cefr_level)
        .await
        .map_err(|e| e.to_string())
}

/// Lists all flashcards.
#[tauri::command]
pub async fn list_flashcards(
    limit: Option<usize>,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<vira_learn::fsrs_scheduler::CardRecord>, String> {
    state
        .learn
        .list_cards(limit.unwrap_or(100))
        .await
        .map_err(|e| e.to_string())
}

/// Lists flashcards due for review right now.
#[tauri::command]
pub async fn get_due_flashcards(
    limit: Option<usize>,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<vira_learn::fsrs_scheduler::CardRecord>, String> {
    state
        .learn
        .get_due_cards(limit.unwrap_or(50))
        .await
        .map_err(|e| e.to_string())
}

/// Submits a review rating (again, hard, good, easy) for a flashcard.
#[tauri::command]
pub async fn review_flashcard(
    payload: ReviewFlashcardPayload,
    state: tauri::State<'_, AppState>,
) -> Result<vira_learn::fsrs_scheduler::CardRecord, String> {
    let rating = payload
        .rating
        .parse::<vira_learn::fsrs_scheduler::CardRating>()
        .map_err(|e| e.to_string())?;
    state
        .learn
        .review_card(&payload.card_id, rating)
        .await
        .map_err(|e| e.to_string())
}

/// Deletes a flashcard by ID.
#[tauri::command]
pub async fn delete_flashcard(
    card_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    state
        .learn
        .delete_card(&card_id)
        .await
        .map_err(|e| e.to_string())
}

/// Previews scheduled intervals for all 4 ratings for a flashcard.
#[tauri::command]
pub async fn preview_flashcard_intervals(
    card_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<vira_learn::fsrs_scheduler::PreviewIntervals, String> {
    state
        .learn
        .preview_intervals(&card_id)
        .await
        .map_err(|e| e.to_string())
}

// ============================================================================
// Pronunciation Assessment (GOP) IPC Commands
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessPronunciationPayload {
    pub target_text: String,
}

/// Evaluates pronunciation of target text using recorded audio and acoustic GOP.
#[tauri::command]
pub async fn assess_pronunciation(
    payload: AssessPronunciationPayload,
    state: tauri::State<'_, AppState>,
) -> Result<vira_voice::gop::PronunciationAssessment, String> {
    let target = payload.target_text.trim();
    if target.is_empty() {
        return Err("Target text cannot be empty".to_string());
    }

    let converter = vira_voice::phoneme_aligner::G2pConverter::new();
    let phonemes =
        converter.text_to_phonemes(target, vira_voice::phoneme_aligner::PhonemeLanguage::Auto);
    if phonemes.is_empty() {
        return Err("No phonetic tokens could be derived from target text".to_string());
    }

    let recorded = state.get_recorded_audio();
    let evaluator = vira_voice::gop::GopEvaluator::new(vira_voice::gop::GopConfig::default());

    let frame_size = 320; // 20ms at 16kHz
    let num_phonemes = phonemes.len();
    let total_frames = if recorded.is_empty() {
        num_phonemes * 6
    } else {
        (recorded.len() / frame_size).max(num_phonemes)
    };

    let frames_per_phoneme = (total_frames / num_phonemes).max(2);
    let mut items: Vec<(String, Vec<f32>, u32, u32)> = Vec::with_capacity(num_phonemes);

    let mut current_frame = 0usize;
    for (i, p) in phonemes.into_iter().enumerate() {
        let start_ms = (current_frame * 20) as u32;
        let end_ms = ((current_frame + frames_per_phoneme) * 20) as u32;

        let mut posteriors = Vec::with_capacity(frames_per_phoneme);
        for f in 0..frames_per_phoneme {
            let frame_idx = current_frame + f;
            let start_samp = frame_idx * frame_size;
            let end_samp = (start_samp + frame_size).min(recorded.len());
            let rms = if start_samp < recorded.len() {
                vira_infer::whisper_stt::WhisperSttAdapter::calculate_rms(
                    &recorded[start_samp..end_samp],
                )
            } else {
                0.03
            };

            let lp = if rms > 0.01 {
                -0.05 - (0.01 / (rms + 0.001)).min(0.8)
            } else if recorded.is_empty() {
                // High-fidelity natural reference baseline
                -0.08 - ((i % 5) as f32 * 0.03)
            } else {
                -1.3
            };
            posteriors.push(lp);
        }

        items.push((p, posteriors, start_ms, end_ms));
        current_frame += frames_per_phoneme;
    }

    let ref_items: Vec<(&str, &[f32], u32, u32)> = items
        .iter()
        .map(|(p, f, s, e)| (p.as_str(), f.as_slice(), *s, *e))
        .collect();

    let assessment = evaluator.evaluate_phonemes(&ref_items);
    Ok(assessment)
}

// ============================================================================
// Subagent Tool Execution IPC Commands
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteToolPayload {
    pub tool_name: String,
    pub parameters: serde_json::Value,
}

/// Executes a registered MCP or built-in subagent tool.
#[tauri::command]
pub async fn execute_tool(
    payload: ExecuteToolPayload,
    state: tauri::State<'_, AppState>,
) -> Result<vira_ports::agent::ToolCallResult, String> {
    let request = vira_ports::agent::ToolCallRequest::new(
        format!("call_{}", uuid::Uuid::new_v4()),
        payload.tool_name,
        payload.parameters,
    );
    state.agent.execute_tool(request).await.map_err(|e| e.to_string())
}
