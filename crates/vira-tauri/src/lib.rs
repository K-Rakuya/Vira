//! Tauri 2.0 Host & IPC Bridge for Vira Desktop Companion.

#![deny(clippy::all)]

pub mod commands;
pub mod events;
pub mod state;

use std::sync::Arc;
use tauri::Manager;
use tokio_util::sync::CancellationToken;
use vira_domain::viseme::VisemeWeights;
use vira_infer::kokoro_tts::KokoroTtsAdapter;
use vira_infer::openai_compat::OpenAiCompatLlmAdapter;
use vira_infer::whisper_stt::WhisperSttAdapter;
use vira_ports::audio::AudioPlaybackPort;
use vira_voice::capture::{CaptureConfig, CaptureEngine};
use vira_voice::playback::{PlaybackConfig, PlaybackEngine};

use crate::events::TauriEventEmitter;
use crate::state::AppState;

/// Spawns a dedicated 60Hz (~16.6ms) background loop that synchronizes Viseme blendshape
/// weights against the `MasterAudioClock` playhead and pushes them to the avatar stage.
pub fn spawn_viseme_tick_loop(
    state: AppState,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(16));
        while !cancel.is_cancelled() {
            interval.tick().await;

            let is_playing = state.audio_playback.is_playing();
            let is_gen = state.is_generating();
            let current_conv_state = state.get_conversation_state();

            // Detect speech completion: audio buffer drained, no active LLM/TTS generation,
            // and state is still in Speaking.
            if current_conv_state == vira_domain::state::ConversationState::Speaking && !is_playing && !is_gen {
                state.force_state(vira_domain::state::ConversationState::Idle);
                state.emit_subtitle("", vira_domain::state::ConversationState::Idle);
                *state.active_visemes.write().unwrap() = Vec::new();
                state.emit_viseme_tick(VisemeWeights::zero(), false);
                continue;
            }

            let now_ms = state.clock.current_time_ms();

            let weights = if is_playing {
                let visemes = state.active_visemes.read().unwrap();
                state.clock.sample_viseme_at_ms(&visemes, now_ms)
            } else {
                VisemeWeights::zero()
            };

            state.emit_viseme_tick(weights, is_playing);
        }
    })
}

/// Initializes and launches the Tauri 2.0 application host.
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let handle = app.handle().clone();
            let emitter = Arc::new(TauriEventEmitter::new(handle));

            // Native playback engine with virtual fallback for headless/container environments
            let playback = PlaybackEngine::new_native(PlaybackConfig::default()).unwrap_or_else(|_| {
                let (virtual_pb, _) = PlaybackEngine::new_virtual(48000, 96000);
                virtual_pb
            });
            let clock = playback.clock().clone();
            let pb_sample_rate = playback.sample_rate();

            // Native audio capture with virtual fallback
            let capture = CaptureEngine::new_native(CaptureConfig::default()).unwrap_or_else(|_| {
                let (virtual_cap, _) = CaptureEngine::new_virtual(16000, 512);
                virtual_cap
            });

            // Default AI inference adapters with synchronized sample rate
            let llm = Arc::new(OpenAiCompatLlmAdapter::new(
                "http://127.0.0.1:8080/v1".to_string(),
                "default".to_string(),
                None,
            ));
            let tts = Arc::new(KokoroTtsAdapter::new().with_sample_rate(pb_sample_rate));
            let stt = Arc::new(WhisperSttAdapter::new());

            let app_state = AppState::new(
                llm,
                tts,
                stt,
                Arc::new(playback),
                Arc::new(capture),
                clock,
                emitter,
            );

            // Spawn 60Hz lip-sync broadcast loop
            let loop_cancel = CancellationToken::new();
            spawn_viseme_tick_loop(app_state.clone(), loop_cancel);

            // Register managed state in Tauri container
            app.manage(app_state);

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            crate::commands::send_message,
            crate::commands::trigger_gesture,
            crate::commands::play_gesture,
            crate::commands::set_emotion,
            crate::commands::set_tutor_mode,
            crate::commands::toggle_mic,
            crate::commands::save_settings,
            crate::commands::get_app_state,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Vira Tauri application");
}
