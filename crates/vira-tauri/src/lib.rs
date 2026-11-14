//! Tauri 2.0 Host & IPC Bridge for Vira Desktop Companion.

#![deny(clippy::all)]

pub mod commands;
pub mod events;
pub mod state;

use std::sync::Arc;
use tauri::Manager;
use tokio_util::sync::CancellationToken;
use vira_agent::{ToolCallExecutor, ToolRegistry};
use vira_domain::viseme::VisemeWeights;
use vira_infer::kokoro_tts::KokoroTtsAdapter;
use vira_infer::openai_compat::OpenAiCompatLlmAdapter;
use vira_infer::whisper_stt::WhisperSttAdapter;
use vira_learn::service::SpacedRepetitionService;
use vira_ports::audio::AudioPlaybackPort;
use vira_ports::store::StorePort;
use vira_rag::{HybridRagConfig, HybridRagEngine};
use vira_store::SqliteStore;
use vira_voice::capture::{CaptureConfig, CaptureEngine};
use vira_voice::playback::{PlaybackConfig, PlaybackEngine};

use crate::events::TauriEventEmitter;
use crate::state::{AppSettingsPayload, AppState};

/// Spawns a dedicated 60Hz (~16.6ms) background loop that synchronizes Viseme blendshape
/// weights against the `MasterAudioClock` playhead and pushes them to the avatar stage.
pub fn spawn_viseme_tick_loop(
    state: AppState,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(16));
        let mut was_active = false;
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
                was_active = false;
                continue;
            }

            let is_active = is_playing || is_gen || current_conv_state == vira_domain::state::ConversationState::Speaking;

            if !is_active {
                if was_active {
                    // Emit one zero viseme tick to reset the avatar mouth when entering idle
                    state.emit_viseme_tick(VisemeWeights::zero(), false);
                    was_active = false;
                }
                // Adaptive Viseme Gating: pause sending IPC ticks until audio playback or generation resumes
                continue;
            }

            was_active = true;
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
    let _runtime = if tokio::runtime::Handle::try_current().is_err() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build Tokio runtime");
        Some(rt)
    } else {
        None
    };
    let _guard = _runtime.as_ref().map(|rt| rt.enter());

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

            // Local application data directory for SQLite and RAG index
            let data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("data"));
            let _ = std::fs::create_dir_all(&data_dir);

            // Persistent local SQLite store
            let db_path = data_dir.join("vira.db");
            let store: Arc<dyn StorePort> = Arc::new(
                SqliteStore::open(&db_path)
                    .unwrap_or_else(|_| SqliteStore::open_in_memory().expect("in-memory sqlite store")),
            );

            // Persistent Hybrid RAG Engine
            let rag_dir = data_dir.join("rag");
            let rag = Arc::new(
                HybridRagEngine::new_persistent(&rag_dir, HybridRagConfig::default()).unwrap_or_else(
                    |_| {
                        HybridRagEngine::new_in_memory(HybridRagConfig::default())
                            .expect("in-memory RAG engine")
                    },
                ),
            );

            // FSRS Spaced Repetition Service
            let learn = Arc::new(SpacedRepetitionService::new(store.clone()));

            // Subagent Tool Registry and Executor
            let mut tool_registry = ToolRegistry::new();
            tool_registry.register_builtin_tools();
            tool_registry.register_knowledge_search(rag.clone());
            let agent = Arc::new(ToolCallExecutor::new(Arc::new(tokio::sync::RwLock::new(
                tool_registry,
            ))));

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
                rag,
                store.clone(),
                learn,
                agent,
                clock,
                emitter,
            );

            // Restore saved settings and chat history from SQLite store on boot
            let state_for_init = app_state.clone();
            let store_for_init = store.clone();
            tokio::spawn(async move {
                if let Ok(Some(settings_json)) = store_for_init.get_setting("app_settings").await {
                    if let Ok(saved_settings) =
                        serde_json::from_str::<AppSettingsPayload>(&settings_json)
                    {
                        if saved_settings.llm_endpoint.is_some()
                            || saved_settings.llm_model.is_some()
                            || saved_settings.llm_api_key.is_some()
                        {
                            let endpoint = saved_settings
                                .llm_endpoint
                                .clone()
                                .unwrap_or_else(|| "http://127.0.0.1:8080/v1".to_string());
                            let model = saved_settings
                                .llm_model
                                .clone()
                                .unwrap_or_else(|| "default".to_string());
                            let api_key = saved_settings.llm_api_key.clone();
                            let new_llm = Arc::new(OpenAiCompatLlmAdapter::new(
                                endpoint, model, api_key,
                            ));
                            state_for_init.set_dynamic_llm(new_llm);
                        }

                        if saved_settings.tts_endpoint.is_some()
                            || saved_settings.tts_voice.is_some()
                            || saved_settings.tts_speed.is_some()
                        {
                            let mut tts = KokoroTtsAdapter::new().with_sample_rate(pb_sample_rate);
                            if let Some(ref voice) = saved_settings.tts_voice {
                                tts = tts.with_voice(voice);
                            }
                            if let Some(speed) = saved_settings.tts_speed {
                                tts = tts.with_speed(speed);
                            }
                            if let Some(ref ep) = saved_settings.tts_endpoint {
                                tts = tts.with_endpoint(ep);
                            }
                            state_for_init.set_dynamic_tts(Arc::new(tts));
                        }

                        if saved_settings.stt_endpoint.is_some() {
                            let mut stt = WhisperSttAdapter::new();
                            if let Some(ref ep) = saved_settings.stt_endpoint {
                                stt = stt.with_endpoint(ep);
                            }
                            state_for_init.set_dynamic_stt(Arc::new(stt));
                        }

                        *state_for_init.settings.write().unwrap() = saved_settings;
                    }
                }

                if let Ok(history_msgs) = store_for_init.get_history("default_session", 50).await {
                    if !history_msgs.is_empty() {
                        *state_for_init.history.write().await = history_msgs;
                    }
                }
            });

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
            crate::commands::ingest_document,
            crate::commands::list_documents,
            crate::commands::delete_document,
            crate::commands::search_knowledge,
            crate::commands::create_flashcard,
            crate::commands::list_flashcards,
            crate::commands::get_due_flashcards,
            crate::commands::review_flashcard,
            crate::commands::delete_flashcard,
            crate::commands::preview_flashcard_intervals,
            crate::commands::assess_pronunciation,
            crate::commands::execute_tool,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Vira Tauri application");
}
