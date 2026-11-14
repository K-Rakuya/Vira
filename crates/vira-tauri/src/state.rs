use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use vira_agent::{ToolCallExecutor, ToolRegistry};
use vira_domain::message::Message;
use vira_domain::state::{ConversationState, StateTransitionError, StateTransitionEvent};
use vira_domain::viseme::VisemeWeights;
use vira_infer::openai_compat::{MockLlmTransport, OpenAiCompatLlmAdapter};
use vira_infer::{SherpaSttAdapter, SherpaTtsAdapter};
use vira_learn::service::SpacedRepetitionService;
use vira_ports::audio::{AudioCapturePort, AudioPlaybackPort};
use vira_ports::llm::LlmPort;
use vira_ports::store::StorePort;
use vira_ports::stt::SttPort;
use vira_ports::tts::{TimedViseme, TtsPort};
use vira_rag::{HybridRagConfig, HybridRagEngine};
use vira_store::SqliteStore;
use vira_voice::capture::CaptureEngine;
use vira_voice::clock::MasterAudioClock;
use vira_voice::playback::PlaybackEngine;

use crate::events::{
    AppEventEmitter, AudioStatePayload, BinaryVisemeFrame, MockEventEmitter, NullEventEmitter,
    SubtitleUpdatePayload, VisemeTickPayload, EVENT_AUDIO_STATE, EVENT_STATE_CHANGE,
    EVENT_SUBTITLE_UPDATE, EVENT_VISEME_TICK,
};

/// Language tutor modes supported by Vira.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TutorMode {
    English,
    Japanese,
    Vietnamese,
    #[default]
    GeneralAssistant,
}

impl TutorMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            TutorMode::English => "english",
            TutorMode::Japanese => "japanese",
            TutorMode::Vietnamese => "vietnamese",
            TutorMode::GeneralAssistant => "general_assistant",
        }
    }

    pub fn system_prompt(&self) -> &'static str {
        match self {
            TutorMode::English => {
                "You are an encouraging and fluent English language tutor. Engage in friendly conversation, provide corrections when appropriate, and explain grammar or vocabulary clearly."
            }
            TutorMode::Japanese => {
                "あなたは親切で自然な日本語のチューターです。ユーザーの日本語学習をサポートし、丁寧で親しみやすい会話を行ってください。"
            }
            TutorMode::Vietnamese => {
                "Bạn là trợ lý và gia sư tiếng Việt thân thiện, thông minh. Hãy đồng hành cùng người dùng bằng giọng văn tự nhiên, ấm áp."
            }
            TutorMode::GeneralAssistant => {
                "You are Vira, an ultra-low-latency 3D desktop AI companion with full voice, lip-sync, and emotional awareness."
            }
        }
    }
}

impl std::fmt::Display for TutorMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for TutorMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().replace(['-', ' '], "_").as_str() {
            "english" | "en" | "english_tutor" | "englishtutor" => Ok(TutorMode::English),
            "japanese" | "ja" | "jp" | "japanese_tutor" | "japanesetutor" => Ok(TutorMode::Japanese),
            "vietnamese" | "vi" | "vn" | "vietnamese_tutor" | "vietnamesetutor" => Ok(TutorMode::Vietnamese),
            "general_assistant" | "general" | "assistant" | "default" | "general_tutor" | "generaltutor" => {
                Ok(TutorMode::GeneralAssistant)
            }
            other => Err(format!("Unknown tutor mode: '{other}'")),
        }
    }
}

/// User application settings payload for persistent configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AppSettingsPayload {
    #[serde(alias = "llm_base_url")]
    pub llm_endpoint: Option<String>,
    pub llm_model: Option<String>,
    pub llm_api_key: Option<String>,
    pub tts_voice: Option<String>,
    #[serde(alias = "speech_rate")]
    pub tts_speed: Option<f32>,
    pub stt_model: Option<String>,
    pub tutor_mode: Option<String>,
    pub rag_enabled: Option<bool>,
    pub system_prompt: Option<String>,
    pub temperature: Option<f32>,
    pub tts_endpoint: Option<String>,
    pub stt_endpoint: Option<String>,
}

/// Comprehensive state snapshot returned by `get_app_state`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GetAppStateResponse {
    pub state: ConversationState,
    pub emotion: String,
    pub emotion_weight: f32,
    pub tutor_mode: TutorMode,
    pub mic_active: bool,
    pub is_speaking: bool,
    pub current_subtitle: String,
    pub sample_clock: u64,
    pub current_time_ms: u64,
}

/// Lock-free atomic conversation state wrapper using `AtomicU8` with CAS transitions.
#[derive(Debug)]
pub struct AtomicConversationState {
    pub inner: AtomicU8,
}

impl AtomicConversationState {
    pub const fn new(initial: ConversationState) -> Self {
        Self {
            inner: AtomicU8::new(Self::to_u8(initial)),
        }
    }

    pub const fn to_u8(state: ConversationState) -> u8 {
        match state {
            ConversationState::Idle => 0,
            ConversationState::Listening => 1,
            ConversationState::Thinking => 2,
            ConversationState::Speaking => 3,
            ConversationState::Interrupted => 4,
        }
    }

    pub const fn from_u8(val: u8) -> ConversationState {
        match val {
            0 => ConversationState::Idle,
            1 => ConversationState::Listening,
            2 => ConversationState::Thinking,
            3 => ConversationState::Speaking,
            4 => ConversationState::Interrupted,
            _ => ConversationState::Idle,
        }
    }

    #[inline(always)]
    pub fn get(&self) -> ConversationState {
        Self::from_u8(self.inner.load(Ordering::Acquire))
    }

    #[inline(always)]
    pub fn set(&self, state: ConversationState) {
        self.inner.store(Self::to_u8(state), Ordering::Release);
    }

    /// Compare-and-swap state transition conforming to domain state machine rules.
    /// Returns `(previous_state, next_state)` on success.
    pub fn transition_to(
        &self,
        target: ConversationState,
    ) -> Result<(ConversationState, ConversationState), StateTransitionError> {
        let mut current_raw = self.inner.load(Ordering::Acquire);
        loop {
            let current = Self::from_u8(current_raw);
            if current == target {
                return Ok((current, current));
            }
            let next = current.transition_to(target)?;

            match self.inner.compare_exchange_weak(
                current_raw,
                Self::to_u8(next),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok((current, next)),
                Err(actual) => current_raw = actual,
            }
        }
    }

    /// Performs an explicit atomic compare-and-swap operation.
    pub fn compare_exchange(
        &self,
        current: ConversationState,
        new: ConversationState,
    ) -> Result<ConversationState, ConversationState> {
        self.inner
            .compare_exchange(
                Self::to_u8(current),
                Self::to_u8(new),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(Self::from_u8)
            .map_err(Self::from_u8)
    }

    #[inline(always)]
    pub fn as_atomic(&self) -> &AtomicU8 {
        &self.inner
    }
}

impl Default for AtomicConversationState {
    fn default() -> Self {
        Self::new(ConversationState::Idle)
    }
}

impl std::ops::Deref for AtomicConversationState {
    type Target = AtomicU8;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Core application state managed by Tauri and accessed across IPC command threads.
#[derive(Clone)]
pub struct AppState {
    pub llm: Arc<dyn LlmPort>,
    pub dynamic_llm: Arc<RwLock<Option<Arc<dyn LlmPort>>>>,
    pub tts: Arc<dyn TtsPort>,
    pub dynamic_tts: Arc<RwLock<Option<Arc<dyn TtsPort>>>>,
    pub stt: Arc<dyn SttPort>,
    pub dynamic_stt: Arc<RwLock<Option<Arc<dyn SttPort>>>>,
    pub audio_playback: Arc<dyn AudioPlaybackPort>,
    pub audio_capture: Arc<dyn AudioCapturePort>,
    pub rag: Arc<HybridRagEngine>,
    pub store: Arc<dyn StorePort>,
    pub learn: Arc<SpacedRepetitionService>,
    pub agent: Arc<ToolCallExecutor>,
    pub clock: MasterAudioClock,
    pub conversation_state: Arc<AtomicConversationState>,
    pub turn_epoch: Arc<AtomicU64>,
    pub emotion: Arc<RwLock<(String, f32)>>,
    pub tutor_mode: Arc<RwLock<TutorMode>>,
    pub mic_active: Arc<AtomicBool>,
    pub current_subtitle: Arc<RwLock<String>>,
    pub active_cancellation: Arc<Mutex<Option<CancellationToken>>>,
    pub history: Arc<tokio::sync::RwLock<Vec<Message>>>,
    pub emitter: Arc<dyn AppEventEmitter>,
    pub active_visemes: Arc<RwLock<Vec<TimedViseme>>>,
    pub settings: Arc<RwLock<AppSettingsPayload>>,
    pub is_generating: Arc<AtomicBool>,
    pub turn_lock: Arc<tokio::sync::Mutex<()>>,
    pub recorded_audio: Arc<Mutex<Vec<f32>>>,
    pub mic_task_cancel: Arc<Mutex<Option<CancellationToken>>>,
}

impl AppState {
    /// Creates a production `AppState` with explicit dependencies.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        llm: Arc<dyn LlmPort>,
        tts: Arc<dyn TtsPort>,
        stt: Arc<dyn SttPort>,
        audio_playback: Arc<dyn AudioPlaybackPort>,
        audio_capture: Arc<dyn AudioCapturePort>,
        rag: Arc<HybridRagEngine>,
        store: Arc<dyn StorePort>,
        learn: Arc<SpacedRepetitionService>,
        agent: Arc<ToolCallExecutor>,
        clock: MasterAudioClock,
        emitter: Arc<dyn AppEventEmitter>,
    ) -> Self {
        Self {
            llm,
            dynamic_llm: Arc::new(RwLock::new(None)),
            tts,
            dynamic_tts: Arc::new(RwLock::new(None)),
            stt,
            dynamic_stt: Arc::new(RwLock::new(None)),
            audio_playback,
            audio_capture,
            rag,
            store,
            learn,
            agent,
            clock,
            conversation_state: Arc::new(AtomicConversationState::new(ConversationState::Idle)),
            turn_epoch: Arc::new(AtomicU64::new(0)),
            emotion: Arc::new(RwLock::new(("neutral".to_string(), 1.0))),
            tutor_mode: Arc::new(RwLock::new(TutorMode::GeneralAssistant)),
            mic_active: Arc::new(AtomicBool::new(false)),
            current_subtitle: Arc::new(RwLock::new(String::new())),
            active_cancellation: Arc::new(Mutex::new(None)),
            history: Arc::new(tokio::sync::RwLock::new(Vec::new())),
            emitter,
            active_visemes: Arc::new(RwLock::new(Vec::new())),
            settings: Arc::new(RwLock::new(AppSettingsPayload::default())),
            is_generating: Arc::new(AtomicBool::new(false)),
            turn_lock: Arc::new(tokio::sync::Mutex::new(())),
            recorded_audio: Arc::new(Mutex::new(Vec::new())),
            mic_task_cancel: Arc::new(Mutex::new(None)),
        }
    }

    /// Creates a headless/mock `AppState` ideal for automated unit and integration tests.
    pub fn new_headless() -> Self {
        Self::new_headless_with_emitter(Arc::new(MockEventEmitter::new()))
    }

    /// Creates a headless `AppState` configured with a specific event emitter.
    pub fn new_headless_with_emitter(emitter: Arc<dyn AppEventEmitter>) -> Self {
        let (playback, _) = PlaybackEngine::new_virtual(48000, 96000);
        let clock = playback.clock().clone();
        let pb_sample_rate = playback.sample_rate();
        let (capture, _) = CaptureEngine::new_virtual(16000, 512);

        let transport = Arc::new(MockLlmTransport::from_deltas(&[
            "[emotion:happy][gesture:wave] Hello! I am Vira.",
        ]));
        let llm = Arc::new(
            OpenAiCompatLlmAdapter::new(
                "http://127.0.0.1:8080/v1".to_string(),
                "default".to_string(),
                None,
            )
            .with_transport(transport),
        );
        let tts = Arc::new(SherpaTtsAdapter::virtual_fallback().with_sample_rate(pb_sample_rate));
        let stt = Arc::new(SherpaSttAdapter::virtual_fallback());
        let rag = Arc::new(
            HybridRagEngine::new_in_memory(HybridRagConfig::default())
                .expect("in-memory RAG engine"),
        );
        let store = Arc::new(SqliteStore::open_in_memory().expect("in-memory SqliteStore"));
        let learn = Arc::new(SpacedRepetitionService::new(store.clone()));

        let mut tool_registry = ToolRegistry::new();
        tool_registry.register_builtin_tools();
        let agent = Arc::new(ToolCallExecutor::new(Arc::new(tokio::sync::RwLock::new(
            tool_registry,
        ))));

        Self::new(
            llm,
            tts,
            stt,
            Arc::new(playback),
            Arc::new(capture),
            rag,
            store,
            learn,
            agent,
            clock,
            emitter,
        )
    }

    /// Replaces the audio capture port (useful for injecting `VirtualAudioFeed` during tests).
    pub fn with_audio_capture(mut self, capture: Arc<dyn AudioCapturePort>) -> Self {
        self.audio_capture = capture;
        self
    }

    /// Obtains the active LLM port (either dynamically reconfigured or initial fallback).
    pub fn get_llm(&self) -> Arc<dyn LlmPort> {
        if let Some(ref dynamic) = *self.dynamic_llm.read().unwrap() {
            return dynamic.clone();
        }
        self.llm.clone()
    }

    /// Dynamically reconfigures the active LLM instance.
    pub fn set_dynamic_llm(&self, llm: Arc<dyn LlmPort>) {
        *self.dynamic_llm.write().unwrap() = Some(llm);
    }

    /// Obtains the active TTS port (either dynamically reconfigured or initial fallback).
    pub fn get_tts(&self) -> Arc<dyn TtsPort> {
        if let Some(ref dynamic) = *self.dynamic_tts.read().unwrap() {
            return dynamic.clone();
        }
        self.tts.clone()
    }

    /// Dynamically reconfigures the active TTS instance.
    pub fn set_dynamic_tts(&self, tts: Arc<dyn TtsPort>) {
        *self.dynamic_tts.write().unwrap() = Some(tts);
    }

    /// Obtains the active STT port (either dynamically reconfigured or initial fallback).
    pub fn get_stt(&self) -> Arc<dyn SttPort> {
        if let Some(ref dynamic) = *self.dynamic_stt.read().unwrap() {
            return dynamic.clone();
        }
        self.stt.clone()
    }

    /// Dynamically reconfigures the active STT instance.
    pub fn set_dynamic_stt(&self, stt: Arc<dyn SttPort>) {
        *self.dynamic_stt.write().unwrap() = Some(stt);
    }

    /// Returns a copy of the recorded microphone audio samples.
    pub fn get_recorded_audio(&self) -> Vec<f32> {
        self.recorded_audio.lock().unwrap().clone()
    }

    /// Appends PCM audio samples to the microphone recording buffer.
    pub fn append_recorded_audio(&self, samples: &[f32]) {
        self.recorded_audio.lock().unwrap().extend_from_slice(samples);
    }

    /// Takes the recorded microphone audio buffer, clearing it atomically.
    pub fn take_recorded_audio(&self) -> Vec<f32> {
        let mut guard = self.recorded_audio.lock().unwrap();
        let samples = guard.clone();
        guard.clear();
        samples
    }

    /// Clears the microphone recording buffer.
    pub fn clear_recorded_audio(&self) {
        self.recorded_audio.lock().unwrap().clear();
    }

    /// Returns true if an active LLM generation or TTS synthesis pipeline is in progress.
    pub fn is_generating(&self) -> bool {
        self.is_generating.load(Ordering::Acquire)
    }

    /// Sets the active generation flag.
    pub fn set_generating(&self, val: bool) {
        self.is_generating.store(val, Ordering::Release);
    }

    /// Read the current conversation state via wait-free atomic load.
    #[inline(always)]
    pub fn get_conversation_state(&self) -> ConversationState {
        self.conversation_state.get()
    }

    /// Transition to target state, validating against domain rules via atomic CAS and emitting a transition event.
    pub fn transition_state(
        &self,
        target: ConversationState,
    ) -> Result<ConversationState, StateTransitionError> {
        let (current, next) = self.conversation_state.transition_to(target)?;

        if current != next {
            let _ = self.emitter.emit(
                EVENT_STATE_CHANGE,
                serde_json::to_value(StateTransitionEvent::new(current, next)).unwrap_or_default(),
            );
        }

        Ok(next)
    }

    /// Forcibly sets conversation state without transition checks (e.g. error resets).
    pub fn force_state(&self, state: ConversationState) {
        let current = self.get_conversation_state();
        self.conversation_state.set(state);

        let _ = self.emitter.emit(
            EVENT_STATE_CHANGE,
            serde_json::to_value(StateTransitionEvent::new(current, state)).unwrap_or_default(),
        );
    }

    /// Cancels active LLM generation, flushes audio output buffers, resets lip-sync timeline and clock,
    /// and increments turn_epoch for O(1) instant barge-in cancellation.
    pub fn cancel_active_speech(&self) {
        self.turn_epoch.fetch_add(1, Ordering::SeqCst);
        if let Some(token) = self.active_cancellation.lock().unwrap().take() {
            token.cancel();
        }
        let _ = self.audio_playback.flush();
        let _ = self.audio_playback.reset_sample_clock();
        self.clock.reset();
        *self.active_visemes.write().unwrap() = Vec::new();
    }

    /// Obtains the current conversational turn epoch.
    #[inline(always)]
    pub fn current_turn_epoch(&self) -> u64 {
        self.turn_epoch.load(Ordering::Acquire)
    }

    /// Advances the turn epoch to start a new conversational turn and returns the new epoch value.
    #[inline(always)]
    pub fn start_new_turn(&self) -> u64 {
        self.turn_epoch.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Returns true if the specified epoch matches the active conversational turn epoch.
    #[inline(always)]
    pub fn is_current_turn(&self, epoch: u64) -> bool {
        self.turn_epoch.load(Ordering::Acquire) == epoch
    }

    /// Creates a packed 40-byte `BinaryVisemeFrame` for the given weights and playback state.
    pub fn create_binary_viseme_frame(&self, weights: VisemeWeights, is_speaking: bool) -> BinaryVisemeFrame {
        BinaryVisemeFrame::new(
            weights,
            self.clock.current_sample(),
            self.clock.current_time_ms(),
            is_speaking,
        )
    }

    /// Emits a subtitle update event and caches the active subtitle string.
    pub fn emit_subtitle(&self, text: &str, state: ConversationState) {
        *self.current_subtitle.write().unwrap() = text.to_string();
        let payload = SubtitleUpdatePayload::new(text, state);
        if let Ok(val) = serde_json::to_value(payload) {
            let _ = self.emitter.emit(EVENT_SUBTITLE_UPDATE, val);
        }
    }

    /// Emits an audio state event (RMS volume level, playing/capturing, barge-in).
    pub fn emit_audio_state(&self, rms_level: f32, barge_in_triggered: bool) {
        let payload = AudioStatePayload::new(
            rms_level,
            self.audio_playback.is_playing(),
            self.mic_active.load(Ordering::Acquire),
            barge_in_triggered,
        );
        if let Ok(val) = serde_json::to_value(payload) {
            let _ = self.emitter.emit(EVENT_AUDIO_STATE, val);
        }
    }

    /// Emits a 60Hz Viseme lip-sync tick event.
    pub fn emit_viseme_tick(&self, weights: VisemeWeights, is_speaking: bool) {
        let payload = VisemeTickPayload::new(
            weights,
            self.clock.current_sample(),
            self.clock.current_time_ms(),
            is_speaking,
        );
        if let Ok(val) = serde_json::to_value(payload) {
            let _ = self.emitter.emit(EVENT_VISEME_TICK, val);
        }
    }

    /// Obtains a full snapshot of the companion's current operational state.
    pub fn get_snapshot(&self) -> GetAppStateResponse {
        let emotion = self.emotion.read().unwrap().clone();
        GetAppStateResponse {
            state: self.get_conversation_state(),
            emotion: emotion.0,
            emotion_weight: emotion.1,
            tutor_mode: *self.tutor_mode.read().unwrap(),
            mic_active: self.mic_active.load(Ordering::Acquire),
            is_speaking: self.audio_playback.is_playing(),
            current_subtitle: self.current_subtitle.read().unwrap().clone(),
            sample_clock: self.clock.current_sample(),
            current_time_ms: self.clock.current_time_ms(),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new_headless_with_emitter(Arc::new(NullEventEmitter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tutor_mode_parsing() {
        assert_eq!("english".parse::<TutorMode>().unwrap(), TutorMode::English);
        assert_eq!("english_tutor".parse::<TutorMode>().unwrap(), TutorMode::English);
        assert_eq!("englishtutor".parse::<TutorMode>().unwrap(), TutorMode::English);
        assert_eq!("japanese".parse::<TutorMode>().unwrap(), TutorMode::Japanese);
        assert_eq!("japanese_tutor".parse::<TutorMode>().unwrap(), TutorMode::Japanese);
        assert_eq!("vietnamese".parse::<TutorMode>().unwrap(), TutorMode::Vietnamese);
        assert_eq!("vietnameseTutor".parse::<TutorMode>().unwrap(), TutorMode::Vietnamese);
        assert_eq!(
            "general_assistant".parse::<TutorMode>().unwrap(),
            TutorMode::GeneralAssistant
        );
        assert_eq!("general".parse::<TutorMode>().unwrap(), TutorMode::GeneralAssistant);
        assert_eq!("default".parse::<TutorMode>().unwrap(), TutorMode::GeneralAssistant);
        assert!("invalid_mode".parse::<TutorMode>().is_err());
    }

    #[test]
    fn test_app_settings_deserialization_aliases() {
        let json = r#"{
            "llm_base_url": "http://127.0.0.1:11434/v1",
            "llm_model": "llama3.2",
            "speech_rate": 1.15,
            "tutor_mode": "vietnameseTutor"
        }"#;
        let payload: AppSettingsPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.llm_endpoint.as_deref(), Some("http://127.0.0.1:11434/v1"));
        assert_eq!(payload.llm_model.as_deref(), Some("llama3.2"));
        assert_eq!(payload.tts_speed, Some(1.15));
        assert_eq!(payload.tutor_mode.as_deref(), Some("vietnameseTutor"));
    }

    #[test]
    fn test_app_state_initialization_and_snapshot() {
        let state = AppState::new_headless();
        let snapshot = state.get_snapshot();
        assert_eq!(snapshot.state, ConversationState::Idle);
        assert_eq!(snapshot.emotion, "neutral");
        assert_eq!(snapshot.tutor_mode, TutorMode::GeneralAssistant);
        assert!(!snapshot.mic_active);
        assert!(!snapshot.is_speaking);
        assert_eq!(snapshot.sample_clock, 0);
    }

    #[test]
    fn test_state_transitions_and_events() {
        let emitter = Arc::new(MockEventEmitter::new());
        let state = AppState::new_headless_with_emitter(emitter.clone());

        assert!(state.transition_state(ConversationState::Thinking).is_ok());
        assert_eq!(state.get_conversation_state(), ConversationState::Thinking);

        assert!(state.transition_state(ConversationState::Speaking).is_ok());
        assert_eq!(state.get_conversation_state(), ConversationState::Speaking);

        assert!(state.transition_state(ConversationState::Idle).is_ok());
        assert_eq!(state.get_conversation_state(), ConversationState::Idle);

        // Invalid transition: Idle -> Speaking directly
        assert!(state.transition_state(ConversationState::Speaking).is_err());

        let events = emitter.find_events(EVENT_STATE_CHANGE);
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn test_atomic_conversation_state_cas_transitions() {
        let atomic_state = AtomicConversationState::new(ConversationState::Idle);
        assert_eq!(atomic_state.get(), ConversationState::Idle);

        // Valid CAS transition
        let res = atomic_state.transition_to(ConversationState::Listening);
        assert_eq!(res, Ok((ConversationState::Idle, ConversationState::Listening)));
        assert_eq!(atomic_state.get(), ConversationState::Listening);

        // Invalid CAS transition (Listening -> Speaking directly)
        let err = atomic_state.transition_to(ConversationState::Speaking);
        assert!(err.is_err());
        assert_eq!(atomic_state.get(), ConversationState::Listening);

        // Explicit compare_exchange
        let cas_ok = atomic_state.compare_exchange(ConversationState::Listening, ConversationState::Thinking);
        assert_eq!(cas_ok, Ok(ConversationState::Listening));
        assert_eq!(atomic_state.get(), ConversationState::Thinking);

        // Mismatched expected state
        let cas_err = atomic_state.compare_exchange(ConversationState::Idle, ConversationState::Speaking);
        assert_eq!(cas_err, Err(ConversationState::Thinking));
    }

    #[test]
    fn test_turn_epoch_instant_barge_in() {
        let state = AppState::new_headless();
        let epoch0 = state.current_turn_epoch();
        assert_eq!(epoch0, 0);

        let epoch1 = state.start_new_turn();
        assert_eq!(epoch1, 1);
        assert!(state.is_current_turn(1));
        assert!(!state.is_current_turn(0));

        // Barge-in cancels active speech and advances epoch
        state.cancel_active_speech();
        let epoch2 = state.current_turn_epoch();
        assert!(epoch2 > epoch1);
        assert!(!state.is_current_turn(epoch1));
    }
}
