use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use vira_domain::state::ConversationState;
use vira_domain::viseme::VisemeWeights;

pub const EVENT_VISEME_TICK: &str = "viseme_tick";
pub const EVENT_SUBTITLE_UPDATE: &str = "subtitle_update";
pub const EVENT_AUDIO_STATE: &str = "audio_state";
pub const EVENT_GESTURE: &str = "gesture_event";
pub const EVENT_EMOTION: &str = "emotion_event";
pub const EVENT_STATE_CHANGE: &str = "state_change";

/// Payload emitted for 60Hz Viseme lip-sync ticks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisemeTickPayload {
    pub weights: VisemeWeights,
    pub sample_clock: u64,
    pub timestamp_ms: u64,
    pub is_speaking: bool,
}

impl VisemeTickPayload {
    pub fn new(weights: VisemeWeights, sample_clock: u64, timestamp_ms: u64, is_speaking: bool) -> Self {
        Self {
            weights,
            sample_clock,
            timestamp_ms,
            is_speaking,
        }
    }
}

/// Payload emitted when subtitle text or conversational state updates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtitleUpdatePayload {
    pub text: String,
    pub state: ConversationState,
    pub timestamp: u64,
}

impl SubtitleUpdatePayload {
    pub fn new(text: impl Into<String>, state: ConversationState) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            text: text.into(),
            state,
            timestamp,
        }
    }
}

/// Payload emitted when audio levels or playback/capture states change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioStatePayload {
    pub rms_level: f32,
    pub is_playing: bool,
    pub is_capturing: bool,
    pub barge_in_triggered: bool,
}

impl AudioStatePayload {
    pub fn new(rms_level: f32, is_playing: bool, is_capturing: bool, barge_in_triggered: bool) -> Self {
        Self {
            rms_level: rms_level.clamp(0.0, 1.0),
            is_playing,
            is_capturing,
            barge_in_triggered,
        }
    }
}

/// Payload emitted when an avatar gesture is triggered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GesturePayload {
    pub name: String,
    pub duration: f32,
}

/// Payload emitted when an avatar emotion is updated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionPayload {
    pub name: String,
    pub weight: f32,
}

/// Abstraction for pushing event updates to the frontend webview or test listeners.
pub trait AppEventEmitter: Send + Sync {
    fn emit(&self, event: &str, payload: serde_json::Value) -> Result<(), String>;
}

/// Production event emitter forwarding to Tauri webview windows.
#[derive(Clone)]
pub struct TauriEventEmitter {
    app_handle: tauri::AppHandle,
}

impl TauriEventEmitter {
    pub fn new(app_handle: tauri::AppHandle) -> Self {
        Self { app_handle }
    }
}

impl AppEventEmitter for TauriEventEmitter {
    fn emit(&self, event: &str, payload: serde_json::Value) -> Result<(), String> {
        use tauri::Emitter;
        self.app_handle
            .emit(event, payload)
            .map_err(|e| e.to_string())
    }
}

/// Mock event emitter capturing all emitted events for tests.
#[derive(Debug, Default, Clone)]
pub struct MockEventEmitter {
    events: Arc<Mutex<Vec<(String, serde_json::Value)>>>,
}

impl MockEventEmitter {
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn recorded_events(&self) -> Vec<(String, serde_json::Value)> {
        self.events.lock().unwrap().clone()
    }

    pub fn clear(&self) {
        self.events.lock().unwrap().clear();
    }

    pub fn find_events(&self, event_name: &str) -> Vec<serde_json::Value> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|(name, _)| name == event_name)
            .map(|(_, val)| val.clone())
            .collect()
    }
}

impl AppEventEmitter for MockEventEmitter {
    fn emit(&self, event: &str, payload: serde_json::Value) -> Result<(), String> {
        self.events.lock().unwrap().push((event.to_string(), payload));
        Ok(())
    }
}

/// Null event emitter that silently drops events.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullEventEmitter;

impl AppEventEmitter for NullEventEmitter {
    fn emit(&self, _event: &str, _payload: serde_json::Value) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_event_emitter_records_and_filters() {
        let emitter = MockEventEmitter::new();
        assert!(emitter.recorded_events().is_empty());

        emitter
            .emit(
                EVENT_SUBTITLE_UPDATE,
                serde_json::to_value(SubtitleUpdatePayload::new("hello", ConversationState::Speaking)).unwrap(),
            )
            .unwrap();

        emitter
            .emit(
                EVENT_AUDIO_STATE,
                serde_json::to_value(AudioStatePayload::new(0.5, true, false, false)).unwrap(),
            )
            .unwrap();

        let events = emitter.recorded_events();
        assert_eq!(events.len(), 2);

        let subs = emitter.find_events(EVENT_SUBTITLE_UPDATE);
        assert_eq!(subs.len(), 1);
        let parsed: SubtitleUpdatePayload = serde_json::from_value(subs[0].clone()).unwrap();
        assert_eq!(parsed.text, "hello");
        assert_eq!(parsed.state, ConversationState::Speaking);
    }
}
