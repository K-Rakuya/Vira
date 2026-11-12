use async_trait::async_trait;
use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use vira_domain::message::{Message, Role};
use vira_domain::viseme::VisemeWeights;

use vira_ports::audio::{AudioCapturePort, AudioError, AudioPlaybackPort};
use vira_ports::llm::{LlmError, LlmPort, LlmStreamChunk};
use vira_ports::store::{Flashcard, SessionSummary, StoreError, StorePort, UserProfile};
use vira_ports::stt::{SttError, SttEvent, SttPort};
use vira_ports::tts::{TimedViseme, TtsAudioChunk, TtsError, TtsOutput, TtsPort, TtsStreamEvent};

// -----------------------------------------------------------------------------
// 1. LLM Port Contract Verification
// -----------------------------------------------------------------------------
struct DummyLlm;

#[async_trait]
impl LlmPort for DummyLlm {
    async fn stream_chat(
        &self,
        messages: &[Message],
        sender: mpsc::Sender<LlmStreamChunk>,
        cancel: CancellationToken,
    ) -> Result<(), LlmError> {
        if cancel.is_cancelled() {
            return Err(LlmError::Cancelled);
        }
        for (i, _msg) in messages.iter().enumerate() {
            sender
                .send(LlmStreamChunk::new(format!("chunk_{i}")))
                .await
                .map_err(|e| LlmError::Channel(e.to_string()))?;
        }
        sender
            .send(LlmStreamChunk::done())
            .await
            .map_err(|e| LlmError::Channel(e.to_string()))?;
        Ok(())
    }
}

#[tokio::test]
async fn test_llm_port_contract() {
    let dummy = DummyLlm;
    let (tx, mut rx) = mpsc::channel(4);
    let cancel = CancellationToken::new();

    let msgs = vec![Message::user("Hello")];
    assert!(dummy.stream_chat(&msgs, tx, cancel).await.is_ok());

    let chunk1 = rx.recv().await.expect("chunk 1");
    assert_eq!(chunk1.delta, "chunk_0");
    assert!(!chunk1.is_final);

    let chunk2 = rx.recv().await.expect("chunk 2");
    assert!(chunk2.is_final);
}

#[tokio::test]
async fn test_llm_port_cancellation() {
    let dummy = DummyLlm;
    let (tx, _rx) = mpsc::channel(1);
    let cancel = CancellationToken::new();
    cancel.cancel();

    let result = dummy.stream_chat(&[], tx, cancel).await;
    assert!(matches!(result, Err(LlmError::Cancelled)));
}

// -----------------------------------------------------------------------------
// 2. STT Port Contract Verification
// -----------------------------------------------------------------------------
struct DummyStt;

#[async_trait]
impl SttPort for DummyStt {
    async fn stream_transcribe(
        &self,
        mut audio_receiver: mpsc::Receiver<Vec<f32>>,
        event_sender: mpsc::Sender<SttEvent>,
        cancel: CancellationToken,
    ) -> Result<(), SttError> {
        tokio::select! {
            _ = cancel.cancelled() => Err(SttError::Cancelled),
            chunk = audio_receiver.recv() => {
                if let Some(samples) = chunk {
                    let event = SttEvent::final_result(format!("heard {} samples", samples.len()));
                    event_sender.send(event).await.map_err(|e| SttError::Channel(e.to_string()))?;
                }
                Ok(())
            }
        }
    }

    async fn transcribe_slice(
        &self,
        samples: &[f32],
        _sample_rate: u32,
    ) -> Result<SttEvent, SttError> {
        Ok(SttEvent::final_result(format!("slice: {} samples", samples.len())))
    }
}

#[tokio::test]
async fn test_stt_port_contract() {
    let dummy = DummyStt;

    // Test slice
    let slice_res = dummy.transcribe_slice(&[0.0, 0.5, -0.5], 16000).await.expect("transcribe slice");
    assert_eq!(slice_res.text, "slice: 3 samples");
    assert!(slice_res.is_final);

    // Test stream
    let (audio_tx, audio_rx) = mpsc::channel(2);
    let (event_tx, mut event_rx) = mpsc::channel(2);
    let cancel = CancellationToken::new();

    audio_tx.send(vec![0.1; 160]).await.unwrap();

    dummy.stream_transcribe(audio_rx, event_tx, cancel).await.expect("stream transcribe");
    let event = event_rx.recv().await.expect("stt event");
    assert_eq!(event.text, "heard 160 samples");
}

// -----------------------------------------------------------------------------
// 3. TTS Port Contract Verification
// -----------------------------------------------------------------------------
struct DummyTts;

#[async_trait]
impl TtsPort for DummyTts {
    async fn stream_synthesize(
        &self,
        text: &str,
        sender: mpsc::Sender<TtsStreamEvent>,
        cancel: CancellationToken,
    ) -> Result<(), TtsError> {
        if cancel.is_cancelled() {
            return Err(TtsError::Cancelled);
        }

        let audio = TtsAudioChunk {
            samples: vec![0.1, 0.2, 0.3],
            sample_rate: self.sample_rate(),
        };
        sender.send(TtsStreamEvent::Audio(audio)).await.map_err(|e| TtsError::Channel(e.to_string()))?;

        let viseme = TimedViseme::new(50, VisemeWeights::new(0.8, 0.1, 0.0, 0.0, 0.0));
        sender.send(TtsStreamEvent::Viseme(viseme)).await.map_err(|e| TtsError::Channel(e.to_string()))?;
        sender.send(TtsStreamEvent::Done).await.map_err(|e| TtsError::Channel(e.to_string()))?;

        let _ = text;
        Ok(())
    }

    async fn synthesize(&self, text: &str) -> Result<TtsOutput, TtsError> {
        let _ = text;
        Ok(TtsOutput {
            samples: vec![0.1, 0.2],
            sample_rate: self.sample_rate(),
            visemes: vec![TimedViseme::new(0, VisemeWeights::zero())],
        })
    }

    fn sample_rate(&self) -> u32 {
        24000
    }
}

#[tokio::test]
async fn test_tts_port_contract() {
    let dummy = DummyTts;
    assert_eq!(dummy.sample_rate(), 24000);

    let (tx, mut rx) = mpsc::channel(4);
    let cancel = CancellationToken::new();

    dummy.stream_synthesize("Hello world", tx, cancel).await.expect("tts stream");

    let event1 = rx.recv().await.expect("event 1");
    match event1 {
        TtsStreamEvent::Audio(chunk) => {
            assert_eq!(chunk.samples.len(), 3);
            assert_eq!(chunk.sample_rate, 24000);
        }
        _ => panic!("expected audio event"),
    }

    let event2 = rx.recv().await.expect("event 2");
    match event2 {
        TtsStreamEvent::Viseme(tv) => {
            assert_eq!(tv.timestamp_ms, 50);
            assert_eq!(tv.weights.aa, 0.8);
        }
        _ => panic!("expected viseme event"),
    }

    let event3 = rx.recv().await.expect("event 3");
    assert!(matches!(event3, TtsStreamEvent::Done));
}

// -----------------------------------------------------------------------------
// 4. Audio Playback and Capture Port Contracts
// -----------------------------------------------------------------------------
struct DummyPlayback {
    sample_clock: std::sync::atomic::AtomicU64,
    buffered: std::sync::atomic::AtomicUsize,
}

impl AudioPlaybackPort for DummyPlayback {
    fn enqueue_samples(&self, samples: &[f32]) -> Result<usize, AudioError> {
        self.buffered.fetch_add(samples.len(), std::sync::atomic::Ordering::SeqCst);
        Ok(samples.len())
    }

    fn flush(&self) -> Result<(), AudioError> {
        self.buffered.store(0, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn get_sample_clock(&self) -> u64 {
        self.sample_clock.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn sample_rate(&self) -> u32 {
        48000
    }

    fn buffered_samples(&self) -> usize {
        self.buffered.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn buffer_capacity(&self) -> usize {
        8192
    }
}

#[test]
fn test_audio_playback_port_contract() {
    let playback = DummyPlayback {
        sample_clock: std::sync::atomic::AtomicU64::new(48000),
        buffered: std::sync::atomic::AtomicUsize::new(0),
    };

    assert_eq!(playback.sample_rate(), 48000);
    assert_eq!(playback.get_sample_clock(), 48000);
    assert_eq!(playback.current_time_ms(), 1000);
    assert_eq!(playback.buffered_samples(), 0);
    assert!(!playback.is_playing());

    let enqueued = playback.enqueue_samples(&[0.1; 512]).expect("enqueue");
    assert_eq!(enqueued, 512);
    assert_eq!(playback.buffered_samples(), 512);
    assert!(playback.is_playing());

    playback.flush().expect("flush");
    assert_eq!(playback.buffered_samples(), 0);
    assert!(!playback.is_playing());
}

struct DummyCapture;

#[async_trait]
impl AudioCapturePort for DummyCapture {
    fn start_capture(&self) -> Result<(), AudioError> {
        Ok(())
    }

    fn stop_capture(&self) -> Result<(), AudioError> {
        Ok(())
    }

    fn is_capturing(&self) -> bool {
        true
    }

    fn sample_rate(&self) -> u32 {
        16000
    }

    async fn read_chunk(&self, _cancel: CancellationToken) -> Result<Vec<f32>, AudioError> {
        Ok(vec![0.0; 160])
    }

    async fn stream_capture(
        &self,
        sender: mpsc::Sender<Vec<f32>>,
        cancel: CancellationToken,
    ) -> Result<(), AudioError> {
        if cancel.is_cancelled() {
            return Err(AudioError::Cancelled);
        }
        sender.send(vec![0.0; 160]).await.map_err(|e| AudioError::Stream(e.to_string()))?;
        Ok(())
    }
}

#[tokio::test]
async fn test_audio_capture_port_contract() {
    let capture = DummyCapture;
    assert!(capture.is_capturing());
    assert_eq!(capture.sample_rate(), 16000);

    let cancel = CancellationToken::new();
    let chunk = capture.read_chunk(cancel.clone()).await.expect("read chunk");
    assert_eq!(chunk.len(), 160);

    let (tx, mut rx) = mpsc::channel(1);
    capture.stream_capture(tx, cancel).await.expect("stream capture");
    let streamed = rx.recv().await.expect("streamed chunk");
    assert_eq!(streamed.len(), 160);
}

// -----------------------------------------------------------------------------
// 5. Store Port Contract Verification
// -----------------------------------------------------------------------------
struct DummyStore {
    messages: tokio::sync::Mutex<Vec<(String, Message)>>,
    settings: tokio::sync::Mutex<HashMap<String, String>>,
    profile: tokio::sync::Mutex<Option<UserProfile>>,
    flashcards: tokio::sync::Mutex<Vec<Flashcard>>,
}

#[async_trait]
impl StorePort for DummyStore {
    async fn save_message(&self, session_id: &str, message: &Message) -> Result<(), StoreError> {
        let mut msgs = self.messages.lock().await;
        msgs.push((session_id.to_string(), message.clone()));
        Ok(())
    }

    async fn get_history(&self, session_id: &str, limit: usize) -> Result<Vec<Message>, StoreError> {
        let msgs = self.messages.lock().await;
        let filtered: Vec<Message> = msgs
            .iter()
            .filter(|(s, _)| s == session_id)
            .map(|(_, m)| m.clone())
            .take(limit)
            .collect();
        Ok(filtered)
    }

    async fn delete_session(&self, session_id: &str) -> Result<(), StoreError> {
        let mut msgs = self.messages.lock().await;
        msgs.retain(|(s, _)| s != session_id);
        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<SessionSummary>, StoreError> {
        let msgs = self.messages.lock().await;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for (sid, _) in msgs.iter() {
            *counts.entry(sid.clone()).or_insert(0) += 1;
        }
        let summaries = counts
            .into_iter()
            .map(|(id, count)| SessionSummary {
                id,
                title: None,
                message_count: count,
                created_at: 1000,
                updated_at: 2000,
            })
            .collect();
        Ok(summaries)
    }

    async fn get_user_profile(&self, _user_id: &str) -> Result<Option<UserProfile>, StoreError> {
        let p = self.profile.lock().await;
        Ok(p.clone())
    }

    async fn save_user_profile(&self, profile: &UserProfile) -> Result<(), StoreError> {
        let mut p = self.profile.lock().await;
        *p = Some(profile.clone());
        Ok(())
    }

    async fn save_flashcard(&self, card: &Flashcard) -> Result<(), StoreError> {
        let mut cards = self.flashcards.lock().await;
        if let Some(pos) = cards.iter().position(|c| c.id == card.id) {
            cards[pos] = card.clone();
        } else {
            cards.push(card.clone());
        }
        Ok(())
    }

    async fn get_flashcard(&self, id: &str) -> Result<Option<Flashcard>, StoreError> {
        let cards = self.flashcards.lock().await;
        Ok(cards.iter().find(|c| c.id == id).cloned())
    }

    async fn get_due_flashcards(&self, current_time: u64, limit: usize) -> Result<Vec<Flashcard>, StoreError> {
        let cards = self.flashcards.lock().await;
        let due: Vec<Flashcard> = cards
            .iter()
            .filter(|c| c.due <= current_time)
            .take(limit)
            .cloned()
            .collect();
        Ok(due)
    }

    async fn list_flashcards(&self, limit: usize) -> Result<Vec<Flashcard>, StoreError> {
        let cards = self.flashcards.lock().await;
        Ok(cards.iter().take(limit).cloned().collect())
    }

    async fn get_setting(&self, key: &str) -> Result<Option<String>, StoreError> {
        let map = self.settings.lock().await;
        Ok(map.get(key).cloned())
    }

    async fn set_setting(&self, key: &str, value: &str) -> Result<(), StoreError> {
        let mut map = self.settings.lock().await;
        map.insert(key.to_string(), value.to_string());
        Ok(())
    }
}

#[tokio::test]
async fn test_store_port_contract() {
    let store = DummyStore {
        messages: tokio::sync::Mutex::new(Vec::new()),
        settings: tokio::sync::Mutex::new(HashMap::new()),
        profile: tokio::sync::Mutex::new(None),
        flashcards: tokio::sync::Mutex::new(Vec::new()),
    };

    // Messages & Sessions
    let msg = Message::new(Role::User, "Hello Vira");
    store.save_message("session_1", &msg).await.expect("save message");
    let history = store.get_history("session_1", 10).await.expect("get history");
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].raw_content, "Hello Vira");

    let sessions = store.list_sessions().await.expect("list sessions");
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].message_count, 1);

    // Profile
    let mut prefs = HashMap::new();
    prefs.insert("theme".to_string(), "cosmic".to_string());
    let profile = UserProfile {
        user_id: "user_1".to_string(),
        display_name: "Learner".to_string(),
        native_language: "vi".to_string(),
        target_language: "en".to_string(),
        cefr_level: "B2".to_string(),
        preferences: prefs,
    };
    store.save_user_profile(&profile).await.expect("save profile");
    let loaded_profile = store.get_user_profile("user_1").await.expect("get profile");
    assert_eq!(loaded_profile, Some(profile));

    // Flashcard
    let card = Flashcard {
        id: "card_1".to_string(),
        front: "serendipity".to_string(),
        back: "sự tình cờ may mắn".to_string(),
        cefr_level: Some("C1".to_string()),
        state: 0,
        step: Some(1),
        stability: 2.5,
        difficulty: 4.0,
        due: 1500,
        last_review: None,
        reps: 0,
        lapses: 0,
    };
    store.save_flashcard(&card).await.expect("save flashcard");
    let due_cards = store.get_due_flashcards(1600, 10).await.expect("get due cards");
    assert_eq!(due_cards.len(), 1);
    assert_eq!(due_cards[0].front, "serendipity");

    // Settings
    store.set_setting("audio_gain", "1.2").await.expect("set setting");
    let val = store.get_setting("audio_gain").await.expect("get setting");
    assert_eq!(val.as_deref(), Some("1.2"));
}
