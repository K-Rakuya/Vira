use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use vira_domain::message::{Message, Role};
use vira_domain::state::CharacterEmotion;
use vira_domain::viseme::VisemeWeights;

use vira_ports::audio::{AudioCapturePort, AudioError, AudioPlaybackPort};
use vira_ports::llm::{
    extract_emotion_tag, extract_gesture_tag, strip_tags, LlmError, LlmPort, LlmStreamChunk,
};
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
            if cancel.is_cancelled() {
                return Err(LlmError::Cancelled);
            }
            let raw_delta = format!("[emotion:happy][gesture:wave] chunk_{i}");
            let chunk = LlmStreamChunk::from_delta_extracting_tags(&raw_delta);
            sender
                .send(chunk)
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
    assert_eq!(chunk1.emotion.as_deref(), Some("happy"));
    assert_eq!(chunk1.gesture.as_deref(), Some("wave"));
    assert_eq!(chunk1.emotion_as_domain(), Some(CharacterEmotion::Happy));

    let chunk2 = rx.recv().await.expect("chunk 2");
    assert!(chunk2.is_final);
    assert!(chunk2.is_empty());
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

#[test]
fn test_llm_tag_utilities() {
    let text = "[emotion:surprised] [gesture:tilt] Thật kỳ diệu!";
    assert_eq!(extract_emotion_tag(text), Some("surprised".to_string()));
    assert_eq!(extract_gesture_tag(text), Some("tilt".to_string()));
    assert_eq!(strip_tags(text), "Thật kỳ diệu!");
}

// -----------------------------------------------------------------------------
// 2. STT Port Contract Verification
// -----------------------------------------------------------------------------
struct DummyStt {
    expected_rate: u32,
}

#[async_trait]
impl SttPort for DummyStt {
    async fn stream_transcribe(
        &self,
        mut audio_receiver: mpsc::Receiver<Vec<f32>>,
        event_sender: mpsc::Sender<SttEvent>,
        cancel: CancellationToken,
    ) -> Result<(), SttError> {
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return Err(SttError::Cancelled),
                chunk = audio_receiver.recv() => {
                    match chunk {
                        Some(samples) => {
                            let event = SttEvent::final_result(format!("heard {} samples", samples.len()))
                                .with_confidence(0.95)
                                .with_language("vi");
                            event_sender.send(event).await.map_err(|e| SttError::Channel(e.to_string()))?;
                        }
                        None => break,
                    }
                }
            }
        }
        Ok(())
    }

    async fn transcribe_slice(
        &self,
        samples: &[f32],
        _sample_rate: u32,
    ) -> Result<SttEvent, SttError> {
        if samples.is_empty() {
            return Err(SttError::InvalidAudio("empty audio samples".to_string()));
        }
        Ok(SttEvent::final_result(format!(
            "slice: {} samples",
            samples.len()
        )))
    }

    fn sample_rate(&self) -> u32 {
        self.expected_rate
    }
}

#[tokio::test]
async fn test_stt_port_contract() {
    let dummy = DummyStt {
        expected_rate: 16000,
    };
    assert_eq!(dummy.sample_rate(), 16000);

    // Test slice
    let slice_res = dummy
        .transcribe_slice(&[0.0, 0.5, -0.5], 16000)
        .await
        .expect("transcribe slice");
    assert_eq!(slice_res.text, "slice: 3 samples");
    assert!(slice_res.is_final);

    let empty_err = dummy.transcribe_slice(&[], 16000).await;
    assert!(matches!(empty_err, Err(SttError::InvalidAudio(_))));

    // Test stream
    let (audio_tx, audio_rx) = mpsc::channel(2);
    let (event_tx, mut event_rx) = mpsc::channel(2);
    let cancel = CancellationToken::new();

    audio_tx.send(vec![0.1; 160]).await.unwrap();
    drop(audio_tx);

    dummy
        .stream_transcribe(audio_rx, event_tx, cancel)
        .await
        .expect("stream transcribe");
    let event = event_rx.recv().await.expect("stt event");
    assert_eq!(event.text, "heard 160 samples");
    assert_eq!(event.confidence, Some(0.95));
    assert_eq!(event.language.as_deref(), Some("vi"));
}

#[tokio::test]
async fn test_stt_port_cancellation() {
    let dummy = DummyStt {
        expected_rate: 16000,
    };
    let (_audio_tx, audio_rx) = mpsc::channel(2);
    let (event_tx, _event_rx) = mpsc::channel(2);
    let cancel = CancellationToken::new();
    cancel.cancel();

    let res = dummy.stream_transcribe(audio_rx, event_tx, cancel).await;
    assert!(matches!(res, Err(SttError::Cancelled)));
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

        let audio = TtsAudioChunk::new(vec![0.1, 0.2, 0.3], self.sample_rate());
        sender
            .send(TtsStreamEvent::Audio(audio))
            .await
            .map_err(|e| TtsError::Channel(e.to_string()))?;

        let viseme = TimedViseme::new(50, VisemeWeights::new(0.8, 0.1, 0.0, 0.0, 0.0));
        sender
            .send(TtsStreamEvent::Viseme(viseme))
            .await
            .map_err(|e| TtsError::Channel(e.to_string()))?;
        sender
            .send(TtsStreamEvent::Done)
            .await
            .map_err(|e| TtsError::Channel(e.to_string()))?;

        let _ = text;
        Ok(())
    }

    async fn synthesize(&self, text: &str) -> Result<TtsOutput, TtsError> {
        let _ = text;
        Ok(TtsOutput::new(
            vec![0.1, 0.2, 0.3, 0.4],
            self.sample_rate(),
            vec![TimedViseme::new(0, VisemeWeights::zero())],
        ))
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

    dummy
        .stream_synthesize("Hello world", tx, cancel)
        .await
        .expect("tts stream");

    let event1 = rx.recv().await.expect("event 1");
    match event1 {
        TtsStreamEvent::Audio(chunk) => {
            assert_eq!(chunk.len(), 3);
            assert_eq!(chunk.sample_rate, 24000);
            assert!(!chunk.is_empty());
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

    // One-shot synthesis test
    let output = dummy.synthesize("test one shot").await.expect("synthesize");
    assert_eq!(output.len(), 4);
    assert_eq!(output.sample_rate, 24000);
    assert_eq!(output.visemes.len(), 1);
}

#[tokio::test]
async fn test_tts_port_cancellation() {
    let dummy = DummyTts;
    let (tx, _rx) = mpsc::channel(1);
    let cancel = CancellationToken::new();
    cancel.cancel();

    let res = dummy.stream_synthesize("cancel me", tx, cancel).await;
    assert!(matches!(res, Err(TtsError::Cancelled)));
}

// -----------------------------------------------------------------------------
// 4. Audio Playback and Capture Port Contracts
// -----------------------------------------------------------------------------
struct DummyPlayback {
    sample_clock: AtomicU64,
    buffered: AtomicUsize,
    capacity: usize,
    sample_rate: u32,
}

impl AudioPlaybackPort for DummyPlayback {
    fn enqueue_samples(&self, samples: &[f32]) -> Result<usize, AudioError> {
        let avail = self.available_space();
        let to_write = samples.len().min(avail);
        self.buffered.fetch_add(to_write, Ordering::SeqCst);
        Ok(to_write)
    }

    fn flush(&self) -> Result<(), AudioError> {
        self.buffered.store(0, Ordering::SeqCst);
        Ok(())
    }

    fn get_sample_clock(&self) -> u64 {
        self.sample_clock.load(Ordering::SeqCst)
    }

    fn reset_sample_clock(&self) -> Result<(), AudioError> {
        self.sample_clock.store(0, Ordering::SeqCst);
        Ok(())
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn buffered_samples(&self) -> usize {
        self.buffered.load(Ordering::SeqCst)
    }

    fn buffer_capacity(&self) -> usize {
        self.capacity
    }
}

#[test]
fn test_audio_playback_port_contract() {
    let playback = DummyPlayback {
        sample_clock: AtomicU64::new(48000),
        buffered: AtomicUsize::new(0),
        capacity: 8192,
        sample_rate: 48000,
    };

    assert_eq!(playback.sample_rate(), 48000);
    assert_eq!(playback.get_sample_clock(), 48000);
    assert_eq!(playback.current_sample(), 48000);
    assert_eq!(playback.current_time_ms(), 1000);
    assert_eq!(playback.buffered_samples(), 0);
    assert_eq!(playback.available_space(), 8192);
    assert!(!playback.is_playing());
    assert!(playback.is_empty());

    let enqueued = playback.enqueue_samples(&[0.1; 512]).expect("enqueue");
    assert_eq!(enqueued, 512);
    assert_eq!(playback.buffered_samples(), 512);
    assert_eq!(playback.available_space(), 8192 - 512);
    assert!(playback.is_playing());
    assert!(!playback.is_empty());

    // Test enqueue_samples_exact success
    assert!(playback.enqueue_samples_exact(&[0.2; 512]).is_ok());
    assert_eq!(playback.buffered_samples(), 1024);

    // Test enqueue_samples_exact failure when exceeding capacity
    let overflow_res = playback.enqueue_samples_exact(&[0.3; 10000]);
    assert!(matches!(
        overflow_res,
        Err(AudioError::BufferOverflow { .. })
    ));

    // Instant flush (<10ms)
    playback.flush().expect("flush");
    assert_eq!(playback.buffered_samples(), 0);
    assert!(!playback.is_playing());
    assert!(playback.is_empty());

    // Reset sample clock
    playback.reset_sample_clock().expect("reset clock");
    assert_eq!(playback.get_sample_clock(), 0);
    assert_eq!(playback.current_time_ms(), 0);
}

struct DummyCapture {
    capturing: AtomicBool,
    sample_rate: u32,
}

#[async_trait]
impl AudioCapturePort for DummyCapture {
    fn start_capture(&self) -> Result<(), AudioError> {
        self.capturing.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn stop_capture(&self) -> Result<(), AudioError> {
        self.capturing.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn is_capturing(&self) -> bool {
        self.capturing.load(Ordering::SeqCst)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn read_chunk(&self, cancel: CancellationToken) -> Result<Vec<f32>, AudioError> {
        if cancel.is_cancelled() {
            return Err(AudioError::Cancelled);
        }
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
        sender
            .send(vec![0.0; 160])
            .await
            .map_err(|e| AudioError::Stream(e.to_string()))?;
        Ok(())
    }
}

#[tokio::test]
async fn test_audio_capture_port_contract() {
    let capture = DummyCapture {
        capturing: AtomicBool::new(false),
        sample_rate: 16000,
    };
    assert!(!capture.is_capturing());
    assert_eq!(capture.sample_rate(), 16000);

    capture.start_capture().expect("start capture");
    assert!(capture.is_capturing());

    capture.stop_capture().expect("stop capture");
    assert!(!capture.is_capturing());

    let cancel = CancellationToken::new();
    let chunk = capture.read_chunk(cancel.clone()).await.expect("read chunk");
    assert_eq!(chunk.len(), 160);

    let (tx, mut rx) = mpsc::channel(1);
    capture
        .stream_capture(tx, cancel.clone())
        .await
        .expect("stream capture");
    let streamed = rx.recv().await.expect("streamed chunk");
    assert_eq!(streamed.len(), 160);

    cancel.cancel();
    assert!(matches!(
        capture.read_chunk(cancel.clone()).await,
        Err(AudioError::Cancelled)
    ));
    let (tx2, _rx2) = mpsc::channel(1);
    assert!(matches!(
        capture.stream_capture(tx2, cancel).await,
        Err(AudioError::Cancelled)
    ));
}

// -----------------------------------------------------------------------------
// 5. Store Port Contract Verification
// -----------------------------------------------------------------------------
#[derive(Default)]
struct DummyStore {
    messages: tokio::sync::Mutex<Vec<(String, Message)>>,
    sessions: tokio::sync::Mutex<HashMap<String, SessionSummary>>,
    settings: tokio::sync::Mutex<HashMap<String, String>>,
    profile: tokio::sync::Mutex<Option<UserProfile>>,
    flashcards: tokio::sync::Mutex<Vec<Flashcard>>,
}

#[async_trait]
impl StorePort for DummyStore {
    async fn save_message(&self, session_id: &str, message: &Message) -> Result<(), StoreError> {
        let mut msgs = self.messages.lock().await;
        msgs.push((session_id.to_string(), message.clone()));

        let mut sessions = self.sessions.lock().await;
        let summary = sessions
            .entry(session_id.to_string())
            .or_insert_with(|| SessionSummary::new(session_id));
        summary.message_count += 1;
        summary.updated_at = message.timestamp;
        Ok(())
    }

    async fn get_history(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<Message>, StoreError> {
        let msgs = self.messages.lock().await;
        let mut session_msgs: Vec<Message> = msgs
            .iter()
            .filter(|(s, _)| s == session_id)
            .map(|(_, m)| m.clone())
            .collect();
        // Return latest messages up to limit in chronological order
        if session_msgs.len() > limit {
            session_msgs = session_msgs.split_off(session_msgs.len() - limit);
        }
        Ok(session_msgs)
    }

    async fn delete_session(&self, session_id: &str) -> Result<(), StoreError> {
        let mut msgs = self.messages.lock().await;
        msgs.retain(|(s, _)| s != session_id);
        let mut sessions = self.sessions.lock().await;
        sessions.remove(session_id);
        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<SessionSummary>, StoreError> {
        let sessions = self.sessions.lock().await;
        Ok(sessions.values().cloned().collect())
    }

    async fn update_session_title(&self, session_id: &str, title: &str) -> Result<(), StoreError> {
        let mut sessions = self.sessions.lock().await;
        if let Some(s) = sessions.get_mut(session_id) {
            s.title = Some(title.to_string());
            Ok(())
        } else {
            Err(StoreError::NotFound(format!("session {session_id}")))
        }
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

    async fn delete_flashcard(&self, id: &str) -> Result<(), StoreError> {
        let mut cards = self.flashcards.lock().await;
        let initial_len = cards.len();
        cards.retain(|c| c.id != id);
        if cards.len() < initial_len {
            Ok(())
        } else {
            Err(StoreError::NotFound(format!("flashcard {id}")))
        }
    }

    async fn count_flashcards(&self) -> Result<usize, StoreError> {
        let cards = self.flashcards.lock().await;
        Ok(cards.len())
    }

    async fn get_due_flashcards(
        &self,
        current_time: u64,
        limit: usize,
    ) -> Result<Vec<Flashcard>, StoreError> {
        let cards = self.flashcards.lock().await;
        let due: Vec<Flashcard> = cards
            .iter()
            .filter(|c| c.is_due(current_time))
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

    async fn delete_setting(&self, key: &str) -> Result<(), StoreError> {
        let mut map = self.settings.lock().await;
        map.remove(key);
        Ok(())
    }
}

#[tokio::test]
async fn test_store_port_contract() {
    let store = Arc::new(DummyStore::default());

    // Messages & Sessions
    let msg1 = Message::new(Role::User, "Hello Vira");
    let msg2 = Message::new(Role::Assistant, "Xin chào bạn!");
    store
        .save_message("session_1", &msg1)
        .await
        .expect("save message 1");
    store
        .save_message("session_1", &msg2)
        .await
        .expect("save message 2");

    let history = store
        .get_history("session_1", 10)
        .await
        .expect("get history");
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].raw_content, "Hello Vira");
    assert_eq!(history[1].raw_content, "Xin chào bạn!");

    // History limit test
    let limited = store
        .get_history("session_1", 1)
        .await
        .expect("get history limited");
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].raw_content, "Xin chào bạn!");

    // Session title update & listing
    store
        .update_session_title("session_1", "Vietnamese Basics")
        .await
        .expect("update session title");
    let sessions = store.list_sessions().await.expect("list sessions");
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].message_count, 2);
    assert_eq!(
        sessions[0].title.as_deref(),
        Some("Vietnamese Basics")
    );

    // Profile operations
    let profile = UserProfile::new("user_1", "Learner", "vi", "en", "B2")
        .with_preference("theme", "cosmic");
    store
        .save_user_profile(&profile)
        .await
        .expect("save profile");
    let loaded_profile = store
        .get_user_profile("user_1")
        .await
        .expect("get profile");
    assert_eq!(loaded_profile, Some(profile));

    // Flashcard operations: save, get, list, count, due, delete
    let card1 = Flashcard::new("card_1", "serendipity", "sự tình cờ may mắn")
        .with_cefr_level("C1")
        .with_due(1500);
    let card2 = Flashcard::new("card_2", "ephemeral", "phù du").with_due(3000);

    store.save_flashcard(&card1).await.expect("save card 1");
    store.save_flashcard(&card2).await.expect("save card 2");

    assert_eq!(store.count_flashcards().await.expect("count"), 2);

    let fetched = store
        .get_flashcard("card_1")
        .await
        .expect("get flashcard");
    assert_eq!(fetched.as_ref().map(|c| c.front.as_str()), Some("serendipity"));

    let due_cards = store
        .get_due_flashcards(2000, 10)
        .await
        .expect("get due cards");
    assert_eq!(due_cards.len(), 1);
    assert_eq!(due_cards[0].id, "card_1");

    let all_cards = store.list_flashcards(10).await.expect("list flashcards");
    assert_eq!(all_cards.len(), 2);

    store
        .delete_flashcard("card_1")
        .await
        .expect("delete flashcard");
    assert_eq!(store.count_flashcards().await.expect("count"), 1);
    assert_eq!(
        store.get_flashcard("card_1").await.expect("get after delete"),
        None
    );

    // Settings operations: set, get, delete
    store
        .set_setting("audio_gain", "1.2")
        .await
        .expect("set setting");
    let val = store.get_setting("audio_gain").await.expect("get setting");
    assert_eq!(val.as_deref(), Some("1.2"));

    store
        .delete_setting("audio_gain")
        .await
        .expect("delete setting");
    assert_eq!(
        store.get_setting("audio_gain").await.expect("get after delete"),
        None
    );

    // Delete session
    store
        .delete_session("session_1")
        .await
        .expect("delete session");
    assert_eq!(
        store.get_history("session_1", 10).await.expect("get deleted"),
        vec![]
    );
    assert_eq!(store.list_sessions().await.expect("list after delete").len(), 0);
}
