use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use vira_domain::message::Message;
use vira_infer::kokoro_tts::KokoroTtsAdapter;
use vira_infer::openai_compat::{
    MockLlmTransport, OpenAiCompatLlmAdapter, StreamingMode,
};
use vira_infer::whisper_stt::WhisperSttAdapter;
use vira_ports::llm::{LlmError, LlmPort};
use vira_ports::stt::{SttError, SttPort};
use vira_ports::tts::{TtsError, TtsPort, TtsStreamEvent};

#[test]
fn test_adapter_config_initialization() {
    let adapter = OpenAiCompatLlmAdapter::new(
        "http://127.0.0.1:8080/v1".to_string(),
        "default".to_string(),
        None,
    );
    assert_eq!(adapter.base_url(), "http://127.0.0.1:8080/v1");
    assert_eq!(adapter.model(), "default");
    assert_eq!(adapter.api_key(), None);
}

#[tokio::test]
async fn test_openai_compat_streaming_llm_port_with_tags() {
    let deltas = vec![
        "[emotion:happy]",
        "[gesture:wave]",
        " Xin chào bạn! ",
        "[gesture:nod]",
        "Tôi là Vira, ",
        "người bạn đồng hành của bạn.",
    ];

    let adapter = OpenAiCompatLlmAdapter::mock(
        "http://127.0.0.1:8080/v1".to_string(),
        "default".to_string(),
        &deltas,
    );

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let history = vec![Message::user("Xin chào!")];

    let res = adapter.stream_chat(&history, tx, cancel).await;
    assert!(res.is_ok(), "streaming chat should succeed");

    let mut chunks = Vec::new();
    while let Some(chunk) = rx.recv().await {
        chunks.push(chunk);
    }

    assert!(!chunks.is_empty(), "must receive chunks");

    // Check emotion and gesture tag extraction
    let emotions: Vec<_> = chunks.iter().filter_map(|c| c.emotion.as_deref()).collect();
    let gestures: Vec<_> = chunks.iter().filter_map(|c| c.gesture.as_deref()).collect();

    assert!(emotions.contains(&"happy"), "must extract 'happy' emotion");
    assert!(gestures.contains(&"wave"), "must extract 'wave' gesture");
    assert!(gestures.contains(&"nod"), "must extract 'nod' gesture");

    // Check speakable text is present and does not contain raw tag brackets
    let speakable_text: String = chunks
        .iter()
        .map(|c| c.delta.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(!speakable_text.contains("[emotion:"));
    assert!(!speakable_text.contains("[gesture:"));
    assert!(speakable_text.contains("Xin chào bạn!"));

    // Check final done chunk
    let last = chunks.last().expect("must have final chunk");
    assert!(last.is_final);
}

#[tokio::test]
async fn test_openai_compat_streaming_incremental_token_mode() {
    let deltas = vec![
        "[emotion:surprised] ",
        "Thật ",
        "bất ",
        "ngờ!",
    ];

    let adapter = OpenAiCompatLlmAdapter::mock(
        "http://127.0.0.1:8080/v1".to_string(),
        "default".to_string(),
        &deltas,
    )
    .with_streaming_mode(StreamingMode::IncrementalToken);

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    adapter
        .stream_chat(&[Message::user("Surprise me")], tx, cancel)
        .await
        .expect("stream success");

    let mut chunks = Vec::new();
    while let Some(chunk) = rx.recv().await {
        chunks.push(chunk);
    }

    assert!(chunks.iter().any(|c| c.emotion.as_deref() == Some("surprised")));
    assert!(chunks.iter().any(|c| c.is_final));
}

#[tokio::test]
async fn test_openai_compat_barge_in_cancellation_during_generation() {
    let deltas = vec!["Đây là một câu rất dài ", "và sẽ bị cắt ngang ", "khi người dùng nói..."];
    let transport = Arc::new(MockLlmTransport::from_deltas(&deltas).with_delay_ms(30));
    let adapter = OpenAiCompatLlmAdapter::new(
        "http://127.0.0.1:8080/v1".to_string(),
        "default".to_string(),
        None,
    )
    .with_transport(transport);

    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(15)).await;
        cancel_clone.cancel();
    });

    let res = adapter
        .stream_chat(&[Message::user("Hello")], tx, cancel)
        .await;

    assert_eq!(res, Err(LlmError::Cancelled));

    // Even if cancelled, receiver should not hang
    let mut received = 0;
    while (rx.recv().await).is_some() {
        received += 1;
    }
    assert!(received <= 2, "must abort quickly after cancellation");
}

#[tokio::test]
async fn test_openai_compat_transport_failure() {
    let transport = Arc::new(
        MockLlmTransport::new(vec![]).with_error(LlmError::Network("connection refused".into())),
    );
    let adapter = OpenAiCompatLlmAdapter::new(
        "http://127.0.0.1:8080/v1".to_string(),
        "default".to_string(),
        None,
    )
    .with_transport(transport);

    let (tx, _rx) = mpsc::channel(16);
    let cancel = CancellationToken::new();

    let res = adapter
        .stream_chat(&[Message::user("hi")], tx, cancel)
        .await;

    assert_eq!(
        res,
        Err(LlmError::Network("connection refused".to_string()))
    );
}

#[tokio::test]
async fn test_kokoro_tts_port_synthesis_and_60hz_visemes() {
    let adapter = KokoroTtsAdapter::new()
        .with_voice("vi_vn")
        .with_speed(1.0);

    assert_eq!(adapter.sample_rate(), 24000);

    let text = "Xin chào các bạn, tôi là người máy ảo.";
    let output = adapter.synthesize(text).await.expect("tts synthesize success");

    assert_eq!(output.sample_rate, 24000);
    assert!(!output.samples.is_empty(), "must produce audio samples");
    assert!(!output.visemes.is_empty(), "must produce viseme frames");
    assert!(output.duration_ms() > 0);

    // Verify all visemes have clamped weights [0.0, 1.0]
    for frame in &output.visemes {
        assert!(frame.weights.aa >= 0.0 && frame.weights.aa <= 1.0);
        assert!(frame.weights.ih >= 0.0 && frame.weights.ih <= 1.0);
        assert!(frame.weights.ou >= 0.0 && frame.weights.ou <= 1.0);
        assert!(frame.weights.ee >= 0.0 && frame.weights.ee <= 1.0);
        assert!(frame.weights.oh >= 0.0 && frame.weights.oh <= 1.0);
    }

    // Verify 60Hz tick intervals (~16ms between frames)
    if output.visemes.len() >= 2 {
        let diff = output.visemes[1].timestamp_ms - output.visemes[0].timestamp_ms;
        assert_eq!(diff, 16, "viseme timeline must be 60Hz (~16ms ticks)");
    }
}

#[tokio::test]
async fn test_kokoro_tts_streaming_synthesis_flow() {
    let adapter = KokoroTtsAdapter::new();
    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let synth_handle = tokio::spawn(async move {
        adapter
            .stream_synthesize("Tôi đang nói một câu chào.", tx, cancel)
            .await
    });

    let mut audio_chunks = 0;
    let mut viseme_batches = 0;
    let mut got_done = false;

    while let Some(event) = rx.recv().await {
        match event {
            TtsStreamEvent::Audio(chunk) => {
                assert!(!chunk.is_empty());
                assert_eq!(chunk.sample_rate, 24000);
                audio_chunks += 1;
            }
            TtsStreamEvent::Visemes(visemes) => {
                assert!(!visemes.is_empty());
                viseme_batches += 1;
            }
            TtsStreamEvent::Done => {
                got_done = true;
                break;
            }
            _ => {}
        }
    }

    let res = synth_handle.await.expect("join task");
    assert!(res.is_ok(), "synthesis must succeed: {res:?}");
    assert!(audio_chunks > 0, "must stream audio chunks");
    assert!(viseme_batches > 0, "must stream viseme frames");
    assert!(got_done, "must emit Done at conclusion");
}

#[tokio::test]
async fn test_kokoro_tts_barge_in_cancellation() {
    let adapter = KokoroTtsAdapter::new();
    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    cancel.cancel(); // Immediate cancel
    let res = adapter
        .stream_synthesize("Một câu nói rất dài và bị cắt đứt.", tx, cancel)
        .await;

    assert_eq!(res, Err(TtsError::Cancelled));
}

#[tokio::test]
async fn test_kokoro_tts_empty_text() {
    let adapter = KokoroTtsAdapter::new();
    let output = adapter.synthesize("   ").await.expect("empty text ok");
    assert!(output.is_empty());
    assert_eq!(output.duration_ms(), 0);
}

#[tokio::test]
async fn test_whisper_stt_port_contract() {
    let adapter = WhisperSttAdapter::new()
        .with_language("vi")
        .with_mock_responses(vec!["Chào trợ lý".to_string()]);

    assert_eq!(adapter.sample_rate(), 16000);
    assert_eq!(adapter.language(), "vi");

    // Generate test audio (1 second, non-silent)
    let samples: Vec<f32> = (0..16000)
        .map(|i| (i as f32 * 0.05).sin() * 0.25)
        .collect();

    let res = adapter
        .transcribe_slice(&samples, 16000)
        .await
        .expect("transcription success");

    assert_eq!(res.text, "Chào trợ lý");
    assert!(res.is_final);
    assert_eq!(res.language.as_deref(), Some("vi"));
    assert_eq!(res.duration_ms, Some(1000));
}

#[tokio::test]
async fn test_whisper_stt_streaming_transcription() {
    let adapter = WhisperSttAdapter::new()
        .with_mock_responses(vec!["Bật đèn phòng khách".to_string()]);

    let (audio_tx, audio_rx) = mpsc::channel(16);
    let (event_tx, mut event_rx) = mpsc::channel(16);
    let cancel = CancellationToken::new();

    let speech_chunk: Vec<f32> = (0..3200)
        .map(|i| (i as f32 * 0.05).sin() * 0.2)
        .collect();

    tokio::spawn(async move {
        audio_tx.send(speech_chunk).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        drop(audio_tx);
    });

    let res = adapter
        .stream_transcribe(audio_rx, event_tx, cancel)
        .await;
    assert!(res.is_ok());

    let mut events = Vec::new();
    while let Some(evt) = event_rx.recv().await {
        events.push(evt);
    }

    assert!(!events.is_empty());
    let final_evt = events.iter().find(|e| e.is_final).expect("has final event");
    assert_eq!(final_evt.text, "Bật đèn phòng khách");
}

#[tokio::test]
async fn test_whisper_stt_barge_in_cancellation() {
    let adapter = WhisperSttAdapter::new();
    let (_audio_tx, audio_rx) = mpsc::channel(16);
    let (event_tx, _event_rx) = mpsc::channel(16);
    let cancel = CancellationToken::new();

    cancel.cancel();
    let res = adapter
        .stream_transcribe(audio_rx, event_tx, cancel)
        .await;

    assert_eq!(res, Err(SttError::Cancelled));
}

#[tokio::test]
async fn test_whisper_stt_silence_detection() {
    let adapter = WhisperSttAdapter::new();
    let silence = vec![0.0f32; 8000]; // 0.5s silence

    let res = adapter
        .transcribe_slice(&silence, 16000)
        .await
        .expect("transcribe success");

    assert_eq!(res.text, "");
    assert_eq!(res.duration_ms, Some(500));
}

#[tokio::test]
async fn test_whisper_stt_invalid_sample_rate() {
    let adapter = WhisperSttAdapter::new();
    let res = adapter.transcribe_slice(&[0.1, -0.1], 0).await;
    assert!(matches!(res, Err(SttError::InvalidAudio(_))));
}
