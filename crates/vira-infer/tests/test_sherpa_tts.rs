use std::path::PathBuf;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use vira_infer::{
    OfflineModelPaths, SherpaTtsAdapter, SHERPA_TTS_DEFAULT_SAMPLE_RATE,
};
use vira_ports::tts::{TtsError, TtsPort, TtsStreamEvent};

#[tokio::test]
async fn test_sherpa_tts_adapter_initialization_and_defaults() {
    let adapter = SherpaTtsAdapter::new()
        .with_speed(1.2)
        .with_speaker_id(1)
        .with_chunk_ms(40);

    assert_eq!(adapter.sample_rate(), SHERPA_TTS_DEFAULT_SAMPLE_RATE);
    assert_eq!(adapter.config().speed, 1.2);
    assert_eq!(adapter.config().speaker_id, 1);
    assert_eq!(adapter.config().chunk_ms, 40);
    let fallback = SherpaTtsAdapter::virtual_fallback();
    assert!(!fallback.is_neural_active());
}

#[tokio::test]
async fn test_sherpa_tts_one_shot_synthesize_vietnamese() {
    let adapter = SherpaTtsAdapter::virtual_fallback();
    let text = "Xin chào bạn! Tôi là trợ lý ảo Vira.";
    let output = adapter.synthesize(text).await.expect("synthesis success");

    assert_eq!(output.sample_rate, SHERPA_TTS_DEFAULT_SAMPLE_RATE);
    assert!(!output.samples.is_empty(), "audio samples must be generated");
    assert!(!output.visemes.is_empty(), "visemes must be generated");
    assert!(output.duration_ms() > 0);

    // Verify 60Hz blendshape weights are strictly clamped to [0.0, 1.0]
    for v in &output.visemes {
        assert!(
            v.weights.aa >= 0.0 && v.weights.aa <= 1.0,
            "aa blendshape clamped"
        );
        assert!(
            v.weights.ih >= 0.0 && v.weights.ih <= 1.0,
            "ih blendshape clamped"
        );
        assert!(
            v.weights.ou >= 0.0 && v.weights.ou <= 1.0,
            "ou blendshape clamped"
        );
        assert!(
            v.weights.ee >= 0.0 && v.weights.ee <= 1.0,
            "ee blendshape clamped"
        );
        assert!(
            v.weights.oh >= 0.0 && v.weights.oh <= 1.0,
            "oh blendshape clamped"
        );
    }
}

#[tokio::test]
async fn test_sherpa_tts_one_shot_synthesize_english() {
    let adapter = SherpaTtsAdapter::virtual_fallback();
    let text = "Good morning! Welcome to the Vira companion system.";
    let output = adapter.synthesize(text).await.expect("synthesis success");

    assert_eq!(output.sample_rate, SHERPA_TTS_DEFAULT_SAMPLE_RATE);
    assert!(!output.samples.is_empty());
    assert!(!output.visemes.is_empty());
}

#[tokio::test]
async fn test_sherpa_tts_stream_synthesize_flow() {
    let adapter = SherpaTtsAdapter::new().with_chunk_ms(30);
    let (tx, mut rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let consumer_handle = tokio::spawn(async move {
        let mut audio_chunks = 0;
        let mut viseme_batches = 0;
        let mut got_done = false;

        while let Some(event) = rx.recv().await {
            match event {
                TtsStreamEvent::Audio(chunk) => {
                    assert!(!chunk.is_empty());
                    assert!(
                        chunk.sample_rate == 22050 || chunk.sample_rate == SHERPA_TTS_DEFAULT_SAMPLE_RATE,
                        "expected sample rate 22050 or 24000, got {}",
                        chunk.sample_rate
                    );
                    audio_chunks += 1;
                }
                TtsStreamEvent::Visemes(batch) => {
                    assert!(!batch.is_empty());
                    for v in &batch {
                        assert!(v.weights.aa >= 0.0 && v.weights.aa <= 1.0);
                        assert!(v.weights.ih >= 0.0 && v.weights.ih <= 1.0);
                        assert!(v.weights.ou >= 0.0 && v.weights.ou <= 1.0);
                        assert!(v.weights.ee >= 0.0 && v.weights.ee <= 1.0);
                        assert!(v.weights.oh >= 0.0 && v.weights.oh <= 1.0);
                    }
                    viseme_batches += 1;
                }
                TtsStreamEvent::Done => {
                    got_done = true;
                }
                _ => {}
            }
        }

        (audio_chunks, viseme_batches, got_done)
    });

    adapter
        .stream_synthesize("Hôm nay thời tiết rất đẹp.", tx, cancel)
        .await
        .expect("stream synthesis success");

    let (audio_chunks, viseme_batches, got_done) = consumer_handle.await.expect("consumer task");
    assert!(audio_chunks > 0, "must stream audio chunks");
    assert!(viseme_batches > 0, "must stream viseme batches");
    assert!(got_done, "must emit Done event");
}

#[tokio::test]
async fn test_sherpa_tts_barge_in_instant_cancellation() {
    let adapter = SherpaTtsAdapter::new();
    let (tx, _rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    cancel.cancel(); // Pre-cancel to simulate instant barge-in

    let res = adapter
        .stream_synthesize("Một đoạn văn bản rất dài cần tổng hợp giọng nói.", tx, cancel)
        .await;

    assert_eq!(res, Err(TtsError::Cancelled));
}

#[tokio::test]
async fn test_sherpa_tts_empty_and_whitespace_input() {
    let adapter = SherpaTtsAdapter::new();

    let output = adapter.synthesize("   ").await.expect("empty synthesize");
    assert!(output.samples.is_empty());

    let (tx, mut rx) = mpsc::channel(16);
    let cancel = CancellationToken::new();
    adapter
        .stream_synthesize("", tx, cancel)
        .await
        .expect("empty stream synthesize");

    let event = rx.recv().await.expect("receive done");
    assert_eq!(event, TtsStreamEvent::Done);
}

#[tokio::test]
async fn test_sherpa_tts_with_missing_model_paths_falls_back_gracefully() {
    let paths = OfflineModelPaths::new(PathBuf::from("/nonexistent/vira/models"));
    let adapter = SherpaTtsAdapter::with_model_paths(paths);

    assert!(!adapter.is_neural_active());
    let output = adapter.synthesize("Fallback test").await.expect("synthesize success");
    assert!(!output.samples.is_empty());
}

#[tokio::test]
async fn test_sherpa_tts_neural_vietnamese_synthesis_no_beeps() {
    let adapter = SherpaTtsAdapter::new();
    if !adapter.is_piper_active() {
        return; // Skip if models not present on disk
    }

    let text = "Xin chào bạn, tôi là trợ lý ảo Vira.";
    let output = adapter.synthesize(text).await.expect("neural vietnamese synthesis");

    assert_eq!(output.sample_rate, 22050, "Piper Vietnamese sample rate is 22050Hz");
    assert!(!output.samples.is_empty(), "audio samples must be generated");
    assert!(!output.visemes.is_empty(), "visemes must be generated");
    assert!(output.duration_ms() > 500, "audio duration must be realistic");

    // Verify it is real speech (active amplitude)
    let max_amp = output.samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
    assert!(max_amp > 0.05, "audio amplitude must reflect active speech");
}

#[tokio::test]
async fn test_sherpa_tts_neural_kokoro_english_synthesis() {
    let adapter = SherpaTtsAdapter::new();
    if !adapter.is_kokoro_active() {
        return; // Skip if models not present on disk
    }

    let text = "Good morning! Welcome to the Vira companion system.";
    let output = adapter.synthesize(text).await.expect("neural english synthesis");

    assert_eq!(output.sample_rate, 24000, "Kokoro English sample rate is 24000Hz");
    assert!(!output.samples.is_empty(), "audio samples must be generated");
    assert!(!output.visemes.is_empty(), "visemes must be generated");
    assert!(output.duration_ms() > 500, "audio duration must be realistic");

    let max_amp = output.samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
    assert!(max_amp > 0.05, "audio amplitude must reflect active speech");
}
