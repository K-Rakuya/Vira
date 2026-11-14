use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use vira_infer::{
    OfflineModelPaths, SherpaSttAdapter,
};
use vira_ports::stt::{SttError, SttPort};

#[tokio::test]
async fn test_sherpa_stt_adapter_initialization_and_defaults() {
    let adapter = SherpaSttAdapter::new()
        .with_language("vi")
        .with_sample_rate(16000)
        .with_vad_threshold(0.01);

    assert_eq!(adapter.language(), "vi");
    assert_eq!(adapter.sample_rate(), 16000);
    assert_eq!(adapter.config().vad_energy_threshold, 0.01);
    let fallback = SherpaSttAdapter::virtual_fallback();
    assert!(!fallback.is_neural_active());
}

#[tokio::test]
async fn test_sherpa_stt_transcribe_silence_returns_empty() {
    let adapter = SherpaSttAdapter::new();
    let silent_samples = vec![0.0f32; 16000]; // 1 second of silence
    let res = adapter
        .transcribe_slice(&silent_samples, 16000)
        .await
        .expect("transcribe success");

    assert_eq!(res.text, "");
    assert_eq!(res.duration_ms, Some(1000));
    assert!(res.is_final);
}

#[tokio::test]
async fn test_sherpa_stt_transcribe_slice_mock_response() {
    let adapter =
        SherpaSttAdapter::new().with_mock_responses(vec!["Chào Vira nhé".to_string()]);

    let speech_samples: Vec<f32> = (0..16000)
        .map(|i| (i as f32 * 0.1).sin() * 0.2)
        .collect();

    let res = adapter
        .transcribe_slice(&speech_samples, 16000)
        .await
        .expect("transcribe success");

    assert_eq!(res.text, "Chào Vira nhé");
    assert_eq!(res.confidence, Some(0.96));
    assert_eq!(res.language.as_deref(), Some("vi"));
    assert_eq!(res.duration_ms, Some(1000));
}

#[tokio::test]
async fn test_sherpa_stt_stream_transcribe_cancellation() {
    let adapter = SherpaSttAdapter::new();
    let (_audio_tx, audio_rx) = mpsc::channel(16);
    let (event_tx, _event_rx) = mpsc::channel(16);
    let cancel = CancellationToken::new();

    cancel.cancel(); // Pre-cancel
    let res = adapter
        .stream_transcribe(audio_rx, event_tx, cancel)
        .await;

    assert_eq!(res, Err(SttError::Cancelled));
}

#[tokio::test]
async fn test_sherpa_stt_stream_transcribe_flow() {
    let adapter = SherpaSttAdapter::new()
        .with_mock_responses(vec!["Hôm nay trời có mưa không?".to_string()]);

    let (audio_tx, audio_rx) = mpsc::channel(16);
    let (event_tx, mut event_rx) = mpsc::channel(16);
    let cancel = CancellationToken::new();

    let speech_chunk: Vec<f32> = (0..3200)
        .map(|i| (i as f32 * 0.1).sin() * 0.3)
        .collect();

    tokio::spawn(async move {
        audio_tx.send(speech_chunk).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        drop(audio_tx); // close stream (EOF)
    });

    adapter
        .stream_transcribe(audio_rx, event_tx, cancel)
        .await
        .expect("stream transcribe success");

    let mut events = Vec::new();
    while let Some(evt) = event_rx.recv().await {
        events.push(evt);
    }

    assert!(!events.is_empty(), "must receive at least final event");
    let final_evt = events.iter().find(|e| e.is_final).expect("has final event");
    assert_eq!(final_evt.text, "Hôm nay trời có mưa không?");
}

#[tokio::test]
async fn test_sherpa_stt_invalid_sample_rate_fails() {
    let adapter = SherpaSttAdapter::new();
    let res = adapter.transcribe_slice(&[0.1, 0.2], 0).await;
    assert!(matches!(res, Err(SttError::InvalidAudio(_))));
}

#[tokio::test]
async fn test_sherpa_stt_with_missing_model_paths_falls_back_gracefully() {
    let paths = OfflineModelPaths::new(PathBuf::from("/nonexistent/models"));
    let mut config = vira_infer::SherpaSttConfig::default();
    config.model_paths = paths;

    let adapter = SherpaSttAdapter::from_config(config);
    assert!(!adapter.is_neural_active());

    let silent = vec![0.0f32; 1600];
    let res = adapter.transcribe_slice(&silent, 16000).await.expect("success");
    assert_eq!(res.text, "");
}
