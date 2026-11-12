use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use vira_domain::viseme::VisemeWeights;
use vira_ports::audio::{AudioCapturePort, AudioPlaybackPort};
use vira_ports::tts::TimedViseme;
use vira_voice::barge_in::BargeInController;
use vira_voice::capture::CaptureEngine;
use vira_voice::clock::MasterAudioClock;
use vira_voice::playback::PlaybackEngine;
use vira_voice::ring_buffer::AudioRingBuffer;

#[test]
fn test_master_sample_clock_progression() {
    let clock = MasterAudioClock::new(48000);
    assert_eq!(clock.current_sample(), 0);
    assert_eq!(clock.current_time_ms(), 0);

    clock.advance_samples(48000);
    assert_eq!(clock.current_sample(), 48000);
    assert_eq!(clock.current_time_ms(), 1000);
}

#[test]
fn test_ring_buffer_instant_flush() {
    let (mut producer, mut consumer) = AudioRingBuffer::create(1024);
    producer.push_slice(&[0.5; 512]).expect("push success");
    assert_eq!(consumer.slots(), 512);

    producer.flush();
    assert_eq!(consumer.slots(), 0);
}

#[test]
fn test_playback_engine_audio_port_integration() {
    let (playback, mut consumer) = PlaybackEngine::new_virtual(48000, 2048);
    let port: &dyn AudioPlaybackPort = &playback;

    assert_eq!(port.sample_rate(), 48000);
    assert_eq!(port.buffer_capacity(), 2048);
    assert_eq!(port.buffered_samples(), 0);
    assert!(port.is_empty());
    assert!(!port.is_playing());

    // Enqueue 480 samples (10ms at 48kHz)
    let pcm = vec![0.3f32; 480];
    let written = port.enqueue_samples(&pcm).expect("enqueue successful");
    assert_eq!(written, 480);
    assert_eq!(port.buffered_samples(), 480);
    assert!(port.is_playing());
    assert_eq!(port.available_space(), 2048 - 480);

    // Audio callback consumer pops 480 samples
    let mut out_buffer = vec![0.0f32; 480];
    let popped = consumer.pop_slice(&mut out_buffer);
    assert_eq!(popped, 480);
    playback.clock().advance_samples(popped as u64);

    assert_eq!(port.get_sample_clock(), 480);
    assert_eq!(port.current_time_ms(), 10);

    // Flush
    port.flush().expect("flush successful");
    assert_eq!(port.buffered_samples(), 0);
    assert!(port.is_empty());

    // Clock reset
    port.reset_sample_clock().expect("reset clock");
    assert_eq!(port.get_sample_clock(), 0);
    assert_eq!(port.current_time_ms(), 0);
}

#[test]
fn test_barge_in_instant_flush_and_cancel() {
    let (playback, _consumer) = PlaybackEngine::new_virtual(48000, 4096);
    let playback = Arc::new(playback);

    // Assistant is speaking
    playback.enqueue_samples(&[0.4; 1000]).unwrap();
    assert!(playback.is_playing());

    let controller = BargeInController::new(playback.clone());
    let cancel = controller.cancellation_token();
    assert!(!cancel.is_cancelled());

    // Trigger user interruption
    let report = controller.trigger_barge_in().unwrap();
    assert_eq!(report.interrupted_at_ms, 0);

    // Verify token cancelled immediately
    assert!(cancel.is_cancelled());

    // Verify playback stopped immediately
    assert_eq!(playback.buffered_samples(), 0);
    assert!(!playback.is_playing());

    // Verify re-armed cancellation token is fresh
    let fresh_token = controller.cancellation_token();
    assert!(!fresh_token.is_cancelled());
}

#[test]
fn test_barge_in_energy_detection_during_playback() {
    let (playback, _consumer) = PlaybackEngine::new_virtual(48000, 2048);
    let playback = Arc::new(playback);
    let controller = BargeInController::new(playback.clone());
    controller.set_energy_threshold(0.04);

    let loud_input = vec![0.2f32; 256]; // RMS = 0.2 > 0.04

    // 1. Playback idle -> no barge in triggered
    let res = controller.check_pcm_frame(&loud_input).unwrap();
    assert!(res.is_none());

    // 2. Playback playing -> speech triggers barge in!
    playback.enqueue_samples(&[0.1; 512]).unwrap();
    let res = controller.check_pcm_frame(&loud_input).unwrap();
    assert!(res.is_some());
    assert!(!playback.is_playing());
}

#[tokio::test]
async fn test_capture_engine_streaming_and_cancellation() {
    let (capture, feed) = CaptureEngine::new_virtual(16000, 512);
    let capture = Arc::new(capture);

    let (tx, mut rx) = tokio::sync::mpsc::channel(10);
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let capture_clone = capture.clone();

    let task = tokio::spawn(async move {
        capture_clone.stream_capture(tx, cancel_clone).await
    });

    let frame = vec![0.123f32; 512];
    let frame_clone = frame.clone();
    let feeder = tokio::spawn(async move {
        for _ in 0..50 {
            feed.feed_chunk(frame_clone.clone());
            tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        }
    });

    let received = tokio::time::timeout(tokio::time::Duration::from_secs(1), rx.recv())
        .await
        .expect("timeout waiting for chunk")
        .expect("received captured chunk");
    assert_eq!(received, frame);

    cancel.cancel();
    let res = task.await.expect("join handle");
    assert!(res.is_ok());
    feeder.abort();
}

#[test]
fn test_clock_viseme_sampling_60hz_sync() {
    let clock = MasterAudioClock::new(48000);

    let visemes = vec![
        TimedViseme::new(0, VisemeWeights::new(0.0, 0.0, 0.0, 0.0, 0.0)),
        TimedViseme::new(100, VisemeWeights::new(1.0, 0.0, 0.0, 0.0, 0.0)), // AA vowel at 100ms
        TimedViseme::new(200, VisemeWeights::new(0.0, 1.0, 0.0, 0.0, 0.0)), // IH vowel at 200ms
    ];

    // At 60Hz, 1 frame is ~16.666ms (800 samples at 48kHz)
    // Frame 0: 0ms -> AA = 0.0
    assert_eq!(clock.sample_viseme(&visemes).aa, 0.0);

    // Frame 3: ~50ms (2400 samples) -> AA = 0.5
    clock.advance_samples(2400);
    assert_eq!(clock.current_time_ms(), 50);
    let v_50 = clock.sample_viseme(&visemes);
    assert!((v_50.aa - 0.5).abs() < 1e-4);

    // Frame 6: 100ms (4800 samples total) -> AA = 1.0
    clock.advance_samples(2400);
    assert_eq!(clock.current_time_ms(), 100);
    let v_100 = clock.sample_viseme(&visemes);
    assert!((v_100.aa - 1.0).abs() < 1e-4);

    // Frame 9: 150ms -> AA = 0.5, IH = 0.5
    clock.advance_samples(2400);
    assert_eq!(clock.current_time_ms(), 150);
    let v_150 = clock.sample_viseme(&visemes);
    assert!((v_150.aa - 0.5).abs() < 1e-4);
    assert!((v_150.ih - 0.5).abs() < 1e-4);
}

#[test]
fn test_multi_thread_audio_pipeline_stress() {
    let (playback, mut consumer) = PlaybackEngine::new_virtual(48000, 8192);
    let playback = Arc::new(playback);

    let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let running_clone = running.clone();
    let clock = playback.clock().clone();

    // Spawn high-priority consumer thread simulating OS audio callback
    let consumer_handle = std::thread::spawn(move || {
        let mut buffer = [0.0f32; 256];
        let mut total_rendered = 0u64;

        while running_clone.load(std::sync::atomic::Ordering::Relaxed) {
            let read = consumer.pop_slice(&mut buffer);
            if read > 0 {
                clock.advance_samples(read as u64);
                total_rendered += read as u64;
            }
            std::thread::sleep(std::time::Duration::from_micros(500));
        }
        total_rendered
    });

    // Producer writes chunks
    let chunk = [0.1f32; 256];
    for _ in 0..50 {
        let _ = playback.enqueue_samples(&chunk);
        std::thread::sleep(std::time::Duration::from_micros(300));
    }

    // Interleave barge-in flush
    playback.flush().unwrap();
    assert_eq!(playback.buffered_samples(), 0);

    // Write more chunks after flush
    for _ in 0..50 {
        let _ = playback.enqueue_samples(&chunk);
        std::thread::sleep(std::time::Duration::from_micros(300));
    }

    // Stop consumer
    running.store(false, std::sync::atomic::Ordering::Relaxed);
    let total_rendered = consumer_handle.join().expect("consumer thread");

    assert!(total_rendered > 0);
    assert!(playback.get_sample_clock() > 0);
}
