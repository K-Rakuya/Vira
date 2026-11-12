use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;
use vira_ports::audio::{AudioCapturePort, AudioError};

use crate::ring_buffer::AudioRingBuffer;

/// Configuration parameters for microphone audio capture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureConfig {
    /// Sample rate in Hz (default: 16000 Hz, standard for VAD and STT).
    pub sample_rate: u32,
    /// Channel count (default: 1 for mono).
    pub channels: u16,
    /// Frame chunk size dispatched to downstream VAD/STT (default: 512 samples = 32ms at 16kHz).
    pub chunk_size: usize,
    /// Specific input audio device name, or `None` for default.
    pub device_name: Option<String>,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            chunk_size: 512,
            device_name: None,
        }
    }
}

/// Simulated / virtual microphone feed handle for feeding PCM data into `CaptureEngine`.
#[derive(Clone)]
pub struct VirtualAudioFeed {
    sender: broadcast::Sender<Vec<f32>>,
}

impl VirtualAudioFeed {
    /// Injects a chunk of PCM audio samples into the capture stream.
    pub fn feed_chunk(&self, chunk: Vec<f32>) {
        let _ = self.sender.send(chunk);
    }
}

/// Internal wrapper for cpal::Stream to allow safe storage in Send + Sync engine.
struct StreamHandle(#[allow(dead_code)] cpal::Stream);
// Safety: Underlying stream handles on desktop targets can be held and dropped across threads.
unsafe impl Send for StreamHandle {}
unsafe impl Sync for StreamHandle {}

/// Real-time microphone audio capture engine powered by `cpal` and lock-free SPSC buffers.
///
/// Implements `AudioCapturePort` from `vira-ports`.
pub struct CaptureEngine {
    sample_rate: u32,
    chunk_size: usize,
    is_capturing: Arc<AtomicBool>,
    broadcast_tx: broadcast::Sender<Vec<f32>>,
    _stream: Option<StreamHandle>,
    _worker_cancel: Option<CancellationToken>,
}

impl CaptureEngine {
    /// Creates a virtual/headless capture engine for testing and programmatic audio streaming.
    ///
    /// Returns the `CaptureEngine` and a `VirtualAudioFeed` to inject simulated microphone audio.
    pub fn new_virtual(sample_rate: u32, chunk_size: usize) -> (Self, VirtualAudioFeed) {
        let (broadcast_tx, _) = broadcast::channel(128);
        let feed = VirtualAudioFeed {
            sender: broadcast_tx.clone(),
        };

        let engine = Self {
            sample_rate,
            chunk_size,
            is_capturing: Arc::new(AtomicBool::new(true)),
            broadcast_tx,
            _stream: None,
            _worker_cancel: None,
        };

        (engine, feed)
    }

    /// Initializes a live native audio capture engine using `cpal` and the system's microphone.
    ///
    /// The high-priority OS audio callback pushes directly to a lock-free `AudioRingBuffer`
    /// without mutexes or heap allocations. A background async task consumes chunks and broadcasts
    /// them downstream.
    pub fn new_native(config: CaptureConfig) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = if let Some(ref name) = config.device_name {
            host.input_devices()
                .map_err(|e| AudioError::Device(e.to_string()))?
                .find(|d| d.name().map(|n| n == *name).unwrap_or(false))
                .ok_or_else(|| AudioError::Device(format!("Capture device '{name}' not found")))?
        } else {
            host.default_input_device()
                .ok_or_else(|| AudioError::Device("No default input audio device found".into()))?
        };

        let stream_config = cpal::StreamConfig {
            channels: config.channels,
            sample_rate: cpal::SampleRate(config.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let ring_capacity = config.chunk_size.max(512) * 16;
        let (mut producer, mut consumer) = AudioRingBuffer::create(ring_capacity);

        let (broadcast_tx, _) = broadcast::channel(128);
        let is_capturing = Arc::new(AtomicBool::new(true));
        let is_capturing_clone = is_capturing.clone();

        let notify = Arc::new(tokio::sync::Notify::new());
        let notify_audio = notify.clone();
        let channels = config.channels as usize;

        let err_callback = |err| {
            eprintln!("Audio capture stream error: {err}");
        };

        let stream = device
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if !is_capturing_clone.load(Ordering::Relaxed) {
                        return;
                    }

                    if channels == 1 {
                        let _ = producer.push_partial(data);
                    } else {
                        // Downmix multi-channel to mono via stack buffer - zero heap allocations
                        let mut mono_buf = [0.0f32; 512];
                        for chunk in data.chunks(512 * channels) {
                            let frames = chunk.len() / channels;
                            for (i, mono) in mono_buf[..frames].iter_mut().enumerate() {
                                let mut sum = 0.0f32;
                                for ch in 0..channels {
                                    sum += chunk[i * channels + ch];
                                }
                                *mono = sum / channels as f32;
                            }
                            let _ = producer.push_partial(&mono_buf[..frames]);
                        }
                    }
                    notify_audio.notify_one();
                },
                err_callback,
                None,
            )
            .map_err(|e| AudioError::Stream(e.to_string()))?;

        stream
            .play()
            .map_err(|e| AudioError::Stream(e.to_string()))?;

        // Background worker to consume from lock-free ring buffer and dispatch chunks to broadcast channel
        let worker_cancel = CancellationToken::new();
        let worker_cancel_task = worker_cancel.clone();
        let worker_tx = broadcast_tx.clone();
        let chunk_size = config.chunk_size;
        let is_capturing_worker = is_capturing.clone();

        tokio::spawn(async move {
            let mut chunk_buf = vec![0.0f32; chunk_size];
            while !worker_cancel_task.is_cancelled() {
                if consumer.slots() >= chunk_size {
                    let read = consumer.pop_slice(&mut chunk_buf);
                    if read == chunk_size && is_capturing_worker.load(Ordering::Relaxed) {
                        let _ = worker_tx.send(chunk_buf.clone());
                    }
                } else {
                    tokio::select! {
                        _ = worker_cancel_task.cancelled() => break,
                        _ = notify.notified() => {},
                        _ = tokio::time::sleep(tokio::time::Duration::from_millis(5)) => {},
                    }
                }
            }
        });

        Ok(Self {
            sample_rate: config.sample_rate,
            chunk_size: config.chunk_size,
            is_capturing,
            broadcast_tx,
            _stream: Some(StreamHandle(stream)),
            _worker_cancel: Some(worker_cancel),
        })
    }

    /// Returns the chunk size in samples.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }
}

impl Drop for CaptureEngine {
    fn drop(&mut self) {
        if let Some(ref cancel) = self._worker_cancel {
            cancel.cancel();
        }
    }
}

#[async_trait]
impl AudioCapturePort for CaptureEngine {
    fn start_capture(&self) -> Result<(), AudioError> {
        self.is_capturing.store(true, Ordering::Release);
        Ok(())
    }

    fn stop_capture(&self) -> Result<(), AudioError> {
        self.is_capturing.store(false, Ordering::Release);
        Ok(())
    }

    fn is_capturing(&self) -> bool {
        self.is_capturing.load(Ordering::Acquire)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn read_chunk(&self, cancel: CancellationToken) -> Result<Vec<f32>, AudioError> {
        let mut rx = self.broadcast_tx.subscribe();

        loop {
            tokio::select! {
                _ = cancel.cancelled() => return Err(AudioError::Cancelled),
                recv_res = rx.recv() => {
                    match recv_res {
                        Ok(chunk) => {
                            if self.is_capturing() {
                                return Ok(chunk);
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(broadcast::error::RecvError::Closed) => {
                            return Err(AudioError::Stream("Capture channel closed".into()));
                        }
                    }
                }
            }
        }
    }

    async fn stream_capture(
        &self,
        sender: Sender<Vec<f32>>,
        cancel: CancellationToken,
    ) -> Result<(), AudioError> {
        let mut rx = self.broadcast_tx.subscribe();

        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                recv_res = rx.recv() => {
                    match recv_res {
                        Ok(chunk) => {
                            if self.is_capturing() && sender.send(chunk).await.is_err() {
                                // Receiver channel dropped
                                break;
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_virtual_capture_engine_read_chunk() {
        let (engine, feed) = CaptureEngine::new_virtual(16000, 512);
        assert_eq!(engine.sample_rate(), 16000);
        assert_eq!(engine.chunk_size(), 512);
        assert!(engine.is_capturing());

        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        let handle = tokio::spawn(async move {
            engine.read_chunk(cancel_clone).await
        });

        let test_data = vec![0.5; 512];
        let test_clone = test_data.clone();
        let feeder = tokio::spawn(async move {
            for _ in 0..50 {
                feed.feed_chunk(test_clone.clone());
                tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
            }
        });

        let result = tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .expect("timeout waiting for read_chunk")
            .unwrap()
            .unwrap();
        assert_eq!(result, test_data);
        feeder.abort();
    }

    #[tokio::test]
    async fn test_virtual_capture_engine_cancellation() {
        let (engine, _) = CaptureEngine::new_virtual(16000, 512);
        let cancel = CancellationToken::new();

        cancel.cancel();
        let result = engine.read_chunk(cancel).await;
        assert_eq!(result, Err(AudioError::Cancelled));
    }

    #[tokio::test]
    async fn test_capture_pause_and_resume() {
        let (engine, _) = CaptureEngine::new_virtual(16000, 512);
        assert!(engine.is_capturing());

        engine.stop_capture().unwrap();
        assert!(!engine.is_capturing());

        engine.start_capture().unwrap();
        assert!(engine.is_capturing());
    }

    #[test]
    fn test_capture_config_serde() {
        let config = CaptureConfig {
            sample_rate: 16000,
            channels: 1,
            chunk_size: 512,
            device_name: Some("default_mic".to_string()),
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: CaptureConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, decoded);
    }
}
