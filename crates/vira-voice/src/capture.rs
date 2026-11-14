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

/// Continuous streaming linear interpolation resampler that preserves phase and sample
/// continuity across arbitrary audio chunk boundaries without allocations or phase drift.
#[derive(Debug, Clone)]
pub struct LinearResampler {
    from_rate: u32,
    to_rate: u32,
    ratio: f64,
    phase: f64,
    last_sample: f32,
    has_last_sample: bool,
}

impl LinearResampler {
    /// Creates a new streaming resampler between sample rates.
    pub fn new(from_rate: u32, to_rate: u32) -> Self {
        let ratio = if to_rate > 0 {
            from_rate as f64 / to_rate as f64
        } else {
            1.0
        };
        Self {
            from_rate,
            to_rate,
            ratio,
            phase: 0.0,
            last_sample: 0.0,
            has_last_sample: false,
        }
    }

    /// Resamples an input slice of mono PCM samples into an existing stack or heap buffer `output`.
    /// Returns the number of resampled samples written into `output`.
    ///
    /// Real-time safe: zero heap allocations, zero locks.
    pub fn resample_into_slice(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        if input.is_empty() || self.ratio <= 0.0 {
            return 0;
        }

        if self.from_rate == self.to_rate {
            let n = input.len().min(output.len());
            output[..n].copy_from_slice(&input[..n]);
            return n;
        }

        if !self.has_last_sample {
            self.last_sample = input[0];
            self.has_last_sample = true;
            self.phase = 0.0;
        }

        let in_len = input.len();
        let mut t = self.phase;
        let mut out_idx = 0;

        while t < in_len as f64 && out_idx < output.len() {
            let idx0 = t.floor() as isize;
            let frac = (t - t.floor()) as f32;

            let (s0, s1) = if idx0 < 0 {
                (self.last_sample, input[0])
            } else if (idx0 as usize + 1) < in_len {
                (input[idx0 as usize], input[idx0 as usize + 1])
            } else if idx0 as usize == in_len - 1 && frac < 1e-6 {
                (input[idx0 as usize], input[idx0 as usize])
            } else {
                // Next sample is in the upcoming chunk; break and retain phase
                break;
            };

            output[out_idx] = s0 + frac * (s1 - s0);
            out_idx += 1;
            t += self.ratio;
        }

        self.phase = t - in_len as f64;
        self.last_sample = input[in_len - 1];
        out_idx
    }

    /// Resamples an input slice of mono PCM samples, appending resampled samples into `output`.
    pub fn resample_chunk(&mut self, input: &[f32], output: &mut Vec<f32>) {
        if input.is_empty() || self.ratio <= 0.0 {
            return;
        }

        if self.from_rate == self.to_rate {
            output.extend_from_slice(input);
            return;
        }

        if !self.has_last_sample {
            self.last_sample = input[0];
            self.has_last_sample = true;
            self.phase = 0.0;
        }

        let in_len = input.len();
        let mut t = self.phase;

        while t < in_len as f64 {
            let idx0 = t.floor() as isize;
            let frac = (t - t.floor()) as f32;

            let (s0, s1) = if idx0 < 0 {
                (self.last_sample, input[0])
            } else if (idx0 as usize + 1) < in_len {
                (input[idx0 as usize], input[idx0 as usize + 1])
            } else if idx0 as usize == in_len - 1 && frac < 1e-6 {
                (input[idx0 as usize], input[idx0 as usize])
            } else {
                // Next sample is in the upcoming chunk; break and retain phase
                break;
            };

            output.push(s0 + frac * (s1 - s0));
            t += self.ratio;
        }

        self.phase = t - in_len as f64;
        self.last_sample = input[in_len - 1];
    }
}

/// One-shot linear interpolation resampler.
pub fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() || from_rate == 0 || to_rate == 0 {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = (samples.len() as f64 * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_idx = i as f64 / ratio;
        let idx0 = src_idx.floor() as usize;
        let frac = (src_idx - idx0 as f64) as f32;
        let s0 = samples.get(idx0).copied().unwrap_or(0.0);
        let s1 = samples.get(idx0 + 1).copied().unwrap_or(s0);
        out.push(s0 + frac * (s1 - s0));
    }
    out
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
    /// Automatically queries `device.default_input_config()` to detect the hardware's native
    /// sample rate and channel layout (e.g. PipeWire stereo 48000Hz or 44100Hz).
    /// If native channels > 1 or sample rate != 16000Hz, downmixes to mono and resamples
    /// down to 16000Hz before pushing to the lock-free `AudioRingBuffer`.
    ///
    /// Outputs clean 16000Hz mono audio chunks of `config.chunk_size` (default 512 samples / 32ms).
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

        // Query native hardware configuration
        let default_config = device.default_input_config().ok();
        let (native_channels, native_sample_rate, sample_format) = match default_config {
            Some(ref supported) => (
                supported.channels(),
                supported.sample_rate().0,
                supported.sample_format(),
            ),
            None => (config.channels, config.sample_rate, cpal::SampleFormat::F32),
        };

        let stream_config = cpal::StreamConfig {
            channels: native_channels,
            sample_rate: cpal::SampleRate(native_sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let ring_capacity = config.chunk_size.max(512) * 16;
        let (mut producer, mut consumer) = AudioRingBuffer::create(ring_capacity);

        let (broadcast_tx, _) = broadcast::channel(128);
        let is_capturing = Arc::new(AtomicBool::new(true));
        let is_capturing_clone = is_capturing.clone();

        let notify = Arc::new(tokio::sync::Notify::new());
        let notify_audio = notify.clone();
        let channels = native_channels as usize;
        let target_sample_rate = config.sample_rate;

        let err_callback = |err| {
            eprintln!("Audio capture stream error: {err}");
        };

        let stream = match sample_format {
            cpal::SampleFormat::I16 => {
                let mut mono_buf = [0.0f32; 1024];
                let mut resample_buf = [0.0f32; 1024];
                let mut resampler = if native_sample_rate != target_sample_rate {
                    Some(LinearResampler::new(native_sample_rate, target_sample_rate))
                } else {
                    None
                };

                device.build_input_stream(
                    &stream_config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        if !is_capturing_clone.load(Ordering::Relaxed) {
                            return;
                        }

                        if channels == 1 {
                            for chunk in data.chunks(1024) {
                                for (dst, &src) in mono_buf[..chunk.len()].iter_mut().zip(chunk.iter()) {
                                    *dst = src as f32 / 32768.0;
                                }
                                if let Some(ref mut r) = resampler {
                                    let written = r.resample_into_slice(&mono_buf[..chunk.len()], &mut resample_buf);
                                    if written > 0 {
                                        let _ = producer.push_partial(&resample_buf[..written]);
                                    }
                                } else {
                                    let _ = producer.push_partial(&mono_buf[..chunk.len()]);
                                }
                            }
                        } else {
                            for chunk in data.chunks(1024 * channels) {
                                let frames = chunk.len() / channels;
                                for (i, mono) in mono_buf[..frames].iter_mut().enumerate() {
                                    let mut sum = 0.0f32;
                                    for ch in 0..channels {
                                        sum += chunk[i * channels + ch] as f32 / 32768.0;
                                    }
                                    *mono = sum / channels as f32;
                                }
                                if let Some(ref mut r) = resampler {
                                    let written = r.resample_into_slice(&mono_buf[..frames], &mut resample_buf);
                                    if written > 0 {
                                        let _ = producer.push_partial(&resample_buf[..written]);
                                    }
                                } else {
                                    let _ = producer.push_partial(&mono_buf[..frames]);
                                }
                            }
                        }
                        notify_audio.notify_one();
                    },
                    err_callback,
                    None,
                )
            }
            _ => {
                let mut mono_buf = [0.0f32; 1024];
                let mut resample_buf = [0.0f32; 1024];
                let mut resampler = if native_sample_rate != target_sample_rate {
                    Some(LinearResampler::new(native_sample_rate, target_sample_rate))
                } else {
                    None
                };

                device.build_input_stream(
                    &stream_config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if !is_capturing_clone.load(Ordering::Relaxed) {
                            return;
                        }

                        if channels == 1 {
                            if let Some(ref mut r) = resampler {
                                for chunk in data.chunks(1024) {
                                    let written = r.resample_into_slice(chunk, &mut resample_buf);
                                    if written > 0 {
                                        let _ = producer.push_partial(&resample_buf[..written]);
                                    }
                                }
                            } else {
                                let _ = producer.push_partial(data);
                            }
                        } else {
                            for chunk in data.chunks(1024 * channels) {
                                let frames = chunk.len() / channels;
                                for (i, mono) in mono_buf[..frames].iter_mut().enumerate() {
                                    let mut sum = 0.0f32;
                                    for ch in 0..channels {
                                        sum += chunk[i * channels + ch];
                                    }
                                    *mono = sum / channels as f32;
                                }
                                if let Some(ref mut r) = resampler {
                                    let written = r.resample_into_slice(&mono_buf[..frames], &mut resample_buf);
                                    if written > 0 {
                                        let _ = producer.push_partial(&resample_buf[..written]);
                                    }
                                } else {
                                    let _ = producer.push_partial(&mono_buf[..frames]);
                                }
                            }
                        }
                        notify_audio.notify_one();
                    },
                    err_callback,
                    None,
                )
            }
        }
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

    #[test]
    fn test_linear_resampler_streaming_equivalence() {
        // Generate a 48000Hz test signal (sine wave)
        let sample_count = 4800; // 100ms at 48kHz
        let mut signal_48k = Vec::with_capacity(sample_count);
        for i in 0..sample_count {
            let t = i as f32 / 48000.0;
            signal_48k.push((2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }

        // 1. One-shot reference resampling (48k -> 16k)
        let oneshot = resample_linear(&signal_48k, 48000, 16000);
        assert_eq!(oneshot.len(), 1600); // exactly 1/3 samples

        // 2. Streaming resampler fed in small chunks (e.g. 240 samples per chunk = 5ms)
        let mut resampler = LinearResampler::new(48000, 16000);
        let mut streamed = Vec::new();
        for chunk in signal_48k.chunks(240) {
            resampler.resample_chunk(chunk, &mut streamed);
        }

        // Verify streamed output matches one-shot with high precision
        assert!((streamed.len() as isize - oneshot.len() as isize).abs() <= 1);
        let compare_len = streamed.len().min(oneshot.len());
        for i in 0..compare_len {
            assert!(
                (streamed[i] - oneshot[i]).abs() < 1e-4,
                "Sample mismatch at {i}: streamed={}, oneshot={}",
                streamed[i],
                oneshot[i]
            );
        }
    }

    #[test]
    fn test_linear_resampler_non_integer_ratio() {
        // 44100Hz -> 16000Hz (ratio = 2.75625)
        let sample_count = 4410; // 100ms
        let mut signal_44k = Vec::with_capacity(sample_count);
        for i in 0..sample_count {
            let t = i as f32 / 44100.0;
            signal_44k.push((2.0 * std::f32::consts::PI * 1000.0 * t).sin());
        }

        let mut resampler = LinearResampler::new(44100, 16000);
        let mut streamed = Vec::new();
        for chunk in signal_44k.chunks(256) {
            let mut buf = [0.0f32; 512];
            let written = resampler.resample_into_slice(chunk, &mut buf);
            streamed.extend_from_slice(&buf[..written]);
        }

        // Expected output samples: ~1600 samples
        assert!((streamed.len() as isize - 1600).abs() <= 5);
        for &s in &streamed {
            assert!(s >= -1.0 && s <= 1.0);
        }
    }

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

    #[tokio::test]
    async fn test_native_device_info_and_creation() {
        let host = cpal::default_host();
        if let Some(device) = host.default_input_device() {
            if let Ok(config) = device.default_input_config() {
                println!(
                    "Default input config: channels={}, sample_rate={}, format={:?}",
                    config.channels(),
                    config.sample_rate().0,
                    config.sample_format()
                );
            }
            let res = CaptureEngine::new_native(CaptureConfig::default());
            assert!(res.is_ok(), "Native capture engine should successfully initialize");
            let engine = res.unwrap();
            assert_eq!(engine.sample_rate(), 16000);
            assert_eq!(engine.chunk_size(), 512);
        }
    }
}
