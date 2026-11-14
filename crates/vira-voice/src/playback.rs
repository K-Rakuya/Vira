use std::sync::Mutex;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::{Deserialize, Serialize};
use vira_ports::audio::{AudioError, AudioPlaybackPort};

use crate::clock::MasterAudioClock;
use crate::ring_buffer::{AudioConsumer, AudioProducer, AudioRingBuffer};

/// Configuration parameters for native audio playback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlaybackConfig {
    /// Sample rate in Hz (default: 48000).
    pub sample_rate: u32,
    /// Channel count (default: 2 for stereo).
    pub channels: u16,
    /// Ring buffer capacity in samples (default: 2880000, 60.0 seconds at 48kHz).
    pub buffer_capacity: usize,
    /// Specific output audio device name, or `None` for default.
    pub device_name: Option<String>,
}

impl Default for PlaybackConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            buffer_capacity: 2880000,
            device_name: None,
        }
    }
}

/// Internal wrapper for cpal::Stream to allow safe storage in Send + Sync engine.
struct StreamHandle(#[allow(dead_code)] cpal::Stream);
// Safety: Underlying stream handles on desktop targets can be held and dropped across threads.
unsafe impl Send for StreamHandle {}
unsafe impl Sync for StreamHandle {}

/// Real-time native audio playback engine powered by `cpal` and lock-free SPSC buffers.
///
/// Implements `AudioPlaybackPort` from `vira-ports`.
pub struct PlaybackEngine {
    clock: MasterAudioClock,
    producer: Mutex<AudioProducer>,
    capacity: usize,
    sample_rate: u32,
    _stream: Option<StreamHandle>,
}

impl PlaybackEngine {
    /// Creates a virtual/headless playback engine without opening physical audio hardware.
    ///
    /// Returns the `PlaybackEngine` (which acts as the producer) and the paired `AudioConsumer`.
    /// Ideal for unit tests, offline synthesis rendering, or custom streaming pipelines.
    pub fn new_virtual(sample_rate: u32, capacity: usize) -> (Self, AudioConsumer) {
        let clock = MasterAudioClock::new(sample_rate);
        let (producer, consumer) = AudioRingBuffer::create(capacity);

        let engine = Self {
            clock,
            producer: Mutex::new(producer),
            capacity,
            sample_rate,
            _stream: None,
        };

        (engine, consumer)
    }

    /// Initializes a live native audio playback engine using `cpal` and the system's default sound card.
    pub fn new_native(config: PlaybackConfig) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = if let Some(ref name) = config.device_name {
            host.output_devices()
                .map_err(|e| AudioError::Device(e.to_string()))?
                .find(|d| d.name().map(|n| n == *name).unwrap_or(false))
                .ok_or_else(|| AudioError::Device(format!("Audio device '{name}' not found")))?
        } else {
            host.default_output_device()
                .ok_or_else(|| AudioError::Device("No default output audio device found".into()))?
        };

        let stream_config = cpal::StreamConfig {
            channels: config.channels,
            sample_rate: cpal::SampleRate(config.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let clock = MasterAudioClock::new(config.sample_rate);
        let (producer, mut consumer) = AudioRingBuffer::create(config.buffer_capacity);

        let clock_clone = clock.clone();
        let channels = config.channels as usize;

        let err_callback = |err| {
            eprintln!("Audio playback stream error: {err}");
        };

        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if channels == 1 {
                        let read = consumer.pop_slice(data);
                        clock_clone.advance_samples(read as u64);
                    } else {
                        let mut mono_buf = [0.0f32; 512];
                        let mut total_rendered = 0u64;
                        for chunk in data.chunks_mut(512 * channels) {
                            let frames = chunk.len() / channels;
                            let read = consumer.pop_slice(&mut mono_buf[..frames]);
                            total_rendered += read as u64;
                            for (i, &sample) in mono_buf[..frames].iter().enumerate() {
                                for ch in 0..channels {
                                    chunk[i * channels + ch] = sample;
                                }
                            }
                        }
                        clock_clone.advance_samples(total_rendered);
                    }
                },
                err_callback,
                None,
            )
            .map_err(|e| AudioError::Stream(e.to_string()))?;

        stream
            .play()
            .map_err(|e| AudioError::Stream(e.to_string()))?;

        Ok(Self {
            clock,
            producer: Mutex::new(producer),
            capacity: config.buffer_capacity,
            sample_rate: config.sample_rate,
            _stream: Some(StreamHandle(stream)),
        })
    }

    /// Returns a reference to the shared master sample clock.
    pub fn clock(&self) -> &MasterAudioClock {
        &self.clock
    }
}

impl AudioPlaybackPort for PlaybackEngine {
    fn enqueue_samples(&self, samples: &[f32]) -> Result<usize, AudioError> {
        let mut producer = self.producer.lock().map_err(|e| {
            AudioError::Other(format!("Failed to acquire playback producer lock: {e}"))
        })?;
        producer.push_partial(samples)
    }

    fn enqueue_samples_exact(&self, samples: &[f32]) -> Result<(), AudioError> {
        let mut producer = self.producer.lock().map_err(|e| {
            AudioError::Other(format!("Failed to acquire playback producer lock: {e}"))
        })?;
        producer.push_slice(samples)
    }

    fn flush(&self) -> Result<(), AudioError> {
        let mut producer = self.producer.lock().map_err(|e| {
            AudioError::Other(format!("Failed to acquire playback producer lock: {e}"))
        })?;
        producer.flush();
        Ok(())
    }

    fn get_sample_clock(&self) -> u64 {
        self.clock.current_sample()
    }

    fn reset_sample_clock(&self) -> Result<(), AudioError> {
        self.clock.reset();
        Ok(())
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn current_time_ms(&self) -> u64 {
        self.clock.current_time_ms()
    }

    fn buffered_samples(&self) -> usize {
        if let Ok(producer) = self.producer.lock() {
            producer.buffered_samples()
        } else {
            0
        }
    }

    fn buffer_capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtual_playback_engine_lifecycle() {
        let (engine, mut consumer) = PlaybackEngine::new_virtual(48000, 1024);
        assert_eq!(engine.sample_rate(), 48000);
        assert_eq!(engine.buffer_capacity(), 1024);
        assert_eq!(engine.buffered_samples(), 0);
        assert!(engine.is_empty());
        assert!(!engine.is_playing());

        let samples = [0.25f32; 256];
        let written = engine.enqueue_samples(&samples).unwrap();
        assert_eq!(written, 256);
        assert_eq!(engine.buffered_samples(), 256);
        assert!(engine.is_playing());
        assert!(!engine.is_empty());

        // Consumer reads 128 samples
        let mut out = [0.0; 128];
        let read = consumer.pop_slice(&mut out);
        assert_eq!(read, 128);
        engine.clock().advance_samples(read as u64);

        assert_eq!(engine.get_sample_clock(), 128);
        assert_eq!(engine.current_sample(), 128);

        // Instant flush
        engine.flush().unwrap();
        assert_eq!(engine.buffered_samples(), 0);
        assert_eq!(consumer.slots(), 0);
        assert!(engine.is_empty());
    }

    #[test]
    fn test_playback_overflow_handling() {
        let (engine, _) = PlaybackEngine::new_virtual(48000, 64);
        let samples = [0.1f32; 100];

        // Exact push must overflow
        let err = engine.enqueue_samples_exact(&samples);
        assert!(matches!(err, Err(AudioError::BufferOverflow { requested: 100, available: 64 })));

        // Partial push must write up to capacity
        let written = engine.enqueue_samples(&samples).unwrap();
        assert_eq!(written, 64);
        assert_eq!(engine.buffered_samples(), 64);
    }
}
