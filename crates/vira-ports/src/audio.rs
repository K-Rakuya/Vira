use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

/// Errors occurring during audio capture or playback.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioError {
    #[error("Audio device error: {0}")]
    Device(String),
    #[error("Audio buffer overflow: requested {requested} samples, available {available}")]
    BufferOverflow { requested: usize, available: usize },
    #[error("Audio buffer underflow: requested {requested} samples, available {available}")]
    BufferUnderflow { requested: usize, available: usize },
    #[error("Audio stream error: {0}")]
    Stream(String),
    #[error("Audio operation cancelled")]
    Cancelled,
    #[error("Other audio error: {0}")]
    Other(String),
}

/// Hexagonal Outbound Port for real-time Audio Playback.
///
/// Designed to interact directly with lock-free SPSC ring buffers and OS audio callbacks
/// (e.g. cpal) without dynamic heap allocation or blocking locks in the fast path.
pub trait AudioPlaybackPort: Send + Sync {
    /// Non-blocking enqueue of raw PCM 32-bit float samples to the playback ring buffer.
    /// Returns the number of samples actually enqueued into the buffer.
    fn enqueue_samples(&self, samples: &[f32]) -> Result<usize, AudioError>;

    /// Convenience method to enqueue an entire slice of samples.
    /// If the full slice cannot be accommodated in the buffer, returns `AudioError::BufferOverflow`.
    fn enqueue_samples_exact(&self, samples: &[f32]) -> Result<(), AudioError> {
        let written = self.enqueue_samples(samples)?;
        if written < samples.len() {
            Err(AudioError::BufferOverflow {
                requested: samples.len(),
                available: written,
            })
        } else {
            Ok(())
        }
    }

    /// Instantly clears/flushes playback buffers (<10ms) when user barge-in occurs.
    /// After a successful flush, `buffered_samples()` must return 0.
    fn flush(&self) -> Result<(), AudioError>;

    /// Returns the monotonic hardware sample clock (total count of samples pushed to sound card).
    fn get_sample_clock(&self) -> u64;

    /// Alias for `get_sample_clock()`.
    fn current_sample(&self) -> u64 {
        self.get_sample_clock()
    }

    /// Resets the hardware sample clock counter to zero if supported by the underlying engine.
    fn reset_sample_clock(&self) -> Result<(), AudioError> {
        Ok(())
    }

    /// Hardware output sample rate in Hz (e.g. 48000).
    fn sample_rate(&self) -> u32;

    /// Returns current playback time in milliseconds derived from the master sample clock.
    fn current_time_ms(&self) -> u64 {
        let rate = self.sample_rate();
        if rate == 0 {
            0
        } else {
            self.get_sample_clock().saturating_mul(1000) / rate as u64
        }
    }

    /// Returns the number of queued PCM samples awaiting playback.
    fn buffered_samples(&self) -> usize;

    /// Returns the maximum capacity of the playback ring buffer in samples.
    fn buffer_capacity(&self) -> usize;

    /// Returns the number of available write slots remaining in the playback ring buffer.
    fn available_space(&self) -> usize {
        self.buffer_capacity()
            .saturating_sub(self.buffered_samples())
    }

    /// Returns true if audio is actively playing or queued in the buffer.
    fn is_playing(&self) -> bool {
        self.buffered_samples() > 0
    }

    /// Returns true if the playback buffer is completely empty.
    fn is_empty(&self) -> bool {
        self.buffered_samples() == 0
    }
}

/// Hexagonal Inbound Port for real-time Audio Capture.
///
/// Captures microphone audio, interacts with Acoustic Echo Cancellation (AEC),
/// and streams PCM audio frames to downstream VAD and STT pipelines.
#[async_trait]
pub trait AudioCapturePort: Send + Sync {
    /// Starts capturing audio from the input device.
    fn start_capture(&self) -> Result<(), AudioError>;

    /// Stops audio capture.
    fn stop_capture(&self) -> Result<(), AudioError>;

    /// Returns true if audio capture is currently active.
    fn is_capturing(&self) -> bool;

    /// Input sample rate in Hz (e.g. 16000 or 48000).
    fn sample_rate(&self) -> u32;

    /// Reads a single chunk of captured PCM f32 samples asynchronously.
    async fn read_chunk(&self, cancel: CancellationToken) -> Result<Vec<f32>, AudioError>;

    /// Continuously captures audio and dispatches sample chunks to `sender` until cancelled.
    async fn stream_capture(
        &self,
        sender: Sender<Vec<f32>>,
        cancel: CancellationToken,
    ) -> Result<(), AudioError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockPlayback {
        rate: u32,
        clock: u64,
        buffered: usize,
        capacity: usize,
    }

    impl AudioPlaybackPort for MockPlayback {
        fn enqueue_samples(&self, samples: &[f32]) -> Result<usize, AudioError> {
            let space = self.available_space();
            let to_write = samples.len().min(space);
            Ok(to_write)
        }
        fn flush(&self) -> Result<(), AudioError> {
            Ok(())
        }
        fn get_sample_clock(&self) -> u64 {
            self.clock
        }
        fn sample_rate(&self) -> u32 {
            self.rate
        }
        fn buffered_samples(&self) -> usize {
            self.buffered
        }
        fn buffer_capacity(&self) -> usize {
            self.capacity
        }
    }

    #[test]
    fn test_audio_playback_default_methods() {
        let pb = MockPlayback {
            rate: 48000,
            clock: 24000,
            buffered: 128,
            capacity: 1024,
        };
        assert_eq!(pb.current_time_ms(), 500);
        assert_eq!(pb.current_sample(), 24000);
        assert!(pb.is_playing());
        assert!(!pb.is_empty());
        assert_eq!(pb.available_space(), 1024 - 128);
        assert!(pb.reset_sample_clock().is_ok());

        // Exact enqueue success
        assert!(pb.enqueue_samples_exact(&[0.0; 256]).is_ok());

        // Exact enqueue overflow
        let err = pb.enqueue_samples_exact(&[0.0; 2000]);
        assert!(matches!(err, Err(AudioError::BufferOverflow { .. })));

        let pb_empty = MockPlayback {
            rate: 48000,
            clock: 0,
            buffered: 0,
            capacity: 512,
        };
        assert_eq!(pb_empty.current_time_ms(), 0);
        assert!(!pb_empty.is_playing());
        assert!(pb_empty.is_empty());
        assert_eq!(pb_empty.available_space(), 512);

        let pb_zero_rate = MockPlayback {
            rate: 0,
            clock: 1000,
            buffered: 0,
            capacity: 512,
        };
        assert_eq!(pb_zero_rate.current_time_ms(), 0);
    }

    #[test]
    fn test_audio_error_display() {
        let overflow = AudioError::BufferOverflow {
            requested: 512,
            available: 256,
        };
        assert_eq!(
            overflow.to_string(),
            "Audio buffer overflow: requested 512 samples, available 256"
        );

        let underflow = AudioError::BufferUnderflow {
            requested: 128,
            available: 0,
        };
        assert_eq!(
            underflow.to_string(),
            "Audio buffer underflow: requested 128 samples, available 0"
        );
    }
}
