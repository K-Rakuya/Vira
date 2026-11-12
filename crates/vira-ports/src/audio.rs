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

    /// Instantly clears/flushes playback buffers (<10ms) when user barge-in occurs.
    fn flush(&self) -> Result<(), AudioError>;

    /// Returns the monotonic hardware sample clock (total count of samples pushed to sound card).
    fn get_sample_clock(&self) -> u64;

    /// Hardware output sample rate in Hz (e.g. 48000).
    fn sample_rate(&self) -> u32;

    /// Returns current playback time in milliseconds derived from the master sample clock.
    fn current_time_ms(&self) -> u64 {
        let rate = self.sample_rate();
        if rate == 0 {
            0
        } else {
            (self.get_sample_clock() * 1000) / rate as u64
        }
    }

    /// Returns the number of queued PCM samples awaiting playback.
    fn buffered_samples(&self) -> usize;

    /// Returns the maximum capacity of the playback ring buffer in samples.
    fn buffer_capacity(&self) -> usize;

    /// Returns true if audio is actively playing or queued in the buffer.
    fn is_playing(&self) -> bool {
        self.buffered_samples() > 0
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
    }

    impl AudioPlaybackPort for MockPlayback {
        fn enqueue_samples(&self, _samples: &[f32]) -> Result<usize, AudioError> {
            Ok(0)
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
            1024
        }
    }

    #[test]
    fn test_audio_playback_default_methods() {
        let pb = MockPlayback {
            rate: 48000,
            clock: 24000,
            buffered: 128,
        };
        assert_eq!(pb.current_time_ms(), 500);
        assert!(pb.is_playing());

        let pb_empty = MockPlayback {
            rate: 48000,
            clock: 0,
            buffered: 0,
        };
        assert_eq!(pb_empty.current_time_ms(), 0);
        assert!(!pb_empty.is_playing());

        let pb_zero_rate = MockPlayback {
            rate: 0,
            clock: 1000,
            buffered: 0,
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
    }
}

