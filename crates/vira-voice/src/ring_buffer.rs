use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use vira_ports::audio::AudioError;

/// Real-time SPSC Audio Ring Buffer built on top of `rtrb::RingBuffer`.
///
/// Provides zero-allocation, lock-free communication between background async tasks (producer)
/// and high-priority OS audio callbacks (consumer).
/// Supports sub-millisecond atomic buffer flushing for instant (<10ms) user barge-in.
#[derive(Debug)]
pub struct AudioRingBuffer;

impl AudioRingBuffer {
    /// Creates a lock-free SPSC ring buffer pair with the specified sample capacity.
    pub fn create(capacity: usize) -> (AudioProducer, AudioConsumer) {
        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(capacity);
        let flush_generation = Arc::new(AtomicU64::new(0));

        let p = AudioProducer {
            inner: producer,
            flush_generation: flush_generation.clone(),
            capacity,
            flushed: false,
        };

        let c = AudioConsumer {
            inner: consumer,
            flush_generation,
            seen_generation: 0,
            capacity,
        };

        (p, c)
    }
}

/// Producer endpoint for writing PCM float samples to the ring buffer.
///
/// Thread-safe for writing from worker or async threads.
pub struct AudioProducer {
    inner: rtrb::Producer<f32>,
    flush_generation: Arc<AtomicU64>,
    capacity: usize,
    flushed: bool,
}

impl AudioProducer {
    /// Returns the maximum capacity in samples.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of available write slots currently remaining.
    #[inline]
    pub fn slots(&self) -> usize {
        self.inner.slots()
    }

    /// Returns true if no write slots are available.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.inner.is_full()
    }

    /// Returns the estimated number of buffered samples awaiting playback.
    /// Returns 0 immediately after a flush.
    #[inline]
    pub fn buffered_samples(&self) -> usize {
        if self.flushed {
            0
        } else {
            self.capacity.saturating_sub(self.inner.slots())
        }
    }

    /// Pushes an exact slice of PCM 32-bit float samples into the ring buffer.
    ///
    /// If there is insufficient capacity to fit the entire slice, returns `AudioError::BufferOverflow`.
    pub fn push_slice(&mut self, samples: &[f32]) -> Result<(), AudioError> {
        let available = self.inner.slots();
        if samples.len() > available {
            return Err(AudioError::BufferOverflow {
                requested: samples.len(),
                available,
            });
        }

        for &sample in samples {
            if self.inner.push(sample).is_err() {
                return Err(AudioError::BufferOverflow {
                    requested: samples.len(),
                    available: self.inner.slots(),
                });
            }
        }

        self.flushed = false;
        Ok(())
    }

    /// Pushes as many samples as possible from `samples` into the ring buffer.
    ///
    /// Returns the number of samples actually written. Never overflows.
    pub fn push_partial(&mut self, samples: &[f32]) -> Result<usize, AudioError> {
        let available = self.inner.slots();
        let to_write = samples.len().min(available);

        for &sample in &samples[..to_write] {
            if self.inner.push(sample).is_err() {
                break;
            }
        }

        if to_write > 0 {
            self.flushed = false;
        }
        Ok(to_write)
    }

    /// Pushes a single float sample.
    pub fn push(&mut self, sample: f32) -> Result<(), AudioError> {
        self.inner.push(sample).map_err(|_| AudioError::BufferOverflow {
            requested: 1,
            available: self.inner.slots(),
        })?;
        self.flushed = false;
        Ok(())
    }

    /// Instantly triggers a flush of all pending audio in the buffer.
    ///
    /// The consumer will discard all queued audio on its next check or audio callback cycle.
    #[inline]
    pub fn flush(&mut self) {
        self.flushed = true;
        self.flush_generation.fetch_add(1, Ordering::SeqCst);
    }
}

// Safety: AudioProducer is Send across threads.
unsafe impl Send for AudioProducer {}

/// Consumer endpoint for reading PCM float samples from the ring buffer.
///
/// Designed for execution inside high-priority OS audio hardware callbacks (e.g. `cpal`).
/// Completely lock-free and wait-free without heap allocations.
pub struct AudioConsumer {
    inner: rtrb::Consumer<f32>,
    flush_generation: Arc<AtomicU64>,
    seen_generation: u64,
    capacity: usize,
}

impl AudioConsumer {
    /// Internal helper to drain the buffer if a flush was signaled by the producer.
    fn check_and_apply_flush(&mut self) {
        let current_gen = self.flush_generation.load(Ordering::Acquire);
        if current_gen > self.seen_generation {
            self.seen_generation = current_gen;
            let unread = self.inner.slots();
            if unread > 0 {
                if let Ok(chunk) = self.inner.read_chunk(unread) {
                    chunk.commit_all();
                }
            }
        }
    }

    /// Returns the maximum capacity in samples.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of readable samples currently buffered.
    /// Automatically applies pending flushes if signaled.
    pub fn slots(&mut self) -> usize {
        self.check_and_apply_flush();
        self.inner.slots()
    }

    /// Returns true if no audio samples are currently buffered.
    pub fn is_empty(&mut self) -> bool {
        self.slots() == 0
    }

    /// Pops a single sample if available. Returns None if empty or flushed.
    pub fn pop(&mut self) -> Option<f32> {
        self.check_and_apply_flush();
        self.inner.pop().ok()
    }

    /// Reads samples into `output`.
    ///
    /// If fewer samples are available than `output.len()`, the remainder of `output`
    /// is zeroed with silence (0.0). Returns the number of actual audio samples read.
    pub fn pop_slice(&mut self, output: &mut [f32]) -> usize {
        self.check_and_apply_flush();

        let available = self.inner.slots();
        let to_read = output.len().min(available);

        for out in output.iter_mut().take(to_read) {
            if let Ok(sample) = self.inner.pop() {
                *out = sample;
            } else {
                *out = 0.0;
            }
        }

        // Fill remaining output buffer with silence
        for out in output.iter_mut().skip(to_read) {
            *out = 0.0;
        }

        to_read
    }
}

// Safety: AudioConsumer is Send across threads.
unsafe impl Send for AudioConsumer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_push_and_pop() {
        let (mut producer, mut consumer) = AudioRingBuffer::create(1024);
        assert_eq!(producer.capacity(), 1024);
        assert_eq!(consumer.capacity(), 1024);
        assert_eq!(consumer.slots(), 0);
        assert!(consumer.is_empty());

        let samples = [0.1, 0.2, 0.3, 0.4];
        producer.push_slice(&samples).expect("push_slice should succeed");
        assert_eq!(consumer.slots(), 4);
        assert!(!consumer.is_empty());

        let mut output = [0.0; 4];
        let read = consumer.pop_slice(&mut output);
        assert_eq!(read, 4);
        assert_eq!(output, samples);
        assert_eq!(consumer.slots(), 0);
        assert!(consumer.is_empty());
    }

    #[test]
    fn test_ring_buffer_underflow_silence_padding() {
        let (mut producer, mut consumer) = AudioRingBuffer::create(256);
        producer.push_slice(&[1.0, 2.0]).unwrap();

        let mut output = [9.9; 6];
        let read = consumer.pop_slice(&mut output);
        assert_eq!(read, 2);
        assert_eq!(output, [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ring_buffer_overflow_error() {
        let (mut producer, _) = AudioRingBuffer::create(4);
        let err = producer.push_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(matches!(err, Err(AudioError::BufferOverflow { requested: 5, available: 4 })));
    }

    #[test]
    fn test_ring_buffer_partial_push() {
        let (mut producer, mut consumer) = AudioRingBuffer::create(4);
        let written = producer.push_partial(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(written, 4);
        assert_eq!(consumer.slots(), 4);
    }
}
