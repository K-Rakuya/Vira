use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use vira_ports::audio::AudioError;

/// Real-time SPSC Audio Ring Buffer built on top of `rtrb::RingBuffer`.
///
/// Provides zero-allocation, lock-free communication between background async tasks (producer)
/// and high-priority OS audio callbacks (consumer).
/// Supports sub-millisecond atomic buffer flushing for instant (<10ms) user barge-in without
/// discarding any new audio written after the flush.
#[derive(Debug)]
pub struct AudioRingBuffer;

impl AudioRingBuffer {
    /// Creates a lock-free SPSC ring buffer pair with the specified sample capacity.
    pub fn create(capacity: usize) -> (AudioProducer, AudioConsumer) {
        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(capacity);
        let total_written = Arc::new(AtomicU64::new(0));
        let total_read = Arc::new(AtomicU64::new(0));
        let discard_target = Arc::new(AtomicU64::new(0));

        let p = AudioProducer {
            inner: producer,
            total_written: total_written.clone(),
            total_read: total_read.clone(),
            discard_target: discard_target.clone(),
            capacity,
        };

        let c = AudioConsumer {
            inner: consumer,
            total_read,
            discard_target,
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
    total_written: Arc<AtomicU64>,
    total_read: Arc<AtomicU64>,
    discard_target: Arc<AtomicU64>,
    capacity: usize,
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

    /// Returns the number of valid queued samples awaiting playback.
    /// Excludes flushed / discarded samples even before the consumer runs.
    #[inline]
    pub fn buffered_samples(&self) -> usize {
        let written = self.total_written.load(Ordering::Acquire);
        let discard = self.discard_target.load(Ordering::Acquire);
        let read = self.total_read.load(Ordering::Acquire);
        let effective_read = discard.max(read);
        written.saturating_sub(effective_read) as usize
    }

    /// Monotonic count of total samples pushed by this producer.
    #[inline]
    pub fn total_written(&self) -> u64 {
        self.total_written.load(Ordering::Acquire)
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

        self.inner
            .push_entire_slice(samples)
            .map_err(|_| AudioError::BufferOverflow {
                requested: samples.len(),
                available: self.inner.slots(),
            })?;

        self.total_written
            .fetch_add(samples.len() as u64, Ordering::Release);
        Ok(())
    }

    /// Pushes as many samples as possible from `samples` into the ring buffer.
    ///
    /// Returns the number of samples actually written. Never overflows.
    pub fn push_partial(&mut self, samples: &[f32]) -> Result<usize, AudioError> {
        let (written, _) = self.inner.push_partial_slice(samples);
        let count = written.len();
        if count > 0 {
            self.total_written
                .fetch_add(count as u64, Ordering::Release);
        }
        Ok(count)
    }

    /// Pushes a single float sample.
    pub fn push(&mut self, sample: f32) -> Result<(), AudioError> {
        self.inner.push(sample).map_err(|_| AudioError::BufferOverflow {
            requested: 1,
            available: self.inner.slots(),
        })?;
        self.total_written.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Instantly triggers a flush of all pending audio currently written up to this point.
    ///
    /// Any audio pushed AFTER this flush call will be preserved and played normally.
    #[inline]
    pub fn flush(&mut self) {
        let current_written = self.total_written.load(Ordering::Acquire);
        self.discard_target
            .fetch_max(current_written, Ordering::SeqCst);
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
    total_read: Arc<AtomicU64>,
    discard_target: Arc<AtomicU64>,
    capacity: usize,
}

impl AudioConsumer {
    /// Internal helper to drain samples marked for discard during barge-in.
    #[inline]
    fn apply_discard(&mut self) {
        let discard_target = self.discard_target.load(Ordering::Acquire);
        let current_read = self.total_read.load(Ordering::Acquire);

        if discard_target > current_read {
            let needed = (discard_target - current_read) as usize;
            let available = self.inner.slots();
            let to_discard = needed.min(available);

            if to_discard > 0 {
                if let Ok(chunk) = self.inner.read_chunk(to_discard) {
                    chunk.commit_all();
                    self.total_read
                        .fetch_add(to_discard as u64, Ordering::Release);
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
        self.apply_discard();
        self.inner.slots()
    }

    /// Returns true if no audio samples are currently buffered.
    pub fn is_empty(&mut self) -> bool {
        self.slots() == 0
    }

    /// Monotonic count of total samples read or discarded by this consumer.
    #[inline]
    pub fn total_read(&self) -> u64 {
        self.total_read.load(Ordering::Acquire)
    }

    /// Pops a single sample if available. Returns None if empty or flushed.
    pub fn pop(&mut self) -> Option<f32> {
        self.apply_discard();
        if let Ok(sample) = self.inner.pop() {
            self.total_read.fetch_add(1, Ordering::Release);
            Some(sample)
        } else {
            None
        }
    }

    /// Reads samples into `output` using zero-copy / SIMD memory copy.
    ///
    /// If fewer samples are available than `output.len()`, the remainder of `output`
    /// is zeroed with silence (0.0). Returns the number of actual audio samples read.
    pub fn pop_slice(&mut self, output: &mut [f32]) -> usize {
        self.apply_discard();

        let (read_slice, unread_slice) = self.inner.pop_partial_slice(output);
        let count = read_slice.len();

        if count > 0 {
            self.total_read.fetch_add(count as u64, Ordering::Release);
        }

        // Fill remainder with silence
        unread_slice.fill(0.0);

        count
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

    #[test]
    fn test_ring_buffer_preserves_audio_after_flush() {
        let (mut producer, mut consumer) = AudioRingBuffer::create(1024);
        producer.push_slice(&[1.0; 100]).expect("push old");
        assert_eq!(consumer.slots(), 100);
        assert_eq!(producer.buffered_samples(), 100);

        // Interruption / flush
        producer.flush();
        assert_eq!(producer.buffered_samples(), 0);

        // Push new utterance immediately
        producer.push_slice(&[2.0; 50]).expect("push new");
        assert_eq!(producer.buffered_samples(), 50);

        // Consumer reads: old 100 should be discarded, new 50 should be read!
        assert_eq!(consumer.slots(), 50);
        let mut out = [0.0f32; 50];
        let read = consumer.pop_slice(&mut out);
        assert_eq!(read, 50);
        assert_eq!(out, [2.0f32; 50]);
        assert_eq!(producer.buffered_samples(), 0);
    }

    #[test]
    fn test_ring_buffer_accounting_during_read() {
        let (mut producer, mut consumer) = AudioRingBuffer::create(1024);
        producer.push_slice(&[0.5; 300]).unwrap();
        assert_eq!(producer.buffered_samples(), 300);

        let mut out = [0.0; 100];
        let read = consumer.pop_slice(&mut out);
        assert_eq!(read, 100);
        assert_eq!(producer.buffered_samples(), 200);

        producer.flush();
        assert_eq!(producer.buffered_samples(), 0);
        assert_eq!(consumer.slots(), 0);
    }
}
