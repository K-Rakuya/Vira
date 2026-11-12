use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use vira_domain::viseme::VisemeWeights;
use vira_ports::tts::TimedViseme;

/// Monotonic master audio clock tracking actual audio samples rendered to sound hardware.
///
/// Designed as the high-precision playhead clock for 60Hz Viseme lip-sync and audio-visual synchronization.
/// Uses `Arc<AtomicU64>` so multiple components (audio callback, animation renderer, barge-in controller)
/// can share, inspect, and advance the sample clock lock-free and thread-safely.
#[derive(Debug, Clone)]
pub struct MasterAudioClock {
    sample_rate: u32,
    samples_rendered: Arc<AtomicU64>,
}

impl MasterAudioClock {
    /// Creates a new `MasterAudioClock` configured with the specified sample rate in Hz.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            samples_rendered: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Returns the configured sample rate in Hz.
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Monotonically reads the current sample count rendered to the hardware playhead.
    #[inline]
    pub fn current_sample(&self) -> u64 {
        self.samples_rendered.load(Ordering::Acquire)
    }

    /// Computes the current playback time in milliseconds based on the sample rate.
    #[inline]
    pub fn current_time_ms(&self) -> u64 {
        if self.sample_rate == 0 {
            0
        } else {
            (self.current_sample() as u128 * 1000 / self.sample_rate as u128) as u64
        }
    }

    /// Computes the current playback time in seconds (float).
    #[inline]
    pub fn current_time_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.current_sample() as f64 / self.sample_rate as f64
        }
    }

    /// Advances the playhead clock by `count` samples.
    ///
    /// Called directly by the OS audio output callback as PCM buffers are flushed to the DAC.
    #[inline]
    pub fn advance_samples(&self, count: u64) {
        self.samples_rendered.fetch_add(count, Ordering::Release);
    }

    /// Resets the playhead clock back to 0.
    #[inline]
    pub fn reset(&self) {
        self.samples_rendered.store(0, Ordering::Release);
    }

    /// Explicitly sets the sample playhead to a target sample index.
    #[inline]
    pub fn set_sample(&self, sample: u64) {
        self.samples_rendered.store(sample, Ordering::Release);
    }

    /// Converts milliseconds to sample count at this clock's sample rate.
    #[inline]
    pub fn ms_to_samples(&self, ms: u64) -> u64 {
        (ms as u128 * self.sample_rate as u128 / 1000) as u64
    }

    /// Converts sample count to milliseconds at this clock's sample rate.
    #[inline]
    pub fn samples_to_ms(&self, samples: u64) -> u64 {
        if self.sample_rate == 0 {
            0
        } else {
            (samples as u128 * 1000 / self.sample_rate as u128) as u64
        }
    }

    /// Calculates the current video/animation frame index for a given target frame rate (e.g. 60.0 FPS).
    #[inline]
    pub fn current_frame(&self, fps: f32) -> u64 {
        if fps <= 0.0 {
            0
        } else {
            (self.current_time_seconds() * fps as f64).floor() as u64
        }
    }

    /// Samples and linearly interpolates the active viseme blendshapes at the current playhead time.
    ///
    /// Enables 60Hz Viseme lip-sync aligned precisely to the audio speaker output.
    pub fn sample_viseme(&self, visemes: &[TimedViseme]) -> VisemeWeights {
        if visemes.is_empty() {
            return VisemeWeights::zero();
        }

        let now_ms = self.current_time_ms();

        // Before first keyframe
        if now_ms <= visemes[0].timestamp_ms {
            return visemes[0].weights;
        }

        // At or beyond last keyframe
        if now_ms >= visemes[visemes.len() - 1].timestamp_ms {
            return visemes[visemes.len() - 1].weights;
        }

        // Binary search to find keyframe bracket [i, i+1]
        let idx = match visemes.binary_search_by_key(&now_ms, |v| v.timestamp_ms) {
            Ok(exact) => return visemes[exact].weights,
            Err(insertion_point) => insertion_point - 1,
        };

        let v0 = &visemes[idx];
        let v1 = &visemes[idx + 1];

        let span = v1.timestamp_ms.saturating_sub(v0.timestamp_ms);
        if span == 0 {
            return v0.weights;
        }

        let elapsed = now_ms.saturating_sub(v0.timestamp_ms);
        let alpha = (elapsed as f32 / span as f32).clamp(0.0, 1.0);

        v0.weights.lerp(&v1.weights, alpha)
    }
}

impl Default for MasterAudioClock {
    fn default() -> Self {
        Self::new(48000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_conversion_and_math() {
        let clock = MasterAudioClock::new(48000);
        assert_eq!(clock.sample_rate(), 48000);
        assert_eq!(clock.current_sample(), 0);
        assert_eq!(clock.current_time_ms(), 0);
        assert_eq!(clock.current_time_seconds(), 0.0);

        assert_eq!(clock.ms_to_samples(1000), 48000);
        assert_eq!(clock.ms_to_samples(500), 24000);
        assert_eq!(clock.samples_to_ms(48000), 1000);
        assert_eq!(clock.samples_to_ms(24000), 500);

        clock.advance_samples(24000);
        assert_eq!(clock.current_sample(), 24000);
        assert_eq!(clock.current_time_ms(), 500);
        assert!((clock.current_time_seconds() - 0.5).abs() < f64::EPSILON);
        assert_eq!(clock.current_frame(60.0), 30);

        clock.set_sample(96000);
        assert_eq!(clock.current_sample(), 96000);
        assert_eq!(clock.current_time_ms(), 2000);
        assert_eq!(clock.current_frame(60.0), 120);

        clock.reset();
        assert_eq!(clock.current_sample(), 0);
        assert_eq!(clock.current_time_ms(), 0);
    }

    #[test]
    fn test_clock_viseme_sampling_and_interpolation() {
        let clock = MasterAudioClock::new(48000);

        // Empty visemes
        assert_eq!(clock.sample_viseme(&[]), VisemeWeights::zero());

        let visemes = vec![
            TimedViseme::new(0, VisemeWeights::new(0.0, 0.0, 0.0, 0.0, 0.0)),
            TimedViseme::new(100, VisemeWeights::new(1.0, 0.0, 0.0, 0.0, 0.0)),
            TimedViseme::new(200, VisemeWeights::new(0.0, 1.0, 0.0, 0.0, 0.0)),
        ];

        // At 0ms -> exactly keyframe 0
        assert_eq!(clock.sample_viseme(&visemes).aa, 0.0);

        // At 50ms -> midpoint between 0ms and 100ms
        clock.set_sample(clock.ms_to_samples(50));
        let sample_50 = clock.sample_viseme(&visemes);
        assert!((sample_50.aa - 0.5).abs() < 1e-4);

        // At 100ms -> exact keyframe 1
        clock.set_sample(clock.ms_to_samples(100));
        let sample_100 = clock.sample_viseme(&visemes);
        assert!((sample_100.aa - 1.0).abs() < 1e-4);

        // At 150ms -> midpoint between keyframe 1 and 2
        clock.set_sample(clock.ms_to_samples(150));
        let sample_150 = clock.sample_viseme(&visemes);
        assert!((sample_150.aa - 0.5).abs() < 1e-4);
        assert!((sample_150.ih - 0.5).abs() < 1e-4);

        // Beyond 200ms -> clamp to last keyframe
        clock.set_sample(clock.ms_to_samples(300));
        let sample_300 = clock.sample_viseme(&visemes);
        assert_eq!(sample_300.ih, 1.0);
    }
}
