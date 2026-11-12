use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use vira_ports::audio::{AudioError, AudioPlaybackPort};

/// Report generated when user barge-in interruption is triggered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BargeInReport {
    /// Playback timestamp in milliseconds when interruption was triggered.
    pub interrupted_at_ms: u64,
    /// Master hardware sample playhead count at interruption.
    pub interrupted_sample: u64,
}

/// Controller responsible for detecting user speech interruption (barge-in),
/// immediately flushing playback audio within 1 callback cycle (<10ms),
/// and cancelling active downstream LLM and TTS tasks.
pub struct BargeInController {
    playback: Arc<dyn AudioPlaybackPort>,
    cancel_token: Mutex<CancellationToken>,
    energy_threshold_bits: AtomicU32,
    enabled: AtomicBool,
}

impl BargeInController {
    /// Creates a new `BargeInController` tied to the given audio playback port.
    pub fn new(playback: Arc<dyn AudioPlaybackPort>) -> Self {
        Self {
            playback,
            cancel_token: Mutex::new(CancellationToken::new()),
            energy_threshold_bits: AtomicU32::new(0.03f32.to_bits()), // default RMS speech threshold
            enabled: AtomicBool::new(true),
        }
    }

    /// Creates a `BargeInController` with an explicit initial cancellation token.
    pub fn with_token(
        playback: Arc<dyn AudioPlaybackPort>,
        token: CancellationToken,
    ) -> Self {
        Self {
            playback,
            cancel_token: Mutex::new(token),
            energy_threshold_bits: AtomicU32::new(0.03f32.to_bits()),
            enabled: AtomicBool::new(true),
        }
    }

    /// Returns a clone of the current active cancellation token.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel_token.lock().unwrap().clone()
    }

    /// Resets the cancellation token to a fresh uncancelled token for the next conversational turn.
    pub fn renew_cancellation_token(&self) -> CancellationToken {
        let mut guard = self.cancel_token.lock().unwrap();
        let new_token = CancellationToken::new();
        *guard = new_token.clone();
        new_token
    }

    /// Sets the RMS audio energy threshold for speech interruption detection.
    /// Completely lock-free.
    pub fn set_energy_threshold(&self, threshold: f32) {
        let clamped = threshold.max(0.0);
        self.energy_threshold_bits
            .store(clamped.to_bits(), Ordering::Release);
    }

    /// Returns the current RMS audio energy threshold.
    /// Completely lock-free.
    pub fn energy_threshold(&self) -> f32 {
        f32::from_bits(self.energy_threshold_bits.load(Ordering::Acquire))
    }

    /// Enables or disables automatic energy-based barge-in triggering.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Returns true if barge-in triggering is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Computes Root-Mean-Square (RMS) amplitude of a PCM frame.
    /// Filters out non-finite samples (NaN / Inf) for numeric robustness.
    #[inline]
    pub fn compute_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut sum_squares = 0.0f32;
        let mut valid_count = 0usize;
        for &s in samples {
            if s.is_finite() {
                sum_squares += s * s;
                valid_count += 1;
            }
        }
        if valid_count == 0 {
            0.0
        } else {
            (sum_squares / valid_count as f32).sqrt()
        }
    }

    /// Checks an incoming captured PCM audio frame against the energy threshold.
    ///
    /// If playback is currently active (`is_playing() == true`) and frame energy exceeds threshold,
    /// triggers barge-in immediately and returns the `BargeInReport`.
    pub fn check_pcm_frame(&self, samples: &[f32]) -> Result<Option<BargeInReport>, AudioError> {
        if !self.is_enabled() {
            return Ok(None);
        }

        if !self.playback.is_playing() {
            return Ok(None);
        }

        let rms = Self::compute_rms(samples);
        let threshold = self.energy_threshold();

        if rms >= threshold {
            let report = self.trigger_barge_in()?;
            Ok(Some(report))
        } else {
            Ok(None)
        }
    }

    /// Manually or externally triggers a barge-in event (e.g. from neural VAD).
    ///
    /// Instantly (<10ms):
    /// 1. Flushes playback buffers to silence sound card output.
    /// 2. Cancels current cancellation token, stopping LLM and TTS tasks.
    /// 3. Re-arms a new cancellation token for the upcoming speech response.
    pub fn trigger_barge_in(&self) -> Result<BargeInReport, AudioError> {
        let interrupted_at_ms = self.playback.current_time_ms();
        let interrupted_sample = self.playback.get_sample_clock();

        // 1. Flush playback audio buffer
        self.playback.flush()?;

        // 2. Cancel running LLM/TTS pipelines and re-arm a new token
        let old_token = {
            let mut guard = self.cancel_token.lock().unwrap();
            let old = guard.clone();
            *guard = CancellationToken::new();
            old
        };
        old_token.cancel();

        Ok(BargeInReport {
            interrupted_at_ms,
            interrupted_sample,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playback::PlaybackEngine;

    #[test]
    fn test_barge_in_trigger_flushes_and_cancels() {
        let (playback, _consumer) = PlaybackEngine::new_virtual(48000, 1024);
        let playback = Arc::new(playback);

        // Enqueue some audio to simulate active speaking
        playback.enqueue_samples(&[0.5; 512]).unwrap();
        assert!(playback.is_playing());
        assert_eq!(playback.buffered_samples(), 512);

        let controller = BargeInController::new(playback.clone());
        let token = controller.cancellation_token();
        assert!(!token.is_cancelled());

        let report = controller.trigger_barge_in().unwrap();
        assert_eq!(report.interrupted_at_ms, 0);

        // Verify token cancelled
        assert!(token.is_cancelled());

        // Verify playback buffer flushed
        assert_eq!(playback.buffered_samples(), 0);
        assert!(!playback.is_playing());

        // New token was re-armed
        let new_token = controller.cancellation_token();
        assert!(!new_token.is_cancelled());
    }

    #[test]
    fn test_barge_in_pcm_energy_detection() {
        let (playback, _consumer) = PlaybackEngine::new_virtual(48000, 1024);
        let playback = Arc::new(playback);
        let controller = BargeInController::new(playback.clone());
        controller.set_energy_threshold(0.05);

        // When playback is idle, even loud speech does NOT trigger barge-in
        let loud_speech = vec![0.5f32; 256];
        let res = controller.check_pcm_frame(&loud_speech).unwrap();
        assert!(res.is_none());

        // When playback is playing:
        playback.enqueue_samples(&[0.1; 512]).unwrap();
        assert!(playback.is_playing());

        // Quiet background noise (< 0.05 RMS) does NOT trigger barge-in
        let quiet_noise = vec![0.01f32; 256];
        let res_quiet = controller.check_pcm_frame(&quiet_noise).unwrap();
        assert!(res_quiet.is_none());
        assert!(playback.is_playing());

        // Loud speech (> 0.05 RMS) DOES trigger barge-in
        let res_loud = controller.check_pcm_frame(&loud_speech).unwrap();
        assert!(res_loud.is_some());
        assert!(!playback.is_playing());
        assert_eq!(playback.buffered_samples(), 0);
    }

    #[test]
    fn test_barge_in_rms_nan_inf_robustness() {
        let non_finite = vec![f32::NAN, f32::INFINITY, 0.5f32, -0.5f32];
        let rms = BargeInController::compute_rms(&non_finite);
        assert!((rms - 0.5).abs() < 1e-4);

        let all_nan = vec![f32::NAN, f32::NAN];
        assert_eq!(BargeInController::compute_rms(&all_nan), 0.0);
        assert_eq!(BargeInController::compute_rms(&[]), 0.0);
    }

    #[test]
    fn test_barge_in_report_serde() {
        let report = BargeInReport {
            interrupted_at_ms: 1234,
            interrupted_sample: 59232,
        };
        let json = serde_json::to_string(&report).expect("serialize report");
        let decoded: BargeInReport = serde_json::from_str(&json).expect("deserialize report");
        assert_eq!(report, decoded);
    }
}
