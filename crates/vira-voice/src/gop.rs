//! Goodness of Pronunciation (GOP) Assessment Engine.
//!
//! Provides phoneme-level pronunciation evaluation using acoustic frame posterior
//! probabilities, normalized scoring (0..100), categorization (`Correct`, `MinorFlaw`,
//! `Mispronounced`), and integration with 3D VRM blendshape visemes (`aa`, `ih`, `ou`, `ee`, `oh`)
//! to visually demonstrate correct mouth shapes to language learners.

use std::fmt;
use serde::{Deserialize, Serialize};
use vira_domain::viseme::VisemeWeights;

/// Qualitative accuracy level for a phoneme or complete utterance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccuracyLevel {
    /// Phoneme was pronounced correctly within native target range (score >= correct threshold).
    Correct,
    /// Phoneme has minor acoustic flaw or slight non-native accent (minor_threshold <= score < correct_threshold).
    MinorFlaw,
    /// Phoneme was severely mispronounced or replaced with incorrect sound (score < minor_threshold).
    Mispronounced,
}

/// Splits a phoneme string into its base phonetic token and optional tone/stress tag.
/// Examples:
/// - "AA_T2" -> ("AA", Some("T2"))
/// - "OW_HORN_T5" -> ("OW_HORN", Some("T5"))
/// - "AA1" -> ("AA", Some("1"))
/// - "IY0" -> ("IY", Some("0"))
/// - "TH" -> ("TH", None)
pub fn split_phoneme_base_and_tag(phoneme: &str) -> (&str, Option<&str>) {
    let trimmed = phoneme.trim();
    // Check for Vietnamese tone tag _T1.._T6
    if let Some(pos) = trimmed.rfind("_T") {
        let tag = &trimmed[pos + 1..];
        let base = &trimmed[..pos];
        if !base.is_empty() && tag.len() >= 2 && tag.chars().skip(1).all(|c| c.is_ascii_digit()) {
            return (base, Some(tag));
        }
    }

    // Check for ARPAbet lexical stress digits (0, 1, 2) at end: e.g. AA1, IY0, AW2
    let digits_count = trimmed.chars().rev().take_while(|c| c.is_ascii_digit()).count();
    if digits_count > 0 && digits_count < trimmed.len() {
        let split_idx = trimmed.len() - digits_count;
        let base = &trimmed[..split_idx];
        let tag = &trimmed[split_idx..];
        return (base, Some(tag));
    }

    (trimmed, None)
}

impl AccuracyLevel {
    /// Determines the accuracy level based on numeric score and configured thresholds.
    pub fn from_score(score: f32, correct_thresh: f32, minor_thresh: f32) -> Self {
        if score.is_nan() || score < minor_thresh {
            AccuracyLevel::Mispronounced
        } else if score < correct_thresh {
            AccuracyLevel::MinorFlaw
        } else {
            AccuracyLevel::Correct
        }
    }

    /// Returns true if the pronunciation is considered acceptable (Correct or MinorFlaw).
    pub fn is_passing(&self) -> bool {
        matches!(self, AccuracyLevel::Correct | AccuracyLevel::MinorFlaw)
    }

    /// Returns true if the pronunciation was fully correct.
    pub fn is_correct(&self) -> bool {
        matches!(self, AccuracyLevel::Correct)
    }
}

impl fmt::Display for AccuracyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccuracyLevel::Correct => write!(f, "Correct"),
            AccuracyLevel::MinorFlaw => write!(f, "MinorFlaw"),
            AccuracyLevel::Mispronounced => write!(f, "Mispronounced"),
        }
    }
}

/// Evaluation score and timing interval for a single target phoneme.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonemeScore {
    /// Phonetic token symbol (e.g., "AA", "TH", "S", or IPA "/θ/").
    pub phoneme: String,
    /// Normalized score in range [0.0, 100.0].
    pub score: f32,
    /// Categorized accuracy level.
    pub accuracy: AccuracyLevel,
    /// Start timestamp in milliseconds relative to audio start.
    pub start_ms: u32,
    /// End timestamp in milliseconds relative to audio start.
    pub end_ms: u32,
    /// Viseme mouth blendshape weights demonstrating the correct articulation to the learner.
    pub viseme_hint: Option<VisemeWeights>,
    /// Name of the primary VRM 1.0 viseme preset (e.g. "aa", "ih", "ou", "ee", "oh").
    pub recommended_viseme: Option<String>,
    /// Raw acoustic log posterior probability before normalization, if available.
    pub log_posterior: Option<f32>,
}

/// Backward compatibility alias matching brief specification.
pub type PhonemeResult = PhonemeScore;

impl PhonemeScore {
    /// Constructs a new `PhonemeScore` with default thresholds (80.0 / 50.0).
    pub fn new(phoneme: impl Into<String>, score: f32, start_ms: u32, end_ms: u32) -> Self {
        let phoneme_str = phoneme.into();
        let clamped_score = if score.is_nan() {
            0.0
        } else {
            score.clamp(0.0, 100.0)
        };
        let accuracy = AccuracyLevel::from_score(clamped_score, 80.0, 50.0);
        let viseme = phoneme_to_viseme(&phoneme_str);
        let preset = phoneme_to_vrm_preset(&phoneme_str);

        Self {
            phoneme: phoneme_str,
            score: clamped_score,
            accuracy,
            start_ms,
            end_ms,
            viseme_hint: Some(viseme),
            recommended_viseme: Some(preset.to_string()),
            log_posterior: None,
        }
    }

    /// Sets explicit accuracy level.
    pub fn with_accuracy(mut self, accuracy: AccuracyLevel) -> Self {
        self.accuracy = accuracy;
        self
    }

    /// Sets explicit viseme blendshape weights.
    pub fn with_viseme_hint(mut self, viseme: VisemeWeights) -> Self {
        self.viseme_hint = Some(viseme);
        self
    }

    /// Sets raw log posterior value.
    pub fn with_log_posterior(mut self, log_posterior: f32) -> Self {
        self.log_posterior = Some(log_posterior);
        self
    }

    /// Re-evaluates accuracy using custom thresholds.
    pub fn with_thresholds(mut self, correct_thresh: f32, minor_thresh: f32) -> Self {
        self.accuracy = AccuracyLevel::from_score(self.score, correct_thresh, minor_thresh);
        self
    }

    /// Returns true if this phoneme was marked as mispronounced.
    pub fn is_mispronounced(&self) -> bool {
        self.accuracy == AccuracyLevel::Mispronounced
    }

    /// Duration of the phoneme segment in milliseconds.
    pub fn duration_ms(&self) -> u32 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Comprehensive pronunciation assessment result for an utterance or phrase.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PronunciationAssessment {
    /// Overall utterance score from 0.0 to 100.0.
    pub overall_score: f32,
    /// Utterance-level accuracy classification.
    pub accuracy: AccuracyLevel,
    /// Detailed score and timing for each individual target phoneme.
    pub phoneme_scores: Vec<PhonemeScore>,
    /// Number of mispronounced phonemes detected.
    pub mispronounced_count: usize,
    /// Total count of target phonemes evaluated.
    pub total_phonemes: usize,
    /// Optional fluency score (rhythm, pauses, speaking rate).
    pub fluency_score: Option<f32>,
    /// Optional completeness score (ratio of pronounced words/phonemes).
    pub completeness_score: Option<f32>,
}

/// Backward compatibility alias matching brief specification.
pub type PronunciationScore = PronunciationAssessment;

impl PronunciationAssessment {
    /// Creates a new assessment from a list of phoneme scores.
    pub fn new(phoneme_scores: Vec<PhonemeScore>, correct_thresh: f32, minor_thresh: f32) -> Self {
        let total_phonemes = phoneme_scores.len();
        if total_phonemes == 0 {
            return Self {
                overall_score: 100.0,
                accuracy: AccuracyLevel::Correct,
                phoneme_scores,
                mispronounced_count: 0,
                total_phonemes: 0,
                fluency_score: Some(100.0),
                completeness_score: Some(100.0),
            };
        }

        let mispronounced_count = phoneme_scores
            .iter()
            .filter(|p| p.is_mispronounced())
            .count();

        // Calculate weighted score by duration, using a minimum effective duration of 1ms
        // per phoneme so zero-duration phonemes are never erased from the overall score.
        let total_duration: u32 = phoneme_scores.iter().map(|p| p.duration_ms().max(1)).sum();
        let weighted_sum: f32 = phoneme_scores
            .iter()
            .map(|p| {
                let score = if p.score.is_nan() { 0.0 } else { p.score };
                score * (p.duration_ms().max(1) as f32)
            })
            .sum();

        let raw_score = weighted_sum / (total_duration as f32);
        let overall_score = if raw_score.is_nan() {
            0.0
        } else {
            raw_score.clamp(0.0, 100.0)
        };

        let accuracy = AccuracyLevel::from_score(overall_score, correct_thresh, minor_thresh);

        Self {
            overall_score,
            accuracy,
            phoneme_scores,
            mispronounced_count,
            total_phonemes,
            fluency_score: None,
            completeness_score: None,
        }
    }

    /// Checks if overall score meets or exceeds a given passing threshold.
    pub fn is_passing(&self, threshold: f32) -> bool {
        self.overall_score >= threshold
    }

    /// Returns references to all phonemes that were mispronounced.
    pub fn mispronounced_phonemes(&self) -> Vec<&PhonemeScore> {
        self.phoneme_scores
            .iter()
            .filter(|p| p.is_mispronounced())
            .collect()
    }

    /// Collects viseme blendshape hints for all mispronounced phonemes so the VRM avatar
    /// can animate them in tutorial feedback mode.
    pub fn viseme_hints_for_mispronounced(&self) -> Vec<VisemeWeights> {
        self.phoneme_scores
            .iter()
            .filter(|p| p.is_mispronounced())
            .filter_map(|p| p.viseme_hint)
            .collect()
    }

    /// Collects viseme blendshape hints across all target phonemes in sequence.
    pub fn all_viseme_hints(&self) -> Vec<VisemeWeights> {
        self.phoneme_scores
            .iter()
            .filter_map(|p| p.viseme_hint)
            .collect()
    }
}

/// Assessment result for a single word in an utterance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordAssessment {
    /// Target or recognized word string.
    pub word: String,
    /// Accuracy score for this word in range [0.0, 100.0].
    pub accuracy_score: f32,
    /// Error category if any: `None` (correct), "Mispronounced", "Omission", "Insertion".
    pub error_type: Option<String>,
    /// Phone-level scores and alignments for this word.
    pub phonemes: Vec<PhonemeScore>,
}

impl WordAssessment {
    /// Creates a new `WordAssessment`.
    pub fn new(
        word: impl Into<String>,
        accuracy_score: f32,
        error_type: Option<String>,
        phonemes: Vec<PhonemeScore>,
    ) -> Self {
        Self {
            word: word.into(),
            accuracy_score: if accuracy_score.is_nan() {
                0.0
            } else {
                accuracy_score.clamp(0.0, 100.0)
            },
            error_type,
            phonemes,
        }
    }

    /// Returns true if this word was pronounced correctly.
    pub fn is_correct(&self) -> bool {
        self.error_type.is_none() || self.error_type.as_deref() == Some("None")
    }

    /// Returns true if this word was mispronounced.
    pub fn is_mispronounced(&self) -> bool {
        self.error_type.as_deref() == Some("Mispronounced")
    }

    /// Returns true if this word was omitted.
    pub fn is_omission(&self) -> bool {
        self.error_type.as_deref() == Some("Omission")
    }

    /// Returns true if this word was inserted.
    pub fn is_insertion(&self) -> bool {
        self.error_type.as_deref() == Some("Insertion")
    }
}

/// Fluency metrics measuring speech tempo, pauses, and hesitations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FluencyMetrics {
    /// Speaking rate in words per minute (WPM).
    pub words_per_minute: f32,
    /// Ratio of pause duration (>= 250ms) relative to total audio duration in range [0.0, 1.0].
    pub pause_duration_ratio: f32,
    /// Total count of hesitation intervals (>= 250ms).
    pub hesitation_count: u32,
    /// Overall normalized fluency score in range [0.0, 100.0].
    pub fluency_score: f32,
}

impl FluencyMetrics {
    /// Calculates fluency metrics from word count, total audio duration (ms), and pause intervals (ms).
    pub fn calculate(
        word_count: usize,
        total_duration_ms: u32,
        pause_durations_ms: &[u32],
    ) -> Self {
        if total_duration_ms == 0 || word_count == 0 {
            return Self {
                words_per_minute: 0.0,
                pause_duration_ratio: 0.0,
                hesitation_count: 0,
                fluency_score: 0.0,
            };
        }

        let duration_sec = total_duration_ms as f32 / 1000.0;
        let words_per_minute = ((word_count as f32 / duration_sec) * 60.0).max(0.0);

        // Standard pauses defined as intervals >= 250ms
        let pause_threshold_ms = 250;
        let significant_pauses: Vec<u32> = pause_durations_ms
            .iter()
            .copied()
            .filter(|&p| p >= pause_threshold_ms)
            .collect();

        let total_pause_ms: u32 = significant_pauses.iter().sum();
        let pause_duration_ratio =
            (total_pause_ms as f32 / total_duration_ms as f32).clamp(0.0, 1.0);
        let hesitation_count = significant_pauses.len() as u32;

        // Native speaking tempo is 110 - 150 WPM (allow up to 160)
        let wpm_score = if words_per_minute >= 110.0 && words_per_minute <= 160.0 {
            100.0
        } else if words_per_minute < 110.0 {
            (words_per_minute / 110.0) * 100.0
        } else {
            (100.0 - (words_per_minute - 160.0) * 0.5).max(30.0)
        };

        // Penalize pause ratio above 15%
        let pause_penalty = if pause_duration_ratio > 0.15 {
            (pause_duration_ratio - 0.15) * 50.0
        } else {
            0.0
        };

        // Penalize hesitations beyond 2 occurrences
        let hesitation_penalty = if hesitation_count > 2 {
            ((hesitation_count - 2) as f32) * 5.0
        } else {
            0.0
        };

        let raw_fluency = wpm_score - pause_penalty - hesitation_penalty;
        let fluency_score = if raw_fluency.is_nan() {
            0.0
        } else {
            raw_fluency.clamp(0.0, 100.0)
        };

        Self {
            words_per_minute,
            pause_duration_ratio,
            hesitation_count,
            fluency_score,
        }
    }
}

/// Maps overall pronunciation score to CEFR level (A1, A2, B1, B2, C1, C2).
pub fn score_to_cefr_level(score: f32) -> &'static str {
    if score.is_nan() || score < 40.0 {
        "A1"
    } else if score < 55.0 {
        "A2"
    } else if score < 70.0 {
        "B1"
    } else if score < 80.0 {
        "B2"
    } else if score < 90.0 {
        "C1"
    } else {
        "C2"
    }
}

/// Comprehensive multi-dimensional pronunciation assessment report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PronunciationReport {
    /// Overall composite score in range [0.0, 100.0].
    pub overall_score: f32,
    /// CEFR language proficiency level: "A1", "A2", "B1", "B2", "C1", "C2".
    pub cefr_level: String,
    /// Acoustic accuracy score (GOP and phoneme match) in range [0.0, 100.0].
    pub accuracy_score: f32,
    /// Fluency score (WPM, pauses, hesitations) in range [0.0, 100.0].
    pub fluency_score: f32,
    /// Completeness score (omissions and insertions) in range [0.0, 100.0].
    pub completeness_score: f32,
    /// Prosody score (intonation, rhythm, energy) in range [0.0, 100.0].
    pub prosody_score: f32,
    /// Detailed assessments for individual words.
    pub words: Vec<WordAssessment>,
    /// Flattened list of individual phoneme scores and timings.
    pub phoneme_scores: Vec<PhonemeScore>,
    /// Actionable instructional feedback tips for the learner.
    pub feedback_tips: Vec<String>,
}

impl PronunciationReport {
    /// Constructs a `PronunciationReport` computing `overall_score` and `cefr_level` automatically.
    ///
    /// Weights:
    /// - Accuracy: 50%
    /// - Fluency: 25%
    /// - Completeness: 15%
    /// - Prosody: 10%
    pub fn new(
        accuracy_score: f32,
        fluency_score: f32,
        completeness_score: f32,
        prosody_score: f32,
        words: Vec<WordAssessment>,
        phoneme_scores: Vec<PhonemeScore>,
        feedback_tips: Vec<String>,
    ) -> Self {
        let acc = if accuracy_score.is_nan() {
            0.0
        } else {
            accuracy_score.clamp(0.0, 100.0)
        };
        let flu = if fluency_score.is_nan() {
            0.0
        } else {
            fluency_score.clamp(0.0, 100.0)
        };
        let comp = if completeness_score.is_nan() {
            0.0
        } else {
            completeness_score.clamp(0.0, 100.0)
        };
        let pro = if prosody_score.is_nan() {
            0.0
        } else {
            prosody_score.clamp(0.0, 100.0)
        };

        let overall_raw = 0.50 * acc + 0.25 * flu + 0.15 * comp + 0.10 * pro;
        let overall_score = if overall_raw.is_nan() {
            0.0
        } else {
            overall_raw.clamp(0.0, 100.0)
        };
        let cefr_level = score_to_cefr_level(overall_score).to_string();

        Self {
            overall_score,
            cefr_level,
            accuracy_score: acc,
            fluency_score: flu,
            completeness_score: comp,
            prosody_score: pro,
            words,
            phoneme_scores,
            feedback_tips,
        }
    }

    /// Checks if overall score meets or exceeds a given passing threshold.
    pub fn is_passing(&self, threshold: f32) -> bool {
        self.overall_score >= threshold
    }
}

/// Configuration options for GOP calculation and scoring thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GopConfig {
    /// Minimum score required for `AccuracyLevel::Correct` (default: 80.0).
    pub correct_threshold: f32,
    /// Minimum score required for `AccuracyLevel::MinorFlaw` (default: 50.0).
    pub minor_flaw_threshold: f32,
    /// Temperature scaling factor for exponential posterior mapping (default: 1.0).
    pub temperature: f32,
    /// Floor for normalized score (default: 0.0).
    pub min_score: f32,
    /// Ceiling for normalized score (default: 100.0).
    pub max_score: f32,
    /// Duration of each acoustic frame in milliseconds (default: 20).
    pub frame_duration_ms: u32,
}

impl GopConfig {
    /// Returns a validated and sanitized copy with legal mathematical bounds.
    pub fn sanitized(&self) -> Self {
        let mut cfg = self.clone();
        if cfg.temperature.is_nan() || cfg.temperature <= 1e-4 {
            cfg.temperature = 1.0;
        }
        if cfg.min_score.is_nan() {
            cfg.min_score = 0.0;
        }
        if cfg.max_score.is_nan() {
            cfg.max_score = 100.0;
        }
        if cfg.min_score > cfg.max_score {
            std::mem::swap(&mut cfg.min_score, &mut cfg.max_score);
        }
        if cfg.minor_flaw_threshold > cfg.correct_threshold {
            std::mem::swap(&mut cfg.minor_flaw_threshold, &mut cfg.correct_threshold);
        }
        if cfg.frame_duration_ms == 0 {
            cfg.frame_duration_ms = 20;
        }
        cfg
    }
}

impl Default for GopConfig {
    fn default() -> Self {
        Self {
            correct_threshold: 80.0,
            minor_flaw_threshold: 50.0,
            temperature: 1.0,
            min_score: 0.0,
            max_score: 100.0,
            frame_duration_ms: 20,
        }
    }
}

/// Core Goodness of Pronunciation (GOP) Evaluator.
///
/// Computes phoneme-level and utterance-level pronunciation quality from acoustic frame
/// posteriors or log-probabilities.
#[derive(Debug, Clone, Default)]
pub struct GopEvaluator {
    pub config: GopConfig,
}

impl GopEvaluator {
    /// Creates a new `GopEvaluator` with custom configuration.
    pub fn new(config: GopConfig) -> Self {
        Self {
            config: config.sanitized(),
        }
    }

    /// Sets custom classification thresholds.
    pub fn with_thresholds(mut self, correct: f32, minor: f32) -> Self {
        self.config.correct_threshold = correct;
        self.config.minor_flaw_threshold = minor;
        self.config = self.config.sanitized();
        self
    }

    /// Calculates a normalized score [0.0, 100.0] from a log posterior probability.
    ///
    /// Mathematical formula:
    /// - For $L = \log P(p \mid O) \le 0$:
    ///   $\text{Score} = \text{clamp}(100.0 \times \exp(L / T), \text{min}, \text{max})$
    /// - When $L = 0.0$ (perfect certainty), score is 100.0.
    /// - When $L = -0.15$, score is $\approx 86.07 \ge 80.0$ (Correct).
    /// - When $L = -0.50$, score is $\approx 60.65$ (MinorFlaw).
    /// - When $L \le -1.0$, score is $\le 36.79$ (Mispronounced).
    pub fn evaluate_score(&self, log_posterior: f32) -> f32 {
        let cfg = self.config.sanitized();
        if log_posterior.is_nan() || log_posterior <= -20.0 {
            return cfg.min_score;
        }
        if log_posterior >= 0.0 {
            return cfg.max_score;
        }

        let prob = (log_posterior / cfg.temperature).exp();
        (prob * 100.0).clamp(cfg.min_score, cfg.max_score)
    }

    /// Calculates normalized score [0.0, 100.0] from a Log-Likelihood Ratio (LLR):
    /// $\text{LLR} = \log P(p \mid O) - \max_{q \neq p} \log P(q \mid O)$.
    ///
    /// Uses standard logistic sigmoid transformation:
    /// $\text{Score} = 100.0 / (1.0 + \exp(-\text{LLR} / T))$.
    pub fn evaluate_llr_score(&self, llr: f32) -> f32 {
        let cfg = self.config.sanitized();
        if llr.is_nan() || llr <= -10.0 {
            return cfg.min_score;
        }
        if llr >= 10.0 {
            return cfg.max_score;
        }

        let sigmoid = 1.0 / (1.0 + (-llr / cfg.temperature).exp());
        (sigmoid * 100.0).clamp(cfg.min_score, cfg.max_score)
    }

    /// Evaluates a target phoneme from a slice of acoustic frame log-posteriors.
    pub fn evaluate_phoneme(
        &self,
        target_phoneme: &str,
        frame_log_posteriors: &[f32],
        start_ms: u32,
        end_ms: u32,
    ) -> PhonemeScore {
        let valid_frames: Vec<f32> = frame_log_posteriors
            .iter()
            .copied()
            .filter(|lp| !lp.is_nan())
            .collect();

        let mean_lp = if valid_frames.is_empty() {
            -10.0
        } else {
            let sum: f32 = valid_frames.iter().sum();
            sum / (valid_frames.len() as f32)
        };

        let score = self.evaluate_score(mean_lp);
        let accuracy = AccuracyLevel::from_score(
            score,
            self.config.correct_threshold,
            self.config.minor_flaw_threshold,
        );
        let viseme = phoneme_to_viseme(target_phoneme);
        let preset = phoneme_to_vrm_preset(target_phoneme);

        PhonemeScore {
            phoneme: target_phoneme.to_string(),
            score,
            accuracy,
            start_ms,
            end_ms,
            viseme_hint: Some(viseme),
            recommended_viseme: Some(preset.to_string()),
            log_posterior: Some(mean_lp),
        }
    }

    /// Evaluates multiple phonemes with their aligned frame posteriors and timestamp ranges.
    pub fn evaluate_phonemes(
        &self,
        items: &[(&str, &[f32], u32, u32)],
    ) -> PronunciationAssessment {
        let scores: Vec<PhonemeScore> = items
            .iter()
            .map(|(ph, frames, s, e)| self.evaluate_phoneme(ph, frames, *s, *e))
            .collect();

        PronunciationAssessment::new(
            scores,
            self.config.correct_threshold,
            self.config.minor_flaw_threshold,
        )
    }

    /// Builds a full assessment from already constructed `PhonemeScore` list.
    pub fn evaluate_assessment(&self, scores: Vec<PhonemeScore>) -> PronunciationAssessment {
        PronunciationAssessment::new(
            scores,
            self.config.correct_threshold,
            self.config.minor_flaw_threshold,
        )
    }

    /// Aligns target words with spoken words to calculate completeness score and identify omissions / insertions.
    pub fn calculate_completeness(
        &self,
        target_words: &[&str],
        spoken_words: &[&str],
    ) -> (f32, Vec<WordAssessment>) {
        let m = target_words.len();
        let n = spoken_words.len();

        if m == 0 && n == 0 {
            return (100.0, Vec::new());
        }

        if m == 0 {
            let words = spoken_words
                .iter()
                .map(|&w| WordAssessment::new(w, 0.0, Some("Insertion".to_string()), vec![]))
                .collect();
            return (0.0, words);
        }

        if n == 0 {
            let words = target_words
                .iter()
                .map(|&w| WordAssessment::new(w, 0.0, Some("Omission".to_string()), vec![]))
                .collect();
            return (0.0, words);
        }

        let normalize = |s: &str| {
            s.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        };

        let norm_targets: Vec<String> = target_words.iter().map(|&w| normalize(w)).collect();
        let norm_spoken: Vec<String> = spoken_words.iter().map(|&w| normalize(w)).collect();

        // Levenshtein / DTW dynamic programming on word tokens
        let mut dp = DtwMatrix::new(m + 1, n + 1);
        const COST_INS: f32 = 1.0;
        const COST_DEL: f32 = 1.0;

        for i in 1..=m {
            dp.set(i, 0, i as f32 * COST_DEL);
        }
        for j in 1..=n {
            dp.set(0, j, j as f32 * COST_INS);
        }

        for i in 1..=m {
            for j in 1..=n {
                let cost_sub = if norm_targets[i - 1] == norm_spoken[j - 1] {
                    0.0
                } else {
                    1.0
                };
                let match_cost = dp.get(i - 1, j - 1) + cost_sub;
                let del_cost = dp.get(i - 1, j) + COST_DEL;
                let ins_cost = dp.get(i, j - 1) + COST_INS;
                dp.set(i, j, match_cost.min(del_cost).min(ins_cost));
            }
        }

        // Backtrack
        let mut aligned_ops = Vec::new();
        let mut i = m;
        let mut j = n;

        while i > 0 || j > 0 {
            if i > 0 && j > 0 {
                let cost_sub = if norm_targets[i - 1] == norm_spoken[j - 1] {
                    0.0
                } else {
                    1.0
                };
                let match_cost = dp.get(i - 1, j - 1) + cost_sub;
                let del_cost = dp.get(i - 1, j) + COST_DEL;
                let cur = dp.get(i, j);

                if (cur - match_cost).abs() < 1e-4 {
                    if cost_sub == 0.0 {
                        aligned_ops.push((Some(i - 1), Some(j - 1), "Match"));
                    } else {
                        aligned_ops.push((Some(i - 1), Some(j - 1), "Sub"));
                    }
                    i -= 1;
                    j -= 1;
                    continue;
                } else if (cur - del_cost).abs() < 1e-4 {
                    aligned_ops.push((Some(i - 1), None, "Del"));
                    i -= 1;
                    continue;
                } else {
                    aligned_ops.push((None, Some(j - 1), "Ins"));
                    j -= 1;
                    continue;
                }
            } else if i > 0 {
                aligned_ops.push((Some(i - 1), None, "Del"));
                i -= 1;
            } else {
                aligned_ops.push((None, Some(j - 1), "Ins"));
                j -= 1;
            }
        }

        aligned_ops.reverse();

        let mut assessments = Vec::with_capacity(aligned_ops.len());
        let mut match_count = 0usize;
        let mut ins_count = 0usize;

        for (t_idx, s_idx, op) in aligned_ops {
            match op {
                "Match" => {
                    match_count += 1;
                    let word = target_words[t_idx.unwrap()].to_string();
                    assessments.push(WordAssessment::new(word, 100.0, None, vec![]));
                }
                "Sub" => {
                    let word = target_words[t_idx.unwrap()].to_string();
                    assessments.push(WordAssessment::new(
                        word,
                        50.0,
                        Some("Mispronounced".to_string()),
                        vec![],
                    ));
                }
                "Del" => {
                    let word = target_words[t_idx.unwrap()].to_string();
                    assessments.push(WordAssessment::new(
                        word,
                        0.0,
                        Some("Omission".to_string()),
                        vec![],
                    ));
                }
                "Ins" => {
                    ins_count += 1;
                    let word = spoken_words[s_idx.unwrap()].to_string();
                    assessments.push(WordAssessment::new(
                        word,
                        0.0,
                        Some("Insertion".to_string()),
                        vec![],
                    ));
                }
                _ => {}
            }
        }

        let base_completeness = (match_count as f32 / m as f32) * 100.0;
        let insertion_penalty = ins_count as f32 * 5.0;
        let completeness = (base_completeness - insertion_penalty).clamp(0.0, 100.0);

        (completeness, assessments)
    }

    /// Evaluates speech prosody from pitch (F0 in Hz) dynamics, energy (RMS) contours,
    /// or phoneme duration distributions.
    pub fn calculate_prosody(
        &self,
        pitch_values: Option<&[f32]>,
        energy_values: Option<&[f32]>,
        phonemes: &[PhonemeScore],
    ) -> f32 {
        let mut scores = Vec::new();

        if let Some(pitch) = pitch_values {
            let voiced: Vec<f32> = pitch
                .iter()
                .copied()
                .filter(|&p| !p.is_nan() && p > 50.0 && p < 500.0)
                .collect();

            if voiced.len() >= 2 {
                let mean: f32 = voiced.iter().sum::<f32>() / voiced.len() as f32;
                let var: f32 = voiced
                    .iter()
                    .map(|&p| (p - mean).powi(2))
                    .sum::<f32>()
                    / voiced.len() as f32;
                let std_dev = var.sqrt();
                let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

                let pitch_score = if cv < 0.05 {
                    (cv / 0.05) * 45.0 + 10.0
                } else if cv <= 0.35 {
                    80.0 + ((cv - 0.08).max(0.0) / (0.35 - 0.08)).min(1.0) * 20.0
                } else {
                    (80.0 - (cv - 0.35) * 60.0).max(40.0)
                };
                scores.push((pitch_score, 0.65));
            }
        }

        if let Some(energy) = energy_values {
            let active: Vec<f32> = energy
                .iter()
                .copied()
                .filter(|&e| !e.is_nan() && e > 0.005)
                .collect();

            if active.len() >= 2 {
                let mean: f32 = active.iter().sum::<f32>() / active.len() as f32;
                let var: f32 = active
                    .iter()
                    .map(|&e| (e - mean).powi(2))
                    .sum::<f32>()
                    / active.len() as f32;
                let std_dev = var.sqrt();
                let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

                let energy_score = (cv / 0.35 * 30.0 + 70.0).clamp(50.0, 100.0);
                scores.push((energy_score, 0.35));
            }
        }

        if scores.is_empty() {
            if phonemes.is_empty() {
                return 0.0;
            }
            let durations: Vec<f32> = phonemes
                .iter()
                .map(|p| p.duration_ms().max(1) as f32)
                .collect();

            let mean: f32 = durations.iter().sum::<f32>() / durations.len() as f32;
            let var: f32 = durations
                .iter()
                .map(|&d| (d - mean).powi(2))
                .sum::<f32>()
                / durations.len() as f32;
            let cv = var.sqrt() / mean;

            return if cv < 0.10 {
                55.0
            } else {
                (80.0 + (cv - 0.10).min(0.40) * 40.0).clamp(60.0, 95.0)
            };
        }

        let total_weight: f32 = scores.iter().map(|(_, w)| w).sum();
        let weighted_score: f32 = scores.iter().map(|(s, w)| s * w).sum();
        (weighted_score / total_weight).clamp(0.0, 100.0)
    }

    /// Evaluates a comprehensive multi-dimensional pronunciation report including
    /// Accuracy, Fluency, Completeness, Prosody, CEFR level, word breakdown, and tips.
    pub fn evaluate_comprehensive_report(
        &self,
        target_text: &str,
        target_words: &[&str],
        word_phonemes: &[(&str, Vec<PhonemeScore>)],
        audio_duration_ms: u32,
        pause_durations_ms: &[u32],
        pitch_values: Option<&[f32]>,
        energy_values: Option<&[f32]>,
    ) -> PronunciationReport {
        let derived_targets: Vec<&str> = if target_words.is_empty() && !target_text.is_empty() {
            target_text.split_whitespace().collect()
        } else {
            target_words.to_vec()
        };

        let spoken_word_names: Vec<&str> = word_phonemes.iter().map(|(w, _)| *w).collect();
        let (completeness_score, aligned_assessments) =
            self.calculate_completeness(&derived_targets, &spoken_word_names);

        let mut wp_map: std::collections::HashMap<String, Vec<PhonemeScore>> =
            std::collections::HashMap::new();
        for (w, phones) in word_phonemes {
            wp_map.insert(w.to_lowercase(), phones.clone());
        }

        let mut words = Vec::with_capacity(derived_targets.len().max(word_phonemes.len()));
        let mut all_phonemes = Vec::new();
        let mut mispronounced_phonemes = Vec::new();
        let mut omitted_words = Vec::new();

        for mut wa in aligned_assessments {
            let key = wa.word.to_lowercase();
            if let Some(phones) = wp_map.remove(&key) {
                if !phones.is_empty() {
                    let total_dur: u32 = phones.iter().map(|p| p.duration_ms().max(1)).sum();
                    let sum_score: f32 = phones
                        .iter()
                        .map(|p| p.score * (p.duration_ms().max(1) as f32))
                        .sum();
                    let word_acc = sum_score / total_dur as f32;
                    wa.accuracy_score = word_acc.clamp(0.0, 100.0);

                    let has_mispronounced = phones.iter().any(|p| p.is_mispronounced());
                    if has_mispronounced {
                        wa.error_type = Some("Mispronounced".to_string());
                        for p in &phones {
                            if p.is_mispronounced() {
                                mispronounced_phonemes.push(p.phoneme.clone());
                            }
                        }
                    }

                    all_phonemes.extend(phones.clone());
                    wa.phonemes = phones;
                }
            } else if wa.error_type.as_deref() == Some("Omission") {
                omitted_words.push(wa.word.clone());
            }
            words.push(wa);
        }

        // Calculate accuracy score across all spoken phonemes
        let accuracy_score = if all_phonemes.is_empty() {
            0.0
        } else {
            let total_dur: u32 = all_phonemes.iter().map(|p| p.duration_ms().max(1)).sum();
            let sum_score: f32 = all_phonemes
                .iter()
                .map(|p| p.score * (p.duration_ms().max(1) as f32))
                .sum();
            (sum_score / total_dur as f32).clamp(0.0, 100.0)
        };

        let fluency = FluencyMetrics::calculate(
            spoken_word_names.len(),
            audio_duration_ms,
            pause_durations_ms,
        );

        let prosody_score = self.calculate_prosody(pitch_values, energy_values, &all_phonemes);

        mispronounced_phonemes.sort();
        mispronounced_phonemes.dedup();

        let mut feedback_tips = Vec::new();
        if !mispronounced_phonemes.is_empty() {
            feedback_tips.push(format!(
                "Cần chú ý phát âm chuẩn các âm: {}",
                mispronounced_phonemes.join(", ")
            ));
        }
        if !omitted_words.is_empty() {
            feedback_tips.push(format!(
                "Các từ bị bỏ sót hoặc chưa rõ: {}",
                omitted_words.join(", ")
            ));
        }
        if fluency.words_per_minute > 0.0 && fluency.words_per_minute < 100.0 {
            feedback_tips.push(format!(
                "Tốc độ nói còn chậm ({:.0} WPM, chuẩn bản ngữ: 110-150 WPM). Hãy tập đọc liền mạch hơn.",
                fluency.words_per_minute
            ));
        }
        if fluency.pause_duration_ratio > 0.25 {
            feedback_tips.push(format!(
                "Tỷ lệ khoảng lặng khá cao ({:.0}%). Cố gắng giảm ngập ngừng giữa các từ.",
                fluency.pause_duration_ratio * 100.0
            ));
        }
        if fluency.hesitation_count > 2 {
            feedback_tips.push(format!(
                "Có {} lần ngập ngừng kéo dài. Cố gắng giữ hơi thở đều.",
                fluency.hesitation_count
            ));
        }
        if prosody_score < 70.0 && prosody_score > 0.0 {
            feedback_tips.push(
                "Ngữ điệu câu còn đều đều (monotone). Hãy nhấn nhá trọng âm và ngữ điệu tự nhiên hơn."
                    .to_string(),
            );
        }

        let overall_raw = 0.50 * accuracy_score
            + 0.25 * fluency.fluency_score
            + 0.15 * completeness_score
            + 0.10 * prosody_score;
        let overall_score = overall_raw.clamp(0.0, 100.0);

        if overall_score >= 90.0 {
            feedback_tips.push(
                "Phát âm xuất sắc, lưu loát và tự nhiên đạt chuẩn bản ngữ (C2)!".to_string(),
            );
        } else if overall_score >= 80.0 {
            feedback_tips.push("Phát âm rất tốt và rõ ràng (C1)! Tiếp tục phát huy.".to_string());
        }

        PronunciationReport::new(
            accuracy_score,
            fluency.fluency_score,
            completeness_score,
            prosody_score,
            words,
            all_phonemes,
            feedback_tips,
        )
    }
}

/// Maps any phoneme string (ARPAbet, IPA, or Vietnamese orthography) to 3D avatar VRM
/// blendshape weights `(aa, ih, ou, ee, oh)` clamped strictly to `[0.0, 1.0]`.
pub fn phoneme_to_viseme(phoneme: &str) -> VisemeWeights {
    let (base_raw, _) = split_phoneme_base_and_tag(phoneme);
    let base = base_raw.trim().to_uppercase();

    match base.as_str() {
        // Open vowels (VRM 'aa')
        "AA" | "AH" | "AE" | "AW" | "AY" | "/A/" | "/Æ/" | "/Ɑ/" | "/Ʌ/" | "/AƱ/" | "/AꞮ/"
        | "A" | "AW_OPEN" | "AA_HAT" => {
            VisemeWeights::new(0.90, 0.05, 0.0, 0.10, 0.05)
        }

        // High front vowels & palatals (VRM 'ih')
        "IY" | "IH" | "Y" | "/I/" | "/Ɪ/" | "/Iː/" | "/J/" | "I" => {
            VisemeWeights::new(0.05, 0.90, 0.0, 0.25, 0.0)
        }

        // High back rounded vowels & labio-velars (VRM 'ou')
        "UW" | "UH" | "W" | "/U/" | "/Ʊ/" | "/Uː/" | "/W/" | "U" | "UW_HORN" => {
            VisemeWeights::new(0.0, 0.0, 0.90, 0.0, 0.20)
        }

        // Mid front vowels (VRM 'ee')
        "EH" | "EY" | "ER" | "/E/" | "/Ɛ/" | "/EꞮ/" | "/Ə/" | "/Ɜː/" | "E" | "EE" | "IE" => {
            VisemeWeights::new(0.10, 0.25, 0.0, 0.85, 0.0)
        }

        // Mid/back rounded vowels (VRM 'oh')
        "AO" | "OW" | "OY" | "/O/" | "/Ɔ/" | "/OƱ/" | "/ƆꞮ/" | "/Ɒ/" | "O" | "OW_HORN"
        | "UO" | "UOW" => {
            VisemeWeights::new(0.10, 0.0, 0.30, 0.0, 0.90)
        }

        // Bilabial consonants (lips closed)
        "B" | "P" | "M" | "/B/" | "/P/" | "/M/" => {
            VisemeWeights::zero()
        }

        // Labiodentals (upper teeth on lower lip)
        "F" | "V" | "/F/" | "/V/" => {
            VisemeWeights::new(0.05, 0.35, 0.0, 0.30, 0.0)
        }

        // Dental fricatives (tongue between teeth)
        "TH" | "DH" | "/Θ/" | "/Ð/" => {
            VisemeWeights::new(0.15, 0.35, 0.0, 0.25, 0.0)
        }

        // Sibilants & postalveolars
        "S" | "Z" | "SH" | "ZH" | "CH" | "JH" | "/S/" | "/Z/" | "/Ʃ/" | "/Ʒ/" | "/TƩ/"
        | "/DƷ/" => {
            VisemeWeights::new(0.05, 0.45, 0.10, 0.35, 0.0)
        }

        // Alveolar stops, nasals, liquids
        "T" | "D" | "N" | "L" | "/T/" | "/D/" | "/N/" | "/L/" => {
            VisemeWeights::new(0.10, 0.30, 0.0, 0.25, 0.0)
        }

        // Velar stops & nasals
        "K" | "G" | "NG" | "KH" | "/K/" | "/Ɡ/" | "/Ŋ/" | "/X/" => {
            VisemeWeights::new(0.25, 0.10, 0.0, 0.15, 0.0)
        }

        // Glottal / Aspirate
        "HH" | "H" | "/H/" => {
            VisemeWeights::new(0.30, 0.10, 0.10, 0.10, 0.10)
        }

        // Rhotic / retroflex
        "R" | "/R/" | "/ɹ/" | "TR" => {
            VisemeWeights::new(0.05, 0.15, 0.40, 0.10, 0.30)
        }

        // Palatal nasal / glide
        "NH" | "/Ɲ/" => {
            VisemeWeights::new(0.10, 0.60, 0.0, 0.30, 0.0)
        }

        // Default neutral mouth shape
        _ => VisemeWeights::zero(),
    }
}

/// Identifies the primary VRM 1.0 expression preset name (`"aa"`, `"ih"`, `"ou"`, `"ee"`, `"oh"`)
/// for a given phoneme token.
pub fn phoneme_to_vrm_preset(phoneme: &str) -> &'static str {
    let viseme = phoneme_to_viseme(phoneme);
    let weights = [
        ("aa", viseme.aa),
        ("ih", viseme.ih),
        ("ou", viseme.ou),
        ("ee", viseme.ee),
        ("oh", viseme.oh),
    ];

    let mut max_name = "aa";
    let mut max_val = -1.0;

    for (name, val) in weights {
        if val > max_val {
            max_val = val;
            max_name = name;
        }
    }

    if max_val <= 0.05 {
        "aa"
    } else {
        max_name
    }
}

/// Contiguous 1D flat buffer for DTW and dynamic programming matrices.
///
/// Replaces disjoint `Vec<Vec<f32>>` with a single contiguous heap allocation
/// indexed by `row * cols + col`. This eliminates multiple heap allocations/deallocations
/// and 2-level pointer chasing, preventing L1 data cache thrashing.
#[derive(Debug, Clone, PartialEq)]
pub struct DtwMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl DtwMatrix {
    /// Creates a new zeroed DTW matrix with dimensions `rows` x `cols`.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0f32; rows * cols],
        }
    }

    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Retrieves element at `(r, c)`.
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    /// Sets element at `(r, c)`.
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f32) {
        self.data[r * self.cols + c] = val;
    }

    /// Returns a slice for row `r`.
    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Returns a mutable slice for row `r`.
    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        let start = r * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// Direct access to underlying contiguous slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Direct mutable access to underlying contiguous slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

/// Dynamic Time Warping (DTW) distance calculation using a rolling 2-row cache.
///
/// Requires only O(N) memory (2 * (N + 1) floats) rather than O(M * N), fitting 100%
/// inside the 32KB L1 data cache to completely avoid cache eviction overhead.
pub fn dtw_distance_rolling<F>(m: usize, n: usize, mut dist_fn: F) -> f32
where
    F: FnMut(usize, usize) -> f32,
{
    if m == 0 && n == 0 {
        return 0.0;
    }
    const INDEL_COST: f32 = 1.0;
    if m == 0 {
        return n as f32 * INDEL_COST;
    }
    if n == 0 {
        return m as f32 * INDEL_COST;
    }

    let cols = n + 1;
    let mut prev_row = vec![0.0f32; cols];
    let mut curr_row = vec![0.0f32; cols];

    for j in 1..=n {
        prev_row[j] = (j as f32) * INDEL_COST;
    }

    for i in 1..=m {
        curr_row[0] = (i as f32) * INDEL_COST;
        for j in 1..=n {
            let dist = dist_fn(i - 1, j - 1);
            let match_cost = prev_row[j - 1] + dist;
            let del_cost = prev_row[j] + INDEL_COST;
            let ins_cost = curr_row[j - 1] + INDEL_COST;
            curr_row[j] = match_cost.min(del_cost).min(ins_cost);
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluator_score_formula() {
        let evaluator = GopEvaluator::default();
        let s0 = evaluator.evaluate_score(0.0);
        assert!((s0 - 100.0).abs() < 1e-4);

        let s1 = evaluator.evaluate_score(-0.15);
        assert!(s1 >= 80.0, "Expected >= 80.0, got {s1}");

        let s2 = evaluator.evaluate_score(-0.5);
        assert!((50.0..80.0).contains(&s2), "Expected MinorFlaw range, got {s2}");

        let s3 = evaluator.evaluate_score(-1.5);
        assert!(s3 < 50.0, "Expected Mispronounced, got {s3}");

        let s_nan = evaluator.evaluate_score(f32::NAN);
        assert_eq!(s_nan, 0.0);

        let s_pos = evaluator.evaluate_score(0.5);
        assert_eq!(s_pos, 100.0);
    }

    #[test]
    fn test_llr_score_formula() {
        let evaluator = GopEvaluator::default();
        let high = evaluator.evaluate_llr_score(3.0);
        assert!(high >= 90.0);

        let zero = evaluator.evaluate_llr_score(0.0);
        assert!((zero - 50.0).abs() < 1e-3);

        let low = evaluator.evaluate_llr_score(-3.0);
        assert!(low <= 10.0);
    }

    #[test]
    fn test_viseme_mapping_coverage() {
        let v_aa = phoneme_to_viseme("AA");
        assert!(v_aa.aa > 0.8);
        assert_eq!(phoneme_to_vrm_preset("AA"), "aa");

        let v_iy = phoneme_to_viseme("IY");
        assert!(v_iy.ih > 0.8);
        assert_eq!(phoneme_to_vrm_preset("IY"), "ih");

        let v_uw = phoneme_to_viseme("UW");
        assert!(v_uw.ou > 0.8);
        assert_eq!(phoneme_to_vrm_preset("UW"), "ou");

        let v_eh = phoneme_to_viseme("EH");
        assert!(v_eh.ee > 0.8);
        assert_eq!(phoneme_to_vrm_preset("EH"), "ee");

        let v_ow = phoneme_to_viseme("OW");
        assert!(v_ow.oh > 0.8);
        assert_eq!(phoneme_to_vrm_preset("OW"), "oh");

        let v_p = phoneme_to_viseme("P");
        assert!(v_p.is_silent());
    }

    #[test]
    fn test_pronunciation_assessment_mispronounced() {
        let evaluator = GopEvaluator::default();
        let p1 = evaluator.evaluate_phoneme("TH", &[-0.05, -0.1], 0, 100);
        assert_eq!(p1.accuracy, AccuracyLevel::Correct);

        let p2 = evaluator.evaluate_phoneme("IY", &[-1.8, -2.1], 100, 200);
        assert_eq!(p2.accuracy, AccuracyLevel::Mispronounced);

        let assess = evaluator.evaluate_assessment(vec![p1, p2]);
        assert_eq!(assess.mispronounced_count, 1);
        assert_eq!(assess.mispronounced_phonemes().len(), 1);
        assert_eq!(assess.mispronounced_phonemes()[0].phoneme, "IY");

        let hints = assess.viseme_hints_for_mispronounced();
        assert_eq!(hints.len(), 1);
        assert!(hints[0].ih > 0.8);
    }

    #[test]
    fn test_dtw_matrix_and_rolling_cache() {
        let mut mat = DtwMatrix::new(3, 4);
        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 4);
        assert_eq!(mat.len(), 12);
        assert!(!mat.is_empty());

        mat.set(1, 2, 4.5);
        assert_eq!(mat.get(1, 2), 4.5);
        assert_eq!(mat.row(1)[2], 4.5);

        mat.row_mut(2)[3] = 9.0;
        assert_eq!(mat.get(2, 3), 9.0);

        // Test rolling distance equivalence
        let cost = dtw_distance_rolling(3, 3, |i, j| if i == j { 0.0 } else { 0.5 });
        assert_eq!(cost, 0.0);

        let cost_diff = dtw_distance_rolling(3, 3, |_i, _j| 0.5);
        assert!((cost_diff - 1.5).abs() < 1e-4);
    }
}
