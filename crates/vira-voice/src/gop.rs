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
}
