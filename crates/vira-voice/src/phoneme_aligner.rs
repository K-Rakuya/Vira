//! Phoneme Alignment and G2P (Grapheme-to-Phoneme) Engine.
//!
//! Provides:
//! - Multilingual G2P conversion (English ARPAbet/IPA + Vietnamese syllable decomposition).
//! - Monotonic CTC acoustic frame alignment (Wav2Vec2 / CTC acoustic model output).
//! - Dynamic Time Warping (DTW) sequence alignment between target text phonemes
//!   and recognized microphone phonemes.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Supported languages for G2P phonetic transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PhonemeLanguage {
    #[default]
    English,
    Vietnamese,
    Auto,
}

/// Errors occurring during phoneme alignment.
#[derive(Debug, Error, PartialEq)]
pub enum AlignmentError {
    #[error("Target phoneme sequence is empty")]
    EmptyTargetSequence,
    #[error("Acoustic frame sequence is empty")]
    EmptyAcousticFrames,
    #[error("Frame count ({frames}) is less than target phoneme count ({phonemes})")]
    InsufficientFrames { frames: usize, phonemes: usize },
    #[error("Vocabulary does not contain phoneme: {0}")]
    PhonemeNotFoundInVocab(String),
}

/// Result of aligning a target phoneme against acoustic frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonemeFrameAlignment {
    /// Target phoneme symbol.
    pub phoneme: String,
    /// Starting frame index in acoustic frame sequence (inclusive).
    pub start_frame: usize,
    /// Ending frame index in acoustic frame sequence (inclusive).
    pub end_frame: usize,
    /// Start timestamp in milliseconds.
    pub start_ms: u32,
    /// End timestamp in milliseconds.
    pub end_ms: u32,
    /// Posterior probability or log-posterior values across the aligned frames.
    pub frame_log_posteriors: Vec<f32>,
    /// Arithmetic mean of log-posterior across the aligned segment.
    pub mean_log_posterior: f32,
}

/// Monotonic CTC acoustic aligner.
///
/// Uses dynamic programming (Viterbi decoding) to monotonically segment acoustic frames
/// into target phoneme intervals without reordering or skipping.
#[derive(Debug, Clone)]
pub struct CtcAligner {
    pub frame_duration_ms: u32,
}

impl Default for CtcAligner {
    fn default() -> Self {
        Self {
            frame_duration_ms: 20,
        }
    }
}

impl CtcAligner {
    pub fn new(frame_duration_ms: u32) -> Self {
        Self { frame_duration_ms }
    }

    /// Monotonically aligns a target phoneme sequence against a matrix of frame log-posteriors
    /// [num_frames][vocab_size].
    pub fn align_log_posteriors(
        &self,
        target_phonemes: &[String],
        frame_log_posteriors: &[Vec<f32>],
        vocab: &[String],
    ) -> Result<Vec<PhonemeFrameAlignment>, AlignmentError> {
        let num_phonemes = target_phonemes.len();
        if num_phonemes == 0 {
            return Err(AlignmentError::EmptyTargetSequence);
        }

        let num_frames = frame_log_posteriors.len();
        if num_frames == 0 {
            return Err(AlignmentError::EmptyAcousticFrames);
        }

        if num_frames < num_phonemes {
            return Err(AlignmentError::InsufficientFrames {
                frames: num_frames,
                phonemes: num_phonemes,
            });
        }

        // Map vocabulary strings to indices
        let vocab_map: HashMap<&str, usize> = vocab
            .iter()
            .enumerate()
            .map(|(idx, s)| (s.as_str(), idx))
            .collect();

        let mut target_indices = Vec::with_capacity(num_phonemes);
        for ph in target_phonemes {
            if let Some(&idx) = vocab_map.get(ph.as_str()) {
                target_indices.push(idx);
            } else {
                return Err(AlignmentError::PhonemeNotFoundInVocab(ph.clone()));
            }
        }

        // Viterbi forward table: dp[t][k] is max cumulative log-posterior for aligning frames 0..=t to phonemes 0..=k
        // backpointer[t][k] stores true if transition came from k-1 (advance), false if from k (stay)
        const NEG_INF: f32 = -1e9;
        let mut dp = vec![vec![NEG_INF; num_phonemes]; num_frames];
        let mut backpointer = vec![vec![false; num_phonemes]; num_frames];

        // Frame 0 initialization: can only be at phoneme 0
        let p0_idx = target_indices[0];
        dp[0][0] = frame_log_posteriors[0].get(p0_idx).copied().unwrap_or(NEG_INF);

        // Fill dynamic programming matrix
        for t in 1..num_frames {
            for k in 0..num_phonemes {
                // Must have at least k frames before frame t
                if t < k {
                    continue;
                }
                // Option 1: stay at phoneme k
                let stay_score = dp[t - 1][k];

                // Option 2: advance from phoneme k - 1
                let (best_prev, advanced) = if k > 0 && dp[t - 1][k - 1] > stay_score {
                    (dp[t - 1][k - 1], true)
                } else {
                    (stay_score, false)
                };

                if best_prev > NEG_INF / 2.0 {
                    let phoneme_vocab_idx = target_indices[k];
                    let frame_score = frame_log_posteriors[t]
                        .get(phoneme_vocab_idx)
                        .copied()
                        .unwrap_or(NEG_INF);

                    dp[t][k] = best_prev + frame_score;
                    backpointer[t][k] = advanced;
                }
            }
        }

        // Backtrack optimal alignment path from frame num_frames - 1 at phoneme num_phonemes - 1
        let mut frame_to_phoneme = vec![0usize; num_frames];
        let mut curr_k = num_phonemes - 1;

        for t in (0..num_frames).rev() {
            frame_to_phoneme[t] = curr_k;
            if t > 0 && backpointer[t][curr_k] {
                curr_k = curr_k.saturating_sub(1);
            }
        }

        // Aggregate frame intervals per phoneme
        let mut alignments = Vec::with_capacity(num_phonemes);
        let mut k = 0;
        let mut t = 0;

        while k < num_phonemes {
            let start_frame = t;
            let mut end_frame = t;
            let mut posteriors = Vec::new();

            while t < num_frames && frame_to_phoneme[t] == k {
                let p_idx = target_indices[k];
                let lp = frame_log_posteriors[t].get(p_idx).copied().unwrap_or(-10.0);
                posteriors.push(lp);
                end_frame = t;
                t += 1;
            }

            // Fallback safety if empty
            if posteriors.is_empty() {
                let p_idx = target_indices[k];
                let lp = frame_log_posteriors[start_frame.min(num_frames - 1)]
                    .get(p_idx)
                    .copied()
                    .unwrap_or(-10.0);
                posteriors.push(lp);
            }

            let mean_lp = posteriors.iter().sum::<f32>() / (posteriors.len() as f32);
            let start_ms = (start_frame as u32) * self.frame_duration_ms;
            let end_ms = ((end_frame + 1) as u32) * self.frame_duration_ms;

            alignments.push(PhonemeFrameAlignment {
                phoneme: target_phonemes[k].clone(),
                start_frame,
                end_frame,
                start_ms,
                end_ms,
                frame_log_posteriors: posteriors,
                mean_log_posterior: mean_lp,
            });

            k += 1;
        }

        Ok(alignments)
    }
}

/// Status of phoneme match in DTW sequence alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchStatus {
    /// Phoneme was pronounced correctly and matched.
    ExactMatch,
    /// Phoneme was substituted with a different phoneme.
    Substitution,
    /// Extra unrecognized phoneme was inserted in speech.
    Insertion,
    /// Expected phoneme was completely omitted/deleted.
    Deletion,
}

/// Paired phoneme match detail from sequence alignment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentMatch {
    pub target: Option<String>,
    pub recognized: Option<String>,
    pub status: MatchStatus,
    pub cost: f32,
}

/// Overall results of Dynamic Time Warping alignment between target and recognized sequences.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DtwResult {
    pub matches: Vec<AlignmentMatch>,
    pub total_cost: f32,
    pub normalized_distance: f32,
    pub accuracy_percent: f32,
    pub correct_count: usize,
    pub substitutions_count: usize,
    pub insertions_count: usize,
    pub deletions_count: usize,
}

/// Dynamic Time Warping (DTW) sequence aligner.
#[derive(Debug, Clone, Default)]
pub struct DtwAligner;

impl DtwAligner {
    pub fn new() -> Self {
        Self
    }

    /// Computes distance between two phonemes taking phonetic similarity into account.
    pub fn phoneme_distance(p1: &str, p2: &str) -> f32 {
        if p1.eq_ignore_ascii_case(p2) {
            return 0.0;
        }

        let is_vowel = |p: &str| -> bool {
            let u = p.trim_end_matches(|c: char| c.is_ascii_digit()).to_uppercase();
            matches!(
                u.as_str(),
                "AA" | "AE" | "AH" | "AO" | "AW" | "AY" | "EH" | "ER" | "EY" | "IH" | "IY"
                    | "OW" | "OY" | "UH" | "UW" | "A" | "AW_OPEN" | "AA_HAT" | "E" | "EE"
                    | "I" | "O" | "OW_HORN" | "U" | "UW_HORN" | "IE" | "UO" | "UOW"
            )
        };

        let v1 = is_vowel(p1);
        let v2 = is_vowel(p2);

        if v1 && v2 {
            // Both are vowels: partial confusion penalty
            0.4
        } else if !v1 && !v2 {
            // Both are consonants: partial confusion penalty
            0.6
        } else {
            // Cross vowel/consonant substitution: severe penalty
            1.0
        }
    }

    /// Aligns expected target phonemes with recognized phonemes using Dynamic Time Warping.
    pub fn align(&self, expected: &[String], recognized: &[String]) -> DtwResult {
        let m = expected.len();
        let n = recognized.len();

        if m == 0 && n == 0 {
            return DtwResult {
                matches: Vec::new(),
                total_cost: 0.0,
                normalized_distance: 0.0,
                accuracy_percent: 100.0,
                correct_count: 0,
                substitutions_count: 0,
                insertions_count: 0,
                deletions_count: 0,
            };
        }

        if m == 0 {
            let matches: Vec<AlignmentMatch> = recognized
                .iter()
                .map(|r| AlignmentMatch {
                    target: None,
                    recognized: Some(r.clone()),
                    status: MatchStatus::Insertion,
                    cost: 1.0,
                })
                .collect();
            return DtwResult {
                matches,
                total_cost: n as f32,
                normalized_distance: 1.0,
                accuracy_percent: 0.0,
                correct_count: 0,
                substitutions_count: 0,
                insertions_count: n,
                deletions_count: 0,
            };
        }

        if n == 0 {
            let matches: Vec<AlignmentMatch> = expected
                .iter()
                .map(|e| AlignmentMatch {
                    target: Some(e.clone()),
                    recognized: None,
                    status: MatchStatus::Deletion,
                    cost: 1.0,
                })
                .collect();
            return DtwResult {
                matches,
                total_cost: m as f32,
                normalized_distance: 1.0,
                accuracy_percent: 0.0,
                correct_count: 0,
                substitutions_count: 0,
                insertions_count: 0,
                deletions_count: m,
            };
        }

        const INDEL_COST: f32 = 1.0;
        let mut dp = vec![vec![0.0f32; n + 1]; m + 1];

        for (i, row) in dp.iter_mut().enumerate().take(m + 1).skip(1) {
            row[0] = (i as f32) * INDEL_COST;
        }
        for (j, cell) in dp[0].iter_mut().enumerate().take(n + 1).skip(1) {
            *cell = (j as f32) * INDEL_COST;
        }

        for i in 1..=m {
            for j in 1..=n {
                let dist = Self::phoneme_distance(&expected[i - 1], &recognized[j - 1]);
                let match_cost = dp[i - 1][j - 1] + dist;
                let del_cost = dp[i - 1][j] + INDEL_COST;
                let ins_cost = dp[i][j - 1] + INDEL_COST;

                dp[i][j] = match_cost.min(del_cost).min(ins_cost);
            }
        }

        // Backtrack
        let mut matches = Vec::new();
        let mut i = m;
        let mut j = n;

        while i > 0 || j > 0 {
            if i > 0 && j > 0 {
                let dist = Self::phoneme_distance(&expected[i - 1], &recognized[j - 1]);
                let match_cost = dp[i - 1][j - 1] + dist;
                let del_cost = dp[i - 1][j] + INDEL_COST;

                let min_val = dp[i][j];
                if (min_val - match_cost).abs() < 1e-4 {
                    let status = if dist == 0.0 {
                        MatchStatus::ExactMatch
                    } else {
                        MatchStatus::Substitution
                    };
                    matches.push(AlignmentMatch {
                        target: Some(expected[i - 1].clone()),
                        recognized: Some(recognized[j - 1].clone()),
                        status,
                        cost: dist,
                    });
                    i -= 1;
                    j -= 1;
                    continue;
                } else if (min_val - del_cost).abs() < 1e-4 {
                    matches.push(AlignmentMatch {
                        target: Some(expected[i - 1].clone()),
                        recognized: None,
                        status: MatchStatus::Deletion,
                        cost: INDEL_COST,
                    });
                    i -= 1;
                    continue;
                } else {
                    matches.push(AlignmentMatch {
                        target: None,
                        recognized: Some(recognized[j - 1].clone()),
                        status: MatchStatus::Insertion,
                        cost: INDEL_COST,
                    });
                    j -= 1;
                    continue;
                }
            } else if i > 0 {
                matches.push(AlignmentMatch {
                    target: Some(expected[i - 1].clone()),
                    recognized: None,
                    status: MatchStatus::Deletion,
                    cost: INDEL_COST,
                });
                i -= 1;
            } else {
                matches.push(AlignmentMatch {
                    target: None,
                    recognized: Some(recognized[j - 1].clone()),
                    status: MatchStatus::Insertion,
                    cost: INDEL_COST,
                });
                j -= 1;
            }
        }

        matches.reverse();

        let total_cost = dp[m][n];
        let max_len = m.max(n) as f32;
        let normalized_distance = if max_len > 0.0 {
            (total_cost / max_len).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let mut correct_count = 0;
        let mut substitutions_count = 0;
        let mut insertions_count = 0;
        let mut deletions_count = 0;

        for m_item in &matches {
            match m_item.status {
                MatchStatus::ExactMatch => correct_count += 1,
                MatchStatus::Substitution => substitutions_count += 1,
                MatchStatus::Insertion => insertions_count += 1,
                MatchStatus::Deletion => deletions_count += 1,
            }
        }

        let accuracy_percent = ((1.0 - normalized_distance) * 100.0).clamp(0.0, 100.0);

        DtwResult {
            matches,
            total_cost,
            normalized_distance,
            accuracy_percent,
            correct_count,
            substitutions_count,
            insertions_count,
            deletions_count,
        }
    }
}

/// Grapheme-to-Phoneme (G2P) transcription system.
///
/// Converts written words or full phrases into sequences of standard phonetic tokens.
#[derive(Debug, Clone)]
pub struct G2pConverter {
    dictionary: HashMap<String, Vec<String>>,
}

impl Default for G2pConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl G2pConverter {
    pub fn new() -> Self {
        let mut converter = Self {
            dictionary: HashMap::new(),
        };
        converter.populate_english_lexicon();
        converter
    }

    /// Adds or overrides a custom pronunciation in the lexicon.
    pub fn add_custom_word(&mut self, word: &str, phonemes: Vec<String>) {
        self.dictionary.insert(word.trim().to_lowercase(), phonemes);
    }

    /// Automatically detects language from input text (Vietnamese diacritics vs English).
    pub fn detect_language(text: &str) -> PhonemeLanguage {
        for c in text.chars() {
            match c {
                'à' | 'á' | 'ả' | 'ã' | 'ạ' | 'ă' | 'ắ' | 'ằ' | 'ẳ' | 'ẵ' | 'ặ' | 'â' | 'ấ'
                | 'ầ' | 'ẩ' | 'ẫ' | 'ậ' | 'è' | 'é' | 'ẻ' | 'ẽ' | 'ẹ' | 'ê' | 'ế' | 'ề' | 'ể'
                | 'ễ' | 'ệ' | 'ì' | 'í' | 'ỉ' | 'ĩ' | 'ị' | 'ò' | 'ó' | 'ỏ' | 'õ' | 'ọ' | 'ô'
                | 'ố' | 'ồ' | 'ổ' | 'ỗ' | 'ộ' | 'ơ' | 'ớ' | 'ờ' | 'ở' | 'ỡ' | 'ợ' | 'ù' | 'ú'
                | 'ủ' | 'ũ' | 'ụ' | 'ư' | 'ứ' | 'ừ' | 'ử' | 'ữ' | 'ự' | 'ỳ' | 'ý' | 'ỷ' | 'ỹ'
                | 'ỵ' | 'đ' | 'À' | 'Á' | 'Ả' | 'Ã' | 'Ạ' | 'Ă' | 'Ắ' | 'Ằ' | 'Ẳ' | 'Ẵ' | 'Ặ'
                | 'Â' | 'Ấ' | 'Ầ' | 'Ẩ' | 'Ẫ' | 'Ậ' | 'È' | 'É' | 'Ẻ' | 'Ẽ' | 'Ẹ' | 'Ê' | 'Ế'
                | 'Ề' | 'Ể' | 'Ễ' | 'Ệ' | 'Ì' | 'Í' | 'Ỉ' | 'Ĩ' | 'Ị' | 'Ò' | 'Ó' | 'Ỏ' | 'Õ'
                | 'Ọ' | 'Ô' | 'Ố' | 'Ồ' | 'Ổ' | 'Ỗ' | 'Ộ' | 'Ơ' | 'Ớ' | 'Ờ' | 'Ở' | 'Ỡ' | 'Ợ'
                | 'Ù' | 'Ú' | 'Ủ' | 'Ũ' | 'Ụ' | 'Ư' | 'Ứ' | 'Ừ' | 'Ử' | 'Ữ' | 'Ự' | 'Ỳ' | 'Ý'
                | 'Ỷ' | 'Ỹ' | 'Ỵ' | 'Đ' => return PhonemeLanguage::Vietnamese,
                _ => {}
            }
        }
        PhonemeLanguage::English
    }

    /// Converts raw text to a vector of phoneme symbols.
    pub fn text_to_phonemes(&self, text: &str, language: PhonemeLanguage) -> Vec<String> {
        let lang = match language {
            PhonemeLanguage::Auto => Self::detect_language(text),
            other => other,
        };

        let words: Vec<&str> = text
            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|w| !w.is_empty())
            .collect();

        let mut phonemes = Vec::new();

        for word in words {
            match lang {
                PhonemeLanguage::English => {
                    phonemes.extend(self.english_word_to_phonemes(word));
                }
                PhonemeLanguage::Vietnamese => {
                    phonemes.extend(self.vietnamese_word_to_phonemes(word));
                }
                PhonemeLanguage::Auto => unreachable!(),
            }
        }

        phonemes
    }

    /// English word G2P with dictionary lookup and rule-based fallback.
    fn english_word_to_phonemes(&self, word: &str) -> Vec<String> {
        let clean = word.trim().to_lowercase();
        if let Some(list) = self.dictionary.get(&clean) {
            return list.clone();
        }

        // Rule-based fallback for unseen English words
        self.english_rule_based_g2p(&clean)
    }

    /// Rule-based phonetic generator for arbitrary English letter patterns.
    fn english_rule_based_g2p(&self, word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();
        let mut phonemes = Vec::new();
        let mut i = 0;

        while i < len {
            // Check 4-character suffixes
            if i + 4 <= len {
                let slice: String = chars[i..i + 4].iter().collect();
                if slice == "tion" {
                    phonemes.push("SH".to_string());
                    phonemes.push("AH".to_string());
                    phonemes.push("N".to_string());
                    i += 4;
                    continue;
                } else if slice == "sion" {
                    phonemes.push("ZH".to_string());
                    phonemes.push("AH".to_string());
                    phonemes.push("N".to_string());
                    i += 4;
                    continue;
                }
            }

            // Check 3-character sequences
            if i + 3 <= len {
                let slice: String = chars[i..i + 3].iter().collect();
                if slice == "ing" && i > 0 {
                    phonemes.push("IH".to_string());
                    phonemes.push("NG".to_string());
                    i += 3;
                    continue;
                }
            }

            // Check 2-character digraphs
            if i + 2 <= len {
                let slice: String = chars[i..i + 2].iter().collect();
                match slice.as_str() {
                    "th" => {
                        phonemes.push("TH".to_string());
                        i += 2;
                        continue;
                    }
                    "sh" => {
                        phonemes.push("SH".to_string());
                        i += 2;
                        continue;
                    }
                    "ch" => {
                        phonemes.push("CH".to_string());
                        i += 2;
                        continue;
                    }
                    "ph" => {
                        phonemes.push("F".to_string());
                        i += 2;
                        continue;
                    }
                    "wh" => {
                        phonemes.push("W".to_string());
                        i += 2;
                        continue;
                    }
                    "ck" => {
                        phonemes.push("K".to_string());
                        i += 2;
                        continue;
                    }
                    "ng" => {
                        phonemes.push("NG".to_string());
                        i += 2;
                        continue;
                    }
                    "qu" => {
                        phonemes.push("K".to_string());
                        phonemes.push("W".to_string());
                        i += 2;
                        continue;
                    }
                    "ee" | "ea" => {
                        phonemes.push("IY".to_string());
                        i += 2;
                        continue;
                    }
                    "oo" => {
                        phonemes.push("UW".to_string());
                        i += 2;
                        continue;
                    }
                    "ai" | "ay" => {
                        phonemes.push("EY".to_string());
                        i += 2;
                        continue;
                    }
                    "oi" | "oy" => {
                        phonemes.push("OY".to_string());
                        i += 2;
                        continue;
                    }
                    "ou" | "ow" => {
                        phonemes.push("AW".to_string());
                        i += 2;
                        continue;
                    }
                    _ => {}
                }
            }

            // Single letters
            match chars[i] {
                'a' => phonemes.push("AE".to_string()),
                'e' => phonemes.push("EH".to_string()),
                'i' => phonemes.push("IH".to_string()),
                'o' => phonemes.push("AA".to_string()),
                'u' => phonemes.push("AH".to_string()),
                'b' => phonemes.push("B".to_string()),
                'c' => phonemes.push("K".to_string()),
                'd' => phonemes.push("D".to_string()),
                'f' => phonemes.push("F".to_string()),
                'g' => phonemes.push("G".to_string()),
                'h' => phonemes.push("HH".to_string()),
                'j' => phonemes.push("JH".to_string()),
                'k' => phonemes.push("K".to_string()),
                'l' => phonemes.push("L".to_string()),
                'm' => phonemes.push("M".to_string()),
                'n' => phonemes.push("N".to_string()),
                'p' => phonemes.push("P".to_string()),
                'r' => phonemes.push("R".to_string()),
                's' => phonemes.push("S".to_string()),
                't' => phonemes.push("T".to_string()),
                'v' => phonemes.push("V".to_string()),
                'w' => phonemes.push("W".to_string()),
                'x' => {
                    phonemes.push("K".to_string());
                    phonemes.push("S".to_string());
                }
                'y' => phonemes.push("Y".to_string()),
                'z' => phonemes.push("Z".to_string()),
                _ => {}
            }
            i += 1;
        }

        phonemes
    }

    /// Vietnamese syllable decomposition G2P.
    ///
    /// Deconstructs Vietnamese syllables into (initial onset, medial glide, nucleus vowel with tone, coda).
    pub fn vietnamese_word_to_phonemes(&self, word: &str) -> Vec<String> {
        let lower = word.trim().to_lowercase();
        if lower.is_empty() {
            return Vec::new();
        }

        let mut phonemes = Vec::new();
        let chars: Vec<char> = lower.chars().collect();
        let len = chars.len();
        let mut i = 0;

        // 1. Initial onset consonant (phụ âm đầu)
        let mut initial = None;
        if i + 3 <= len {
            let slice: String = chars[i..i + 3].iter().collect();
            if slice == "ngh" {
                initial = Some("NG");
                i += 3;
            }
        }
        if initial.is_none() && i + 2 <= len {
            let slice: String = chars[i..i + 2].iter().collect();
            match slice.as_str() {
                "ng" => {
                    initial = Some("NG");
                    i += 2;
                }
                "nh" => {
                    initial = Some("NH");
                    i += 2;
                }
                "th" => {
                    initial = Some("TH");
                    i += 2;
                }
                "ch" => {
                    initial = Some("CH");
                    i += 2;
                }
                "kh" => {
                    initial = Some("KH");
                    i += 2;
                }
                "ph" => {
                    initial = Some("F");
                    i += 2;
                }
                "tr" => {
                    initial = Some("TR");
                    i += 2;
                }
                "gi" => {
                    initial = Some("Z");
                    i += 2;
                }
                "gh" => {
                    initial = Some("G");
                    i += 2;
                }
                "qu" => {
                    initial = Some("K_W");
                    i += 2;
                }
                _ => {}
            }
        }
        if initial.is_none() && i < len {
            match chars[i] {
                'b' => {
                    initial = Some("B");
                    i += 1;
                }
                'c' | 'k' => {
                    initial = Some("K");
                    i += 1;
                }
                'd' => {
                    initial = Some("Z");
                    i += 1;
                }
                'đ' => {
                    initial = Some("D");
                    i += 1;
                }
                'g' => {
                    initial = Some("G");
                    i += 1;
                }
                'h' => {
                    initial = Some("HH");
                    i += 1;
                }
                'l' => {
                    initial = Some("L");
                    i += 1;
                }
                'm' => {
                    initial = Some("M");
                    i += 1;
                }
                'n' => {
                    initial = Some("N");
                    i += 1;
                }
                'p' => {
                    initial = Some("P");
                    i += 1;
                }
                'r' => {
                    initial = Some("R");
                    i += 1;
                }
                's' | 'x' => {
                    initial = Some("S");
                    i += 1;
                }
                't' => {
                    initial = Some("T");
                    i += 1;
                }
                'v' => {
                    initial = Some("V");
                    i += 1;
                }
                _ => {}
            }
        }

        if let Some(init) = initial {
            if init == "K_W" {
                phonemes.push("K".to_string());
                phonemes.push("W".to_string());
            } else {
                phonemes.push(init.to_string());
            }
        }

        // 2. Medial & Nucleus Vowel & Tone extraction
        while i < len {
            let (vowel_ph, tone, consumed) = match_vietnamese_vowel(&chars[i..]);
            if consumed > 0 {
                let token = if let Some(t) = tone {
                    format!("{vowel_ph}_{t}")
                } else {
                    vowel_ph.to_string()
                };
                phonemes.push(token);
                i += consumed;
            } else {
                // Final consonant (phụ âm cuối) or unknown
                if i + 2 <= len {
                    let coda: String = chars[i..i + 2].iter().collect();
                    match coda.as_str() {
                        "ng" => {
                            phonemes.push("NG".to_string());
                            i += 2;
                            continue;
                        }
                        "nh" => {
                            phonemes.push("NH".to_string());
                            i += 2;
                            continue;
                        }
                        "ch" => {
                            phonemes.push("CH".to_string());
                            i += 2;
                            continue;
                        }
                        _ => {}
                    }
                }
                match chars[i] {
                    'c' => phonemes.push("K".to_string()),
                    'm' => phonemes.push("M".to_string()),
                    'n' => phonemes.push("N".to_string()),
                    'p' => phonemes.push("P".to_string()),
                    't' => phonemes.push("T".to_string()),
                    'y' | 'i' => phonemes.push("IY".to_string()),
                    'u' | 'o' => phonemes.push("UW".to_string()),
                    _ => {}
                }
                i += 1;
            }
        }

        phonemes
    }

    fn populate_english_lexicon(&mut self) {
        let entries: &[(&str, &[&str])] = &[
            ("hello", &["HH", "AH", "L", "OW"]),
            ("hi", &["HH", "AY"]),
            ("hey", &["HH", "EY"]),
            ("good", &["G", "UH", "D"]),
            ("morning", &["M", "AO", "R", "N", "IH", "NG"]),
            ("afternoon", &["AE", "F", "T", "ER", "N", "UW", "N"]),
            ("evening", &["IY", "V", "N", "IH", "NG"]),
            ("night", &["N", "AY", "T"]),
            ("how", &["HH", "AW"]),
            ("are", &["AA", "R"]),
            ("you", &["Y", "UW"]),
            ("fine", &["F", "AY", "N"]),
            ("great", &["G", "R", "EY", "T"]),
            ("thank", &["TH", "AE", "NG", "K"]),
            ("thanks", &["TH", "AE", "NG", "K", "S"]),
            ("what", &["W", "AH", "T"]),
            ("where", &["W", "EH", "R"]),
            ("when", &["W", "EH", "N"]),
            ("why", &["W", "AY"]),
            ("which", &["W", "IH", "CH"]),
            ("who", &["HH", "UW"]),
            ("is", &["IH", "Z"]),
            ("am", &["AE", "M"]),
            ("was", &["W", "AA", "Z"]),
            ("were", &["W", "ER"]),
            ("be", &["B", "IY"]),
            ("have", &["HH", "AE", "V"]),
            ("has", &["HH", "AE", "Z"]),
            ("do", &["D", "UW"]),
            ("english", &["IH", "NG", "G", "L", "IH", "SH"]),
            ("vietnamese", &["V", "IY", "EH", "T", "N", "AA", "M", "IY", "Z"]),
            ("language", &["L", "AE", "NG", "G", "W", "AH", "JH"]),
            ("learn", &["L", "ER", "N"]),
            ("pronunciation", &["P", "R", "OW", "N", "AH", "N", "S", "IY", "EY", "SH", "AH", "N"]),
            ("speak", &["S", "P", "IY", "K"]),
            ("speech", &["S", "P", "IY", "CH"]),
            ("listen", &["L", "IH", "S", "AH", "N"]),
            ("sound", &["S", "AW", "N", "D"]),
            ("voice", &["V", "OY", "S"]),
            ("practice", &["P", "R", "AE", "K", "T", "AH", "S"]),
            ("apple", &["AE", "P", "AH", "L"]),
            ("cat", &["K", "AE", "T"]),
            ("dog", &["D", "AO", "G"]),
            ("water", &["W", "AA", "T", "ER"]),
            ("coffee", &["K", "AA", "F", "IY"]),
            ("tea", &["T", "IY"]),
            ("computer", &["K", "AH", "M", "P", "Y", "UW", "T", "ER"]),
            ("world", &["W", "ER", "L", "D"]),
            ("people", &["P", "IY", "P", "AH", "L"]),
            ("avatar", &["AE", "V", "AH", "T", "AA", "R"]),
            ("companion", &["K", "AH", "M", "P", "AE", "N", "Y", "AH", "N"]),
            ("vira", &["V", "IH", "R", "AA"]),
            ("yes", &["Y", "EH", "S"]),
            ("no", &["N", "OW"]),
            ("please", &["P", "L", "IY", "Z"]),
            ("welcome", &["W", "EH", "L", "K", "AH", "M"]),
            ("friend", &["F", "R", "EH", "N", "D"]),
            ("think", &["TH", "IH", "NG", "K"]),
            ("the", &["DH", "AH"]),
            ("this", &["DH", "IH", "S"]),
            ("that", &["DH", "AE", "T"]),
            ("there", &["DH", "EH", "R"]),
        ];

        for (word, phones) in entries {
            self.dictionary.insert(
                word.to_string(),
                phones.iter().map(|p| p.to_string()).collect(),
            );
        }
    }
}

/// Identifies vowel phoneme and tone tag (T1=ngang, T2=huyền, T3=hỏi, T4=ngã, T5=sắc, T6=nặng).
fn match_vietnamese_vowel(chars: &[char]) -> (&'static str, Option<&'static str>, usize) {
    if chars.is_empty() {
        return ("", None, 0);
    }

    // Check diphthongs / triphthongs
    if chars.len() >= 2 {
        let two: String = chars[..2].iter().collect();
        match two.as_str() {
            "iê" | "yê" | "ia" => return ("IE", None, 2),
            "iế" | "yế" | "ía" => return ("IE", Some("T5"), 2),
            "iề" | "yề" | "ìa" => return ("IE", Some("T2"), 2),
            "iể" | "yể" | "ỉa" => return ("IE", Some("T3"), 2),
            "iễ" | "yễ" | "ĩa" => return ("IE", Some("T4"), 2),
            "iệ" | "yệ" | "ịa" => return ("IE", Some("T6"), 2),

            "uô" | "ua" => return ("UO", None, 2),
            "uố" | "úa" => return ("UO", Some("T5"), 2),
            "uồ" | "ùa" => return ("UO", Some("T2"), 2),
            "uổ" | "ủa" => return ("UO", Some("T3"), 2),
            "uỗ" | "ũa" => return ("UO", Some("T4"), 2),
            "uộ" | "ụa" => return ("UO", Some("T6"), 2),

            "ươ" | "ưa" => return ("UOW", None, 2),
            "ướ" | "ứa" => return ("UOW", Some("T5"), 2),
            "ườ" | "ừa" => return ("UOW", Some("T2"), 2),
            "ưở" | "ửa" => return ("UOW", Some("T3"), 2),
            "ưỡng" | "ữa" => return ("UOW", Some("T4"), 2),
            "ượ" | "ựa" => return ("UOW", Some("T6"), 2),
            _ => {}
        }
    }

    // Single vowels with tones
    let c = chars[0];
    let (ph, tone) = match c {
        'a' => ("AA", None),
        'á' => ("AA", Some("T5")),
        'à' => ("AA", Some("T2")),
        'ả' => ("AA", Some("T3")),
        'ã' => ("AA", Some("T4")),
        'ạ' => ("AA", Some("T6")),

        'ă' => ("AW", None),
        'ắ' => ("AW", Some("T5")),
        'ằ' => ("AW", Some("T2")),
        'ẳ' => ("AW", Some("T3")),
        'ẵ' => ("AW", Some("T4")),
        'ặ' => ("AW", Some("T6")),

        'â' => ("AA_HAT", None),
        'ấ' => ("AA_HAT", Some("T5")),
        'ầ' => ("AA_HAT", Some("T2")),
        'ẩ' => ("AA_HAT", Some("T3")),
        'ẫ' => ("AA_HAT", Some("T4")),
        'ậ' => ("AA_HAT", Some("T6")),

        'e' => ("EH", None),
        'é' => ("EH", Some("T5")),
        'è' => ("EH", Some("T2")),
        'ẻ' => ("EH", Some("T3")),
        'ẽ' => ("EH", Some("T4")),
        'ẹ' => ("EH", Some("T6")),

        'ê' => ("EE", None),
        'ế' => ("EE", Some("T5")),
        'ề' => ("EE", Some("T2")),
        'ể' => ("EE", Some("T3")),
        'ễ' => ("EE", Some("T4")),
        'ệ' => ("EE", Some("T6")),

        'i' | 'y' => ("IY", None),
        'í' | 'ý' => ("IY", Some("T5")),
        'ì' | 'ỳ' => ("IY", Some("T2")),
        'ỉ' | 'ỷ' => ("IY", Some("T3")),
        'ĩ' | 'ỹ' => ("IY", Some("T4")),
        'ị' | 'ỵ' => ("IY", Some("T6")),

        'o' => ("AW_OPEN", None),
        'ó' => ("AW_OPEN", Some("T5")),
        'ò' => ("AW_OPEN", Some("T2")),
        'ỏ' => ("AW_OPEN", Some("T3")),
        'õ' => ("AW_OPEN", Some("T4")),
        'ọ' => ("AW_OPEN", Some("T6")),

        'ô' => ("OW", None),
        'ố' => ("OW", Some("T5")),
        'ồ' => ("OW", Some("T2")),
        'ổ' => ("OW", Some("T3")),
        'ỗ' => ("OW", Some("T4")),
        'ộ' => ("OW", Some("T6")),

        'ơ' => ("OW_HORN", None),
        'ớ' => ("OW_HORN", Some("T5")),
        'ờ' => ("OW_HORN", Some("T2")),
        'ở' => ("OW_HORN", Some("T3")),
        'ỡ' => ("OW_HORN", Some("T4")),
        'ợ' => ("OW_HORN", Some("T6")),

        'u' => ("UW", None),
        'ú' => ("UW", Some("T5")),
        'ù' => ("UW", Some("T2")),
        'ủ' => ("UW", Some("T3")),
        'ũ' => ("UW", Some("T4")),
        'ụ' => ("UW", Some("T6")),

        'ư' => ("UW_HORN", None),
        'ứ' => ("UW_HORN", Some("T5")),
        'ừ' => ("UW_HORN", Some("T2")),
        'ử' => ("UW_HORN", Some("T3")),
        'ữ' => ("UW_HORN", Some("T4")),
        'ự' => ("UW_HORN", Some("T6")),

        _ => return ("", None, 0),
    };

    (ph, tone, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_g2p_english_lexicon_and_fallback() {
        let g2p = G2pConverter::new();

        let hello = g2p.text_to_phonemes("Hello", PhonemeLanguage::English);
        assert_eq!(hello, vec!["HH", "AH", "L", "OW"]);

        let pron = g2p.text_to_phonemes("pronunciation", PhonemeLanguage::English);
        assert!(!pron.is_empty());
        assert_eq!(pron[0], "P");

        // Fallback for unseen word
        let fallback = g2p.text_to_phonemes("sunshine", PhonemeLanguage::English);
        assert!(fallback.contains(&"S".to_string()));
        assert!(fallback.contains(&"SH".to_string()));
    }

    #[test]
    fn test_g2p_vietnamese_syllable() {
        let g2p = G2pConverter::new();

        let phones = g2p.text_to_phonemes("xin chào", PhonemeLanguage::Vietnamese);
        assert!(phones.contains(&"S".to_string()));
        assert!(phones.contains(&"IY".to_string()));
        assert!(phones.contains(&"N".to_string()));
        assert!(phones.contains(&"CH".to_string()));
        assert!(phones.contains(&"AA_T2".to_string()));

        let lang = G2pConverter::detect_language("Chào bạn, bạn có khỏe không?");
        assert_eq!(lang, PhonemeLanguage::Vietnamese);

        let lang_en = G2pConverter::detect_language("Good morning Vira!");
        assert_eq!(lang_en, PhonemeLanguage::English);
    }

    #[test]
    fn test_dtw_alignment_match_and_substitutions() {
        let aligner = DtwAligner::new();
        let target = vec!["HH".to_string(), "EH".to_string(), "L".to_string(), "OW".to_string()];
        let recognized = vec!["HH".to_string(), "AE".to_string(), "L".to_string(), "OW".to_string()];

        let res = aligner.align(&target, &recognized);
        assert_eq!(res.correct_count, 3);
        assert_eq!(res.substitutions_count, 1);
        assert!(res.accuracy_percent > 80.0);
    }

    #[test]
    fn test_ctc_monotonic_alignment() {
        let aligner = CtcAligner::new(20);
        let target = vec!["HH".to_string(), "OW".to_string()];
        let vocab = vec!["HH".to_string(), "OW".to_string(), "AA".to_string()];

        // 4 frames: frames 0, 1 high on HH; frames 2, 3 high on OW
        let frame0 = vec![-0.1, -5.0, -5.0];
        let frame1 = vec![-0.2, -4.0, -5.0];
        let frame2 = vec![-4.0, -0.15, -5.0];
        let frame3 = vec![-5.0, -0.1, -5.0];
        let frames = vec![frame0, frame1, frame2, frame3];

        let alignments = aligner.align_log_posteriors(&target, &frames, &vocab).unwrap();
        assert_eq!(alignments.len(), 2);
        assert_eq!(alignments[0].phoneme, "HH");
        assert_eq!(alignments[0].start_frame, 0);
        assert_eq!(alignments[1].phoneme, "OW");
        assert_eq!(alignments[1].end_frame, 3);
        assert!(alignments[0].mean_log_posterior >= -0.2);
    }
}
