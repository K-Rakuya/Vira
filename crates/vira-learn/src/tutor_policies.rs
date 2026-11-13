//! Pedagogical tutor policies and CEFR language proficiency modeling.
//!
//! Provides CEFR competency evaluation, communicative recast instruction generation
//! (subtly correcting learner grammar within natural conversational replies without interrupting),
//! and vocabulary/syntax complexity modulation.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;
use vira_domain::message::{Message, Role};
use vira_ports::store::UserProfile;

/// Common European Framework of Reference for Languages (CEFR) proficiency level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CefrLevel {
    /// Breakthrough / Beginner.
    A1,
    /// Waystage / Elementary.
    A2,
    /// Threshold / Intermediate.
    B1,
    /// Vantage / Upper Intermediate.
    B2,
    /// Effective Operational Proficiency / Advanced.
    C1,
    /// Mastery / Native-level Proficiency.
    C2,
}

impl CefrLevel {
    /// Returns standardized uppercase string (e.g. "A1", "B2").
    pub fn as_str(&self) -> &'static str {
        match self {
            CefrLevel::A1 => "A1",
            CefrLevel::A2 => "A2",
            CefrLevel::B1 => "B1",
            CefrLevel::B2 => "B2",
            CefrLevel::C1 => "C1",
            CefrLevel::C2 => "C2",
        }
    }

    /// Returns human-readable proficiency title.
    pub fn title(&self) -> &'static str {
        match self {
            CefrLevel::A1 => "Beginner (Breakthrough)",
            CefrLevel::A2 => "Elementary (Waystage)",
            CefrLevel::B1 => "Intermediate (Threshold)",
            CefrLevel::B2 => "Upper Intermediate (Vantage)",
            CefrLevel::C1 => "Advanced (Effective Proficiency)",
            CefrLevel::C2 => "Mastery (Proficient Native-like)",
        }
    }

    /// Recommended sentence length range (min_words, max_words).
    pub fn target_sentence_length(&self) -> (usize, usize) {
        match self {
            CefrLevel::A1 => (3, 8),
            CefrLevel::A2 => (6, 12),
            CefrLevel::B1 => (10, 18),
            CefrLevel::B2 => (14, 25),
            CefrLevel::C1 => (18, 35),
            CefrLevel::C2 => (20, 50),
        }
    }

    /// Pedagogical vocabulary guidance for this level.
    pub fn vocabulary_profile(&self) -> &'static str {
        match self {
            CefrLevel::A1 => "Use top 500 high-frequency words, concrete everyday objects, basic verbs in present tense.",
            CefrLevel::A2 => "Use top 1500 common words, routine activities, simple comparisons, straightforward adjectives.",
            CefrLevel::B1 => "Use top 3000 words, connected phrases, opinion markers ('in my opinion', 'furthermore'), mixed tenses.",
            CefrLevel::B2 => "Use rich vocabulary (top 5000 words), idiomatic collocations, technical/abstract terms, varied discourse markers.",
            CefrLevel::C1 => "Use sophisticated academic/literary vocabulary, subtle nuance, precise metaphors, and implicit expressions.",
            CefrLevel::C2 => "Use native-level mastery, culturally rich idioms, stylistic wordplay, and complex specialized terminology.",
        }
    }

    /// Pedagogical grammar guidance for this level.
    pub fn grammar_profile(&self) -> &'static str {
        match self {
            CefrLevel::A1 => "Stick to Present Simple, basic interrogatives (What/Where/Who), and direct S-V-O syntax.",
            CefrLevel::A2 => "Use Past Simple, Present Continuous, basic future with 'going to', and simple compound sentences (and/but/because).",
            CefrLevel::B1 => "Introduce Present Perfect, first/second conditionals, relative clauses (who/which/that), and modal auxiliaries.",
            CefrLevel::B2 => "Employ passive voice, third conditionals, reported speech, varied subordinate clauses, and phrasal verbs.",
            CefrLevel::C1 => "Employ mixed conditionals, inversion ('rarely have I seen'), participle clauses, and complex cleft sentences.",
            CefrLevel::C2 => "Full syntactic flexibility with native intuition: subtle ellipsis, fronting, discourse cohesion, and varied registers.",
        }
    }

    /// Next higher CEFR level if not already C2.
    pub fn next_level(&self) -> Option<CefrLevel> {
        match self {
            CefrLevel::A1 => Some(CefrLevel::A2),
            CefrLevel::A2 => Some(CefrLevel::B1),
            CefrLevel::B1 => Some(CefrLevel::B2),
            CefrLevel::B2 => Some(CefrLevel::C1),
            CefrLevel::C1 => Some(CefrLevel::C2),
            CefrLevel::C2 => None,
        }
    }

    /// Previous lower CEFR level if not already A1.
    pub fn prev_level(&self) -> Option<CefrLevel> {
        match self {
            CefrLevel::A1 => None,
            CefrLevel::A2 => Some(CefrLevel::A1),
            CefrLevel::B1 => Some(CefrLevel::A2),
            CefrLevel::B2 => Some(CefrLevel::B1),
            CefrLevel::C1 => Some(CefrLevel::B2),
            CefrLevel::C2 => Some(CefrLevel::C1),
        }
    }

    /// Heuristically evaluates a sample text to assess estimated CEFR proficiency level.
    pub fn assess_text(text: &str) -> CefrLevel {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return CefrLevel::A1;
        }

        let words: Vec<&str> = trimmed
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return CefrLevel::A1;
        }

        let total_words = words.len();
        let total_chars: usize = words.iter().map(|w| w.chars().count()).sum();
        let avg_word_len = (total_chars as f32) / (total_words as f32);

        // Count sentences
        let sentence_count = text
            .chars()
            .filter(|c| *c == '.' || *c == '!' || *c == '?' || *c == '\n')
            .count()
            .max(1);
        let avg_sent_len = (total_words as f32) / (sentence_count as f32);

        // Lexical diversity (Type-Token Ratio)
        let unique_words: HashSet<String> = words.iter().map(|w| w.to_lowercase()).collect();
        let ttr = (unique_words.len() as f32) / (total_words as f32);

        // Advanced discourse markers and grammatical indicators
        let lower = text.to_lowercase();
        let mut complexity_score = 0;

        let b1_markers = ["because", "although", "while", "since", "in my opinion", "prefer", "recently"];
        let b2_markers = ["furthermore", "moreover", "however", "consequently", "whereas", "in contrast", "despite", "nevertheless"];
        let c1_c2_markers = ["notwithstanding", "paradoxically", "ubiquitous", "inevitably", "subsequently", "thereby", "predominantly", "dichotomy"];

        for marker in b1_markers {
            if lower.contains(marker) {
                complexity_score += 1;
            }
        }
        for marker in b2_markers {
            if lower.contains(marker) {
                complexity_score += 2;
            }
        }
        for marker in c1_c2_markers {
            if lower.contains(marker) {
                complexity_score += 4;
            }
        }

        // Scoring logic combining length, lexical diversity, and discourse markers
        if complexity_score >= 6 || (avg_word_len >= 5.8 && avg_sent_len >= 18.0 && ttr >= 0.70) {
            if complexity_score >= 10 || (avg_word_len >= 6.2 && avg_sent_len >= 22.0) {
                CefrLevel::C2
            } else {
                CefrLevel::C1
            }
        } else if (complexity_score >= 3 && avg_sent_len >= 8.0) || (avg_word_len >= 4.8 && avg_sent_len >= 13.0) {
            CefrLevel::B2
        } else if (complexity_score >= 1 && avg_sent_len >= 6.0) || (avg_word_len >= 4.2 && avg_sent_len >= 9.0) {
            CefrLevel::B1
        } else if avg_word_len >= 3.6 && avg_sent_len >= 4.5 {
            CefrLevel::A2
        } else {
            CefrLevel::A1
        }
    }
}

impl fmt::Display for CefrLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Error returned when CEFR level string is unrecognized.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[error("Invalid CEFR level: '{0}' (expected A1, A2, B1, B2, C1, or C2)")]
pub struct CefrParseError(pub String);

impl FromStr for CefrLevel {
    type Err = CefrParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_uppercase().as_str() {
            "A1" => Ok(CefrLevel::A1),
            "A2" => Ok(CefrLevel::A2),
            "B1" => Ok(CefrLevel::B1),
            "B2" => Ok(CefrLevel::B2),
            "C1" => Ok(CefrLevel::C1),
            "C2" => Ok(CefrLevel::C2),
            other => Err(CefrParseError(other.to_string())),
        }
    }
}

/// A detected grammatical issue or unnatural phrasing in the learner's message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarIssue {
    /// Linguistic category (e.g. "Tense / Agreement", "L1 Transfer", "Preposition").
    pub category: String,
    /// The erroneous or non-native phrase found in the learner's utterance.
    pub faulty_phrase: String,
    /// Recommended natural correction.
    pub suggested_correction: String,
    /// Pedagogical explanation for why this is corrected.
    pub explanation: String,
}

/// Style of recast delivery by the AI tutor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecastMode {
    /// Seamless conversational reformulation without explicit metalinguistic commentary.
    Implicit,
    /// Conversational reformulation with gentle emphasis (e.g. mirroring the corrected form).
    GentleRecast,
    /// Direct model followed immediately by conversational continuation.
    ExplicitRecast,
}

/// Comprehensive evaluation report of a learner's conversational performance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnerEvaluation {
    /// Total conversational turns submitted by learner.
    pub total_user_messages: usize,
    /// Total words spoken by learner.
    pub total_words: usize,
    /// Count of distinct vocabulary words.
    pub unique_words: usize,
    /// Type-Token Ratio [0.0, 1.0].
    pub lexical_diversity: f32,
    /// Average words per utterance.
    pub avg_sentence_length: f32,
    /// Assessed current CEFR proficiency level.
    pub assessed_cefr: CefrLevel,
    /// Count of identified grammatical issues.
    pub detected_grammar_issues: usize,
    /// Actionable pedagogical recommendation.
    pub recommendation: String,
}

/// Manager for pedagogical policies, CEFR language modulation, and conversational recasts.
#[derive(Debug, Clone)]
pub struct TutorPolicyManager {
    target_level: CefrLevel,
    native_language: String,
    target_language: String,
    recast_mode: RecastMode,
    recast_enabled: bool,
}

impl Default for TutorPolicyManager {
    fn default() -> Self {
        Self::new(CefrLevel::B1)
    }
}

/// Helper to verify word boundaries around matched substrings.
fn has_word_boundary(text: &str, start: usize, end: usize) -> bool {
    let before_ok = if start == 0 {
        true
    } else {
        text[..start]
            .chars()
            .next_back()
            .map(|c| !c.is_alphanumeric())
            .unwrap_or(true)
    };

    let after_ok = if end >= text.len() {
        true
    } else {
        text[end..]
            .chars()
            .next()
            .map(|c| !c.is_alphanumeric())
            .unwrap_or(true)
    };

    before_ok && after_ok
}

/// Helper to check if a phrase occurs with proper word boundaries on both ends.
fn contains_word_phrase(text_lower: &str, phrase: &str) -> bool {
    let mut search_from = 0;
    while let Some(rel_idx) = text_lower[search_from..].find(phrase) {
        let abs_start = search_from + rel_idx;
        let abs_end = abs_start + phrase.len();
        if has_word_boundary(text_lower, abs_start, abs_end) {
            return true;
        }
        search_from = abs_start + 1;
        if search_from >= text_lower.len() {
            break;
        }
    }
    false
}

impl TutorPolicyManager {
    /// Creates a new tutor policy manager targeting a specific CEFR level.
    pub fn new(target_level: CefrLevel) -> Self {
        Self {
            target_level,
            native_language: "vi".to_string(),
            target_language: "en".to_string(),
            recast_mode: RecastMode::GentleRecast,
            recast_enabled: true,
        }
    }

    /// Initializes a tutor policy manager from an existing `UserProfile`.
    pub fn from_profile(profile: &UserProfile) -> Self {
        let target_level = profile
            .cefr_level
            .parse::<CefrLevel>()
            .unwrap_or(CefrLevel::B1);

        Self {
            target_level,
            native_language: profile.native_language.clone(),
            target_language: profile.target_language.clone(),
            recast_mode: RecastMode::GentleRecast,
            recast_enabled: true,
        }
    }

    /// Sets native and target learning languages.
    pub fn with_languages(
        mut self,
        native: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        self.native_language = native.into();
        self.target_language = target.into();
        self
    }

    /// Sets the recast mode.
    pub fn with_recast_mode(mut self, mode: RecastMode) -> Self {
        self.recast_mode = mode;
        self
    }

    /// Sets whether conversational recast prompts are generated.
    pub fn with_recast_enabled(mut self, enabled: bool) -> Self {
        self.recast_enabled = enabled;
        self
    }

    /// Returns current CEFR level.
    pub fn target_level(&self) -> CefrLevel {
        self.target_level
    }

    /// Updates target CEFR level.
    pub fn set_target_level(&mut self, level: CefrLevel) {
        self.target_level = level;
    }

    /// Returns the active recast mode.
    pub fn recast_mode(&self) -> RecastMode {
        self.recast_mode
    }

    /// Detects common English grammatical issues and Vietnamese-English L1 transfer errors.
    pub fn detect_grammar_issues(text: &str) -> Vec<GrammarIssue> {
        let mut issues = Vec::new();
        let lower = text.to_lowercase();

        // 1. Vietnamese-English L1 transfer: "very like" -> "really like / like ... very much"
        if contains_word_phrase(&lower, "very like") {
            issues.push(GrammarIssue {
                category: "L1 Transfer (Word Order)".to_string(),
                faulty_phrase: "very like".to_string(),
                suggested_correction: "really like / like ... very much".to_string(),
                explanation: "'Very' cannot directly modify verbs in English; use 'really' or place 'very much' at the end.".to_string(),
            });
        }

        // 2. Vietnamese-English transfer: "no have" -> "there is no / don't have"
        if contains_word_phrase(&lower, "no have") {
            issues.push(GrammarIssue {
                category: "L1 Transfer (Negation)".to_string(),
                faulty_phrase: "no have".to_string(),
                suggested_correction: "don't have / there is no".to_string(),
                explanation: "English uses auxiliary verb negation ('don't have') or existential 'there is no'.".to_string(),
            });
        }

        // 3. Preposition error: "depend of" / "depends of" -> "depend on / depends on"
        if contains_word_phrase(&lower, "depend of") {
            issues.push(GrammarIssue {
                category: "Preposition Collocation".to_string(),
                faulty_phrase: "depend of".to_string(),
                suggested_correction: "depend on".to_string(),
                explanation: "The verb 'depend' collocates with 'on', not 'of'.".to_string(),
            });
        } else if contains_word_phrase(&lower, "depends of")
            || contains_word_phrase(&lower, "depended of")
            || contains_word_phrase(&lower, "depending of")
        {
            issues.push(GrammarIssue {
                category: "Preposition Collocation".to_string(),
                faulty_phrase: "depends of".to_string(),
                suggested_correction: "depends on".to_string(),
                explanation: "The verb 'depend' collocates with 'on', not 'of'.".to_string(),
            });
        }

        // 4. Preposition error: "good in [skill/subject/gerund]" -> "good at"
        {
            let mut search_from = 0;
            while let Some(rel_idx) = lower[search_from..].find("good in") {
                let start = search_from + rel_idx;
                let end = start + "good in".len();
                if has_word_boundary(&lower, start, end) {
                    let remainder = lower[end..].trim_start();
                    let next_word: String = remainder
                        .chars()
                        .take_while(|c| c.is_alphabetic())
                        .collect();
                    let non_error_exceptions = [
                        "general", "itself", "terms", "person", "order", "case", "fact", "part",
                        "the", "a", "my", "your", "our", "their", "his", "her", "mind", "front",
                    ];
                    if !next_word.is_empty() && !non_error_exceptions.contains(&next_word.as_str()) {
                        issues.push(GrammarIssue {
                            category: "Preposition Collocation".to_string(),
                            faulty_phrase: format!("good in {next_word}"),
                            suggested_correction: "good at".to_string(),
                            explanation: "Use 'good at' when referring to skills, abilities, subjects, or activities.".to_string(),
                        });
                        break;
                    }
                }
                search_from = end;
                if search_from >= lower.len() {
                    break;
                }
            }
        }

        // 5. Redundant preposition: "marry with" -> "marry"
        for phrase in ["marry with", "marries with", "married with"] {
            let mut search_from = 0;
            while let Some(rel_idx) = lower[search_from..].find(phrase) {
                let start = search_from + rel_idx;
                let end = start + phrase.len();
                if has_word_boundary(&lower, start, end) {
                    let remainder = lower[end..].trim_start();
                    let next_word: String = remainder
                        .chars()
                        .take_while(|c| c.is_alphabetic())
                        .collect();
                    if next_word != "children" && next_word != "kids" {
                        issues.push(GrammarIssue {
                            category: "Transitive Verb Usage".to_string(),
                            faulty_phrase: phrase.to_string(),
                            suggested_correction: "marry / married to".to_string(),
                            explanation: "'Marry' is a transitive verb (e.g. 'marry him'); as an adjective use 'married to'.".to_string(),
                        });
                        break;
                    }
                }
                search_from = end;
                if search_from >= lower.len() {
                    break;
                }
            }
        }

        // 6. Redundant preposition: "discuss about" -> "discuss"
        for phrase in ["discuss about", "discusses about", "discussed about", "discussing about"] {
            if contains_word_phrase(&lower, phrase) {
                issues.push(GrammarIssue {
                    category: "Transitive Verb Usage".to_string(),
                    faulty_phrase: phrase.to_string(),
                    suggested_correction: "discuss / talk about".to_string(),
                    explanation: "'Discuss' directly takes an object without 'about'.".to_string(),
                });
                break;
            }
        }

        // 7. Double comparative: "more better" -> "better"
        for phrase in ["more better", "more bigger", "more faster", "more easier", "more higher", "more smaller"] {
            if contains_word_phrase(&lower, phrase) {
                let base = phrase.strip_prefix("more ").unwrap_or(phrase);
                issues.push(GrammarIssue {
                    category: "Comparative Form".to_string(),
                    faulty_phrase: phrase.to_string(),
                    suggested_correction: base.to_string(),
                    explanation: format!("'{base}' is already the comparative form of 'good'; do not add 'more'."),
                });
                break;
            }
        }

        // 8. Stative verb error: "I am agree" -> "I agree"
        for phrase in [
            "i am agree", "i'm agree", "we are agree", "we're agree", "they are agree", "they're agree",
            "he is agree", "he's agree", "she is agree", "she's agree",
        ] {
            if contains_word_phrase(&lower, phrase) {
                issues.push(GrammarIssue {
                    category: "Verb Structure".to_string(),
                    faulty_phrase: phrase.to_string(),
                    suggested_correction: "I agree".to_string(),
                    explanation: "'Agree' is already a verb; do not use the auxiliary 'am/is/are' with it in Present Simple.".to_string(),
                });
                break;
            }
        }

        // 9. Uncountable noun plural: "informations" / "advices"
        if contains_word_phrase(&lower, "informations") {
            issues.push(GrammarIssue {
                category: "Countability".to_string(),
                faulty_phrase: "informations".to_string(),
                suggested_correction: "information / pieces of information".to_string(),
                explanation: "'Information' is uncountable in English and has no plural 's'.".to_string(),
            });
        }
        if contains_word_phrase(&lower, "advices") {
            issues.push(GrammarIssue {
                category: "Countability".to_string(),
                faulty_phrase: "advices".to_string(),
                suggested_correction: "advice / pieces of advice".to_string(),
                explanation: "'Advice' is uncountable; use 'pieces of advice' if quantification is needed.".to_string(),
            });
        }

        // 10. Missing past tense with explicit past adverbials: "yesterday I go", "yesterday I see", etc.
        {
            let time_markers = ["yesterday", "last night", "last week", "last year", "last month"];
            let present_to_past = [
                ("go", "went"),
                ("see", "saw"),
                ("buy", "bought"),
                ("eat", "ate"),
                ("meet", "met"),
                ("have", "had"),
                ("make", "made"),
                ("come", "came"),
                ("take", "took"),
                ("get", "got"),
                ("tell", "told"),
                ("write", "wrote"),
            ];

            for marker in time_markers {
                if let Some(idx) = lower.find(marker) {
                    if has_word_boundary(&lower, idx, idx + marker.len()) {
                        let after_marker = lower[idx + marker.len()..].trim_start();
                        let cleaned_after = after_marker.trim_start_matches(|c: char| c == ',' || c == ';' || c.is_whitespace());
                        let words: Vec<&str> = cleaned_after.split_whitespace().collect();
                        if words.len() >= 2 {
                            let subj = words[0];
                            let verb = words[1].trim_matches(|c: char| !c.is_alphabetic());
                            if matches!(subj, "i" | "we" | "he" | "she" | "they" | "you") {
                                let display_subj = if subj == "i" { "I" } else { subj };
                                for (pres, past) in present_to_past {
                                    if verb == pres {
                                        issues.push(GrammarIssue {
                                            category: "Past Tense Inconsistency".to_string(),
                                            faulty_phrase: format!("{marker} {display_subj} {pres}"),
                                            suggested_correction: format!("{marker} {display_subj} {past}"),
                                            explanation: format!("With past time markers such as '{marker}', verbs must be in Past Simple form ('{past}')."),
                                        });
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 11. Third person singular agreement: "he go", "she go", "he don't", "she don't"
        if contains_word_phrase(&lower, "he don't")
            || contains_word_phrase(&lower, "she don't")
            || contains_word_phrase(&lower, "it don't")
        {
            issues.push(GrammarIssue {
                category: "Subject-Verb Agreement".to_string(),
                faulty_phrase: "he/she don't".to_string(),
                suggested_correction: "he/she doesn't".to_string(),
                explanation: "Third-person singular subjects (he/she/it) require 'doesn't' instead of 'don't' in Present Simple.".to_string(),
            });
        }

        for subj in ["he", "she"] {
            let phrase = format!("{subj} go");
            let mut search_from = 0;
            while let Some(rel_idx) = lower[search_from..].find(&phrase) {
                let start = search_from + rel_idx;
                let end = start + phrase.len();
                if has_word_boundary(&lower, start, end) {
                    let prefix = lower[..start].trim_end();
                    let last_word = prefix
                        .split_whitespace()
                        .next_back()
                        .unwrap_or("")
                        .trim_matches(|c: char| !c.is_alphabetic());

                    let modal_aux_preceding = [
                        "does", "doesn't", "did", "didn't", "can", "can't", "could", "couldn't",
                        "will", "won't", "would", "wouldn't", "shall", "should", "shouldn't",
                        "may", "might", "must", "to", "let", "makes", "made",
                    ];

                    let is_question_inversion = modal_aux_preceding.contains(&last_word);

                    if !is_question_inversion {
                        issues.push(GrammarIssue {
                            category: "Subject-Verb Agreement".to_string(),
                            faulty_phrase: format!("{subj} go"),
                            suggested_correction: format!("{subj} goes"),
                            explanation: "Third-person singular subjects (he/she/it) require the -s/-es inflection in Present Simple.".to_string(),
                        });
                        break;
                    }
                }
                search_from = end;
                if search_from >= lower.len() {
                    break;
                }
            }
        }

        issues
    }

    /// Builds a targeted recast directive for the AI system prompt if issues were identified.
    pub fn build_recast_instruction(&self, issues: &[GrammarIssue]) -> Option<String> {
        if !self.recast_enabled || issues.is_empty() {
            return None;
        }

        let mut lines = Vec::new();
        lines.push("[PEDAGOGICAL RECAST DIRECTIVE]".to_string());
        lines.push(
            "The learner made the following grammar/phrasing error(s) in their message:".to_string(),
        );

        for issue in issues {
            lines.push(format!(
                "- Error: '{}' -> Correct Model: '{}' ({})",
                issue.faulty_phrase, issue.suggested_correction, issue.explanation
            ));
        }

        match self.recast_mode {
            RecastMode::Implicit => {
                lines.push(
                    "INSTRUCTION: Seamlessly recast and model the correct phrasing within your natural conversational reply. DO NOT overtly correct or lecture the learner; maintain continuous, engaging conversation."
                        .to_string(),
                );
            }
            RecastMode::GentleRecast => {
                lines.push(
                    "INSTRUCTION: Naturally reformulate the correct sentence structure in your opening conversational turn (e.g. 'Oh, you really like...' or 'I see, you went to...'), then continue the conversation with an engaging follow-up question. Keep your tone supportive and warm."
                        .to_string(),
                );
            }
            RecastMode::ExplicitRecast => {
                lines.push(
                    "INSTRUCTION: Gently validate their meaning and provide the natural phrasing before transitioning smoothly back into the conversation."
                        .to_string(),
                );
            }
        }

        Some(lines.join("\n"))
    }

    /// Generates full pedagogical system instructions tailored to the learner's CEFR level.
    pub fn generate_system_instruction(&self) -> String {
        let (min_w, max_w) = self.target_level.target_sentence_length();
        let vocab_guide = self.target_level.vocabulary_profile();
        let grammar_guide = self.target_level.grammar_profile();

        format!(
            "[PEDAGOGICAL POLICY: CEFR LEVEL {} ({})]\n\
            You are Vira, an empathetic and highly skilled language companion.\n\
            The learner's target proficiency level is {}.\n\
            1. Sentence Complexity: Target average sentence length of {} to {} words. Avoid overly convoluted clause nesting.\n\
            2. Vocabulary Modulation: {}\n\
            3. Grammar Scaffolding: {}\n\
            4. Conversational Recasting: When the learner makes mistakes, perform subtle conversational recasts—modeling the correct idiom or grammatical form naturally within your conversational flow rather than interrupting or criticizing.\n\
            5. Communicative Engagement: End responses with natural, level-appropriate open-ended prompts to encourage learner output.",
            self.target_level.as_str(),
            self.target_level.title(),
            self.target_level.as_str(),
            min_w,
            max_w,
            vocab_guide,
            grammar_guide
        )
    }

    /// Enriches conversation history by prepending or updating the pedagogical system prompt
    /// and injecting recast directives for the most recent user turn.
    pub fn enrich_messages(&self, history: &[Message]) -> Vec<Message> {
        let mut enriched = Vec::new();

        // Check if the latest message was from user and has errors
        let latest_user_msg = history.iter().rev().find(|m| m.role == Role::User);
        let recast_instruction = if let Some(user_msg) = latest_user_msg {
            let issues = Self::detect_grammar_issues(&user_msg.raw_content);
            self.build_recast_instruction(&issues)
        } else {
            None
        };

        let base_policy = self.generate_system_instruction();
        let combined_system_prompt = match recast_instruction {
            Some(recast) => format!("{}\n\n{}", base_policy, recast),
            None => base_policy,
        };

        // If existing system message exists at the start, replace or enhance it
        let mut system_found = false;
        for (i, msg) in history.iter().enumerate() {
            if i == 0 && msg.role == Role::System {
                system_found = true;
                // Strip any previously injected pedagogical policy & recast blocks
                let base_raw = if let Some(idx) = msg.raw_content.find("[PEDAGOGICAL POLICY:") {
                    msg.raw_content[..idx].trim_end()
                } else {
                    msg.raw_content.as_str()
                };

                let merged_content = if base_raw.is_empty() {
                    combined_system_prompt.clone()
                } else {
                    format!("{}\n\n{}", base_raw, combined_system_prompt)
                };
                enriched.push(Message::system(merged_content));
            } else {
                enriched.push(msg.clone());
            }
        }

        if !system_found {
            enriched.insert(0, Message::system(combined_system_prompt));
        }

        enriched
    }

    /// Evaluates the learner's performance across user messages in a session.
    pub fn evaluate_learner_performance(&self, messages: &[Message]) -> LearnerEvaluation {
        let user_messages: Vec<&Message> = messages.iter().filter(|m| m.role == Role::User).collect();

        if user_messages.is_empty() {
            return LearnerEvaluation {
                total_user_messages: 0,
                total_words: 0,
                unique_words: 0,
                lexical_diversity: 0.0,
                avg_sentence_length: 0.0,
                assessed_cefr: self.target_level,
                detected_grammar_issues: 0,
                recommendation: "Begin conversing with Vira to establish your baseline profile."
                    .to_string(),
            };
        }

        let combined_text: String = user_messages
            .iter()
            .map(|m| m.raw_content.as_str())
            .collect::<Vec<&str>>()
            .join(" ");

        let words: Vec<&str> = combined_text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
            .filter(|w| !w.is_empty())
            .collect();

        let total_words = words.len();
        let unique_words: HashSet<String> = words.iter().map(|w| w.to_lowercase()).collect();
        let lexical_diversity = if total_words > 0 {
            (unique_words.len() as f32) / (total_words as f32)
        } else {
            0.0
        };

        let sentence_count = combined_text
            .chars()
            .filter(|c| *c == '.' || *c == '!' || *c == '?' || *c == '\n')
            .count()
            .max(user_messages.len());
        let avg_sentence_length = if sentence_count > 0 {
            (total_words as f32) / (sentence_count as f32)
        } else {
            0.0
        };

        let assessed_cefr = CefrLevel::assess_text(&combined_text);

        let mut total_issues = 0;
        for msg in &user_messages {
            let issues = Self::detect_grammar_issues(&msg.raw_content);
            total_issues += issues.len();
        }

        let recommendation = if assessed_cefr > self.target_level {
            format!(
                "Your performance exhibits strong competency reaching {}. Ready to progress from {}!",
                assessed_cefr.as_str(),
                self.target_level.as_str()
            )
        } else if total_issues == 0 && lexical_diversity > 0.65 {
            format!(
                "Excellent precision and fluency at {}. Maintain consistent practice.",
                self.target_level.as_str()
            )
        } else {
            format!(
                "Continue conversational practice at {} with focus on natural collocations and tense consistency.",
                self.target_level.as_str()
            )
        };

        LearnerEvaluation {
            total_user_messages: user_messages.len(),
            total_words,
            unique_words: unique_words.len(),
            lexical_diversity,
            avg_sentence_length,
            assessed_cefr,
            detected_grammar_issues: total_issues,
            recommendation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tutor_manager_from_profile() {
        let profile = UserProfile::new("u1", "Bob", "vi", "en", "B2");
        let mut manager = TutorPolicyManager::from_profile(&profile);

        assert_eq!(manager.target_level(), CefrLevel::B2);
        manager.set_target_level(CefrLevel::C1);
        assert_eq!(manager.target_level(), CefrLevel::C1);
    }

    #[test]
    fn test_tutor_manager_with_languages_and_recast_modes() {
        let manager = TutorPolicyManager::new(CefrLevel::A2)
            .with_languages("vi", "en")
            .with_recast_mode(RecastMode::ExplicitRecast)
            .with_recast_enabled(false);

        assert_eq!(manager.recast_mode(), RecastMode::ExplicitRecast);

        // When recast is disabled, building recast instruction returns None
        let issues = TutorPolicyManager::detect_grammar_issues("I very like football.");
        assert!(manager.build_recast_instruction(&issues).is_none());
    }

    #[test]
    fn test_detect_all_grammar_patterns() {
        let t1 = TutorPolicyManager::detect_grammar_issues("I no have any book.");
        assert!(t1.iter().any(|i| i.faulty_phrase == "no have"));

        let t2 = TutorPolicyManager::detect_grammar_issues("She is very good in math.");
        assert!(t2.iter().any(|i| i.suggested_correction == "good at"));

        let t3 = TutorPolicyManager::detect_grammar_issues("I want to marry with her.");
        assert!(t3.iter().any(|i| i.suggested_correction.contains("marry")));

        let t4 = TutorPolicyManager::detect_grammar_issues("Let us discuss about this problem.");
        assert!(t4.iter().any(|i| i.suggested_correction.contains("discuss")));

        let t5 = TutorPolicyManager::detect_grammar_issues("This idea is more better.");
        assert!(t5.iter().any(|i| i.suggested_correction == "better"));

        let t6 = TutorPolicyManager::detect_grammar_issues("I am agree with you.");
        assert!(t6.iter().any(|i| i.suggested_correction == "I agree"));

        let t7 = TutorPolicyManager::detect_grammar_issues("Give me some informations and advices.");
        assert!(t7.iter().any(|i| i.faulty_phrase == "informations"));
        assert!(t7.iter().any(|i| i.faulty_phrase == "advices"));

        let t8 = TutorPolicyManager::detect_grammar_issues("He go to gym every day and she don't eat sugar.");
        assert!(t8.iter().any(|i| i.category.contains("Subject-Verb Agreement")));
    }

    #[test]
    fn test_learner_evaluation_with_empty_history() {
        let manager = TutorPolicyManager::new(CefrLevel::B1);
        let eval = manager.evaluate_learner_performance(&[]);

        assert_eq!(eval.total_user_messages, 0);
        assert_eq!(eval.total_words, 0);
        assert_eq!(eval.assessed_cefr, CefrLevel::B1);
    }
}

