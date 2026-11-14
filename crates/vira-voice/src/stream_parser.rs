use std::collections::HashSet;
use serde::{Deserialize, Serialize};
use vira_domain::CharacterEmotion;

use crate::chunker::{ChunkerConfig, SentenceChunker};

/// Default LangIns avatar gesture identifiers recognized in shortened tag form (e.g. `[wave]`).
pub const DEFAULT_GESTURES: &[&str] = &[
    "wave", "nod", "think", "idea", "praise", "bow", "clap", "shake", "shrug", "point",
    "salute", "cheer", "laugh", "sigh", "tilt", "wink", "dance", "confused", "idle",
    "explain", "shy", "encourage", "surprise", "relax",
];

/// Events produced during streaming LLM parsing for avatar animation and voice synthesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParsedStreamEvent {
    /// Avatar facial expression change event (e.g. "happy", "surprised").
    Emotion(String),
    /// Avatar body/hand gesture animation trigger event (e.g. "wave", "nod").
    Gesture(String),
    /// Internal reasoning scratchpad block (e.g. `<thought>...</thought>` or `<think>...</think>`).
    /// Strictly stripped from TTS audio.
    Thought(String),
    /// Grammatically complete, clean sentence or clause chunk ready for TTS audio synthesis.
    SpeakableText(String),
}

impl ParsedStreamEvent {
    /// Creates a new `ParsedStreamEvent::Emotion`.
    pub fn emotion(name: impl Into<String>) -> Self {
        Self::Emotion(name.into())
    }

    /// Creates a new `ParsedStreamEvent::Gesture`.
    pub fn gesture(name: impl Into<String>) -> Self {
        Self::Gesture(name.into())
    }

    /// Creates a new `ParsedStreamEvent::Thought`.
    pub fn thought(text: impl Into<String>) -> Self {
        Self::Thought(text.into())
    }

    /// Creates a new `ParsedStreamEvent::SpeakableText`.
    pub fn speakable(text: impl Into<String>) -> Self {
        Self::SpeakableText(text.into())
    }

    /// Returns true if this event is an emotion change.
    pub fn is_emotion(&self) -> bool {
        matches!(self, Self::Emotion(_))
    }

    /// Returns true if this event is a gesture animation.
    pub fn is_gesture(&self) -> bool {
        matches!(self, Self::Gesture(_))
    }

    /// Returns true if this event is a thought scratchpad block.
    pub fn is_thought(&self) -> bool {
        matches!(self, Self::Thought(_))
    }

    /// Returns true if this event is speakable text for TTS.
    pub fn is_speakable(&self) -> bool {
        matches!(self, Self::SpeakableText(_))
    }

    /// Returns the inner string if this is an Emotion event.
    pub fn as_emotion(&self) -> Option<&str> {
        match self {
            Self::Emotion(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the inner string if this is a Gesture event.
    pub fn as_gesture(&self) -> Option<&str> {
        match self {
            Self::Gesture(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the inner string if this is a Thought event.
    pub fn as_thought(&self) -> Option<&str> {
        match self {
            Self::Thought(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the inner string if this is a SpeakableText event.
    pub fn as_speakable(&self) -> Option<&str> {
        match self {
            Self::SpeakableText(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Attempts to parse this event as a domain `CharacterEmotion`.
    pub fn to_character_emotion(&self) -> Option<CharacterEmotion> {
        match self {
            Self::Emotion(s) => s.parse().ok(),
            _ => None,
        }
    }
}

/// Streaming tag parser and sentence chunker for LangIns LLM output.
///
/// Strips inline avatar control tags (`[emotion:...]`, `[gesture:...]`, and shortened
/// gestures like `[wave]`), emits them immediately for UI avatar state, and streams clean
/// grammatical sentence chunks to the TTS queue to minimize Time-to-First-Audio.
#[derive(Debug, Clone)]
pub struct StreamParser {
    chunker: SentenceChunker,
    pending_raw: String,
    last_char: Option<char>,
    tag_just_stripped: bool,
    custom_gestures: HashSet<String>,
    in_thought: Option<String>,
}

impl Default for StreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamParser {
    /// Creates a new `StreamParser` with default chunker settings and standard gesture list.
    pub fn new() -> Self {
        Self {
            chunker: SentenceChunker::new(),
            pending_raw: String::new(),
            last_char: None,
            tag_just_stripped: false,
            custom_gestures: HashSet::new(),
            in_thought: None,
        }
    }

    /// Creates a `StreamParser` with custom `ChunkerConfig`.
    pub fn with_chunker_config(config: ChunkerConfig) -> Self {
        Self {
            chunker: SentenceChunker::with_config(config),
            pending_raw: String::new(),
            last_char: None,
            tag_just_stripped: false,
            custom_gestures: HashSet::new(),
            in_thought: None,
        }
    }

    /// Registers a custom gesture identifier for shortened tag recognition (e.g. `[my_custom_anim]`).
    pub fn register_gesture(&mut self, gesture: impl Into<String>) {
        self.custom_gestures.insert(gesture.into().to_ascii_lowercase());
    }

    /// Checks if a string matches a recognized avatar gesture (case-insensitive).
    pub fn is_known_gesture(&self, val: &str) -> bool {
        let lower = val.trim().to_ascii_lowercase();
        DEFAULT_GESTURES.contains(&lower.as_str()) || self.custom_gestures.contains(&lower)
    }

    /// Checks if a string matches a recognized avatar emotion preset (case-insensitive).
    pub fn is_known_emotion(&self, val: &str) -> bool {
        let lower = val.trim().to_ascii_lowercase();
        matches!(
            lower.as_str(),
            "neutral" | "happy" | "sad" | "angry" | "surprised" | "relaxed"
        )
    }

    /// Resets all internal buffers and state machine.
    pub fn reset(&mut self) {
        self.pending_raw.clear();
        self.chunker.reset();
        self.last_char = None;
        self.tag_just_stripped = false;
        self.in_thought = None;
    }

    fn push_text_to_chunker(&mut self, text: &str, events: &mut Vec<ParsedStreamEvent>) {
        if let Some(last) = text.chars().last() {
            self.last_char = Some(last);
        }
        let chunks = self.chunker.feed(text);
        for chunk in chunks {
            events.push(ParsedStreamEvent::SpeakableText(chunk));
        }
    }

    fn on_tag_removed(&mut self, c_idx: usize) {
        let after_tag_char = self.pending_raw[c_idx + 1..].chars().next();
        let need_space_before = self.last_char.is_some_and(|c| c.is_alphanumeric());

        self.pending_raw.drain(..=c_idx);

        if let Some(next_c) = after_tag_char {
            if need_space_before && next_c.is_alphanumeric() {
                self.pending_raw.insert(0, ' ');
            }
            self.tag_just_stripped = false;
        } else {
            self.tag_just_stripped = need_space_before;
        }
    }

    /// Ingests an incoming token chunk from LLM stream and returns any extracted avatar events,
    /// thought scratchpad blocks, and complete speakable sentence chunks.
    pub fn feed(&mut self, text: &str) -> Vec<ParsedStreamEvent> {
        if self.tag_just_stripped {
            if let Some(first_c) = text.chars().next() {
                if first_c.is_alphanumeric() {
                    self.pending_raw.push(' ');
                }
            }
            self.tag_just_stripped = false;
        }

        self.pending_raw.push_str(text);
        let mut events = Vec::new();

        loop {
            // 1. If currently inside a thought or tool block, wait for matching closing tag
            if let Some(ref tag_name) = self.in_thought.clone() {
                let lower = self.pending_raw.to_ascii_lowercase();
                let close_tag = format!("</{}>", tag_name);
                if let Some(close_idx) = lower.find(&close_tag) {
                    let thought_content = self.pending_raw[..close_idx].trim().to_string();
                    self.pending_raw.drain(..close_idx + close_tag.len());
                    self.in_thought = None;
                    if !thought_content.is_empty() {
                        events.push(ParsedStreamEvent::Thought(thought_content));
                    }
                    continue;
                } else {
                    // Still inside thought block, waiting for closing tag
                    break;
                }
            }

            // 2. Locate potential tags: [ for gestures/emotions and <thought> / <think>
            let bracket_open = self.pending_raw.find('[');
            let lower_raw = self.pending_raw.to_ascii_lowercase();
            let thought_open = lower_raw.find("<thought>").map(|idx| (idx, "thought", 9));
            let think_open = lower_raw.find("<think>").map(|idx| (idx, "think", 7));

            // Select earliest matching complete tag
            let earliest_thought = match (thought_open, think_open) {
                (Some(a), Some(b)) => if a.0 <= b.0 { Some(a) } else { Some(b) },
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };

            // 3. Check for incomplete '<' opening tag fragment at or near end of pending_raw
            let incomplete_lt = lower_raw.rfind('<').and_then(|lt_idx| {
                let tail = &lower_raw[lt_idx..];
                let is_prefix = "<thought>".starts_with(tail) || "<think>".starts_with(tail);
                if is_prefix && earliest_thought.is_none_or(|(t_idx, _, _)| lt_idx < t_idx) {
                    Some(lt_idx)
                } else {
                    None
                }
            });

            if let Some(lt_idx) = incomplete_lt {
                // Bracket tag before incomplete tag?
                if let Some(b_idx) = bracket_open {
                    if b_idx < lt_idx {
                        // Bracket tag is earlier, handle it below
                    } else {
                        // Flush text before incomplete '<' tag and wait for tag completion
                        if lt_idx > 0 {
                            let text_before = self.pending_raw[..lt_idx].to_string();
                            self.push_text_to_chunker(&text_before, &mut events);
                            self.pending_raw.drain(..lt_idx);
                        }
                        break;
                    }
                } else {
                    // Flush text before incomplete '<' tag and wait for tag completion
                    if lt_idx > 0 {
                        let text_before = self.pending_raw[..lt_idx].to_string();
                        self.push_text_to_chunker(&text_before, &mut events);
                        self.pending_raw.drain(..lt_idx);
                    }
                    break;
                }
            }

            match (bracket_open, earliest_thought) {
                (None, None) => {
                    // No open brackets or thought tags present; entire pending text is speakable
                    let text = std::mem::take(&mut self.pending_raw);
                    self.push_text_to_chunker(&text, &mut events);
                    break;
                }
                (Some(b_idx), Some((t_idx, tag_name, tag_len))) if t_idx < b_idx => {
                    // Thought opening tag appears before bracket tag
                    if t_idx > 0 {
                        let text_before = self.pending_raw[..t_idx].to_string();
                        self.push_text_to_chunker(&text_before, &mut events);
                        self.pending_raw.drain(..t_idx);
                    }
                    self.pending_raw.drain(..tag_len);
                    self.in_thought = Some(tag_name.to_string());
                    continue;
                }
                (None, Some((t_idx, tag_name, tag_len))) => {
                    // Thought opening tag present with no bracket
                    if t_idx > 0 {
                        let text_before = self.pending_raw[..t_idx].to_string();
                        self.push_text_to_chunker(&text_before, &mut events);
                        self.pending_raw.drain(..t_idx);
                    }
                    self.pending_raw.drain(..tag_len);
                    self.in_thought = Some(tag_name.to_string());
                    continue;
                }
                (Some(idx), _) => {
                    // Bracket tag appears first
                    if idx > 0 {
                        let text_before = self.pending_raw[..idx].to_string();
                        self.push_text_to_chunker(&text_before, &mut events);
                        self.pending_raw.drain(..idx);
                    }

                    // Now self.pending_raw starts with '['
                    let close_idx = self.pending_raw.find(']');
                    match close_idx {
                        Some(c_idx) => {
                            let inner_slice = &self.pending_raw[1..c_idx];
                            // If there is another '[' before ']', the first '[' is plain text
                            if let Some(second_open) = inner_slice.find('[') {
                                let non_tag = self.pending_raw[..1 + second_open].to_string();
                                self.pending_raw.drain(..1 + second_open);
                                self.push_text_to_chunker(&non_tag, &mut events);
                                continue;
                            }

                            let inner = inner_slice.trim();

                            if let Some((kind, val)) = inner.split_once(':') {
                                let kind = kind.trim().to_ascii_lowercase();
                                let val = val.trim().to_ascii_lowercase();
                                if kind == "emotion" && !val.is_empty() {
                                    events.push(ParsedStreamEvent::Emotion(val));
                                    self.on_tag_removed(c_idx);
                                } else if kind == "gesture" && !val.is_empty() {
                                    events.push(ParsedStreamEvent::Gesture(val));
                                    self.on_tag_removed(c_idx);
                                } else {
                                    let non_tag = self.pending_raw[..=c_idx].to_string();
                                    self.pending_raw.drain(..=c_idx);
                                    self.push_text_to_chunker(&non_tag, &mut events);
                                }
                            } else if self.is_known_gesture(inner) {
                                events.push(ParsedStreamEvent::Gesture(inner.to_ascii_lowercase()));
                                self.on_tag_removed(c_idx);
                            } else if self.is_known_emotion(inner) {
                                events.push(ParsedStreamEvent::Emotion(inner.to_ascii_lowercase()));
                                self.on_tag_removed(c_idx);
                            } else {
                                // Not a tag (e.g. citation `[1]` or markdown `[Link]`). Treat as regular text.
                                let non_tag = self.pending_raw[..=c_idx].to_string();
                                self.pending_raw.drain(..=c_idx);
                                self.push_text_to_chunker(&non_tag, &mut events);
                            }
                        }
                        None => {
                            // Incomplete bracket tag (e.g. `[em` or `[gesture:wa`).
                            let slice = &self.pending_raw[1..];
                            let has_invalid_tag_char = slice.chars().any(|c| {
                                matches!(c, '.' | '!' | '?' | ',' | ';' | '\n' | '\r' | '(' | ')' | '{' | '}')
                            });
                            let has_second_open = slice.contains('[');

                            if has_invalid_tag_char || self.pending_raw.len() > 48 {
                                let not_tag = std::mem::take(&mut self.pending_raw);
                                self.push_text_to_chunker(&not_tag, &mut events);
                            } else if has_second_open {
                                let second_open = slice.find('[').unwrap();
                                let not_tag = self.pending_raw[..1 + second_open].to_string();
                                self.pending_raw.drain(..1 + second_open);
                                self.push_text_to_chunker(&not_tag, &mut events);
                            }
                            // Otherwise wait for the closing bracket in subsequent chunks
                            break;
                        }
                    }
                }
            }
        }

        events
    }

    /// Concludes the stream, flushing any trailing bracket fragments, thoughts, and pending sentence chunks.
    pub fn finish(&mut self) -> Vec<ParsedStreamEvent> {
        let mut events = Vec::new();

        if let Some(_tag_name) = self.in_thought.take() {
            let thought_leftover = std::mem::take(&mut self.pending_raw);
            let trimmed = thought_leftover.trim();
            if !trimmed.is_empty() {
                events.push(ParsedStreamEvent::Thought(trimmed.to_string()));
            }
        } else if !self.pending_raw.is_empty() {
            let leftover = std::mem::take(&mut self.pending_raw);
            let chunks = self.chunker.feed(&leftover);
            for chunk in chunks {
                events.push(ParsedStreamEvent::SpeakableText(chunk));
            }
        }

        let flushed = self.chunker.finish();
        for chunk in flushed {
            events.push(ParsedStreamEvent::SpeakableText(chunk));
        }

        self.last_char = None;
        self.tag_just_stripped = false;
        events
    }

    /// Alias for `finish()`.
    pub fn flush(&mut self) -> Vec<ParsedStreamEvent> {
        self.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parsed_stream_event_helpers_and_serde() {
        let e = ParsedStreamEvent::emotion("happy");
        assert!(e.is_emotion());
        assert_eq!(e.as_emotion(), Some("happy"));
        assert_eq!(e.to_character_emotion(), Some(CharacterEmotion::Happy));

        let g = ParsedStreamEvent::gesture("wave");
        assert!(g.is_gesture());
        assert_eq!(g.as_gesture(), Some("wave"));
        assert_eq!(g.to_character_emotion(), None);

        let s = ParsedStreamEvent::speakable("Hello!");
        assert!(s.is_speakable());
        assert_eq!(s.as_speakable(), Some("Hello!"));

        // Serde test
        let json = serde_json::to_string(&e).unwrap();
        let parsed: ParsedStreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(e, parsed);
    }

    #[test]
    fn test_stream_parser_case_insensitivity_and_whitespace() {
        let mut parser = StreamParser::new();
        let events = parser.feed("[EMOTION: HAPPY] [GESTURE: WAVE] [SURPRISED] Chào bạn!");

        assert_eq!(
            events,
            vec![
                ParsedStreamEvent::Emotion("happy".to_string()),
                ParsedStreamEvent::Gesture("wave".to_string()),
                ParsedStreamEvent::Emotion("surprised".to_string()),
                ParsedStreamEvent::SpeakableText("Chào bạn!".to_string()),
            ]
        );
    }

    #[test]
    fn test_stream_parser_custom_registered_gestures() {
        let mut parser = StreamParser::new();
        parser.register_gesture("fly_kiss");

        let events = parser.feed("[fly_kiss] Tạm biệt bạn nhé!");
        assert_eq!(
            events,
            vec![
                ParsedStreamEvent::Gesture("fly_kiss".to_string()),
                ParsedStreamEvent::SpeakableText("Tạm biệt bạn nhé!".to_string()),
            ]
        );
    }

    #[test]
    fn test_stream_parser_unclosed_bracket_exceeding_max_len() {
        let mut parser = StreamParser::new();
        let long_text = "[This bracketed string is intentionally much longer than sixty-four characters and should be treated as text. Tiếp tục câu khác!";
        let events = parser.feed(long_text);

        assert!(!events.is_empty());
        assert!(events.iter().any(|e| matches!(e, ParsedStreamEvent::SpeakableText(_))));
    }
}
