use std::collections::HashSet;
use serde::{Deserialize, Serialize};
use vira_domain::CharacterEmotion;

use crate::chunker::{ChunkerConfig, SentenceChunker};

/// Default LangIns avatar gesture identifiers recognized in shortened tag form (e.g. `[wave]`).
pub const DEFAULT_GESTURES: &[&str] = &[
    "wave", "nod", "think", "idea", "praise", "bow", "clap", "shake", "shrug", "point",
    "salute", "cheer", "laugh", "sigh", "tilt", "wink", "dance", "confused", "idle",
];

/// Events produced during streaming LLM parsing for avatar animation and voice synthesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParsedStreamEvent {
    /// Avatar facial expression change event (e.g. "happy", "surprised").
    Emotion(String),
    /// Avatar body/hand gesture animation trigger event (e.g. "wave", "nod").
    Gesture(String),
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
    custom_gestures: HashSet<String>,
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
            custom_gestures: HashSet::new(),
        }
    }

    /// Creates a `StreamParser` with custom `ChunkerConfig`.
    pub fn with_chunker_config(config: ChunkerConfig) -> Self {
        Self {
            chunker: SentenceChunker::with_config(config),
            pending_raw: String::new(),
            custom_gestures: HashSet::new(),
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
    }

    /// Ingests an incoming token chunk from LLM stream and returns any extracted avatar events
    /// and complete speakable sentence chunks.
    pub fn feed(&mut self, text: &str) -> Vec<ParsedStreamEvent> {
        self.pending_raw.push_str(text);
        let mut events = Vec::new();

        loop {
            let open_idx = self.pending_raw.find('[');
            match open_idx {
                None => {
                    // No open brackets present; the entire pending string is speakable text
                    let text = std::mem::take(&mut self.pending_raw);
                    let chunks = self.chunker.feed(&text);
                    for chunk in chunks {
                        events.push(ParsedStreamEvent::SpeakableText(chunk));
                    }
                    break;
                }
                Some(idx) => {
                    // If there is text before the opening bracket, feed it to the sentence chunker
                    if idx > 0 {
                        let text_before = self.pending_raw[..idx].to_string();
                        let chunks = self.chunker.feed(&text_before);
                        for chunk in chunks {
                            events.push(ParsedStreamEvent::SpeakableText(chunk));
                        }
                        self.pending_raw.drain(..idx);
                    }

                    // Now self.pending_raw starts with '['
                    let close_idx = self.pending_raw.find(']');
                    match close_idx {
                        Some(c_idx) => {
                            let inner = self.pending_raw[1..c_idx].trim();

                            if let Some((kind, val)) = inner.split_once(':') {
                                let kind = kind.trim().to_ascii_lowercase();
                                let val = val.trim().to_ascii_lowercase();
                                if kind == "emotion" && !val.is_empty() {
                                    events.push(ParsedStreamEvent::Emotion(val));
                                    self.pending_raw.drain(..=c_idx);
                                } else if kind == "gesture" && !val.is_empty() {
                                    events.push(ParsedStreamEvent::Gesture(val));
                                    self.pending_raw.drain(..=c_idx);
                                } else {
                                    let non_tag = self.pending_raw[..=c_idx].to_string();
                                    self.pending_raw.drain(..=c_idx);
                                    let chunks = self.chunker.feed(&non_tag);
                                    for chunk in chunks {
                                        events.push(ParsedStreamEvent::SpeakableText(chunk));
                                    }
                                }
                            } else if self.is_known_gesture(inner) {
                                events.push(ParsedStreamEvent::Gesture(inner.to_ascii_lowercase()));
                                self.pending_raw.drain(..=c_idx);
                            } else if self.is_known_emotion(inner) {
                                events.push(ParsedStreamEvent::Emotion(inner.to_ascii_lowercase()));
                                self.pending_raw.drain(..=c_idx);
                            } else {
                                // Not a tag (e.g. citation `[1]` or markdown `[Link]`). Treat as regular text.
                                let non_tag = self.pending_raw[..=c_idx].to_string();
                                self.pending_raw.drain(..=c_idx);
                                let chunks = self.chunker.feed(&non_tag);
                                for chunk in chunks {
                                    events.push(ParsedStreamEvent::SpeakableText(chunk));
                                }
                            }
                        }
                        None => {
                            // Incomplete bracket tag (e.g. `[em` or `[gesture:wa`).
                            // If it exceeds reasonable tag length or spans newlines, it's regular text.
                            if self.pending_raw.len() > 64 || self.pending_raw.contains('\n') {
                                let not_tag = std::mem::take(&mut self.pending_raw);
                                let chunks = self.chunker.feed(&not_tag);
                                for chunk in chunks {
                                    events.push(ParsedStreamEvent::SpeakableText(chunk));
                                }
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

    /// Concludes the stream, flushing any trailing bracket fragments and pending sentence chunks.
    pub fn finish(&mut self) -> Vec<ParsedStreamEvent> {
        let mut events = Vec::new();

        if !self.pending_raw.is_empty() {
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
