//! Semantic sentence-aware text chunker with sliding window overlap.
//!
//! Splits study documents, notes, and lessons into semantic chunks preserving
//! sentence boundaries and maintaining context across chunk transitions.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::Deref;

/// A single extracted document chunk with position metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentChunk {
    /// Unique identifier for this chunk (e.g. "doc123_chunk_0").
    pub id: String,
    /// Parent document identifier.
    pub document_id: String,
    /// Zero-based sequential index within the parent document.
    pub index: usize,
    /// Text content of this chunk.
    pub text: String,
    /// Starting character index in original source text.
    pub start_char: usize,
    /// Ending character index (exclusive) in original source text.
    pub end_char: usize,
}

impl Deref for DocumentChunk {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.text
    }
}

impl AsRef<str> for DocumentChunk {
    fn as_ref(&self) -> &str {
        &self.text
    }
}

impl fmt::Display for DocumentChunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

/// Configuration and engine for sentence-aware sliding window text chunking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunker {
    /// Maximum character length per chunk (default: 500).
    pub max_chunk_size: usize,
    /// Character overlap between adjacent chunks (default: 50).
    pub overlap: usize,
    /// Whether to respect sentence boundaries before splitting words (default: true).
    pub sentence_aware: bool,
}

impl Default for TextChunker {
    fn default() -> Self {
        Self {
            max_chunk_size: 500,
            overlap: 50,
            sentence_aware: true,
        }
    }
}

impl TextChunker {
    /// Creates a new `TextChunker` with specified maximum chunk size and overlap.
    pub fn new(max_chunk_size: usize, overlap: usize) -> Self {
        let max_chunk_size = max_chunk_size.max(10);
        let overlap = overlap.min(max_chunk_size.saturating_sub(1));
        Self {
            max_chunk_size,
            overlap,
            sentence_aware: true,
        }
    }

    /// Toggles sentence boundary awareness.
    pub fn with_sentence_aware(mut self, enabled: bool) -> Self {
        self.sentence_aware = enabled;
        self
    }

    /// Splits `text` into chunks for an anonymous document (`doc_default`).
    pub fn chunk(&self, text: &str) -> Vec<DocumentChunk> {
        self.chunk_document("doc_default", text)
    }

    /// Splits `text` into chunks associated with `document_id`.
    pub fn chunk_document(&self, document_id: &str, text: &str) -> Vec<DocumentChunk> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }

        if !self.sentence_aware {
            return self.chunk_by_characters(document_id, text);
        }

        // 1. Segment text into raw sentence units with character offsets
        let sentences = self.split_into_sentences(text);
        if sentences.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let mut current_sentences: Vec<(usize, usize, &str)> = Vec::new();
        let mut current_len = 0;

        for (start, end, sentence) in sentences {
            let sent_len = sentence.chars().count();

            // If a single sentence exceeds max_chunk_size, flush existing and sub-split
            if sent_len > self.max_chunk_size {
                if !current_sentences.is_empty() {
                    self.flush_chunk(document_id, &mut chunks, &current_sentences);
                    current_sentences.clear();
                    current_len = 0;
                }

                // Sub-split this long sentence on word boundaries
                let sub_chunks = self.split_long_sentence(document_id, chunks.len(), sentence, start);
                chunks.extend(sub_chunks);
                continue;
            }

            if current_len + sent_len > self.max_chunk_size && !current_sentences.is_empty() {
                // Emit chunk
                self.flush_chunk(document_id, &mut chunks, &current_sentences);

                // Prepare sliding window overlap from trailing sentences
                let mut overlap_sentences = Vec::new();
                let mut overlap_len = 0;
                for s in current_sentences.iter().rev() {
                    let slen = s.2.chars().count();
                    if overlap_len + slen <= self.overlap {
                        overlap_sentences.push(*s);
                        overlap_len += slen;
                    } else {
                        break;
                    }
                }
                overlap_sentences.reverse();
                current_sentences = overlap_sentences;
                current_len = overlap_len;
            }

            current_sentences.push((start, end, sentence));
            current_len += sent_len;
        }

        if !current_sentences.is_empty() {
            self.flush_chunk(document_id, &mut chunks, &current_sentences);
        }

        chunks
    }

    fn flush_chunk(
        &self,
        document_id: &str,
        chunks: &mut Vec<DocumentChunk>,
        sentences: &[(usize, usize, &str)],
    ) {
        if sentences.is_empty() {
            return;
        }
        let start_char = sentences.first().unwrap().0;
        let end_char = sentences.last().unwrap().1;
        let text: String = sentences
            .iter()
            .map(|(_, _, s)| s.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        if text.trim().is_empty() {
            return;
        }

        let index = chunks.len();
        chunks.push(DocumentChunk {
            id: format!("{document_id}_chunk_{index}"),
            document_id: document_id.to_string(),
            index,
            text,
            start_char,
            end_char,
        });
    }

    /// Splits text into sentences respecting Vietnamese, English, and CJK punctuation, decimals, and abbreviations.
    pub fn split_into_sentences<'a>(&self, text: &'a str) -> Vec<(usize, usize, &'a str)> {
        let mut results = Vec::new();
        let chars_info: Vec<(usize, usize, char)> = text
            .char_indices()
            .enumerate()
            .map(|(char_idx, (byte_idx, ch))| (char_idx, byte_idx, ch))
            .collect();
        let total = chars_info.len();

        let mut start_char_idx = 0;
        let mut start_byte_idx = 0;

        let mut i = 0;
        while i < total {
            let (_, _byte_idx, ch) = chars_info[i];

            // Potential sentence terminators: '.', '!', '?', newline, or CJK punctuation
            let is_terminator = ch == '.'
                || ch == '!'
                || ch == '?'
                || ch == '\n'
                || ch == '。'
                || ch == '！'
                || ch == '？'
                || ch == '…';

            if is_terminator {
                if ch == '.' {
                    // 1. Decimal number check (e.g. 3.14 or 98.5%)
                    let prev_is_digit = i > 0 && chars_info[i - 1].2.is_ascii_digit();
                    let next_is_digit = i + 1 < total && chars_info[i + 1].2.is_ascii_digit();
                    if prev_is_digit && next_is_digit {
                        i += 1;
                        continue;
                    }

                    // 2. Repeated dot check (ellipsis like "...")
                    if i + 1 < total && chars_info[i + 1].2 == '.' {
                        while i + 1 < total && chars_info[i + 1].2 == '.' {
                            i += 1;
                        }
                    }

                    // 3. Check for abbreviations (whole word / token match)
                    if self.is_abbreviation_dot(&chars_info, i, total, text) {
                        i += 1;
                        continue;
                    }
                }

                // Sentence boundary confirmed
                let end_char_idx = i + 1;
                let end_byte_idx = if i + 1 < total {
                    chars_info[i + 1].1
                } else {
                    text.len()
                };

                let raw_slice = &text[start_byte_idx..end_byte_idx];
                let trimmed = raw_slice.trim();

                if !trimmed.is_empty() {
                    // Compute accurate character offsets for trimmed sentence
                    let leading_ws_count = raw_slice
                        .chars()
                        .take_while(|c| c.is_whitespace())
                        .count();
                    let trimmed_char_count = trimmed.chars().count();

                    let sentence_start_char = start_char_idx + leading_ws_count;
                    let sentence_end_char = sentence_start_char + trimmed_char_count;

                    results.push((sentence_start_char, sentence_end_char, trimmed));
                }

                start_char_idx = end_char_idx;
                start_byte_idx = end_byte_idx;
            }

            i += 1;
        }

        if start_byte_idx < text.len() {
            let raw_slice = &text[start_byte_idx..];
            let trimmed = raw_slice.trim();
            if !trimmed.is_empty() {
                let leading_ws_count = raw_slice
                    .chars()
                    .take_while(|c| c.is_whitespace())
                    .count();
                let trimmed_char_count = trimmed.chars().count();
                let sentence_start_char = start_char_idx + leading_ws_count;
                let sentence_end_char = sentence_start_char + trimmed_char_count;

                results.push((sentence_start_char, sentence_end_char, trimmed));
            }
        }

        results
    }

    /// Checks if a dot at index `dot_idx` is part of an abbreviation.
    fn is_abbreviation_dot(
        &self,
        chars: &[(usize, usize, char)],
        dot_idx: usize,
        total: usize,
        text: &str,
    ) -> bool {
        // Find token start (stop at whitespace or punctuation like '(' or quote)
        let mut token_start = dot_idx;
        while token_start > 0 {
            let prev_ch = chars[token_start - 1].2;
            if prev_ch.is_whitespace() || matches!(prev_ch, '(' | '[' | '{' | '"' | '\'' | '“' | '«') {
                break;
            }
            token_start -= 1;
        }

        let word_chars: String = chars[token_start..dot_idx].iter().map(|(_, _, c)| *c).collect();
        let word_lower = word_chars.to_lowercase();

        // Multi-dot abbreviations check (e.g. "v.v.", "e.g.", "i.e.")
        let next_ch = if dot_idx + 1 < total { Some(chars[dot_idx + 1].2) } else { None };

        // Check if token starts with "v.v", "e.g", "i.e"
        let full_token_byte_start = chars[token_start].1;
        let token_slice = &text[full_token_byte_start..];
        let token_lower = token_slice.to_lowercase();

        for multidot in &["v.v.", "v.v", "e.g.", "e.g", "i.e.", "i.e"] {
            if token_lower.starts_with(multidot) {
                // If dot_idx is the first dot, never split
                if word_lower == "v" || word_lower == "e" || word_lower == "i" {
                    return true;
                }
                // If dot_idx is the second dot, check if followed by lowercase letter
                if let Some(next) = next_ch {
                    if next == ' ' {
                        if let Some(after_space) = chars.get(dot_idx + 2) {
                            if after_space.2.is_lowercase() || after_space.2 == ',' {
                                return true;
                            }
                        }
                    } else if next.is_alphabetic() || next == ',' {
                        return true;
                    }
                }
            }
        }

        // Single-dot titles and abbreviations:
        // English: mr, mrs, ms, dr, prof, sr, jr, vs, etc
        // Vietnamese: ts, ths, tp, gs, thay, co
        let known_abbrevs = [
            "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
            "ts", "ths", "tp", "gs",
        ];

        if known_abbrevs.contains(&word_lower.as_str()) {
            // If it's a known abbreviation:
            // Check if followed by space, or uppercase (e.g. "TP.HCM"), or digit
            if let Some(next) = next_ch {
                if next == ' ' || next.is_uppercase() || next.is_ascii_digit() || next == ',' {
                    return true;
                }
            }
        }

        false
    }

    fn split_long_sentence(
        &self,
        document_id: &str,
        start_index: usize,
        sentence: &str,
        sentence_start_char: usize,
    ) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();

        // Extract words with character offsets relative to sentence
        let mut word_entries: Vec<(usize, usize, &str)> = Vec::new();
        let chars_info: Vec<(usize, usize, char)> = sentence
            .char_indices()
            .enumerate()
            .map(|(c_idx, (b_idx, ch))| (c_idx, b_idx, ch))
            .collect();
        let total_chars = chars_info.len();

        let mut in_word = false;
        let mut w_start_char = 0;
        let mut w_start_byte = 0;

        for (c_idx, b_idx, ch) in &chars_info {
            if ch.is_whitespace() {
                if in_word {
                    let word_str = &sentence[w_start_byte..*b_idx];
                    word_entries.push((w_start_char, *c_idx, word_str));
                    in_word = false;
                }
            } else if !in_word {
                in_word = true;
                w_start_char = *c_idx;
                w_start_byte = *b_idx;
            }
        }
        if in_word {
            let word_str = &sentence[w_start_byte..];
            word_entries.push((w_start_char, total_chars, word_str));
        }

        if word_entries.is_empty() {
            return chunks;
        }

        let mut current_words: Vec<(usize, usize, &str)> = Vec::new();
        let mut current_len = 0;
        let mut chunk_idx = start_index;

        for (w_start, w_end, word) in word_entries {
            let wlen = w_end - w_start;

            if current_len + wlen + 1 > self.max_chunk_size && !current_words.is_empty() {
                let first_word = current_words.first().unwrap();
                let last_word = current_words.last().unwrap();
                let chunk_text = current_words
                    .iter()
                    .map(|(_, _, w)| *w)
                    .collect::<Vec<_>>()
                    .join(" ");

                chunks.push(DocumentChunk {
                    id: format!("{document_id}_chunk_{chunk_idx}"),
                    document_id: document_id.to_string(),
                    index: chunk_idx,
                    text: chunk_text,
                    start_char: sentence_start_char + first_word.0,
                    end_char: sentence_start_char + last_word.1,
                });
                chunk_idx += 1;

                // Carry over words within overlap
                let mut overlap_words = Vec::new();
                let mut overlap_len = 0;
                for w in current_words.iter().rev() {
                    let w_size = w.1 - w.0;
                    if overlap_len + w_size < self.overlap {
                        overlap_words.push(*w);
                        overlap_len += w_size + 1;
                    } else {
                        break;
                    }
                }
                overlap_words.reverse();
                current_words = overlap_words;
                current_len = overlap_len;
            }

            current_words.push((w_start, w_end, word));
            current_len += wlen + 1;
        }

        if !current_words.is_empty() {
            let first_word = current_words.first().unwrap();
            let last_word = current_words.last().unwrap();
            let chunk_text = current_words
                .iter()
                .map(|(_, _, w)| *w)
                .collect::<Vec<_>>()
                .join(" ");

            chunks.push(DocumentChunk {
                id: format!("{document_id}_chunk_{chunk_idx}"),
                document_id: document_id.to_string(),
                index: chunk_idx,
                text: chunk_text,
                start_char: sentence_start_char + first_word.0,
                end_char: sentence_start_char + last_word.1,
            });
        }

        chunks
    }

    fn chunk_by_characters(&self, document_id: &str, text: &str) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let step = if self.max_chunk_size > self.overlap {
            self.max_chunk_size - self.overlap
        } else {
            self.max_chunk_size
        };

        let mut i = 0;
        let mut index = 0;
        while i < chars.len() {
            let end = (i + self.max_chunk_size).min(chars.len());
            let chunk_str: String = chars[i..end].iter().collect();
            chunks.push(DocumentChunk {
                id: format!("{document_id}_chunk_{index}"),
                document_id: document_id.to_string(),
                index,
                text: chunk_str,
                start_char: i,
                end_char: end,
            });
            index += 1;
            i += step;
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunker_basic_sentences() {
        let chunker = TextChunker::new(50, 10);
        let chunks = chunker.chunk("Một hai ba. Bốn năm sáu. Bảy tám chín.");
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].document_id, "doc_default");
    }

    #[test]
    fn test_chunker_abbreviation_not_split() {
        let chunker = TextChunker::new(200, 20);
        let text = "Gặp TS. Nguyễn lúc 9h sáng tại TP. Hồ Chí Minh. Buổi gặp rất bổ ích!";
        let sentences = chunker.split_into_sentences(text);
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn test_chunker_empty_input() {
        let chunker = TextChunker::new(100, 20);
        assert!(chunker.chunk("").is_empty());
        assert!(chunker.chunk("   \n\t  ").is_empty());
    }
}
