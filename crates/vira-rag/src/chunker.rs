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

    /// Splits text into sentences respecting Vietnamese and English punctuation, decimals, and abbreviations.
    pub fn split_into_sentences<'a>(&self, text: &'a str) -> Vec<(usize, usize, &'a str)> {
        let mut results = Vec::new();
        let mut start_idx = 0;
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let total = chars.len();

        let mut i = 0;
        while i < total {
            let (_, ch) = chars[i];

            // Potential sentence terminators: '.', '!', '?', or newline
            if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
                // Check if this is an abbreviation, decimal number, or ellipsis
                if ch == '.' {
                    // Decimal number check (e.g. 3.14)
                    let prev_is_digit = i > 0 && chars[i - 1].1.is_ascii_digit();
                    let next_is_digit = i + 1 < total && chars[i + 1].1.is_ascii_digit();
                    if prev_is_digit && next_is_digit {
                        i += 1;
                        continue;
                    }

                    // Repeated dot check (...)
                    if i + 1 < total && chars[i + 1].1 == '.' {
                        while i + 1 < total && chars[i + 1].1 == '.' {
                            i += 1;
                        }
                    }

                    // Abbreviation check (e.g., "Mr.", "Dr.", "e.g.", "v.v.")
                    let lookback_start = i.saturating_sub(10);
                    let prior_slice: String = chars[lookback_start..i].iter().map(|(_, c)| *c).collect();
                    let is_abbrev = ["mr", "ms", "mrs", "dr", "prof", "tp", "ths", "ts", "etc", "vs", "eg", "ie", "vv"]
                        .iter()
                        .any(|&abbr| {
                            let lower = prior_slice.to_lowercase();
                            lower.ends_with(abbr)
                        });
                    if is_abbrev && i + 1 < total && chars[i + 1].1 == ' ' {
                        i += 1;
                        continue;
                    }
                }

                // Sentence boundary confirmed
                let end_pos = if i + 1 < total {
                    chars[i + 1].0
                } else {
                    text.len()
                };

                let slice = &text[start_idx..end_pos];
                if !slice.trim().is_empty() {
                    results.push((start_idx, end_pos, slice.trim()));
                }
                start_idx = end_pos;
            }
            i += 1;
        }

        if start_idx < text.len() {
            let slice = &text[start_idx..];
            if !slice.trim().is_empty() {
                results.push((start_idx, text.len(), slice.trim()));
            }
        }

        results
    }

    fn split_long_sentence(
        &self,
        document_id: &str,
        start_index: usize,
        sentence: &str,
        offset_base: usize,
    ) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let mut current_words = Vec::new();
        let mut current_len = 0;
        let mut chunk_idx = start_index;

        for word in words {
            let wlen = word.chars().count();
            if current_len + wlen + 1 > self.max_chunk_size && !current_words.is_empty() {
                let text = current_words.join(" ");
                chunks.push(DocumentChunk {
                    id: format!("{document_id}_chunk_{chunk_idx}"),
                    document_id: document_id.to_string(),
                    index: chunk_idx,
                    text,
                    start_char: offset_base,
                    end_char: offset_base + sentence.len(),
                });
                chunk_idx += 1;
                current_words.clear();
                current_len = 0;
            }
            current_words.push(word);
            current_len += wlen + 1;
        }

        if !current_words.is_empty() {
            let text = current_words.join(" ");
            chunks.push(DocumentChunk {
                id: format!("{document_id}_chunk_{chunk_idx}"),
                document_id: document_id.to_string(),
                index: chunk_idx,
                text,
                start_char: offset_base,
                end_char: offset_base + sentence.len(),
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
