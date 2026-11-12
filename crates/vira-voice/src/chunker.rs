use serde::{Deserialize, Serialize};

/// Configuration parameters for streaming sentence and clause chunking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkerConfig {
    /// Punctuation marks that strictly terminate a full sentence.
    /// Default: `['.', '!', '?', '\n', '。', '！', '？']`
    pub sentence_delimiters: Vec<char>,

    /// Punctuation marks that can split a clause for early TTFA (<1.0s).
    /// Default: `[',', ';', ':', '—']`
    pub clause_delimiters: Vec<char>,

    /// Whether to split on clause delimiters (comma, semicolon) for lower TTFA.
    /// Default: true
    pub split_on_clauses: bool,

    /// Minimum character count before splitting on a clause delimiter.
    /// Avoids breaking micro-fragments like "Ừ," or "Vâng," into tiny TTS audio chunks.
    /// Default: 12
    pub min_clause_chars: usize,

    /// Maximum character limit before forcing a boundary on the last word break.
    /// Default: 160
    pub max_chunk_chars: usize,

    /// Common abbreviations that should NOT trigger a sentence split on period.
    pub abbreviations: Vec<String>,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            sentence_delimiters: vec!['.', '!', '?', '\n', '。', '！', '？'],
            clause_delimiters: vec![',', ';', ':', '—'],
            split_on_clauses: true,
            min_clause_chars: 12,
            max_chunk_chars: 160,
            abbreviations: vec![
                "mr".to_string(),
                "mrs".to_string(),
                "ms".to_string(),
                "dr".to_string(),
                "prof".to_string(),
                "sr".to_string(),
                "jr".to_string(),
                "vs".to_string(),
                "etc".to_string(),
                "tp".to_string(),
                "ths".to_string(),
                "ts".to_string(),
                "eg".to_string(),
                "ie".to_string(),
            ],
        }
    }
}

/// Streaming sentence and clause chunker for realtime TTS audio generation.
///
/// Buffers streaming token fragments and cuts them into grammatically sound sentences
/// or clauses to minimize Time-to-First-Audio (TTFA < 1.0s).
#[derive(Debug, Clone, Default)]
pub struct SentenceChunker {
    config: ChunkerConfig,
    buffer: String,
}

impl SentenceChunker {
    /// Creates a new `SentenceChunker` with default configuration.
    pub fn new() -> Self {
        Self {
            config: ChunkerConfig::default(),
            buffer: String::new(),
        }
    }

    /// Creates a new `SentenceChunker` with custom configuration.
    pub fn with_config(config: ChunkerConfig) -> Self {
        Self {
            config,
            buffer: String::new(),
        }
    }

    /// Returns a reference to the active chunker configuration.
    pub fn config(&self) -> &ChunkerConfig {
        &self.config
    }

    /// Returns a mutable reference to the active chunker configuration.
    pub fn config_mut(&mut self) -> &mut ChunkerConfig {
        &mut self.config
    }

    /// Returns the currently pending (un-flushed) buffered text.
    pub fn pending_text(&self) -> &str {
        &self.buffer
    }

    /// Returns true if the internal buffer is completely empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Resets all internal buffers.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Ingests a streaming text fragment and returns any complete sentence or clause chunks.
    pub fn feed(&mut self, text: &str) -> Vec<String> {
        self.buffer.push_str(text);
        let mut chunks = Vec::new();

        while let Some((chunk_end, next_start)) = self.find_next_chunk() {
            let chunk = self.buffer[..chunk_end].trim().to_string();
            self.buffer.drain(..next_start);
            if !chunk.is_empty() {
                chunks.push(chunk);
            }
        }

        chunks
    }

    /// Flushes any remaining buffered text when the LLM stream concludes.
    pub fn finish(&mut self) -> Vec<String> {
        let mut chunks = Vec::new();

        while let Some((chunk_end, next_start)) = self.find_next_chunk() {
            let chunk = self.buffer[..chunk_end].trim().to_string();
            self.buffer.drain(..next_start);
            if !chunk.is_empty() {
                chunks.push(chunk);
            }
        }

        let remaining = self.buffer.trim().to_string();
        self.buffer.clear();
        if !remaining.is_empty() {
            chunks.push(remaining);
        }

        chunks
    }

    /// Alias for `finish()`.
    pub fn flush(&mut self) -> Vec<String> {
        self.finish()
    }

    /// Scans `self.buffer` to find the next split point (chunk_end_byte, next_start_byte).
    fn find_next_chunk(&self) -> Option<(usize, usize)> {
        let text = &self.buffer;

        // Skip leading whitespace to locate the start of the chunk
        let start_byte = match text.char_indices().find(|(_, c)| !c.is_whitespace()) {
            Some((idx, _)) => idx,
            None => return None,
        };

        let search_slice = &text[start_byte..];

        for (offset, ch) in search_slice.char_indices() {
            let curr_byte = start_byte + offset;
            let after_curr = curr_byte + ch.len_utf8();

            // 1. Newlines always force a chunk boundary
            if ch == '\n' || ch == '\r' {
                let chunk_end = curr_byte;
                let next_start = text[after_curr..]
                    .char_indices()
                    .find(|(_, c)| !c.is_whitespace())
                    .map(|(i, _)| after_curr + i)
                    .unwrap_or(text.len());
                return Some((chunk_end, next_start));
            }

            // 2. Strong sentence delimiters (. ! ? etc.)
            if self.config.sentence_delimiters.contains(&ch) {
                let prev_char = text[..curr_byte].chars().last();
                let next_char = text[after_curr..].chars().next();

                // Numbers with decimals like 3.14
                if ch == '.' {
                    if let (Some(p), Some(n)) = (prev_char, next_char) {
                        if p.is_ascii_digit() && n.is_ascii_digit() {
                            continue;
                        }
                    }
                    if prev_char.is_some_and(|p| p.is_ascii_digit()) && next_char.is_none() {
                        // Could be mid-stream decimal number e.g. "3." waiting for "14"
                        return None;
                    }

                    // Abbreviation check (e.g. Dr., tp., Prof.)
                    let prefix = &text[start_byte..curr_byte];
                    let word = prefix
                        .split(|c: char| c.is_whitespace() || c == '(' || c == '[' || c == '"' || c == '\'')
                        .next_back()
                        .unwrap_or("");
                    let clean_word = word
                        .trim_matches(|c: char| !c.is_alphabetic())
                        .to_ascii_lowercase();

                    if self
                        .config
                        .abbreviations
                        .iter()
                        .any(|abbr| abbr.eq_ignore_ascii_case(&clean_word))
                    {
                        continue;
                    }
                }

                // Consume consecutive delimiters (e.g. "...", "!!!", "?!")
                let mut delim_end = after_curr;
                while let Some((_, next_delim)) = text[delim_end..].char_indices().next() {
                    if self.config.sentence_delimiters.contains(&next_delim)
                        || self.config.clause_delimiters.contains(&next_delim)
                    {
                        delim_end += next_delim.len_utf8();
                    } else {
                        break;
                    }
                }

                // Consume closing quotes or brackets (e.g. ." or !))
                let mut chunk_end = delim_end;
                while let Some((_, closing)) = text[chunk_end..].char_indices().next() {
                    if is_closing_bracket_or_quote(closing) {
                        chunk_end += closing.len_utf8();
                    } else {
                        break;
                    }
                }

                let next_start = text[chunk_end..]
                    .char_indices()
                    .find(|(_, c)| !c.is_whitespace())
                    .map(|(i, _)| chunk_end + i)
                    .unwrap_or(text.len());

                return Some((chunk_end, next_start));
            }

            // 3. Clause delimiters (, ; : —) for lower TTFA
            if self.config.split_on_clauses && self.config.clause_delimiters.contains(&ch) {
                let prev_char = text[..curr_byte].chars().last();
                let next_char = text[after_curr..].chars().next();

                // Numbers with commas like 100,000
                if ch == ',' {
                    if let (Some(p), Some(n)) = (prev_char, next_char) {
                        if p.is_ascii_digit() && n.is_ascii_digit() {
                            continue;
                        }
                    }
                    if prev_char.is_some_and(|p| p.is_ascii_digit()) && next_char.is_none() {
                        return None;
                    }
                }

                let clause_chars = text[start_byte..curr_byte].chars().count();
                if clause_chars >= self.config.min_clause_chars {
                    let mut chunk_end = after_curr;
                    while let Some((_, closing)) = text[chunk_end..].char_indices().next() {
                        if is_closing_bracket_or_quote(closing) {
                            chunk_end += closing.len_utf8();
                        } else {
                            break;
                        }
                    }

                    let next_start = text[chunk_end..]
                        .char_indices()
                        .find(|(_, c)| !c.is_whitespace())
                        .map(|(i, _)| chunk_end + i)
                        .unwrap_or(text.len());

                    return Some((chunk_end, next_start));
                }
            }

            // 4. Max length safeguard for run-on sentences
            let current_chars = text[start_byte..curr_byte].chars().count();
            if current_chars >= self.config.max_chunk_chars {
                if let Some(last_ws_offset) = text[start_byte..curr_byte].rfind(char::is_whitespace) {
                    let chunk_end = start_byte + last_ws_offset;
                    let next_start = text[chunk_end..]
                        .char_indices()
                        .find(|(_, c)| !c.is_whitespace())
                        .map(|(i, _)| chunk_end + i)
                        .unwrap_or(text.len());
                    return Some((chunk_end, next_start));
                } else {
                    return Some((curr_byte, curr_byte));
                }
            }
        }

        None
    }
}

fn is_closing_bracket_or_quote(c: char) -> bool {
    matches!(c, '"' | '\'' | '”' | '’' | '»' | ')' | ']' | '}')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunker_empty_and_whitespace() {
        let mut chunker = SentenceChunker::new();
        assert!(chunker.is_empty());
        assert_eq!(chunker.feed(""), Vec::<String>::new());
        assert_eq!(chunker.feed("   \t  "), Vec::<String>::new());
        assert_eq!(chunker.finish(), Vec::<String>::new());
    }

    #[test]
    fn test_chunker_pending_text_and_reset() {
        let mut chunker = SentenceChunker::new();
        let _ = chunker.feed("Đang chờ xử lý");
        assert_eq!(chunker.pending_text(), "Đang chờ xử lý");
        assert!(!chunker.is_empty());

        chunker.reset();
        assert!(chunker.is_empty());
        assert_eq!(chunker.pending_text(), "");
    }

    #[test]
    fn test_chunker_config_serde_roundtrip() {
        let config = ChunkerConfig::default();
        let json = serde_json::to_string(&config).expect("serialize config");
        let deserialized: ChunkerConfig = serde_json::from_str(&json).expect("deserialize config");
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_chunker_crlf_newlines() {
        let mut chunker = SentenceChunker::new();
        let chunks = chunker.feed("Dòng 1\r\nDòng 2\r\nDòng 3");
        assert_eq!(chunks, vec!["Dòng 1", "Dòng 2"]);

        let flushed = chunker.finish();
        assert_eq!(flushed, vec!["Dòng 3"]);
    }

    #[test]
    fn test_chunker_run_on_sentence_without_spaces() {
        let config = ChunkerConfig {
            max_chunk_chars: 20,
            split_on_clauses: false,
            ..Default::default()
        };
        let mut chunker = SentenceChunker::with_config(config);
        let huge_word = "A".repeat(50);
        let chunks = chunker.feed(&huge_word);
        assert!(!chunks.is_empty());
        let flushed = chunker.finish();
        assert!(!flushed.is_empty());
    }
}
