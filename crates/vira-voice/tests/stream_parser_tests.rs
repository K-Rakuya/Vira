use vira_voice::chunker::{ChunkerConfig, SentenceChunker};
use vira_voice::stream_parser::{ParsedStreamEvent, StreamParser};

#[test]
fn test_stream_parser_extracts_tags_and_sentences() {
    let mut parser = StreamParser::new();
    let events = parser.feed("[emotion:happy][gesture:wave] Xin chào bạn! Tôi là Vira. [emotion:neutral]");

    assert!(events.contains(&ParsedStreamEvent::Emotion("happy".to_string())));
    assert!(events.contains(&ParsedStreamEvent::Gesture("wave".to_string())));
    assert!(events.contains(&ParsedStreamEvent::SpeakableText("Xin chào bạn!".to_string())));
    assert!(events.contains(&ParsedStreamEvent::SpeakableText("Tôi là Vira.".to_string())));
    assert!(events.contains(&ParsedStreamEvent::Emotion("neutral".to_string())));
}

#[test]
fn test_stream_parser_shortened_gesture_and_emotion_tags() {
    let mut parser = StreamParser::new();
    let events = parser.feed("[wave] Chào buổi sáng! [nod] Hôm nay thời tiết thế nào? [think]");

    assert_eq!(
        events,
        vec![
            ParsedStreamEvent::Gesture("wave".to_string()),
            ParsedStreamEvent::SpeakableText("Chào buổi sáng!".to_string()),
            ParsedStreamEvent::Gesture("nod".to_string()),
            ParsedStreamEvent::SpeakableText("Hôm nay thời tiết thế nào?".to_string()),
            ParsedStreamEvent::Gesture("think".to_string()),
        ]
    );

    let mut parser2 = StreamParser::new();
    let events2 = parser2.feed("[surprised] Thật không thể tin được! [praise]");
    assert_eq!(
        events2,
        vec![
            ParsedStreamEvent::Emotion("surprised".to_string()),
            ParsedStreamEvent::SpeakableText("Thật không thể tin được!".to_string()),
            ParsedStreamEvent::Gesture("praise".to_string()),
        ]
    );
}

#[test]
fn test_stream_parser_chunked_streaming_feed() {
    let mut parser = StreamParser::new();

    // LLM tokens arriving incrementally
    let c1 = parser.feed("[em");
    assert!(c1.is_empty());

    let c2 = parser.feed("otion:happy] Xin ");
    assert_eq!(c2, vec![ParsedStreamEvent::Emotion("happy".to_string())]);

    let c3 = parser.feed("chào bạn! [ges");
    assert_eq!(
        c3,
        vec![ParsedStreamEvent::SpeakableText("Xin chào bạn!".to_string())]
    );

    let c4 = parser.feed("ture:nod] Rất vui ");
    assert_eq!(c4, vec![ParsedStreamEvent::Gesture("nod".to_string())]);

    let c5 = parser.feed("được gặp bạn. ");
    assert_eq!(
        c5,
        vec![ParsedStreamEvent::SpeakableText("Rất vui được gặp bạn.".to_string())]
    );

    let c6 = parser.feed("Hẹn gặp lại nhé");
    assert!(c6.is_empty()); // No punctuation yet

    let c7 = parser.finish();
    assert_eq!(
        c7,
        vec![ParsedStreamEvent::SpeakableText("Hẹn gặp lại nhé".to_string())]
    );
}

#[test]
fn test_stream_parser_preserves_non_tag_brackets() {
    let mut parser = StreamParser::new();
    let events = parser.feed("Tham khảo tài liệu [1] và [Rust Book] để biết thêm chi tiết.");

    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0],
        ParsedStreamEvent::SpeakableText("Tham khảo tài liệu [1] và [Rust Book] để biết thêm chi tiết.".to_string())
    );
}

#[test]
fn test_stream_parser_unclosed_tag_at_finish() {
    let mut parser = StreamParser::new();
    let c1 = parser.feed("Đang xử lý dữ liệu [chưa xong");
    assert!(c1.is_empty());

    let c2 = parser.finish();
    assert_eq!(
        c2,
        vec![ParsedStreamEvent::SpeakableText("Đang xử lý dữ liệu [chưa xong".to_string())]
    );
}

#[test]
fn test_stream_parser_reset() {
    let mut parser = StreamParser::new();
    let _ = parser.feed("[emotion:happy] Đoạn này bỏ dở");
    parser.reset();

    let events = parser.feed("Câu mới hoàn chỉnh!");
    assert_eq!(
        events,
        vec![ParsedStreamEvent::SpeakableText("Câu mới hoàn chỉnh!".to_string())]
    );
}

#[test]
fn test_numbered_lists_not_split() {
    let mut chunker = SentenceChunker::new();
    let chunks = chunker.feed("1. Bước đầu tiên là chuẩn bị.\n2. Bước thứ hai là thực hiện.");
    assert_eq!(
        chunks,
        vec![
            "1. Bước đầu tiên là chuẩn bị.",
            "2. Bước thứ hai là thực hiện."
        ]
    );
}

#[test]
fn test_timestamp_with_colon_not_split() {
    let mut chunker = SentenceChunker::new();
    let chunks = chunker.feed("Cuộc họp sẽ bắt đầu lúc 9:30 sáng nay tại phòng họp.");
    assert_eq!(
        chunks,
        vec!["Cuộc họp sẽ bắt đầu lúc 9:30 sáng nay tại phòng họp."]
    );
}

#[test]
fn test_pure_punctuation_leading_ellipsis() {
    let mut chunker = SentenceChunker::new();
    let chunks = chunker.feed("... Chào bạn! Tôi là Vira.");
    assert_eq!(
        chunks,
        vec!["... Chào bạn!", "Tôi là Vira."]
    );
}

#[test]
fn test_vietnamese_multi_dot_abbreviations() {
    let mut chunker = SentenceChunker::new();
    let chunks = chunker.feed("Hôm nay chúng ta gặp Th.S. Nam và PGS.TS. Bình. Buổi gặp rất quan trọng!");
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0], "Hôm nay chúng ta gặp Th.S. Nam và PGS.TS. Bình.");
    assert_eq!(chunks[1], "Buổi gặp rất quan trọng!");
}

#[test]
fn test_unclosed_bracket_does_not_swallow_subsequent_tag() {
    let mut parser = StreamParser::new();
    let events = parser.feed("[ghi chú [wave] Xin chào bạn!");
    assert!(
        events.contains(&ParsedStreamEvent::Gesture("wave".to_string())),
        "Tag [wave] after unclosed bracket must not be swallowed"
    );
}

#[test]
fn test_words_not_smashed_around_tags() {
    let mut parser = StreamParser::new();
    let events = parser.feed("Xin chào[wave]bạn nhé!");
    let flushed = parser.finish();
    let all_events = [events, flushed].concat();
    assert!(
        all_events.contains(&ParsedStreamEvent::Gesture("wave".to_string())),
        "Tag should be extracted"
    );
    assert!(
        all_events.contains(&ParsedStreamEvent::SpeakableText("Xin chào bạn nhé!".to_string())),
        "Words 'chào' and 'bạn' should have a space separating them, not 'chàobạn'"
    );
}

#[test]
fn test_sentence_chunker_basic_delimiters() {
    let mut chunker = SentenceChunker::new();
    let chunks = chunker.feed("Câu thứ nhất. Câu thứ hai! Câu thứ ba? Dòng mới\nTiếp tục câu.");

    assert_eq!(
        chunks,
        vec![
            "Câu thứ nhất.",
            "Câu thứ hai!",
            "Câu thứ ba?",
            "Dòng mới",
            "Tiếp tục câu."
        ]
    );
}

#[test]
fn test_sentence_chunker_numbers_and_decimals_not_split() {
    let mut chunker = SentenceChunker::new();
    let chunks = chunker.feed("Giá trị Pi là 3.14159 và giá sản phẩm là 100,000 VND thì rất tốt. Bạn có hiểu không?");

    assert_eq!(chunks.len(), 2);
    assert_eq!(
        chunks[0],
        "Giá trị Pi là 3.14159 và giá sản phẩm là 100,000 VND thì rất tốt."
    );
    assert_eq!(chunks[1], "Bạn có hiểu không?");
}

#[test]
fn test_sentence_chunker_abbreviations_not_split() {
    let mut chunker = SentenceChunker::new();
    let chunks = chunker.feed("Gặp Dr. John lúc 9h sáng tại tp. HCM. Buổi họp rất quan trọng!");

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0], "Gặp Dr. John lúc 9h sáng tại tp. HCM.");
    assert_eq!(chunks[1], "Buổi họp rất quan trọng!");
}

#[test]
fn test_sentence_chunker_ellipsis_and_repeated_punctuation() {
    let mut chunker = SentenceChunker::new();
    let chunks = chunker.feed("Chờ một lát... Thật tuyệt vời!!! Có đúng không?! Đúng vậy.");

    assert_eq!(
        chunks,
        vec![
            "Chờ một lát...",
            "Thật tuyệt vời!!!",
            "Có đúng không?!",
            "Đúng vậy."
        ]
    );
}

#[test]
fn test_sentence_chunker_clause_delimiters_for_ttfa() {
    let config = ChunkerConfig {
        split_on_clauses: true,
        min_clause_chars: 10,
        ..Default::default()
    };
    let mut chunker = SentenceChunker::with_config(config);

    // Clause delimiter ',' with length >= 10 should split
    let chunks = chunker.feed("Xin chào bạn, tôi là trợ lý ảo Vira; rất hân hạnh hỗ trợ bạn.");
    assert_eq!(
        chunks,
        vec![
            "Xin chào bạn,",
            "tôi là trợ lý ảo Vira;",
            "rất hân hạnh hỗ trợ bạn."
        ]
    );
}

#[test]
fn test_sentence_chunker_short_clauses_not_split() {
    let config = ChunkerConfig {
        split_on_clauses: true,
        min_clause_chars: 15,
        ..Default::default()
    };
    let mut chunker = SentenceChunker::with_config(config);

    // "Ừ," is only 2 chars, below min_clause_chars: 15, so it should not split at ','
    let chunks = chunker.feed("Ừ, tôi đã hiểu điều bạn vừa nói.");
    assert_eq!(chunks, vec!["Ừ, tôi đã hiểu điều bạn vừa nói."]);
}

#[test]
fn test_sentence_chunker_quotes_and_brackets_attached_to_sentence() {
    let mut chunker = SentenceChunker::new();
    let chunks = chunker.feed("Cô ấy nói: \"Xin chào!\" Rồi mỉm cười.");

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0], "Cô ấy nói: \"Xin chào!\"");
    assert_eq!(chunks[1], "Rồi mỉm cười.");
}

#[test]
fn test_sentence_chunker_max_length_safeguard() {
    let config = ChunkerConfig {
        max_chunk_chars: 40,
        split_on_clauses: false,
        ..Default::default()
    };
    let mut chunker = SentenceChunker::with_config(config);

    // A run-on sentence without punctuation should split at the last space before max_chunk_chars
    let chunks = chunker.feed("Đây là một chuỗi văn bản rất dài không hề có dấu chấm phẩy gì cả để kiểm tra");
    assert!(!chunks.is_empty());
    for c in &chunks {
        assert!(c.chars().count() <= 40);
    }
}

#[test]
fn test_unclosed_bracket_with_punctuation_does_not_stall() {
    let mut parser = StreamParser::new();
    // Contains punctuation inside unclosed bracket; should NOT stall waiting for 48 chars or finish
    let events = parser.feed("[Draft 1.0! Xin chào bạn.");
    assert_eq!(
        events,
        vec![
            ParsedStreamEvent::SpeakableText("[Draft 1.0!".to_string()),
            ParsedStreamEvent::SpeakableText("Xin chào bạn.".to_string())
        ]
    );
}

#[test]
fn test_initial_abbreviations_rowling() {
    let mut chunker = SentenceChunker::new();
    let chunks = chunker.feed("Tác giả J. K. Rowling đã viết tác phẩm này. Bạn đã đọc chưa?");
    assert_eq!(
        chunks,
        vec![
            "Tác giả J. K. Rowling đã viết tác phẩm này.",
            "Bạn đã đọc chưa?"
        ]
    );
}

#[test]
fn test_fullwidth_cjk_punctuation_and_attached_quotes() {
    let mut chunker = SentenceChunker::new();
    let chunks = chunker.feed("「こんにちは！」 次の文です。");
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0], "「こんにちは！」");
    assert_eq!(chunks[1], "次の文です。");
}

#[test]
fn test_markdown_links_not_swallowed() {
    let mut parser = StreamParser::new();
    let events = parser.feed("[wave] Xem tại [Rust Lang](https://www.rust-lang.org) để bắt đầu. [happy]");
    assert_eq!(
        events,
        vec![
            ParsedStreamEvent::Gesture("wave".to_string()),
            ParsedStreamEvent::SpeakableText("Xem tại [Rust Lang](https://www.rust-lang.org) để bắt đầu.".to_string()),
            ParsedStreamEvent::Emotion("happy".to_string()),
        ]
    );
}

#[test]
fn test_multiple_consecutive_tags() {
    let mut parser = StreamParser::new();
    let events = parser.feed("[happy][wave][nod] Chào bạn!");
    assert_eq!(
        events,
        vec![
            ParsedStreamEvent::Emotion("happy".to_string()),
            ParsedStreamEvent::Gesture("wave".to_string()),
            ParsedStreamEvent::Gesture("nod".to_string()),
            ParsedStreamEvent::SpeakableText("Chào bạn!".to_string()),
        ]
    );
}

#[test]
fn test_url_with_colon_not_split_early() {
    let mut chunker = SentenceChunker::new();
    let chunks = chunker.feed("Hãy truy cập http://localhost:8080/api ngay hôm nay. Hẹn gặp lại!");
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0], "Hãy truy cập http://localhost:8080/api ngay hôm nay.");
    assert_eq!(chunks[1], "Hẹn gặp lại!");
}
