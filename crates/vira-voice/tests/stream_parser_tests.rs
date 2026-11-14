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

#[test]
fn test_stream_parser_sync_gestures_explain_shy_encourage() {
    let mut parser = StreamParser::new();
    let events = parser.feed("[explain] Để tôi giải thích cho bạn. [shy] Hì hì.");
    assert!(events.contains(&ParsedStreamEvent::Gesture("explain".to_string())));
    assert!(events.contains(&ParsedStreamEvent::Gesture("shy".to_string())));
    assert!(events.contains(&ParsedStreamEvent::SpeakableText("Để tôi giải thích cho bạn.".to_string())));
}

#[test]
fn test_stream_parser_dual_channel_thought_scratchpad() {
    let mut parser = StreamParser::new();
    let events = parser.feed("<thought>Người dùng đang hỏi về ngữ pháp tiếng Nhật N3.</thought>Dưới đây là cấu trúc câu.");
    assert_eq!(events.len(), 2);
    assert_eq!(
        events[0],
        ParsedStreamEvent::Thought("Người dùng đang hỏi về ngữ pháp tiếng Nhật N3.".to_string())
    );
    assert_eq!(
        events[1],
        ParsedStreamEvent::SpeakableText("Dưới đây là cấu trúc câu.".to_string())
    );

    // Also verify <think>...</think> variant
    let mut parser2 = StreamParser::new();
    let events2 = parser2.feed("<think>Analyzing user question</think>Chào bạn! [happy]");
    assert_eq!(
        events2[0],
        ParsedStreamEvent::Thought("Analyzing user question".to_string())
    );
    assert!(events2.contains(&ParsedStreamEvent::Emotion("happy".to_string())));
    assert!(events2.contains(&ParsedStreamEvent::SpeakableText("Chào bạn!".to_string())));

    // Verify thoughts NEVER enter speakable text
    for ev in events.iter().chain(events2.iter()) {
        if let ParsedStreamEvent::SpeakableText(ref text) = ev {
            assert!(!text.contains("Người dùng đang hỏi"));
            assert!(!text.contains("Analyzing user"));
        }
    }
}

#[test]
fn test_stream_parser_streaming_thought_across_chunks() {
    let mut parser = StreamParser::new();
    let e1 = parser.feed("<think>Bước 1: suy luận. ");
    assert!(e1.is_empty(), "Unclosed thought must not emit premature events");

    let e2 = parser.feed("Bước 2: tổng hợp.</think>Xin chào!");
    assert_eq!(e2.len(), 2);
    assert_eq!(
        e2[0],
        ParsedStreamEvent::Thought("Bước 1: suy luận. Bước 2: tổng hợp.".to_string())
    );
    assert_eq!(e2[1], ParsedStreamEvent::SpeakableText("Xin chào!".to_string()));
}

#[test]
fn test_stream_parser_tag_fragmentation_at_1_byte_boundaries() {
    let mut parser = StreamParser::new();

    // Opening tag split across chunks: "<", "th", "ought>"
    let e1 = parser.feed("Xin chào bạn. <");
    assert_eq!(e1.len(), 1);
    assert_eq!(e1[0], ParsedStreamEvent::SpeakableText("Xin chào bạn.".to_string()));

    let e2 = parser.feed("th");
    assert!(e2.is_empty(), "Incomplete '<th' must not leak to speech");

    let e3 = parser.feed("ought>Đây là suy nghĩ bí mật</");
    assert!(e3.is_empty(), "Inside thought must not leak to speech");

    let e4 = parser.feed("thought>Chúc một ngày tốt lành!");
    assert_eq!(e4.len(), 2);
    assert_eq!(
        e4[0],
        ParsedStreamEvent::Thought("Đây là suy nghĩ bí mật".to_string())
    );
    assert_eq!(
        e4[1],
        ParsedStreamEvent::SpeakableText("Chúc một ngày tốt lành!".to_string())
    );
}

#[test]
fn test_stream_parser_think_tag_fragmentation_at_1_byte_boundaries() {
    let mut parser = StreamParser::new();

    // Opening tag split across chunks: "<", "thi", "nk>"
    let e1 = parser.feed("Hôm nay trời đẹp. <");
    assert_eq!(e1.len(), 1);
    assert_eq!(e1[0], ParsedStreamEvent::SpeakableText("Hôm nay trời đẹp.".to_string()));

    let e2 = parser.feed("thi");
    assert!(e2.is_empty(), "Incomplete '<thi' must not leak to speech");

    let e3 = parser.feed("nk>Suy nghĩ cẩn thận</");
    assert!(e3.is_empty(), "Inside think must not leak to speech");

    let e4 = parser.feed("think>Cùng bắt đầu nhé!");
    assert_eq!(e4.len(), 2);
    assert_eq!(
        e4[0],
        ParsedStreamEvent::Thought("Suy nghĩ cẩn thận".to_string())
    );
    assert_eq!(
        e4[1],
        ParsedStreamEvent::SpeakableText("Cùng bắt đầu nhé!".to_string())
    );
}

#[test]
fn test_stream_parser_asterisk_actions_and_emotions() {
    let mut parser = StreamParser::new();
    let events = parser.feed("*cười tươi, vẫy tay chào* Xin chào bạn! *gật đầu* Chúc một ngày tốt lành.");
    assert!(events.contains(&ParsedStreamEvent::Emotion("happy".to_string())));
    assert!(events.contains(&ParsedStreamEvent::Gesture("wave".to_string())));
    assert!(events.contains(&ParsedStreamEvent::SpeakableText("Xin chào bạn!".to_string())));
    assert!(events.contains(&ParsedStreamEvent::Gesture("nod".to_string())));
    assert!(events.contains(&ParsedStreamEvent::SpeakableText("Chúc một ngày tốt lành.".to_string())));

    // Verify speakable text contains zero asterisks and zero action tags
    for ev in &events {
        if let ParsedStreamEvent::SpeakableText(ref text) = ev {
            assert!(!text.contains('*'));
            assert!(!text.contains("cười tươi"));
            assert!(!text.contains("vẫy tay"));
            assert!(!text.contains("gật đầu"));
        }
    }
}

#[test]
fn test_stream_parser_parentheses_stage_directions() {
    let mut parser = StreamParser::new();
    let events = parser.feed("(nghiêng đầu) Bạn có khỏe không? (cười nhẹ) Tôi rất vui khi được hỗ trợ bạn.");
    assert!(events.contains(&ParsedStreamEvent::Gesture("tilt".to_string())));
    assert!(events.contains(&ParsedStreamEvent::SpeakableText("Bạn có khỏe không?".to_string())));
    assert!(events.contains(&ParsedStreamEvent::Emotion("happy".to_string())));
    assert!(events.contains(&ParsedStreamEvent::SpeakableText("Tôi rất vui khi được hỗ trợ bạn.".to_string())));

    for ev in &events {
        if let ParsedStreamEvent::SpeakableText(ref text) = ev {
            assert!(!text.contains("nghiêng đầu"));
            assert!(!text.contains("cười nhẹ"));
        }
    }
}

#[test]
fn test_stream_parser_split_asterisk_and_paren_chunk_fragmentation() {
    let mut parser = StreamParser::new();

    // Split across chunks: "*vẫy", " tay*", " Xin ", "chào!"
    let c1 = parser.feed("*vẫy");
    assert!(c1.is_empty(), "Incomplete '*vẫy' must not be flushed prematurely");

    let c2 = parser.feed(" tay* Xin ");
    assert_eq!(c2, vec![ParsedStreamEvent::Gesture("wave".to_string())]);

    let c3 = parser.feed("chào bạn! (nghiêng");
    assert_eq!(c3, vec![ParsedStreamEvent::SpeakableText("Xin chào bạn!".to_string())]);

    let c4 = parser.feed(" đầu) Hôm nay thế nào?");
    assert_eq!(
        c4,
        vec![
            ParsedStreamEvent::Gesture("tilt".to_string()),
            ParsedStreamEvent::SpeakableText("Hôm nay thế nào?".to_string()),
        ]
    );
}

#[test]
fn test_stream_parser_clean_dialogue_zero_stage_directions() {
    let test_stream = "<thought>Analyzing user question and mood</thought>\
                       *cười tươi, vẫy tay chào* Xin chào bạn! \
                       (nghiêng đầu) Rất vui được gặp lại bạn. \
                       *gật đầu* [happy] Hôm nay chúng ta cùng học tiếp nhé!";

    let mut parser = StreamParser::new();
    let mut all_events = Vec::new();

    // Stream in random-sized chunks (2, 3, 5 chars)
    let mut pos = 0;
    let chars: Vec<char> = test_stream.chars().collect();
    let step_sizes = [2, 3, 5, 1, 4, 2, 7];
    let mut step_idx = 0;

    while pos < chars.len() {
        let step = step_sizes[step_idx % step_sizes.len()];
        step_idx += 1;
        let end = (pos + step).min(chars.len());
        let chunk: String = chars[pos..end].iter().collect();
        all_events.extend(parser.feed(&chunk));
        pos = end;
    }
    all_events.extend(parser.finish());

    let thoughts: Vec<_> = all_events.iter().filter(|e| e.is_thought()).collect();
    let emotions: Vec<_> = all_events.iter().filter(|e| e.is_emotion()).collect();
    let gestures: Vec<_> = all_events.iter().filter(|e| e.is_gesture()).collect();
    let speakable: Vec<_> = all_events.iter().filter(|e| e.is_speakable()).collect();

    assert_eq!(thoughts.len(), 1);
    assert_eq!(thoughts[0].as_thought().unwrap(), "Analyzing user question and mood");
    assert!(emotions.iter().any(|e| e.as_emotion() == Some("happy")));
    assert!(gestures.iter().any(|g| g.as_gesture() == Some("wave")));
    assert!(gestures.iter().any(|g| g.as_gesture() == Some("tilt")));
    assert!(gestures.iter().any(|g| g.as_gesture() == Some("nod")));

    // Strict invariant: Speakable text contains ONLY spoken dialogue with zero thoughts, zero asterisks, and zero stage directions
    for s in &speakable {
        let text = s.as_speakable().unwrap();
        assert!(!text.contains('*'), "Speakable text must not contain asterisks: {text}");
        assert!(!text.contains('<'), "Speakable text must not contain '<': {text}");
        assert!(!text.contains('>'), "Speakable text must not contain '>': {text}");
        assert!(!text.contains('['), "Speakable text must not contain '[': {text}");
        assert!(!text.contains(']'), "Speakable text must not contain ']': {text}");
        assert!(!text.contains("cười tươi"), "Speakable text must not contain action 'cười tươi': {text}");
        assert!(!text.contains("vẫy tay"), "Speakable text must not contain action 'vẫy tay': {text}");
        assert!(!text.contains("nghiêng đầu"), "Speakable text must not contain action 'nghiêng đầu': {text}");
        assert!(!text.contains("gật đầu"), "Speakable text must not contain action 'gật đầu': {text}");
    }
}

#[test]
fn test_stream_parser_generative_anatomical_anchors() {
    let mut parser = StreamParser::new();
    let stream = "*chống cằm* Để tôi xem. *che miệng cười* Thật thú vị! *đặt tay lên tim* Rất cảm ơn bạn. *day thái dương* Tôi cần suy nghĩ một chút.";
    let events = parser.feed(stream);
    let flushed = parser.finish();
    let all = [events, flushed].concat();

    let gestures: Vec<_> = all.iter().filter_map(|e| e.as_gesture()).collect();
    assert!(gestures.contains(&"chin_touch"));
    assert!(gestures.contains(&"mouth_cover"));
    assert!(gestures.contains(&"hand_on_heart"));
    assert!(gestures.contains(&"temple_rub"));
}

