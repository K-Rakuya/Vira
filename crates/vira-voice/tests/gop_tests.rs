use vira_voice::gop::{
    phoneme_to_viseme, phoneme_to_vrm_preset, AccuracyLevel, GopConfig, GopEvaluator,
    PhonemeResult, PhonemeScore, PronunciationAssessment,
};
use vira_voice::phoneme_aligner::{
    AlignmentError, CtcAligner, DtwAligner, G2pConverter, MatchStatus, PhonemeLanguage,
};

#[test]
fn test_phoneme_scoring_thresholds() {
    let evaluator = GopEvaluator::default();
    let result = evaluator.evaluate_score(-0.15); // log posterior
    assert!(result >= 80.0, "Score for -0.15 log-posterior should be >= 80.0, got {result}");

    // Also verify PhonemeResult and PhonemeScore work identically
    let ph_res: PhonemeResult = evaluator.evaluate_phoneme("AA", &[-0.15], 0, 100);
    assert_eq!(ph_res.accuracy, AccuracyLevel::Correct);
    assert!(ph_res.score >= 80.0);

    let ph_score: PhonemeScore = ph_res;
    assert_eq!(ph_score.phoneme, "AA");
}

#[test]
fn test_accuracy_level_boundaries() {
    let evaluator = GopEvaluator::default();

    // Correct threshold (default >= 80.0)
    let s_correct = evaluator.evaluate_score(-0.1); // ~90.5
    assert!(s_correct >= 80.0);
    assert_eq!(
        AccuracyLevel::from_score(s_correct, 80.0, 50.0),
        AccuracyLevel::Correct
    );

    // MinorFlaw threshold (default 50.0 <= score < 80.0)
    let s_flaw = evaluator.evaluate_score(-0.4); // ~67.0
    assert!((50.0..80.0).contains(&s_flaw));
    assert_eq!(
        AccuracyLevel::from_score(s_flaw, 80.0, 50.0),
        AccuracyLevel::MinorFlaw
    );

    // Mispronounced threshold (default < 50.0)
    let s_mis = evaluator.evaluate_score(-0.8); // ~44.9
    assert!(s_mis < 50.0);
    assert_eq!(
        AccuracyLevel::from_score(s_mis, 80.0, 50.0),
        AccuracyLevel::Mispronounced
    );
}

#[test]
fn test_extreme_and_invalid_log_posteriors() {
    let evaluator = GopEvaluator::default();

    // Perfect probability (log prob = 0.0)
    assert_eq!(evaluator.evaluate_score(0.0), 100.0);

    // Positive log prob clamped
    assert_eq!(evaluator.evaluate_score(1.5), 100.0);

    // Negative infinity / extreme negative
    assert_eq!(evaluator.evaluate_score(-25.0), 0.0);
    assert_eq!(evaluator.evaluate_score(f32::NEG_INFINITY), 0.0);

    // NaN safety
    assert_eq!(evaluator.evaluate_score(f32::NAN), 0.0);
}

#[test]
fn test_custom_gop_config() {
    let config = GopConfig {
        correct_threshold: 85.0,
        minor_flaw_threshold: 60.0,
        temperature: 1.5,
        min_score: 5.0,
        max_score: 95.0,
        frame_duration_ms: 10,
    };
    let evaluator = GopEvaluator::new(config);

    let score = evaluator.evaluate_score(0.0);
    assert_eq!(score, 95.0);

    let low_score = evaluator.evaluate_score(-30.0);
    assert_eq!(low_score, 5.0);
}

#[test]
fn test_llr_score_calculation() {
    let evaluator = GopEvaluator::default();

    let pos_llr = evaluator.evaluate_llr_score(2.5);
    assert!(pos_llr > 90.0);

    let zero_llr = evaluator.evaluate_llr_score(0.0);
    assert!((zero_llr - 50.0).abs() < 1e-3);

    let neg_llr = evaluator.evaluate_llr_score(-2.5);
    assert!(neg_llr < 10.0);

    assert_eq!(evaluator.evaluate_llr_score(f32::NAN), 0.0);
    assert_eq!(evaluator.evaluate_llr_score(15.0), 100.0);
    assert_eq!(evaluator.evaluate_llr_score(-15.0), 0.0);
}

#[test]
fn test_viseme_hints_for_avatar_vrm() {
    // Vowels
    let v_aa = phoneme_to_viseme("AA");
    assert!(v_aa.aa > 0.8, "AA should activate aa blendshape");
    assert_eq!(phoneme_to_vrm_preset("AA"), "aa");

    let v_iy = phoneme_to_viseme("IY");
    assert!(v_iy.ih > 0.8, "IY should activate ih blendshape");
    assert_eq!(phoneme_to_vrm_preset("IY"), "ih");

    let v_uw = phoneme_to_viseme("UW");
    assert!(v_uw.ou > 0.8, "UW should activate ou blendshape");
    assert_eq!(phoneme_to_vrm_preset("UW"), "ou");

    let v_eh = phoneme_to_viseme("EH");
    assert!(v_eh.ee > 0.8, "EH should activate ee blendshape");
    assert_eq!(phoneme_to_vrm_preset("EH"), "ee");

    let v_ow = phoneme_to_viseme("OW");
    assert!(v_ow.oh > 0.8, "OW should activate oh blendshape");
    assert_eq!(phoneme_to_vrm_preset("OW"), "oh");

    // Bilabial consonants (lips closed)
    let v_b = phoneme_to_viseme("B");
    assert!(v_b.is_silent());

    let v_m = phoneme_to_viseme("M");
    assert!(v_m.is_silent());

    // Dental fricative (TH)
    let v_th = phoneme_to_viseme("TH");
    assert!(v_th.ih > 0.2 && v_th.aa > 0.1);

    // Vietnamese vowel with tone
    let v_vn = phoneme_to_viseme("AA_T2");
    assert!(v_vn.aa > 0.8, "AA_T2 should activate aa blendshape but got {}", v_vn.aa);

    let v_ie = phoneme_to_viseme("IE_T6");
    assert!(v_ie.ee > 0.8, "IE_T6 should activate ee blendshape but got {}", v_ie.ee);

    let v_ow = phoneme_to_viseme("OW_HORN_T4");
    assert!(v_ow.oh > 0.8, "OW_HORN_T4 should activate oh blendshape but got {}", v_ow.oh);
}

#[test]
fn test_phoneme_score_and_assessment_aggregation() {
    let evaluator = GopEvaluator::default();

    // 3 phonemes: 2 correct, 1 mispronounced
    let p1 = evaluator.evaluate_phoneme("TH", &[-0.05, -0.08], 0, 80);
    let p2 = evaluator.evaluate_phoneme("IH", &[-0.10, -0.12], 80, 160);
    let p3 = evaluator.evaluate_phoneme("NG", &[-1.80, -2.10], 160, 240);

    assert_eq!(p1.accuracy, AccuracyLevel::Correct);
    assert_eq!(p2.accuracy, AccuracyLevel::Correct);
    assert_eq!(p3.accuracy, AccuracyLevel::Mispronounced);

    let scores = vec![p1, p2, p3];
    let assessment = evaluator.evaluate_assessment(scores);

    assert_eq!(assessment.total_phonemes, 3);
    assert_eq!(assessment.mispronounced_count, 1);
    assert_eq!(assessment.mispronounced_phonemes().len(), 1);
    assert_eq!(assessment.mispronounced_phonemes()[0].phoneme, "NG");

    // Zero-duration deleted phoneme must NOT be erased from overall score
    let zero_dur_p1 = PhonemeScore::new("AA", 100.0, 0, 100);
    let zero_dur_p2 = PhonemeScore::new("B", 0.0, 100, 100); // 0ms duration, score 0
    let zero_assess = evaluator.evaluate_assessment(vec![zero_dur_p1, zero_dur_p2]);
    assert!(
        zero_assess.overall_score < 100.0,
        "Zero duration phoneme should reduce overall score, got {}",
        zero_assess.overall_score
    );

    // Viseme hints for avatar feedback
    let mis_hints = assessment.viseme_hints_for_mispronounced();
    assert_eq!(mis_hints.len(), 1);

    let all_hints = assessment.all_viseme_hints();
    assert_eq!(all_hints.len(), 3);

    // Serialization roundtrip
    let serialized = serde_json::to_string(&assessment).expect("Failed to serialize");
    let deserialized: PronunciationAssessment =
        serde_json::from_str(&serialized).expect("Failed to deserialize");
    assert_eq!(assessment.overall_score, deserialized.overall_score);
    assert_eq!(assessment.mispronounced_count, deserialized.mispronounced_count);
}

#[test]
fn test_g2p_english_lexicon_and_rules() {
    let g2p = G2pConverter::new();

    // Lexicon lookup
    let hello = g2p.text_to_phonemes("hello", PhonemeLanguage::English);
    assert_eq!(hello, vec!["HH", "AH", "L", "OW"]);

    let hello_quotes = g2p.text_to_phonemes("“hello”", PhonemeLanguage::English);
    assert_eq!(hello_quotes, vec!["HH", "AH", "L", "OW"], "Quotes broke dictionary lookup: got {:?}", hello_quotes);

    let they = g2p.text_to_phonemes("they", PhonemeLanguage::English);
    assert_eq!(they, vec!["DH", "EY"]);

    let dont = g2p.text_to_phonemes("don't", PhonemeLanguage::English);
    assert_eq!(dont, vec!["D", "OW", "N", "T"]);

    let world = g2p.text_to_phonemes("world", PhonemeLanguage::English);
    assert_eq!(world, vec!["W", "ER", "L", "D"]);

    let vira = g2p.text_to_phonemes("Vira", PhonemeLanguage::English);
    assert_eq!(vira, vec!["V", "IH", "R", "AA"]);

    // Rule-based fallback for arbitrary words
    let rules_word = g2p.text_to_phonemes("station", PhonemeLanguage::English);
    assert!(rules_word.contains(&"SH".to_string()));
    assert!(rules_word.contains(&"AH".to_string()));
    assert!(rules_word.contains(&"N".to_string()));

    // Phonics rules: silent k in knee, silent w in write
    let knee = g2p.text_to_phonemes("knee", PhonemeLanguage::English);
    assert_eq!(knee, vec!["N", "IY"]);

    let write_word = g2p.text_to_phonemes("write", PhonemeLanguage::English);
    assert_eq!(write_word[0], "R");

    // Custom word addition
    let mut custom_g2p = G2pConverter::new();
    custom_g2p.add_custom_word("supercalifragilistic", vec!["S".to_string(), "UW".to_string()]);
    let custom_res = custom_g2p.text_to_phonemes("supercalifragilistic", PhonemeLanguage::English);
    assert_eq!(custom_res, vec!["S", "UW"]);
}

#[test]
fn test_g2p_vietnamese_syllables_and_diacritics() {
    let g2p = G2pConverter::new();

    // Vietnamese phrase
    let phones = g2p.text_to_phonemes("Xin chào Việt Nam đồng điệu", PhonemeLanguage::Vietnamese);
    assert!(!phones.is_empty());
    assert!(phones.contains(&"S".to_string()));
    assert!(phones.contains(&"IY".to_string()));
    assert!(phones.contains(&"N".to_string()));
    assert!(phones.contains(&"CH".to_string()));
    assert!(phones.contains(&"AA_T2".to_string())); // Chào has huyền tone (T2)
    assert!(phones.contains(&"V".to_string())); // Việt has v -> V
    assert!(phones.contains(&"IE_T6".to_string())); // Việt has iê + nặng (T6)
    assert!(phones.contains(&"D".to_string())); // đồng / điệu has đ -> D

    // Diphthong ưỡng bug check
    let duong = g2p.text_to_phonemes("dưỡng", PhonemeLanguage::Vietnamese);
    assert_eq!(duong, vec!["Z", "UOW_T4", "NG"]);

    // gi with tone on i check
    let gi = g2p.text_to_phonemes("gì", PhonemeLanguage::Vietnamese);
    assert_eq!(gi, vec!["Z", "IY_T2"]);

    let gieng = g2p.text_to_phonemes("giếng", PhonemeLanguage::Vietnamese);
    assert_eq!(gieng, vec!["Z", "IE_T5", "NG"]);

    let gio = g2p.text_to_phonemes("gió", PhonemeLanguage::Vietnamese);
    assert_eq!(gio, vec!["Z", "AW_OPEN_T5"]);

    // Language auto-detection
    assert_eq!(
        G2pConverter::detect_language("Hôm nay thời tiết rất đẹp."),
        PhonemeLanguage::Vietnamese
    );
    assert_eq!(
        G2pConverter::detect_language("The quick brown fox jumps over the lazy dog."),
        PhonemeLanguage::English
    );
}

#[test]
fn test_ctc_monotonic_frame_alignment_and_errors() {
    let aligner = CtcAligner::new(20);
    let target = vec!["HH".to_string(), "OW".to_string()];
    let vocab = vec!["HH".to_string(), "OW".to_string(), "AA".to_string()];

    // Error case: target empty
    let err_empty = aligner.align_log_posteriors(&[], &[vec![-0.1]], &vocab);
    assert_eq!(err_empty, Err(AlignmentError::EmptyTargetSequence));

    // Error case: frames empty
    let err_frames = aligner.align_log_posteriors(&target, &[], &vocab);
    assert_eq!(err_frames, Err(AlignmentError::EmptyAcousticFrames));

    // Error case: fewer frames than phonemes (1 frame for 2 phonemes)
    let err_insuf = aligner.align_log_posteriors(&target, &[vec![-0.1, -0.2, -0.3]], &vocab);
    assert!(matches!(err_insuf, Err(AlignmentError::InsufficientFrames { .. })));

    // Error case: phoneme not in vocab
    let missing_target = vec!["UNKNOWN_PHONEME".to_string()];
    let err_vocab = aligner.align_log_posteriors(
        &missing_target,
        &[vec![-0.1, -0.2, -0.3]],
        &vocab,
    );
    assert!(matches!(err_vocab, Err(AlignmentError::PhonemeNotFoundInVocab(_))));

    // Dimension mismatch error
    let err_dim = aligner.align_log_posteriors(&target, &[vec![-0.1], vec![-0.1]], &vocab);
    assert!(matches!(err_dim, Err(AlignmentError::FrameDimensionMismatch { .. })));

    // Valid 5-frame alignment
    let frames = vec![
        vec![-0.1, -4.0, -5.0], // frame 0: HH
        vec![-0.2, -3.5, -5.0], // frame 1: HH
        vec![-0.3, -3.0, -4.0], // frame 2: HH
        vec![-4.0, -0.1, -5.0], // frame 3: OW
        vec![-5.0, -0.05, -5.0], // frame 4: OW
    ];
    let alignment = aligner.align_log_posteriors(&target, &frames, &vocab).unwrap();
    assert_eq!(alignment.len(), 2);

    let hh = &alignment[0];
    assert_eq!(hh.phoneme, "HH");
    assert_eq!(hh.start_frame, 0);
    assert_eq!(hh.end_frame, 2);
    assert_eq!(hh.start_ms, 0);
    assert_eq!(hh.end_ms, 60);
    assert!(hh.mean_log_posterior >= -0.25);

    let ow = &alignment[1];
    assert_eq!(ow.phoneme, "OW");
    assert_eq!(ow.start_frame, 3);
    assert_eq!(ow.end_frame, 4);
    assert_eq!(ow.start_ms, 60);
    assert_eq!(ow.end_ms, 100);
    assert!(ow.mean_log_posterior >= -0.1);
}

#[test]
fn test_dtw_sequence_alignment() {
    let dtw = DtwAligner::new();

    // 1. Identical sequences -> 100% accuracy
    let target = vec!["HH".to_string(), "EH".to_string(), "L".to_string(), "OW".to_string()];
    let res_exact = dtw.align(&target, &target);
    assert_eq!(res_exact.accuracy_percent, 100.0);
    assert_eq!(res_exact.correct_count, 4);
    assert_eq!(res_exact.matches[0].status, MatchStatus::ExactMatch);
    assert_eq!(res_exact.substitutions_count, 0);
    assert_eq!(res_exact.insertions_count, 0);
    assert_eq!(res_exact.deletions_count, 0);
    assert_eq!(dtw.align_distance_rolling(&target, &target), res_exact.total_cost);

    // ARPAbet stress variation: AA vs AA1 is not penalized as substitution
    let dist_stress = DtwAligner::phoneme_distance("AA", "AA1");
    assert_eq!(dist_stress, 0.0);

    // Vietnamese tone difference (AA_T2 vs AA_T5) is a minor flaw (0.2), not a full substitution (1.0)
    let dist_tone = DtwAligner::phoneme_distance("AA_T2", "AA_T5");
    assert!((dist_tone - 0.2).abs() < 1e-4);

    // 2. Vowel substitution ("EH" replaced with "AE")
    let rec_sub = vec!["HH".to_string(), "AE".to_string(), "L".to_string(), "OW".to_string()];
    let res_sub = dtw.align(&target, &rec_sub);
    assert_eq!(res_sub.correct_count, 3);
    assert_eq!(res_sub.substitutions_count, 1);
    assert!(res_sub.accuracy_percent > 85.0);
    assert!((dtw.align_distance_rolling(&target, &rec_sub) - res_sub.total_cost).abs() < 1e-4);

    // 3. Deletion ("L" dropped)
    let rec_del = vec!["HH".to_string(), "EH".to_string(), "OW".to_string()];
    let res_del = dtw.align(&target, &rec_del);
    assert_eq!(res_del.deletions_count, 1);
    assert!((dtw.align_distance_rolling(&target, &rec_del) - res_del.total_cost).abs() < 1e-4);

    // 4. Insertion (extra sound "S" inserted at end)
    let rec_ins = vec![
        "HH".to_string(),
        "EH".to_string(),
        "L".to_string(),
        "OW".to_string(),
        "S".to_string(),
    ];
    let res_ins = dtw.align(&target, &rec_ins);
    assert_eq!(res_ins.insertions_count, 1);
    assert!((dtw.align_distance_rolling(&target, &rec_ins) - res_ins.total_cost).abs() < 1e-4);

    // 5. Empty inputs
    let res_empty = dtw.align(&[], &[]);
    assert_eq!(res_empty.accuracy_percent, 100.0);
    assert_eq!(dtw.align_distance_rolling(&[], &[]), 0.0);
}

#[test]
fn test_end_to_end_pronunciation_pipeline() {
    let g2p = G2pConverter::new();
    let ctc = CtcAligner::new(20);
    let evaluator = GopEvaluator::default();

    // Step 1: Text to phonemes
    let expected_text = "hello";
    let phonemes = g2p.text_to_phonemes(expected_text, PhonemeLanguage::English);
    assert_eq!(phonemes, vec!["HH", "AH", "L", "OW"]);

    // Step 2: Simulated acoustic model frame probabilities
    let vocab = phonemes.clone();
    let frames = vec![
        vec![-0.1, -5.0, -5.0, -5.0], // frame 0: HH
        vec![-0.1, -5.0, -5.0, -5.0], // frame 1: HH
        vec![-5.0, -0.2, -5.0, -5.0], // frame 2: AH
        vec![-5.0, -0.15, -5.0, -5.0], // frame 3: AH
        vec![-5.0, -5.0, -0.1, -5.0], // frame 4: L
        vec![-5.0, -5.0, -0.2, -5.0], // frame 5: L
        vec![-5.0, -5.0, -5.0, -0.05], // frame 6: OW
        vec![-5.0, -5.0, -5.0, -0.1], // frame 7: OW
    ];

    // Step 3: Monotonic alignment
    let aligned_segments = ctc.align_log_posteriors(&phonemes, &frames, &vocab).unwrap();
    assert_eq!(aligned_segments.len(), 4);

    // Step 4: GOP evaluation
    let mut phoneme_scores = Vec::new();
    for seg in aligned_segments {
        let score = evaluator.evaluate_phoneme(
            &seg.phoneme,
            &seg.frame_log_posteriors,
            seg.start_ms,
            seg.end_ms,
        );
        phoneme_scores.push(score);
    }

    let assessment = evaluator.evaluate_assessment(phoneme_scores);
    assert!(assessment.overall_score >= 80.0);
    assert_eq!(assessment.accuracy, AccuracyLevel::Correct);
    assert_eq!(assessment.mispronounced_count, 0);

    // Check VRM viseme hints are generated
    let visemes = assessment.all_viseme_hints();
    assert_eq!(visemes.len(), 4);
    assert_eq!(assessment.phoneme_scores[3].recommended_viseme.as_deref(), Some("oh"));
}
