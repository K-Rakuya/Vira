use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use vira_domain::message::{Message, Role};
use vira_learn::fsrs_scheduler::{
    CardRating, CardRecord, CardReviewState, FsrsScheduler,
};
use vira_learn::service::{LearnError, SpacedRepetitionPort, SpacedRepetitionService};
use vira_learn::tutor_policies::{
    CefrLevel, RecastMode, TutorPolicyManager,
};
use vira_ports::store::{Flashcard, SessionSummary, StoreError, StorePort, UserProfile};

// -----------------------------------------------------------------------------
// Mock Store for Hexagonal Integration
// -----------------------------------------------------------------------------

#[derive(Default)]
struct MockStore {
    cards: Mutex<HashMap<String, Flashcard>>,
}

#[async_trait]
impl StorePort for MockStore {
    async fn save_message(&self, _session_id: &str, _message: &Message) -> Result<(), StoreError> {
        Ok(())
    }

    async fn get_history(
        &self,
        _session_id: &str,
        _limit: usize,
    ) -> Result<Vec<Message>, StoreError> {
        Ok(Vec::new())
    }

    async fn delete_session(&self, _session_id: &str) -> Result<(), StoreError> {
        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<SessionSummary>, StoreError> {
        Ok(Vec::new())
    }

    async fn update_session_title(
        &self,
        _session_id: &str,
        _title: &str,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    async fn get_user_profile(&self, _user_id: &str) -> Result<Option<UserProfile>, StoreError> {
        Ok(None)
    }

    async fn save_user_profile(&self, _profile: &UserProfile) -> Result<(), StoreError> {
        Ok(())
    }

    async fn save_flashcard(&self, card: &Flashcard) -> Result<(), StoreError> {
        let mut cards = self.cards.lock().unwrap();
        cards.insert(card.id.clone(), card.clone());
        Ok(())
    }

    async fn get_flashcard(&self, id: &str) -> Result<Option<Flashcard>, StoreError> {
        let cards = self.cards.lock().unwrap();
        Ok(cards.get(id).cloned())
    }

    async fn delete_flashcard(&self, id: &str) -> Result<(), StoreError> {
        let mut cards = self.cards.lock().unwrap();
        cards.remove(id);
        Ok(())
    }

    async fn count_flashcards(&self) -> Result<usize, StoreError> {
        let cards = self.cards.lock().unwrap();
        Ok(cards.len())
    }

    async fn get_due_flashcards(
        &self,
        current_time: u64,
        limit: usize,
    ) -> Result<Vec<Flashcard>, StoreError> {
        let cards = self.cards.lock().unwrap();
        let mut due: Vec<Flashcard> = cards
            .values()
            .filter(|c| c.is_due(current_time))
            .cloned()
            .collect();
        due.truncate(limit);
        Ok(due)
    }

    async fn list_flashcards(&self, limit: usize) -> Result<Vec<Flashcard>, StoreError> {
        let cards = self.cards.lock().unwrap();
        let mut list: Vec<Flashcard> = cards.values().cloned().collect();
        list.truncate(limit);
        Ok(list)
    }

    async fn get_setting(&self, _key: &str) -> Result<Option<String>, StoreError> {
        Ok(None)
    }

    async fn set_setting(&self, _key: &str, _value: &str) -> Result<(), StoreError> {
        Ok(())
    }

    async fn delete_setting(&self, _key: &str) -> Result<(), StoreError> {
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Test 1: Card Scheduling Progression (from Brief)
// -----------------------------------------------------------------------------

#[test]
fn test_card_scheduling_progression() {
    let scheduler = FsrsScheduler::new();
    let card = scheduler.new_card("serendipity", "sự tình cờ may mắn");
    let reviewed = scheduler.review(&card, CardRating::Good);
    assert!(reviewed.scheduled_days > 0);
}

// -----------------------------------------------------------------------------
// Test 2: Card Rating and Enum Utilities
// -----------------------------------------------------------------------------

#[test]
fn test_card_rating_conversions_and_display() {
    assert_eq!(CardRating::Again.as_u32(), 1);
    assert_eq!(CardRating::Hard.as_u32(), 2);
    assert_eq!(CardRating::Good.as_u32(), 3);
    assert_eq!(CardRating::Easy.as_u32(), 4);

    assert_eq!(CardRating::from_u32(1).unwrap(), CardRating::Again);
    assert_eq!(CardRating::from_u32(2).unwrap(), CardRating::Hard);
    assert_eq!(CardRating::from_u32(3).unwrap(), CardRating::Good);
    assert_eq!(CardRating::from_u32(4).unwrap(), CardRating::Easy);
    assert!(CardRating::from_u32(0).is_err());
    assert!(CardRating::from_u32(5).is_err());

    assert_eq!(CardRating::Again.to_string(), "again");
    assert_eq!("easy".parse::<CardRating>().unwrap(), CardRating::Easy);
    assert_eq!("4".parse::<CardRating>().unwrap(), CardRating::Easy);
    assert!("invalid".parse::<CardRating>().is_err());
}

// -----------------------------------------------------------------------------
// Test 3: CardRecord and Domain Flashcard Bridge
// -----------------------------------------------------------------------------

#[test]
fn test_card_record_flashcard_conversion_and_serde() {
    let record = CardRecord::new("ubiquitous", "phổ biến khắp nơi")
        .with_id("card_101")
        .with_cefr("C1");

    assert_eq!(record.id, "card_101");
    assert_eq!(record.front, "ubiquitous");
    assert_eq!(record.back, "phổ biến khắp nơi");
    assert_eq!(record.cefr_level.as_deref(), Some("C1"));
    assert_eq!(record.state, CardReviewState::New);
    assert_eq!(record.reps, 0);
    assert_eq!(record.lapses, 0);

    // Convert to domain Flashcard
    let flashcard = record.to_flashcard();
    assert_eq!(flashcard.id, "card_101");
    assert_eq!(flashcard.front, "ubiquitous");
    assert_eq!(flashcard.back, "phổ biến khắp nơi");
    assert_eq!(flashcard.cefr_level.as_deref(), Some("C1"));
    assert_eq!(flashcard.state, 0);

    // Convert back from Flashcard
    let restored = CardRecord::from_flashcard(&flashcard);
    assert_eq!(restored.id, record.id);
    assert_eq!(restored.front, record.front);
    assert_eq!(restored.back, record.back);
    assert_eq!(restored.cefr_level, record.cefr_level);
    assert_eq!(restored.state, record.state);

    // Serde roundtrip
    let json = serde_json::to_string(&record).expect("serialize record");
    let deserialized: CardRecord = serde_json::from_str(&json).expect("deserialize record");
    assert_eq!(record, deserialized);
}

// -----------------------------------------------------------------------------
// Test 4: Multi-Step FSRS Spaced Repetition Progression
// -----------------------------------------------------------------------------

#[test]
fn test_fsrs_multi_step_progression_and_stability_growth() {
    let scheduler = FsrsScheduler::with_retention(0.9);
    let now_ms = 1_700_000_000_000u64; // arbitrary base epoch

    let card = scheduler.new_card("ephemeral", "chóng tàn, phù du");
    assert_eq!(card.reps, 0);
    assert_eq!(card.state, CardReviewState::New);

    // First review: Good
    let rev1 = scheduler.review_at(&card, CardRating::Good, now_ms);
    assert_eq!(rev1.reps, 1);
    assert_eq!(rev1.lapses, 0);
    assert_eq!(rev1.state, CardReviewState::Review);
    assert!(rev1.stability > 0.0);
    assert!(rev1.difficulty > 0.0);
    assert!(rev1.scheduled_days >= 1);
    assert!(rev1.due_timestamp > now_ms);

    // Second review after scheduled interval has passed
    let review_2_time = rev1.due_timestamp;
    let rev2 = scheduler.review_at(&rev1, CardRating::Good, review_2_time);
    assert_eq!(rev2.reps, 2);
    assert!(rev2.stability >= rev1.stability);
    assert!(rev2.scheduled_days >= rev1.scheduled_days);

    // Third review: Easy
    let review_3_time = rev2.due_timestamp;
    let rev3 = scheduler.review_at(&rev2, CardRating::Easy, review_3_time);
    assert_eq!(rev3.reps, 3);
    assert!(rev3.stability > rev2.stability);
    assert!(rev3.scheduled_days >= rev2.scheduled_days);

    // Fourth review: Lapse with Again
    let review_4_time = rev3.due_timestamp;
    let rev4 = scheduler.review_at(&rev3, CardRating::Again, review_4_time);
    assert_eq!(rev4.reps, 3); // reps not incremented on fail
    assert_eq!(rev4.lapses, 1);
    assert_eq!(rev4.state, CardReviewState::Relearning);
    assert!(rev4.stability < rev3.stability);
}

// -----------------------------------------------------------------------------
// Test 5: Preview Intervals Across All Ratings
// -----------------------------------------------------------------------------

#[test]
fn test_preview_intervals_order() {
    let scheduler = FsrsScheduler::new();
    let card = scheduler.new_card("meticulous", "tỉ mỉ, cẩn thận");

    let previews = scheduler.preview_intervals(&card);

    // Again interval is the shortest, Easy is the longest
    assert!(previews.again <= previews.hard);
    assert!(previews.hard <= previews.good);
    assert!(previews.good <= previews.easy);

    assert_eq!(previews.for_rating(CardRating::Again), previews.again);
    assert_eq!(previews.for_rating(CardRating::Hard), previews.hard);
    assert_eq!(previews.for_rating(CardRating::Good), previews.good);
    assert_eq!(previews.for_rating(CardRating::Easy), previews.easy);
}

// -----------------------------------------------------------------------------
// Test 6: CEFR Level Progression and Text Assessment
// -----------------------------------------------------------------------------

#[test]
fn test_cefr_level_properties_and_parsing() {
    assert_eq!("A1".parse::<CefrLevel>().unwrap(), CefrLevel::A1);
    assert_eq!("b2".parse::<CefrLevel>().unwrap(), CefrLevel::B2);
    assert_eq!("C2".parse::<CefrLevel>().unwrap(), CefrLevel::C2);
    assert!("Z9".parse::<CefrLevel>().is_err());

    assert_eq!(CefrLevel::A1.next_level(), Some(CefrLevel::A2));
    assert_eq!(CefrLevel::C1.next_level(), Some(CefrLevel::C2));
    assert_eq!(CefrLevel::C2.next_level(), None);

    assert_eq!(CefrLevel::C2.prev_level(), Some(CefrLevel::C1));
    assert_eq!(CefrLevel::A1.prev_level(), None);

    let (min_w, max_w) = CefrLevel::B1.target_sentence_length();
    assert!(min_w < max_w);
}

#[test]
fn test_cefr_text_heuristic_assessment() {
    // Beginner A1 sample
    let a1_text = "I like apples. Cat is cute.";
    assert_eq!(CefrLevel::assess_text(a1_text), CefrLevel::A1);

    // Intermediate B1 sample with discourse marker and compound structure
    let b1_text = "In my opinion, learning English is very enjoyable because we can make international friends.";
    let b1_level = CefrLevel::assess_text(b1_text);
    assert!(b1_level >= CefrLevel::A2);

    // Advanced C1/C2 sample with complex subordinate clauses and academic vocabulary
    let c1_text = "Notwithstanding the ubiquitous prevalence of digital automation, the fundamental dichotomy between human cognitive intuition and algorithmic precision inevitably persists. Furthermore, empirical investigations subsequently substantiate this hypothesis.";
    let c1_level = CefrLevel::assess_text(c1_text);
    assert!(c1_level >= CefrLevel::B2);

    // Empty text fallback
    assert_eq!(CefrLevel::assess_text(""), CefrLevel::A1);
    assert_eq!(CefrLevel::assess_text("   "), CefrLevel::A1);
}

// -----------------------------------------------------------------------------
// Test 7: Grammar Error Detection and Pedagogical Recasts
// -----------------------------------------------------------------------------

#[test]
fn test_grammar_error_detection_and_recasts() {
    // Vietnamese L1 transfer: "very like"
    let issues = TutorPolicyManager::detect_grammar_issues("I very like reading books in the library.");
    assert!(!issues.is_empty());
    assert!(issues.iter().any(|i| i.category.contains("L1 Transfer") && i.faulty_phrase == "very like"));

    // Preposition error: "depend of"
    let prep_issues = TutorPolicyManager::detect_grammar_issues("Our holiday plans depend of the weather forecast.");
    assert!(prep_issues.iter().any(|i| i.suggested_correction == "depend on"));

    // Clean text has no detected issues
    let clean_issues = TutorPolicyManager::detect_grammar_issues("I really enjoy going for a walk in the evening.");
    assert!(clean_issues.is_empty());

    // Recast directive generation
    let manager = TutorPolicyManager::new(CefrLevel::B1).with_recast_mode(RecastMode::GentleRecast);
    let directive = manager.build_recast_instruction(&issues);
    assert!(directive.is_some());
    let prompt_str = directive.unwrap();
    assert!(prompt_str.contains("[PEDAGOGICAL RECAST DIRECTIVE]"));
    assert!(prompt_str.contains("really like"));
}

// -----------------------------------------------------------------------------
// Test 8: Pedagogical Message Enrichment with Domain Message
// -----------------------------------------------------------------------------

#[test]
fn test_tutor_policy_message_enrichment() {
    let manager = TutorPolicyManager::new(CefrLevel::B2);

    // User message with mistake
    let history = vec![
        Message::system("Base conversation assistant instructions."),
        Message::user("Yesterday I see my old teacher and we discuss about science."),
    ];

    let enriched = manager.enrich_messages(&history);
    assert_eq!(enriched.len(), 2);
    assert_eq!(enriched[0].role, Role::System);

    // System prompt should contain pedagogical guidance and recast directives
    let sys_content = &enriched[0].raw_content;
    assert!(sys_content.contains("PEDAGOGICAL POLICY: CEFR LEVEL B2"));
    assert!(sys_content.contains("RECAST DIRECTIVE"));
    assert!(sys_content.contains("yesterday I"));
}

// -----------------------------------------------------------------------------
// Test 9: Learner Evaluation Metrics
// -----------------------------------------------------------------------------

#[test]
fn test_learner_evaluation_summary() {
    let manager = TutorPolicyManager::new(CefrLevel::B1);
    let messages = vec![
        Message::system("System instructions"),
        Message::user("Hello! I want to practice English today."),
        Message::assistant("Hello! What topic would you like to explore?"),
        Message::user("In my opinion, travelling abroad is very exciting although it is expensive."),
    ];

    let eval = manager.evaluate_learner_performance(&messages);
    assert_eq!(eval.total_user_messages, 2);
    assert!(eval.total_words > 10);
    assert!(eval.lexical_diversity > 0.0);
    assert!(eval.recommendation.len() > 10);
}

// -----------------------------------------------------------------------------
// Test 10: Asynchronous SpacedRepetitionService with StorePort Integration
// -----------------------------------------------------------------------------

#[tokio::test]
async fn test_async_spaced_repetition_service_flow() {
    let mock_store = Arc::new(MockStore::default());
    let service = SpacedRepetitionService::new(mock_store.clone());

    // Create a new card
    let card = service
        .create_card("ubiquitous", "phổ biến khắp nơi", Some("C1".to_string()))
        .await
        .expect("create card");

    assert_eq!(card.front, "ubiquitous");
    assert_eq!(card.cefr_level.as_deref(), Some("C1"));

    // Verify stored
    let count = service.count_cards().await.expect("count");
    assert_eq!(count, 1);

    let retrieved = service
        .get_card(&card.id)
        .await
        .expect("get")
        .expect("card exists");
    assert_eq!(retrieved.id, card.id);

    // Preview intervals
    let previews = service
        .preview_intervals(&card.id)
        .await
        .expect("preview");
    assert!(previews.good >= 1);

    // Review card with Good
    let reviewed = service
        .review_card(&card.id, CardRating::Good)
        .await
        .expect("review");
    assert_eq!(reviewed.reps, 1);
    assert_eq!(reviewed.state, CardReviewState::Review);
    assert!(reviewed.scheduled_days >= 1);

    // Stored card should now reflect updated reps and state
    let updated_stored = service
        .get_card(&card.id)
        .await
        .expect("get updated")
        .expect("card exists");
    assert_eq!(updated_stored.reps, 1);
    assert_eq!(updated_stored.state, CardReviewState::Review);

    // Non-existent card returns error
    let err = service
        .review_card("unknown_id", CardRating::Good)
        .await
        .unwrap_err();
    assert!(matches!(err, LearnError::CardNotFound(_)));

    // Delete card
    service.delete_card(&card.id).await.expect("delete");
    assert_eq!(service.count_cards().await.expect("count"), 0);
}

// -----------------------------------------------------------------------------
// Test 11: Multi-threaded Concurrency Stress Test
// -----------------------------------------------------------------------------

#[tokio::test]
async fn test_fsrs_concurrent_reviews_stress() {
    let mock_store = Arc::new(MockStore::default());
    let service = Arc::new(SpacedRepetitionService::new(mock_store));

    // Spawn 20 concurrent card creations and reviews
    let mut tasks = Vec::new();
    for i in 0..20 {
        let s = service.clone();
        tasks.push(tokio::spawn(async move {
            let front = format!("word_{i}");
            let back = format!("meaning_{i}");
            let card = s.create_card(&front, &back, None).await.unwrap();
            let reviewed = s.review_card(&card.id, CardRating::Good).await.unwrap();
            assert_eq!(reviewed.reps, 1);
        }));
    }

    for task in tasks {
        task.await.unwrap();
    }

    assert_eq!(service.count_cards().await.unwrap(), 20);
}

// -----------------------------------------------------------------------------
// Test 12: Reviewer Skepticism Tests
// -----------------------------------------------------------------------------

#[test]
fn test_grammar_detector_false_positives_and_negatives() {
    // 1. "bravery like" should NOT be flagged as "very like"
    let issues = TutorPolicyManager::detect_grammar_issues("Such bravery like that is admirable.");
    assert!(
        !issues.iter().any(|i| i.faulty_phrase == "very like"),
        "False positive: 'bravery like' was flagged as 'very like'!"
    );

    // 2. Correct question "Does he go to school?" should NOT be flagged as subject-verb error!
    let issues = TutorPolicyManager::detect_grammar_issues("Does he go to school every day?");
    assert!(
        !issues.iter().any(|i| i.category.contains("Subject-Verb Agreement")),
        "False positive: 'Does he go' was flagged as an error!"
    );

    // 3. "Yesterday, I see..." with comma should be detected
    let issues = TutorPolicyManager::detect_grammar_issues("Yesterday, I see my friend at the park.");
    assert!(
        issues.iter().any(|i| i.category.contains("Past Tense")),
        "False negative: 'Yesterday, I see' with comma was not detected!"
    );

    // 4. "It depends of the weather." should be detected
    let issues = TutorPolicyManager::detect_grammar_issues("It depends of the weather tomorrow.");
    assert!(
        issues.iter().any(|i| i.category.contains("Preposition")),
        "False negative: 'depends of' was not detected!"
    );

    // 5. "He is good in swimming." should be detected
    let issues = TutorPolicyManager::detect_grammar_issues("He is good in swimming and chess.");
    assert!(
        issues.iter().any(|i| i.suggested_correction == "good at"),
        "False negative: 'good in swimming' was not detected!"
    );
}

#[test]
fn test_system_prompt_enrichment_does_not_duplicate_or_accumulate_stale_recast() {
    let manager = TutorPolicyManager::new(CefrLevel::B2);

    // Turn 1: user with error
    let history_turn_1 = vec![
        Message::system("Base system instructions."),
        Message::user("I very like learning languages."),
    ];
    let enriched_1 = manager.enrich_messages(&history_turn_1);
    assert_eq!(enriched_1.len(), 2);
    let sys_1 = &enriched_1[0].raw_content;
    assert!(sys_1.contains("[PEDAGOGICAL RECAST DIRECTIVE]"));

    // Turn 2: assistant replied, user sends a clean message without error
    let mut history_turn_2 = enriched_1;
    history_turn_2.push(Message::assistant("I really like that too! What languages do you know?"));
    history_turn_2.push(Message::user("I speak Vietnamese and I am learning English."));

    let enriched_2 = manager.enrich_messages(&history_turn_2);
    let sys_2 = &enriched_2[0].raw_content;

    // Must NOT contain duplicate pedagogical policy blocks
    let count_pedagogical_blocks = sys_2.matches("[PEDAGOGICAL POLICY: CEFR LEVEL").count();
    assert_eq!(
        count_pedagogical_blocks, 1,
        "System prompt accumulated duplicate pedagogical policy blocks: {}",
        sys_2
    );

    // Turn 2 had NO error, so stale recast directive from Turn 1 MUST be cleared!
    assert!(
        !sys_2.contains("[PEDAGOGICAL RECAST DIRECTIVE]"),
        "Stale recast directive from turn 1 was not removed on turn 2: {}",
        sys_2
    );
}

#[test]
fn test_card_record_from_flashcard_preserves_scheduled_days() {
    let scheduler = FsrsScheduler::new();
    let card = scheduler.new_card("serendipity", "sự tình cờ may mắn");
    let now = 1_700_000_000_000u64;
    let reviewed = scheduler.review_at(&card, CardRating::Good, now);
    assert!(reviewed.scheduled_days >= 1);

    // Convert to Flashcard (as saved in DB)
    let flashcard = reviewed.to_flashcard();

    // Reconstruct CardRecord (as fetched from DB)
    let restored = CardRecord::from_flashcard(&flashcard);
    assert_eq!(
        restored.scheduled_days, reviewed.scheduled_days,
        "from_flashcard failed to recover scheduled_days!"
    );
}

#[test]
fn test_rating_and_state_serde_flexibility() {
    // Rating deserialization from lowercase string, capitalized string, and integer
    let r1: CardRating = serde_json::from_str("\"again\"").unwrap();
    let r2: CardRating = serde_json::from_str("\"Good\"").unwrap();
    let r3: CardRating = serde_json::from_str("3").unwrap();
    let r4: CardRating = serde_json::from_str("4").unwrap();

    assert_eq!(r1, CardRating::Again);
    assert_eq!(r2, CardRating::Good);
    assert_eq!(r3, CardRating::Good);
    assert_eq!(r4, CardRating::Easy);

    // Rating serialization outputs lowercase canonical representation
    let json_rating = serde_json::to_string(&CardRating::Hard).unwrap();
    assert_eq!(json_rating, "\"hard\"");

    // ReviewState deserialization from string and integer
    let s1: CardReviewState = serde_json::from_str("\"learning\"").unwrap();
    let s2: CardReviewState = serde_json::from_str("2").unwrap();
    assert_eq!(s1, CardReviewState::Learning);
    assert_eq!(s2, CardReviewState::Review);
    assert_eq!(s2.to_string(), "review");
}

#[test]
fn test_fsrs_non_monotonic_and_nan_resilience() {
    let scheduler = FsrsScheduler::new();
    let mut card = scheduler.new_card("paradox", "nghịch lý");

    // Corrupt memory state parameters (NaN, negative, infinite)
    card.stability = f32::NAN;
    card.difficulty = f32::NAN;
    card.last_review = Some(2_000_000_000);

    // Clock jumps backwards into the past (now < last_review)
    let past_time = 1_000_000_000;
    let reviewed = scheduler.review_at(&card, CardRating::Good, past_time);

    // Must gracefully sanitize without panicking and return valid finite values
    assert!(reviewed.stability.is_finite());
    assert!(reviewed.difficulty.is_finite());
    assert!(reviewed.stability > 0.0);
    assert!(reviewed.difficulty >= 1.0 && reviewed.difficulty <= 10.0);
    assert_eq!(reviewed.last_review, Some(past_time));
}

#[tokio::test]
async fn test_service_explicit_timestamp_methods() {
    let mock_store = Arc::new(MockStore::default());
    let service = SpacedRepetitionService::new(mock_store);

    let card = service
        .create_card("cognition", "nhận thức", Some("B2".to_string()))
        .await
        .unwrap();

    let t0 = 1_700_000_000_000u64;
    let rev = service
        .review_card_at(&card.id, CardRating::Good, t0)
        .await
        .unwrap();
    assert_eq!(rev.last_review, Some(t0));

    // At t0, card is not due because it is scheduled in the future
    let due_at_t0 = service.get_due_cards_at(t0, 10).await.unwrap();
    assert!(due_at_t0.is_empty());

    // In the future after due_timestamp, it is due
    let due_at_future = service
        .get_due_cards_at(rev.due_timestamp + 1000, 10)
        .await
        .unwrap();
    assert_eq!(due_at_future.len(), 1);
    assert_eq!(due_at_future[0].id, card.id);
}

#[test]
fn test_cefr_short_text_false_b1_suppression() {
    // A 3-word utterance with "because" should not immediately jump to B1
    let short_text = "I cry because sad.";
    let level = CefrLevel::assess_text(short_text);
    assert!(level <= CefrLevel::A2);
}
