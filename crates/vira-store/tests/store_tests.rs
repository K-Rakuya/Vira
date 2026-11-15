use std::collections::HashMap;
use std::sync::Arc;
use vira_domain::message::{Message, Role};
use vira_learn::fsrs_scheduler::CardRating;
use vira_learn::service::{SpacedRepetitionPort, SpacedRepetitionService};
use vira_ports::store::{Flashcard, StoreError, StorePort, UserProfile};
use vira_store::sqlite::SqliteStore;

#[tokio::test]
async fn test_save_and_retrieve_chat_history() {
    let store = SqliteStore::open_in_memory().expect("open sqlite");
    store.run_migrations().expect("migrations");

    let msg = Message::new(Role::User, "Hello Vira")
        .with_speakable_content("Hello Vira");
    store.save_message("session_1", &msg).await.expect("save message");

    let history = store.get_history("session_1", 10).await.expect("get history");
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].raw_content, "Hello Vira");
    assert_eq!(history[0].speakable_content, "Hello Vira");
    assert_eq!(history[0].role, Role::User);
}

#[tokio::test]
async fn test_history_ordering_and_limit() {
    let store = SqliteStore::open_in_memory().expect("open sqlite");

    // Insert 5 messages with explicit increasing timestamps
    for i in 1..=5 {
        let role = if i % 2 == 1 { Role::User } else { Role::Assistant };
        let msg = Message::new(role, format!("Msg {i}")).with_timestamp(1000 + i * 10);
        store.save_message("session_ord", &msg).await.expect("save msg");
    }

    // Limit 0 returns empty
    let empty = store.get_history("session_ord", 0).await.expect("limit 0");
    assert!(empty.is_empty());

    // Limit 2 returns the latest 2 messages in chronological order (Msg 4, Msg 5)
    let latest_two = store.get_history("session_ord", 2).await.expect("limit 2");
    assert_eq!(latest_two.len(), 2);
    assert_eq!(latest_two[0].raw_content, "Msg 4");
    assert_eq!(latest_two[1].raw_content, "Msg 5");

    // Limit 10 returns all 5 messages in chronological order
    let all = store.get_history("session_ord", 10).await.expect("limit 10");
    assert_eq!(all.len(), 5);
    for (idx, m) in all.iter().enumerate() {
        assert_eq!(m.raw_content, format!("Msg {}", idx + 1));
    }
}

#[tokio::test]
async fn test_sessions_lifecycle_and_summary() {
    let store = SqliteStore::open_in_memory().expect("open sqlite");

    // Saving messages should automatically create session records
    let msg1 = Message::new(Role::User, "How do you say thank you in Vietnamese?");
    let msg2 = Message::new(Role::Assistant, "Cảm ơn bạn!");
    store.save_message("sess_vn", &msg1).await.expect("save 1");
    store.save_message("sess_vn", &msg2).await.expect("save 2");

    let sessions = store.list_sessions().await.expect("list sessions");
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, "sess_vn");
    assert_eq!(sessions[0].message_count, 2);
    assert!(sessions[0].title.is_none());

    // Update title
    store
        .update_session_title("sess_vn", "Vietnamese Polite Greetings")
        .await
        .expect("update title");

    let updated_sessions = store.list_sessions().await.expect("list sessions");
    assert_eq!(
        updated_sessions[0].title.as_deref(),
        Some("Vietnamese Polite Greetings")
    );

    // Updating title for a non-existent session returns NotFound error
    let err = store
        .update_session_title("non_existent_session", "New Title")
        .await
        .expect_err("should fail");
    assert!(matches!(err, StoreError::NotFound(_)));

    // Deleting session cascades to delete its messages
    store.delete_session("sess_vn").await.expect("delete session");
    let after_delete_sessions = store.list_sessions().await.expect("list sessions");
    assert!(after_delete_sessions.is_empty());

    let remaining_messages = store.get_history("sess_vn", 10).await.expect("get history");
    assert!(remaining_messages.is_empty());
}

#[tokio::test]
async fn test_flashcard_crud_and_fsrs_queries() {
    let store = SqliteStore::open_in_memory().expect("open sqlite");

    // Initial count is 0
    assert_eq!(store.count_flashcards().await.expect("count"), 0);

    // Create flashcard
    let card1 = Flashcard::new("card_1", "serendipity", "sự may mắn tình cờ")
        .with_cefr_level("C1")
        .with_fsrs(2.5, 4.0, 1, 1, 0)
        .with_due(5000);

    let card2 = Flashcard::new("card_2", "hello", "xin chào")
        .with_cefr_level("A1")
        .with_fsrs(10.0, 2.0, 2, 5, 0)
        .with_due(10000);

    store.save_flashcard(&card1).await.expect("save card 1");
    store.save_flashcard(&card2).await.expect("save card 2");

    assert_eq!(store.count_flashcards().await.expect("count"), 2);

    // Get by ID
    let fetched = store.get_flashcard("card_1").await.expect("get card").expect("exists");
    assert_eq!(fetched.id, "card_1");
    assert_eq!(fetched.front, "serendipity");
    assert_eq!(fetched.back, "sự may mắn tình cờ");
    assert_eq!(fetched.cefr_level.as_deref(), Some("C1"));
    assert_eq!(fetched.state, 1);
    assert!((fetched.stability - 2.5).abs() < f32::EPSILON);
    assert!((fetched.difficulty - 4.0).abs() < f32::EPSILON);
    assert_eq!(fetched.due, 5000);

    // Non-existent ID returns None
    assert!(store.get_flashcard("card_unknown").await.expect("query").is_none());

    // Due flashcards query
    // At timestamp 4000: neither is due
    let due_4000 = store.get_due_flashcards(4000, 10).await.expect("due 4000");
    assert_eq!(due_4000.len(), 0);

    // At timestamp 6000: only card_1 is due
    let due_6000 = store.get_due_flashcards(6000, 10).await.expect("due 6000");
    assert_eq!(due_6000.len(), 1);
    assert_eq!(due_6000[0].id, "card_1");

    // At timestamp 15000: both are due, ordered by due ASC
    let due_15000 = store.get_due_flashcards(15000, 10).await.expect("due 15000");
    assert_eq!(due_15000.len(), 2);
    assert_eq!(due_15000[0].id, "card_1");
    assert_eq!(due_15000[1].id, "card_2");

    // Due query with limit
    let due_limited = store.get_due_flashcards(15000, 1).await.expect("due limited");
    assert_eq!(due_limited.len(), 1);
    assert_eq!(due_limited[0].id, "card_1");

    // List all flashcards
    let listed = store.list_flashcards(10).await.expect("list cards");
    assert_eq!(listed.len(), 2);

    // Update existing card
    let mut updated_card1 = card1;
    updated_card1.reps = 2;
    updated_card1.stability = 5.0;
    updated_card1.due = 20000;
    store.save_flashcard(&updated_card1).await.expect("update card 1");

    let re_fetched = store.get_flashcard("card_1").await.expect("re-fetch").expect("found");
    assert_eq!(re_fetched.reps, 2);
    assert!((re_fetched.stability - 5.0).abs() < f32::EPSILON);
    assert_eq!(re_fetched.due, 20000);

    // Delete flashcard
    store.delete_flashcard("card_1").await.expect("delete card 1");
    assert_eq!(store.count_flashcards().await.expect("count"), 1);
    assert!(store.get_flashcard("card_1").await.expect("get").is_none());

    // Deleting nonexistent card returns NotFound
    let del_err = store.delete_flashcard("card_1").await.expect_err("already deleted");
    assert!(matches!(del_err, StoreError::NotFound(_)));
}

#[tokio::test]
async fn test_user_profile_crud() {
    let store = SqliteStore::open_in_memory().expect("open sqlite");

    // Non-existent profile returns None
    assert!(store.get_user_profile("user_primary").await.expect("get").is_none());

    let mut prefs = HashMap::new();
    prefs.insert("theme".to_string(), "cosmic-glass".to_string());
    prefs.insert("tutor_rigor".to_string(), "strict".to_string());

    let profile = UserProfile {
        user_id: "user_primary".to_string(),
        display_name: "Lan Anh".to_string(),
        native_language: "vi".to_string(),
        target_language: "en".to_string(),
        cefr_level: "B2".to_string(),
        preferences: prefs,
    };

    store.save_user_profile(&profile).await.expect("save profile");

    let loaded = store
        .get_user_profile("user_primary")
        .await
        .expect("get profile")
        .expect("found profile");

    assert_eq!(loaded.user_id, "user_primary");
    assert_eq!(loaded.display_name, "Lan Anh");
    assert_eq!(loaded.native_language, "vi");
    assert_eq!(loaded.target_language, "en");
    assert_eq!(loaded.cefr_level, "B2");
    assert_eq!(
        loaded.preferences.get("theme").map(String::as_str),
        Some("cosmic-glass")
    );
    assert_eq!(
        loaded.preferences.get("tutor_rigor").map(String::as_str),
        Some("strict")
    );

    // Update profile
    let mut updated_profile = loaded;
    updated_profile.cefr_level = "C1".to_string();
    updated_profile
        .preferences
        .insert("audio_speed".to_string(), "1.2".to_string());
    store.save_user_profile(&updated_profile).await.expect("update profile");

    let re_loaded = store
        .get_user_profile("user_primary")
        .await
        .expect("get")
        .expect("found");
    assert_eq!(re_loaded.cefr_level, "C1");
    assert_eq!(
        re_loaded.preferences.get("audio_speed").map(String::as_str),
        Some("1.2")
    );
}

#[tokio::test]
async fn test_settings_kv_store() {
    let store = SqliteStore::open_in_memory().expect("open sqlite");

    // Non-existent key returns None
    assert!(store.get_setting("audio_device").await.expect("get").is_none());

    // Insert setting
    store.set_setting("audio_device", "BlackHole 2ch").await.expect("set");
    assert_eq!(
        store.get_setting("audio_device").await.expect("get").as_deref(),
        Some("BlackHole 2ch")
    );

    // Update setting
    store.set_setting("audio_device", "Scarlett 2i2 USB").await.expect("update");
    assert_eq!(
        store.get_setting("audio_device").await.expect("get").as_deref(),
        Some("Scarlett 2i2 USB")
    );

    // Delete setting
    store.delete_setting("audio_device").await.expect("delete");
    assert!(store.get_setting("audio_device").await.expect("get").is_none());
}

#[tokio::test]
async fn test_disk_file_persistence_and_wal_mode() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let db_path = temp_dir.path().join("vira_test.db");

    // Open file-backed SQLite database
    let store = SqliteStore::open(&db_path).expect("open file db");

    // Verify WAL journal mode is active
    let journal = store.journal_mode().await.expect("journal mode");
    assert_eq!(journal, "wal");

    // Write persistent data
    let msg = Message::new(Role::User, "Persistent message across restarts");
    store.save_message("persistent_sess", &msg).await.expect("save message");
    store
        .set_setting("app_version", "0.1.0-alpha")
        .await
        .expect("set setting");

    let card = Flashcard::new("persist_c1", "resilient", "kiên cường, bất khuất")
        .with_cefr_level("B2")
        .with_due(9999);
    store.save_flashcard(&card).await.expect("save card");

    // Drop first store instance to simulate application shutdown
    drop(store);

    // Reopen the exact same file in a new instance
    let reopened = SqliteStore::open(&db_path).expect("reopen file db");

    // Verify data persisted
    let history = reopened
        .get_history("persistent_sess", 10)
        .await
        .expect("get history");
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].raw_content, "Persistent message across restarts");

    let ver = reopened.get_setting("app_version").await.expect("get setting");
    assert_eq!(ver.as_deref(), Some("0.1.0-alpha"));

    let loaded_card = reopened
        .get_flashcard("persist_c1")
        .await
        .expect("get card")
        .expect("found card");
    assert_eq!(loaded_card.front, "resilient");
    assert_eq!(loaded_card.back, "kiên cường, bất khuất");
}

#[tokio::test]
async fn test_concurrent_async_tasks() {
    let store = SqliteStore::open_in_memory().expect("open sqlite");
    let mut handles = Vec::new();

    // Spawn 15 concurrent tasks inserting messages into separate sessions
    for task_idx in 0..15 {
        let store_clone = store.clone();
        let handle = tokio::spawn(async move {
            let session_id = format!("concurrent_sess_{task_idx}");
            for msg_idx in 0..5 {
                let msg = Message::new(
                    Role::User,
                    format!("Task {task_idx} Message {msg_idx}"),
                );
                store_clone
                    .save_message(&session_id, &msg)
                    .await
                    .expect("concurrent save");
            }

            let history = store_clone
                .get_history(&session_id, 10)
                .await
                .expect("concurrent get");
            assert_eq!(history.len(), 5);
        });
        handles.push(handle);
    }

    for h in handles {
        h.await.expect("task join");
    }

    let sessions = store.list_sessions().await.expect("list sessions");
    assert_eq!(sessions.len(), 15);
}

#[tokio::test]
async fn test_integration_with_spaced_repetition_service() {
    let sqlite_store = SqliteStore::open_in_memory().expect("open sqlite");
    let store_port: Arc<dyn StorePort> = Arc::new(sqlite_store.clone());

    // Connect to vira-learn SpacedRepetitionService
    let service = SpacedRepetitionService::new(store_port);

    // Create a new card through the service
    let created = service
        .create_card("ephemeral", "lasting for a very short time", Some("C2".to_string()))
        .await
        .expect("create card");

    assert_eq!(created.front, "ephemeral");
    assert_eq!(created.cefr_level.as_deref(), Some("C2"));

    // Review card with rating 'Good'
    let reviewed = service
        .review_card(&created.id, CardRating::Good)
        .await
        .expect("review card");

    assert!(reviewed.scheduled_days > 0);
    assert_eq!(reviewed.reps, 1);

    // Verify card state persisted directly in SqliteStore
    let direct_card = sqlite_store
        .get_flashcard(&created.id)
        .await
        .expect("get flashcard directly")
        .expect("card exists");

    assert_eq!(direct_card.front, "ephemeral");
    assert_eq!(direct_card.reps, 1);
    assert!(direct_card.stability > 0.0);
}

#[tokio::test]
async fn test_session_timestamps_no_regression() {
    let store = SqliteStore::open_in_memory().expect("open sqlite");

    // Save a message at ts=2000
    let msg1 = Message::new(Role::User, "Msg at 2000").with_timestamp(2000);
    store.save_message("sess_regress", &msg1).await.expect("save 1");

    let sessions = store.list_sessions().await.expect("list");
    assert_eq!(sessions[0].created_at, 2000);
    assert_eq!(sessions[0].updated_at, 2000);

    // Now save an older message at ts=1000
    let msg2 = Message::new(Role::User, "Msg at 1000").with_timestamp(1000);
    store.save_message("sess_regress", &msg2).await.expect("save 2");

    let sessions = store.list_sessions().await.expect("list");
    // created_at should adjust to earliest message (1000)
    // updated_at should NOT regress to 1000; it must remain 2000!
    assert_eq!(sessions[0].created_at, 1000);
    assert_eq!(sessions[0].updated_at, 2000);
}

#[tokio::test]
async fn test_flashcard_nan_resilience() {
    let store = SqliteStore::open_in_memory().expect("open sqlite");
    let mut card = Flashcard::new("nan_card", "test", "test");
    card.stability = f32::NAN;
    card.difficulty = f32::INFINITY;
    store.save_flashcard(&card).await.expect("save nan card");

    let loaded = store
        .get_flashcard("nan_card")
        .await
        .expect("get")
        .expect("exists");
    assert!(loaded.stability.is_finite());
    assert!(loaded.difficulty.is_finite());
}

#[tokio::test]
async fn test_flashcard_due_deterministic_tie_breaking() {
    let store = SqliteStore::open_in_memory().expect("open sqlite");

    // Insert 3 cards with the EXACT same due timestamp in non-alphabetical order
    let c_z = Flashcard::new("card_z", "zebra", "ngựa vằn").with_due(5000);
    let c_a = Flashcard::new("card_a", "apple", "quả táo").with_due(5000);
    let c_m = Flashcard::new("card_m", "mango", "quả xoài").with_due(5000);

    store.save_flashcard(&c_z).await.expect("save z");
    store.save_flashcard(&c_a).await.expect("save a");
    store.save_flashcard(&c_m).await.expect("save m");

    let due = store.get_due_flashcards(5000, 10).await.expect("due");
    assert_eq!(due.len(), 3);
    // Should be deterministically sorted by (due ASC, id ASC): card_a, card_m, card_z
    assert_eq!(due[0].id, "card_a");
    assert_eq!(due[1].id, "card_m");
    assert_eq!(due[2].id, "card_z");
}

#[tokio::test]
async fn test_run_migrations_under_concurrency() {
    let store = SqliteStore::open_in_memory().expect("open sqlite");

    // Spawn a task that holds connection for a short duration while executing queries
    let store_clone = store.clone();
    let handle = tokio::spawn(async move {
        for _ in 0..10 {
            let msg = Message::new(Role::User, "ping");
            store_clone.save_message("busy_sess", &msg).await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
    });

    // Run migrations from another task concurrently - should wait and succeed, NOT fail with try_lock
    for _ in 0..5 {
        tokio::time::sleep(std::time::Duration::from_millis(3)).await;
        store.run_migrations().expect("migrations should not fail under contention");
    }

    handle.await.unwrap();
}

#[tokio::test]
async fn test_user_profile_corrupted_json_returns_error() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let db_path = temp_dir.path().join("corrupt_profile.db");
    let store = SqliteStore::open(&db_path).expect("open");

    // Save a valid profile first
    let profile = UserProfile::new("u_corrupt", "Corrupt User", "vi", "en", "B2");
    store.save_user_profile(&profile).await.expect("save profile");

    // Directly corrupt the preferences_json field in SQLite
    {
        let conn = rusqlite::Connection::open(&db_path).expect("open raw");
        conn.execute(
            "UPDATE user_profiles SET preferences_json = '{invalid_json' WHERE user_id = 'u_corrupt'",
            [],
        )
        .expect("corrupt json");
    }

    // Now get_user_profile should return StoreError::Serialization, NOT silently wipe it!
    let res = store.get_user_profile("u_corrupt").await;
    assert!(
        matches!(res, Err(StoreError::Serialization(_))),
        "Expected Serialization error, got: {:?}",
        res
    );
}

#[tokio::test]
async fn test_list_sessions_deterministic_tie_breaking() {
    let store = SqliteStore::open_in_memory().expect("open sqlite");

    let m1 = Message::new(Role::User, "m1").with_timestamp(5000);
    store.save_message("sess_z", &m1).await.expect("save z");
    store.save_message("sess_a", &m1).await.expect("save a");
    store.save_message("sess_m", &m1).await.expect("save m");

    let sessions = store.list_sessions().await.expect("list");
    assert_eq!(sessions.len(), 3);
    // Ordered by (updated_at DESC, id DESC): sess_z, sess_m, sess_a
    assert_eq!(sessions[0].id, "sess_z");
    assert_eq!(sessions[1].id, "sess_m");
    assert_eq!(sessions[2].id, "sess_a");
}

#[test]
fn test_sync_save_and_retrieve_chat_history() {
    let store = SqliteStore::open_in_memory().expect("open sqlite");
    store.run_migrations().expect("migrations");

    let msg = Message::new(Role::User, "Hello Vira Sync");
    store
        .save_message_sync("session_sync_1", &msg)
        .expect("sync save");

    let history = store
        .get_history_sync("session_sync_1", 10)
        .expect("sync get history");
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].raw_content, "Hello Vira Sync");
}

#[tokio::test]
async fn test_nested_directory_creation_on_open() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let deeply_nested = temp_dir
        .path()
        .join("sub1")
        .join("sub2")
        .join("deep_store.db");

    let store = SqliteStore::open(&deeply_nested).expect("open deeply nested db");
    let card = Flashcard::new("c_nested", "depth", "độ sâu");
    store.save_flashcard(&card).await.expect("save flashcard");

    let loaded = store
        .get_flashcard("c_nested")
        .await
        .expect("query")
        .expect("found");
    assert_eq!(loaded.front, "depth");
}

#[tokio::test]
async fn test_rich_card_record_sqlite_persistence() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let db_path = temp_dir.path().join("rich_cards.db");
    let store = SqliteStore::open(&db_path).expect("open store");

    // 1. Create a rich flashcard with phonetic, context, POS, and tags
    let rich_card = Flashcard::new("c_rich_1", "serendipity", "sự tình cờ may mắn")
        .with_cefr_level("C1")
        .with_phonetic("/ˌser.ənˈdɪp.ə.ti/")
        .with_context(
            "Finding this cozy cafe was pure serendipity.",
            Some("Tìm thấy quán cà phê ấm cúng này đúng là một sự tình cờ may mắn.".to_string()),
        )
        .with_part_of_speech("noun")
        .with_tags(vec!["ielts".to_string(), "vocabulary".to_string(), "c1".to_string()]);

    store.save_flashcard(&rich_card).await.expect("save rich card");

    // 2. Fetch and assert all rich fields are accurately retrieved
    let loaded = store
        .get_flashcard("c_rich_1")
        .await
        .expect("get rich card")
        .expect("card exists");

    assert_eq!(loaded.id, "c_rich_1");
    assert_eq!(loaded.front, "serendipity");
    assert_eq!(loaded.back, "sự tình cờ may mắn");
    assert_eq!(loaded.cefr_level.as_deref(), Some("C1"));
    assert_eq!(loaded.phonetic.as_deref(), Some("/ˌser.ənˈdɪp.ə.ti/"));
    assert_eq!(
        loaded.context_sentence.as_deref(),
        Some("Finding this cozy cafe was pure serendipity.")
    );
    assert_eq!(
        loaded.context_translation.as_deref(),
        Some("Tìm thấy quán cà phê ấm cúng này đúng là một sự tình cờ may mắn.")
    );
    assert_eq!(loaded.part_of_speech.as_deref(), Some("noun"));
    assert_eq!(
        loaded.tags,
        vec!["ielts".to_string(), "vocabulary".to_string(), "c1".to_string()]
    );

    // 3. Update rich card and verify persistence
    let mut updated_card = loaded.clone();
    updated_card.tags.push("favorites".to_string());
    updated_card.part_of_speech = Some("noun/phrase".to_string());
    store.save_flashcard(&updated_card).await.expect("update rich card");

    let re_loaded = store
        .get_flashcard("c_rich_1")
        .await
        .expect("get updated card")
        .expect("card exists");
    assert_eq!(
        re_loaded.tags,
        vec![
            "ielts".to_string(),
            "vocabulary".to_string(),
            "c1".to_string(),
            "favorites".to_string()
        ]
    );
    assert_eq!(re_loaded.part_of_speech.as_deref(), Some("noun/phrase"));

    // 4. Test listing and due queries preserve rich fields
    let list = store.list_flashcards(10).await.expect("list cards");
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].phonetic.as_deref(), Some("/ˌser.ənˈdɪp.ə.ti/"));
    assert_eq!(list[0].tags.len(), 4);

    let due = store.get_due_flashcards(u64::MAX, 10).await.expect("due cards");
    assert_eq!(due.len(), 1);
    assert_eq!(due[0].phonetic.as_deref(), Some("/ˌser.ənˈdɪp.ə.ti/"));

    // 5. Backwards compatibility: legacy card without rich fields defaults cleanly
    let legacy_card = Flashcard::new("c_legacy", "apple", "quả táo");
    store.save_flashcard(&legacy_card).await.expect("save legacy card");

    let loaded_legacy = store
        .get_flashcard("c_legacy")
        .await
        .expect("get legacy card")
        .expect("exists");
    assert_eq!(loaded_legacy.phonetic, None);
    assert_eq!(loaded_legacy.context_sentence, None);
    assert_eq!(loaded_legacy.context_translation, None);
    assert_eq!(loaded_legacy.part_of_speech, None);
    assert!(loaded_legacy.tags.is_empty());
}
