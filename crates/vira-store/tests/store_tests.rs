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
