use std::collections::HashMap;
use vira_rag::chunker::TextChunker;
use vira_rag::hybrid_rrf::{reciprocal_rank_fusion, reciprocal_rank_fusion_weighted};
use vira_rag::lancedb_vector::LanceDbVectorStore;
use vira_rag::tantivy_bm25::TantivyBm25Index;
use vira_rag::vector_store::{cosine_similarity, LocalVectorStore, VectorRecord};
use vira_rag::{HybridRagConfig, HybridRagEngine};

#[test]
fn test_text_chunking_with_overlap() {
    let text = "Câu một. Câu hai. Câu ba. Câu bốn. Câu năm.";
    let chunker = TextChunker::new(50, 10);
    let chunks = chunker.chunk(text);
    assert!(!chunks.is_empty());
    assert_eq!(chunks[0].document_id, "doc_default");
}

#[test]
fn test_rrf_scoring_order() {
    let bm25_ranks = vec!["docA".to_string(), "docB".to_string()];
    let vec_ranks = vec!["docB".to_string(), "docC".to_string()];
    let fused = reciprocal_rank_fusion(&bm25_ranks, &vec_ranks, 60);
    assert_eq!(fused[0].doc_id, "docB");
    assert!(fused[0].score > fused[1].score);
}

#[test]
fn test_rrf_weighted_custom_weights() {
    let bm25_ranks = vec!["keyword_match".to_string()];
    let vec_ranks = vec!["semantic_match".to_string()];

    // Vector search has 4x higher weight
    let fused = reciprocal_rank_fusion_weighted(&bm25_ranks, &vec_ranks, 60, 1.0, 4.0);
    assert_eq!(fused[0].doc_id, "semantic_match");
    assert_eq!(fused[1].doc_id, "keyword_match");
}

#[test]
fn test_chunking_preserves_sentences_and_vietnamese() {
    let text = "Tiếng Việt là ngôn ngữ có thanh điệu. Học phát âm chuẩn giúp bạn tự tin hơn khi giao tiếp! Hãy luyện tập mỗi ngày cùng Vira.";
    let chunker = TextChunker::new(60, 15);
    let chunks = chunker.chunk(text);
    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert!(!chunk.text.is_empty());
    }
}

#[test]
fn test_chunker_abbreviation_and_decimal_protection() {
    let text = "Theo TS. Trần Văn Nam, tỉ lệ đỗ là 98.5% tại TP. Hồ Chí Minh. Đây là con số ấn tượng!";
    let chunker = TextChunker::new(200, 20);
    let sentences = chunker.split_into_sentences(text);
    assert_eq!(sentences.len(), 2);
    assert!(sentences[0].2.contains("TS."));
    assert!(sentences[0].2.contains("98.5%"));
    assert!(sentences[0].2.contains("TP."));
}

#[test]
fn test_chunker_cats_and_multidot_abbreviations() {
    let chunker = TextChunker::new(200, 20);
    // "cats." ends with "ts", should NOT be treated as abbreviation
    let text1 = "I love cats. They are cute.";
    let s1 = chunker.split_into_sentences(text1);
    assert_eq!(s1.len(), 2, "Word 'cats' ending in 'ts' should not prevent sentence split");

    // "v.v." multi-dot abbreviation
    let text2 = "Học Toán, Lý, Hóa, v.v. tại trường phổ thông. Kỳ thi rất khó.";
    let s2 = chunker.split_into_sentences(text2);
    assert_eq!(s2.len(), 2, "v.v. abbreviation should not cause split after first v.");

    // "e.g." multi-dot abbreviation
    let text3 = "Common fruits, e.g. apples and oranges, are healthy. Eat them daily.";
    let s3 = chunker.split_into_sentences(text3);
    assert_eq!(s3.len(), 2, "e.g. abbreviation should not cause split after first e.");
}

#[test]
fn test_long_sentence_chunk_offsets() {
    let chunker = TextChunker::new(40, 10);
    let text = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen.";
    let chunks = chunker.chunk(text);
    assert!(chunks.len() >= 2);
    // Each chunk should have distinct and monotonically increasing offsets
    for i in 1..chunks.len() {
        assert!(
            chunks[i].start_char >= chunks[i - 1].start_char,
            "Chunk start_char must advance, got chunk[{}].start_char={}, chunk[{}].start_char={}",
            i - 1, chunks[i - 1].start_char, i, chunks[i].start_char
        );
        assert_ne!(
            chunks[i].start_char, chunks[i - 1].start_char,
            "Subsequent chunks must not share identical start_char"
        );
    }
}

#[test]
fn test_cjk_sentence_delimiters() {
    let chunker = TextChunker::new(200, 20);
    let text = "これは日本語です。それは本です！どれですか？";
    let sentences = chunker.split_into_sentences(text);
    assert_eq!(sentences.len(), 3, "CJK full stop and exclamation/question marks should split sentences");
}

#[test]
fn test_tantivy_query_parser_special_characters() {
    let index = TantivyBm25Index::create_in_ram().expect("ram index");
    index
        .add_chunk("c1", "d1", "C++ Programming", "C++ is a compiled systems language.", None)
        .expect("add");
    index.commit().expect("commit");

    // Should not panic or return QueryParser error on C++, ???, or boolean ops
    let res1 = index.search("C++", 5);
    assert!(res1.is_ok(), "Searching 'C++' must not fail with query parser error");

    let res2 = index.search("???", 5);
    assert!(res2.is_ok(), "Searching '???' must not fail with query parser error");

    let res3 = index.search("+ - && || !", 5);
    assert!(res3.is_ok(), "Searching operators must not fail with query parser error");
}

#[test]
fn test_cosine_similarity_edge_cases() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

    let c = vec![0.0, 1.0, 0.0];
    assert!((cosine_similarity(&a, &c)).abs() < 1e-6);

    let zero = vec![0.0, 0.0, 0.0];
    assert_eq!(cosine_similarity(&a, &zero), 0.0);

    let empty: Vec<f32> = vec![];
    assert_eq!(cosine_similarity(&empty, &a), 0.0);
}

#[test]
fn test_tantivy_bm25_standalone_index_and_search() {
    let index = TantivyBm25Index::create_in_ram().expect("tantivy in ram");
    index
        .add_chunk("c_voc_1", "doc_voc", "Vocabulary", "Ephemeral means lasting for a very short time.", None)
        .expect("add chunk");
    index.commit().expect("commit");

    let hits = index.search("lasting short time", 3).expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].chunk_id, "c_voc_1");
    assert_eq!(hits[0].doc_id, "doc_voc");
}

#[test]
fn test_vector_store_standalone_crud() {
    let mut store = LocalVectorStore::new();
    let rec1 = VectorRecord::new("c1", "d1", "Alpha", vec![1.0, 0.0]);
    let rec2 = VectorRecord::new("c2", "d2", "Beta", vec![0.0, 1.0]);

    store.insert(rec1);
    store.insert(rec2);
    assert_eq!(store.count(), 2);

    let hits = store.search(&[1.0, 0.0], 1);
    assert_eq!(hits[0].chunk_id, "c1");

    assert!(store.delete_by_chunk_id("c1"));
    assert_eq!(store.count(), 1);
}

#[test]
fn test_lancedb_vector_store_tables() {
    let db = LanceDbVectorStore::new_in_memory();
    let table = db.open_table("grammar_rules").expect("open table");
    let rec = VectorRecord::new("r1", "doc_g", "Rules of conditionals", vec![0.5, 0.5]);
    table.add(vec![rec]).expect("add");

    assert_eq!(table.count(), 1);
    let hits = table.search(&[0.5, 0.5], 2).expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].chunk_id, "r1");
}

#[tokio::test]
async fn test_bm25_and_vector_hybrid_retrieval() {
    let engine = HybridRagEngine::new_in_memory(HybridRagConfig::default())
        .expect("failed to initialize in-memory HybridRagEngine");

    let doc_text = "Thì hiện tại hoàn thành (Present Perfect) dùng để diễn đạt hành động đã xảy ra trong quá khứ nhưng kết quả còn lưu lại ở hiện tại. Cấu trúc: S + have/has + V3/ed.";
    let mut meta = HashMap::new();
    meta.insert("grammar_topic".to_string(), "present_perfect".to_string());

    let chunks = engine
        .ingest_document("doc_grammar_01", "Ngữ pháp tiếng Anh cơ bản", doc_text, Some(meta))
        .await
        .expect("ingest document");

    assert!(!chunks.is_empty());

    let results = engine
        .search("quá khứ hiện tại cấu trúc have has", 3)
        .await
        .expect("hybrid search");

    assert!(!results.is_empty());
    assert_eq!(results[0].document_id, "doc_grammar_01");
    assert_eq!(results[0].citation, "[Ngữ pháp tiếng Anh cơ bản] (doc_grammar_01)");
    assert!(results[0].score > 0.0);
}

#[tokio::test]
async fn test_persistent_hybrid_rag_lifecycle_and_delete() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("rag_db");

    // Phase 1: Ingest into persistent store
    {
        let engine = HybridRagEngine::new_persistent(&db_path, HybridRagConfig::default())
            .expect("init persistent engine");

        engine
            .ingest_document(
                "doc_ielts_01",
                "IELTS Speaking Tips",
                "Fluency and coherence are critical criteria. Do not pause for too long when searching for words.",
                None,
            )
            .await
            .expect("ingest");

        assert_eq!(engine.count_chunks(), 1);
    }

    // Phase 2: Reload from disk in a fresh instance
    {
        let engine = HybridRagEngine::new_persistent(&db_path, HybridRagConfig::default())
            .expect("reload persistent engine");

        let hits = engine.search("fluency coherence criteria", 3).await.expect("search");
        assert!(!hits.is_empty());
        assert_eq!(hits[0].document_id, "doc_ielts_01");
        assert_eq!(hits[0].title, "IELTS Speaking Tips");
        assert_eq!(hits[0].citation, "[IELTS Speaking Tips] (doc_ielts_01)");
        assert!(hits[0].text.contains("Fluency"));

        // Delete document
        engine.delete_document("doc_ielts_01").await.expect("delete");
        assert_eq!(engine.count_chunks(), 0);

        let hits_after_del = engine.search("fluency", 3).await.expect("search after delete");
        assert!(hits_after_del.is_empty());
    }
}

#[tokio::test]
async fn test_rag_port_contract() {
    use vira_ports::RagPort;

    let engine = HybridRagEngine::new_in_memory(HybridRagConfig::default()).expect("engine");
    let port: &dyn RagPort = &engine;

    let count = port
        .ingest(
            "doc_port_1",
            "Vocabulary Builder",
            "Serendipity means finding something good without looking for it.",
            None,
        )
        .await
        .expect("ingest via port");

    assert!(count >= 1);
    assert_eq!(port.count(), count);

    let hits = port.search("serendipity looking", 2).await.expect("port search");
    assert!(!hits.is_empty());
    assert_eq!(hits[0].doc_id, "doc_port_1");
    assert_eq!(hits[0].title, "Vocabulary Builder");
    assert_eq!(hits[0].citation, "[Vocabulary Builder] (doc_port_1)");

    port.delete_document("doc_port_1").await.expect("port delete");
    assert_eq!(port.count(), 0);
}

#[test]
fn test_lancedb_vector_store_advanced_crud() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = LanceDbVectorStore::connect(dir.path()).expect("connect");

    let table1 = store.open_table("table_a").expect("open a");
    table1
        .add(vec![
            VectorRecord::new("chunk_1", "doc_1", "Text 1", vec![1.0, 0.0]),
            VectorRecord::new("chunk_2", "doc_2", "Text 2", vec![0.0, 1.0]),
        ])
        .expect("add");

    assert_eq!(table1.count(), 2);
    assert!(table1.delete_by_chunk_id("chunk_1").expect("del chunk"));
    assert_eq!(table1.count(), 1);

    table1.clear().expect("clear");
    assert_eq!(table1.count(), 0);

    let table2 = store.open_table("table_b").expect("open b");
    table2
        .add(vec![VectorRecord::new("chunk_3", "doc_3", "Text 3", vec![0.5, 0.5])])
        .expect("add");

    let tables = store.list_tables().expect("list");
    assert!(tables.contains(&"table_a".to_string()));
    assert!(tables.contains(&"table_b".to_string()));

    assert!(store.drop_table("table_a").expect("drop"));
    let tables_after = store.list_tables().expect("list after");
    assert!(!tables_after.contains(&"table_a".to_string()));
}

#[tokio::test]
async fn test_hybrid_rag_empty_and_whitespace_search() {
    let engine = HybridRagEngine::new_in_memory(HybridRagConfig::default()).expect("engine");
    let results = engine.search("   ", 5).await.expect("search empty");
    assert!(results.is_empty());

    let results_zero_limit = engine.search("something", 0).await.expect("search limit 0");
    assert!(results_zero_limit.is_empty());
}

#[test]
fn test_top_k_min_heap_selection_accuracy() {
    let mut store = LocalVectorStore::new();

    // Insert 5 records with known angles relative to [1.0, 0.0]
    // Vector Record 1: [0.1, 0.99] -> low similarity with [1.0, 0.0] (~0.10)
    // Vector Record 2: [0.9, 0.43] -> high similarity (~0.90)
    // Vector Record 3: [0.5, 0.86] -> medium similarity (~0.50)
    // Vector Record 4: [0.98, 0.19] -> highest similarity (~0.98)
    // Vector Record 5: [0.2, 0.97] -> low similarity (~0.20)
    store.insert(VectorRecord::new("c1", "doc1", "Doc 1", vec![0.1, 0.99]));
    store.insert(VectorRecord::new("c2", "doc2", "Doc 2", vec![0.9, 0.43]));
    store.insert(VectorRecord::new("c3", "doc3", "Doc 3", vec![0.5, 0.86]));
    store.insert(VectorRecord::new("c4", "doc4", "Doc 4", vec![0.98, 0.19]));
    store.insert(VectorRecord::new("c5", "doc5", "Doc 5", vec![0.2, 0.97]));

    // Query for top 2 records closest to [1.0, 0.0]
    let hits = store.search(&[1.0, 0.0], 2);
    assert_eq!(hits.len(), 2, "Should return exactly limit=2 items");

    // The top 2 must be c4 (score ~0.98) and c2 (score ~0.90), strictly ordered descending
    assert_eq!(hits[0].chunk_id, "c4");
    assert_eq!(hits[1].chunk_id, "c2");
    assert!(hits[0].score > hits[1].score);
    assert!(hits[1].score > 0.85);
}

#[test]
fn test_simd_dot_product_performance_and_accuracy() {
    let a: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
    let b: Vec<f32> = (0..384).map(|i| (i as f32 * 0.02).cos()).collect();
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..10_000 {
        sum += vira_rag::vector_store::dot_product(&a, &b);
    }
    let elapsed = start.elapsed();
    assert!(sum.is_finite());
    assert!(
        elapsed.as_millis() < 50,
        "10,000 384-d dot products should finish in < 50ms, took {:?}",
        elapsed
    );
}

