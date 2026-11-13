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
    assert!(results[0].citation.contains("Ngữ pháp tiếng Anh cơ bản"));
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
        assert!(hits[0].text.contains("Fluency"));

        // Delete document
        engine.delete_document("doc_ielts_01").await.expect("delete");
        assert_eq!(engine.count_chunks(), 0);

        let hits_after_del = engine.search("fluency", 3).await.expect("search after delete");
        assert!(hits_after_del.is_empty());
    }
}

#[tokio::test]
async fn test_hybrid_rag_empty_and_whitespace_search() {
    let engine = HybridRagEngine::new_in_memory(HybridRagConfig::default()).expect("engine");
    let results = engine.search("   ", 5).await.expect("search empty");
    assert!(results.is_empty());

    let results_zero_limit = engine.search("something", 0).await.expect("search limit 0");
    assert!(results_zero_limit.is_empty());
}
