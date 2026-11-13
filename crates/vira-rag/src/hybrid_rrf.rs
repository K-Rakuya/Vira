//! Reciprocal Rank Fusion (RRF) algorithm for Hybrid Search.
//!
//! Combines keyword BM25 ranking and semantic vector search ranking into a
//! unified, balanced relevance score using the standard formula:
//! RRF_score(d) = sum_{m in models} (weight_m / (k + rank_m(d)))

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of reciprocal rank fusion for a candidate document or chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusedDoc {
    /// Document or chunk identifier.
    pub doc_id: String,
    /// Combined reciprocal rank fusion score (higher is more relevant).
    pub score: f64,
    /// 1-based rank in BM25 results, if present.
    pub bm25_rank: Option<usize>,
    /// 1-based rank in Vector results, if present.
    pub vector_rank: Option<usize>,
}

/// Computes reciprocal rank fusion over BM25 and Vector rankings with smoothing constant `k`.
///
/// Default standard value of `k` is 60.
pub fn reciprocal_rank_fusion(
    bm25_ranks: &[String],
    vec_ranks: &[String],
    k: usize,
) -> Vec<FusedDoc> {
    reciprocal_rank_fusion_weighted(bm25_ranks, vec_ranks, k, 1.0, 1.0)
}

/// Computes weighted reciprocal rank fusion with custom weights for BM25 and Vector results.
pub fn reciprocal_rank_fusion_weighted(
    bm25_ranks: &[String],
    vec_ranks: &[String],
    k: usize,
    bm25_weight: f64,
    vec_weight: f64,
) -> Vec<FusedDoc> {
    let k_f64 = k.max(1) as f64;
    let mut scores: HashMap<String, (f64, Option<usize>, Option<usize>)> = HashMap::new();

    for (idx, doc_id) in bm25_ranks.iter().enumerate() {
        let rank = idx + 1; // 1-based
        let entry = scores.entry(doc_id.clone()).or_insert((0.0, None, None));
        entry.0 += bm25_weight / (k_f64 + rank as f64);
        entry.1 = Some(rank);
    }

    for (idx, doc_id) in vec_ranks.iter().enumerate() {
        let rank = idx + 1; // 1-based
        let entry = scores.entry(doc_id.clone()).or_insert((0.0, None, None));
        entry.0 += vec_weight / (k_f64 + rank as f64);
        entry.2 = Some(rank);
    }

    let mut fused: Vec<FusedDoc> = scores
        .into_iter()
        .map(|(doc_id, (score, bm25_rank, vector_rank))| FusedDoc {
            doc_id,
            score,
            bm25_rank,
            vector_rank,
        })
        .collect();

    // Sort descending by score; tie-break deterministically by doc_id
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });

    fused
}

/// Computes reciprocal rank fusion over an arbitrary number of ranked result lists.
pub fn multi_list_rrf(
    ranked_lists: &[(&str, &[String], f64)],
    k: usize,
) -> Vec<FusedDoc> {
    let k_f64 = k.max(1) as f64;
    let mut scores: HashMap<String, f64> = HashMap::new();

    for (_name, list, weight) in ranked_lists {
        for (idx, doc_id) in list.iter().enumerate() {
            let rank = idx + 1;
            *scores.entry(doc_id.clone()).or_insert(0.0) += *weight / (k_f64 + rank as f64);
        }
    }

    let mut fused: Vec<FusedDoc> = scores
        .into_iter()
        .map(|(doc_id, score)| FusedDoc {
            doc_id,
            score,
            bm25_rank: None,
            vector_rank: None,
        })
        .collect();

    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });

    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_both_sources_boosts_item() {
        let bm25 = vec!["docA".to_string(), "docB".to_string()];
        let vec_ranks = vec!["docB".to_string(), "docC".to_string()];
        let fused = reciprocal_rank_fusion(&bm25, &vec_ranks, 60);

        assert_eq!(fused[0].doc_id, "docB");
        assert!(fused[0].bm25_rank.is_some());
        assert!(fused[0].vector_rank.is_some());
        assert!(fused[0].score > fused[1].score);
    }

    #[test]
    fn test_rrf_empty_lists() {
        let empty: Vec<String> = Vec::new();
        let fused = reciprocal_rank_fusion(&empty, &empty, 60);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_rrf_weighted_preference() {
        let bm25 = vec!["docA".to_string()];
        let vec_ranks = vec!["docB".to_string()];
        // Vector has 3x weight
        let fused = reciprocal_rank_fusion_weighted(&bm25, &vec_ranks, 60, 1.0, 3.0);
        assert_eq!(fused[0].doc_id, "docB");
    }
}
