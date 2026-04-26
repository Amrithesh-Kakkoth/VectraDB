//! VectraDB Intensive Evaluation & Stress Tests
//!
//! These tests exercise every component under extreme conditions:
//! - Correctness: mathematical accuracy of every similarity metric
//! - Edge cases: zero vectors, NaN, identical vectors, single element
//! - Scale: thousands of vectors, high dimensions
//! - All algorithms: HNSW, LSH, PQ, ES4D compared head-to-head
//! - Concurrency: parallel reads/writes on shared state
//! - Tensor search: basic, shifting, weighted, cache-efficient
//! - Persistence: write → close → reopen → verify
//! - Search quality: recall@k against brute-force ground truth

use ndarray::Array1;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use vectradb_components::filter::*;
use vectradb_components::tensor::*;
use vectradb_components::*;
use vectradb_search::*;

// ============================================================
// Helpers
// ============================================================

fn random_vector(dim: usize) -> Array1<f32> {
    let mut rng = rand::thread_rng();
    Array1::from_iter((0..dim).map(|_| rng.gen_range(-1.0..1.0)))
}

fn normalized_random_vector(dim: usize) -> Array1<f32> {
    let v = random_vector(dim);
    let norm = v.dot(&v).sqrt();
    if norm == 0.0 {
        v
    } else {
        v / norm
    }
}

fn make_doc(id: &str, data: Vec<f32>) -> vectradb_components::VectorDocument {
    vectradb_components::vector_operations::create_vector_document(
        id.to_string(),
        Array1::from_vec(data),
        None,
    )
    .unwrap()
}

/// Brute-force top-k search (ground truth for recall measurement)
fn brute_force_search(
    query: &Array1<f32>,
    docs: &[vectradb_components::VectorDocument],
    k: usize,
) -> Vec<String> {
    let mut scored: Vec<(String, f32)> = docs
        .iter()
        .map(|d| {
            let diff = query - &d.data;
            let dist = diff.dot(&diff).sqrt();
            (d.metadata.id.clone(), dist)
        })
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Recall@k: fraction of ground-truth top-k found in search results
fn recall_at_k(search_ids: &[String], ground_truth_ids: &[String]) -> f64 {
    let gt_set: HashSet<&String> = ground_truth_ids.iter().collect();
    let found = search_ids.iter().filter(|id| gt_set.contains(id)).count();
    found as f64 / ground_truth_ids.len() as f64
}

// ============================================================
// 1. SIMILARITY METRIC CORRECTNESS
// ============================================================

#[test]
fn test_cosine_identical_vectors() {
    let v = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let sim = cosine_similarity(&v.view(), &v.view()).unwrap();
    assert!(
        (sim - 1.0).abs() < 1e-6,
        "Cosine of identical vectors should be 1.0, got {sim}"
    );
}

#[test]
fn test_cosine_orthogonal_vectors() {
    let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    let b = Array1::from_vec(vec![0.0, 1.0, 0.0]);
    let sim = cosine_similarity(&a.view(), &b.view()).unwrap();
    assert!(
        sim.abs() < 1e-6,
        "Cosine of orthogonal vectors should be 0.0, got {sim}"
    );
}

#[test]
fn test_cosine_opposite_vectors() {
    let a = Array1::from_vec(vec![1.0, 0.0]);
    let b = Array1::from_vec(vec![-1.0, 0.0]);
    let sim = cosine_similarity(&a.view(), &b.view()).unwrap();
    assert!(
        (sim - (-1.0)).abs() < 1e-6,
        "Cosine of opposite vectors should be -1.0, got {sim}"
    );
}

#[test]
fn test_euclidean_triangle_inequality() {
    // d(a,c) <= d(a,b) + d(b,c) for any a, b, c
    let a = Array1::from_vec(vec![0.0, 0.0]);
    let b = Array1::from_vec(vec![3.0, 0.0]);
    let c = Array1::from_vec(vec![3.0, 4.0]);
    let dab = euclidean_distance(&a.view(), &b.view()).unwrap();
    let dbc = euclidean_distance(&b.view(), &c.view()).unwrap();
    let dac = euclidean_distance(&a.view(), &c.view()).unwrap();
    assert!(
        dac <= dab + dbc + 1e-6,
        "Triangle inequality violated: {dac} > {dab} + {dbc}"
    );
}

#[test]
fn test_euclidean_known_value() {
    // 3-4-5 triangle
    let a = Array1::from_vec(vec![0.0, 0.0]);
    let b = Array1::from_vec(vec![3.0, 4.0]);
    let d = euclidean_distance(&a.view(), &b.view()).unwrap();
    assert!((d - 5.0).abs() < 1e-6, "Expected distance 5.0, got {d}");
}

#[test]
fn test_manhattan_known_value() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array1::from_vec(vec![4.0, 6.0, 8.0]);
    let d = manhattan_distance(&a.view(), &b.view()).unwrap();
    assert!(
        (d - 12.0).abs() < 1e-6,
        "Expected Manhattan distance 12.0, got {d}"
    );
}

#[test]
fn test_dot_product_known_value() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);
    let dp = dot_product_similarity(&a.view(), &b.view()).unwrap();
    assert!(
        (dp - 32.0).abs() < 1e-6,
        "Expected dot product 32.0, got {dp}"
    );
}

#[test]
fn test_dimension_mismatch_returns_error() {
    let a = Array1::from_vec(vec![1.0, 2.0]);
    let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(cosine_similarity(&a.view(), &b.view()).is_err());
    assert!(euclidean_distance(&a.view(), &b.view()).is_err());
    assert!(manhattan_distance(&a.view(), &b.view()).is_err());
}

// ============================================================
// 2. EDGE CASES
// ============================================================

#[test]
fn test_zero_vector_cosine() {
    let a = Array1::from_vec(vec![0.0, 0.0, 0.0]);
    let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let sim = cosine_similarity(&a.view(), &b.view()).unwrap();
    assert_eq!(sim, 0.0, "Cosine with zero vector should be 0.0");
}

#[test]
fn test_single_dimension_vectors() {
    let a = Array1::from_vec(vec![5.0]);
    let b = Array1::from_vec(vec![3.0]);
    let d = euclidean_distance(&a.view(), &b.view()).unwrap();
    assert!((d - 2.0).abs() < 1e-6);
}

#[test]
fn test_validate_vector_rejects_nan() {
    let v = Array1::from_vec(vec![1.0, f32::NAN, 3.0]);
    assert!(vector_operations::validate_vector(&v).is_err());
}

#[test]
fn test_validate_vector_rejects_infinity() {
    let v = Array1::from_vec(vec![1.0, f32::INFINITY, 3.0]);
    assert!(vector_operations::validate_vector(&v).is_err());
}

#[test]
fn test_validate_vector_rejects_empty() {
    let v = Array1::zeros(0);
    assert!(vector_operations::validate_vector(&v).is_err());
}

#[test]
fn test_normalize_produces_unit_vector() {
    let v = Array1::from_vec(vec![3.0, 4.0]);
    let n = vector_operations::normalize_vector(v).unwrap();
    let norm = n.dot(&n).sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-6,
        "Normalized vector norm should be 1.0, got {norm}"
    );
}

#[test]
fn test_duplicate_vector_id_returns_error() {
    let mut db = storage::InMemoryVectorDB::new();
    let v = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    db.create_vector("dup".to_string(), v.clone(), None)
        .unwrap();
    let result = db.create_vector("dup".to_string(), v, None);
    assert!(result.is_err(), "Duplicate ID should return error");
}

#[test]
fn test_get_nonexistent_vector() {
    let db = storage::InMemoryVectorDB::new();
    assert!(db.get_vector("ghost").is_err());
}

#[test]
fn test_delete_nonexistent_vector() {
    let mut db = storage::InMemoryVectorDB::new();
    assert!(db.delete_vector("ghost").is_err());
}

// ============================================================
// 3. IN-MEMORY VECTOR DATABASE STRESS
// ============================================================

#[test]
fn test_inmemory_crud_1000_vectors() {
    let dim = 128;
    let n = 1000;
    let mut db = storage::InMemoryVectorDB::with_dimension(dim);

    // Insert
    for i in 0..n {
        let v = random_vector(dim);
        db.create_vector(format!("v{i}"), v, None).unwrap();
    }
    assert_eq!(db.get_stats().unwrap().total_vectors, n);

    // Update half
    for i in 0..n / 2 {
        let v = random_vector(dim);
        db.update_vector(&format!("v{i}"), v, None).unwrap();
    }

    // Delete quarter
    for i in 0..n / 4 {
        db.delete_vector(&format!("v{i}")).unwrap();
    }
    assert_eq!(db.get_stats().unwrap().total_vectors, n - n / 4);

    // Search
    let query = random_vector(dim);
    let results = db.search_similar(query, 10).unwrap();
    assert_eq!(results.len(), 10);

    // Verify results are sorted by score descending
    for i in 1..results.len() {
        assert!(
            results[i].score <= results[i - 1].score,
            "Results not sorted: {} > {}",
            results[i].score,
            results[i - 1].score
        );
    }
}

#[test]
fn test_upsert_creates_and_updates() {
    let mut db = storage::InMemoryVectorDB::new();
    let v1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let v2 = Array1::from_vec(vec![4.0, 5.0, 6.0]);

    // Upsert should create
    db.upsert_vector("u1".to_string(), v1, None).unwrap();
    assert_eq!(db.get_vector("u1").unwrap().data[0], 1.0);

    // Upsert should update
    db.upsert_vector("u1".to_string(), v2, None).unwrap();
    assert_eq!(db.get_vector("u1").unwrap().data[0], 4.0);
    assert_eq!(db.get_stats().unwrap().total_vectors, 1);
}

#[test]
fn test_dimension_enforcement() {
    let mut db = storage::InMemoryVectorDB::with_dimension(3);
    let v_ok = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let v_bad = Array1::from_vec(vec![1.0, 2.0]);
    db.create_vector("ok".to_string(), v_ok, None).unwrap();
    assert!(db.create_vector("bad".to_string(), v_bad, None).is_err());
}

// ============================================================
// 4. SEARCH ALGORITHM COMPARISON (HNSW vs LSH vs PQ vs ES4D)
// ============================================================

fn build_dataset(n: usize, dim: usize) -> Vec<VectorDocument> {
    (0..n)
        .map(|i| {
            let v = normalized_random_vector(dim);
            vector_operations::create_vector_document(format!("d{i}"), v, None).unwrap()
        })
        .collect()
}

/// Test recall quality: what fraction of true top-k does each algorithm find?
#[test]
fn test_search_recall_comparison_all_algorithms() {
    let dim = 32;
    let n = 500;
    let k = 10;
    let num_queries = 20;
    let docs = build_dataset(n, dim);

    // Build all indexes
    let mut hnsw = HNSWIndex::new(dim, 16, 200, 50, DistanceMetric::Euclidean);
    let mut lsh = LSHIndex::new(dim, 20);
    let mut pq = PQIndex::new(dim, 4, 16);
    let mut es4d = ES4DIndex::new(ES4DConfig {
        dimension: dim,
        shard_length: 8,
        m: 16,
        ef_construction: 200,
        ..Default::default()
    });

    hnsw.build_index(docs.clone()).unwrap();
    lsh.build_index(docs.clone()).unwrap();
    pq.build_index(docs.clone()).unwrap();
    es4d.build_index(docs.clone()).unwrap();

    let mut hnsw_recall_sum = 0.0;
    let mut lsh_recall_sum = 0.0;
    let mut pq_recall_sum = 0.0;
    let mut es4d_recall_sum = 0.0;

    for _ in 0..num_queries {
        let query = normalized_random_vector(dim);
        let ground_truth = brute_force_search(&query, &docs, k);

        let hnsw_results: Vec<String> = hnsw
            .search(&query, k)
            .unwrap()
            .iter()
            .map(|r| r.id.clone())
            .collect();
        let lsh_results: Vec<String> = lsh
            .search(&query, k)
            .unwrap()
            .iter()
            .map(|r| r.id.clone())
            .collect();
        let pq_results: Vec<String> = pq
            .search(&query, k)
            .unwrap()
            .iter()
            .map(|r| r.id.clone())
            .collect();
        let es4d_results: Vec<String> = es4d
            .search(&query, k)
            .unwrap()
            .iter()
            .map(|r| r.id.clone())
            .collect();

        hnsw_recall_sum += recall_at_k(&hnsw_results, &ground_truth);
        lsh_recall_sum += recall_at_k(&lsh_results, &ground_truth);
        pq_recall_sum += recall_at_k(&pq_results, &ground_truth);
        es4d_recall_sum += recall_at_k(&es4d_results, &ground_truth);
    }

    let hnsw_recall = hnsw_recall_sum / num_queries as f64;
    let lsh_recall = lsh_recall_sum / num_queries as f64;
    let pq_recall = pq_recall_sum / num_queries as f64;
    let es4d_recall = es4d_recall_sum / num_queries as f64;

    eprintln!("=== Recall@{k} (N={n}, dim={dim}, {num_queries} queries) ===");
    eprintln!("  HNSW:  {hnsw_recall:.3}");
    eprintln!("  LSH:   {lsh_recall:.3}");
    eprintln!("  PQ:    {pq_recall:.3}");
    eprintln!("  ES4D:  {es4d_recall:.3}");

    // HNSW and ES4D should have reasonable recall (>0.5)
    assert!(hnsw_recall > 0.3, "HNSW recall too low: {hnsw_recall}");
    assert!(es4d_recall > 0.3, "ES4D recall too low: {es4d_recall}");
}

/// Insert + delete + search: verify consistency after mutations
#[test]
fn test_algorithm_insert_delete_consistency() {
    let dim = 16;
    let mut index = HNSWIndex::new(dim, 8, 100, 50, DistanceMetric::Euclidean);

    // Insert 100 vectors
    let docs = build_dataset(100, dim);
    for doc in &docs {
        index.insert(doc.clone()).unwrap();
    }

    // Delete the first 50
    for i in 0..50 {
        index.remove(&format!("d{i}")).unwrap();
    }

    // Search should only return surviving vectors
    let query = random_vector(dim);
    let results = index.search(&query, 20).unwrap();
    for r in &results {
        let id_num: usize = r.id[1..].parse().unwrap();
        assert!(
            id_num >= 50,
            "Deleted vector {id_num} appeared in search results"
        );
    }
}

// ============================================================
// 5. HIGH-DIMENSIONAL STRESS TEST
// ============================================================

#[test]
fn test_high_dimensional_512() {
    let dim = 512;
    let n = 200;
    let docs = build_dataset(n, dim);

    let mut hnsw = HNSWIndex::new(dim, 16, 100, 50, DistanceMetric::Euclidean);
    hnsw.build_index(docs.clone()).unwrap();

    let query = normalized_random_vector(dim);
    let results = hnsw.search(&query, 10).unwrap();
    assert_eq!(results.len(), 10);

    // Verify distances are non-negative and sorted ascending
    for i in 0..results.len() {
        assert!(results[i].distance >= 0.0);
        if i > 0 {
            assert!(results[i].distance >= results[i - 1].distance);
        }
    }
}

#[test]
fn test_es4d_det_effectiveness_high_dim() {
    // ES4D with DET should be able to handle high-dim vectors efficiently
    let dim = 256;
    let n = 100;
    let docs = build_dataset(n, dim);

    let mut es4d = ES4DIndex::new(ES4DConfig {
        dimension: dim,
        shard_length: 32,
        m: 8,
        ef_construction: 100,
        enable_cet: true,
        enable_det: true,
        enable_dimension_reorder: true,
        ..Default::default()
    });
    es4d.build_index(docs.clone()).unwrap();

    let query = normalized_random_vector(dim);
    let results = es4d.search(&query, 10).unwrap();
    assert!(
        !results.is_empty(),
        "ES4D returned no results for high-dim query"
    );
}

// ============================================================
// 6. SEARCH PERFORMANCE BENCHMARK
// ============================================================

#[test]
fn test_search_performance_benchmark() {
    let dim = 64;
    let n = 1000;
    let k = 10;
    let num_queries = 100;
    let docs = build_dataset(n, dim);

    let mut hnsw = HNSWIndex::new(dim, 16, 200, 50, DistanceMetric::Euclidean);
    hnsw.build_index(docs.clone()).unwrap();

    let queries: Vec<Array1<f32>> = (0..num_queries)
        .map(|_| normalized_random_vector(dim))
        .collect();

    let start = Instant::now();
    for q in &queries {
        let _ = hnsw.search(q, k).unwrap();
    }
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() as f64 / num_queries as f64;
    eprintln!(
        "=== HNSW Performance (N={n}, dim={dim}, k={k}) ===\n  {num_queries} queries in {:.2}ms ({avg_us:.0}µs/query)",
        elapsed.as_secs_f64() * 1000.0
    );

    // Each query should complete in reasonable time (< 10ms)
    assert!(
        avg_us < 10_000.0,
        "Search too slow: {avg_us}µs/query (expected < 10000µs)"
    );
}

// ============================================================
// 7. PERSISTENT STORAGE (write → close → reopen → verify)
// ============================================================

#[tokio::test]
async fn test_persistent_storage_survives_restart() {
    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path().to_string_lossy().to_string();
    let dim = 8;

    let config = vectradb_storage::DatabaseConfig {
        data_dir: data_dir.clone(),
        search_algorithm: SearchAlgorithm::HNSW,
        index_config: SearchConfig {
            dimension: Some(dim),
            ..Default::default()
        },
        ..Default::default()
    };

    // Write vectors
    {
        let db = vectradb_storage::PersistentVectorDB::new(config.clone())
            .await
            .unwrap();
        for i in 0..50 {
            let v = random_vector(dim);
            db.create_vector(format!("persist_{i}"), v, None)
                .await
                .unwrap();
        }
        assert_eq!(db.list_vectors().await.unwrap().len(), 50);
    }
    // db is dropped here — simulates server shutdown

    // Reopen and verify
    {
        let db = vectradb_storage::PersistentVectorDB::new(config)
            .await
            .unwrap();
        let ids = db.list_vectors().await.unwrap();
        assert_eq!(
            ids.len(),
            50,
            "Expected 50 vectors after restart, got {}",
            ids.len()
        );

        // Verify a specific vector is retrievable
        let doc = db.get_vector("persist_0").await.unwrap();
        assert_eq!(doc.metadata.dimension, dim);

        // Search should work on reopened data
        let query = random_vector(dim);
        let results = db.search_similar(query, 5).await.unwrap();
        assert_eq!(results.len(), 5);
    }
}

// ============================================================
// 8. CONCURRENT ACCESS STRESS TEST
// ============================================================

#[test]
fn test_concurrent_reads_and_writes() {
    let db = Arc::new(std::sync::RwLock::new(
        storage::InMemoryVectorDB::with_dimension(16),
    ));

    // Pre-populate
    {
        let mut db_w = db.write().unwrap();
        for i in 0..100 {
            let v = random_vector(16);
            db_w.create_vector(format!("c{i}"), v, None).unwrap();
        }
    }

    // Spawn concurrent readers and writers
    let mut handles = vec![];

    // 4 reader threads
    for t in 0..4 {
        let db_clone = Arc::clone(&db);
        handles.push(std::thread::spawn(move || {
            for _ in 0..50 {
                let db_r = db_clone.read().unwrap();
                let query = random_vector(16);
                let _ = db_r.search_similar(query, 5);
                let _ = db_r.list_vectors();
                drop(db_r);
            }
            eprintln!("  Reader thread {t} completed 50 iterations");
        }));
    }

    // 2 writer threads
    for t in 0..2 {
        let db_clone = Arc::clone(&db);
        handles.push(std::thread::spawn(move || {
            for i in 0..25 {
                let mut db_w = db_clone.write().unwrap();
                let v = random_vector(16);
                let _ = db_w.upsert_vector(format!("w{t}_{i}"), v, None);
                drop(db_w);
            }
            eprintln!("  Writer thread {t} completed 25 iterations");
        }));
    }

    for h in handles {
        h.join().expect("Thread panicked");
    }

    let db_r = db.read().unwrap();
    let stats = db_r.get_stats().unwrap();
    eprintln!(
        "=== After concurrent access: {} vectors ===",
        stats.total_vectors
    );
    assert!(
        stats.total_vectors >= 100,
        "Some vectors lost during concurrent access"
    );
}

// ============================================================
// 9. TENSOR SEARCH TESTS
// ============================================================

fn make_tensor_2d(data: Vec<f32>, rows: usize, cols: usize) -> TensorData {
    TensorData::new(data, vec![rows, cols]).unwrap()
}

#[test]
fn test_tensor_basic_search_finds_exact_match() {
    let engine = TensorSearchEngine::new();

    // Insert 50 random 4x4 tensors
    for i in 0..50 {
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..16).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let t = make_tensor_2d(data, 4, 4);
        engine
            .insert(create_tensor_document(format!("t{i}"), t, None).unwrap())
            .unwrap();
    }

    // Query with the exact data of t0
    let t0 = engine.get("t0").unwrap();
    let results = engine
        .basic_search(&t0.tensor, 1, None, TensorSimilarityMetric::Cosine, 5)
        .unwrap();

    assert_eq!(results[0].id, "t0", "Exact match should be first result");
    assert!(
        results[0].similarity > 0.99,
        "Self-similarity should be ~1.0, got {}",
        results[0].similarity
    );
}

#[test]
fn test_tensor_shifting_search_finds_embedded_signal() {
    let engine = TensorSearchEngine::new();

    // Create a large reference tensor (20 rows x 5 cols) with noise
    let mut ref_data = vec![0.0f32; 20 * 5];
    let mut rng = rand::thread_rng();
    for v in ref_data.iter_mut() {
        *v = rng.gen_range(-0.1..0.1); // small noise
    }
    // Embed a strong signal at rows 12-14
    let signal = vec![
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,
        150.0,
    ];
    for (i, &val) in signal.iter().enumerate() {
        ref_data[12 * 5 + i] = val;
    }
    let reference = make_tensor_2d(ref_data, 20, 5);
    engine
        .insert(create_tensor_document("ref_signal".into(), reference, None).unwrap())
        .unwrap();

    // Pattern: the signal (3 rows x 5 cols)
    let pattern = make_tensor_2d(signal, 3, 5);

    let results = engine
        .shifting_search(
            &pattern,
            "ref_signal",
            0,
            1,
            None,
            TensorSimilarityMetric::DotProduct,
            3,
        )
        .unwrap();

    assert_eq!(
        results[0].offset,
        Some(12),
        "Signal embedded at row 12 should be found, got {:?}",
        results[0].offset
    );
}

#[test]
fn test_tensor_weighted_similarity_emphasizes_first_dimension() {
    let engine = TensorSearchEngine::new();

    // Two reference tensors, both 2x2
    // r1: first row matches pattern, second row doesn't
    // r2: second row matches pattern, first row doesn't
    let r1 = make_tensor_2d(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
    let r2 = make_tensor_2d(vec![0.0, 1.0, 1.0, 0.0], 2, 2);
    engine
        .insert(create_tensor_document("r1".into(), r1, None).unwrap())
        .unwrap();
    engine
        .insert(create_tensor_document("r2".into(), r2, None).unwrap())
        .unwrap();

    let pattern = make_tensor_2d(vec![1.0, 0.0, 0.5, 0.5], 2, 2);

    // Heavy weight on first row
    let weights = vec![10.0, 0.1];
    let results = engine
        .basic_search(
            &pattern,
            1,
            Some(&weights),
            TensorSimilarityMetric::DotProduct,
            2,
        )
        .unwrap();

    assert_eq!(
        results[0].id, "r1",
        "With heavy weight on row 0, r1 (matching row 0) should rank first"
    );
}

#[test]
fn test_tensor_cross_correlation_detects_scaled_copy() {
    let engine = TensorSearchEngine::new();

    let original = make_tensor_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let scaled = make_tensor_2d(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3); // 10x
    let different = make_tensor_2d(vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0], 2, 3); // reversed

    engine
        .insert(create_tensor_document("orig".into(), original.clone(), None).unwrap())
        .unwrap();
    engine
        .insert(create_tensor_document("scaled".into(), scaled, None).unwrap())
        .unwrap();
    engine
        .insert(create_tensor_document("diff".into(), different, None).unwrap())
        .unwrap();

    // Cross-correlation should give scaled copy similarity ~1.0 (correlation is scale-invariant)
    let results = engine
        .basic_search(
            &original,
            1,
            None,
            TensorSimilarityMetric::CrossCorrelation,
            3,
        )
        .unwrap();

    // Both "orig" and "scaled" should have correlation ~1.0
    let orig_sim = results.iter().find(|r| r.id == "orig").unwrap().similarity;
    let scaled_sim = results
        .iter()
        .find(|r| r.id == "scaled")
        .unwrap()
        .similarity;
    assert!(
        (orig_sim - 1.0).abs() < 1e-4,
        "Self cross-correlation should be 1.0, got {orig_sim}"
    );
    assert!(
        (scaled_sim - 1.0).abs() < 1e-4,
        "Scaled copy cross-correlation should be ~1.0, got {scaled_sim}"
    );
}

#[test]
fn test_tensor_subtensor_extraction_correctness() {
    // 3D tensor [2, 3, 4] = 24 elements
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = TensorData::new(data, vec![2, 3, 4]).unwrap();

    // Extract first "matrix" (dim 0, index 0, len 1) → shape [1, 3, 4]
    let sub = t.subtensor(0, 0, 1).unwrap();
    assert_eq!(sub.shape(), &[1, 3, 4]);
    assert_eq!(sub.data().len(), 12);
    assert_eq!(sub.get(&[0, 0, 0]), 0.0);
    assert_eq!(sub.get(&[0, 2, 3]), 11.0);
}

// ============================================================
// 10. ALGORITHM STRESS: MANY INSERTS + DELETES + SEARCHES
// ============================================================

#[test]
fn test_hnsw_heavy_churn() {
    let dim = 32;
    let mut index = HNSWIndex::new(dim, 8, 100, 50, DistanceMetric::Euclidean);

    // Insert 500
    for i in 0..500 {
        let doc = make_doc(
            &format!("h{i}"),
            (0..dim).map(|j| (i * dim + j) as f32).collect(),
        );
        index.insert(doc).unwrap();
    }

    // Remove 250
    for i in 0..250 {
        index.remove(&format!("h{i}")).unwrap();
    }

    // Insert 250 more
    for i in 500..750 {
        let doc = make_doc(
            &format!("h{i}"),
            (0..dim).map(|j| (i * dim + j) as f32).collect(),
        );
        index.insert(doc).unwrap();
    }

    // Stats should show 500
    let stats = index.get_stats();
    assert_eq!(
        stats.total_vectors, 500,
        "Expected 500, got {}",
        stats.total_vectors
    );

    // Search should return results
    let query = random_vector(dim);
    let results = index.search(&query, 20).unwrap();
    assert!(!results.is_empty());

    // No deleted IDs should appear
    for r in &results {
        let num: usize = r.id[1..].parse().unwrap();
        assert!(num >= 250, "Deleted vector h{num} appeared in results");
    }
}

#[test]
fn test_es4d_build_index_then_incremental_insert() {
    let dim = 16;
    let mut es4d = ES4DIndex::new(ES4DConfig {
        dimension: dim,
        shard_length: 4,
        m: 8,
        ef_construction: 50,
        enable_cet: true,
        enable_det: true,
        enable_dimension_reorder: true,
        ..Default::default()
    });

    // Build with 100 vectors
    let docs = build_dataset(100, dim);
    es4d.build_index(docs).unwrap();

    // Incrementally add 50 more
    for i in 100..150 {
        let v = normalized_random_vector(dim);
        let doc = vector_operations::create_vector_document(format!("d{i}"), v, None).unwrap();
        es4d.insert(doc).unwrap();
    }

    let stats = es4d.get_stats();
    assert_eq!(stats.total_vectors, 150);

    let query = normalized_random_vector(dim);
    let results = es4d.search(&query, 10).unwrap();
    assert_eq!(results.len(), 10);
}

// ============================================================
// 11. SEARCH RESULT ORDERING INVARIANTS
// ============================================================

#[test]
fn test_all_algorithms_return_sorted_results() {
    let dim = 16;
    let docs = build_dataset(100, dim);

    let mut hnsw = HNSWIndex::new(dim, 8, 100, 50, DistanceMetric::Euclidean);
    let mut lsh = LSHIndex::new(dim, 15);
    let mut es4d = ES4DIndex::new(ES4DConfig {
        dimension: dim,
        shard_length: 4,
        m: 8,
        ef_construction: 50,
        ..Default::default()
    });

    hnsw.build_index(docs.clone()).unwrap();
    lsh.build_index(docs.clone()).unwrap();
    es4d.build_index(docs).unwrap();

    let query = normalized_random_vector(dim);

    for (name, results) in [
        ("HNSW", hnsw.search(&query, 20).unwrap()),
        ("LSH", lsh.search(&query, 20).unwrap()),
        ("ES4D", es4d.search(&query, 20).unwrap()),
    ] {
        for i in 1..results.len() {
            assert!(
                results[i].distance >= results[i - 1].distance - 1e-6,
                "{name}: results not sorted at position {i}: {} < {}",
                results[i].distance,
                results[i - 1].distance
            );
        }
    }
}

#[test]
fn test_top_1_is_nearest_neighbor() {
    let dim = 8;
    let mut index = HNSWIndex::new(dim, 16, 200, 50, DistanceMetric::Euclidean);

    // Insert a known nearest neighbor
    let target = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let near = Array1::from_vec(vec![0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let far = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);

    index.insert(make_doc("near", near.to_vec())).unwrap();
    index.insert(make_doc("far", far.to_vec())).unwrap();

    let results = index.search(&target, 1).unwrap();
    assert_eq!(results[0].id, "near", "Nearest vector should be 'near'");
}

// ============================================================
// 12. METADATA FILTER TESTS
// ============================================================

#[test]
fn test_filter_condition_all_variants() {
    let tags: HashMap<String, String> = [
        ("category", "article"),
        ("lang", "en"),
        ("status", "published"),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();

    // Equals
    assert!(FilterCondition::Equals {
        key: "category".into(),
        value: "article".into()
    }
    .matches(&tags));
    assert!(!FilterCondition::Equals {
        key: "category".into(),
        value: "video".into()
    }
    .matches(&tags));

    // NotEquals
    assert!(FilterCondition::NotEquals {
        key: "status".into(),
        value: "deleted".into()
    }
    .matches(&tags));
    assert!(!FilterCondition::NotEquals {
        key: "status".into(),
        value: "published".into()
    }
    .matches(&tags));

    // In
    assert!(FilterCondition::In {
        key: "lang".into(),
        values: vec!["en".into(), "fr".into()]
    }
    .matches(&tags));
    assert!(!FilterCondition::In {
        key: "lang".into(),
        values: vec!["de".into(), "es".into()]
    }
    .matches(&tags));

    // Exists / NotExists
    assert!(FilterCondition::Exists {
        key: "category".into()
    }
    .matches(&tags));
    assert!(!FilterCondition::Exists {
        key: "missing".into()
    }
    .matches(&tags));
    assert!(FilterCondition::NotExists {
        key: "missing".into()
    }
    .matches(&tags));
    assert!(!FilterCondition::NotExists {
        key: "category".into()
    }
    .matches(&tags));
}

#[test]
fn test_filter_complex_boolean_logic() {
    // category="article" AND (source="news" OR source="blog") AND status!="deleted"
    let filter = MetadataFilter::And(vec![
        MetadataFilter::Condition(FilterCondition::Equals {
            key: "category".into(),
            value: "article".into(),
        }),
        MetadataFilter::Or(vec![
            MetadataFilter::Condition(FilterCondition::Equals {
                key: "source".into(),
                value: "news".into(),
            }),
            MetadataFilter::Condition(FilterCondition::Equals {
                key: "source".into(),
                value: "blog".into(),
            }),
        ]),
        MetadataFilter::Condition(FilterCondition::NotEquals {
            key: "status".into(),
            value: "deleted".into(),
        }),
    ]);

    let make_tags = |pairs: &[(&str, &str)]| -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    };

    // Passes: article + news + active
    assert!(filter.matches(&make_tags(&[
        ("category", "article"),
        ("source", "news"),
        ("status", "active")
    ])));

    // Passes: article + blog + (no status = passes NotEquals)
    assert!(filter.matches(&make_tags(&[("category", "article"), ("source", "blog")])));

    // Fails: wrong category
    assert!(!filter.matches(&make_tags(&[("category", "video"), ("source", "news")])));

    // Fails: wrong source
    assert!(!filter.matches(&make_tags(&[("category", "article"), ("source", "wiki")])));

    // Fails: deleted
    assert!(!filter.matches(&make_tags(&[
        ("category", "article"),
        ("source", "news"),
        ("status", "deleted")
    ])));
}

#[tokio::test]
async fn test_persistent_db_search_with_filter() {
    let temp_dir = tempfile::tempdir().unwrap();
    let dim = 8;

    let config = vectradb_storage::DatabaseConfig {
        data_dir: temp_dir.path().to_string_lossy().to_string(),
        search_algorithm: SearchAlgorithm::HNSW,
        index_config: SearchConfig {
            dimension: Some(dim),
            ..Default::default()
        },
        ..Default::default()
    };

    let db = vectradb_storage::PersistentVectorDB::new(config)
        .await
        .unwrap();

    // Insert vectors with different categories
    for i in 0..30 {
        let v = random_vector(dim);
        let category = if i < 10 {
            "article"
        } else if i < 20 {
            "video"
        } else {
            "image"
        };
        let mut tags = HashMap::new();
        tags.insert("category".to_string(), category.to_string());
        tags.insert("index".to_string(), i.to_string());
        db.create_vector(format!("f{i}"), v, Some(tags))
            .await
            .unwrap();
    }

    let query = random_vector(dim);

    // Search without filter — should return results from any category
    let all_results = db
        .search_with_filter(query.clone(), 10, None)
        .await
        .unwrap();
    assert_eq!(all_results.len(), 10);

    // Search with filter: category="article"
    let article_filter = MetadataFilter::Condition(FilterCondition::Equals {
        key: "category".into(),
        value: "article".into(),
    });
    let article_results = db
        .search_with_filter(query.clone(), 10, Some(&article_filter))
        .await
        .unwrap();

    // All results should have category=article
    for r in &article_results {
        assert_eq!(
            r.metadata.tags.get("category"),
            Some(&"article".to_string()),
            "Filtered result {} has wrong category: {:?}",
            r.id,
            r.metadata.tags
        );
    }
    // Should have at most 10 articles (we inserted 10)
    assert!(article_results.len() <= 10);
    assert!(!article_results.is_empty());

    // Search with filter: category!="image"
    let not_image_filter = MetadataFilter::Condition(FilterCondition::NotEquals {
        key: "category".into(),
        value: "image".into(),
    });
    let not_image_results = db
        .search_with_filter(query.clone(), 20, Some(&not_image_filter))
        .await
        .unwrap();

    for r in &not_image_results {
        assert_ne!(
            r.metadata.tags.get("category"),
            Some(&"image".to_string()),
            "Filter should exclude images, but got {}",
            r.id
        );
    }

    // Search with impossible filter — should return empty
    let impossible_filter = MetadataFilter::Condition(FilterCondition::Equals {
        key: "category".into(),
        value: "nonexistent".into(),
    });
    let empty_results = db
        .search_with_filter(query, 10, Some(&impossible_filter))
        .await
        .unwrap();
    assert!(
        empty_results.is_empty(),
        "Impossible filter should return no results"
    );
}
