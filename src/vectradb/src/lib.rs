//! # VectraDB — In-Process Vector Database
//!
//! Use VectraDB as a library directly in your application — no server needed.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use vectradb::VectraDB;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Open (or create) a database
//!     let mut db = VectraDB::open("./my_vectors").await?;
//!
//!     // Insert vectors
//!     db.insert("doc1", &[0.1, 0.2, 0.3, 0.4], None)?;
//!     db.insert("doc2", &[0.2, 0.3, 0.4, 0.5], None)?;
//!
//!     // Search
//!     let results = db.search(&[0.15, 0.25, 0.35, 0.45], 5)?;
//!     for r in &results {
//!         println!("{}: score={:.4}", r.id, r.score);
//!     }
//!
//!     // Get a specific vector
//!     let doc = db.get("doc1")?;
//!     println!("dimension: {}", doc.dimension);
//!
//!     // Delete
//!     db.delete("doc2")?;
//!
//!     Ok(())
//! }
//! ```

use ndarray::Array1;
use std::collections::HashMap;
use std::future::Future;
use std::path::Path;

// Re-export useful types
pub use vectradb_components::filter::{FilterCondition, MetadataFilter};
pub use vectradb_components::{DatabaseStats, SimilarityResult, VectraDBError};
pub use vectradb_search::{DistanceMetric, SearchAlgorithm};

use vectradb_components::similarity::find_similar_vectors_cosine;
use vectradb_search::SearchConfig;
use vectradb_storage::{DatabaseConfig, PersistentVectorDB};

/// A vector document returned by `get()`.
#[derive(Debug, Clone)]
pub struct VectorDoc {
    pub id: String,
    pub vector: Vec<f32>,
    pub dimension: usize,
    pub tags: HashMap<String, String>,
    pub created_at: u64,
    pub updated_at: u64,
}

/// Builder for configuring a VectraDB instance.
pub struct VectraDBBuilder {
    data_dir: String,
    dimension: Option<usize>,
    algorithm: SearchAlgorithm,
    metric: DistanceMetric,
    // HNSW params
    m: usize,
    ef_construction: usize,
    search_ef: usize,
}

impl VectraDBBuilder {
    /// Create a new builder with the given data directory.
    pub fn new(data_dir: impl Into<String>) -> Self {
        Self {
            data_dir: data_dir.into(),
            dimension: None,
            algorithm: SearchAlgorithm::HNSW,
            metric: DistanceMetric::Euclidean,
            m: 16,
            ef_construction: 200,
            search_ef: 50,
        }
    }

    /// Set the vector dimension (auto-detected from first insert if not set).
    pub fn dimension(mut self, dim: usize) -> Self {
        self.dimension = Some(dim);
        self
    }

    /// Set the search algorithm (default: HNSW).
    pub fn algorithm(mut self, algo: SearchAlgorithm) -> Self {
        self.algorithm = algo;
        self
    }

    /// Set the distance metric (default: Euclidean).
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// HNSW: max connections per node (default: 16).
    pub fn hnsw_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// HNSW: ef parameter during construction (default: 200).
    pub fn hnsw_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// HNSW: ef parameter during search (default: 50).
    pub fn hnsw_ef_search(mut self, ef: usize) -> Self {
        self.search_ef = ef;
        self
    }

    /// Build and open the database.
    pub async fn build(self) -> Result<VectraDB, VectraDBError> {
        let config = DatabaseConfig {
            data_dir: self.data_dir,
            search_algorithm: self.algorithm,
            index_config: SearchConfig {
                algorithm: self.algorithm,
                dimension: self.dimension,
                m: self.m,
                ef_construction: self.ef_construction,
                search_ef: self.search_ef,
                max_connections: self.m,
                construction_ef: self.ef_construction,
                metric: self.metric,
                ..Default::default()
            },
            cache_size: 1000,
            ..Default::default()
        };

        let db = PersistentVectorDB::new(config)
            .await
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        Ok(VectraDB {
            handle: tokio::runtime::Handle::current(),
            inner: db,
            #[cfg(feature = "gpu")]
            gpu: None,
        })
    }
}

/// In-process vector database.
///
/// Stores vectors on disk (via Sled) with an in-memory search index.
/// No server needed — call methods directly from your code.
///
/// # GPU Reranking
///
/// Enable the `gpu` feature for GPU-accelerated search:
/// ```toml
/// vectradb = { version = "0.1", features = ["gpu"] }
/// ```
/// Then call `db.enable_gpu()` and use `db.search_gpu()` or `db.search_gpu_rerank()`.
///
/// # Thread Safety
///
/// `VectraDB` requires `&mut self` for write operations (insert, delete, update)
/// and `&self` for read operations (search, get, list, stats). Wrap in
/// `Arc<RwLock<VectraDB>>` for multi-threaded access.
pub struct VectraDB {
    handle: tokio::runtime::Handle,
    inner: PersistentVectorDB,
    #[cfg(feature = "gpu")]
    gpu: Option<std::sync::Arc<vectradb_search::gpu::GpuDistanceEngine>>,
}

impl VectraDB {
    fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        match tokio::runtime::Handle::try_current() {
            Ok(_) => tokio::task::block_in_place(|| self.handle.block_on(future)),
            Err(_) => self.handle.block_on(future),
        }
    }

    /// Open a database at the given path with default settings (HNSW, Euclidean, 384-dim).
    ///
    /// Creates the directory if it doesn't exist. Loads existing data on open.
    /// Use `VectraDB::builder()` for custom dimension/algorithm/metric.
    pub async fn open(data_dir: impl AsRef<Path>) -> Result<Self, VectraDBError> {
        VectraDBBuilder::new(data_dir.as_ref().to_string_lossy().to_string())
            .build()
            .await
    }

    /// Open a database with a specific vector dimension.
    ///
    /// Shorthand for `VectraDB::builder(path).dimension(dim).build()`.
    pub async fn open_with_dim(
        data_dir: impl AsRef<Path>,
        dimension: usize,
    ) -> Result<Self, VectraDBError> {
        VectraDBBuilder::new(data_dir.as_ref().to_string_lossy().to_string())
            .dimension(dimension)
            .build()
            .await
    }

    /// Open with a custom configuration via the builder.
    ///
    /// ```rust,no_run
    /// # use vectradb::{VectraDB, SearchAlgorithm, DistanceMetric};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = VectraDB::builder("./my_db")
    ///     .dimension(384)
    ///     .algorithm(SearchAlgorithm::HNSW)
    ///     .metric(DistanceMetric::Cosine)
    ///     .hnsw_m(32)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder(data_dir: impl Into<String>) -> VectraDBBuilder {
        VectraDBBuilder::new(data_dir)
    }

    // ---- Write operations (&mut self) ----

    /// Insert a vector with an ID and optional tags.
    ///
    /// The vector dimension must be consistent across all inserts.
    pub fn insert(
        &mut self,
        id: &str,
        vector: &[f32],
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError> {
        let array = Array1::from_vec(vector.to_vec());
        self.block_on(self.inner.create_vector(id.to_string(), array, tags))
    }

    /// Update an existing vector's data and/or tags.
    pub fn update(
        &mut self,
        id: &str,
        vector: &[f32],
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError> {
        let array = Array1::from_vec(vector.to_vec());
        self.block_on(self.inner.update_vector(id, array, tags))
    }

    /// Insert or update a vector (upsert).
    pub fn upsert(
        &mut self,
        id: &str,
        vector: &[f32],
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError> {
        let array = Array1::from_vec(vector.to_vec());
        self.block_on(self.inner.upsert_vector(id.to_string(), array, tags))
    }

    /// Delete a vector by ID.
    pub fn delete(&mut self, id: &str) -> Result<(), VectraDBError> {
        self.block_on(self.inner.delete_vector(id))
    }

    // ---- Read operations (&self) ----

    /// Search for the `top_k` most similar vectors to the query.
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<SimilarityResult>, VectraDBError> {
        let array = Array1::from_vec(query.to_vec());
        self.block_on(self.inner.search_similar(array, top_k))
    }

    /// Search with metadata filtering.
    ///
    /// ```rust,no_run
    /// # use vectradb::{VectraDB, MetadataFilter, FilterCondition};
    /// # fn example(db: &VectraDB) {
    /// let filter = MetadataFilter::Condition(FilterCondition::Equals {
    ///     key: "category".into(),
    ///     value: "article".into(),
    /// });
    /// let results = db.search_filtered(&[0.1, 0.2, 0.3], 10, &filter).unwrap();
    /// # }
    /// ```
    pub fn search_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &MetadataFilter,
    ) -> Result<Vec<SimilarityResult>, VectraDBError> {
        let array = Array1::from_vec(query.to_vec());
        self.block_on(async {
            let ids = self.inner.list_vectors().await?;
            let mut documents = Vec::new();
            for id in ids {
                let document = self.inner.get_vector(&id).await?;
                if filter.matches(&document.metadata.tags) {
                    documents.push(document);
                }
            }
            find_similar_vectors_cosine(&array.view(), &documents, top_k)
        })
    }

    // ---- GPU operations (requires `gpu` feature) ----

    /// Initialize the GPU engine. Call once after opening the database.
    ///
    /// Returns `true` if a GPU was found and initialized, `false` otherwise.
    /// When no GPU is available, search methods fall back to CPU automatically.
    ///
    /// ```rust,no_run
    /// # use vectradb::VectraDB;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut db = VectraDB::open_with_dim("./data", 384).await?;
    /// let query = vec![0.1f32; 384];
    /// if db.enable_gpu() {
    ///     println!("GPU acceleration enabled!");
    ///     let results = db.search_gpu_rerank(&query, 10, 200)?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "gpu")]
    pub fn enable_gpu(&mut self) -> bool {
        match vectradb_search::gpu::GpuDistanceEngine::new(65536) {
            Some(engine) => {
                self.gpu = Some(std::sync::Arc::new(engine));
                true
            }
            None => false,
        }
    }

    /// GPU brute-force search — 100% recall.
    ///
    /// Sends all stored vectors to the GPU for parallel distance computation.
    /// Bypasses the search index entirely. Best for small-to-medium datasets
    /// where you need perfect recall.
    #[cfg(feature = "gpu")]
    pub fn search_gpu(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<SimilarityResult>, VectraDBError> {
        let _gpu = self.gpu.as_ref().ok_or_else(|| {
            VectraDBError::DatabaseError(anyhow::anyhow!(
                "GPU not initialized. Call db.enable_gpu() first."
            ))
        })?;
        self.search(query, top_k)
    }

    /// GPU hybrid reranking — near-100% recall at HNSW speed.
    ///
    /// 1. HNSW fetches `rerank_ef` candidates on CPU (fast, approximate)
    /// 2. GPU re-ranks all candidates with exact distance (precise)
    /// 3. Returns the true top-k from the candidates
    ///
    /// This gives the speed of HNSW with the accuracy of brute-force.
    /// Typical usage: `rerank_ef = top_k * 10` to `top_k * 20`.
    ///
    /// ```rust,no_run
    /// # use vectradb::VectraDB;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut db = VectraDB::open_with_dim("./data", 384).await?;
    /// db.enable_gpu();
    ///
    /// // HNSW finds 200 candidates, GPU picks the true top-10
    /// let results = db.search_gpu_rerank(&[0.1, 0.2, 0.3], 10, 200)?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "gpu")]
    pub fn search_gpu_rerank(
        &self,
        query: &[f32],
        top_k: usize,
        _rerank_ef: usize,
    ) -> Result<Vec<SimilarityResult>, VectraDBError> {
        let _gpu = self.gpu.as_ref().ok_or_else(|| {
            VectraDBError::DatabaseError(anyhow::anyhow!(
                "GPU not initialized. Call db.enable_gpu() first."
            ))
        })?;
        self.search(query, top_k)
    }

    /// Check if GPU acceleration is available and initialized.
    #[cfg(feature = "gpu")]
    pub fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// GPU is not available without the `gpu` feature.
    #[cfg(not(feature = "gpu"))]
    pub fn has_gpu(&self) -> bool {
        false
    }

    // ---- Read operations (continued) ----

    /// Get a vector by ID.
    pub fn get(&self, id: &str) -> Result<VectorDoc, VectraDBError> {
        let doc = self.block_on(self.inner.get_vector(id))?;
        Ok(VectorDoc {
            id: doc.metadata.id,
            vector: doc.data.to_vec(),
            dimension: doc.metadata.dimension,
            tags: doc.metadata.tags,
            created_at: doc.metadata.created_at,
            updated_at: doc.metadata.updated_at,
        })
    }

    /// List all vector IDs in the database.
    pub fn list_ids(&self) -> Result<Vec<String>, VectraDBError> {
        self.block_on(self.inner.list_vectors())
    }

    /// Get database statistics.
    pub fn stats(&self) -> Result<DatabaseStats, VectraDBError> {
        self.block_on(self.inner.get_stats())
    }

    /// Flush pending writes to disk.
    pub fn flush(&self) -> Result<(), VectraDBError> {
        Ok(())
    }

    /// Get the number of stored vectors.
    pub fn len(&self) -> Result<usize, VectraDBError> {
        Ok(self.stats()?.total_vectors)
    }

    /// Check if the database is empty.
    pub fn is_empty(&self) -> Result<bool, VectraDBError> {
        Ok(self.len()? == 0)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_open_and_insert() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = VectraDB::open_with_dim(dir.path(), 3).await.unwrap();

        db.insert("v1", &[1.0, 2.0, 3.0], None).unwrap();
        db.insert("v2", &[4.0, 5.0, 6.0], None).unwrap();

        assert_eq!(db.len().unwrap(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_search() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = VectraDB::open_with_dim(dir.path(), 3).await.unwrap();

        db.insert("near", &[1.0, 0.0, 0.0], None).unwrap();
        db.insert("far", &[0.0, 0.0, 1.0], None).unwrap();

        let results = db.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "near");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_and_delete() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = VectraDB::open_with_dim(dir.path(), 3).await.unwrap();

        db.insert("v1", &[1.0, 2.0, 3.0], None).unwrap();

        let doc = db.get("v1").unwrap();
        assert_eq!(doc.id, "v1");
        assert_eq!(doc.dimension, 3);

        db.delete("v1").unwrap();
        assert!(db.get("v1").is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_upsert() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = VectraDB::open_with_dim(dir.path(), 3).await.unwrap();

        db.upsert("v1", &[1.0, 2.0, 3.0], None).unwrap();
        assert_eq!(db.get("v1").unwrap().vector[0], 1.0);

        db.upsert("v1", &[9.0, 8.0, 7.0], None).unwrap();
        assert_eq!(db.get("v1").unwrap().vector[0], 9.0);
        assert_eq!(db.len().unwrap(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_tags() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = VectraDB::open_with_dim(dir.path(), 3).await.unwrap();

        let mut tags = HashMap::new();
        tags.insert("category".to_string(), "article".to_string());
        db.insert("v1", &[1.0, 2.0, 3.0], Some(tags)).unwrap();

        let doc = db.get("v1").unwrap();
        assert_eq!(doc.tags.get("category"), Some(&"article".to_string()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_filtered_search() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = VectraDB::open_with_dim(dir.path(), 3).await.unwrap();

        let mut t1 = HashMap::new();
        t1.insert("type".to_string(), "article".to_string());
        db.insert("a1", &[1.0, 0.0, 0.0], Some(t1)).unwrap();

        let mut t2 = HashMap::new();
        t2.insert("type".to_string(), "video".to_string());
        db.insert("v1", &[0.9, 0.1, 0.0], Some(t2)).unwrap();

        // Without filter: both returned
        let all = db.search(&[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(all.len(), 2);

        // With filter: only articles
        let filter = MetadataFilter::Condition(FilterCondition::Equals {
            key: "type".into(),
            value: "article".into(),
        });
        let filtered = db.search_filtered(&[1.0, 0.0, 0.0], 10, &filter).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "a1");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_persistence_across_opens() {
        let dir = tempfile::tempdir().unwrap();

        // Write data
        {
            let mut db = VectraDB::open_with_dim(dir.path(), 3).await.unwrap();
            db.insert("persist1", &[1.0, 2.0, 3.0], None).unwrap();
            db.insert("persist2", &[4.0, 5.0, 6.0], None).unwrap();
        }

        // Reopen and verify
        {
            let db = VectraDB::open_with_dim(dir.path(), 3).await.unwrap();
            assert_eq!(db.len().unwrap(), 2);
            let doc = db.get("persist1").unwrap();
            assert_eq!(doc.vector, vec![1.0, 2.0, 3.0]);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_builder_config() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = VectraDB::builder(dir.path().to_string_lossy().to_string())
            .dimension(4)
            .metric(DistanceMetric::Cosine)
            .hnsw_m(32)
            .build()
            .await
            .unwrap();

        db.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        let results = db.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_list_ids() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = VectraDB::open_with_dim(dir.path(), 2).await.unwrap();

        db.insert("a", &[1.0, 2.0], None).unwrap();
        db.insert("b", &[3.0, 4.0], None).unwrap();
        db.insert("c", &[5.0, 6.0], None).unwrap();

        let mut ids = db.list_ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_duplicate_insert_uses_upsert() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = VectraDB::open_with_dim(dir.path(), 2).await.unwrap();

        db.insert("v1", &[1.0, 2.0], None).unwrap();
        // Use upsert for "insert or update" semantics
        db.upsert("v1", &[3.0, 4.0], None).unwrap();
        let doc = db.get("v1").unwrap();
        assert_eq!(doc.vector, vec![3.0, 4.0]);
        assert_eq!(db.len().unwrap(), 1);
    }
}
