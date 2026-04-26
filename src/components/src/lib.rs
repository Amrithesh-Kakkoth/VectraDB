use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// Re-export commonly used types
pub use ndarray::{Array1, ArrayView1};

/// Vector database error types
#[derive(Error, Debug)]
pub enum VectraDBError {
    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Vector already exists: {id}")]
    VectorAlreadyExists { id: String },
    #[error("Vector not found: {id}")]
    VectorNotFound { id: String },
    #[error("Vector already exists: {id}")]
    DuplicateVector { id: String },
    #[error("Invalid vector data")]
    InvalidVector,
    #[error("Database error: {0}")]
    DatabaseError(#[from] anyhow::Error),
}

/// Vector metadata structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    pub id: String,
    pub dimension: usize,
    pub created_at: u64,
    pub updated_at: u64,
    pub tags: HashMap<String, String>,
}

/// Vector document structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDocument {
    pub metadata: VectorMetadata,
    pub data: Array1<f32>,
}

/// Vector similarity result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    pub id: String,
    pub score: f32,
    pub metadata: VectorMetadata,
}

/// Shared vector input shape for batch APIs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorInput {
    pub id: String,
    pub vector: Vec<f32>,
    pub tags: HashMap<String, String>,
}

/// Per-item batch write status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchWriteItemStatus {
    pub id: String,
    pub code: String,
    pub message: String,
}

impl BatchWriteItemStatus {
    pub fn ok(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            code: "OK".to_string(),
            message: String::new(),
        }
    }

    pub fn error(
        id: impl Into<String>,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            code: code.into(),
            message: message.into(),
        }
    }

    pub fn is_ok(&self) -> bool {
        self.code == "OK"
    }
}

/// Order-preserving batch write response
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BatchWriteResponse {
    pub statuses: Vec<BatchWriteItemStatus>,
}

/// Vector database trait for different implementations
pub trait VectorDatabase {
    /// Create a new vector in the database
    fn create_vector(
        &mut self,
        id: String,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError>;

    /// Fetch a vector by ID
    fn get_vector(&self, id: &str) -> Result<VectorDocument, VectraDBError>;

    /// Update an existing vector
    fn update_vector(
        &mut self,
        id: &str,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError>;

    /// Delete a vector by ID
    fn delete_vector(&mut self, id: &str) -> Result<(), VectraDBError>;

    /// Upsert (insert or update) a vector
    fn upsert_vector(
        &mut self,
        id: String,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError>;

    /// Search for similar vectors
    fn search_similar(
        &self,
        query_vector: Array1<f32>,
        top_k: usize,
    ) -> Result<Vec<SimilarityResult>, VectraDBError>;

    /// Get all vector IDs
    fn list_vectors(&self) -> Result<Vec<String>, VectraDBError>;

    /// Get database statistics
    fn get_stats(&self) -> Result<DatabaseStats, VectraDBError>;
}

/// Database statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatabaseStats {
    pub total_vectors: usize,
    pub dimension: usize,
    pub memory_usage: u64,
}

// Module declarations
pub mod filter;
pub mod indexing;
pub mod similarity;
pub mod storage;
pub mod tensor;
pub mod vector_operations;

#[cfg(feature = "gpu")]
pub mod gpu;

// Re-export main functionality
pub use similarity::*;
pub use vector_operations::*;
