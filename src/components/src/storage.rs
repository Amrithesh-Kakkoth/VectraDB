use crate::{DatabaseStats, VectorDatabase, VectorDocument, VectraDBError};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

/// In-memory storage implementation for the vector database
pub struct InMemoryVectorDB {
    vectors: RwLock<HashMap<String, VectorDocument>>,
    dimension: Option<usize>,
}

impl InMemoryVectorDB {
    /// Create a new in-memory vector database
    pub fn new() -> Self {
        Self {
            vectors: RwLock::new(HashMap::new()),
            dimension: None,
        }
    }

    /// Create a new in-memory vector database with fixed dimension
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            vectors: RwLock::new(HashMap::new()),
            dimension: Some(dimension),
        }
    }

    /// Get current memory usage (approximate)
    fn calculate_memory_usage(&self) -> u64 {
        let vectors = self.vectors.read().unwrap_or_else(|e| e.into_inner());
        let mut total_size = 0;

        for (id, doc) in vectors.iter() {
            total_size += id.len() + doc.data.len() * 4; // 4 bytes per f32
            total_size += doc.metadata.tags.len() * 16; // Rough estimate for tags
        }

        total_size as u64
    }
}

impl Default for InMemoryVectorDB {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorDatabase for InMemoryVectorDB {
    fn create_vector(
        &mut self,
        id: String,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError> {
        // Check dimension consistency if dimension is fixed
        if let Some(expected_dim) = self.dimension {
            if vector.len() != expected_dim {
                return Err(VectraDBError::DimensionMismatch {
                    expected: expected_dim,
                    actual: vector.len(),
                });
            }
        } else {
            // Set dimension on first vector
            self.dimension = Some(vector.len());
        }

        let doc = crate::vector_operations::create_vector_document(id.clone(), vector, tags)?;

        let mut vectors = self.vectors.write().unwrap_or_else(|e| e.into_inner());
        if vectors.contains_key(&id) {
            return Err(VectraDBError::VectorAlreadyExists { id });
        }

        vectors.insert(id, doc);
        Ok(())
    }

    fn get_vector(&self, id: &str) -> Result<VectorDocument, VectraDBError> {
        let vectors = self.vectors.read().unwrap_or_else(|e| e.into_inner());
        vectors
            .get(id)
            .cloned()
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })
    }

    fn update_vector(
        &mut self,
        id: &str,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError> {
        // Check dimension consistency
        if let Some(expected_dim) = self.dimension {
            if vector.len() != expected_dim {
                return Err(VectraDBError::DimensionMismatch {
                    expected: expected_dim,
                    actual: vector.len(),
                });
            }
        }

        let mut vectors = self.vectors.write().unwrap_or_else(|e| e.into_inner());
        let doc = vectors
            .get_mut(id)
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })?;

        let updated_doc =
            crate::vector_operations::update_vector_document(doc.clone(), vector, tags)?;
        *doc = updated_doc;
        Ok(())
    }

    fn delete_vector(&mut self, id: &str) -> Result<(), VectraDBError> {
        let mut vectors = self.vectors.write().unwrap_or_else(|e| e.into_inner());
        vectors
            .remove(id)
            .map(|_| ())
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })
    }

    fn upsert_vector(
        &mut self,
        id: String,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError> {
        // Check dimension consistency
        if let Some(expected_dim) = self.dimension {
            if vector.len() != expected_dim {
                return Err(VectraDBError::DimensionMismatch {
                    expected: expected_dim,
                    actual: vector.len(),
                });
            }
        } else {
            // Set dimension on first vector
            self.dimension = Some(vector.len());
        }

        let mut vectors = self.vectors.write().unwrap_or_else(|e| e.into_inner());

        use std::collections::hash_map::Entry;
        match vectors.entry(id.clone()) {
            Entry::Occupied(mut entry) => {
                // Update existing vector
                let updated_doc = crate::vector_operations::update_vector_document(
                    entry.get().clone(),
                    vector,
                    tags,
                )?;
                *entry.get_mut() = updated_doc;
            }
            Entry::Vacant(entry) => {
                // Create new vector
                let doc = crate::vector_operations::create_vector_document(id, vector, tags)?;
                entry.insert(doc);
            }
        }

        Ok(())
    }

    fn search_similar(
        &self,
        query_vector: Array1<f32>,
        top_k: usize,
    ) -> Result<Vec<crate::SimilarityResult>, VectraDBError> {
        let vectors = self.vectors.read().unwrap_or_else(|e| e.into_inner());
        let documents: Vec<VectorDocument> = vectors.values().cloned().collect();

        crate::similarity::find_similar_vectors_cosine(&query_vector.view(), &documents, top_k)
    }

    fn list_vectors(&self) -> Result<Vec<String>, VectraDBError> {
        let vectors = self.vectors.read().unwrap_or_else(|e| e.into_inner());
        Ok(vectors.keys().cloned().collect())
    }

    fn get_stats(&self) -> Result<DatabaseStats, VectraDBError> {
        let vectors = self.vectors.read().unwrap_or_else(|e| e.into_inner());
        let total_vectors = vectors.len();
        let dimension = self.dimension.unwrap_or(0);
        let memory_usage = self.calculate_memory_usage();

        Ok(DatabaseStats {
            total_vectors,
            dimension,
            memory_usage,
        })
    }
}

/// Persistent storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub data_dir: String,
    pub max_file_size: u64,
    pub compression_enabled: bool,
    pub cache_size: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: "./data".to_string(),
            max_file_size: 1024 * 1024 * 1024, // 1GB
            compression_enabled: true,
            cache_size: 1000,
        }
    }
}

/// Storage trait for different storage backends
pub trait StorageBackend {
    fn save_vector(&mut self, id: &str, document: &VectorDocument) -> Result<(), VectraDBError>;
    fn load_vector(&self, id: &str) -> Result<VectorDocument, VectraDBError>;
    fn delete_vector(&mut self, id: &str) -> Result<(), VectraDBError>;
    fn list_vector_ids(&self) -> Result<Vec<String>, VectraDBError>;
    fn exists(&self, id: &str) -> Result<bool, VectraDBError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_db_creation() {
        let mut db = InMemoryVectorDB::new();
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        assert!(db
            .create_vector("test_id".to_string(), vector, None)
            .is_ok());
        assert!(db.get_vector("test_id").is_ok());
    }

    #[test]
    fn test_in_memory_db_dimension_check() {
        let mut db = InMemoryVectorDB::with_dimension(3);
        let vector = Array1::from_vec(vec![1.0, 2.0]); // Wrong dimension

        assert!(db
            .create_vector("test_id".to_string(), vector, None)
            .is_err());
    }

    #[test]
    fn test_in_memory_db_upsert() {
        let mut db = InMemoryVectorDB::new();
        let vector1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let vector2 = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        // First upsert should create
        assert!(db
            .upsert_vector("test_id".to_string(), vector1, None)
            .is_ok());

        // Second upsert should update
        assert!(db
            .upsert_vector("test_id".to_string(), vector2, None)
            .is_ok());

        let doc = db.get_vector("test_id").unwrap();
        assert_eq!(doc.data[0], 4.0);
    }

    #[test]
    fn test_in_memory_db_stats() {
        let mut db = InMemoryVectorDB::new();
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        db.create_vector("test_id".to_string(), vector, None)
            .unwrap();
        let stats = db.get_stats().unwrap();

        assert_eq!(stats.total_vectors, 1);
        assert_eq!(stats.dimension, 3);
        assert!(stats.memory_usage > 0);
    }
}
