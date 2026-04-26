use crate::{VectorDocument, VectorMetadata, VectraDBError};
use ndarray::Array1;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Create a new vector document
pub fn create_vector_document(
    id: String,
    vector: Array1<f32>,
    tags: Option<HashMap<String, String>>,
) -> Result<VectorDocument, VectraDBError> {
    validate_vector(&vector)?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let metadata = VectorMetadata {
        id: id.clone(),
        dimension: vector.len(),
        created_at: now,
        updated_at: now,
        tags: tags.unwrap_or_default(),
    };

    Ok(VectorDocument {
        metadata,
        data: vector,
    })
}

/// Update vector document with new data
pub fn update_vector_document(
    mut doc: VectorDocument,
    vector: Array1<f32>,
    tags: Option<HashMap<String, String>>,
) -> Result<VectorDocument, VectraDBError> {
    // Validate dimension consistency
    if doc.metadata.dimension != vector.len() {
        return Err(VectraDBError::DimensionMismatch {
            expected: doc.metadata.dimension,
            actual: vector.len(),
        });
    }

    validate_vector(&vector)?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Update metadata
    doc.metadata.updated_at = now;
    if let Some(new_tags) = tags {
        doc.metadata.tags = new_tags;
    }

    // Update vector data
    doc.data = vector;

    Ok(doc)
}

/// Validate vector data
pub fn validate_vector(vector: &Array1<f32>) -> Result<(), VectraDBError> {
    if vector.is_empty() {
        return Err(VectraDBError::InvalidVector);
    }

    // Check for NaN or infinite values
    for &value in vector.iter() {
        if !value.is_finite() {
            return Err(VectraDBError::InvalidVector);
        }
    }

    Ok(())
}

/// Normalize vector to unit length
pub fn normalize_vector(mut vector: Array1<f32>) -> Result<Array1<f32>, VectraDBError> {
    validate_vector(&vector)?;

    let norm = vector.dot(&vector).sqrt();
    if norm == 0.0 {
        return Err(VectraDBError::InvalidVector);
    }

    vector /= norm;
    Ok(vector)
}

/// Create a zero vector of specified dimension
pub fn create_zero_vector(dimension: usize) -> Array1<f32> {
    Array1::zeros(dimension)
}

/// Create a random vector of specified dimension (for testing)
#[cfg(test)]
pub fn create_random_vector(dimension: usize) -> Array1<f32> {
    use ndarray::Array;
    use rand::Rng;

    let mut rng = rand::thread_rng();
    Array::from_iter((0..dimension).map(|_| rng.gen::<f32>()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_create_vector_document() {
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut tags = HashMap::new();
        tags.insert("category".to_string(), "test".to_string());

        let doc =
            create_vector_document("test_id".to_string(), vector.clone(), Some(tags)).unwrap();

        assert_eq!(doc.metadata.id, "test_id");
        assert_eq!(doc.metadata.dimension, 3);
        assert_eq!(doc.data, vector);
        assert_eq!(doc.metadata.tags.get("category"), Some(&"test".to_string()));
    }

    #[test]
    fn test_validate_vector() {
        let valid_vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(validate_vector(&valid_vector).is_ok());

        let invalid_vector = Array1::from_vec(vec![f32::NAN, 2.0, 3.0]);
        assert!(validate_vector(&invalid_vector).is_err());

        let empty_vector = Array1::zeros(0);
        assert!(validate_vector(&empty_vector).is_err());
    }

    #[test]
    fn test_normalize_vector() {
        let vector = Array1::from_vec(vec![3.0, 4.0]);
        let normalized = normalize_vector(vector).unwrap();

        let expected_norm = (normalized.dot(&normalized)).sqrt();
        assert!((expected_norm - 1.0).abs() < 1e-6);
    }
}
