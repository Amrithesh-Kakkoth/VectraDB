#![allow(non_local_definitions)]

use ndarray::Array1;
use pyo3::prelude::*;
use std::collections::HashMap;
use vectradb_components::{DatabaseStats, SimilarityResult, VectorDocument, VectorInput};
use vectradb_search::SearchAlgorithm;
use vectradb_storage::{DatabaseConfig, PersistentVectorDB};

#[pyclass]
pub struct VectraDB {
    rt: tokio::runtime::Runtime,
    db: PersistentVectorDB,
}

#[pymethods]
impl VectraDB {
    #[new]
    #[pyo3(signature = (data_dir=None, search_algorithm=None, dimension=None, search_ef=None))]
    pub fn new(
        data_dir: Option<String>,
        search_algorithm: Option<String>,
        dimension: Option<usize>,
        search_ef: Option<usize>,
    ) -> PyResult<Self> {
        let mut index_config = vectradb_search::SearchConfig::default();
        if let Some(dim) = dimension {
            index_config.dimension = Some(dim);
        }
        if let Some(ef) = search_ef {
            index_config.search_ef = ef;
        }

        let config = DatabaseConfig {
            data_dir: data_dir.unwrap_or_else(|| "./vectradb_data".to_string()),
            search_algorithm: match search_algorithm
                .unwrap_or_else(|| "hnsw".to_string())
                .as_str()
            {
                "hnsw" => SearchAlgorithm::HNSW,
                "lsh" => SearchAlgorithm::LSH,
                "pq" => SearchAlgorithm::PQ,
                "es4d" => SearchAlgorithm::ES4D,
                _ => SearchAlgorithm::HNSW,
            },
            index_config,
            ..Default::default()
        };

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let db = rt
            .block_on(PersistentVectorDB::new(config))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { rt, db })
    }

    pub fn create_vector(
        &mut self,
        id: String,
        vector: Vec<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> PyResult<()> {
        let array_vector = Array1::from_vec(vector);
        self.rt
            .block_on(self.db.create_vector(id, array_vector, tags))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }

    pub fn get_vector(&self, id: &str) -> PyResult<PyVectorDocument> {
        let document = self
            .rt
            .block_on(self.db.get_vector(id))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyKeyError, _>(e.to_string()))?;
        Ok(PyVectorDocument::from(document))
    }

    pub fn update_vector(
        &mut self,
        id: &str,
        vector: Vec<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> PyResult<()> {
        let array_vector = Array1::from_vec(vector);
        self.rt
            .block_on(self.db.update_vector(id, array_vector, tags))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }

    pub fn delete_vector(&mut self, id: &str) -> PyResult<()> {
        self.rt
            .block_on(self.db.delete_vector(id))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyKeyError, _>(e.to_string()))?;
        Ok(())
    }

    pub fn upsert_vector(
        &mut self,
        id: String,
        vector: Vec<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> PyResult<()> {
        let array_vector = Array1::from_vec(vector);
        self.rt
            .block_on(self.db.upsert_vector(id, array_vector, tags))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }

    pub fn search_similar(
        &self,
        py: Python<'_>,
        query_vector: Vec<f32>,
        top_k: Option<usize>,
    ) -> PyResult<Vec<PySimilarityResult>> {
        let array_query = Array1::from_vec(query_vector);
        let k = top_k.unwrap_or(10);
        let results = py
            .allow_threads(|| self.rt.block_on(self.db.search_similar(array_query, k)))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(results.into_iter().map(PySimilarityResult::from).collect())
    }

    pub fn batch_create(
        &mut self,
        py: Python<'_>,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        tags: Option<Vec<Option<HashMap<String, String>>>>,
    ) -> PyResult<(usize, usize)> {
        let items: Vec<VectorInput> = ids
            .into_iter()
            .zip(vectors)
            .enumerate()
            .map(|(index, (id, vector))| VectorInput {
                id,
                vector,
                tags: tags
                    .as_ref()
                    .and_then(|all| all.get(index).cloned())
                    .flatten()
                    .unwrap_or_default(),
            })
            .collect();

        let total = items.len();
        let response = py
            .allow_threads(|| self.rt.block_on(self.db.batch_create_vectors(items)))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let inserted = response
            .statuses
            .iter()
            .filter(|status| status.is_ok())
            .count();
        Ok((inserted, total))
    }

    #[cfg(feature = "gpu")]
    #[pyo3(signature = (query_vector, top_k=None, rerank_ef=None))]
    pub fn search_gpu(
        &self,
        query_vector: Vec<f32>,
        top_k: Option<usize>,
        rerank_ef: Option<usize>,
    ) -> PyResult<Vec<PySimilarityResult>> {
        let _ = rerank_ef;
        let results = self
            .rt
            .block_on(
                self.db
                    .search_similar(Array1::from_vec(query_vector), top_k.unwrap_or(10)),
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(results.into_iter().map(PySimilarityResult::from).collect())
    }

    pub fn has_gpu(&self) -> bool {
        false
    }

    pub fn list_vectors(&self) -> PyResult<Vec<String>> {
        self.rt
            .block_on(self.db.list_vectors())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    pub fn get_stats(&self) -> PyResult<PyDatabaseStats> {
        let stats = self
            .rt
            .block_on(self.db.get_stats())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyDatabaseStats::from(stats))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyVectorDocument {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub vector: Vec<f32>,
    #[pyo3(get)]
    pub dimension: usize,
    #[pyo3(get)]
    pub created_at: u64,
    #[pyo3(get)]
    pub updated_at: u64,
    #[pyo3(get)]
    pub tags: HashMap<String, String>,
}

impl From<VectorDocument> for PyVectorDocument {
    fn from(doc: VectorDocument) -> Self {
        Self {
            id: doc.metadata.id,
            vector: doc.data.to_vec(),
            dimension: doc.metadata.dimension,
            created_at: doc.metadata.created_at,
            updated_at: doc.metadata.updated_at,
            tags: doc.metadata.tags,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PySimilarityResult {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub tags: HashMap<String, String>,
}

impl From<SimilarityResult> for PySimilarityResult {
    fn from(result: SimilarityResult) -> Self {
        Self {
            id: result.id,
            score: result.score,
            tags: result.metadata.tags,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyDatabaseStats {
    #[pyo3(get)]
    pub total_vectors: usize,
    #[pyo3(get)]
    pub dimension: usize,
    #[pyo3(get)]
    pub memory_usage: u64,
}

impl From<DatabaseStats> for PyDatabaseStats {
    fn from(stats: DatabaseStats) -> Self {
        Self {
            total_vectors: stats.total_vectors,
            dimension: stats.dimension,
            memory_usage: stats.memory_usage,
        }
    }
}

#[pymodule]
fn vectradb_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VectraDB>()?;
    m.add_class::<PyVectorDocument>()?;
    m.add_class::<PySimilarityResult>()?;
    m.add_class::<PyDatabaseStats>()?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_vector_document_conversion() {
        let doc = VectorDocument {
            metadata: vectradb_components::VectorMetadata {
                id: "test".to_string(),
                dimension: 3,
                created_at: 1_234_567_890,
                updated_at: 1_234_567_890,
                tags: HashMap::new(),
            },
            data: Array1::from_vec(vec![1.0, 2.0, 3.0]),
        };

        let py_doc = PyVectorDocument::from(doc);
        assert_eq!(py_doc.id, "test");
        assert_eq!(py_doc.dimension, 3);
        assert_eq!(py_doc.vector, vec![1.0, 2.0, 3.0]);
    }
}
