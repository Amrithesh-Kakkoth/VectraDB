use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, oneshot};
use vectradb_components::{
    filter::MetadataFilter,
    vector_operations::{create_vector_document, update_vector_document, validate_vector},
    BatchWriteItemStatus, BatchWriteResponse, DatabaseStats, SimilarityResult, VectorDocument,
    VectorInput, VectorMetadata, VectraDBError,
};
use vectradb_search::{AdvancedSearch, DistanceMetric, HNSWIndex, SearchAlgorithm, SearchConfig};

const STORAGE_VERSION: u32 = 2;
const MANIFEST_FILE: &str = "MANIFEST.json";
const WAL_FILE: &str = "WAL.log";
const SEGMENTS_DIR: &str = "segments";
const SEGMENT_DATA_FILE: &str = "segment.bin";
const INDEX_META_FILE: &str = "index-meta.json";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DurabilityMode {
    Batch,
    Async,
    Strict,
}

impl Default for DurabilityMode {
    fn default() -> Self {
        Self::Batch
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub data_dir: String,
    pub search_algorithm: SearchAlgorithm,
    pub index_config: SearchConfig,
    pub cache_size: usize,
    pub durability_mode: DurabilityMode,
    pub commit_interval_ms: u64,
    pub commit_max_vectors: usize,
    pub commit_max_bytes: usize,
    pub segment_max_vectors: usize,
    pub segment_max_bytes: usize,
    pub segment_max_age_secs: u64,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            data_dir: "./vectradb_data".to_string(),
            search_algorithm: SearchAlgorithm::HNSW,
            index_config: SearchConfig::default(),
            cache_size: 1000,
            durability_mode: DurabilityMode::Batch,
            commit_interval_ms: 10,
            commit_max_vectors: 2000,
            commit_max_bytes: 4 * 1024 * 1024,
            segment_max_vectors: 50_000,
            segment_max_bytes: 64 * 1024 * 1024,
            segment_max_age_secs: 30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MigrationStats {
    pub migrated_vectors: usize,
    pub sealed_segments: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    search_algorithm: SearchAlgorithm,
    dimension: Option<usize>,
    compacted_seq: u64,
    current_mutable_segment_id: u64,
    next_segment_id: u64,
    sealed_segments: Vec<SealedSegmentMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SealedSegmentMeta {
    id: u64,
    doc_count: usize,
    tombstone_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SegmentDiskData {
    id: u64,
    documents: Vec<VectorDocument>,
    tombstones: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexMeta {
    algorithm: String,
    built_at: u64,
    doc_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WalBatch {
    records: Vec<WalRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WalRecord {
    seq: u64,
    operation: WalOperation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum WalOperation {
    Upsert { document: VectorDocument },
    Delete { id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RecordLocation {
    Mutable { segment_id: u64 },
    Sealed { segment_id: u64 },
}

#[derive(Debug, Clone)]
enum OverlayEntry {
    Upsert(VectorDocument),
    Delete,
}

#[derive(Debug, Clone, Copy)]
enum WriteMode {
    Create,
    Update,
    Upsert,
    Delete,
}

#[derive(Debug, Clone)]
struct WriteItem {
    id: String,
    vector: Option<Array1<f32>>,
    tags: Option<HashMap<String, String>>,
}

struct PendingRequest {
    mode: WriteMode,
    items: Vec<WriteItem>,
    estimated_bytes: usize,
    responder: oneshot::Sender<BatchWriteResponse>,
}

struct EvaluatedBatch {
    responses: Vec<BatchWriteResponse>,
    wal_records: Vec<WalRecord>,
    next_seq: u64,
    dimension: Option<usize>,
}

struct MutableSegment {
    id: u64,
    created_at: u64,
    documents: HashMap<String, VectorDocument>,
    tombstones: HashSet<String>,
    raw_vector_bytes: usize,
    max_seq: u64,
}

impl MutableSegment {
    fn new(id: u64, created_at: u64) -> Self {
        Self {
            id,
            created_at,
            documents: HashMap::new(),
            tombstones: HashSet::new(),
            raw_vector_bytes: 0,
            max_seq: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.documents.is_empty() && self.tombstones.is_empty()
    }

    fn should_seal(&self, config: &DatabaseConfig) -> bool {
        if self.is_empty() {
            return false;
        }

        if self.documents.len() >= config.segment_max_vectors {
            return true;
        }

        if self.raw_vector_bytes >= config.segment_max_bytes {
            return true;
        }

        now_secs().saturating_sub(self.created_at) >= config.segment_max_age_secs
    }

    fn apply_upsert(&mut self, document: VectorDocument, seq: u64) {
        if let Some(previous) = self
            .documents
            .insert(document.metadata.id.clone(), document.clone())
        {
            self.raw_vector_bytes = self
                .raw_vector_bytes
                .saturating_sub(previous.data.len().saturating_mul(4));
        }
        self.raw_vector_bytes = self
            .raw_vector_bytes
            .saturating_add(document.data.len().saturating_mul(4));
        self.tombstones.remove(&document.metadata.id);
        self.max_seq = self.max_seq.max(seq);
    }

    fn apply_delete(&mut self, id: &str, seq: u64) {
        if let Some(previous) = self.documents.remove(id) {
            self.raw_vector_bytes = self
                .raw_vector_bytes
                .saturating_sub(previous.data.len().saturating_mul(4));
        }
        self.tombstones.insert(id.to_string());
        self.max_seq = self.max_seq.max(seq);
    }
}

struct SealedSegment {
    id: u64,
    docs: Vec<VectorDocument>,
    doc_index: HashMap<String, usize>,
    path: PathBuf,
    index: RwLock<Option<HNSWIndex>>,
}

impl SealedSegment {
    fn from_disk_data(path: PathBuf, data: SegmentDiskData) -> Self {
        let doc_index = data
            .documents
            .iter()
            .enumerate()
            .map(|(idx, doc)| (doc.metadata.id.clone(), idx))
            .collect();

        Self {
            id: data.id,
            docs: data.documents,
            doc_index,
            path,
            index: RwLock::new(None),
        }
    }

    fn get_document(&self, id: &str) -> Option<VectorDocument> {
        self.doc_index
            .get(id)
            .and_then(|idx| self.docs.get(*idx))
            .cloned()
    }

    fn search(
        &self,
        query: &Array1<f32>,
        top_k: usize,
    ) -> Result<Vec<SearchCandidate>, VectraDBError> {
        if self.docs.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(index) = self.index.read().unwrap().as_ref() {
            let indexed = index.search(query, top_k.saturating_mul(4).max(top_k))?;
            return Ok(indexed
                .into_iter()
                .filter_map(|result| {
                    self.get_document(&result.id)
                        .map(|document| SearchCandidate {
                            location: RecordLocation::Sealed {
                                segment_id: self.id,
                            },
                            result: SimilarityResult {
                                id: result.id,
                                score: result.similarity,
                                metadata: document.metadata,
                            },
                        })
                })
                .collect());
        }

        exact_search_candidates(
            &self.docs,
            query,
            top_k.saturating_mul(4).max(top_k),
            RecordLocation::Sealed {
                segment_id: self.id,
            },
        )
    }
}

struct SearchCandidate {
    location: RecordLocation,
    result: SimilarityResult,
}

struct EngineState {
    manifest: Manifest,
    mutable_segment: MutableSegment,
    id_map: HashMap<String, RecordLocation>,
    next_seq: u64,
}

struct StorageCore {
    config: DatabaseConfig,
    manifest_path: PathBuf,
    wal_path: PathBuf,
    segments_dir: PathBuf,
    state: RwLock<EngineState>,
    sealed_segments: RwLock<BTreeMap<u64, Arc<SealedSegment>>>,
}

#[derive(Clone)]
pub struct PersistentVectorDB {
    core: Arc<StorageCore>,
    write_tx: mpsc::Sender<PendingRequest>,
}

impl PersistentVectorDB {
    pub fn config(&self) -> &DatabaseConfig {
        &self.core.config
    }

    pub async fn new(config: DatabaseConfig) -> Result<Self, VectraDBError> {
        if config.search_algorithm != SearchAlgorithm::HNSW {
            return Err(VectraDBError::DatabaseError(anyhow::anyhow!(
                "The v2 ingest engine currently supports only HNSW"
            )));
        }

        let root = PathBuf::from(&config.data_dir);
        fs::create_dir_all(&root).map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        let manifest_path = root.join(MANIFEST_FILE);
        let wal_path = root.join(WAL_FILE);
        let segments_dir = root.join(SEGMENTS_DIR);

        let manifest = if manifest_path.exists() {
            load_manifest(&manifest_path)?
        } else {
            if dir_has_entries(&root)? {
                return Err(VectraDBError::DatabaseError(anyhow::anyhow!(
                    "Legacy or mixed storage layout detected in {}. Run `vectradb-server migrate` into a clean destination.",
                    root.display()
                )));
            }

            fs::create_dir_all(&segments_dir)
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
            File::create(&wal_path)
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

            let manifest = Manifest {
                version: STORAGE_VERSION,
                search_algorithm: config.search_algorithm,
                dimension: config.index_config.dimension,
                compacted_seq: 0,
                current_mutable_segment_id: 1,
                next_segment_id: 2,
                sealed_segments: Vec::new(),
            };
            write_manifest(&manifest_path, &manifest)?;
            manifest
        };

        if manifest.version != STORAGE_VERSION {
            return Err(VectraDBError::DatabaseError(anyhow::anyhow!(
                "Unsupported storage version {}",
                manifest.version
            )));
        }

        fs::create_dir_all(&segments_dir)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
        if !wal_path.exists() {
            File::create(&wal_path)
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
        }

        let mut id_map = HashMap::new();
        let mut sealed_segments = BTreeMap::new();
        let mut dimension = manifest.dimension;

        for meta in &manifest.sealed_segments {
            let segment_dir = segments_dir.join(format!("{:020}", meta.id));
            let disk = load_segment(&segment_dir)?;
            for document in &disk.documents {
                dimension = dimension.or(Some(document.metadata.dimension));
                id_map.insert(
                    document.metadata.id.clone(),
                    RecordLocation::Sealed {
                        segment_id: meta.id,
                    },
                );
            }
            for tombstone in &disk.tombstones {
                id_map.remove(tombstone);
            }
            sealed_segments.insert(
                meta.id,
                Arc::new(SealedSegment::from_disk_data(segment_dir, disk)),
            );
        }

        let mut mutable_segment =
            MutableSegment::new(manifest.current_mutable_segment_id, now_secs());
        let mut max_seq = manifest.compacted_seq;

        for record in read_wal_records(&wal_path)? {
            if record.seq <= manifest.compacted_seq {
                continue;
            }
            max_seq = max_seq.max(record.seq);
            match record.operation {
                WalOperation::Upsert { document } => {
                    dimension = dimension.or(Some(document.metadata.dimension));
                    mutable_segment.apply_upsert(document.clone(), record.seq);
                    id_map.insert(
                        document.metadata.id.clone(),
                        RecordLocation::Mutable {
                            segment_id: mutable_segment.id,
                        },
                    );
                }
                WalOperation::Delete { id } => {
                    mutable_segment.apply_delete(&id, record.seq);
                    id_map.remove(&id);
                }
            }
        }

        let state = EngineState {
            manifest: Manifest {
                dimension,
                ..manifest
            },
            mutable_segment,
            id_map,
            next_seq: max_seq.saturating_add(1),
        };

        let core = Arc::new(StorageCore {
            config,
            manifest_path,
            wal_path,
            segments_dir,
            state: RwLock::new(state),
            sealed_segments: RwLock::new(sealed_segments),
        });

        let (write_tx, write_rx) = mpsc::channel(256);
        let db = Self {
            core: core.clone(),
            write_tx,
        };

        tokio::spawn(writer_loop(core.clone(), write_rx));

        let existing_segments: Vec<Arc<SealedSegment>> = core
            .sealed_segments
            .read()
            .unwrap()
            .values()
            .cloned()
            .collect();
        for segment in existing_segments {
            schedule_segment_index(core.clone(), segment);
        }

        Ok(db)
    }

    pub async fn migrate_legacy_data(
        legacy_dir: impl AsRef<Path>,
        destination_dir: impl AsRef<Path>,
        mut config: DatabaseConfig,
    ) -> Result<MigrationStats, VectraDBError> {
        let legacy = legacy_dir.as_ref();
        let destination = destination_dir.as_ref();

        let legacy_db =
            sled::open(legacy).map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
        let vectors_tree = legacy_db
            .open_tree("vectors")
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
        let metadata_tree = legacy_db
            .open_tree("metadata")
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        let mut batch = Vec::new();
        let mut migrated = 0usize;

        config.data_dir = destination.to_string_lossy().to_string();
        let db = PersistentVectorDB::new(config).await?;

        for item in vectors_tree.iter() {
            let (id_bytes, vector_bytes) =
                item.map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
            let id = String::from_utf8(id_bytes.to_vec())
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
            let metadata_bytes = metadata_tree
                .get(id.as_bytes())
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?
                .ok_or_else(|| VectraDBError::VectorNotFound { id: id.clone() })?;
            let metadata: VectorMetadata = bincode::deserialize(&metadata_bytes)
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
            let data: Array1<f32> = bincode::deserialize(&vector_bytes)
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

            batch.push(VectorInput {
                id,
                vector: data.to_vec(),
                tags: metadata.tags,
            });

            if batch.len() >= 1000 {
                db.batch_upsert_vectors(batch.clone()).await?;
                migrated += batch.len();
                batch.clear();
            }
        }

        if !batch.is_empty() {
            db.batch_upsert_vectors(batch.clone()).await?;
            migrated += batch.len();
        }

        let stats = db.get_stats().await?;

        Ok(MigrationStats {
            migrated_vectors: migrated,
            sealed_segments: stats.total_vectors,
        })
    }

    pub async fn create_vector(
        &self,
        id: String,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError> {
        let response = self
            .send_write(
                WriteMode::Create,
                vec![WriteItem {
                    id: id.clone(),
                    vector: Some(vector),
                    tags,
                }],
            )
            .await?;
        status_to_result(response.statuses.into_iter().next())
    }

    pub async fn get_vector(&self, id: &str) -> Result<VectorDocument, VectraDBError> {
        let location = {
            let state = self.core.state.read().unwrap();
            state
                .id_map
                .get(id)
                .cloned()
                .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })?
        };

        match location {
            RecordLocation::Mutable { .. } => {
                let state = self.core.state.read().unwrap();
                state
                    .mutable_segment
                    .documents
                    .get(id)
                    .cloned()
                    .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })
            }
            RecordLocation::Sealed { segment_id } => {
                let segments = self.core.sealed_segments.read().unwrap();
                segments
                    .get(&segment_id)
                    .and_then(|segment| segment.get_document(id))
                    .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })
            }
        }
    }

    pub async fn update_vector(
        &self,
        id: &str,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError> {
        let response = self
            .send_write(
                WriteMode::Update,
                vec![WriteItem {
                    id: id.to_string(),
                    vector: Some(vector),
                    tags,
                }],
            )
            .await?;
        status_to_result(response.statuses.into_iter().next())
    }

    pub async fn delete_vector(&self, id: &str) -> Result<(), VectraDBError> {
        let response = self
            .send_write(
                WriteMode::Delete,
                vec![WriteItem {
                    id: id.to_string(),
                    vector: None,
                    tags: None,
                }],
            )
            .await?;
        status_to_result(response.statuses.into_iter().next())
    }

    pub async fn upsert_vector(
        &self,
        id: String,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError> {
        let response = self
            .send_write(
                WriteMode::Upsert,
                vec![WriteItem {
                    id: id.clone(),
                    vector: Some(vector),
                    tags,
                }],
            )
            .await?;
        status_to_result(response.statuses.into_iter().next())
    }

    pub async fn batch_create_vectors(
        &self,
        items: Vec<VectorInput>,
    ) -> Result<BatchWriteResponse, VectraDBError> {
        let write_items = items.into_iter().map(vector_input_to_write_item).collect();
        self.send_write(WriteMode::Create, write_items).await
    }

    pub async fn batch_upsert_vectors(
        &self,
        items: Vec<VectorInput>,
    ) -> Result<BatchWriteResponse, VectraDBError> {
        let write_items = items.into_iter().map(vector_input_to_write_item).collect();
        self.send_write(WriteMode::Upsert, write_items).await
    }

    pub async fn search_similar(
        &self,
        query_vector: Array1<f32>,
        top_k: usize,
    ) -> Result<Vec<SimilarityResult>, VectraDBError> {
        validate_vector(&query_vector)?;
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let active_results = {
            let state = self.core.state.read().unwrap();
            if let Some(expected) = state.manifest.dimension {
                if expected != query_vector.len() {
                    return Err(VectraDBError::DimensionMismatch {
                        expected,
                        actual: query_vector.len(),
                    });
                }
            }

            let active_docs: Vec<VectorDocument> =
                state.mutable_segment.documents.values().cloned().collect();
            exact_search_candidates(
                &active_docs,
                &query_vector,
                top_k.saturating_mul(4).max(top_k),
                RecordLocation::Mutable {
                    segment_id: state.mutable_segment.id,
                },
            )?
        };

        let sealed_segments: Vec<Arc<SealedSegment>> = self
            .core
            .sealed_segments
            .read()
            .unwrap()
            .values()
            .cloned()
            .collect();

        let mut candidates = active_results;
        for segment in sealed_segments {
            candidates.extend(segment.search(&query_vector, top_k)?);
        }

        let state = self.core.state.read().unwrap();
        let mut best_by_id: HashMap<String, SimilarityResult> = HashMap::new();
        for candidate in candidates {
            if state.id_map.get(&candidate.result.id) != Some(&candidate.location) {
                continue;
            }

            match best_by_id.get(&candidate.result.id) {
                Some(existing) if existing.score >= candidate.result.score => {}
                _ => {
                    best_by_id.insert(candidate.result.id.clone(), candidate.result);
                }
            }
        }

        let mut results: Vec<SimilarityResult> = best_by_id.into_values().collect();
        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        results.truncate(top_k);
        Ok(results)
    }

    pub async fn search_similar_with_ef(
        &self,
        query_vector: Array1<f32>,
        top_k: usize,
        _ef: usize,
    ) -> Result<Vec<SimilarityResult>, VectraDBError> {
        self.search_similar(query_vector, top_k).await
    }

    pub async fn search_with_filter(
        &self,
        query_vector: Array1<f32>,
        top_k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SimilarityResult>, VectraDBError> {
        let oversample = if filter.is_some() {
            top_k.saturating_mul(10).max(top_k)
        } else {
            top_k
        };
        let mut results = self.search_similar(query_vector, oversample).await?;
        if let Some(filter) = filter {
            results.retain(|result| filter.matches(&result.metadata.tags));
        }
        results.truncate(top_k);
        Ok(results)
    }

    pub async fn search_by_threshold(
        &self,
        query_vector: Array1<f32>,
        min_similarity: f32,
        max_results: usize,
    ) -> Result<Vec<SimilarityResult>, VectraDBError> {
        let mut results = self
            .search_similar(query_vector, max_results.saturating_mul(4).max(max_results))
            .await?;
        results.retain(|result| result.score >= min_similarity);
        results.truncate(max_results);
        Ok(results)
    }

    #[cfg(feature = "gpu")]
    pub async fn search_gpu(
        &self,
        query_vector: Array1<f32>,
        top_k: usize,
        _gpu: &vectradb_search::gpu::GpuDistanceEngine,
        _metric: DistanceMetric,
    ) -> Result<Vec<SimilarityResult>, VectraDBError> {
        self.search_similar(query_vector, top_k).await
    }

    #[cfg(feature = "gpu")]
    pub async fn search_gpu_rerank(
        &self,
        query_vector: Array1<f32>,
        top_k: usize,
        _rerank_ef: usize,
        _gpu: &vectradb_search::gpu::GpuDistanceEngine,
        _metric: DistanceMetric,
    ) -> Result<Vec<SimilarityResult>, VectraDBError> {
        self.search_similar(query_vector, top_k).await
    }

    pub async fn list_vectors(&self) -> Result<Vec<String>, VectraDBError> {
        let state = self.core.state.read().unwrap();
        let mut ids: Vec<String> = state.id_map.keys().cloned().collect();
        ids.sort();
        Ok(ids)
    }

    pub fn flush(&self) -> Result<(), VectraDBError> {
        maybe_seal_current_segment(&self.core)?;
        OpenOptions::new()
            .read(true)
            .open(&self.core.wal_path)
            .and_then(|file| file.sync_all())
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
        Ok(())
    }

    pub async fn get_stats(&self) -> Result<DatabaseStats, VectraDBError> {
        let state = self.core.state.read().unwrap();
        let sealed = self.core.sealed_segments.read().unwrap();
        let mut memory_usage = state.mutable_segment.raw_vector_bytes as u64;
        for segment in sealed.values() {
            memory_usage = memory_usage.saturating_add(
                segment
                    .docs
                    .iter()
                    .map(|doc| doc.data.len().saturating_mul(4) as u64)
                    .sum::<u64>(),
            );
        }

        Ok(DatabaseStats {
            total_vectors: state.id_map.len(),
            dimension: state
                .manifest
                .dimension
                .or(self.core.config.index_config.dimension)
                .unwrap_or(0),
            memory_usage,
        })
    }

    async fn send_write(
        &self,
        mode: WriteMode,
        items: Vec<WriteItem>,
    ) -> Result<BatchWriteResponse, VectraDBError> {
        let estimated_bytes = items.iter().map(estimate_write_item_bytes).sum();
        let (tx, rx) = oneshot::channel();
        self.write_tx
            .send(PendingRequest {
                mode,
                items,
                estimated_bytes,
                responder: tx,
            })
            .await
            .map_err(|_| {
                VectraDBError::DatabaseError(anyhow::anyhow!("write pipeline is unavailable"))
            })?;

        rx.await.map_err(|_| {
            VectraDBError::DatabaseError(anyhow::anyhow!("write pipeline response dropped"))
        })
    }
}

fn vector_input_to_write_item(input: VectorInput) -> WriteItem {
    WriteItem {
        id: input.id,
        vector: Some(Array1::from_vec(input.vector)),
        tags: if input.tags.is_empty() {
            None
        } else {
            Some(input.tags)
        },
    }
}

async fn writer_loop(core: Arc<StorageCore>, mut rx: mpsc::Receiver<PendingRequest>) {
    let mut wal_file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&core.wal_path)
    {
        Ok(file) => file,
        Err(_) => return,
    };

    let tick_ms = core.config.commit_interval_ms.max(1);
    let mut ticker = tokio::time::interval(Duration::from_millis(tick_ms));
    let mut pending = Vec::new();
    let mut pending_vectors = 0usize;
    let mut pending_bytes = 0usize;

    loop {
        tokio::select! {
            maybe_request = rx.recv() => {
                match maybe_request {
                    Some(request) => {
                        pending_vectors = pending_vectors.saturating_add(request.items.len());
                        pending_bytes = pending_bytes.saturating_add(request.estimated_bytes);
                        pending.push(request);

                        let strict = core.config.durability_mode == DurabilityMode::Strict;
                        let vector_limit = pending_vectors >= core.config.commit_max_vectors;
                        let byte_limit = pending_bytes >= core.config.commit_max_bytes;
                        if strict || vector_limit || byte_limit {
                            flush_pending(&core, &mut wal_file, &mut pending);
                            pending_vectors = 0;
                            pending_bytes = 0;
                        }
                    }
                    None => {
                        if !pending.is_empty() {
                            flush_pending(&core, &mut wal_file, &mut pending);
                        }
                        break;
                    }
                }
            }
            _ = ticker.tick() => {
                if !pending.is_empty() {
                    flush_pending(&core, &mut wal_file, &mut pending);
                    pending_vectors = 0;
                    pending_bytes = 0;
                } else {
                    let _ = maybe_seal_current_segment(&core);
                }
            }
        }
    }
}

fn flush_pending(core: &Arc<StorageCore>, wal_file: &mut File, pending: &mut Vec<PendingRequest>) {
    if pending.is_empty() {
        return;
    }

    let drained: Vec<PendingRequest> = pending.drain(..).collect();
    let fallback_ids: Vec<Vec<String>> = drained
        .iter()
        .map(|request| request.items.iter().map(|item| item.id.clone()).collect())
        .collect();

    match evaluate_requests(core, &drained) {
        Ok(evaluated) => {
            let write_result = append_wal_batch(
                wal_file,
                &evaluated.wal_records,
                core.config.durability_mode,
            )
            .and_then(|_| apply_records(core, &evaluated));

            match write_result {
                Ok(()) => {
                    for (request, response) in
                        drained.into_iter().zip(evaluated.responses.into_iter())
                    {
                        let _ = request.responder.send(response);
                    }
                    let _ = maybe_seal_current_segment(core);
                }
                Err(err) => {
                    for (request, ids) in drained.into_iter().zip(fallback_ids.into_iter()) {
                        let statuses = ids
                            .into_iter()
                            .map(|id| BatchWriteItemStatus::error(id, "INTERNAL", err.to_string()))
                            .collect();
                        let _ = request.responder.send(BatchWriteResponse { statuses });
                    }
                }
            }
        }
        Err(err) => {
            for (request, ids) in drained.into_iter().zip(fallback_ids.into_iter()) {
                let statuses = ids
                    .into_iter()
                    .map(|id| BatchWriteItemStatus::error(id, "INTERNAL", err.to_string()))
                    .collect();
                let _ = request.responder.send(BatchWriteResponse { statuses });
            }
        }
    }
}

fn evaluate_requests(
    core: &StorageCore,
    requests: &[PendingRequest],
) -> Result<EvaluatedBatch, VectraDBError> {
    let state = core.state.read().unwrap();
    let sealed = core.sealed_segments.read().unwrap();

    let mut overlay: HashMap<String, OverlayEntry> = HashMap::new();
    let mut responses = Vec::with_capacity(requests.len());
    let mut wal_records = Vec::new();
    let mut next_seq = state.next_seq;
    let mut dimension = state
        .manifest
        .dimension
        .or(core.config.index_config.dimension);

    for request in requests {
        let mut statuses = Vec::with_capacity(request.items.len());
        for item in &request.items {
            let status = match request.mode {
                WriteMode::Create => evaluate_create(
                    item,
                    &state,
                    &sealed,
                    &mut overlay,
                    &mut wal_records,
                    &mut next_seq,
                    &mut dimension,
                ),
                WriteMode::Update => evaluate_update(
                    item,
                    &state,
                    &sealed,
                    &mut overlay,
                    &mut wal_records,
                    &mut next_seq,
                    &mut dimension,
                ),
                WriteMode::Upsert => evaluate_upsert(
                    item,
                    &state,
                    &sealed,
                    &mut overlay,
                    &mut wal_records,
                    &mut next_seq,
                    &mut dimension,
                ),
                WriteMode::Delete => evaluate_delete(
                    item,
                    &state,
                    &sealed,
                    &mut overlay,
                    &mut wal_records,
                    &mut next_seq,
                ),
            };
            statuses.push(status);
        }
        responses.push(BatchWriteResponse { statuses });
    }

    Ok(EvaluatedBatch {
        responses,
        wal_records,
        next_seq,
        dimension,
    })
}

fn evaluate_create(
    item: &WriteItem,
    state: &EngineState,
    sealed: &BTreeMap<u64, Arc<SealedSegment>>,
    overlay: &mut HashMap<String, OverlayEntry>,
    wal_records: &mut Vec<WalRecord>,
    next_seq: &mut u64,
    dimension: &mut Option<usize>,
) -> BatchWriteItemStatus {
    if effective_document(item.id.as_str(), overlay, state, sealed).is_some() {
        return BatchWriteItemStatus::error(
            item.id.clone(),
            "ALREADY_EXISTS",
            format!("Vector already exists: {}", item.id),
        );
    }

    match build_new_document(item, *dimension) {
        Ok(document) => {
            *dimension = Some(document.metadata.dimension);
            overlay.insert(item.id.clone(), OverlayEntry::Upsert(document.clone()));
            wal_records.push(WalRecord {
                seq: *next_seq,
                operation: WalOperation::Upsert { document },
            });
            *next_seq = next_seq.saturating_add(1);
            BatchWriteItemStatus::ok(item.id.clone())
        }
        Err(err) => error_status(item.id.clone(), err),
    }
}

fn evaluate_update(
    item: &WriteItem,
    state: &EngineState,
    sealed: &BTreeMap<u64, Arc<SealedSegment>>,
    overlay: &mut HashMap<String, OverlayEntry>,
    wal_records: &mut Vec<WalRecord>,
    next_seq: &mut u64,
    dimension: &mut Option<usize>,
) -> BatchWriteItemStatus {
    let Some(existing) = effective_document(item.id.as_str(), overlay, state, sealed) else {
        return BatchWriteItemStatus::error(
            item.id.clone(),
            "NOT_FOUND",
            format!("Vector not found: {}", item.id),
        );
    };

    match update_existing_document(existing, item, *dimension) {
        Ok(document) => {
            *dimension = Some(document.metadata.dimension);
            overlay.insert(item.id.clone(), OverlayEntry::Upsert(document.clone()));
            wal_records.push(WalRecord {
                seq: *next_seq,
                operation: WalOperation::Upsert { document },
            });
            *next_seq = next_seq.saturating_add(1);
            BatchWriteItemStatus::ok(item.id.clone())
        }
        Err(err) => error_status(item.id.clone(), err),
    }
}

fn evaluate_upsert(
    item: &WriteItem,
    state: &EngineState,
    sealed: &BTreeMap<u64, Arc<SealedSegment>>,
    overlay: &mut HashMap<String, OverlayEntry>,
    wal_records: &mut Vec<WalRecord>,
    next_seq: &mut u64,
    dimension: &mut Option<usize>,
) -> BatchWriteItemStatus {
    let current = effective_document(item.id.as_str(), overlay, state, sealed);
    let result = match current {
        Some(existing) => update_existing_document(existing, item, *dimension),
        None => build_new_document(item, *dimension),
    };

    match result {
        Ok(document) => {
            *dimension = Some(document.metadata.dimension);
            overlay.insert(item.id.clone(), OverlayEntry::Upsert(document.clone()));
            wal_records.push(WalRecord {
                seq: *next_seq,
                operation: WalOperation::Upsert { document },
            });
            *next_seq = next_seq.saturating_add(1);
            BatchWriteItemStatus::ok(item.id.clone())
        }
        Err(err) => error_status(item.id.clone(), err),
    }
}

fn evaluate_delete(
    item: &WriteItem,
    state: &EngineState,
    sealed: &BTreeMap<u64, Arc<SealedSegment>>,
    overlay: &mut HashMap<String, OverlayEntry>,
    wal_records: &mut Vec<WalRecord>,
    next_seq: &mut u64,
) -> BatchWriteItemStatus {
    if effective_document(item.id.as_str(), overlay, state, sealed).is_none() {
        return BatchWriteItemStatus::error(
            item.id.clone(),
            "NOT_FOUND",
            format!("Vector not found: {}", item.id),
        );
    }

    overlay.insert(item.id.clone(), OverlayEntry::Delete);
    wal_records.push(WalRecord {
        seq: *next_seq,
        operation: WalOperation::Delete {
            id: item.id.clone(),
        },
    });
    *next_seq = next_seq.saturating_add(1);
    BatchWriteItemStatus::ok(item.id.clone())
}

fn build_new_document(
    item: &WriteItem,
    expected_dimension: Option<usize>,
) -> Result<VectorDocument, VectraDBError> {
    let vector = item
        .vector
        .clone()
        .ok_or_else(|| VectraDBError::InvalidVector)?;
    if let Some(expected) = expected_dimension {
        if vector.len() != expected {
            return Err(VectraDBError::DimensionMismatch {
                expected,
                actual: vector.len(),
            });
        }
    }
    create_vector_document(item.id.clone(), vector, item.tags.clone())
}

fn update_existing_document(
    existing: VectorDocument,
    item: &WriteItem,
    expected_dimension: Option<usize>,
) -> Result<VectorDocument, VectraDBError> {
    let vector = item
        .vector
        .clone()
        .ok_or_else(|| VectraDBError::InvalidVector)?;
    if let Some(expected) = expected_dimension {
        if vector.len() != expected {
            return Err(VectraDBError::DimensionMismatch {
                expected,
                actual: vector.len(),
            });
        }
    }
    update_vector_document(existing, vector, item.tags.clone())
}

fn effective_document(
    id: &str,
    overlay: &HashMap<String, OverlayEntry>,
    state: &EngineState,
    sealed: &BTreeMap<u64, Arc<SealedSegment>>,
) -> Option<VectorDocument> {
    match overlay.get(id) {
        Some(OverlayEntry::Upsert(document)) => return Some(document.clone()),
        Some(OverlayEntry::Delete) => return None,
        None => {}
    }

    match state.id_map.get(id) {
        Some(RecordLocation::Mutable { .. }) => state.mutable_segment.documents.get(id).cloned(),
        Some(RecordLocation::Sealed { segment_id }) => sealed
            .get(segment_id)
            .and_then(|segment| segment.get_document(id)),
        None => None,
    }
}

fn append_wal_batch(
    wal_file: &mut File,
    records: &[WalRecord],
    durability_mode: DurabilityMode,
) -> Result<(), VectraDBError> {
    if records.is_empty() {
        return Ok(());
    }

    let batch = WalBatch {
        records: records.to_vec(),
    };
    let payload =
        bincode::serialize(&batch).map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
    let len = (payload.len() as u32).to_le_bytes();

    wal_file
        .write_all(&len)
        .and_then(|_| wal_file.write_all(&payload))
        .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
    wal_file
        .flush()
        .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

    if durability_mode != DurabilityMode::Async {
        wal_file
            .sync_data()
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
    }

    Ok(())
}

fn apply_records(core: &StorageCore, evaluated: &EvaluatedBatch) -> Result<(), VectraDBError> {
    let mut state = core.state.write().unwrap();
    state.manifest.dimension = evaluated.dimension;
    state.next_seq = evaluated.next_seq;

    for record in &evaluated.wal_records {
        match &record.operation {
            WalOperation::Upsert { document } => {
                let mutable_segment_id = state.mutable_segment.id;
                state
                    .mutable_segment
                    .apply_upsert(document.clone(), record.seq);
                state.id_map.insert(
                    document.metadata.id.clone(),
                    RecordLocation::Mutable {
                        segment_id: mutable_segment_id,
                    },
                );
            }
            WalOperation::Delete { id } => {
                state.mutable_segment.apply_delete(id, record.seq);
                state.id_map.remove(id);
            }
        }
    }

    Ok(())
}

fn maybe_seal_current_segment(core: &Arc<StorageCore>) -> Result<(), VectraDBError> {
    let (manifest, segment_data, sealed_id) = {
        let state = core.state.read().unwrap();
        if !state.mutable_segment.should_seal(&core.config) {
            return Ok(());
        }

        let sealed_id = state.mutable_segment.id;
        let next_mutable_id = state.manifest.next_segment_id;
        let manifest = Manifest {
            version: STORAGE_VERSION,
            search_algorithm: core.config.search_algorithm,
            dimension: state.manifest.dimension,
            compacted_seq: state.mutable_segment.max_seq,
            current_mutable_segment_id: next_mutable_id,
            next_segment_id: next_mutable_id.saturating_add(1),
            sealed_segments: {
                let mut metas = state.manifest.sealed_segments.clone();
                metas.push(SealedSegmentMeta {
                    id: sealed_id,
                    doc_count: state.mutable_segment.documents.len(),
                    tombstone_count: state.mutable_segment.tombstones.len(),
                });
                metas
            },
        };
        let segment_data = SegmentDiskData {
            id: sealed_id,
            documents: state.mutable_segment.documents.values().cloned().collect(),
            tombstones: state.mutable_segment.tombstones.iter().cloned().collect(),
        };

        (manifest, segment_data, sealed_id)
    };

    let segment_dir = core.segments_dir.join(format!("{:020}", sealed_id));
    write_segment(&segment_dir, &segment_data)?;
    write_manifest(&core.manifest_path, &manifest)?;

    let segment = Arc::new(SealedSegment::from_disk_data(segment_dir, segment_data));
    {
        let mut sealed = core.sealed_segments.write().unwrap();
        sealed.insert(sealed_id, segment.clone());
    }
    {
        let mut state = core.state.write().unwrap();
        let live_ids: Vec<String> = state.mutable_segment.documents.keys().cloned().collect();
        for id in live_ids {
            state.id_map.insert(
                id,
                RecordLocation::Sealed {
                    segment_id: sealed_id,
                },
            );
        }
        state.manifest = manifest;
        state.mutable_segment =
            MutableSegment::new(state.manifest.current_mutable_segment_id, now_secs());
    }

    schedule_segment_index(core.clone(), segment);
    Ok(())
}

fn schedule_segment_index(core: Arc<StorageCore>, segment: Arc<SealedSegment>) {
    let config = core.config.clone();
    tokio::spawn(async move {
        let docs = segment.docs.clone();
        let path = segment.path.clone();
        let build = tokio::task::spawn_blocking(move || -> Result<HNSWIndex, VectraDBError> {
            let dimension = config
                .index_config
                .dimension
                .or_else(|| docs.first().map(|doc| doc.metadata.dimension))
                .unwrap_or(0);
            let mut index = HNSWIndex::new(
                dimension,
                config.index_config.m,
                config.index_config.ef_construction,
                config.index_config.search_ef,
                config.index_config.metric,
            );
            index.build_index(docs)?;
            Ok(index)
        })
        .await;

        if let Ok(Ok(index)) = build {
            *segment.index.write().unwrap() = Some(index);
            let _ = fs::write(
                path.join(INDEX_META_FILE),
                serde_json::to_vec_pretty(&IndexMeta {
                    algorithm: "HNSW".to_string(),
                    built_at: now_secs(),
                    doc_count: segment.docs.len(),
                })
                .unwrap_or_default(),
            );
        }
    });
}

fn exact_search_candidates(
    documents: &[VectorDocument],
    query: &Array1<f32>,
    top_k: usize,
    location: RecordLocation,
) -> Result<Vec<SearchCandidate>, VectraDBError> {
    let results = vectradb_components::similarity::find_similar_vectors_euclidean(
        &query.view(),
        documents,
        top_k,
    )?;
    Ok(results
        .into_iter()
        .map(|result| SearchCandidate {
            location: location.clone(),
            result,
        })
        .collect())
}

fn error_status(id: String, error: VectraDBError) -> BatchWriteItemStatus {
    BatchWriteItemStatus::error(id, error_code(&error), error.to_string())
}

fn error_code(error: &VectraDBError) -> &'static str {
    match error {
        VectraDBError::VectorAlreadyExists { .. } | VectraDBError::DuplicateVector { .. } => {
            "ALREADY_EXISTS"
        }
        VectraDBError::VectorNotFound { .. } => "NOT_FOUND",
        VectraDBError::DimensionMismatch { .. } | VectraDBError::InvalidVector => {
            "INVALID_ARGUMENT"
        }
        VectraDBError::DatabaseError(_) => "INTERNAL",
    }
}

fn status_to_result(status: Option<BatchWriteItemStatus>) -> Result<(), VectraDBError> {
    let status = status.ok_or_else(|| {
        VectraDBError::DatabaseError(anyhow::anyhow!("missing batch write status"))
    })?;

    if status.is_ok() {
        return Ok(());
    }

    match status.code.as_str() {
        "ALREADY_EXISTS" => Err(VectraDBError::VectorAlreadyExists { id: status.id }),
        "NOT_FOUND" => Err(VectraDBError::VectorNotFound { id: status.id }),
        "INVALID_ARGUMENT" => {
            if let Some((expected, actual)) = parse_dimension_message(&status.message) {
                Err(VectraDBError::DimensionMismatch { expected, actual })
            } else {
                Err(VectraDBError::InvalidVector)
            }
        }
        _ => Err(VectraDBError::DatabaseError(anyhow::anyhow!(
            status.message
        ))),
    }
}

fn parse_dimension_message(message: &str) -> Option<(usize, usize)> {
    let needle = "Vector dimension mismatch: expected ";
    let rest = message.strip_prefix(needle)?;
    let mut parts = rest.split(", got ");
    let expected = parts.next()?.parse().ok()?;
    let actual = parts.next()?.parse().ok()?;
    Some((expected, actual))
}

fn estimate_write_item_bytes(item: &WriteItem) -> usize {
    let vector_bytes = item
        .vector
        .as_ref()
        .map(|vector| vector.len().saturating_mul(4))
        .unwrap_or(0);
    let tag_bytes: usize = item
        .tags
        .as_ref()
        .map(|tags| tags.iter().map(|(k, v)| k.len() + v.len()).sum())
        .unwrap_or(0);
    item.id.len() + vector_bytes + tag_bytes
}

fn write_manifest(path: &Path, manifest: &Manifest) -> Result<(), VectraDBError> {
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
    fs::write(&tmp, bytes).map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
    fs::rename(&tmp, path).map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
    Ok(())
}

fn load_manifest(path: &Path) -> Result<Manifest, VectraDBError> {
    let bytes = fs::read(path).map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
    serde_json::from_slice(&bytes).map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))
}

fn write_segment(path: &Path, segment: &SegmentDiskData) -> Result<(), VectraDBError> {
    fs::create_dir_all(path).map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
    let payload = bincode::serialize(segment)
        .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
    fs::write(path.join(SEGMENT_DATA_FILE), payload)
        .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
    Ok(())
}

fn load_segment(path: &Path) -> Result<SegmentDiskData, VectraDBError> {
    let bytes = fs::read(path.join(SEGMENT_DATA_FILE))
        .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
    bincode::deserialize(&bytes).map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))
}

fn read_wal_records(path: &Path) -> Result<Vec<WalRecord>, VectraDBError> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(VectraDBError::DatabaseError(anyhow::anyhow!(err))),
    };

    let mut records = Vec::new();
    loop {
        let mut len_bytes = [0u8; 4];
        match file.read_exact(&mut len_bytes) {
            Ok(()) => {}
            Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(VectraDBError::DatabaseError(anyhow::anyhow!(err))),
        }

        let len = u32::from_le_bytes(len_bytes) as usize;
        let mut payload = vec![0u8; len];
        match file.read_exact(&mut payload) {
            Ok(()) => {}
            Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(VectraDBError::DatabaseError(anyhow::anyhow!(err))),
        }

        let batch: WalBatch = bincode::deserialize(&payload)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
        records.extend(batch.records);
    }

    Ok(records)
}

fn dir_has_entries(path: &Path) -> Result<bool, VectraDBError> {
    let mut entries =
        fs::read_dir(path).map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
    Ok(entries.next().is_some())
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sled::Db;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_persistent_db_creation() {
        let temp_dir = tempdir().unwrap();
        let config = DatabaseConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };

        let db = PersistentVectorDB::new(config).await;
        assert!(db.is_ok());
    }

    #[tokio::test]
    async fn test_batch_create_partial_success() {
        let temp_dir = tempdir().unwrap();
        let config = DatabaseConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            index_config: SearchConfig {
                dimension: Some(3),
                ..Default::default()
            },
            ..Default::default()
        };
        let db = PersistentVectorDB::new(config).await.unwrap();
        db.create_vector(
            "dup".to_string(),
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
            None,
        )
        .await
        .unwrap();

        let response = db
            .batch_create_vectors(vec![
                VectorInput {
                    id: "dup".to_string(),
                    vector: vec![1.0, 0.0, 0.0],
                    tags: HashMap::new(),
                },
                VectorInput {
                    id: "good".to_string(),
                    vector: vec![0.0, 1.0, 0.0],
                    tags: HashMap::new(),
                },
                VectorInput {
                    id: "bad".to_string(),
                    vector: vec![f32::NAN, 0.0, 0.0],
                    tags: HashMap::new(),
                },
            ])
            .await
            .unwrap();

        assert_eq!(response.statuses.len(), 3);
        assert_eq!(response.statuses[0].code, "ALREADY_EXISTS");
        assert_eq!(response.statuses[1].code, "OK");
        assert_eq!(response.statuses[2].code, "INVALID_ARGUMENT");
        assert!(db.get_vector("good").await.is_ok());
    }

    #[tokio::test]
    async fn test_search_returns_stored_metadata_without_disk_lookup() {
        let temp_dir = tempdir().unwrap();
        let config = DatabaseConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            index_config: SearchConfig {
                dimension: Some(3),
                ..Default::default()
            },
            segment_max_vectors: 1,
            ..Default::default()
        };
        let db = PersistentVectorDB::new(config).await.unwrap();
        let mut tags = HashMap::new();
        tags.insert("source".to_string(), "unit-test".to_string());

        db.create_vector(
            "doc1".to_string(),
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
            Some(tags),
        )
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;

        let results = db
            .search_similar(Array1::from_vec(vec![1.0, 0.0, 0.0]), 1)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].metadata.dimension, 3);
        assert_eq!(
            results[0].metadata.tags.get("source"),
            Some(&"unit-test".to_string())
        );
    }

    #[tokio::test]
    async fn test_delete_survives_restart() {
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_string_lossy().to_string();
        let config = DatabaseConfig {
            data_dir: path.clone(),
            index_config: SearchConfig {
                dimension: Some(3),
                ..Default::default()
            },
            ..Default::default()
        };

        let db = PersistentVectorDB::new(config.clone()).await.unwrap();
        db.create_vector(
            "doc1".to_string(),
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
            None,
        )
        .await
        .unwrap();
        db.delete_vector("doc1").await.unwrap();

        let reopened = PersistentVectorDB::new(config).await.unwrap();
        assert!(matches!(
            reopened.get_vector("doc1").await,
            Err(VectraDBError::VectorNotFound { .. })
        ));
    }

    #[tokio::test]
    async fn test_recovery_ignores_partial_wal_tail() {
        let temp_dir = tempdir().unwrap();
        let config = DatabaseConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            index_config: SearchConfig {
                dimension: Some(3),
                ..Default::default()
            },
            ..Default::default()
        };
        let db = PersistentVectorDB::new(config.clone()).await.unwrap();
        db.create_vector(
            "stable".to_string(),
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
            None,
        )
        .await
        .unwrap();

        let wal_path = PathBuf::from(&config.data_dir).join(WAL_FILE);
        let mut wal = OpenOptions::new().append(true).open(&wal_path).unwrap();
        wal.write_all(&1234u32.to_le_bytes()).unwrap();
        wal.write_all(&[1, 2, 3]).unwrap();
        wal.flush().unwrap();

        let reopened = PersistentVectorDB::new(config).await.unwrap();
        assert!(reopened.get_vector("stable").await.is_ok());
    }

    #[tokio::test]
    async fn test_legacy_migration_preserves_search() {
        let legacy_dir = tempdir().unwrap();
        let new_dir = tempdir().unwrap();
        let legacy_db: Db = sled::open(legacy_dir.path()).unwrap();
        let vectors_tree = legacy_db.open_tree("vectors").unwrap();
        let metadata_tree = legacy_db.open_tree("metadata").unwrap();

        let doc = create_vector_document(
            "legacy".to_string(),
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
            Some(HashMap::from([(
                "origin".to_string(),
                "legacy".to_string(),
            )])),
        )
        .unwrap();

        vectors_tree
            .insert("legacy".as_bytes(), bincode::serialize(&doc.data).unwrap())
            .unwrap();
        metadata_tree
            .insert(
                "legacy".as_bytes(),
                bincode::serialize(&doc.metadata).unwrap(),
            )
            .unwrap();
        legacy_db.flush().unwrap();
        drop(metadata_tree);
        drop(vectors_tree);
        drop(legacy_db);

        let config = DatabaseConfig {
            data_dir: new_dir.path().to_string_lossy().to_string(),
            index_config: SearchConfig {
                dimension: Some(3),
                ..Default::default()
            },
            ..Default::default()
        };

        let stats = PersistentVectorDB::migrate_legacy_data(
            legacy_dir.path(),
            new_dir.path(),
            config.clone(),
        )
        .await
        .unwrap();
        assert_eq!(stats.migrated_vectors, 1);

        let migrated = PersistentVectorDB::new(config).await.unwrap();
        let results = migrated
            .search_similar(Array1::from_vec(vec![1.0, 0.0, 0.0]), 1)
            .await
            .unwrap();
        assert_eq!(results[0].id, "legacy");
        assert_eq!(
            results[0].metadata.tags.get("origin"),
            Some(&"legacy".to_string())
        );
    }
}
