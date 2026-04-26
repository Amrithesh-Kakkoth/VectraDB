use ndarray::Array1;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use vectradb_components::{
    BatchWriteResponse as CoreBatchWriteResponse, VectorDocument, VectorInput as CoreVectorInput,
    VectraDBError,
};
use vectradb_storage::PersistentVectorDB;

pub mod vectradb {
    tonic::include_proto!("vectradb");
}

use vectradb::{
    vectra_db_server::{VectraDb, VectraDbServer},
    BatchCreateVectorsRequest, BatchUpsertVectorsRequest, BatchWriteItemStatus, BatchWriteResponse,
    CreateVectorRequest, DeleteVectorRequest, DeleteVectorResponse, GetStatsRequest,
    GetVectorRequest, HealthCheckRequest, HealthCheckResponse, ListVectorsRequest,
    ListVectorsResponse, SearchRequest, SearchResponse, SimilarityResult, StatsResponse,
    UpdateVectorRequest, UpsertVectorRequest, VectorInput, VectorMetadata, VectorResponse,
};

pub struct VectraDbService {
    db: Arc<PersistentVectorDB>,
}

impl VectraDbService {
    pub fn new(db: Arc<PersistentVectorDB>) -> Self {
        Self { db }
    }

    pub fn into_service(self) -> VectraDbServer<Self> {
        VectraDbServer::new(self)
    }

    fn map_error(error: VectraDBError) -> Status {
        match error {
            VectraDBError::VectorAlreadyExists { id } | VectraDBError::DuplicateVector { id } => {
                Status::already_exists(format!("Vector already exists: {id}"))
            }
            VectraDBError::VectorNotFound { id } => {
                Status::not_found(format!("Vector not found: {id}"))
            }
            VectraDBError::DimensionMismatch { expected, actual } => Status::invalid_argument(
                format!("Vector dimension mismatch: expected {expected}, got {actual}"),
            ),
            VectraDBError::InvalidVector => Status::invalid_argument("Invalid vector data"),
            VectraDBError::DatabaseError(inner) => Status::internal(inner.to_string()),
        }
    }

    fn document_response(document: VectorDocument) -> VectorResponse {
        VectorResponse {
            id: document.metadata.id,
            vector: document.data.to_vec(),
            dimension: document.metadata.dimension as u64,
            created_at: document.metadata.created_at,
            updated_at: document.metadata.updated_at,
            tags: document.metadata.tags,
        }
    }

    fn batch_response(results: CoreBatchWriteResponse) -> BatchWriteResponse {
        BatchWriteResponse {
            statuses: results
                .statuses
                .into_iter()
                .map(|item| BatchWriteItemStatus {
                    id: item.id,
                    code: item.code,
                    message: item.message,
                })
                .collect(),
        }
    }

    fn into_write_vector(input: VectorInput) -> CoreVectorInput {
        CoreVectorInput {
            id: input.id,
            vector: input.vector,
            tags: input.tags,
        }
    }
}

#[tonic::async_trait]
impl VectraDb for VectraDbService {
    async fn create_vector(
        &self,
        request: Request<CreateVectorRequest>,
    ) -> Result<Response<VectorResponse>, Status> {
        let req = request.into_inner();
        let vector = Array1::from_vec(req.vector);
        if vector.is_empty() {
            return Err(Status::invalid_argument("vector must not be empty"));
        }
        let tags = if req.tags.is_empty() {
            None
        } else {
            Some(req.tags)
        };

        self.db
            .create_vector(req.id.clone(), vector, tags)
            .await
            .map_err(Self::map_error)?;

        let document = self.db.get_vector(&req.id).await.map_err(Self::map_error)?;
        Ok(Response::new(Self::document_response(document)))
    }

    async fn batch_create_vectors(
        &self,
        request: Request<BatchCreateVectorsRequest>,
    ) -> Result<Response<BatchWriteResponse>, Status> {
        let items = request
            .into_inner()
            .items
            .into_iter()
            .map(Self::into_write_vector)
            .collect();
        let results = self
            .db
            .batch_create_vectors(items)
            .await
            .map_err(Self::map_error)?;
        Ok(Response::new(Self::batch_response(results)))
    }

    async fn get_vector(
        &self,
        request: Request<GetVectorRequest>,
    ) -> Result<Response<VectorResponse>, Status> {
        let document = self
            .db
            .get_vector(&request.into_inner().id)
            .await
            .map_err(Self::map_error)?;
        Ok(Response::new(Self::document_response(document)))
    }

    async fn update_vector(
        &self,
        request: Request<UpdateVectorRequest>,
    ) -> Result<Response<VectorResponse>, Status> {
        let req = request.into_inner();
        let vector = Array1::from_vec(req.vector);
        if vector.is_empty() {
            return Err(Status::invalid_argument("vector must not be empty"));
        }
        let tags = if req.tags.is_empty() {
            None
        } else {
            Some(req.tags)
        };

        self.db
            .update_vector(&req.id, vector, tags)
            .await
            .map_err(Self::map_error)?;

        let document = self.db.get_vector(&req.id).await.map_err(Self::map_error)?;
        Ok(Response::new(Self::document_response(document)))
    }

    async fn delete_vector(
        &self,
        request: Request<DeleteVectorRequest>,
    ) -> Result<Response<DeleteVectorResponse>, Status> {
        self.db
            .delete_vector(&request.into_inner().id)
            .await
            .map_err(Self::map_error)?;
        Ok(Response::new(DeleteVectorResponse { success: true }))
    }

    async fn upsert_vector(
        &self,
        request: Request<UpsertVectorRequest>,
    ) -> Result<Response<VectorResponse>, Status> {
        let req = request.into_inner();
        let vector = Array1::from_vec(req.vector);
        if vector.is_empty() {
            return Err(Status::invalid_argument("vector must not be empty"));
        }
        let tags = if req.tags.is_empty() {
            None
        } else {
            Some(req.tags)
        };

        self.db
            .upsert_vector(req.id.clone(), vector, tags)
            .await
            .map_err(Self::map_error)?;

        let document = self.db.get_vector(&req.id).await.map_err(Self::map_error)?;
        Ok(Response::new(Self::document_response(document)))
    }

    async fn batch_upsert_vectors(
        &self,
        request: Request<BatchUpsertVectorsRequest>,
    ) -> Result<Response<BatchWriteResponse>, Status> {
        let items = request
            .into_inner()
            .items
            .into_iter()
            .map(Self::into_write_vector)
            .collect();
        let results = self
            .db
            .batch_upsert_vectors(items)
            .await
            .map_err(Self::map_error)?;
        Ok(Response::new(Self::batch_response(results)))
    }

    async fn search_similar(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let vector = Array1::from_vec(req.vector);
        if vector.is_empty() {
            return Err(Status::invalid_argument("query vector must not be empty"));
        }
        let top_k = (req.top_k as usize).clamp(1, 10_000);

        let start_time = std::time::Instant::now();
        let results = self
            .db
            .search_similar(vector, top_k)
            .await
            .map_err(Self::map_error)?;

        let grpc_results = results
            .into_iter()
            .map(|result| SimilarityResult {
                id: result.id.clone(),
                score: result.score,
                metadata: Some(VectorMetadata {
                    id: result.metadata.id,
                    dimension: result.metadata.dimension as u64,
                    created_at: result.metadata.created_at,
                    updated_at: result.metadata.updated_at,
                    tags: result.metadata.tags,
                }),
            })
            .collect();

        Ok(Response::new(SearchResponse {
            results: grpc_results,
            total_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        }))
    }

    async fn list_vectors(
        &self,
        _request: Request<ListVectorsRequest>,
    ) -> Result<Response<ListVectorsResponse>, Status> {
        let ids = self.db.list_vectors().await.map_err(Self::map_error)?;
        Ok(Response::new(ListVectorsResponse { ids }))
    }

    async fn get_stats(
        &self,
        _request: Request<GetStatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        let stats = self.db.get_stats().await.map_err(Self::map_error)?;
        Ok(Response::new(StatsResponse {
            total_vectors: stats.total_vectors as u64,
            dimension: stats.dimension as u64,
            memory_usage: stats.memory_usage,
        }))
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        Ok(Response::new(HealthCheckResponse {
            status: "healthy".to_string(),
            service: "vectradb-grpc".to_string(),
        }))
    }
}
