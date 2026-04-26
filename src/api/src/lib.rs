use axum::{
    extract::{DefaultBodyLimit, Path, State},
    http::StatusCode,
    middleware,
    response::Json,
    routing::{get, post, put},
    Router,
};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use vectradb_components::{
    filter::{FilterCondition, MetadataFilter},
    similarity::find_similar_vectors_cosine,
    BatchWriteResponse as CoreBatchWriteResponse, DatabaseStats, SimilarityResult, VectorDocument,
    VectorInput as CoreVectorInput, VectraDBError,
};
use vectradb_storage::{DatabaseConfig, PersistentVectorDB};

pub mod auth;
pub mod metrics;
pub mod rate_limit;
pub use auth::AuthConfig;
pub use rate_limit::{RateLimitConfig, RateLimiter};

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<PersistentVectorDB>,
    pub embedder: Option<Arc<dyn vectradb_embeddings::EmbeddingProvider>>,
    pub auth: Arc<AuthConfig>,
    pub rate_limiter: Arc<RateLimiter>,
    pub metrics_handle: Option<metrics_exporter_prometheus::PrometheusHandle>,
    #[cfg(feature = "gpu")]
    pub gpu: Option<Arc<vectradb_search::gpu::GpuDistanceEngine>>,
    pub tfidf: Option<Arc<RwLock<vectradb_tfidf::TfIdfIndex>>>,
    pub rag_pipeline: Option<Arc<vectradb_rag::RagPipeline>>,
    pub graph_agent: Option<Arc<vectradb_rag::GraphAgent>>,
}

#[derive(Debug, Deserialize)]
pub struct CreateVectorRequest {
    pub id: String,
    pub vector: Vec<f32>,
    pub tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateVectorRequest {
    pub vector: Vec<f32>,
    pub tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct UpsertVectorRequest {
    pub vector: Vec<f32>,
    pub tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct VectorInput {
    pub id: String,
    pub vector: Vec<f32>,
    pub tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct BatchWriteRequest {
    pub items: Vec<VectorInput>,
}

#[derive(Debug, Deserialize)]
pub struct TagCondition {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Deserialize)]
pub struct SearchFilter {
    pub must: Option<Vec<TagCondition>>,
    pub must_not: Option<Vec<TagCondition>>,
    pub should: Option<Vec<TagCondition>>,
}

impl SearchFilter {
    fn to_metadata_filter(&self) -> Option<MetadataFilter> {
        let mut parts = Vec::new();

        if let Some(must) = &self.must {
            for condition in must {
                parts.push(MetadataFilter::Condition(FilterCondition::Equals {
                    key: condition.key.clone(),
                    value: condition.value.clone(),
                }));
            }
        }

        if let Some(must_not) = &self.must_not {
            for condition in must_not {
                parts.push(MetadataFilter::Condition(FilterCondition::NotEquals {
                    key: condition.key.clone(),
                    value: condition.value.clone(),
                }));
            }
        }

        if let Some(should) = &self.should {
            if !should.is_empty() {
                parts.push(MetadataFilter::Or(
                    should
                        .iter()
                        .map(|condition| {
                            MetadataFilter::Condition(FilterCondition::Equals {
                                key: condition.key.clone(),
                                value: condition.value.clone(),
                            })
                        })
                        .collect(),
                ));
            }
        }

        match parts.len() {
            0 => None,
            1 => Some(parts.remove(0)),
            _ => Some(MetadataFilter::And(parts)),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub top_k: Option<usize>,
    pub ef_search: Option<usize>,
    pub filter: Option<SearchFilter>,
}

#[derive(Debug, Serialize)]
pub struct VectorResponse {
    pub id: String,
    pub vector: Vec<f32>,
    pub dimension: usize,
    pub created_at: u64,
    pub updated_at: u64,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct BatchWriteItemStatus {
    pub id: String,
    pub code: String,
    pub message: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BatchWriteResponse {
    pub statuses: Vec<BatchWriteItemStatus>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SimilarityResult>,
    pub total_time_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
}

fn classify_error(error: &VectraDBError) -> (StatusCode, &'static str) {
    match error {
        VectraDBError::VectorAlreadyExists { .. } | VectraDBError::DuplicateVector { .. } => {
            (StatusCode::CONFLICT, "Vector already exists")
        }
        VectraDBError::VectorNotFound { .. } => (StatusCode::NOT_FOUND, "Vector not found"),
        VectraDBError::DimensionMismatch { .. } | VectraDBError::InvalidVector => {
            (StatusCode::BAD_REQUEST, "Invalid vector input")
        }
        VectraDBError::DatabaseError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Database error"),
    }
}

fn vector_response_from_document(document: VectorDocument) -> VectorResponse {
    VectorResponse {
        id: document.metadata.id,
        vector: document.data.to_vec(),
        dimension: document.metadata.dimension,
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
                message: if item.message.is_empty() {
                    None
                } else {
                    Some(item.message)
                },
            })
            .collect(),
    }
}

fn validate_request_vector(values: &[f32]) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if values.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid vector".to_string(),
                message: "Vector must not be empty".to_string(),
            }),
        ));
    }

    if values.iter().any(|value| !value.is_finite()) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid vector".to_string(),
                message: "Vector contains NaN or infinite values".to_string(),
            }),
        ));
    }

    Ok(())
}

async fn search_with_optional_filter(
    db: &PersistentVectorDB,
    query_vector: Array1<f32>,
    top_k: usize,
    filter: Option<&MetadataFilter>,
) -> Result<Vec<SimilarityResult>, VectraDBError> {
    if let Some(filter) = filter {
        // Exact filtered fallback until index-level filtering lands on the v2 engine.
        let ids = db.list_vectors().await?;
        let mut documents = Vec::new();
        for id in ids {
            let document = db.get_vector(&id).await?;
            if filter.matches(&document.metadata.tags) {
                documents.push(document);
            }
        }
        return find_similar_vectors_cosine(&query_vector.view(), &documents, top_k);
    }

    db.search_similar(query_vector, top_k).await
}

pub fn create_router(state: AppState) -> Router {
    let auth_state = state.auth.clone();
    let rate_limiter = state.rate_limiter.clone();

    Router::new()
        .route("/health", get(health_check))
        .route("/stats", get(get_stats))
        .route("/vectors", post(create_vector).get(list_vectors))
        .route("/vectors/batch/create", post(batch_create_vectors))
        .route("/vectors/batch/upsert", put(batch_upsert_vectors))
        .route(
            "/vectors/:id",
            get(get_vector).put(update_vector).delete(delete_vector),
        )
        .route("/vectors/:id/upsert", put(upsert_vector))
        .route("/search", post(search_vectors))
        .route("/metrics", get(metrics::metrics_handler))
        .layer(middleware::from_fn(metrics::metrics_middleware))
        .layer(middleware::from_fn_with_state(
            auth_state,
            auth::auth_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            rate_limiter,
            rate_limit::rate_limit_middleware,
        ))
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([
                    axum::http::Method::GET,
                    axum::http::Method::POST,
                    axum::http::Method::PUT,
                    axum::http::Method::DELETE,
                ])
                .allow_headers(Any),
        )
        .with_state(state)
}

async fn health_check() -> Result<Json<HashMap<String, String>>, StatusCode> {
    let mut response = HashMap::new();
    response.insert("status".to_string(), "healthy".to_string());
    response.insert("service".to_string(), "vectradb-api".to_string());
    Ok(Json(response))
}

async fn get_stats(
    State(state): State<AppState>,
) -> Result<Json<DatabaseStats>, (StatusCode, Json<ErrorResponse>)> {
    state.db.get_stats().await.map(Json).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Database error".to_string(),
                message: error.to_string(),
            }),
        )
    })
}

async fn create_vector(
    State(state): State<AppState>,
    Json(request): Json<CreateVectorRequest>,
) -> Result<Json<VectorResponse>, (StatusCode, Json<ErrorResponse>)> {
    validate_request_vector(&request.vector)?;
    let id = request.id.clone();
    let vector = Array1::from_vec(request.vector);

    match state
        .db
        .create_vector(id.clone(), vector, request.tags)
        .await
    {
        Ok(()) => state
            .db
            .get_vector(&id)
            .await
            .map(vector_response_from_document)
            .map(Json)
            .map_err(|error| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Failed to fetch created vector".to_string(),
                        message: error.to_string(),
                    }),
                )
            }),
        Err(error) => {
            let (status, label) = classify_error(&error);
            Err((
                status,
                Json(ErrorResponse {
                    error: label.to_string(),
                    message: error.to_string(),
                }),
            ))
        }
    }
}

async fn batch_create_vectors(
    State(state): State<AppState>,
    Json(request): Json<BatchWriteRequest>,
) -> Result<Json<BatchWriteResponse>, (StatusCode, Json<ErrorResponse>)> {
    let items = request
        .items
        .into_iter()
        .map(|item| {
            if item.vector.is_empty() || item.vector.iter().any(|value| !value.is_finite()) {
                Err((item.id, "Invalid vector payload".to_string()))
            } else {
                Ok(CoreVectorInput {
                    id: item.id,
                    vector: item.vector,
                    tags: item.tags.unwrap_or_default(),
                })
            }
        })
        .collect::<Result<Vec<_>, _>>();

    let items = match items {
        Ok(items) => items,
        Err((id, message)) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Batch write failed".to_string(),
                    message: format!("{message}: {id}"),
                }),
            ));
        }
    };

    state
        .db
        .batch_create_vectors(items)
        .await
        .map(batch_response)
        .map(Json)
        .map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Batch write failed".to_string(),
                    message: error.to_string(),
                }),
            )
        })
}

async fn batch_upsert_vectors(
    State(state): State<AppState>,
    Json(request): Json<BatchWriteRequest>,
) -> Result<Json<BatchWriteResponse>, (StatusCode, Json<ErrorResponse>)> {
    let items = request
        .items
        .into_iter()
        .map(|item| {
            if item.vector.is_empty() || item.vector.iter().any(|value| !value.is_finite()) {
                Err((item.id, "Invalid vector payload".to_string()))
            } else {
                Ok(CoreVectorInput {
                    id: item.id,
                    vector: item.vector,
                    tags: item.tags.unwrap_or_default(),
                })
            }
        })
        .collect::<Result<Vec<_>, _>>();

    let items = match items {
        Ok(items) => items,
        Err((id, message)) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Batch write failed".to_string(),
                    message: format!("{message}: {id}"),
                }),
            ));
        }
    };

    state
        .db
        .batch_upsert_vectors(items)
        .await
        .map(batch_response)
        .map(Json)
        .map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Batch write failed".to_string(),
                    message: error.to_string(),
                }),
            )
        })
}

async fn get_vector(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<VectorResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.db.get_vector(&id).await {
        Ok(document) => Ok(Json(vector_response_from_document(document))),
        Err(error) => {
            let (status, label) = classify_error(&error);
            Err((
                status,
                Json(ErrorResponse {
                    error: label.to_string(),
                    message: error.to_string(),
                }),
            ))
        }
    }
}

async fn update_vector(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<UpdateVectorRequest>,
) -> Result<Json<VectorResponse>, (StatusCode, Json<ErrorResponse>)> {
    validate_request_vector(&request.vector)?;
    let vector = Array1::from_vec(request.vector);

    match state.db.update_vector(&id, vector, request.tags).await {
        Ok(()) => state
            .db
            .get_vector(&id)
            .await
            .map(vector_response_from_document)
            .map(Json)
            .map_err(|error| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Failed to fetch updated vector".to_string(),
                        message: error.to_string(),
                    }),
                )
            }),
        Err(error) => {
            let (status, label) = classify_error(&error);
            Err((
                status,
                Json(ErrorResponse {
                    error: label.to_string(),
                    message: error.to_string(),
                }),
            ))
        }
    }
}

async fn delete_vector(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    match state.db.delete_vector(&id).await {
        Ok(()) => Ok(StatusCode::NO_CONTENT),
        Err(error) => {
            let (status, label) = classify_error(&error);
            Err((
                status,
                Json(ErrorResponse {
                    error: label.to_string(),
                    message: error.to_string(),
                }),
            ))
        }
    }
}

async fn upsert_vector(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<UpsertVectorRequest>,
) -> Result<Json<VectorResponse>, (StatusCode, Json<ErrorResponse>)> {
    validate_request_vector(&request.vector)?;
    let vector = Array1::from_vec(request.vector);

    match state
        .db
        .upsert_vector(id.clone(), vector, request.tags)
        .await
    {
        Ok(()) => state
            .db
            .get_vector(&id)
            .await
            .map(vector_response_from_document)
            .map(Json)
            .map_err(|error| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Failed to fetch upserted vector".to_string(),
                        message: error.to_string(),
                    }),
                )
            }),
        Err(error) => {
            let (status, label) = classify_error(&error);
            Err((
                status,
                Json(ErrorResponse {
                    error: label.to_string(),
                    message: error.to_string(),
                }),
            ))
        }
    }
}

async fn search_vectors(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    validate_request_vector(&request.vector)?;
    let vector = Array1::from_vec(request.vector);
    let top_k = request.top_k.unwrap_or(10).clamp(1, 10000);
    let metadata_filter = request
        .filter
        .as_ref()
        .and_then(|filter| filter.to_metadata_filter());
    let start_time = std::time::Instant::now();

    let results = search_with_optional_filter(&state.db, vector, top_k, metadata_filter.as_ref())
        .await
        .map_err(|error| {
            let (status, label) = classify_error(&error);
            (
                status,
                Json(ErrorResponse {
                    error: label.to_string(),
                    message: error.to_string(),
                }),
            )
        })?;

    metrics::record_search_query();

    Ok(Json(SearchResponse {
        results,
        total_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
    }))
}

async fn list_vectors(
    State(state): State<AppState>,
) -> Result<Json<Vec<String>>, (StatusCode, Json<ErrorResponse>)> {
    state.db.list_vectors().await.map(Json).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Database error".to_string(),
                message: error.to_string(),
            }),
        )
    })
}

pub async fn start_server(
    config: DatabaseConfig,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let db = Arc::new(PersistentVectorDB::new(config).await?);
    let state = AppState {
        db,
        embedder: None,
        auth: Arc::new(AuthConfig::disabled()),
        rate_limiter: Arc::new(RateLimiter::new(RateLimitConfig::disabled())),
        metrics_handle: None,
        #[cfg(feature = "gpu")]
        gpu: None,
        tfidf: None,
        rag_pipeline: None,
        graph_agent: None,
    };

    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    println!("VectraDB API server running on http://0.0.0.0:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectradb_components::BatchWriteItemStatus as CoreBatchWriteItemStatus;

    #[tokio::test]
    async fn test_batch_write_request_shape() {
        let request = BatchWriteRequest {
            items: vec![VectorInput {
                id: "test".to_string(),
                vector: vec![1.0, 2.0, 3.0],
                tags: Some(HashMap::from([("kind".to_string(), "test".to_string())])),
            }],
        };

        assert_eq!(request.items.len(), 1);
        assert_eq!(request.items[0].id, "test");
    }

    #[test]
    fn test_batch_status_mapping() {
        let response = batch_response(CoreBatchWriteResponse {
            statuses: vec![CoreBatchWriteItemStatus::error(
                "id",
                "ALREADY_EXISTS",
                "exists",
            )],
        });
        assert_eq!(response.statuses[0].code, "ALREADY_EXISTS");
    }
}
