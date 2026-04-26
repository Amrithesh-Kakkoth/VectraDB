//! Prometheus metrics middleware and `/metrics` endpoint.
//!
//! Exposes standard HTTP metrics plus VectraDB-specific gauges:
//!
//! **HTTP metrics (per request):**
//! - `vectradb_http_requests_total` — counter by method, path, status
//! - `vectradb_http_request_duration_seconds` — histogram of latency
//!
//! **Database metrics (gauges):**
//! - `vectradb_vectors_total` — current number of stored vectors
//! - `vectradb_search_queries_total` — total search queries processed
//! - `vectradb_rate_limit_rejected_total` — requests rejected by rate limiter
//!
//! # Usage
//! ```bash
//! curl http://localhost:8080/metrics
//! ```
//!
//! Output is Prometheus text format, ready for scraping.

use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::time::Instant;

use crate::AppState;

/// Install the Prometheus metrics recorder.
/// Must be called once at startup before any metrics are recorded.
/// Returns a handle used by the `/metrics` endpoint to render output.
pub fn install_prometheus_recorder() -> PrometheusHandle {
    PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder")
}

/// GET /metrics — render all metrics in Prometheus text format.
pub async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    // Update database gauges on each scrape
    if let Ok(stats) = state.db.get_stats().await {
        gauge!("vectradb_vectors_total").set(stats.total_vectors as f64);
        gauge!("vectradb_dimension").set(stats.dimension as f64);
        gauge!("vectradb_memory_usage_bytes").set(stats.memory_usage as f64);
    }

    let handle = state
        .metrics_handle
        .as_ref()
        .expect("metrics not initialized");
    let body = handle.render();

    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        body,
    )
}

/// Axum middleware that records per-request metrics.
///
/// Tracks:
/// - `vectradb_http_requests_total{method, path, status}` — request count
/// - `vectradb_http_request_duration_seconds{method, path}` — latency histogram
pub async fn metrics_middleware(request: Request<Body>, next: Next) -> Response {
    let method = request.method().to_string();
    let path = normalize_path(request.uri().path());
    let start = Instant::now();

    let response = next.run(request).await;

    let status = response.status().as_u16().to_string();
    let duration = start.elapsed().as_secs_f64();

    counter!(
        "vectradb_http_requests_total",
        "method" => method.clone(),
        "path" => path.clone(),
        "status" => status,
    )
    .increment(1);

    histogram!(
        "vectradb_http_request_duration_seconds",
        "method" => method,
        "path" => path,
    )
    .record(duration);

    response
}

/// Normalize path to avoid high-cardinality labels.
/// `/vectors/abc123` → `/vectors/:id`
fn normalize_path(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').collect();

    if parts.len() >= 3 && parts[1] == "vectors" && parts[2] != "text" && parts[2] != "batch" {
        if parts.len() == 3 {
            return "/vectors/:id".to_string();
        }
        if parts.len() == 4 && parts[3] == "upsert" {
            return "/vectors/:id/upsert".to_string();
        }
    }

    path.to_string()
}

/// Record a search query metric.
pub fn record_search_query() {
    counter!("vectradb_search_queries_total").increment(1);
}

/// Record a rate limit rejection.
pub fn record_rate_limit_rejection() {
    counter!("vectradb_rate_limit_rejected_total").increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_normalization() {
        assert_eq!(normalize_path("/health"), "/health");
        assert_eq!(normalize_path("/stats"), "/stats");
        assert_eq!(normalize_path("/vectors"), "/vectors");
        assert_eq!(normalize_path("/vectors/abc123"), "/vectors/:id");
        assert_eq!(
            normalize_path("/vectors/abc123/upsert"),
            "/vectors/:id/upsert"
        );
        assert_eq!(normalize_path("/search"), "/search");
        assert_eq!(normalize_path("/search/text"), "/search/text");
        assert_eq!(normalize_path("/vectors/text"), "/vectors/text");
        assert_eq!(normalize_path("/vectors/batch"), "/vectors/batch");
        assert_eq!(normalize_path("/metrics"), "/metrics");
    }
}
