use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use tonic::transport::Server;
use vectradb_search::{DistanceMetric, SearchAlgorithm};
use vectradb_storage::{DatabaseConfig, DurabilityMode, PersistentVectorDB};

mod grpc;
use grpc::VectraDbService;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    serve: ServeArgs,
}

#[derive(Subcommand, Debug)]
enum Command {
    Migrate(MigrateArgs),
}

#[derive(Args, Debug, Clone)]
struct ServeArgs {
    #[arg(short = 'D', long, default_value = "./vectradb_data")]
    data_dir: PathBuf,

    #[arg(short, long, default_value = "8080")]
    port: u16,

    #[arg(long, default_value = "50051")]
    grpc_port: u16,

    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    enable_grpc: bool,

    #[arg(short, long, default_value = "hnsw")]
    algorithm: String,

    #[arg(short = 'd', long, default_value = "384")]
    dimension: usize,

    #[arg(long, default_value = "16")]
    max_connections: usize,

    #[arg(long, default_value = "50")]
    search_ef: usize,

    #[arg(long, default_value = "200")]
    construction_ef: usize,

    #[arg(long, default_value = "10")]
    num_hashes: usize,

    #[arg(long, default_value = "1000")]
    num_buckets: usize,

    #[arg(long, default_value = "64")]
    shard_length: usize,

    #[arg(long, default_value = "euclidean")]
    metric: String,

    #[arg(long, default_value = "256")]
    ivf_nlist: usize,

    #[arg(long, default_value = "16")]
    ivf_nprobe: usize,

    #[arg(long, default_value = "batch")]
    durability_mode: String,

    #[arg(long, default_value = "10")]
    commit_interval_ms: u64,

    #[arg(long, default_value = "2000")]
    commit_max_vectors: usize,

    #[arg(long, default_value = "4194304")]
    commit_max_bytes: usize,

    #[arg(long, default_value = "50000")]
    seal_max_vectors: usize,

    #[arg(long, default_value = "67108864")]
    seal_max_bytes: usize,

    #[arg(long, default_value = "30000")]
    seal_max_age_ms: u64,

    #[arg(long, default_value = "1000")]
    cache_size: usize,

    #[arg(long)]
    embedding_provider: Option<String>,

    #[arg(long, default_value = "nomic-embed-text")]
    embedding_model: String,

    #[arg(long)]
    embedding_url: Option<String>,

    #[arg(long)]
    embedding_api_key: Option<String>,

    #[arg(long)]
    api_key: Vec<String>,

    #[arg(long)]
    api_key_readonly: Vec<String>,

    #[arg(long)]
    tls_cert: Option<PathBuf>,

    #[arg(long)]
    tls_key: Option<PathBuf>,

    #[arg(long, default_value = "0")]
    rate_limit: f64,

    #[arg(long, default_value = "100")]
    rate_burst: u32,
}

#[derive(Args, Debug)]
struct MigrateArgs {
    #[arg(long)]
    from: PathBuf,

    #[arg(long)]
    to: PathBuf,

    #[arg(short = 'd', long, default_value = "384")]
    dimension: usize,

    #[arg(short, long, default_value = "hnsw")]
    algorithm: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    env_logger::init();

    match cli.command {
        Some(Command::Migrate(args)) => run_migrate(args).await,
        None => run_server(cli.serve).await,
    }
}

async fn run_migrate(args: MigrateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let algorithm = parse_algorithm(&args.algorithm)?;
    let config = DatabaseConfig {
        data_dir: args.to.to_string_lossy().to_string(),
        search_algorithm: algorithm,
        index_config: vectradb_search::SearchConfig {
            algorithm,
            dimension: Some(args.dimension),
            ..vectradb_search::SearchConfig::default()
        },
        ..DatabaseConfig::default()
    };

    PersistentVectorDB::migrate_legacy_data(&args.from, &args.to, config).await?;
    println!(
        "Migrated legacy storage from {} to {}",
        args.from.display(),
        args.to.display()
    );
    Ok(())
}

async fn run_server(args: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let search_algorithm = parse_algorithm(&args.algorithm)?;
    let metric = parse_metric(&args.metric)?;

    let config = DatabaseConfig {
        data_dir: args.data_dir.to_string_lossy().to_string(),
        search_algorithm,
        index_config: vectradb_search::SearchConfig {
            algorithm: search_algorithm,
            max_connections: args.max_connections,
            search_ef: args.search_ef,
            construction_ef: args.construction_ef,
            m: args.max_connections,
            ef_construction: args.construction_ef,
            num_hashes: args.num_hashes,
            num_buckets: args.num_buckets,
            dimension: Some(args.dimension),
            num_subspaces: Some(8),
            codes_per_subspace: Some(256),
            shard_length: Some(args.shard_length),
            metric,
            ivf_nlist: Some(args.ivf_nlist),
            ivf_nprobe: Some(args.ivf_nprobe),
        },
        durability_mode: parse_durability(&args.durability_mode)?,
        commit_interval_ms: args.commit_interval_ms,
        commit_max_vectors: args.commit_max_vectors,
        commit_max_bytes: args.commit_max_bytes,
        segment_max_vectors: args.seal_max_vectors,
        segment_max_bytes: args.seal_max_bytes,
        segment_max_age_secs: (args.seal_max_age_ms.saturating_add(999)) / 1000,
        cache_size: args.cache_size,
    };

    println!("Starting VectraDB server...");
    println!("Data directory: {}", config.data_dir);
    println!("Search algorithm: {:?}", config.search_algorithm);
    println!("Vector dimension: {}", args.dimension);
    println!("Durability mode: {:?}", config.durability_mode);
    println!("HTTP port: {}", args.port);
    if args.enable_grpc {
        println!("gRPC port: {}", args.grpc_port);
    }

    let db = Arc::new(PersistentVectorDB::new(config.clone()).await?);

    let embedder: Option<Arc<dyn vectradb_embeddings::EmbeddingProvider>> =
        if let Some(provider_name) = &args.embedding_provider {
            let emb_config = vectradb_embeddings::EmbeddingConfig {
                provider: provider_name.clone(),
                model: args.embedding_model.clone(),
                api_url: args.embedding_url.clone(),
                api_key: args.embedding_api_key.clone(),
                dimension: Some(args.dimension),
            };

            match vectradb_embeddings::create_provider(&emb_config) {
                Ok(provider) => {
                    println!(
                        "Embedding provider: {} (model: {})",
                        provider.provider_name(),
                        provider.model_name()
                    );
                    Some(Arc::from(provider))
                }
                Err(error) => {
                    eprintln!("Failed to initialize embedding provider: {error}");
                    std::process::exit(1);
                }
            }
        } else {
            None
        };

    let mut admin_keys = args.api_key;
    if let Ok(env_key) = std::env::var("VECTRADB_API_KEY") {
        admin_keys.push(env_key);
    }
    let mut readonly_keys = args.api_key_readonly;
    if let Ok(env_key) = std::env::var("VECTRADB_API_KEY_READONLY") {
        readonly_keys.push(env_key);
    }
    let auth_config = Arc::new(vectradb_api::AuthConfig::new(admin_keys, readonly_keys));
    if auth_config.enabled {
        println!("API key authentication: enabled");
    }

    #[cfg(feature = "gpu")]
    let gpu_engine: Option<Arc<vectradb_search::gpu::GpuDistanceEngine>> = {
        match vectradb_search::gpu::GpuDistanceEngine::new(100_000) {
            Some(engine) => {
                println!("GPU acceleration: enabled (wgpu)");
                Some(Arc::new(engine))
            }
            None => {
                println!("GPU acceleration: no adapter found, disabled");
                None
            }
        }
    };

    let tls_config = match (&args.tls_cert, &args.tls_key) {
        (Some(cert), Some(key)) => {
            println!(
                "TLS: enabled (cert={}, key={})",
                cert.display(),
                key.display()
            );
            Some((cert.clone(), key.clone()))
        }
        (Some(_), None) | (None, Some(_)) => {
            eprintln!("Error: both --tls-cert and --tls-key must be provided together");
            std::process::exit(1);
        }
        (None, None) => {
            println!("TLS: disabled (use --tls-cert and --tls-key to enable)");
            None
        }
    };

    let rate_config = vectradb_api::RateLimitConfig::new(args.rate_limit, args.rate_burst);
    if rate_config.enabled {
        println!(
            "Rate limiting: {} req/s per IP (burst: {})",
            args.rate_limit, args.rate_burst
        );
    } else {
        println!("Rate limiting: disabled (use --rate-limit to enable)");
    }
    let rate_limiter = Arc::new(vectradb_api::RateLimiter::new(rate_config));

    let http_db = db.clone();
    let http_embedder = embedder.clone();
    let http_auth = auth_config.clone();
    let http_rate_limiter = rate_limiter.clone();
    #[cfg(feature = "gpu")]
    let http_gpu = gpu_engine.clone();
    let http_port = args.port;
    let http_tls = tls_config.clone();

    let metrics_handle = vectradb_api::metrics::install_prometheus_recorder();
    println!("Prometheus metrics: enabled at /metrics");

    let shutdown = async {
        let ctrl_c = tokio::signal::ctrl_c();

        #[cfg(unix)]
        {
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("failed to register SIGTERM handler");
            tokio::select! {
                _ = ctrl_c => {},
                _ = sigterm.recv() => {},
            }
        }

        #[cfg(not(unix))]
        {
            ctrl_c.await.ok();
        }

        println!("\nShutdown signal received. Draining in-flight requests...");
    };

    let shutdown_notify = Arc::new(tokio::sync::Notify::new());
    let http_shutdown = shutdown_notify.clone();
    let grpc_shutdown = shutdown_notify.clone();

    let http_handle = tokio::spawn(async move {
        let state = vectradb_api::AppState {
            db: http_db,
            embedder: http_embedder,
            auth: http_auth,
            rate_limiter: http_rate_limiter,
            metrics_handle: Some(metrics_handle),
            #[cfg(feature = "gpu")]
            gpu: http_gpu,
            tfidf: None,
            rag_pipeline: None,
            graph_agent: None,
        };
        let app = vectradb_api::create_router(state);
        let addr = format!("0.0.0.0:{http_port}");

        if let Some((cert_path, key_path)) = http_tls {
            let tls =
                match axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert_path, &key_path)
                    .await
                {
                    Ok(config) => config,
                    Err(error) => {
                        eprintln!("Failed to load TLS cert/key: {error}");
                        return;
                    }
                };

            println!("VectraDB HTTPS server running on https://{addr}");
            if let Err(error) = axum_server::bind_rustls(addr.parse().unwrap(), tls)
                .serve(app.into_make_service())
                .await
            {
                eprintln!("HTTPS server error: {error}");
            }
        } else {
            let listener = match tokio::net::TcpListener::bind(&addr).await {
                Ok(listener) => listener,
                Err(error) => {
                    eprintln!("Failed to bind HTTP port {http_port}: {error}");
                    return;
                }
            };

            println!("VectraDB HTTP server running on http://{addr}");
            if let Err(error) = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    http_shutdown.notified().await;
                })
                .await
            {
                eprintln!("HTTP server error: {error}");
            }
        }
    });

    if args.enable_grpc {
        let grpc_addr = format!("0.0.0.0:{}", args.grpc_port).parse()?;
        let grpc_service = VectraDbService::new(db.clone()).into_service();

        let grpc_handle = tokio::spawn(async move {
            let mut builder = Server::builder();

            if let Some((cert_path, key_path)) = tls_config {
                let cert_pem = std::fs::read_to_string(&cert_path).unwrap_or_else(|error| {
                    eprintln!("Failed to read TLS cert for gRPC: {error}");
                    std::process::exit(1);
                });
                let key_pem = std::fs::read_to_string(&key_path).unwrap_or_else(|error| {
                    eprintln!("Failed to read TLS key for gRPC: {error}");
                    std::process::exit(1);
                });
                let tls = tonic::transport::ServerTlsConfig::new()
                    .identity(tonic::transport::Identity::from_pem(cert_pem, key_pem));
                builder = builder.tls_config(tls).unwrap_or_else(|error| {
                    eprintln!("Failed to configure gRPC TLS: {error}");
                    std::process::exit(1);
                });
                println!("VectraDB gRPC-TLS server running on {grpc_addr}");
            } else {
                println!("VectraDB gRPC server running on {grpc_addr}");
            }

            if let Err(error) = builder
                .add_service(grpc_service)
                .serve_with_shutdown(grpc_addr, async move {
                    grpc_shutdown.notified().await;
                })
                .await
            {
                eprintln!("gRPC server error: {error}");
            }
        });

        shutdown.await;
        shutdown_notify.notify_waiters();
        let _ = tokio::join!(http_handle, grpc_handle);
    } else {
        shutdown.await;
        shutdown_notify.notify_waiters();
        let _ = http_handle.await;
    }

    println!("VectraDB shutdown complete.");
    Ok(())
}

fn parse_algorithm(value: &str) -> Result<SearchAlgorithm, Box<dyn std::error::Error>> {
    match value.to_lowercase().as_str() {
        "hnsw" => Ok(SearchAlgorithm::HNSW),
        _ => Err(format!(
            "Invalid algorithm: {value}. The v2 storage engine currently supports only hnsw."
        )
        .into()),
    }
}

fn parse_metric(value: &str) -> Result<DistanceMetric, Box<dyn std::error::Error>> {
    match value.to_lowercase().as_str() {
        "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
        "cosine" => Ok(DistanceMetric::Cosine),
        "dot" | "dot_product" | "ip" => Ok(DistanceMetric::DotProduct),
        _ => Err(
            format!("Invalid metric: {value}. Supported metrics: euclidean, cosine, dot").into(),
        ),
    }
}

fn parse_durability(value: &str) -> Result<DurabilityMode, Box<dyn std::error::Error>> {
    match value.to_lowercase().as_str() {
        "batch" => Ok(DurabilityMode::Batch),
        "async" => Ok(DurabilityMode::Async),
        "strict" => Ok(DurabilityMode::Strict),
        _ => Err(format!(
            "Invalid durability mode: {value}. Supported modes: batch, async, strict"
        )
        .into()),
    }
}
