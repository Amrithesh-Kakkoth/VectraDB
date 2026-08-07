

<p align="center">
  <img src="docs/assets/banner.png" alt="VectraDB Banner" />
</p>

<h1 align="center">VectraDB</h1>

<p align="center">
  <strong>Base de datos vectorial de alto rendimiento construida en Rust</strong>
</p>

<p align="center">
  <a href="#features">Características</a> •
  <a href="#quick-start">Inicio Rápido</a> •
  <a href="#search-algorithms">Algoritmos</a> •
  <a href="#rest-api-reference">Referencia de la API</a> •
  <a href="#python-client">Cliente de Python</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/rust-stable-orange.svg" alt="Rust" />
  <a href="https://github.com/Amrithesh-Kakkoth/VectraDB"><img src="https://img.shields.io/github/stars/Amrithesh-Kakkoth/VectraDB" alt="Stars" /></a>
  <a href="https://deepwiki.com/Amrithesh-Kakkoth/VectraDB"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" /></a>
</p>

---

Una base de datos vectorial de alto rendimiento construida en Rust. Úsala como una **biblioteca incrustada** (como SQLite) o como un **servidor independiente** con APIs REST + gRPC.

VectraDB está diseñada para cargas de trabajo de IA/ML como búsqueda semántica, motores de recomendación y pipelines RAG. Admite 7 algoritmos de búsqueda, cálculo de distancia acelerado por SIMD, filtrado de metadatos, soporte integrado para modelos de incrustación (embeddings) y búsqueda de tensores multidimensionales.

## Características

- **7 Algoritmos de búsqueda** — HNSW, ES4D, IVF, SQ8, LSH, PQ y TensorSearch
- **Dos modos** — Biblioteca en proceso (sin servidor) o servidor independiente (REST + gRPC)
- **Aceleración SIMD** — Intrínsecos AVX2, SSE y NEON para el cálculo de distancias
- **Filtrado de metadatos** — Filtros de etiquetas `must` / `must_not` / `should` durante la búsqueda
- **Modelos de incrustación (embeddings)** — Integración integrada con Ollama, OpenAI, HuggingFace, Cohere
- **Almacenamiento persistente** — Backed by Sled, seguro ante fallos, sobrevive a los reinicios
- **Listo para producción** — Autenticación, TLS, limitación de velocidad, métricas de Prometheus, apagado elegante
- **Búsqueda de tensores** — Búsqueda nativa de similitud 2D/3D/nD (no solo vectores 1D)

## Inicio Rápido

### Opción 1: Como biblioteca (no se necesita servidor)

Agrega a tu `Cargo.toml`:
```toml
[dependencies]
vectradb = "0.1"
tokio = { version = "1", features = ["full"] }
```

```rust
use vectradb::VectraDB;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open or create a database
    let mut db = VectraDB::open_with_dim("./my_vectors", 4).await?;

    // Insert vectors with metadata
    db.insert("doc1", &[0.1, 0.2, 0.3, 0.4], None)?;
    db.insert("doc2", &[0.2, 0.3, 0.4, 0.5], None)?;

    // Search for similar vectors
    let results = db.search(&[0.15, 0.25, 0.35, 0.45], 5)?;
    for r in &results {
        println!("{}: score={:.4}", r.id, r.score);
    }

    // Data persists across restarts automatically
    Ok(())
}
```

Configuración avanzada mediante builder:
```rust
use vectradb::{VectraDB, SearchAlgorithm, DistanceMetric};

let db = VectraDB::builder("./my_db")
    .dimension(384)
    .algorithm(SearchAlgorithm::HNSW)
    .metric(DistanceMetric::Cosine)
    .hnsw_m(32)
    .hnsw_ef_construction(200)
    .build()
    .await?;
```

### Opción 2: Como servidor (REST + gRPC)

```bash
# Prerequisites: Rust 1.70+, protoc
# macOS: brew install protobuf
# Ubuntu: sudo apt install protobuf-compiler

git clone https://github.com/Amrithesh-Kakkoth/VectraDB.git
cd VectraDB && cargo build --release

# Start server
./target/release/vectradb-server --enable-grpc -d 384
```

El servidor se inicia en:
- HTTP REST API: `http://localhost:8080`
- gRPC API: `localhost:50051`

### Tus primeros vectores

Una vez que el servidor esté en ejecución, prueba estos comandos en una nueva terminal:

```bash
# 1. Check the server is running
curl http://localhost:8080/health

# 2. Store a vector
curl -X POST http://localhost:8080/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
    "tags": {"title": "Hello World", "source": "demo"}
  }'

# 3. Store another vector
curl -X POST http://localhost:8080/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc2",
    "vector": [0.15, 0.22, 0.28, 0.41, 0.52],
    "tags": {"title": "Similar Document", "source": "demo"}
  }'

# 4. Search for similar vectors
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.12, 0.21, 0.29, 0.42, 0.51], "top_k": 5}'

# 5. Get a specific vector
curl http://localhost:8080/vectors/doc1

# 6. View database stats
curl http://localhost:8080/stats

# 7. Delete a vector
curl -X DELETE http://localhost:8080/vectors/doc1
```

> **Nota:** La dimensión del vector en tus solicitudes debe coincidir con la dimensión configurada del servidor (predeterminado: 384). Los ejemplos anteriores usan vectores de 5 dimensiones por brevedad: inicia el servidor con `-d 5` para usar estos ejemplos, o usa vectores de 384 dimensiones.

### Configuración del servidor

```bash
./target/release/vectradb-server [OPTIONS]

Options:
  -d, --dimension <DIM>          Vector dimension [default: 384]
  -D, --data-dir <DIR>           Data directory [default: ./vectradb_data]
  -p, --port <PORT>              HTTP port [default: 8080]
      --grpc-port <PORT>         gRPC port [default: 50051]
      --enable-grpc              Enable gRPC server [default: true]
  -a, --algorithm <ALGO>         Search algorithm: hnsw, lsh, pq, es4d [default: hnsw]
      --max-connections <N>      HNSW max connections [default: 16]
      --search-ef <N>            HNSW search ef [default: 50]
      --construction-ef <N>      HNSW construction ef [default: 200]
      --shard-length <N>         ES4D shard length for DET [default: 64]
      --auto-flush               Flush to disk after each write [default: true]
```

## Algoritmos de búsqueda

VectraDB admite 7 algoritmos de búsqueda. Todos usan cálculo de distancia acelerado por SIMD (AVX2/SSE/NEON).

| Algoritmo | Velocidad | Memoria | Recall@10 | Mejor para |
|-----------|:-----:|:------:|:---------:|---------|
| **HNSW** | ⚡ Rápido | Alta | 98.5% | Propósito general (predeterminado) |
| **ES4D** | ⚡ Rápido | Alta | 100% | Búsqueda exacta de alta dimensión |
| **IVF** | ⚡⚡ Muy rápido | Alta | 73%* | Conjuntos de datos grandes (1M+ vectores) |
| **SQ8** | Medio | **4x más pequeño** | 100% | Despliegues con restricciones de memoria |
| **LSH** | Medio | Baja | 60% | Búsqueda aproximada basada en hash |
| **PQ** | Rápido | Muy baja | 35% | Compresión extrema |
| **TensorSearch** | — | — | — | Coincidencia de patrones 2D/3D/nD |

*El recall de IVF depende de `nprobe` (clústeres buscados). Un nprobe más alto = mayor recall.

### HNSW (predeterminado)

Grafo Hierarchical Navigable Small World. El mejor equilibrio entre velocidad y precisión para la mayoría de las cargas de trabajo.

```bash
./target/release/vectradb-server -a hnsw --max-connections 16 --construction-ef 200
```

### ES4D

Nuestra implementación del [paper ES4D](https://doi.org/10.1109/ICCD56317.2022.00051), adaptada para usar navegación de grafo HNSW. Añade tres optimizaciones sobre HNSW:

- **DET (Terminación Temprana a Nivel de Dimensión)**: Calcula la distancia en bloques. Si la distancia parcial ya supera el límite, omite el resto, ahorrando CPU en vectores de alta dimensión.
- **Reordenamiento de Dimensiones**: Coloca primero las dimensiones de alta varianza para que DET se active antes.
- **CET (Terminación Temprana a Nivel de Clúster)**: Agrupa previamente los vectores y omite clústeres completos que no pueden contener resultados.

```bash
./target/release/vectradb-server -a es4d --shard-length 64
```

### IVF (Índice de Archivo Invertido)

Particiona los vectores en clústeres, solo busca en los clústeres más cercanos por consulta. Con 1M de vectores y nprobe=10, busca ~1% de los datos.

```bash
./target/release/vectradb-server -a ivf --ivf-nlist 256 --ivf-nprobe 16
```

### SQ8 (Cuantización Escalar)

Comprime cada dimensión de f32 a uint8. **Reducción de memoria de 4x** con prácticamente ninguna pérdida de recall.

```bash
./target/release/vectradb-server -a sq
```

### LSH y PQ

```bash
# LSH — hash-based approximate search
./target/release/vectradb-server -a lsh --num-hashes 10

# PQ — extreme compression (k-means++ trained codebooks)
./target/release/vectradb-server -a pq
```

## Referencia de la API REST

Todos los endpoints devuelven JSON. Las respuestas de error incluyen un campo `error` y `message`.

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Verificación de estado |
| `GET` | `/stats` | Estadísticas de la base de datos |
| `POST` | `/vectors` | Crear un vector |
| `GET` | `/vectors` | Listar todos los IDs de vectores |
| `GET` | `/vectors/:id` | Obtener un vector por ID |
| `PUT` | `/vectors/:id` | Actualizar un vector |
| `DELETE` | `/vectors/:id` | Eliminar un vector |
| `PUT` | `/vectors/:id/upsert` | Crear o actualizar un vector |
| `POST` | `/search` | Buscar vectores similares |

### Crear un Vector

```
POST /vectors
Content-Type: application/json

{
  "id": "my-vector-1",
  "vector": [0.1, 0.2, 0.3, ...],
  "tags": {
    "category": "article",
    "author": "jane"
  }
}
```

- `id` (string, requerido): Identificador único
- `vector` (array de floats, requerido): Debe coincidir con la dimensión configurada
- `tags` (object, opcional): Metadatos clave-valor

### Buscar Vectores Similares

```
POST /search
Content-Type: application/json

{
  "vector": [0.1, 0.2, 0.3, ...],
  "top_k": 10
}
```

- `vector` (array de floats, requerido): Vector de consulta
- `top_k` (entero, opcional): Número de resultados a devolver (predeterminado: 10, máximo: 10000)

**Respuesta:**

```json
{
  "results": [
    {
      "id": "doc1",
      "score": 0.95,
      "metadata": {
        "id": "doc1",
        "dimension": 384,
        "created_at": 1700000000,
        "updated_at": 1700000000,
        "tags": {"category": "article"}
      }
    }
  ],
  "total_time_ms": 0.42
}
```

## API gRPC

La API gRPC proporciona la misma funcionalidad con mejor rendimiento y seguridad de tipos. Consulta [`proto/vectradb.proto`](proto/vectradb.proto) para el esquema completo.

### Pruebas con grpcurl

```bash
# Install grpcurl: https://github.com/fullstorydev/grpcurl

# List available services
grpcurl -plaintext localhost:50051 list

# Health check
grpcurl -plaintext localhost:50051 vectradb.VectraDb/HealthCheck

# Create a vector
grpcurl -plaintext -d '{
  "id": "test1",
  "vector": [0.1, 0.2, 0.3],
  "tags": {"type": "test"}
}' localhost:50051 vectradb.VectraDb/CreateVector

# Search
grpcurl -plaintext -d '{
  "vector": [0.1, 0.2, 0.3],
  "top_k": 5
}' localhost:50051 vectradb.VectraDb/SearchSimilar
```

## Cliente de Python

VectraDB incluye un cliente gRPC de Python para una fácil integración con aplicaciones de Python.

### Configuración

```bash
cd python-client
pip install grpcio grpcio-tools protobuf
python generate_proto.py   # Generate gRPC stubs
pip install -e .
```

### Uso

```python
from vectradb_simple import VectraDB

# Connect (server must be running with --enable-grpc)
with VectraDB(host="localhost", port=50051) as client:
    # Store vectors
    client.create("doc1", [0.1, 0.2, 0.3], {"type": "article"})
    client.create("doc2", [0.2, 0.3, 0.4], {"type": "article"})

    # Search
    results = client.search([0.15, 0.25, 0.35], k=10)
    for r in results.results:
        print(f"  {r.id}: score={r.score:.4f}")

    # Get stats
    stats = client.stats()
    print(f"Total vectors: {stats.total_vectors}")

    # CRUD operations
    vec = client.get("doc1")
    client.update("doc1", [0.11, 0.21, 0.31], {"type": "updated"})
    client.delete("doc2")
```

Consulta [`python-client/README.md`](python-client/README.md) para la documentación completa del cliente de Python.

## Uso de la Biblioteca Rust

También puedes usar VectraDB como una biblioteca Rust en tus propios proyectos:

```rust
use vectradb_components::{VectorDatabase, InMemoryVectorDB};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an in-memory database
    let mut db = InMemoryVectorDB::new();

    // Insert a vector
    let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    db.create_vector("doc1".to_string(), vector, None)?;

    // Search for similar vectors
    let query = Array1::from_vec(vec![1.1, 2.1, 3.1]);
    let results = db.search_similar(query, 5)?;

    for result in &results {
        println!("{}: score={:.4}", result.id, result.score);
    }

    Ok(())
}
```

## Docker

```bash
# Build the image
docker build -t vectradb .

# Run with default settings
docker run -p 8080:8080 -p 50051:50051 vectradb

# Run with custom settings and persistent data
docker run -p 8080:8080 -p 50051:50051 \
  -v ./data:/data \
  vectradb --enable-grpc -d 384 -D /data -a hnsw
```

## Estructura del Proyecto

```
VectraDB/
├── src/
│   ├── vectradb/         In-process library API (use without server)
│   ├── components/       Core types, similarity math, vector operations
│   ├── search/           Search algorithms (HNSW, ES4D, IVF, SQ, LSH, PQ)
│   │   └── simd.rs       SIMD-accelerated distance functions
│   ├── storage/          Sled-based persistent storage
│   ├── api/              Axum REST API + auth + rate limiting + metrics
│   ├── server/           Server binary (HTTP + gRPC + TLS)
│   ├── embeddings/       Embedding providers (Ollama, OpenAI, HF, Cohere)
│   ├── chunkers/         Text chunking utilities
│   ├── tfidf/            TF-IDF sparse text retrieval
│   ├── rag/              RAG pipeline
│   └── eval/             Evaluation framework
├── proto/                Protocol Buffer definitions
├── python-client/        Python gRPC client library
├── tests/                Integration & stress tests (119+ tests)
└── .github/workflows/    CI/CD (build, test, release, Docker)
```

Para una visión general detallada de la arquitectura, consulta [ARCHITECTURE.md](ARCHITECTURE.md).

## Benchmarks (Pruebas de Rendimiento)

Resultados típicos en un CPU de 8 núcleos con 32 GB de RAM:

| Métrica | gRPC | REST |
|--------|------|------|
| Rendimiento de búsqueda (dim=64, k=10, N=50k) | 5,000-8,000 req/s | 1,000-2,000 req/s |
| Latencia p95 (concurrencia=200) | < 20 ms | < 50 ms |

Consulta [BENCHMARKS.md](BENCHMARKS.md) para saber cómo reproducir estos números.

## Contribuciones

¡Damos la bienvenida a las contribuciones! Consulta [CONTRIBUTING.md](CONTRIBUTING.md) para las directrices.

```bash
# Development setup
git clone https://github.com/Amrithesh-Kakkoth/VectraDB.git
cd VectraDB
cargo build
cargo test
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

## Licencia

Licencia MIT. Consulta [LICENSE](LICENSE) para más detalles.

## Enlaces

- [Repositorio en GitHub](https://github.com/Amrithesh-Kakkoth/VectraDB)
- [Guía de Arquitectura](ARCHITECTURE.md)
- [Referencia de la API](ARCHITECTURE.md#api-layer)
- [Pruebas de Rendimiento (Benchmarks)](BENCHMARKS.md)
- [Contribuciones](CONTRIBUTING.md)
- [Cliente de Python](python-client/README.md)
