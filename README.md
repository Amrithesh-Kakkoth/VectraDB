<p align="center">
  <img src="https://img.shields.io/badge/VectraDB-Vector_Database-8B5CF6?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsMTAgNSAxMC01TTIgMTJsMTAgNSAxMC01Ii8+PC9zdmc+&logoColor=white" alt="VectraDB" />
</p>

<h1 align="center">VectraDB</h1>

<p align="center">
  <strong>High-performance vector database built in Rust</strong><br>
  Store, search, and manage high-dimensional vectors with sub-millisecond query times.
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-search-algorithms">Algorithms</a> •
  <a href="#-rest-api">REST API</a> •
  <a href="#-grpc-api">gRPC API</a> •
  <a href="#-python-client">Python Client</a> •
  <a href="ARCHITECTURE.md">Architecture</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" />
  <img src="https://img.shields.io/badge/rust-1.70+-orange.svg" alt="Rust" />
  <img src="https://img.shields.io/badge/API-REST_%2B_gRPC-10b981.svg" alt="API" />
  <img src="https://img.shields.io/badge/storage-Sled-e11d48.svg" alt="Storage" />
  <img src="https://img.shields.io/badge/docker-ready-2563eb.svg" alt="Docker" />
</p>

---

## 🧠 What is a Vector Database?

Modern AI models convert text, images, and other data into **vectors** — arrays of numbers that capture meaning. Similar items produce similar vectors. A vector database lets you:

1. **Store** millions of these vectors with metadata
2. **Search** for the most similar vectors to a query (nearest neighbor search)
3. **Retrieve** the original data associated with each result

> [!TIP]
> **Real-world example:** You embed all your documents using an AI model, store the embeddings in VectraDB, then search for documents similar to a user's question. This is how RAG (Retrieval-Augmented Generation) systems work.

---

## ✨ Features

- 🔍 **4 Search Algorithms** — HNSW, ES4D, LSH, and Product Quantization
- 🌐 **Dual API** — REST (HTTP) and gRPC running concurrently
- 💾 **Persistent Storage** — Data survives restarts via Sled embedded database
- 🐍 **Python Client** — Sync and async gRPC client library
- 📄 **Text Chunking** — Built-in utilities for splitting documents, code, and markdown
- 🐳 **Docker Ready** — Multi-stage Dockerfile included
- ⚡ **High Performance** — Sub-millisecond search, 5,000+ gRPC req/s

---

## 🚀 Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) 1.70+
- [protoc](https://grpc.io/docs/protoc-installation/) (Protocol Buffers compiler)

```bash
# macOS
brew install protobuf

# Ubuntu/Debian
sudo apt install protobuf-compiler
```

### Build and Run

```bash
git clone https://github.com/Amrithesh-Kakkoth/VectraDB.git
cd VectraDB

cargo build --release

./target/release/vectradb-server --enable-grpc
```

The server starts on:
| Service | Address |
|---------|---------|
| HTTP REST API | `http://localhost:8080` |
| gRPC API | `localhost:50051` |

### Your First Vectors

```bash
# 1. Health check
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

# 6. Database stats
curl http://localhost:8080/stats
```

> [!NOTE]
> The vector dimension must match the server's configured dimension (default: 384). The examples above use 5 dimensions for brevity — start the server with `-d 5` to try them.

---

## 🔬 Search Algorithms

| Algorithm | Speed | Memory | Accuracy | Best For |
|-----------|:-----:|:------:|:--------:|----------|
| **HNSW** | ⚡ Fast | 🔴 High | 🟢 High | General purpose (default) |
| **ES4D** | ⚡ Fast | 🔴 High | 🟢 Exact | High-dimensional vectors |
| **LSH** | 🟡 Medium | 🟢 Low | 🟡 Approx | Large datasets, low memory |
| **PQ** | ⚡ Fast | 🟢 Very Low | 🟡 Approx | Huge datasets, memory-critical |

### HNSW (default)

Hierarchical Navigable Small World graph. Best balance of speed and accuracy.

```bash
./target/release/vectradb-server -a hnsw --max-connections 16 --construction-ef 200
```

### ES4D

Our implementation of the [ES4D paper](https://doi.org/10.1109/ICCD56317.2022.00051), adapted for HNSW graph navigation. Three optimizations on top of HNSW:

- **DET** — Computes distance in chunks; skips remaining dimensions when partial distance exceeds cutoff
- **Dimension Reordering** — High-variance dimensions first, so DET triggers earlier
- **CET** — Pre-clusters vectors; skips entire clusters that can't contain results

```bash
./target/release/vectradb-server -a es4d --shard-length 64
```

### LSH & PQ

```bash
# LSH — hash-based approximate search
./target/release/vectradb-server -a lsh --num-hashes 10

# PQ — memory-efficient compressed search
./target/release/vectradb-server -a pq
```

---

## 🌐 REST API

All endpoints return JSON. Error responses include `error` and `message` fields.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Database statistics |
| `POST` | `/vectors` | Create a vector |
| `GET` | `/vectors` | List all vector IDs |
| `GET` | `/vectors/:id` | Get a vector by ID |
| `PUT` | `/vectors/:id` | Update a vector |
| `DELETE` | `/vectors/:id` | Delete a vector |
| `PUT` | `/vectors/:id/upsert` | Create or update |
| `POST` | `/search` | Search for similar vectors |

### Create a Vector

```http
POST /vectors
Content-Type: application/json

{
  "id": "my-vector-1",
  "vector": [0.1, 0.2, 0.3, ...],
  "tags": {"category": "article", "author": "jane"}
}
```

### Search for Similar Vectors

```http
POST /search
Content-Type: application/json

{
  "vector": [0.1, 0.2, 0.3, ...],
  "top_k": 10
}
```

**Response:**

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

---

## 📡 gRPC API

The gRPC API provides the same functionality with better performance. See [`proto/vectradb.proto`](proto/vectradb.proto) for the full schema.

```bash
# List services
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

> [!TIP]
> Install grpcurl from [github.com/fullstorydev/grpcurl](https://github.com/fullstorydev/grpcurl)

---

## 🐍 Python Client

```bash
cd python-client
pip install grpcio grpcio-tools protobuf
python generate_proto.py
pip install -e .
```

```python
from vectradb_simple import VectraDB

with VectraDB(host="localhost", port=50051) as client:
    # Store vectors
    client.create("doc1", [0.1, 0.2, 0.3], {"type": "article"})
    client.create("doc2", [0.2, 0.3, 0.4], {"type": "article"})

    # Search
    results = client.search([0.15, 0.25, 0.35], k=10)
    for r in results.results:
        print(f"  {r.id}: score={r.score:.4f}")

    # Stats
    stats = client.stats()
    print(f"Total vectors: {stats.total_vectors}")
```

See [`python-client/README.md`](python-client/README.md) for full documentation.

---

## 🦀 Rust Library

```rust
use vectradb_components::{VectorDatabase, InMemoryVectorDB};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut db = InMemoryVectorDB::new();

    let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    db.create_vector("doc1".to_string(), vector, None)?;

    let query = Array1::from_vec(vec![1.1, 2.1, 3.1]);
    let results = db.search_similar(query, 5)?;

    for result in &results {
        println!("{}: score={:.4}", result.id, result.score);
    }
    Ok(())
}
```

---

## 🐳 Docker

```bash
# Build
docker build -t vectradb .

# Run
docker run -p 8080:8080 -p 50051:50051 vectradb

# Run with persistent data
docker run -p 8080:8080 -p 50051:50051 \
  -v ./data:/data \
  vectradb --enable-grpc -d 384 -D /data
```

---

## ⚙️ Server Configuration

```
./target/release/vectradb-server [OPTIONS]

Options:
  -d, --dimension <DIM>          Vector dimension [default: 384]
  -D, --data-dir <DIR>           Data directory [default: ./vectradb_data]
  -p, --port <PORT>              HTTP port [default: 8080]
      --grpc-port <PORT>         gRPC port [default: 50051]
      --enable-grpc              Enable gRPC server [default: true]
  -a, --algorithm <ALGO>         hnsw | lsh | pq | es4d [default: hnsw]
      --max-connections <N>      HNSW max connections [default: 16]
      --search-ef <N>            HNSW search ef [default: 50]
      --construction-ef <N>      HNSW construction ef [default: 200]
      --shard-length <N>         ES4D shard length [default: 64]
      --auto-flush               Flush to disk after writes [default: true]
```

---

## 📁 Project Structure

```
VectraDB/
├── src/
│   ├── components/       Core types, similarity math, vector operations
│   ├── search/           Search algorithms (HNSW, LSH, PQ, ES4D)
│   ├── storage/          Sled-based persistent storage
│   ├── api/              Axum REST API handlers
│   ├── server/           Server binary (HTTP + gRPC)
│   └── chunkers/         Text chunking utilities
├── proto/                Protocol Buffer definitions
├── python-client/        Python gRPC client library
├── src_py/               PyO3 native Python bindings
├── bench/                Benchmarking scripts
└── .github/workflows/    CI/CD (build, test, release, Docker)
```

---

## 📊 Benchmarks

| Metric | gRPC | REST |
|--------|:----:|:----:|
| Search throughput (dim=64, k=10, N=50k) | **5,000–8,000** req/s | **1,000–2,000** req/s |
| p95 latency (concurrency=200) | **< 20 ms** | **< 50 ms** |

See [BENCHMARKS.md](BENCHMARKS.md) for how to reproduce.

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/Amrithesh-Kakkoth/VectraDB.git
cd VectraDB
cargo build && cargo test
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](ARCHITECTURE.md) | System design, crate diagram, request flow |
| [Contributing](CONTRIBUTING.md) | Dev setup, code style, PR workflow |
| [Benchmarks](BENCHMARKS.md) | Performance testing methodology |
| [Python Client](python-client/README.md) | Python gRPC client docs |
| [Chunkers](src/chunkers/README.md) | Text chunking utilities |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with 🦀 Rust and ❤️ by the <a href="https://github.com/Amrithesh-Kakkoth/VectraDB">VectraDB</a> team
</p>
