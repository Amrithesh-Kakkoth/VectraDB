<h1 align="center">🤝 Contributing to VectraDB</h1>

<p align="center">
  Thank you for your interest in contributing!<br/>
  This guide will help you get started.
</p>

---

## 🛠️ Development Setup

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Rust | 1.70+ | [rustup.rs](https://rustup.rs/) |
| protoc | any | `brew install protobuf` / `apt install protobuf-compiler` |
| Git | any | [git-scm.com](https://git-scm.com/) |

### First Time Setup

```bash
# Clone
git clone https://github.com/Amrithesh-Kakkoth/VectraDB.git
cd VectraDB

# Build
cargo build

# Test
cargo test -p vectradb-components -p vectradb-search -p vectradb-storage \
           -p vectradb-chunkers -p vectradb-api

# Lint (must pass with zero warnings)
cargo clippy --workspace -- -D warnings

# Format
cargo fmt --all
```

> [!NOTE]
> The `vectradb-py` crate requires a Python development installation to build. It's safe to skip it during development — all other crates build independently.

---

## 🔄 Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write code
- Add tests for new functionality
- Run `cargo fmt --all` before committing
- Run `cargo clippy --workspace -- -D warnings`

### 3. Test

```bash
# All tests
cargo test -p vectradb-components -p vectradb-search -p vectradb-storage \
           -p vectradb-chunkers -p vectradb-api

# Specific crate
cargo test -p vectradb-search

# Specific test
cargo test -p vectradb-search -- es4d::tests::test_es4d_insert_and_search
```

### 4. Submit a Pull Request

```bash
git push -u origin feature/your-feature-name
```

Then open a PR on GitHub with:
- What you changed and why
- Any related issues

---

## 📏 Code Style

| Rule | Details |
|------|---------|
| **Formatting** | `cargo fmt --all` — enforced by CI |
| **Linting** | `cargo clippy -- -D warnings` — zero warnings |
| **Naming** | `snake_case` functions, `PascalCase` types, `SCREAMING_SNAKE` constants |
| **Errors** | Return `Result`, use `thiserror` for types, `anyhow` for ad-hoc |
| **Unsafe** | Avoid. If necessary, document why it's safe. |
| **Tests** | Required for new functionality |

---

## 📁 Where Things Live

```
src/components/   ← Start here to understand the codebase
src/search/       ← Search algorithms (HNSW, LSH, PQ, ES4D)
src/storage/      ← Persistent storage (Sled + search index)
src/api/          ← REST API (Axum)
src/server/       ← Server binary (HTTP + gRPC)
src/chunkers/     ← Text chunking utilities
src_py/           ← PyO3 Python bindings
proto/            ← Protocol Buffer definitions
python-client/    ← Python gRPC client
```

> [!TIP]
> Read [ARCHITECTURE.md](ARCHITECTURE.md) for a visual overview of how components interact.

---

## ➕ Adding a New Search Algorithm

1. Create `src/search/src/my_algo.rs`
2. Implement the `AdvancedSearch` trait
3. Add `pub mod my_algo;` and re-export in `src/search/src/lib.rs`
4. Add a variant to `SearchAlgorithm` enum
5. Add a match arm in `src/storage/src/lib.rs` → `PersistentVectorDB::new()`
6. Add CLI parsing in `src/server/src/main.rs`
7. Add tests
8. Update docs

---

## 💡 Ideas for Contributions

- 🧮 **New distance metrics** — Hamming, Jaccard, etc.
- 🔎 **Metadata filtering** — Filter search results by tags
- 📦 **Batch operations** — Bulk insert/delete endpoints
- 🌊 **gRPC streaming** — Stream large batch operations
- 📊 **Prometheus metrics** — `/metrics` endpoint
- 🟨 **JavaScript client** — Similar to the Python client
- 📈 **Standard benchmarks** — SIFT, GloVe dataset comparisons

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the **MIT License**.

---

<p align="center">
  <a href="README.md">← Back to README</a> •
  <a href="ARCHITECTURE.md">Architecture →</a>
</p>
