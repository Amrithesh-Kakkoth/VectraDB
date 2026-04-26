# VectraDB Python Client

Python client library for VectraDB vector database. This client uses gRPC to communicate with the Rust backend server.

## Installation

```bash
pip install vectradb-client
```

The package ships checked-in protobuf stubs, so importing the client does not require local proto generation.

Or install from source:

```bash
cd python-client
pip install -e .
```

## Quick Start

```python
from vectradb_client import VectraDBClient

# Connect to the VectraDB server
client = VectraDBClient(host="localhost", port=50051)

# Create a vector
vector = client.create_vector(
    id="vec1",
    vector=[0.1, 0.2, 0.3],
    tags={"type": "example", "count": "42"}
)

# Search for similar vectors
response = client.search_similar(
    vector=[0.1, 0.2, 0.3],
    top_k=10,
)

for result in response.results:
    print(f"ID: {result.id}, Score: {result.score}")

# Get database statistics
stats = client.get_stats()
print(f"Total vectors: {stats.total_vectors}")

# Clean up
client.close()
```

## Async Support

```python
from vectradb_client import AsyncVectraDBClient

async def main():
    async with AsyncVectraDBClient(host="localhost", port=50051) as client:
        # Create a vector
        await client.create_vector(
            id="vec1",
            vector=[0.1, 0.2, 0.3],
            tags={"type": "example"}
        )
        
        # Search
        response = await client.search_similar(vector=[0.1, 0.2, 0.3], top_k=10)
        for result in response.results:
            print(f"ID: {result.id}, Score: {result.score}")
```

## API Reference

### VectraDBClient

#### Methods

- `create_vector(id: str, vector: List[float], tags: Optional[Dict[str, str]] = None) -> Vector`
  - Create a new vector and return the stored document
  
- `get_vector(id: str) -> Vector`
  - Retrieve a vector by ID
  
- `update_vector(id: str, vector: List[float], tags: Optional[Dict[str, str]] = None) -> Vector`
  - Update an existing vector and return the stored document
  
- `delete_vector(id: str) -> DeleteResult`
  - Delete a vector by ID
  
- `upsert_vector(id: str, vector: List[float], tags: Optional[Dict[str, str]] = None) -> Vector`
  - Create or update a vector (upsert operation)
  
- `search_similar(vector: List[float], top_k: int = 10) -> SearchResponse`
  - Search for the nearest neighbors of the query vector

- `search(query: List[float], k: int = 10) -> List[SearchResult]`
  - Compatibility alias that returns only `SearchResponse.results`
  
- `list_vectors() -> List[str]`
  - List all stored vector IDs
  
- `get_stats() -> DatabaseStats`
  - Get database statistics (total vectors, memory usage, etc.)
  
- `health_check() -> HealthStatus`
  - Check if the server is healthy and responding

## Requirements

- Python 3.8+
- VectraDB server running with gRPC enabled (`--enable-grpc` flag)

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/
```

Format code:

```bash
black vectradb_client/
```

Type checking:

```bash
mypy vectradb_client/
```

## License

MIT License
