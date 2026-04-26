"""VectraDB Python Client Library

A Python client for VectraDB vector database using gRPC.
"""

from .client import VectraDBClient
from .async_client import AsyncVectraDBClient
from .types import (
    BatchWriteItemStatus,
    BatchWriteResponse,
    DatabaseStats,
    DeleteResult,
    HealthStatus,
    SearchResponse,
    SearchResult,
    VectorInput,
    Vector,
    VectorMetadata,
    VectraDBError,
)

__version__ = "0.1.0"
__all__ = [
    "VectraDBClient",
    "AsyncVectraDBClient",
    "VectorInput",
    "Vector",
    "VectorMetadata",
    "SearchResult",
    "SearchResponse",
    "BatchWriteItemStatus",
    "BatchWriteResponse",
    "DeleteResult",
    "HealthStatus",
    "DatabaseStats",
    "VectraDBError",
]
