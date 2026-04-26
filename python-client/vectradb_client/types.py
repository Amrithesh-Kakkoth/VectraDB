"""Type definitions for the proto-aligned VectraDB client."""

from dataclasses import dataclass
from typing import Dict, List, Optional


class VectraDBError(Exception):
    """Base exception for VectraDB client errors."""


@dataclass
class VectorMetadata:
    """Metadata returned by the VectraDB API."""

    id: str
    dimension: int
    created_at: int
    updated_at: int
    tags: Dict[str, str]


@dataclass
class Vector:
    """A stored vector document."""

    id: str
    vector: List[float]
    dimension: int
    created_at: int
    updated_at: int
    tags: Dict[str, str]

    @property
    def values(self) -> List[float]:
        """Backward-compatible alias for ``vector``."""
        return self.vector

    @property
    def metadata(self) -> Dict[str, str]:
        """Backward-compatible alias for ``tags``."""
        return self.tags


@dataclass
class SearchResult:
    """A search hit returned by the API."""

    id: str
    score: float
    metadata: Optional[VectorMetadata] = None

    @property
    def distance(self) -> float:
        """Approximate compatibility alias for older client code."""
        return 1.0 - self.score


@dataclass
class SearchResponse:
    """Search results returned by the API."""

    results: List[SearchResult]
    total_time_ms: float


@dataclass
class VectorInput:
    """Input item for batch write APIs."""

    id: str
    vector: List[float]
    tags: Dict[str, str]


@dataclass
class BatchWriteItemStatus:
    """Per-item status returned by batch write APIs."""

    id: str
    code: str
    message: str

    @property
    def ok(self) -> bool:
        return self.code == "OK"


@dataclass
class BatchWriteResponse:
    """Order-preserving batch write response."""

    statuses: List[BatchWriteItemStatus]


@dataclass
class DeleteResult:
    """Delete response returned by the API."""

    success: bool

    def __bool__(self) -> bool:
        return self.success


@dataclass
class HealthStatus:
    """Health check response returned by the API."""

    status: str
    service: str

    @property
    def healthy(self) -> bool:
        return self.status == "healthy"

    def __bool__(self) -> bool:
        return self.healthy


@dataclass
class DatabaseStats:
    """Database statistics returned by the API."""

    total_vectors: int
    dimension: int
    memory_usage: int

    @property
    def memory_usage_bytes(self) -> int:
        """Backward-compatible alias for ``memory_usage``."""
        return self.memory_usage

    @property
    def memory_usage_mb(self) -> float:
        return self.memory_usage / (1024 * 1024)
