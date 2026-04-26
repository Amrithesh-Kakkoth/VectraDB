"""Synchronous VectraDB client implementation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import grpc

from . import vectradb_pb2, vectradb_pb2_grpc
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


def _normalize_tags(
    tags: Optional[Mapping[str, object]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, str]:
    source = tags if tags is not None else metadata
    if source is None:
        return {}
    return {str(key): str(value) for key, value in source.items()}


def _resolve_vector(
    vector: Optional[Sequence[float]],
    alias: Optional[Sequence[float]],
    *,
    primary_name: str,
    alias_name: str,
) -> List[float]:
    if vector is not None and alias is not None:
        raise TypeError(f"Pass either '{primary_name}' or '{alias_name}', not both.")
    if vector is None:
        vector = alias
    if vector is None:
        raise TypeError(f"Missing required vector argument '{primary_name}'.")
    return list(vector)


def _vector_from_response(response: vectradb_pb2.VectorResponse) -> Vector:
    return Vector(
        id=response.id,
        vector=list(response.vector),
        dimension=response.dimension,
        created_at=response.created_at,
        updated_at=response.updated_at,
        tags=dict(response.tags),
    )


def _metadata_from_proto(
    metadata: Optional[vectradb_pb2.VectorMetadata],
) -> Optional[VectorMetadata]:
    if metadata is None:
        return None
    return VectorMetadata(
        id=metadata.id,
        dimension=metadata.dimension,
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
        tags=dict(metadata.tags),
    )


def _search_results_from_proto(
    results: Iterable[vectradb_pb2.SimilarityResult],
) -> List[SearchResult]:
    return [
        SearchResult(
            id=result.id,
            score=result.score,
            metadata=_metadata_from_proto(result.metadata),
        )
        for result in results
    ]


def _search_response_from_proto(
    response: vectradb_pb2.SearchResponse,
) -> SearchResponse:
    return SearchResponse(
        results=_search_results_from_proto(response.results),
        total_time_ms=response.total_time_ms,
    )


def _delete_result_from_proto(
    response: vectradb_pb2.DeleteVectorResponse,
) -> DeleteResult:
    return DeleteResult(success=bool(response.success))


def _batch_write_response_from_proto(
    response: vectradb_pb2.BatchWriteResponse,
) -> BatchWriteResponse:
    return BatchWriteResponse(
        statuses=[
            BatchWriteItemStatus(
                id=status.id,
                code=status.code,
                message=status.message,
            )
            for status in response.statuses
        ]
    )


def _health_status_from_proto(
    response: vectradb_pb2.HealthCheckResponse,
) -> HealthStatus:
    return HealthStatus(
        status=response.status,
        service=getattr(response, "service", ""),
    )


class VectraDBClient:
    """Synchronous gRPC client for VectraDB."""

    def __init__(self, host: str = "localhost", port: int = 50051, timeout: int = 30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.address = f"{host}:{port}"
        self.channel = grpc.insecure_channel(self.address)
        self.stub = vectradb_pb2_grpc.VectraDbStub(self.channel)

    def __enter__(self) -> "VectraDBClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self.channel:
            self.channel.close()

    def _rpc(self, rpc_name: str, request):
        try:
            rpc = getattr(self.stub, rpc_name)
            return rpc(request, timeout=self.timeout)
        except grpc.RpcError as exc:
            raise VectraDBError(f"gRPC error: {exc.details()}") from exc

    @staticmethod
    def _vector_input(
        id: str,
        vector: Sequence[float],
        tags: Optional[Mapping[str, object]] = None,
    ) -> vectradb_pb2.VectorInput:
        return vectradb_pb2.VectorInput(
            id=id,
            vector=list(vector),
            tags=_normalize_tags(tags),
        )

    def create_vector(
        self,
        id: str,
        vector: Optional[Sequence[float]] = None,
        tags: Optional[Mapping[str, object]] = None,
        *,
        values: Optional[Sequence[float]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Vector:
        request = vectradb_pb2.CreateVectorRequest(
            id=id,
            vector=_resolve_vector(
                vector,
                values,
                primary_name="vector",
                alias_name="values",
            ),
            tags=_normalize_tags(tags, metadata),
        )
        response = self._rpc("CreateVector", request)
        return _vector_from_response(response)

    def get_vector(self, id: str) -> Vector:
        request = vectradb_pb2.GetVectorRequest(id=id)
        response = self._rpc("GetVector", request)
        return _vector_from_response(response)

    def update_vector(
        self,
        id: str,
        vector: Optional[Sequence[float]] = None,
        tags: Optional[Mapping[str, object]] = None,
        *,
        values: Optional[Sequence[float]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Vector:
        if vector is None and values is None:
            current = self.get_vector(id)
            vector = current.vector
            if metadata is None and tags is None:
                tags = current.tags

        request = vectradb_pb2.UpdateVectorRequest(
            id=id,
            vector=_resolve_vector(
                vector,
                values,
                primary_name="vector",
                alias_name="values",
            ),
            tags=_normalize_tags(tags, metadata),
        )
        response = self._rpc("UpdateVector", request)
        return _vector_from_response(response)

    def delete_vector(self, id: str) -> DeleteResult:
        request = vectradb_pb2.DeleteVectorRequest(id=id)
        response = self._rpc("DeleteVector", request)
        return _delete_result_from_proto(response)

    def upsert_vector(
        self,
        id: str,
        vector: Optional[Sequence[float]] = None,
        tags: Optional[Mapping[str, object]] = None,
        *,
        values: Optional[Sequence[float]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Vector:
        request = vectradb_pb2.UpsertVectorRequest(
            id=id,
            vector=_resolve_vector(
                vector,
                values,
                primary_name="vector",
                alias_name="values",
            ),
            tags=_normalize_tags(tags, metadata),
        )
        response = self._rpc("UpsertVector", request)
        return _vector_from_response(response)

    def batch_create_vectors(
        self,
        items: Sequence[VectorInput],
    ) -> BatchWriteResponse:
        request = vectradb_pb2.BatchCreateVectorsRequest(
            items=[
                self._vector_input(item.id, item.vector, item.tags)
                for item in items
            ]
        )
        response = self._rpc("BatchCreateVectors", request)
        return _batch_write_response_from_proto(response)

    def batch_upsert_vectors(
        self,
        items: Sequence[VectorInput],
    ) -> BatchWriteResponse:
        request = vectradb_pb2.BatchUpsertVectorsRequest(
            items=[
                self._vector_input(item.id, item.vector, item.tags)
                for item in items
            ]
        )
        response = self._rpc("BatchUpsertVectors", request)
        return _batch_write_response_from_proto(response)

    def search_similar(
        self,
        vector: Optional[Sequence[float]] = None,
        top_k: int = 10,
        *,
        query: Optional[Sequence[float]] = None,
        k: Optional[int] = None,
    ) -> SearchResponse:
        if k is not None:
            top_k = k
        request = vectradb_pb2.SearchRequest(
            vector=_resolve_vector(
                vector,
                query,
                primary_name="vector",
                alias_name="query",
            ),
            top_k=top_k,
        )
        response = self._rpc("SearchSimilar", request)
        return _search_response_from_proto(response)

    def search(
        self,
        query: Sequence[float],
        k: int = 10,
    ) -> List[SearchResult]:
        return self.search_similar(query=query, k=k).results

    def list_vectors(self) -> List[str]:
        request = vectradb_pb2.ListVectorsRequest()
        response = self._rpc("ListVectors", request)
        return list(response.ids)

    def get_stats(self) -> DatabaseStats:
        request = vectradb_pb2.GetStatsRequest()
        response = self._rpc("GetStats", request)
        return DatabaseStats(
            total_vectors=response.total_vectors,
            dimension=response.dimension,
            memory_usage=response.memory_usage,
        )

    def health_check(self) -> HealthStatus:
        request = vectradb_pb2.HealthCheckRequest()
        response = self._rpc("HealthCheck", request)
        return _health_status_from_proto(response)

    def is_healthy(self) -> bool:
        return bool(self.health_check())
