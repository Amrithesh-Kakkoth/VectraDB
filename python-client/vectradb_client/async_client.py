"""Asynchronous VectraDB client implementation."""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

import grpc
import grpc.aio

from . import vectradb_pb2, vectradb_pb2_grpc
from .client import (
    _batch_write_response_from_proto,
    _delete_result_from_proto,
    _health_status_from_proto,
    _normalize_tags,
    _resolve_vector,
    _search_response_from_proto,
    _search_results_from_proto,
    _vector_from_response,
)
from .types import (
    BatchWriteResponse,
    DatabaseStats,
    DeleteResult,
    HealthStatus,
    SearchResponse,
    SearchResult,
    VectorInput,
    Vector,
    VectraDBError,
)


class AsyncVectraDBClient:
    """Async gRPC client for VectraDB."""

    def __init__(self, host: str = "localhost", port: int = 50051, timeout: int = 30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.address = f"{host}:{port}"
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub = None

    async def __aenter__(self) -> "AsyncVectraDBClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if self.stub is not None:
            return
        self.channel = grpc.aio.insecure_channel(self.address)
        self.stub = vectradb_pb2_grpc.VectraDbStub(self.channel)

    async def close(self) -> None:
        if self.channel is not None:
            await self.channel.close()
        self.channel = None
        self.stub = None

    async def _rpc(self, rpc_name: str, request):
        if self.stub is None:
            await self.connect()

        try:
            rpc = getattr(self.stub, rpc_name)
            return await rpc(request, timeout=self.timeout)
        except grpc.RpcError as exc:
            raise VectraDBError(f"gRPC error: {exc.details()}") from exc

    async def create_vector(
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
        response = await self._rpc("CreateVector", request)
        return _vector_from_response(response)

    async def get_vector(self, id: str) -> Vector:
        request = vectradb_pb2.GetVectorRequest(id=id)
        response = await self._rpc("GetVector", request)
        return _vector_from_response(response)

    async def update_vector(
        self,
        id: str,
        vector: Optional[Sequence[float]] = None,
        tags: Optional[Mapping[str, object]] = None,
        *,
        values: Optional[Sequence[float]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Vector:
        if vector is None and values is None:
            current = await self.get_vector(id)
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
        response = await self._rpc("UpdateVector", request)
        return _vector_from_response(response)

    async def delete_vector(self, id: str) -> DeleteResult:
        request = vectradb_pb2.DeleteVectorRequest(id=id)
        response = await self._rpc("DeleteVector", request)
        return _delete_result_from_proto(response)

    async def upsert_vector(
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
        response = await self._rpc("UpsertVector", request)
        return _vector_from_response(response)

    async def batch_create_vectors(
        self,
        items: Sequence[VectorInput],
    ) -> BatchWriteResponse:
        request = vectradb_pb2.BatchCreateVectorsRequest(
            items=[
                vectradb_pb2.VectorInput(
                    id=item.id,
                    vector=list(item.vector),
                    tags=_normalize_tags(item.tags),
                )
                for item in items
            ]
        )
        response = await self._rpc("BatchCreateVectors", request)
        return _batch_write_response_from_proto(response)

    async def batch_upsert_vectors(
        self,
        items: Sequence[VectorInput],
    ) -> BatchWriteResponse:
        request = vectradb_pb2.BatchUpsertVectorsRequest(
            items=[
                vectradb_pb2.VectorInput(
                    id=item.id,
                    vector=list(item.vector),
                    tags=_normalize_tags(item.tags),
                )
                for item in items
            ]
        )
        response = await self._rpc("BatchUpsertVectors", request)
        return _batch_write_response_from_proto(response)

    async def search_similar(
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
        response = await self._rpc("SearchSimilar", request)
        return _search_response_from_proto(response)

    async def search(
        self,
        query: Sequence[float],
        k: int = 10,
    ) -> list[SearchResult]:
        return (await self.search_similar(query=query, k=k)).results

    async def list_vectors(self) -> List[str]:
        request = vectradb_pb2.ListVectorsRequest()
        response = await self._rpc("ListVectors", request)
        return list(response.ids)

    async def get_stats(self) -> DatabaseStats:
        request = vectradb_pb2.GetStatsRequest()
        response = await self._rpc("GetStats", request)
        return DatabaseStats(
            total_vectors=response.total_vectors,
            dimension=response.dimension,
            memory_usage=response.memory_usage,
        )

    async def health_check(self) -> HealthStatus:
        request = vectradb_pb2.HealthCheckRequest()
        response = await self._rpc("HealthCheck", request)
        return _health_status_from_proto(response)

    async def is_healthy(self) -> bool:
        return bool(await self.health_check())
