"""Tests for the asynchronous VectraDB client."""

import pytest

from vectradb_client import (
    AsyncVectraDBClient,
    BatchWriteResponse,
    DatabaseStats,
    DeleteResult,
    HealthStatus,
    SearchResponse,
    SearchResult,
    VectorInput,
    Vector,
)


@pytest.mark.asyncio
async def test_async_create_vector_uses_proto_fields(async_client):
    result = await async_client.create_vector(
        "doc1",
        vector=[0.1, 0.2, 0.3],
        tags={"kind": "test"},
    )
    assert isinstance(result, Vector)
    assert result.id == "doc1"

    rpc_name, request, _ = async_client.stub.calls[-1]
    assert rpc_name == "CreateVector"
    assert list(request.vector) == pytest.approx([0.1, 0.2, 0.3])
    assert dict(request.tags) == {"kind": "test"}


@pytest.mark.asyncio
async def test_async_get_vector_maps_response(async_client):
    vector = await async_client.get_vector("doc1")
    assert isinstance(vector, Vector)
    assert vector.id == "doc1"
    assert vector.tags == {"kind": "test"}


@pytest.mark.asyncio
async def test_async_update_vector_can_reuse_existing_vector(async_client):
    result = await async_client.update_vector("doc1", metadata={"kind": "updated"})
    assert isinstance(result, Vector)
    assert result.id == "doc1"

    rpc_name, request, _ = async_client.stub.calls[-1]
    assert rpc_name == "UpdateVector"
    assert list(request.vector) == pytest.approx([0.1, 0.2, 0.3])
    assert dict(request.tags) == {"kind": "updated"}


@pytest.mark.asyncio
async def test_async_search_similar_uses_proto_shape(async_client):
    response = await async_client.search_similar(vector=[0.1, 0.2, 0.3], top_k=5)
    assert isinstance(response, SearchResponse)
    assert len(response.results) == 1
    assert isinstance(response.results[0], SearchResult)
    assert response.results[0].score == pytest.approx(0.98)
    assert response.total_time_ms == pytest.approx(1.5)

    rpc_name, request, _ = async_client.stub.calls[-1]
    assert rpc_name == "SearchSimilar"
    assert request.top_k == 5


@pytest.mark.asyncio
async def test_async_search_alias_returns_results_only(async_client):
    results = await async_client.search([0.1, 0.2, 0.3], k=5)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)


@pytest.mark.asyncio
async def test_async_batch_create_vectors_uses_batch_proto_shape(async_client):
    response = await async_client.batch_create_vectors(
        [
            VectorInput(id="doc1", vector=[0.1, 0.2, 0.3], tags={"kind": "test"}),
            VectorInput(id="doc2", vector=[0.4, 0.5, 0.6], tags={"kind": "test"}),
        ]
    )
    assert isinstance(response, BatchWriteResponse)
    assert len(response.statuses) == 2
    assert all(status.ok for status in response.statuses)

    rpc_name, request, _ = async_client.stub.calls[-1]
    assert rpc_name == "BatchCreateVectors"
    assert len(request.items) == 2


@pytest.mark.asyncio
async def test_async_batch_upsert_vectors_uses_batch_proto_shape(async_client):
    response = await async_client.batch_upsert_vectors(
        [VectorInput(id="doc1", vector=[0.1, 0.2, 0.3], tags={"kind": "test"})]
    )
    assert isinstance(response, BatchWriteResponse)
    assert response.statuses[0].code == "OK"

    rpc_name, request, _ = async_client.stub.calls[-1]
    assert rpc_name == "BatchUpsertVectors"
    assert request.items[0].id == "doc1"


@pytest.mark.asyncio
async def test_async_list_vectors_returns_ids(async_client):
    assert await async_client.list_vectors() == ["doc1", "doc2"]


@pytest.mark.asyncio
async def test_async_get_stats_maps_actual_proto_shape(async_client):
    stats = await async_client.get_stats()
    assert isinstance(stats, DatabaseStats)
    assert stats.memory_usage == 1024


@pytest.mark.asyncio
async def test_async_health_check(async_client):
    health = await async_client.health_check()
    assert isinstance(health, HealthStatus)
    assert health.healthy is True
    assert bool(health) is True


@pytest.mark.asyncio
async def test_async_delete_vector_returns_proto_shape(async_client):
    result = await async_client.delete_vector("doc1")
    assert isinstance(result, DeleteResult)
    assert result.success is True
    assert bool(result) is True
