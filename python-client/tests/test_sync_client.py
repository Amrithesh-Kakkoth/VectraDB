"""Tests for the synchronous VectraDB client."""

import pytest

from vectradb_client import (
    BatchWriteResponse,
    DatabaseStats,
    DeleteResult,
    HealthStatus,
    SearchResponse,
    SearchResult,
    VectorInput,
    Vector,
    VectraDBClient,
)


def test_create_vector_uses_proto_fields(client):
    result = client.create_vector("doc1", vector=[0.1, 0.2, 0.3], tags={"kind": "test"})
    assert isinstance(result, Vector)
    assert result.id == "doc1"

    rpc_name, request, _ = client.stub.calls[-1]
    assert rpc_name == "CreateVector"
    assert list(request.vector) == pytest.approx([0.1, 0.2, 0.3])
    assert dict(request.tags) == {"kind": "test"}


def test_get_vector_maps_response(client):
    vector = client.get_vector("doc1")
    assert isinstance(vector, Vector)
    assert vector.id == "doc1"
    assert vector.vector == pytest.approx([0.1, 0.2, 0.3])
    assert vector.values == pytest.approx([0.1, 0.2, 0.3])
    assert vector.tags == {"kind": "test"}
    assert vector.metadata == {"kind": "test"}


def test_update_vector_can_reuse_existing_vector(client):
    result = client.update_vector("doc1", metadata={"kind": "updated"})
    assert isinstance(result, Vector)
    assert result.id == "doc1"

    rpc_name, request, _ = client.stub.calls[-1]
    assert rpc_name == "UpdateVector"
    assert list(request.vector) == pytest.approx([0.1, 0.2, 0.3])
    assert dict(request.tags) == {"kind": "updated"}


def test_search_similar_uses_proto_shape(client):
    response = client.search_similar(vector=[0.1, 0.2, 0.3], top_k=5)
    assert isinstance(response, SearchResponse)
    assert len(response.results) == 1
    assert isinstance(response.results[0], SearchResult)
    assert response.results[0].score == pytest.approx(0.98)
    assert response.results[0].metadata is not None
    assert response.results[0].metadata.tags == {"kind": "test"}
    assert response.total_time_ms == pytest.approx(1.5)

    rpc_name, request, _ = client.stub.calls[-1]
    assert rpc_name == "SearchSimilar"
    assert request.top_k == 5
    assert list(request.vector) == pytest.approx([0.1, 0.2, 0.3])


def test_search_alias_returns_results_only(client):
    results = client.search([0.1, 0.2, 0.3], k=5)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)


def test_batch_create_vectors_uses_batch_proto_shape(client):
    response = client.batch_create_vectors(
        [
            VectorInput(id="doc1", vector=[0.1, 0.2, 0.3], tags={"kind": "test"}),
            VectorInput(id="doc2", vector=[0.4, 0.5, 0.6], tags={"kind": "test"}),
        ]
    )
    assert isinstance(response, BatchWriteResponse)
    assert len(response.statuses) == 2
    assert all(status.ok for status in response.statuses)

    rpc_name, request, _ = client.stub.calls[-1]
    assert rpc_name == "BatchCreateVectors"
    assert len(request.items) == 2
    assert list(request.items[0].vector) == pytest.approx([0.1, 0.2, 0.3])


def test_batch_upsert_vectors_uses_batch_proto_shape(client):
    response = client.batch_upsert_vectors(
        [VectorInput(id="doc1", vector=[0.1, 0.2, 0.3], tags={"kind": "test"})]
    )
    assert isinstance(response, BatchWriteResponse)
    assert response.statuses[0].code == "OK"

    rpc_name, request, _ = client.stub.calls[-1]
    assert rpc_name == "BatchUpsertVectors"
    assert request.items[0].id == "doc1"


def test_list_vectors_returns_ids(client):
    assert client.list_vectors() == ["doc1", "doc2"]


def test_get_stats_maps_actual_proto_shape(client):
    stats = client.get_stats()
    assert isinstance(stats, DatabaseStats)
    assert stats.total_vectors == 2
    assert stats.dimension == 3
    assert stats.memory_usage == 1024
    assert stats.memory_usage_bytes == 1024


def test_health_check_uses_status_string(client):
    health = client.health_check()
    assert isinstance(health, HealthStatus)
    assert health.healthy is True
    assert bool(health) is True


def test_delete_vector_returns_proto_shape(client):
    result = client.delete_vector("doc1")
    assert isinstance(result, DeleteResult)
    assert result.success is True
    assert bool(result) is True


def test_context_manager():
    with VectraDBClient() as client:
        client.stub = type(
            "Stub",
            (),
            {
                "HealthCheck": lambda self, request, timeout=None: type(
                    "Resp", (), {"status": "healthy"}
                )()
            },
        )()
        assert client.health_check().healthy is True
