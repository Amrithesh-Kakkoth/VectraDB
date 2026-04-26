"""Test configuration and fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytest_asyncio

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from vectradb_client import AsyncVectraDBClient, VectraDBClient
from vectradb_client import vectradb_pb2


class FakeStub:
    def __init__(self):
        self.calls = []

    def CreateVector(self, request, timeout=None):
        self.calls.append(("CreateVector", request, timeout))
        return vectradb_pb2.VectorResponse(id=request.id, vector=request.vector, tags=request.tags)

    def BatchCreateVectors(self, request, timeout=None):
        self.calls.append(("BatchCreateVectors", request, timeout))
        return vectradb_pb2.BatchWriteResponse(
            statuses=[
                vectradb_pb2.BatchWriteItemStatus(id=item.id, code="OK", message="")
                for item in request.items
            ]
        )

    def GetVector(self, request, timeout=None):
        self.calls.append(("GetVector", request, timeout))
        return vectradb_pb2.VectorResponse(
            id=request.id,
            vector=[0.1, 0.2, 0.3],
            dimension=3,
            created_at=1,
            updated_at=2,
            tags={"kind": "test"},
        )

    def UpdateVector(self, request, timeout=None):
        self.calls.append(("UpdateVector", request, timeout))
        return vectradb_pb2.VectorResponse(id=request.id, vector=request.vector, tags=request.tags)

    def DeleteVector(self, request, timeout=None):
        self.calls.append(("DeleteVector", request, timeout))
        return vectradb_pb2.DeleteVectorResponse(success=True)

    def UpsertVector(self, request, timeout=None):
        self.calls.append(("UpsertVector", request, timeout))
        return vectradb_pb2.VectorResponse(id=request.id, vector=request.vector, tags=request.tags)

    def BatchUpsertVectors(self, request, timeout=None):
        self.calls.append(("BatchUpsertVectors", request, timeout))
        return vectradb_pb2.BatchWriteResponse(
            statuses=[
                vectradb_pb2.BatchWriteItemStatus(id=item.id, code="OK", message="")
                for item in request.items
            ]
        )

    def SearchSimilar(self, request, timeout=None):
        self.calls.append(("SearchSimilar", request, timeout))
        return vectradb_pb2.SearchResponse(
            results=[
                vectradb_pb2.SimilarityResult(
                    id="doc1",
                    score=0.98,
                    metadata=vectradb_pb2.VectorMetadata(
                        id="doc1",
                        dimension=3,
                        created_at=1,
                        updated_at=2,
                        tags={"kind": "test"},
                    ),
                )
            ],
            total_time_ms=1.5,
        )

    def ListVectors(self, request, timeout=None):
        self.calls.append(("ListVectors", request, timeout))
        return vectradb_pb2.ListVectorsResponse(ids=["doc1", "doc2"])

    def GetStats(self, request, timeout=None):
        self.calls.append(("GetStats", request, timeout))
        return vectradb_pb2.StatsResponse(total_vectors=2, dimension=3, memory_usage=1024)

    def HealthCheck(self, request, timeout=None):
        self.calls.append(("HealthCheck", request, timeout))
        return vectradb_pb2.HealthCheckResponse(status="healthy", service="vectradb-grpc")


class AsyncFakeStub(FakeStub):
    async def CreateVector(self, request, timeout=None):
        return super().CreateVector(request, timeout)

    async def BatchCreateVectors(self, request, timeout=None):
        return super().BatchCreateVectors(request, timeout)

    async def GetVector(self, request, timeout=None):
        return super().GetVector(request, timeout)

    async def UpdateVector(self, request, timeout=None):
        return super().UpdateVector(request, timeout)

    async def DeleteVector(self, request, timeout=None):
        return super().DeleteVector(request, timeout)

    async def UpsertVector(self, request, timeout=None):
        return super().UpsertVector(request, timeout)

    async def BatchUpsertVectors(self, request, timeout=None):
        return super().BatchUpsertVectors(request, timeout)

    async def SearchSimilar(self, request, timeout=None):
        return super().SearchSimilar(request, timeout)

    async def ListVectors(self, request, timeout=None):
        return super().ListVectors(request, timeout)

    async def GetStats(self, request, timeout=None):
        return super().GetStats(request, timeout)

    async def HealthCheck(self, request, timeout=None):
        return super().HealthCheck(request, timeout)


@pytest.fixture
def client():
    client = VectraDBClient()
    client.stub = FakeStub()
    yield client
    client.close()


@pytest_asyncio.fixture
async def async_client():
    client = AsyncVectraDBClient()
    client.stub = AsyncFakeStub()
    yield client
    await client.close()
