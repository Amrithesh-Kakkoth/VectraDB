"""Optional integration tests against a real VectraDB server."""

from __future__ import annotations

import os

import pytest

from vectradb_client import AsyncVectraDBClient, VectraDBClient


pytestmark = pytest.mark.skipif(
    os.environ.get("VECTRADB_RUN_INTEGRATION") != "1",
    reason="Set VECTRADB_RUN_INTEGRATION=1 to run live integration tests.",
)


def test_server_running():
    client = VectraDBClient()
    try:
        assert client.health_check().healthy is True
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_server_running():
    async with AsyncVectraDBClient() as client:
        assert (await client.health_check()).healthy is True
