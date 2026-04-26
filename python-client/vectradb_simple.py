"""Simple VectraDB Python client that matches the actual proto schema."""

import grpc
from typing import List, Dict, Optional

# Import generated gRPC code
try:
    from vectradb_client import vectradb_pb2, vectradb_pb2_grpc
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vectradb_client'))
    import vectradb_pb2
    import vectradb_pb2_grpc


class VectraDB:
    """Simple VectraDB gRPC client.
    
    Example:
        client = VectraDB()
        client.create("vec1", [0.1, 0.2, 0.3], {"type": "example"})
        results = client.search([0.1, 0.2, 0.3], k=10)
        client.close()
    """
    
    def __init__(self, host: str = "localhost", port: int = 50051):
        """Initialize client connection."""
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = vectradb_pb2_grpc.VectraDbStub(self.channel)
    
    def close(self):
        """Close the connection."""
        self.channel.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            response = self.stub.HealthCheck(vectradb_pb2.HealthCheckRequest())
            return response.status == "healthy"
        except Exception as e:
            print(f"Health check error: {e}")
            return False
    
    def create(self, id: str, vector: List[float], tags: Optional[Dict[str, str]] = None):
        """Create a new vector."""
        request = vectradb_pb2.CreateVectorRequest(
            id=id,
            vector=vector,
            tags=tags or {}
        )
        return self.stub.CreateVector(request)

    def batch_create(self, items: List[Dict[str, object]]):
        """Create many vectors in one RPC."""
        request = vectradb_pb2.BatchCreateVectorsRequest(
            items=[
                vectradb_pb2.VectorInput(
                    id=str(item["id"]),
                    vector=list(item["vector"]),
                    tags={str(k): str(v) for k, v in dict(item.get("tags", {})).items()},
                )
                for item in items
            ]
        )
        return self.stub.BatchCreateVectors(request)
    
    def get(self, id: str):
        """Get a vector by ID."""
        request = vectradb_pb2.GetVectorRequest(id=id)
        return self.stub.GetVector(request)
    
    def update(self, id: str, vector: List[float], tags: Optional[Dict[str, str]] = None):
        """Update a vector (must provide the vector)."""
        request = vectradb_pb2.UpdateVectorRequest(
            id=id,
            vector=vector,
            tags=tags or {}
        )
        return self.stub.UpdateVector(request)
    
    def delete(self, id: str):
        """Delete a vector."""
        request = vectradb_pb2.DeleteVectorRequest(id=id)
        return self.stub.DeleteVector(request)
    
    def upsert(self, id: str, vector: List[float], tags: Optional[Dict[str, str]] = None):
        """Create or update a vector."""
        request = vectradb_pb2.UpsertVectorRequest(
            id=id,
            vector=vector,
            tags=tags or {}
        )
        return self.stub.UpsertVector(request)

    def batch_upsert(self, items: List[Dict[str, object]]):
        """Upsert many vectors in one RPC."""
        request = vectradb_pb2.BatchUpsertVectorsRequest(
            items=[
                vectradb_pb2.VectorInput(
                    id=str(item["id"]),
                    vector=list(item["vector"]),
                    tags={str(k): str(v) for k, v in dict(item.get("tags", {})).items()},
                )
                for item in items
            ]
        )
        return self.stub.BatchUpsertVectors(request)
    
    def search(self, vector: List[float], k: int = 10):
        """Search for similar vectors."""
        request = vectradb_pb2.SearchRequest(
            vector=vector,
            top_k=k
        )
        return self.stub.SearchSimilar(request)
    
    def list(self):
        """List all vector IDs."""
        request = vectradb_pb2.ListVectorsRequest()
        return self.stub.ListVectors(request)
    
    def stats(self):
        """Get database statistics."""
        request = vectradb_pb2.GetStatsRequest()
        return self.stub.GetStats(request)
