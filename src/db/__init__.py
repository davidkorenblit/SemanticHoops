"""
db/__init__.py

Public surface of the database layer:
  - QdrantClient    : thin wrapper around qdrant-client
  - VectorIndexer   : high-level upsert / search interface
"""
from .qdrant_client import QdrantClientWrapper
from .indexer import VectorIndexer

__all__ = ["QdrantClientWrapper", "VectorIndexer"]
