"""Data ingestion components for the retrieval pipeline.

This package exposes class-based modules for loading source data,
splitting documents into chunks, and building/loading the vector store.
"""

from __future__ import annotations

from src.retrieval_pipeline.data.loaders import SUPPORTED_EXTENSIONS, DocumentLoader
from src.retrieval_pipeline.data.splitters import DocumentSplitter
from src.retrieval_pipeline.data.vectorstore import VectorStoreBuilder

__all__ = [
    "DocumentLoader",
    "DocumentSplitter",
    "SUPPORTED_EXTENSIONS",
    "VectorStoreBuilder",
]
