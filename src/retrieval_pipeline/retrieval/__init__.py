"""Retrieval package: index building, retrieval pipeline, and display logic."""

from __future__ import annotations

from src.retrieval_pipeline.retrieval.display import ResultsDisplay
from src.retrieval_pipeline.retrieval.indexing import ingest_source, run_pipeline
from src.retrieval_pipeline.retrieval.pipeline import RetrievalPipeline

__all__ = [
    "ResultsDisplay",
    "RetrievalPipeline",
    "ingest_source",
    "run_pipeline",
]
