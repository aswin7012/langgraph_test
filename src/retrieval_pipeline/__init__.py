"""Retrieval pipeline package.

Provides modular components for document loading, splitting,
vector storage, retrieval, and LangGraph-based query routing.
"""

from __future__ import annotations

from src.retrieval_pipeline.agent.tools import send_email, web_search
from src.retrieval_pipeline.cli.queries import AGENT_QUERIES, CHAIN_QUERIES, DISPLAY_QUERIES, GRAPH_QUERIES
from src.retrieval_pipeline.config import PipelineConfig
from src.retrieval_pipeline.data.loaders import DocumentLoader
from src.retrieval_pipeline.data.splitters import DocumentSplitter
from src.retrieval_pipeline.data.vectorstore import VectorStoreBuilder
from src.retrieval_pipeline.llm.chain import answer, build_rag_chain, format_docs
from src.retrieval_pipeline.llm.prompts import (
    AGENT_SYSTEM_PROMPT,
    CLASSIFY_PROMPT,
    CLASSIFY_SYSTEM_PROMPT,
    GENERAL_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    RAG_PROMPT,
    RAG_SYSTEM_PROMPT,
)
from src.retrieval_pipeline.logging_config import setup_logging
from src.retrieval_pipeline.retrieval.display import ResultsDisplay
from src.retrieval_pipeline.retrieval.pipeline import RetrievalPipeline


def __getattr__(name: str) -> object:
    """Lazy-load graph symbols to avoid circular imports."""
    if name in ("build_graph", "GraphState"):
        from src.retrieval_pipeline.llm import graph as _graph

        return getattr(_graph, name)

    raise AttributeError(f"module 'retrieval_pipeline' has no attribute {name!r}")


__all__ = [
    "AGENT_QUERIES",
    "AGENT_SYSTEM_PROMPT",
    "CHAIN_QUERIES",
    "CLASSIFY_PROMPT",
    "CLASSIFY_SYSTEM_PROMPT",
    "DISPLAY_QUERIES",
    "DocumentLoader",
    "DocumentSplitter",
    "GENERAL_PROMPT",
    "GENERAL_SYSTEM_PROMPT",
    "GRAPH_QUERIES",
    "GraphState",
    "PipelineConfig",
    "RAG_PROMPT",
    "RAG_SYSTEM_PROMPT",
    "ResultsDisplay",
    "RetrievalPipeline",
    "VectorStoreBuilder",
    "answer",
    "build_graph",
    "build_rag_chain",
    "format_docs",
    "send_email",
    "setup_logging",
    "web_search",
]
