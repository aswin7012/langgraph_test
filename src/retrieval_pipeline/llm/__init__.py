"""LLM package: prompts, RAG chain assembly, and graph routing."""

from __future__ import annotations

from src.retrieval_pipeline.llm.chain import answer, build_rag_chain, format_docs
from src.retrieval_pipeline.llm.graph import GraphState, build_graph
from src.retrieval_pipeline.llm.prompts import (
    AGENT_SYSTEM_PROMPT,
    CLASSIFY_PROMPT,
    CLASSIFY_SYSTEM_PROMPT,
    GENERAL_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    RAG_PROMPT,
    RAG_SYSTEM_PROMPT,
)

__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "CLASSIFY_PROMPT",
    "CLASSIFY_SYSTEM_PROMPT",
    "GENERAL_PROMPT",
    "GENERAL_SYSTEM_PROMPT",
    "GraphState",
    "RAG_PROMPT",
    "RAG_SYSTEM_PROMPT",
    "answer",
    "build_graph",
    "build_rag_chain",
    "format_docs",
]
