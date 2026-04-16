"""CLI package for mode parsing and dispatch."""

from __future__ import annotations

from src.retrieval_pipeline.cli.dispatcher import parse_cli_args, run_cli
from src.retrieval_pipeline.cli.mode_handlers import (
    display_queries,
    load_pipeline_from_existing_store,
    run_agent_mode,
    run_chain_mode,
    run_compare_mode,
    run_graph_mode,
    run_reranker_mode,
    run_retriever_mode,
)
from src.retrieval_pipeline.cli.queries import AGENT_QUERIES, CHAIN_QUERIES, DISPLAY_QUERIES, GRAPH_QUERIES

__all__ = [
    "AGENT_QUERIES",
    "CHAIN_QUERIES",
    "DISPLAY_QUERIES",
    "GRAPH_QUERIES",
    "display_queries",
    "load_pipeline_from_existing_store",
    "parse_cli_args",
    "run_agent_mode",
    "run_chain_mode",
    "run_cli",
    "run_compare_mode",
    "run_graph_mode",
    "run_reranker_mode",
    "run_retriever_mode",
]
