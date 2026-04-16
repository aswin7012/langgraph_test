"""Command-line dispatch for retrieval pipeline modes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from loguru import logger

from src.retrieval_pipeline.cli.mode_handlers import (
    load_pipeline_from_existing_store,
    run_agent_mode,
    run_chain_mode,
    run_compare_mode,
    run_graph_mode,
    run_reranker_mode,
    run_retriever_mode,
)
from src.retrieval_pipeline.config import PipelineConfig
from src.retrieval_pipeline.retrieval.indexing import ingest_source
from src.retrieval_pipeline.retrieval.pipeline import RetrievalPipeline

_ARG_MODE_INDEX = 1
_ARG_SOURCE_INDEX = 2
_SUCCESS_EXIT_CODE = 0
_FAILURE_EXIT_CODE = 1
_QUERY_MODES = {"retriever", "reranker", "compare", "chain", "graph", "agent"}


def parse_cli_args(argv: list[str]) -> tuple[str, str | None]:
    """Parse mode and optional source path from argv.

    Defaults to ``retriever`` mode when no mode is provided.
    """
    mode = argv[_ARG_MODE_INDEX] if len(argv) > _ARG_MODE_INDEX else "retriever"
    source_arg = argv[_ARG_SOURCE_INDEX] if len(argv) > _ARG_SOURCE_INDEX else None
    return mode, source_arg


def run_cli(mode: str, source_arg: str | None, config: PipelineConfig) -> int:
    """Execute the selected CLI mode.

    Returns
    -------
    int
        Process-style exit code (``0`` on success, ``1`` on failure).
    """
    if mode == "ingest":
        return _run_ingest_mode(source_arg, config)

    if mode in _QUERY_MODES:
        return _run_query_mode(mode, source_arg, config)

    logger.critical(
        "Unknown mode '{}'. Valid modes: ingest | retriever | reranker | compare | chain | graph | agent",
        mode,
    )
    return _FAILURE_EXIT_CODE


def _run_ingest_mode(source_arg: str | None, config: PipelineConfig) -> int:
    """Run ingest mode and return a process-style exit code."""
    source_path = _resolve_source(source_arg, config)
    is_dir_source = source_path.is_dir()

    if is_dir_source:
        logger.info("Ingest starting | source='{}' (directory)", source_path)
    elif source_path.is_file():
        logger.info("Ingest starting | source='{}' (file)", source_path)
    else:
        logger.critical("Source path does not exist: '{}' — cannot ingest", source_path)
        return _FAILURE_EXIT_CODE

    ingest_source(
        source=str(source_path),
        config=config,
        is_directory=is_dir_source,
    )
    logger.success("Ingest mode complete.")
    return _SUCCESS_EXIT_CODE


def _run_query_mode(mode: str, source_arg: str | None, config: PipelineConfig) -> int:
    """Run a query mode, bootstrapping ingest if needed."""
    pipeline = _load_or_bootstrap_pipeline(source_arg, config)
    if pipeline is None:
        return _FAILURE_EXIT_CODE

    handlers: dict[str, Callable[[RetrievalPipeline, PipelineConfig], None]] = {
        "retriever": run_retriever_mode,
        "reranker": run_reranker_mode,
        "compare": run_compare_mode,
        "graph": run_graph_mode,
        "agent": run_agent_mode,
        "chain": lambda p, _cfg: run_chain_mode(p),
    }
    handlers[mode](pipeline, config)
    return _SUCCESS_EXIT_CODE


def _load_or_bootstrap_pipeline(
    source_arg: str | None,
    config: PipelineConfig,
) -> RetrievalPipeline | None:
    """Load existing index or bootstrap ingest, then reload the pipeline."""
    if source_arg is not None:
        logger.info(
            "Source argument '{}' will be used as ingest bootstrap source "
            "if the index is missing/empty.",
            source_arg,
        )

    try:
        return load_pipeline_from_existing_store(config)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("{}", exc)

    bootstrap_source = _resolve_source(source_arg, config)
    bootstrap_is_dir = bootstrap_source.is_dir()
    bootstrap_exists = bootstrap_is_dir or bootstrap_source.is_file()
    if not bootstrap_exists:
        logger.critical(
            "Cannot bootstrap ingest. Source path '{}' does not exist. "
            "Provide a valid source path or set PIPELINE_SOURCE.",
            bootstrap_source,
        )
        return None

    logger.info(
        "Bootstrapping ingest because index is unavailable or empty | source='{}'",
        bootstrap_source,
    )
    ingest_source(
        source=str(bootstrap_source),
        config=config,
        is_directory=bootstrap_is_dir,
    )

    try:
        return load_pipeline_from_existing_store(config)
    except (FileNotFoundError, ValueError) as reload_exc:
        logger.critical("{}", reload_exc)
        return None


def _resolve_source(path_arg: str | None, config: PipelineConfig) -> Path:
    """Resolve ingest source path from CLI argument or ``PIPELINE_SOURCE``."""
    raw = path_arg or config.pipeline_source
    resolved = Path(raw)
    logger.debug("_resolve_source | raw='{}' resolved='{}'", raw, resolved)
    return resolved
