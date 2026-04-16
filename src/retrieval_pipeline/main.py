"""Thin process entrypoint for the retrieval pipeline CLI.

Implementation details are split into focused modules:
- ``retrieval/indexing.py`` for ingest/indexing routines,
- ``cli/mode_handlers.py`` for query/runtime mode execution,
- ``cli/dispatcher.py`` for argument parsing and dispatch.
"""

from __future__ import annotations

import sys

from src.retrieval_pipeline.cli.dispatcher import parse_cli_args, run_cli
from src.retrieval_pipeline.cli.mode_handlers import load_pipeline_from_existing_store
from src.retrieval_pipeline.config import PipelineConfig
from src.retrieval_pipeline.logging_config import setup_logging
from src.retrieval_pipeline.retrieval.indexing import ingest_source, run_pipeline

__all__ = [
    "ingest_source",
    "load_pipeline_from_existing_store",
    "run_pipeline",
]


if __name__ == "__main__":
    setup_logging()
    cfg = PipelineConfig()
    mode, source_arg = parse_cli_args(sys.argv)
    raise SystemExit(run_cli(mode, source_arg, cfg))
