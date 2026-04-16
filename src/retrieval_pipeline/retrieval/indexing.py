"""Indexing and end-to-end pipeline assembly helpers.

This module owns source loading, chunking, vector indexing, and the optional
in-process query display workflow used by legacy callers.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from langchain_community.vectorstores.utils import filter_complex_metadata
from loguru import logger

from src.retrieval_pipeline.config import PipelineConfig
from src.retrieval_pipeline.data.loaders import DocumentLoader
from src.retrieval_pipeline.data.splitters import DocumentSplitter
from src.retrieval_pipeline.data.vectorstore import VectorStoreBuilder
from src.retrieval_pipeline.retrieval.display import ResultsDisplay
from src.retrieval_pipeline.retrieval.pipeline import RetrievalPipeline


def run_pipeline(
    source: str | pd.DataFrame | list,
    queries: list[str],
    config: PipelineConfig | None = None,
    is_directory: bool = False,
    extensions: list[str] | None = None,
    json_jq_schema: str = ".",
    sql_query: str = "SELECT * FROM documents",
    show: Literal["retriever", "reranker", "both"] = "both",
) -> RetrievalPipeline:
    """Execute load, split, embed/store, pipeline-build, and optional querying.

    Parameters
    ----------
    source : str | pd.DataFrame | list
        Document source — file path, directory path, URL, DataFrame, or list.
    queries : list[str]
        Queries to run against the pipeline after indexing.
    config : PipelineConfig | None
        Pipeline configuration. Defaults to a fresh :class:`PipelineConfig`.
    is_directory : bool
        When ``True``, *source* is treated as a directory root.
    extensions : list[str] | None
        File extensions to restrict directory loading to.
    json_jq_schema : str
        ``jq`` schema for JSON file loading.
    sql_query : str
        SQL statement for SQLite file loading.
    show : Literal["retriever", "reranker", "both"]
        Which results to display: bi-encoder only, reranker only, or both.

    Returns
    -------
    RetrievalPipeline
        The fully initialised retrieval pipeline.
    """
    cfg = config or PipelineConfig()

    logger.info(
        "── Step 1/5: Loading documents from source='{}'",
        source if isinstance(source, str) else type(source).__name__,
    )
    loader = DocumentLoader(json_jq_schema=json_jq_schema, sql_query=sql_query)
    docs = (
        loader.load_directory(str(source), extensions=extensions) if is_directory else loader.load(source)
    )

    if not docs:
        logger.warning("No documents loaded from source='{}' — pipeline may produce empty results", source)
    else:
        logger.success("Step 1 complete | {} document(s) loaded", len(docs))

    logger.info(
        "── Step 2/5: Splitting documents | chunk_size={} overlap={}",
        cfg.chunk_size,
        cfg.chunk_overlap,
    )
    splitter = DocumentSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    chunks = splitter.split(docs)
    logger.success("Step 2 complete | {} chunk(s) produced", len(chunks))

    logger.info("── Step 3/5: Embedding & building vector store | model='{}'", cfg.embedding_model)
    clean_chunks = filter_complex_metadata(chunks)
    if len(clean_chunks) < len(chunks):
        logger.warning(
            "filter_complex_metadata removed {} chunk(s) with unsupported metadata types",
            len(chunks) - len(clean_chunks),
        )
    vs_builder = VectorStoreBuilder(
        embedding_model=cfg.embedding_model,
        chroma_dir=cfg.chroma_dir,
        collection_name=cfg.collection_name,
        device=cfg.device,
    )
    vectorstore, _ = vs_builder.build(clean_chunks)
    logger.success("Step 3 complete | vector store ready at '{}'", cfg.chroma_dir)

    logger.info("── Step 4/5: Building retrieval pipeline | top_k={} top_n={}", cfg.top_k, cfg.top_n)
    pipeline = RetrievalPipeline(
        vectorstore=vectorstore,
        reranker_model=cfg.reranker_model,
        top_k=cfg.top_k,
        top_n=cfg.top_n,
        device=cfg.device,
    )
    logger.success("Step 4 complete | bi-encoder + cross-encoder ready")

    if queries:
        logger.info("── Step 5/5: Running {} query/queries | display='{}'", len(queries), show)
        display = ResultsDisplay()

        for q in queries:
            logger.debug("Querying | q='{}'", q)
            if show == "retriever":
                display.show_retriever(q, pipeline.base_retriever, pipeline.cross_encoder, cfg.top_k)
            elif show == "reranker":
                display.show_reranker(q, pipeline.compression_retriever, cfg.top_n)
            else:
                display.compare(
                    q,
                    pipeline.base_retriever,
                    pipeline.compression_retriever,
                    pipeline.cross_encoder,
                    cfg.top_k,
                    cfg.top_n,
                )

        logger.success("Step 5 complete | all queries processed")
    else:
        logger.debug("── Step 5/5: No queries provided — skipping display")

    return pipeline


def ingest_source(
    source: str | pd.DataFrame | list,
    config: PipelineConfig | None = None,
    is_directory: bool = False,
    extensions: list[str] | None = None,
    json_jq_schema: str = ".",
    sql_query: str = "SELECT * FROM documents",
) -> None:
    """Load, split, and persist documents to Chroma without querying.

    This is the one-time indexing step intended for CLI ``ingest`` mode.
    """
    cfg = config or PipelineConfig()

    logger.info(
        "── Ingest 1/3: Loading documents from source='{}'",
        source if isinstance(source, str) else type(source).__name__,
    )
    loader = DocumentLoader(json_jq_schema=json_jq_schema, sql_query=sql_query)
    docs = (
        loader.load_directory(str(source), extensions=extensions) if is_directory else loader.load(source)
    )

    if not docs:
        logger.warning("No documents loaded from source='{}' — ingest produced an empty index", source)
    else:
        logger.success("Ingest 1/3 complete | {} document(s) loaded", len(docs))

    logger.info(
        "── Ingest 2/3: Splitting documents | chunk_size={} overlap={}",
        cfg.chunk_size,
        cfg.chunk_overlap,
    )
    splitter = DocumentSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    chunks = splitter.split(docs)
    logger.success("Ingest 2/3 complete | {} chunk(s) produced", len(chunks))

    logger.info(
        "── Ingest 3/3: Embedding & persisting vector store | model='{}'",
        cfg.embedding_model,
    )
    clean_chunks = filter_complex_metadata(chunks)
    if len(clean_chunks) < len(chunks):
        logger.warning(
            "filter_complex_metadata removed {} chunk(s) with unsupported metadata types",
            len(chunks) - len(clean_chunks),
        )

    vs_builder = VectorStoreBuilder(
        embedding_model=cfg.embedding_model,
        chroma_dir=cfg.chroma_dir,
        collection_name=cfg.collection_name,
        device=cfg.device,
    )
    vs_builder.build(clean_chunks, load_existing=False)
    logger.success(
        "Ingest 3/3 complete | index persisted at '{}' | collection='{}'",
        cfg.chroma_dir,
        cfg.collection_name,
    )
