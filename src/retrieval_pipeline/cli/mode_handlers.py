"""Runtime mode handlers for existing-index query workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from loguru import logger

from src.retrieval_pipeline.cli.queries import AGENT_QUERIES, CHAIN_QUERIES, DISPLAY_QUERIES, GRAPH_QUERIES
from src.retrieval_pipeline.config import PipelineConfig
from src.retrieval_pipeline.data.vectorstore import VectorStoreBuilder
from src.retrieval_pipeline.llm.chain import answer as llm_answer
from src.retrieval_pipeline.llm.graph import build_graph
from src.retrieval_pipeline.retrieval.display import ResultsDisplay
from src.retrieval_pipeline.retrieval.pipeline import RetrievalPipeline


def load_pipeline_from_existing_store(config: PipelineConfig | None = None) -> RetrievalPipeline:
    """Load an already-indexed Chroma store and build the retrieval pipeline.

    Raises
    ------
    FileNotFoundError
        If the Chroma persistence directory is missing.
    ValueError
        If the vector store exists but contains no vectors.
    """
    cfg = config or PipelineConfig()
    chroma_path = Path(cfg.chroma_dir)

    if not chroma_path.exists():
        msg = (
            f"No vector store found at '{cfg.chroma_dir}'. "
            "Run 'python -m retrieval_pipeline.main ingest [source]' first."
        )
        raise FileNotFoundError(msg)

    logger.info(
        "Loading existing vector store | dir='{}' collection='{}'",
        cfg.chroma_dir,
        cfg.collection_name,
    )
    vs_builder = VectorStoreBuilder(
        embedding_model=cfg.embedding_model,
        chroma_dir=cfg.chroma_dir,
        collection_name=cfg.collection_name,
        device=cfg.device,
    )
    vectorstore, _ = vs_builder.build([], load_existing=True)

    stored_ids = vectorstore.get().get("ids", [])
    if not stored_ids:
        msg = (
            f"Vector store directory '{cfg.chroma_dir}' exists, but collection "
            f"'{cfg.collection_name}' contains 0 vectors. "
            "Run ingest to index your data first."
        )
        raise ValueError(msg)

    logger.info(
        "Building retrieval pipeline from existing index | top_k={} top_n={}",
        cfg.top_k,
        cfg.top_n,
    )
    return RetrievalPipeline(
        vectorstore=vectorstore,
        reranker_model=cfg.reranker_model,
        top_k=cfg.top_k,
        top_n=cfg.top_n,
        device=cfg.device,
    )


def display_queries(
    pipeline: RetrievalPipeline,
    queries: list[str],
    config: PipelineConfig,
    show: Literal["retriever", "reranker", "both"],
) -> None:
    """Run display queries using an already-built retrieval pipeline."""
    if not queries:
        logger.debug("No display queries configured — nothing to show")
        return

    logger.info("Running {} query/queries | display='{}'", len(queries), show)
    display = ResultsDisplay()

    for q in queries:
        logger.debug("Querying | q='{}'", q)
        if show == "retriever":
            display.show_retriever(q, pipeline.base_retriever, pipeline.cross_encoder, config.top_k)
        elif show == "reranker":
            display.show_reranker(q, pipeline.compression_retriever, config.top_n)
        else:
            display.compare(
                q,
                pipeline.base_retriever,
                pipeline.compression_retriever,
                pipeline.cross_encoder,
                config.top_k,
                config.top_n,
            )


def run_chain_mode(pipeline: RetrievalPipeline) -> None:
    """Run predefined chain-mode queries through retrieval + LLM answering."""
    for question in CHAIN_QUERIES:
        logger.info("Chain mode | question='{}'", question)
        docs = pipeline.retrieve(question)
        result = llm_answer(question, docs)
        logger.success("Chain mode answer: {}", result)


def run_graph_mode(pipeline: RetrievalPipeline, config: PipelineConfig) -> None:
    """Run predefined graph-mode queries through the routing graph."""
    app = build_graph(pipeline, config)

    logger.info("Graph mode | {} queries to run", len(GRAPH_QUERIES))
    for q in GRAPH_QUERIES:
        result = app.invoke({"question": q})
        logger.success(
            "Graph result | question='{}' | type='{}' | answer='{}'",
            q,
            result["query_type"],
            result["answer"],
        )


def run_retriever_mode(pipeline: RetrievalPipeline, config: PipelineConfig) -> None:
    """Run display queries showing only bi-encoder retrieval output."""
    display_queries(pipeline, DISPLAY_QUERIES, config, show="retriever")
    logger.success("Retriever mode complete.")


def run_reranker_mode(pipeline: RetrievalPipeline, config: PipelineConfig) -> None:
    """Run display queries showing only reranked output."""
    display_queries(pipeline, DISPLAY_QUERIES, config, show="reranker")
    logger.success("Reranker mode complete.")


def run_compare_mode(pipeline: RetrievalPipeline, config: PipelineConfig) -> None:
    """Run display queries comparing bi-encoder and reranker outputs."""
    display_queries(pipeline, DISPLAY_QUERIES, config, show="both")
    logger.success("Compare mode complete.")


def run_agent_mode(pipeline: RetrievalPipeline, config: PipelineConfig) -> None:
    """Run agent mode with tool-calling (Tavily web search + email)."""
    from langchain.agents import create_agent
    from langchain_groq import ChatGroq

    from src.retrieval_pipeline.agent.tools import send_email, web_search
    from src.retrieval_pipeline.llm.prompts import get_agent_system_prompt

    llm = ChatGroq(
        model=config.groq_model,
        api_key=config.groq_api_key,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        disable_streaming="tool_calling",
        model_kwargs={"parallel_tool_calls": False},
    )

    tools = [web_search, send_email]
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=get_agent_system_prompt(include_send_email=True),
    )

    logger.info("Agent ready | tools={}", [t.name for t in tools])

    for question in AGENT_QUERIES:
        logger.info("Agent mode | question='{}'", question)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": 30},
        )
        answer = result["messages"][-1].content
        logger.success("Agent answer: {}", answer)
