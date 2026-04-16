"""LLM chain module using Groq as the provider.

Build a RAG chain via the pipe (``|``) operator::

    prompt | llm | output_parser

The chain is assembled in :func:`build_rag_chain` and returns a plain
``str`` answer given a *question* and a list of retrieved
:class:`~langchain_core.documents.Document` objects.

Usage example
-------------
>>> from retrieval_pipeline.config import PipelineConfig
>>> from retrieval_pipeline.llm.chain import build_rag_chain
>>> chain = build_rag_chain(PipelineConfig())
>>> result = chain.invoke({"question": "What is EBITDA?", "context": docs})
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from loguru import logger

from src.retrieval_pipeline.config import PipelineConfig
from src.retrieval_pipeline.llm.prompts import RAG_PROMPT


def format_docs(docs: list[Document]) -> str:
    """Concatenate document page content into a single context string.

    Parameters
    ----------
    docs : list[Document]
        Retrieved document chunks.

    Returns
    -------
    str
        Page contents joined by a horizontal rule separator.
    """
    if not docs:
        logger.warning("format_docs | called with empty docs list — returning empty string")
        return ""
    return "\n\n---\n\n".join(d.page_content for d in docs)


def build_rag_chain(config: PipelineConfig | None = None) -> Runnable:
    """Build and return the RAG chain using the pipe operator.

    Chain shape::

        ChatPromptTemplate | ChatGroq | StrOutputParser

    Parameters
    ----------
    config : PipelineConfig | None
        Pipeline configuration. If ``None``, a default instance is used.

    Returns
    -------
    Runnable
        A LangChain runnable that accepts:
        - ``"question"``: str
        - ``"context"``: str

    Raises
    ------
    ValueError
        If ``GROQ_API_KEY`` is not set.
    """
    cfg = config or PipelineConfig()

    if not cfg.groq_api_key:
        msg = "GROQ_API_KEY is not set. Add it to your .env file or export it as an environment variable."
        raise ValueError(msg)

    logger.debug(
        "build_rag_chain | model='{}' max_tokens={} temperature={}",
        cfg.groq_model,
        cfg.llm_max_tokens,
        cfg.llm_temperature,
    )

    llm = ChatGroq(
        model=cfg.groq_model,
        api_key=cfg.groq_api_key,
        max_tokens=cfg.llm_max_tokens,
        temperature=cfg.llm_temperature,
    )

    chain: Runnable = RAG_PROMPT | llm | StrOutputParser()

    logger.info(
        "RAG chain built | prompt | ChatGroq('{}') | StrOutputParser",
        cfg.groq_model,
    )
    return chain


def answer(
    question: str,
    docs: list[Document],
    chain: Runnable | None = None,
    config: PipelineConfig | None = None,
) -> str:
    """Return an LLM answer for *question* grounded in *docs*.

    Parameters
    ----------
    question : str
        The user's question.
    docs : list[Document]
        Pre-retrieved document chunks.
    chain : Runnable | None
        Pre-built RAG chain. When provided, ``config`` is ignored and the
        chain is reused rather than rebuilt on every call.
    config : PipelineConfig | None
        Optional config used to build a chain when *chain* is ``None``.

    Returns
    -------
    str
        The LLM-generated answer.
    """
    rag_chain = chain or build_rag_chain(config)
    context_str = format_docs(docs)

    logger.debug(
        "answer | question='{}' | doc_count={} | context_length={} chars",
        question,
        len(docs),
        len(context_str),
    )

    if not docs:
        logger.warning("answer | no docs provided — LLM will have no context for question='{}'", question)

    result: str = rag_chain.invoke({"question": question, "context": context_str})
    logger.success("answer | answer generated | preview='{}'", result[:120])
    return result
