"""LangGraph workflow for query routing and response generation.

Flow
----
1. Classify query as ``ml`` or ``general``.
2. Route:
   - ``ml``      -> retrieve docs -> answer with RAG chain.
   - ``general`` -> answer directly with a general LLM prompt.

Environment
-----------
Requires ``GROQ_API_KEY`` in environment (typically via ``.env``).
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from loguru import logger

from src.retrieval_pipeline.config import PipelineConfig
from src.retrieval_pipeline.llm.chain import build_rag_chain
from src.retrieval_pipeline.llm.prompts import CLASSIFY_PROMPT, GENERAL_PROMPT
from src.retrieval_pipeline.retrieval.pipeline import RetrievalPipeline


class GraphState(TypedDict):
    """Mutable state passed between LangGraph nodes.

    Keys
    ----
    question : str
        User input question.
    query_type : str
        Classifier output label (``"ml"`` or ``"general"``).
    docs : list[Document]
        Retrieved documents for RAG path.
    answer : str
        Final answer produced by selected path.
    """

    question: str
    query_type: str
    docs: list[Document]
    answer: str


def build_graph(
    pipeline: RetrievalPipeline,
    config: PipelineConfig | None = None,
) -> object:
    """Build and compile the LangGraph query router.

    Parameters
    ----------
    pipeline : RetrievalPipeline
        Retrieval pipeline used by the ``retrieve`` node.
    config : PipelineConfig | None
        Optional runtime config (Groq model, temperature, token cap).

    Returns
    -------
    langgraph.graph.state.CompiledStateGraph
        Compiled graph object exposing ``.invoke(...)``.

    Raises
    ------
    ValueError
        If ``GROQ_API_KEY`` is missing.
    """
    cfg = config or PipelineConfig()

    if not cfg.groq_api_key:
        msg = "GROQ_API_KEY is missing. Add it to .env before running the graph."
        raise ValueError(msg)

    llm = ChatGroq(
        model=cfg.groq_model,
        api_key=cfg.groq_api_key,
        max_tokens=cfg.llm_max_tokens,
        temperature=cfg.llm_temperature,
    )

    classifier_chain = CLASSIFY_PROMPT | llm
    general_chain = GENERAL_PROMPT | llm
    rag_chain = build_rag_chain(cfg)

    def classify(state: GraphState) -> dict[str, str]:
        """Classify question into ``ml`` or ``general`` route."""
        q = state["question"]
        logger.info("[Graph] classify | q='{}'", q)
        out = classifier_chain.invoke({"question": q})
        label = (out.content or "").strip().lower()
        if label not in {"ml", "general"}:
            logger.warning("[Graph] classify | unexpected label='{}' -> forcing 'general'", label)
            label = "general"
        logger.debug("[Graph] classify | label='{}'", label)
        return {"query_type": label}

    def retrieve(state: GraphState) -> dict[str, list[Document]]:
        """Retrieve candidate documents for ML/domain-specific questions."""
        q = state["question"]
        logger.info("[Graph] retrieve | q='{}'", q)
        docs = pipeline.retrieve(q)
        logger.debug("[Graph] retrieve | docs={}", len(docs))
        return {"docs": docs}

    def generate_rag(state: GraphState) -> dict[str, str]:
        """Generate grounded answer from retrieved docs using RAG chain."""
        q = state["question"]
        docs = state.get("docs", [])
        logger.info("[Graph] generate_rag | q='{}' docs={}", q, len(docs))
        context = "\n\n".join(d.page_content for d in docs)
        ans = rag_chain.invoke({"question": q, "context": context})
        return {"answer": ans}

    def generate_general(state: GraphState) -> dict[str, str]:
        """Generate free-form answer for non-ML/general queries."""
        q = state["question"]
        logger.info("[Graph] generate_general | q='{}'", q)
        out = general_chain.invoke({"question": q})
        return {"answer": out.content}

    def route(state: GraphState) -> str:
        """Routing function returning next node name."""
        return "retrieve" if state["query_type"] == "ml" else "generate_general"

    graph = StateGraph(GraphState)

    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_rag", generate_rag)
    graph.add_node("generate_general", generate_general)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route,
        {
            "retrieve": "retrieve",
            "generate_general": "generate_general",
        },
    )
    graph.add_edge("retrieve", "generate_rag")
    graph.add_edge("generate_rag", END)
    graph.add_edge("generate_general", END)

    logger.info("[Graph] compiled successfully")
    return graph.compile()
