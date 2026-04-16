"""Default query sets for each CLI mode.

Edit this file to change what gets run when you launch the pipeline
from the command line. Each mode reads its queries from here.

All values are ``list[str]``. A single query is just a one-element list.

Modes
-----
DISPLAY_QUERIES  — used by ``retriever``, ``reranker``, and ``compare`` modes.
CHAIN_QUERIES    — used by ``chain`` mode (RAG answer via Groq).
GRAPH_QUERIES    — used by ``graph`` mode (LangGraph classifier pipeline).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Retriever / reranker / compare modes
# ---------------------------------------------------------------------------

DISPLAY_QUERIES: list[str] = [
    "What is machine learning",
]

# ---------------------------------------------------------------------------
# Chain mode (full RAG answer)
# ---------------------------------------------------------------------------

CHAIN_QUERIES: list[str] = [
    "What is the machine learning?",
]

# ---------------------------------------------------------------------------
# Graph mode (ML vs general classifier)
# ---------------------------------------------------------------------------

GRAPH_QUERIES: list[str] = [
    "Types of Machine Learning?",
    "Hi, how are we doing today ?",
]

# ---------------------------------------------------------------------------
# Agent mode (tool-calling agent)
# ---------------------------------------------------------------------------

AGENT_QUERIES: list[str] = [
    "Search the web for the latest advancements in AI agents and summarize the top 3 trends in 2026.",
    "HI ,how are you doing and what are your capablities?",
    (
        "Search the web for a quick summary of what LangChain tool-calling agents are, "
        "and email the explanation to vh12234_aiml22@velhightech.com."
    ),
]
