"""Prompt templates for the retrieval pipeline.

All system and human prompt strings live here so they can be found,
read, and edited in one place without touching any logic files.

Prompts
-------
RAG chain (used in ``chain.py``)
    :data:`RAG_SYSTEM_PROMPT` — instructs the LLM to answer from context only.
    :data:`RAG_HUMAN_PROMPT`  — passes the user question.

Query classifier (used in ``graph.py``)
    :data:`CLASSIFY_SYSTEM_PROMPT` — instructs the LLM to label the query type.
    :data:`CLASSIFY_HUMAN_PROMPT`  — passes the user question.

General fallback (used in ``graph.py``)
    :data:`GENERAL_SYSTEM_PROMPT` — instructs the LLM to answer freely.
    :data:`GENERAL_HUMAN_PROMPT`  — passes the user question.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# RAG chain prompts
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT: str = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context excerpts. If the context does not contain enough "
    "information to answer, say so clearly — do not fabricate facts.\n\n"
    "Context:\n{context}"
)

RAG_HUMAN_PROMPT: str = "{question}"

RAG_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_HUMAN_PROMPT),
    ]
)

# ---------------------------------------------------------------------------
# Query classifier prompts
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT: str = "Classify the question as 'ml' or 'general'. Return only one word."

CLASSIFY_HUMAN_PROMPT: str = "{question}"

CLASSIFY_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", CLASSIFY_SYSTEM_PROMPT),
        ("human", CLASSIFY_HUMAN_PROMPT),
    ]
)

# ---------------------------------------------------------------------------
# General fallback prompts
# ---------------------------------------------------------------------------

GENERAL_SYSTEM_PROMPT: str = "You are a helpful assistant."

GENERAL_HUMAN_PROMPT: str = "{question}"

GENERAL_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", GENERAL_SYSTEM_PROMPT),
        ("human", GENERAL_HUMAN_PROMPT),
    ]
)

# ---------------------------------------------------------------------------
# Agent tool-calling prompts (used in ``cli/mode_handlers.py``)
# ---------------------------------------------------------------------------
AGENT_SYSTEM_PROMPT: str = (
    "You are a research assistant. Use the tools available to you when needed.\n\n"
    "## Tool usage\n"
    "- Use web_search when the question requires current information or external knowledge.\n"
    "- Do NOT use web_search for greetings, capability questions, or anything you can "
    "answer directly.\n"
    "- Use send_email ONLY when the user explicitly asks to email something to a specific "
    "recipient address that they provide. Never invent or assume an address.\n"
    "- Before calling send_email, write a clean concise summary (3-5 sentences, plain prose "
    "only — no bullet points, no URLs) as the body. Never paste raw search results into it.\n"
    "- Call send_email exactly once. If it returns 'Success:', stop — do not call it again.\n\n"
    "## General behaviour\n"
    "- Think step by step before deciding whether to use a tool.\n"
    "- Answer directly when no tool is needed.\n"
    "- Always include the actual content in your final reply, not just a delivery status.\n"
    "- Be concise and factual. Never fabricate information.\n"
    "- If a tool returns an error, explain it clearly and tell the user how to fix it."
)


def get_agent_system_prompt(*, include_send_email: bool) -> str:
    """Return the correct agent system prompt based on available tools."""
    return AGENT_SYSTEM_PROMPT
