# 📦 Retrieval Pipeline Setup

## READ THIS FIRST

Read env example and give out necessary config

---

## Install uv (package manager)

```bash
pip install uv
```

## Create virtual environment

```bash
uv venv
```

## Activate virtual environment (bash / linux)

```bash
source .venv/bin/activate
```

## Install project in editable mode

```bash
uv pip install -e .
```

## Install pre-commit hooks

```bash
uv pip install pre-commit
pre-commit install
pre-commit run
```

---

## Run the application

The workflow is now split into two phases:

1. `ingest` once to load, chunk, and persist vectors.
2. Run query modes (`retriever`, `reranker`, `compare`, `chain`, `graph`, `agent`) using the existing index.

Only `ingest` accepts an optional source path argument.

```bash
# Step 1: Build/update vector store (required before query modes)
python -m retrieval_pipeline.main ingest
python -m retrieval_pipeline.main ingest path/to/file.pdf

# Step 2: Query existing vector store

# Bi-encoder results only (default when no mode given)
python -m retrieval_pipeline.main retriever

# Cross-encoder reranked results only
python -m retrieval_pipeline.main reranker

# Bi-encoder vs reranker side-by-side comparison
python -m retrieval_pipeline.main compare

# Full RAG answer via Groq LLM
python -m retrieval_pipeline.main chain

# LangGraph query-routing pipeline (ML vs general classifier)
python -m retrieval_pipeline.main graph

# Tool-calling agent mode
python -m retrieval_pipeline.main agent
```

---

## For Docling

(Codespaces didn’t have it — ignore if you don’t face this error)

```bash
sudo apt-get update
sudo apt-get install -y libgl1
```
