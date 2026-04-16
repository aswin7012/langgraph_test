"""Document splitting and chunking module.

Provides :class:`DocumentSplitter` which routes documents to the correct
LangChain text splitter based on their detected type (Markdown vs. plain).
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from src.retrieval_pipeline.config import CHUNK_OVERLAP, CHUNK_SIZE


class DocumentSplitter:
    """Split documents into chunks for downstream embedding.

    Markdown documents are split with a Markdown-aware splitter;
    all other documents use the recursive character splitter.

    Parameters
    ----------
    chunk_size : int
        Maximum character count per chunk.
    chunk_overlap : int
        Overlap in characters between consecutive chunks.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        """Initialise the splitter with chunk size and overlap settings."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.debug(
            "DocumentSplitter initialised (chunk_size={}, chunk_overlap={}).",
            chunk_size,
            chunk_overlap,
        )

    def split(self, docs: list[Document]) -> list[Document]:
        """Split *docs* into smaller chunks.

        Markdown-sourced documents are routed to :class:`MarkdownTextSplitter`;
        all others use :class:`RecursiveCharacterTextSplitter`.

        Parameters
        ----------
        docs : list[Document]
            Source documents to split.

        Returns
        -------
        list[Document]
            All chunks produced from *docs*.
        """
        markdown_docs = [doc for doc in docs if self._is_markdown_document(doc)]
        other_docs = [doc for doc in docs if not self._is_markdown_document(doc)]

        logger.debug(
            "Splitting {} doc(s): {} Markdown, {} other.",
            len(docs),
            len(markdown_docs),
            len(other_docs),
        )

        chunks: list[Document] = []

        if markdown_docs:
            markdown_splitter = MarkdownTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            markdown_chunks = markdown_splitter.split_documents(markdown_docs)
            logger.debug("Markdown splitter produced {} chunk(s).", len(markdown_chunks))
            chunks.extend(markdown_chunks)

        if other_docs:
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            generic_chunks = recursive_splitter.split_documents(other_docs)
            logger.debug("Recursive splitter produced {} chunk(s).", len(generic_chunks))
            chunks.extend(generic_chunks)

        logger.info(
            "Splitting complete: {} total chunk(s) from {} document(s).",
            len(chunks),
            len(docs),
        )
        return chunks

    @staticmethod
    def _is_markdown_document(doc: Document) -> bool:
        """Return ``True`` when document metadata indicates markdown content."""
        source = str(doc.metadata.get("source", "")).lower()
        return source.endswith((".md", ".markdown"))
