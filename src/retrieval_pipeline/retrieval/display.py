"""Output formatting and terminal display module.

Provides :class:`ResultsDisplay` which renders bi-encoder and cross-encoder
results to the terminal. ``print`` is intentional here — this module is a
terminal UI layer, not a library module.
"""

from __future__ import annotations

import textwrap

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.vectorstores import VectorStoreRetriever

from src.retrieval_pipeline.config import TOP_K, TOP_N


class ResultsDisplay:
    """Render retriever and reranker results to stdout.

    Parameters
    ----------
    width : int
        Column width used when wrapping document text.
    """

    def __init__(self, width: int = 80) -> None:
        """Initialise the display with a fixed terminal column width."""
        self.width = width

    def show_retriever(
        self,
        query: str,
        base_retriever: VectorStoreRetriever,
        cross_encoder: HuggingFaceCrossEncoder,
        top_k: int = TOP_K,
    ) -> None:
        """Print bi-encoder results with their cross-encoder scores.

        Parameters
        ----------
        query : str
            The search query string.
        base_retriever : VectorStoreRetriever
            Bi-encoder retriever to invoke.
        cross_encoder : HuggingFaceCrossEncoder
            Cross-encoder used to score the retrieved documents.
        top_k : int
            Number of results retrieved by the bi-encoder.
        """
        raw_docs = base_retriever.invoke(query)
        pairs = [(query, doc.page_content) for doc in raw_docs]
        scores = cross_encoder.score(pairs)

        print(f"\n{'=' * self.width}")
        print(f"  [BI-ENCODER] Top-{top_k} by cosine similarity  |  QUERY: {query}")
        print(f"{'=' * self.width}")

        for i, (doc, score) in enumerate(zip(raw_docs, scores, strict=False), 1):
            source = doc.metadata.get("source", "unknown")
            print(f"\n  [{i}] cross_score={score:.4f} (retrieval rank #{i}) | source={source}")
            print("  " + "-" * (self.width - 2))
            print(
                textwrap.fill(
                    doc.page_content.strip(),
                    width=self.width,
                    initial_indent="  ",
                    subsequent_indent="  ",
                ),
            )
        print()

    def show_reranker(
        self,
        query: str,
        compression_retriever: ContextualCompressionRetriever,
        top_n: int = TOP_N,
    ) -> None:
        """Print cross-encoder reranked results sorted by score.

        Scores are computed by calling the cross-encoder directly on the
        reranked documents, rather than relying on ``relevance_score`` in
        document metadata which ``CrossEncoderReranker`` does not reliably
        populate in all versions.

        Parameters
        ----------
        query : str
            The search query string.
        compression_retriever : ContextualCompressionRetriever
            Reranking retriever to invoke.
        top_n : int
            Number of results after reranking.
        """
        reranked_docs = compression_retriever.invoke(query)
        pairs = [(query, doc.page_content) for doc in reranked_docs]
        scores = compression_retriever.base_compressor.model.score(pairs)

        ranked = sorted(
            zip(reranked_docs, scores, strict=False),
            key=lambda t: t[1],
            reverse=True,
        )

        print(f"\n{'=' * self.width}")
        print(f"  [RERANKER]   Top-{top_n} after cross-encoder rerank  |  QUERY: {query}")
        print(f"{'=' * self.width}")

        for i, (doc, score) in enumerate(ranked, 1):
            source = doc.metadata.get("source", "unknown")
            print(f"\n  [{i}] reranker_score={score:.4f} | source={source}")
            print("  " + "-" * (self.width - 2))
            print(
                textwrap.fill(
                    doc.page_content.strip(),
                    width=self.width,
                    initial_indent="  ",
                    subsequent_indent="  ",
                ),
            )
        print()

    def compare(
        self,
        query: str,
        base_retriever: VectorStoreRetriever,
        compression_retriever: ContextualCompressionRetriever,
        cross_encoder: HuggingFaceCrossEncoder,
        top_k: int = TOP_K,
        top_n: int = TOP_N,
    ) -> None:
        """Run both retrievers and print their results sequentially.

        Parameters
        ----------
        query : str
            The search query string.
        base_retriever : VectorStoreRetriever
            Bi-encoder retriever.
        compression_retriever : ContextualCompressionRetriever
            Reranking retriever.
        cross_encoder : HuggingFaceCrossEncoder
            Cross-encoder used for scoring in the bi-encoder display only.
        top_k : int
            Bi-encoder candidate count.
        top_n : int
            Post-reranking result count.
        """
        print(f"\n{'#' * self.width}")
        print(f"  QUERY: {query}")
        print(f"{'#' * self.width}")
        self.show_retriever(query, base_retriever, cross_encoder, top_k)
        self.show_reranker(query, compression_retriever, top_n)
