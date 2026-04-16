"""Retrieval and reranking pipeline definition module.

Provides :class:`RetrievalPipeline` which assembles a bi-encoder retriever
and a cross-encoder reranker on top of a Chroma vector store.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from loguru import logger

from src.retrieval_pipeline.config import DEVICE, RERANKER_MODEL, TOP_K, TOP_N


class RetrievalPipeline:
    """Bi-encoder retriever combined with a cross-encoder reranker.

    Parameters
    ----------
    vectorstore : Chroma
        Populated Chroma vector store to search over.
    reranker_model : str
        HuggingFace model identifier for the cross-encoder.
    top_k : int
        Number of candidates the bi-encoder retrieves.
    top_n : int
        Number of results retained after cross-encoder reranking.
    device : str
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(
        self,
        vectorstore: Chroma,
        reranker_model: str = RERANKER_MODEL,
        top_k: int = TOP_K,
        top_n: int = TOP_N,
        device: str = DEVICE,
    ) -> None:
        """Build the retrieval pipeline from *vectorstore*."""
        logger.debug(
            "RetrievalPipeline init | reranker='{}' top_k={} top_n={} device='{}'",
            reranker_model,
            top_k,
            top_n,
            device,
        )

        self.base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )
        logger.debug("Base retriever ready | search_type=similarity | k={}", top_k)

        try:
            self.cross_encoder = HuggingFaceCrossEncoder(
                model_name=reranker_model,
                model_kwargs={"device": device},
            )
            logger.info("Cross-encoder loaded | model='{}'", reranker_model)
        except Exception as exc:
            logger.error("Failed to load cross-encoder '{}': {}", reranker_model, exc)
            raise

        compressor = CrossEncoderReranker(model=self.cross_encoder, top_n=top_n)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.base_retriever,
        )

        logger.success(
            "RetrievalPipeline ready | bi-encoder top-{} -> cross-encoder rerank -> top-{}",
            top_k,
            top_n,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[Document]:
        """Run the full retrieve-then-rerank pipeline for *query*.

        Parameters
        ----------
        query : str
            The search query string.

        Returns
        -------
        list[Document]
            Reranked document chunks, ordered by cross-encoder score.
        """
        logger.debug("RetrievalPipeline.retrieve | query='{}'", query)
        docs: list[Document] = self.compression_retriever.invoke(query)

        if not docs:
            logger.warning("RetrievalPipeline.retrieve | no documents returned for query='{}'", query)
        else:
            logger.info("RetrievalPipeline.retrieve | {} doc(s) returned for query='{}'", len(docs), query)

        return docs
