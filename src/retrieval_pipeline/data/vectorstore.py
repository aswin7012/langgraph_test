"""Vector store management module.

Provides :class:`VectorStoreBuilder` which wraps Chroma and HuggingFace
embeddings into a single buildable unit.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from src.retrieval_pipeline.config import CHROMA_DIR, COLLECTION_NAME, DEVICE, EMBEDDING_MODEL


class VectorStoreBuilder:
    """Build or reload a Chroma vector store from document chunks.

    Parameters
    ----------
    embedding_model : str
        HuggingFace model identifier for the bi-encoder.
    chroma_dir : str
        Filesystem path where Chroma persists its data.
    collection_name : str
        Name of the Chroma collection to create or reuse.
    device : str
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        chroma_dir: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        device: str = DEVICE,
    ) -> None:
        """Initialise the builder with model and storage settings."""
        self.embedding_model = embedding_model
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.device = device

    def build(
        self,
        chunks: Sequence[Document] | None = None,
        *,
        load_existing: bool = False,
    ) -> tuple[Chroma, HuggingFaceEmbeddings]:
        """Build or reload a vector store.

        Parameters
        ----------
        chunks : Sequence[Document] | None
            Chunks to ingest when ``load_existing`` is ``False``.
        load_existing : bool
            When ``True``, skip ingestion and connect to an existing persisted
            Chroma collection.

        Returns
        -------
        tuple[Chroma, HuggingFaceEmbeddings]
            The vector store and the embedding model instance.
        """
        embeddings = self._build_embeddings()

        if load_existing:
            vectorstore = self._connect_store(embeddings)
            vector_count = len(vectorstore.get().get("ids", []))
            logger.info(
                "Existing vector store loaded | collection='{}' vectors={} dir='{}'.",
                self.collection_name,
                vector_count,
                self.chroma_dir,
            )
            return vectorstore, embeddings

        documents = list(chunks or [])
        logger.debug(
            "Building vector store from {} chunk(s) using '{}'.",
            len(documents),
            self.embedding_model,
        )

        if documents:
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=self.collection_name,
                persist_directory=self.chroma_dir,
            )
        else:
            vectorstore = self._connect_store(embeddings)
            logger.warning(
                "No chunks provided; returning collection '{}' without adding vectors.",
                self.collection_name,
            )

        vector_count = len(vectorstore.get().get("ids", []))
        logger.info(
            "Vector store ready | collection='{}' vectors={} dir='{}'.",
            self.collection_name,
            vector_count,
            self.chroma_dir,
        )
        return vectorstore, embeddings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_embeddings(self) -> HuggingFaceEmbeddings:
        """Instantiate and return the HuggingFace embedding model."""
        logger.debug("Loading embedding model: {}.", self.embedding_model)
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
        logger.info("Embedding model loaded: {}.", self.embedding_model)
        return embeddings

    def _connect_store(self, embeddings: HuggingFaceEmbeddings) -> Chroma:
        """Connect to a persisted Chroma collection (or create it if absent)."""
        return Chroma(
            collection_name=self.collection_name,
            persist_directory=self.chroma_dir,
            embedding_function=embeddings,
        )
