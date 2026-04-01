

from __future__ import annotations
import os
from typing import List

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()


# ── Config (override via .env / environment variables) ──────────────────
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "amlgo-index")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class Retriever:
    """
    Wraps a LangChain PineconeVectorStore for similarity search.
    The index must already be populated (run ingest.py first).
    """

    embedder_name: str = EMBEDDING_MODEL

    def __init__(self):
        # Load the embedding model (runs locally via sentence-transformers)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Connect to existing Pinecone index
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self._vectorstore = PineconeVectorStore(
            index=pc.Index(PINECONE_INDEX),
            embedding=self._embeddings,
            text_key="text",
        )

    # ── public ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 4) -> List[Document]:
        """Return the top-k most relevant document chunks."""
        return self._vectorstore.similarity_search(query, k=top_k)

    def index_size(self) -> int:
        """Return the number of vectors stored in the Pinecone index."""
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            stats = pc.Index(PINECONE_INDEX).describe_index_stats()
            return stats.total_vector_count
        except Exception:
            return 0
