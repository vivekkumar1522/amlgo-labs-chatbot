"""
src/pipeline.py
Core RAG pipeline – ties together the retriever and generator.
"""

from __future__ import annotations
import os
from typing import Generator, Tuple, List

from src.retriever import Retriever
from src.generator import Generator as LLMGenerator


class RAGPipeline:
    """
    End-to-end RAG pipeline.
    - Retriever fetches relevant chunks from Pinecone.
    - Generator streams an answer grounded in those chunks.
    """

    def __init__(self):
        self.retriever = Retriever()
        self.generator = LLMGenerator()

    # ── public ──────────────────────────────────────────────────────────

    def stream_answer(
        self, query: str, top_k: int = 4
    ) -> Generator[Tuple[str, List[str]], None, None]:
        """
        Yields (text_chunk, sources_list) tuples.
        sources_list is empty until the very last token so callers
        can display it once streaming finishes.
        """
        # 1. Retrieve relevant passages
        docs = self.retriever.retrieve(query, top_k=top_k)
        sources = [doc.page_content for doc in docs]

        # 2. Stream the LLM response
        for token in self.generator.stream(query=query, context_docs=docs):
            yield token, sources

    def get_stats(self) -> dict:
        """Returns metadata shown in the sidebar."""
        return {
            "model":    self.generator.model_name,
            "chunks":   self.retriever.index_size(),
            "vectordb": "Pinecone",
            "embedder": self.retriever.embedder_name,
        }
