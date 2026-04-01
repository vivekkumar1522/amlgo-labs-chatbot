

from __future__ import annotations
import os
from typing import Generator, List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


# ── Config ───────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
MODEL_NAME    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")   # free on Groq
TEMPERATURE   = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS    = int(os.getenv("LLM_MAX_TOKENS", "512"))

# ── Prompt template ──────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise, helpful document assistant.
Answer the question using ONLY the information provided in the context below.
If the context does not contain enough information to answer, say so clearly.
Do not hallucinate or add information not present in the context.

Context:
---------
{context}
---------

Question: {question}

Answer:""",
)


class Generator:
    """Streams token-by-token answers via Groq's LLaMA / Mistral endpoint."""

    model_name: str = MODEL_NAME

    def __init__(self):
        self._llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            streaming=True,
        )
        self._chain = RAG_PROMPT | self._llm | StrOutputParser()

    # ── public ──────────────────────────────────────────────────────────

    def stream(
        self, query: str, context_docs: List[Document]
    ) -> Generator[str, None, None]:
        """Yield text tokens one at a time."""
        context_text = "\n\n".join(
            f"[Chunk {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(context_docs)
        )
        yield from self._chain.stream(
            {"context": context_text, "question": query}
        )
