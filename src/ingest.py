from __future__ import annotations
import os
import re
import time

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX    = os.getenv("PINECONE_INDEX", "amlgo-index").strip()  # strip accidental spaces
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_DIR          = os.path.join(os.path.dirname(__file__), "..", "data")
CHUNK_SIZE        = 300    # words approx (~1500 chars)
CHUNK_OVERLAP     = 50


# ── Helpers ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove excessive whitespace and common header/footer noise."""
    # Remove page numbers like "Page 1 of 10"
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove non-printable characters except newlines/tabs
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def load_documents():
    """Load PDFs and TXT files from /data using DirectoryLoader."""
    print(f"📂  Loading documents from: {DATA_DIR}")

    # PDF loader
    pdf_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    # TXT loader
    txt_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )

    docs = pdf_loader.load() + txt_loader.load()
    print(f"✅  Loaded {len(docs)} raw pages/documents.")

    if not docs:
        raise ValueError("❌  No documents found in /data. Add PDF or TXT files first.")

    return docs


def chunk_documents(docs):
    """Split documents into 100-300 word chunks with sentence awareness."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 5,       # ~300 words in chars
        chunk_overlap=CHUNK_OVERLAP * 5,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )

    # Clean text before chunking
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    chunks = splitter.split_documents(docs)
    print(f"✅  Created {len(chunks)} chunks.")
    return chunks


def create_pinecone_index():
    """Create Pinecone index if it doesn't exist (dimension=384 for MiniLM)."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX not in existing:
        print(f"🔧  Creating Pinecone index '{PINECONE_INDEX}'…")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=384,           # all-MiniLM-L6-v2 output dim
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until index is ready before returning
        print("⏳  Waiting for index to be ready…")
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            time.sleep(2)
        print("✅  Index created and ready.")
    else:
        print(f"✅  Index '{PINECONE_INDEX}' already exists.")
    
    return pc  # return client so upsert can reuse it


def upsert_to_pinecone(chunks, pc=None):
    """Embed chunks and upsert into Pinecone using LangChain."""
    print("🔢  Loading embedding model…")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if pc is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)

    print(f"⬆️   Upserting {len(chunks)} chunks to Pinecone…")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX,
        pinecone_api_key=PINECONE_API_KEY,
    )
    print("✅  All chunks indexed successfully.")


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀  Amlgo — Document Ingestion Pipeline\n" + "=" * 45)

    if not PINECONE_API_KEY:
        raise EnvironmentError("PINECONE_API_KEY not set. Check your .env file.")

    docs   = load_documents()
    chunks = chunk_documents(docs)
    pc     = create_pinecone_index()
    upsert_to_pinecone(chunks, pc=pc)

    print("\n🎉  Ingestion complete! You can now run: streamlit run app.py\n")