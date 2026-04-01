# ◈ DocMind — RAG Chatbot with Streaming Responses

> A production-ready document Q&A chatbot built with LangChain, Pinecone, Sentence Transformers, and Streamlit.  
> Users just ask questions — all indexing is done beforehand.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
│                    Streamlit (app.py)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │ query
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                            │
│                    src/pipeline.py                           │
│                                                              │
│   ┌──────────────────┐       ┌──────────────────────────┐   │
│   │    Retriever      │       │       Generator           │   │
│   │  src/retriever.py │       │    src/generator.py       │   │
│   │                  │       │                           │   │
│   │  HuggingFace     │       │  Groq API (LLaMA 3)       │   │
│   │  Embeddings      │       │  Streaming token output   │   │
│   │  (MiniLM-L6-v2)  │       │                           │   │
│   └────────┬─────────┘       └───────────────────────────┘   │
│            │                                                  │
│            ▼                                                  │
│   ┌──────────────────┐                                        │
│   │   Pinecone DB    │                                        │
│   │  Vector Search   │                                        │
│   └──────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

**Flow:**
1. Query → Embedding (MiniLM) → Pinecone similarity search → Top-4 chunks
2. Chunks + Query → Prompt → Groq LLaMA 3 → Streamed answer

---

## Folder Structure

```
rag_chatbot/
├── app.py                  # Streamlit UI with streaming
├── requirements.txt
├── .env.example            # Copy to .env and fill in keys
├── README.md
│
├── data/                   # ← Drop your PDF/TXT files here
│
├── src/
│   ├── __init__.py
│   ├── ingest.py           # One-time indexing script
│   ├── retriever.py        # Pinecone + embeddings
│   ├── generator.py        # Groq LLM + streaming
│   └── pipeline.py         # Combines retriever + generator
│
├── chunks/                 # (optional) save chunk previews
├── vectordb/               # (optional) local cache
└── notebooks/              # Exploration notebooks
```

---

## Setup Instructions

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/rag_chatbot.git
cd rag_chatbot
pip install -r requirements.txt
```

### 2. Get your free API keys

| Service | URL | What it's for |
|---------|-----|---------------|
| **Pinecone** | [app.pinecone.io](https://app.pinecone.io) | Vector database (free tier: 1 index) |
| **Groq** | [console.groq.com](https://console.groq.com) | Free LLM API (LLaMA 3, Mistral) |

### 3. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in your PINECONE_API_KEY and GROQ_API_KEY
```

### 4. Add your document(s)

```bash
# Drop your PDF or TXT files into the /data folder
cp your_document.pdf data/
```

### 5. Run the ingestion pipeline (ONE TIME ONLY)

```bash
python -m src.ingest
```

This will:
- Load all files from `/data`
- Clean and chunk them into 100–300 word segments
- Generate embeddings using `all-MiniLM-L6-v2`
- Create a Pinecone index and upsert all vectors

You should see output like:
```
📂  Loading documents from: ./data
✅  Loaded 42 raw pages/documents.
✅  Created 187 chunks.
✅  Index 'docmind-index' already exists.
⬆️   Upserting 187 chunks to Pinecone…
✅  All chunks indexed successfully.
🎉  Ingestion complete! You can now run: streamlit run app.py
```

### 6. Run the chatbot

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Model & Embedding Choices

### Embedding Model: `all-MiniLM-L6-v2`
- Lightweight (80MB), runs on CPU
- 384-dimensional vectors
- Great retrieval quality for English documents
- From the `sentence-transformers` library

### LLM: `llama3-8b-8192` via Groq
- Free tier available on Groq
- Fast inference with streaming support
- Context window: 8192 tokens
- Good instruction-following for RAG tasks

### Vector DB: Pinecone (Serverless)
- Free tier: 1 index, 100K vectors
- Cosine similarity search
- Managed, no infrastructure needed

---

## Sample Queries & Outputs

**Query 1:** *"What are the main privacy rights of users?"*  
✅ Returns accurate answer citing specific clauses from the document.

**Query 2:** *"How is user data stored and protected?"*  
✅ Retrieves relevant sections, streams a grounded answer with source passages.

**Query 3:** *"What happens if I delete my account?"*  
✅ Answers correctly based on retrieved chunks.

**Query 4 (Failure case):** *"What is the CEO's salary?"*  
⚠️ Model correctly responds: "The provided context does not contain information about..."

**Query 5 (Ambiguous):** *"Tell me everything"*  
⚠️ Model asks for clarification or provides a high-level summary based on available chunks.

---

## Known Limitations

- **Hallucination risk**: The model is prompted to stay grounded, but may occasionally infer beyond the retrieved context.
- **Chunk boundary issues**: Some answers may be cut off if the relevant information spans a chunk boundary.
- **CPU-only embeddings**: Embedding 187 chunks takes ~10 seconds on first run; subsequent queries are fast.
- **Groq rate limits**: Free tier has rate limits; add retry logic for production use.

---

## Demo

> 📹 *[Link to demo video / GIF — add after recording]*

---

## Tech Stack

| Component | Library/Service |
|-----------|----------------|
| Framework | LangChain |
| Embeddings | sentence-transformers (HuggingFace) |
| Vector DB | Pinecone (Serverless) |
| LLM | LLaMA 3 via Groq API |
| UI | Streamlit |
| Doc Loading | LangChain DirectoryLoader + PyPDF |
