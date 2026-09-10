#  Intelligent Research Paper Assistant

A multi-document RAG (Retrieval-Augmented Generation) system for asking questions and comparing findings across multiple academic PDFs — with hybrid retrieval, cross-encoder reranking, and citation-grounded answers.

## What it does

Upload one or more research papers as PDFs, then ask natural-language questions against them — including questions that require comparing findings *across* papers. Every answer includes inline `[n]` markers pointing back to the exact excerpt(s) it was generated from, so claims are traceable rather than taken on faith.

## Architecture

```
PDF Upload
   │
   ▼
Layout-aware parsing (pymupdf4llm) ── Markdown output, tables preserved
   │
   ▼
Metadata + table extraction ── title/author/abstract, table detection (regex + PyMuPDF native table finder)
   │
   ▼
Section-aware chunking ── Markdown header splitter → recursive character splitter, junk/low-signal filtering
   │
   ▼
Dual indexing ── BM25 (sparse/lexical) + FAISS (dense/semantic, MiniLM embeddings)
   │
   ▼
Query time:
   Hybrid retrieve (BM25 + FAISS) → Reciprocal Rank Fusion → Cross-encoder rerank
   │
   ▼
Citation-grounded generation (LLM via OpenRouter) → inline [n] citations + source list
```

## Why these design choices

- **Hybrid retrieval (BM25 + FAISS):** BM25 catches exact keyword/terminology matches (important for technical papers with precise jargon); FAISS catches semantic matches even when wording differs. Neither alone is sufficient for research-paper QA.
- **Reciprocal Rank Fusion (RRF):** combines the two ranked lists by *rank position* rather than raw score, avoiding the problem of BM25 and cosine-similarity scores living on incomparable scales. Used here as a cheap prune step (50+50 candidates → top 20) before the more expensive reranker runs.
- **Cross-encoder reranking:** A cross-encoder scores each `(query, doc)` pair jointly, giving much more accurate relevance ranking — but it's too slow to run over the full corpus, hence the funnel.
- **Section-aware chunking:** splitting on Markdown headers before falling back to a character splitter keeps each chunk anchored to a real paper section (e.g. "Results," "Discussion") rather than cutting mid-argument.
- **Citation-grounded generation:** the prompt requires every claim to cite one of the numbered excerpts actually provided — the `[n]` markers are positional labels for *this query's* retrieved excerpts, not the papers' own bibliography numbering.

## Tech stack

| Layer | Tools |
|---|---|
| Backend API | FastAPI, Uvicorn |
| PDF parsing | PyMuPDF (`fitz`), `pymupdf4llm` |
| Chunking | LangChain text splitters |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Dense retrieval | FAISS |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM generation | OpenRouter (OpenAI-compatible API) via LangChain |
| Frontend | Streamlit |

## Project structure

```
papermind/
├── backend/
│   ├── app.py            # FastAPI routes: /upload, /query, /status
│   ├── ingestion.py       # PDF parsing, metadata, table extraction
│   ├── chunking.py        # Section-aware chunking
│   ├── retrieval.py       # Hybrid retrieval, RRF, cross-encoder reranking
│   ├── rag_chain.py       # Prompt template + generation
│   └── requirements.txt
└── frontend/
    ├── streamlit_app.py   # Upload UI + chat interface
    └── requirements.txt
```

## Running locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
export OPENROUTER_API_KEY=your_key_here
uvicorn app:app --reload --port 8000
```

**Frontend** (in a separate terminal):
```bash
cd frontend
pip install -r requirements.txt
export BACKEND_URL=http://localhost:8000
streamlit run streamlit_app.py
```

Then open the Streamlit URL it prints, upload a PDF, and start asking questions.

## Environment variables

| Variable | Where | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | backend | Auth for LLM generation via OpenRouter |
| `BACKEND_URL` | frontend | Points the Streamlit UI at the FastAPI backend |

## Known limitations / possible improvements

- Index is held in memory and rebuilt fresh on every `/upload` — no persistence across restarts.
- Only Free-Tier LLMs are used.
- No automated test suite yet.

