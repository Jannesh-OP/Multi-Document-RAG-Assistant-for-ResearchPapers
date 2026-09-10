"""
FastAPI backend for the Intelligent Research Paper Assistant.

Flow:
  POST /upload -> parse, extract tables, chunk, embed, build BM25+FAISS index
                  (single shared in-memory index; replaced on each upload)
  POST /query  -> hybrid retrieve -> rerank -> citation-grounded answer
  GET  /status -> whether an index is ready, and what's in it
"""
import os
import shutil
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingestion import parse_pdf, extract_metadata, extract_tables_from_markdown, clean_markdown_artifacts,build_abstract_document,extract_tables_pymupdf
from chunking import chunk_paper
from retrieval import build_indices, hybrid_retrieve, rerank as rerank_docs
from rag_chain import build_llm, ask_question

app = FastAPI(title="Intelligent Research Paper Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared in-memory index. Built once on /upload, reused for every
# /query until the next /upload replaces it — no reindexing per question.
STATE = {
    "ready": False,
    "documents": [],
    "bm25_retriever": None,
    "vectorstore": None,
    "llm": None,
    "paper_titles": [],
}


class QueryRequest(BaseModel):
    question: str
    top_n: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


@app.on_event("startup")
def startup():
    STATE["llm"] = build_llm()


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files uploaded")

    tmp_dir = tempfile.mkdtemp()
    try:
        saved_paths = []
        for f in files:
            if not f.filename.lower().endswith(".pdf"):
                continue
            path = os.path.join(tmp_dir, f.filename)
            with open(path, "wb") as out:
                shutil.copyfileobj(f.file, out)
            saved_paths.append(path)

        if not saved_paths:
            raise HTTPException(400, "No valid PDF files found in upload")

        all_prose_chunks, all_table_docs, paper_titles = [], [], []

        for path in saved_paths:
            md_text = parse_pdf(path)
            meta = extract_metadata(path)
            paper_titles.append(meta["title"])
            abstract_doc = build_abstract_document(meta)
            if abstract_doc:
                all_prose_chunks.append(abstract_doc)

            _, cleaned_md = extract_tables_from_markdown(md_text, meta)
            table_docs = extract_tables_pymupdf(path, meta)


            cleaned_md = clean_markdown_artifacts(cleaned_md)


            all_prose_chunks.extend(chunk_paper(cleaned_md, meta))
            all_table_docs.extend(table_docs)

        all_documents = all_prose_chunks + all_table_docs
        if not all_documents:
            raise HTTPException(400, "No extractable content found in uploaded PDFs")

        bm25_retriever, vectorstore, _ = build_indices(all_documents)

        STATE.update({
            "ready": True,
            "documents": all_documents,
            "bm25_retriever": bm25_retriever,
            "vectorstore": vectorstore,
            "paper_titles": paper_titles,
        })

        return {
            "status": "success",
            "papers_processed": len(saved_paths),
            "paper_titles": paper_titles,
            "prose_chunks": len(all_prose_chunks),
            "table_chunks": len(all_table_docs),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not STATE["ready"]:
        raise HTTPException(400, "No documents indexed yet. Upload PDFs first via /upload.")

    candidates = hybrid_retrieve(req.question, STATE["bm25_retriever"], STATE["vectorstore"], top_n=20)
    top_docs = rerank_docs(req.question, candidates, top_n=10)
    answer, sources = ask_question(STATE["llm"], req.question, top_docs)

    return QueryResponse(answer=answer, sources=sources)


@app.get("/status")
async def status():
    return {
        "ready": STATE["ready"],
        "paper_titles": STATE["paper_titles"],
        "total_chunks": len(STATE["documents"]),
    }


@app.get("/")
async def root():
    return {"message": "Intelligent Research Paper Assistant API", "docs": "/docs"}


from dotenv import load_dotenv
load_dotenv()
api_key = os.environ["OPENROUTER_API_KEY"]