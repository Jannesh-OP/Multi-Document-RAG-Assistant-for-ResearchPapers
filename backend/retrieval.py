"""
Hybrid retrieval: BM25 (sparse) + FAISS (dense), fused via Reciprocal Rank
Fusion, then refined with cross-encoder reranking.
"""

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

_cross_encoder = None  # lazy-loaded singleton, reused across requests


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def build_indices(documents: list):
    """Build a fresh BM25 retriever + FAISS vectorstore over the given documents."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(documents, embedding_model)

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 50

    return bm25_retriever, vectorstore, embedding_model


def _reciprocal_rank_fusion(doc_lists, weights=None, k=60, top_n=20):
    """RRF: score(doc) = sum_over_retrievers( weight / (k + rank) )."""
    if weights is None:
        weights = [1.0] * len(doc_lists)

    scores, doc_lookup = {}, {}
    for docs, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(docs):
            key = (doc.page_content, doc.metadata.get("source"))
            scores[key] = scores.get(key, 0) + weight * (1.0 / (k + rank))
            doc_lookup[key] = doc

    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_lookup[key] for key in ranked_keys[:top_n]]



def hybrid_retrieve(query: str, bm25_retriever, vectorstore, weights=(0.4, 0.6), top_n=20):
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
    bm25_results = bm25_retriever.invoke(query)
    faiss_results = faiss_retriever.invoke(query)
    return _reciprocal_rank_fusion([bm25_results, faiss_results], weights=list(weights), top_n=top_n)


def rerank(query: str, docs: list, top_n: int = 10):
    if not docs:
        return []
    cross_encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]
