
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template(
"""You are a research assistant answering questions using ONLY the provided excerpts from academic papers.

Rules:
- Cite sources inline using the [n] tags matching the excerpts below.
- If the question asks to compare across papers, structure your answer by point of comparison, citing each relevant source.
- If the excerpts don't contain enough information, say so explicitly — do not fabricate.

Context excerpts:
{context}

Question: {question}

Answer (with inline [n] citations):"""
)




def build_llm() -> ChatOpenAI:
    api_key = os.environ["OPENROUTER_API_KEY"]
    return ChatOpenAI(
        model="openrouter/free",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
    )


def _format_docs_with_citations(docs) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        src = doc.metadata.get("title", doc.metadata.get("source"))
        section = doc.metadata.get("section") or doc.metadata.get("caption", "")
        formatted.append(f"[{i+1}] (Source: {src} | {section})\n{doc.page_content}")
    return "\n\n".join(formatted)


def ask_question(llm: ChatOpenAI, question: str, docs: list):
    """Runs generation over already-retrieved+reranked docs. Returns (answer, sources)."""
    context = _format_docs_with_citations(docs)
    prompt = RAG_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)

    sources = [
        {
            "title": d.metadata.get("title"),
            "source": d.metadata.get("source"),
            "type": d.metadata.get("type"),
            "section": d.metadata.get("section") or d.metadata.get("caption"),
        }
        for d in docs
    ]
    return response.content, sources
