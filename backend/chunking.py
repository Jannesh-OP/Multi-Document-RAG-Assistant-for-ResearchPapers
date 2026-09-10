"""
Section-aware semantic chunking of cleaned paper text.
"""
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_JUNK_INDICATORS = ["picture text", "ELSEVIER", "<br>", "<mark>", "~~"]


def _is_low_signal(chunk_text: str, min_alpha_ratio: float = 0.3, min_words: int = 8) -> bool:
    words = chunk_text.split()
    if len(words) < min_words:
        return True
    alpha_chars = sum(c.isalpha() for c in chunk_text)
    ratio = alpha_chars / max(len(chunk_text), 1)
    return ratio < min_alpha_ratio


def _is_junk(chunk_text: str) -> bool:
    return any(ind in chunk_text for ind in _JUNK_INDICATORS)


def chunk_paper(cleaned_markdown: str, meta: dict) -> list[Document]:
    """
    Two-stage split: Markdown headers first (section boundaries), then a
    recursive character splitter within any section still too long.
    Drops low-signal / junk chunks.
    """
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS)
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )

    section_docs = header_splitter.split_text(cleaned_markdown)
    chunks = []

    for section_doc in section_docs:
        section_label = " > ".join(
            v for k, v in section_doc.metadata.items() if k in ("h1", "h2", "h3")
        ) or "Front Matter"

        sub_chunks = sub_splitter.split_text(section_doc.page_content)

        for j, chunk_text in enumerate(sub_chunks):
            if _is_low_signal(chunk_text) or _is_junk(chunk_text):
                continue
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "type": "prose",
                    "source": meta["source"],
                    "title": meta["title"],
                    "section": section_label,
                    "chunk_index": j,
                }
            ))

    return chunks
