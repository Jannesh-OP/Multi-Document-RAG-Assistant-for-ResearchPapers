"""
PDF ingestion: layout-aware parsing, metadata extraction, and table extraction.
"""
import os
import re
import uuid

import fitz  # PyMuPDF
import pymupdf4llm
from langchain_core.documents import Document


def parse_pdf(pdf_path: str) -> str:
    """Layout-aware PDF -> Markdown parse (tables render as Markdown tables inline)."""
    return pymupdf4llm.to_markdown(pdf_path)


def extract_metadata(pdf_path: str) -> dict:
    """Pull title/authors from PDF metadata, fall back to first-page heuristics."""
    doc = fitz.open(pdf_path)
    meta = doc.metadata
    first_page_text = doc[0].get_text()

    title = meta.get("title") or ""
    if not title.strip():
        lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]
        title = lines[0] if lines else os.path.basename(pdf_path)

    authors = meta.get("author") or "Unknown"

    abstract_match = re.search(
        r"Abstract[:\s]*(.*?)(?=\n\s*(1\.|Introduction|Keywords))",
        first_page_text, re.DOTALL | re.IGNORECASE
    )
    abstract = abstract_match.group(1).strip()[:800] if abstract_match else ""

    num_pages = doc.page_count
    doc.close()

    return {
        "source": os.path.basename(pdf_path),
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "num_pages": num_pages,
    }


def extract_tables_from_markdown(markdown_text: str, source_meta: dict):
    """
    Detect Markdown table blocks and pull them out as standalone Documents,
    tagged type='table'. Returns (table_documents, markdown_with_tables_removed).
    """
    table_pattern = re.compile(
        r'((?:\|.*\|\r?\n)+\|[\s\-:|]+\|\r?\n(?:\|.*\|\r?\n?)+)'
    )
    tables_found = table_pattern.findall(markdown_text)

    table_docs = []
    cleaned_text = markdown_text

    for i, table_md in enumerate(tables_found):
        table_id = str(uuid.uuid4())[:8]

        idx = markdown_text.find(table_md)
        preceding_text = markdown_text[max(0, idx - 200):idx]
        caption_match = re.search(r'(Table\s+\d+[:\.]?.*)', preceding_text, re.IGNORECASE)
        caption = caption_match.group(1).strip() if caption_match else f"Table {i+1}"

        table_docs.append(Document(
            page_content=table_md.strip(),
            metadata={
                "type": "table",
                "table_id": table_id,
                "caption": caption,
                "source": source_meta["source"],
                "title": source_meta["title"],
            }
        ))
        cleaned_text = cleaned_text.replace(table_md, f"\n[TABLE_REF:{table_id} - {caption}]\n")

    return table_docs, cleaned_text

def build_abstract_document(meta: dict):
    """The abstract is extracted into metadata but was never indexed — fix that
    by giving it its own retrievable Document, tagged type='abstract'."""
    if not meta.get("abstract"):
        return None
    return Document(
        page_content=meta["abstract"],
        metadata={
            "type": "abstract",
            "source": meta["source"],
            "title": meta["title"],
            "section": "Abstract",
        }
    )


def clean_markdown_artifacts(text: str) -> str:
    """Strip OCR/logo/watermark noise that pymupdf4llm sometimes picks up."""
    text = re.sub(r'<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->',
                   '', text, flags=re.DOTALL)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'<mark>(.*?)</mark>', r'\1', text)
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'~~.*?~~', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_tables_pymupdf(pdf_path, source_meta):
    doc = fitz.open(pdf_path)
    table_docs = []
    for page_num, page in enumerate(doc):
        for strategy in ["lines_strict", "lines", "text"]:
            tabs = page.find_tables(strategy=strategy)
            if tabs.tables:
                break
        for t in tabs.tables:
            rows = t.extract()
            markdown = "\n".join(" | ".join(str(c or "") for c in row) for row in rows)
            table_docs.append(Document(
                page_content=markdown,
                metadata={"type": "table", "source": source_meta["source"],
                          "title": source_meta["title"], "page": page_num + 1}
            ))
    doc.close()
    return table_docs
