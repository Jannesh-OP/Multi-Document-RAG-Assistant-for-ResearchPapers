"""
Streamlit UI for the Intelligent Research Paper Assistant.

Upload PDFs -> click Process (blocks until indexed) -> ask unlimited
questions against that index without reindexing.
"""
import os
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Research Paper Assistant", page_icon="📄", layout="wide")
st.title("📄 Intelligent Research Paper Assistant")
st.caption("Upload papers, then ask questions — including comparisons across multiple papers.")

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
    st.session_state.paper_titles = []
    st.session_state.chat_history = []

# ---------------- Upload + Process ----------------
with st.sidebar:
    st.header("1. Upload Papers")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process Documents", type="primary", disabled=not uploaded_files):
        with st.spinner("Parsing PDFs, extracting tables, building index... (~10-30s)"):
            files_payload = [
                ("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files
            ]
            try:
                resp = requests.post(f"{BACKEND_URL}/upload", files=files_payload, timeout=600)
                resp.raise_for_status()
                result = resp.json()

                st.session_state.index_ready = True
                st.session_state.paper_titles = result["paper_titles"]
                st.session_state.chat_history = []  # reset chat on new corpus

                st.success(
                    f"Indexed {result['papers_processed']} paper(s) — "
                    f"{result['prose_chunks']} text chunks, {result['table_chunks']} tables."
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Upload failed: {e}")

    if st.session_state.index_ready:
        st.divider()
        st.subheader("Indexed Papers")
        for t in st.session_state.paper_titles:
            st.write(f"- {t}")

# ---------------- Chat ----------------
st.header("2. Ask Questions")

if not st.session_state.index_ready:
    st.info("Upload and process at least one PDF to start asking questions.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['title']}** — *{s['section']}* ({s['type']})")

    question = st.chat_input("Ask a question, e.g. 'Compare the forecasting methods across these papers'")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and reasoning..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/query",
                        json={"question": question, "top_n": 5},
                        timeout=300,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    st.markdown(result["answer"])
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.markdown(f"**{s['title']}** — *{s['section']}* ({s['type']})")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                except requests.exceptions.RequestException as e:
                    st.error(f"Query failed: {e}")
