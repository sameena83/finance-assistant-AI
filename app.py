"""
app.py

Streamlit UI for the Finance Document Assistant.

Run with:
    uv run streamlit run app.py
"""

import os
import json
import datetime
import streamlit as st
from dotenv import load_dotenv

from pdf_to_text import extract_pdf
from chunker import chunk_document
from ingest import embed_chunks, build_text_index, build_vector_index
from rag import rag

load_dotenv()

# ── Config ──────────────────────────────────────────────────
UPLOAD_FOLDER   = "data/raw"
FEEDBACK_FILE   = "data/feedback.jsonl"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("data", exist_ok=True)


# ══════════════════════════════════════════════════════════
# PAGE SETUP
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Finance Document Assistant",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Finance Document Assistant")
st.caption("Upload financial PDFs and ask questions in plain English.")


# ══════════════════════════════════════════════════════════
# SESSION STATE
# store conversation history across reruns
# ══════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ══════════════════════════════════════════════════════════
# SIDEBAR — FILE UPLOAD
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📁 Upload Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Invoices, bank statements, contracts"
    )

    if uploaded_file:
        # save to data/raw/
        save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name not in st.session_state.indexed_files:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                try:
                    # extract → chunk → embed → index
                    pages  = extract_pdf(save_path)
                    chunks = chunk_document(pages)
                    chunks = embed_chunks(chunks)
                    build_text_index(chunks)
                    build_vector_index(chunks)

                    st.session_state.indexed_files.append(uploaded_file.name)
                    st.success(f"✅ {uploaded_file.name} indexed!")

                except Exception as e:
                    st.error(f"Failed to index: {e}")

    # show indexed files
    if st.session_state.indexed_files:
        st.divider()
        st.subheader("📄 Indexed files")
        for f in st.session_state.indexed_files:
            st.write(f"✅ {f}")

    # show pre-indexed files in data/raw
    existing = [
        f for f in os.listdir(UPLOAD_FOLDER)
        if f.endswith(".pdf")
        and f not in st.session_state.indexed_files
    ]
    if existing:
        st.divider()
        st.subheader("📂 Pre-indexed files")
        for f in existing:
            st.write(f"📄 {f}")


# ══════════════════════════════════════════════════════════
# CHAT HISTORY
# ══════════════════════════════════════════════════════════

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sources"):
                for s in message["sources"]:
                    st.write(
                        f"📄 {s['filename']} — "
                        f"page {s['page']} — "
                        f"score {s['score']}"
                    )


# ══════════════════════════════════════════════════════════
# QUESTION INPUT
# ══════════════════════════════════════════════════════════

question = st.chat_input("Ask a question about your documents...")

if question:
    # show user message
    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({
        "role":    "user",
        "content": question
    })

    # get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                result = rag(question)
                answer = result["answer"]
                sources = result["sources"]

                st.write(answer)

                with st.expander("📎 Sources"):
                    for s in sources:
                        st.write(
                            f"📄 {s['filename']} — "
                            f"page {s['page']} — "
                            f"score {s['score']}"
                        )

                # save to session
                st.session_state.last_result = result
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════
# FEEDBACK
# ══════════════════════════════════════════════════════════

def save_feedback(result: dict, feedback: str) -> None:
    """Save feedback to a JSONL file for monitoring."""
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question":  result["question"],
        "answer":    result["answer"],
        "sources":   result["sources"],
        "feedback":  feedback   # "positive" or "negative"
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


if st.session_state.last_result:
    st.divider()
    st.write("**Was this answer helpful?**")

    col1, col2 = st.columns([1, 8])

    with col1:
        if st.button("👍"):
            save_feedback(st.session_state.last_result, "positive")
            st.success("Thanks!")

    with col2:
        if st.button("👎"):
            save_feedback(st.session_state.last_result, "negative")
            st.info("Thanks for the feedback!")
