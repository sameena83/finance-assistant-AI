"""
ingest.py

Takes chunks from chunker.py, embeds them using OpenAI,
and stores everything in two indexes:

1. Text index (SQLite via sqlitesearch) — for keyword search
2. Vector index (SQLite manually) — for semantic search

Run this once after adding new PDFs:
    uv run python ingest.py
"""

import os
import json
import sqlite3
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from pdf_to_text import extract_all
from chunker import chunk_document

load_dotenv()

# ── Config ──────────────────────────────────────────────────
TEXT_DB_PATH   = os.getenv("DB_PATH", "data/finance.db")
VECTOR_DB_PATH = "data/vectors.db"
EMBED_MODEL    = "text-embedding-3-small"
PDF_FOLDER     = "data/raw"

# ── OpenAI client ───────────────────────────────────────────
openai_client = OpenAI()


# ══════════════════════════════════════════════════════════
# STEP 1 — EMBEDDING
# ══════════════════════════════════════════════════════════

def embed(text: str) -> list[float]:
    """
    Convert a text string into a vector (list of numbers)
    using OpenAI's embedding model.

    Example:
        "Payment due in 30 days" → [0.23, -0.87, 0.45, ...]
    """
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an 'embedding' field to each chunk.

    Input:
        [{"id": "abc", "text": "INVOICE...", ...}, ...]

    Output:
        [{"id": "abc", "text": "INVOICE...", "embedding": [0.23, ...], ...}, ...]
    """
    print(f"\nEmbedding {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embed(chunk["text"])
        print(f"  [{i+1}/{len(chunks)}] embedded chunk {chunk['id']}")

    return chunks


# ══════════════════════════════════════════════════════════
# STEP 2 — TEXT INDEX (keyword search)
# ══════════════════════════════════════════════════════════

def build_text_index(chunks: list[dict]) -> None:
    """
    Store chunks in SQLite for keyword search using sqlitesearch.

    This is the same TextSearchIndex pattern from Module 1 —
    just applied to PDF chunks instead of FAQ documents.
    """
    from sqlitesearch import TextSearchIndex

    print(f"\nBuilding text index → {TEXT_DB_PATH}")

    index = TextSearchIndex(
        text_fields=["text", "section", "filename"],
        keyword_fields=[],
        db_path=TEXT_DB_PATH
    )

    for chunk in chunks:
        index.add({
            "id":       chunk["id"],
            "text":     chunk["text"],
            "section":  chunk["section"],
            "filename": chunk["filename"],
            "page":     str(chunk["page"]),
        })
        print(f"  added: {chunk['id']} ({chunk['filename']} p{chunk['page']})")

    index.close()
    print(f"  text index done — {len(chunks)} chunks stored")


# ══════════════════════════════════════════════════════════
# STEP 3 — VECTOR INDEX (semantic search)
# ══════════════════════════════════════════════════════════

def build_vector_index(chunks: list[dict]) -> None:
    """
    Store chunk vectors in SQLite for semantic search.

    Each row stores:
    - chunk id
    - original text
    - metadata (filename, page, section)
    - embedding as bytes (serialized numpy array)

    Why bytes? SQLite can't store a Python list directly.
    We convert: list → numpy array → bytes → store
    And retrieve: bytes → numpy array → list
    """
    print(f"\nBuilding vector index → {VECTOR_DB_PATH}")

    # Create (or open) the SQLite database
    conn = sqlite3.connect(VECTOR_DB_PATH)
    cur  = conn.cursor()

    # Create the table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id        TEXT PRIMARY KEY,
            text      TEXT,
            filename  TEXT,
            page      INTEGER,
            section   TEXT,
            embedding BLOB        -- stored as bytes
        )
    """)

    for chunk in chunks:
        # Convert list of floats → numpy array → bytes
        vector_bytes = np.array(
            chunk["embedding"], dtype=np.float32
        ).tobytes()

        # Insert into table (replace if same id exists)
        cur.execute("""
            INSERT OR REPLACE INTO embeddings
                (id, text, filename, page, section, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            chunk["id"],
            chunk["text"],
            chunk["filename"],
            chunk["page"],
            chunk["section"],
            vector_bytes
        ))

        print(f"  stored vector: {chunk['id']}")

    conn.commit()
    conn.close()
    print(f"  vector index done — {len(chunks)} vectors stored")


# ══════════════════════════════════════════════════════════
# STEP 4 — COSINE SIMILARITY (for searching later)
# ══════════════════════════════════════════════════════════

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Measure how similar two vectors are.
    Returns a number between -1 and 1:
        1.0  = identical meaning
        0.0  = unrelated
       -1.0  = opposite meaning

    Used in search.py to find the most relevant chunks.
    """
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ══════════════════════════════════════════════════════════
# MAIN — run the full ingestion pipeline
# ══════════════════════════════════════════════════════════

def ingest(folder: str = PDF_FOLDER) -> None:
    """
    Full ingestion pipeline:
    1. Extract text from all PDFs in folder
    2. Chunk all pages
    3. Embed all chunks
    4. Store in text index (keyword search)
    5. Store in vector index (semantic search)
    """
    print("=" * 60)
    print("FINANCE ASSISTANT — INGESTION PIPELINE")
    print("=" * 60)

    # Step 1 — extract
    print(f"\nStep 1: Extracting PDFs from {folder}")
    pages = extract_all(folder)

    if not pages:
        print("No pages extracted. Add PDFs to data/raw/ and try again.")
        return

    # Step 2 — chunk
    print(f"\nStep 2: Chunking {len(pages)} page(s)")
    chunks = chunk_document(pages)
    print(f"  {len(chunks)} chunks created")

    # Step 3 — embed
    print(f"\nStep 3: Embedding chunks")
    chunks = embed_chunks(chunks)

    # Step 4 — text index
    print(f"\nStep 4: Building text index")
    build_text_index(chunks)

    # Step 5 — vector index
    print(f"\nStep 5: Building vector index")
    build_vector_index(chunks)

    print("\n" + "=" * 60)
    print(f"DONE — {len(chunks)} chunks indexed and ready to search")
    print("=" * 60)


if __name__ == "__main__":
    ingest()
