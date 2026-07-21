"""
search.py

Two search modes:
1. Keyword search  — finds exact word matches (sqlitesearch)
2. Semantic search — finds meaning matches (cosine similarity)
3. Hybrid search   — combines both for best results

Usage:
    from search import hybrid_search
    results = hybrid_search("what are the payment terms?")
"""

import os
import sqlite3
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config ──────────────────────────────────────────────────
TEXT_DB_PATH   = os.getenv("DB_PATH", "data/finance.db")
VECTOR_DB_PATH = "data/vectors.db"
EMBED_MODEL    = "text-embedding-3-small"
TOP_K          = 5   # how many results to return

# ── OpenAI client ───────────────────────────────────────────
openai_client = OpenAI()


# ══════════════════════════════════════════════════════════
# HELPER — embed a question
# ══════════════════════════════════════════════════════════

def embed_question(question: str) -> list[float]:
    """
    Convert user question into a vector.
    Same model used in ingest.py — must match!
    """
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=question
    )
    return response.data[0].embedding


# ══════════════════════════════════════════════════════════
# SEARCH 1 — KEYWORD SEARCH
# ══════════════════════════════════════════════════════════

def keyword_search(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Search chunks by exact keywords using sqlitesearch.

    Good for: invoice numbers, names, specific terms
    Example:  "INV-2024-001" → finds the chunk mentioning it
    """
    from sqlitesearch import TextSearchIndex

    index = TextSearchIndex(
        text_fields=["text", "section", "filename"],
        keyword_fields=[],
        db_path=TEXT_DB_PATH
    )

    results = index.search(
        question,
        boost_dict={"text": 2.0, "section": 0.5},
        num_results=top_k
    )

    index.close()

    # add search type label
    for r in results:
        r["search_type"] = "keyword"

    return results


# ══════════════════════════════════════════════════════════
# SEARCH 2 — SEMANTIC SEARCH
# ══════════════════════════════════════════════════════════

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Measure similarity between two vectors.
    Returns 0.0 (unrelated) to 1.0 (identical meaning).
    """
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_search(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Search chunks by meaning using vector similarity.

    Good for: paraphrased questions, synonyms
    Example:  "when do I need to pay?" → finds payment terms chunk
    """
    # Step 1 — embed the question
    question_vector = embed_question(question)

    # Step 2 — load all stored vectors from SQLite
    conn = sqlite3.connect(VECTOR_DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT id, text, filename, page, section, embedding FROM embeddings")
    rows = cur.fetchall()
    conn.close()

    # Step 3 — compare question vector against every chunk vector
    scored = []

    for row in rows:
        chunk_id, text, filename, page, section, vector_bytes = row

        # convert bytes back to list of floats
        chunk_vector = np.frombuffer(vector_bytes, dtype=np.float32).tolist()

        # calculate similarity score
        score = cosine_similarity(question_vector, chunk_vector)

        scored.append({
            "id":          chunk_id,
            "text":        text,
            "filename":    filename,
            "page":        page,
            "section":     section,
            "score":       score,
            "search_type": "semantic"
        })

    # Step 4 — sort by score (highest first) and return top K
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ══════════════════════════════════════════════════════════
# SEARCH 3 — HYBRID SEARCH (keyword + semantic combined)
# ══════════════════════════════════════════════════════════

def hybrid_search(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Combine keyword and semantic search results.

    Strategy:
    - Run both searches
    - Merge results (deduplicate by chunk id)
    - Prefer chunks that appear in BOTH searches
    - Return top K

    This is the main function to use in rag.py
    """
    keyword_results  = keyword_search(question,  top_k=top_k)
    semantic_results = semantic_search(question, top_k=top_k)

    # collect keyword result IDs for boosting
    keyword_ids = {r["id"] for r in keyword_results}

    # merge — semantic results as base, boost if also in keyword
    seen    = set()
    merged  = []

    for result in semantic_results:
        if result["id"] not in seen:
            # boost score if chunk appears in keyword results too
            if result["id"] in keyword_ids:
                result["score"]       += 0.1
                result["search_type"]  = "hybrid"
            merged.append(result)
            seen.add(result["id"])

    # add any keyword results not already in merged
    for result in keyword_results:
        if result["id"] not in seen:
            merged.append(result)
            seen.add(result["id"])

    # re-sort after boosting
    merged.sort(key=lambda x: x.get("score", 0), reverse=True)

    return merged[:top_k]


# ══════════════════════════════════════════════════════════
# MAIN — test all three search modes
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    questions = [
        "what are the payment terms?",
        "what is the IBAN number?",
        "how much is the total invoice amount?",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print("=" * 60)

        results = hybrid_search(question, top_k=2)

        for i, r in enumerate(results):
            print(f"\nResult {i+1} [{r['search_type']}] score={r.get('score', 'n/a'):.3f}")
            print(f"File   : {r['filename']}  page {r['page']}")
            print(f"Section: {r['section']}")
            print(f"Text   : {r['text'][:200]}")
