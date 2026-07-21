"""
evaluate.py

Evaluation pipeline for the Finance Document Assistant.

Three parts:
1. Generate ground truth (A -> Q* pattern from Module 4)
2. Evaluate retrieval (hit rate + MRR)
3. Evaluate answer quality (LLM-as-a-judge)

Run with:
    uv run python evaluate.py
"""

import os
import json
import sqlite3
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

from search import hybrid_search, keyword_search, semantic_search
from rag import rag

load_dotenv()

# ── Config ──────────────────────────────────────────────────
GROUND_TRUTH_PATH = "data/ground_truth/ground_truth.csv"
VECTOR_DB_PATH    = "data/vectors.db"
EVAL_RESULTS_PATH = "data/ground_truth/eval_results.csv"
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

os.makedirs("data/ground_truth", exist_ok=True)

# ── OpenAI client ───────────────────────────────────────────
openai_client = OpenAI()


# ══════════════════════════════════════════════════════════
# PART 1 — GENERATE GROUND TRUTH
# ══════════════════════════════════════════════════════════

class Questions(BaseModel):
    questions: list[str]


DATA_GEN_INSTRUCTIONS = """
You are evaluating a financial document assistant.
Given a chunk of text from a financial document (invoice, bank statement, contract),
generate 5 questions a user might ask that can be answered from this text.

Rules:
- Questions should be natural, like a real user would ask
- Use different words than what appears in the text where possible
- Questions should be specific enough to have a clear answer
- Mix question types: amounts, dates, names, terms, conditions
""".strip()


def generate_questions_for_chunk(chunk: dict) -> tuple[list[dict], object]:
    """
    Generate 5 questions for one chunk using the A -> Q* pattern.
    Returns (records, usage)
    """
    user_prompt = json.dumps({
        "text":     chunk["text"],
        "filename": chunk["filename"],
        "section":  chunk["section"]
    })

    messages = [
        {"role": "system", "content": DATA_GEN_INSTRUCTIONS},
        {"role": "user",   "content": user_prompt}
    ]

    response = openai_client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=messages,
        response_format=Questions
    )

    questions = response.choices[0].message.parsed.questions
    usage     = response.usage

    records = [
        {"question": q, "chunk_id": chunk["id"], "filename": chunk["filename"]}
        for q in questions
    ]

    return records, usage


def load_chunks_from_db() -> list[dict]:
    """Load all indexed chunks from the vector database."""
    conn = sqlite3.connect(VECTOR_DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT id, text, filename, page, section FROM embeddings")
    rows = cur.fetchall()
    conn.close()

    return [
        {"id": r[0], "text": r[1], "filename": r[2], "page": r[3], "section": r[4]}
        for r in rows
    ]


def generate_ground_truth() -> pd.DataFrame:
    """
    Generate ground truth dataset from all indexed chunks.
    Saves to data/ground_truth/ground_truth.csv
    """
    print("\n=== PART 1: Generating Ground Truth ===")

    chunks = load_chunks_from_db()
    print(f"Found {len(chunks)} chunks in index")

    all_records = []
    all_usages  = []

    # use ThreadPoolExecutor for parallel generation (Module 4 pattern)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(generate_questions_for_chunk, chunk): chunk for chunk in chunks}

        for future in tqdm(futures, total=len(chunks), desc="Generating questions"):
            try:
                records, usage = future.result()
                all_records.extend(records)
                all_usages.append(usage)
            except Exception as e:
                print(f"  Error: {e}")

    df = pd.DataFrame(all_records)
    df.to_csv(GROUND_TRUTH_PATH, index=False)

    total_cost = sum(
        (u.prompt_tokens * 0.00015 + u.completion_tokens * 0.0006) / 1000
        for u in all_usages
    )

    print(f"Generated {len(df)} questions from {len(chunks)} chunks")
    print(f"Saved to {GROUND_TRUTH_PATH}")
    print(f"Estimated cost: ${total_cost:.4f}")

    return df


# ══════════════════════════════════════════════════════════
# PART 2 — EVALUATE RETRIEVAL
# ══════════════════════════════════════════════════════════

def hit_rate(ground_truth: pd.DataFrame, search_fn, top_k: int = 5) -> float:
    """
    Hit Rate = % of questions where the correct chunk
               appears in the top K results.

    Example: 8 out of 10 questions → hit rate = 0.80
    """
    hits = 0

    for _, row in ground_truth.iterrows():
        results    = search_fn(row["question"], top_k=top_k)
        result_ids = [r["id"] for r in results]

        if row["chunk_id"] in result_ids:
            hits += 1

    return hits / len(ground_truth)


def mrr(ground_truth: pd.DataFrame, search_fn, top_k: int = 5) -> float:
    """
    MRR (Mean Reciprocal Rank) = average of 1/rank
    where rank is the position of the correct chunk.

    Example:
        correct chunk at rank 1 → score 1.0
        correct chunk at rank 2 → score 0.5
        correct chunk at rank 3 → score 0.33
        not found               → score 0.0
    """
    scores = []

    for _, row in ground_truth.iterrows():
        results    = search_fn(row["question"], top_k=top_k)
        result_ids = [r["id"] for r in results]

        if row["chunk_id"] in result_ids:
            rank = result_ids.index(row["chunk_id"]) + 1
            scores.append(1 / rank)
        else:
            scores.append(0)

    return sum(scores) / len(scores)


def evaluate_retrieval(ground_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Compare three search approaches:
    - Keyword search
    - Semantic search
    - Hybrid search (combined)
    """
    print("\n=== PART 2: Evaluating Retrieval ===")

    search_methods = {
        "keyword":  keyword_search,
        "semantic": semantic_search,
        "hybrid":   hybrid_search,
    }

    results = []

    for name, fn in search_methods.items():
        print(f"  Evaluating {name} search...")
        hr  = hit_rate(ground_truth, fn)
        mrr_score = mrr(ground_truth, fn)

        results.append({
            "method":    name,
            "hit_rate":  round(hr, 3),
            "mrr":       round(mrr_score, 3)
        })

        print(f"    Hit Rate: {hr:.3f}  |  MRR: {mrr_score:.3f}")

    df = pd.DataFrame(results)
    print("\nRetrieval Evaluation Summary:")
    print(df.to_string(index=False))

    return df


# ══════════════════════════════════════════════════════════
# PART 3 — EVALUATE ANSWER QUALITY (LLM-as-a-judge)
# ══════════════════════════════════════════════════════════

class Judgement(BaseModel):
    relevance:   str    # RELEVANT / PARTLY_RELEVANT / NOT_RELEVANT
    explanation: str


JUDGE_INSTRUCTIONS = """
You are evaluating a financial document assistant.
Given a question and an answer, judge if the answer correctly addresses the question.

Respond with:
- relevance: RELEVANT (fully correct), PARTLY_RELEVANT (partially correct), or NOT_RELEVANT (wrong/missing)
- explanation: one sentence explaining your judgment
""".strip()


def judge_answer(question: str, answer: str) -> Judgement:
    """Ask the LLM to judge if an answer is correct."""
    response = openai_client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_INSTRUCTIONS},
            {"role": "user",   "content": f"Question: {question}\nAnswer: {answer}"}
        ],
        response_format=Judgement
    )
    return response.choices[0].message.parsed


def evaluate_answers(ground_truth: pd.DataFrame, sample_size: int = 10) -> pd.DataFrame:
    """
    Evaluate answer quality for a sample of questions.
    Uses LLM-as-a-judge pattern from Module 4.
    """
    print(f"\n=== PART 3: Evaluating Answer Quality (sample={sample_size}) ===")

    sample  = ground_truth.sample(min(sample_size, len(ground_truth)), random_state=42)
    records = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Judging answers"):
        try:
            result   = rag(row["question"])
            answer   = result["answer"]
            judgment = judge_answer(row["question"], answer)

            records.append({
                "question":    row["question"],
                "answer":      answer,
                "relevance":   judgment.relevance,
                "explanation": judgment.explanation
            })
        except Exception as e:
            print(f"  Error: {e}")

    df = pd.DataFrame(records)

    # print summary
    counts = df["relevance"].value_counts()
    print("\nAnswer Quality Summary:")
    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {label}: {count} ({pct:.0f}%)")

    df.to_csv(EVAL_RESULTS_PATH, index=False)
    print(f"\nSaved to {EVAL_RESULTS_PATH}")

    return df


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("FINANCE ASSISTANT — EVALUATION PIPELINE")
    print("=" * 60)

    # Part 1 — generate ground truth
    if os.path.exists(GROUND_TRUTH_PATH):
        print(f"\nLoading existing ground truth from {GROUND_TRUTH_PATH}")
        ground_truth = pd.read_csv(GROUND_TRUTH_PATH)
        print(f"  {len(ground_truth)} questions loaded")
    else:
        ground_truth = generate_ground_truth()

    # Part 2 — evaluate retrieval
    retrieval_results = evaluate_retrieval(ground_truth)

    # Part 3 — evaluate answer quality
    answer_results = evaluate_answers(ground_truth, sample_size=5)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
