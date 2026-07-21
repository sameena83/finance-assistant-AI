"""
rag.py

Connects search and LLM together into a full RAG pipeline.

Flow:
    question → search → build_prompt → LLM → answer

Main function:
    answer = rag("what is the IBAN number?")
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from search import hybrid_search

load_dotenv()

# ── Config ──────────────────────────────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_K        = 3   # how many chunks to retrieve

# ── OpenAI client ───────────────────────────────────────────
openai_client = OpenAI()


# ══════════════════════════════════════════════════════════
# PART 1 — BUILD PROMPT
# ══════════════════════════════════════════════════════════

def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a readable context string.

    Each chunk is labelled with its source file and page
    so the LLM knows where the information came from.

    Example output:
        [Source: 1.pdf, page 1]
        INVOICE # INV-2024-001
        Date: Jul 20, 2026
        ...
    """
    context_parts = []

    for chunk in chunks:
        source = f"[Source: {chunk['filename']}, page {chunk['page']}]"
        context_parts.append(f"{source}\n{chunk['text']}")

    return "\n\n---\n\n".join(context_parts)


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    """
    Build the message list to send to the LLM.

    Returns a list of messages:
    - system message: instructions + retrieved context
    - user message: the question

    Why keep them separate?
    - System = instructions and context (what the LLM knows)
    - User   = the question (what the user wants)
    Mixing them breaks role separation and multi-turn patterns.
    """
    context = build_context(chunks)

    system_message = f"""You are a helpful financial document assistant.
Answer the user's question using ONLY the information from the context below.
If the answer is not in the context, say "I don't have that information in the provided documents."
Be concise and precise. For numbers and amounts, be exact.

CONTEXT:
{context}""".strip()

    return [
        {"role": "system",  "content": system_message},
        {"role": "user",    "content": question}
    ]


# ══════════════════════════════════════════════════════════
# PART 2 — CALL LLM
# ══════════════════════════════════════════════════════════

def ask_llm(messages: list[dict]) -> str:
    """
    Send messages to OpenAI and return the answer text.
    """
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0   # 0 = deterministic, factual answers
                        # good for finance — no hallucinated numbers
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════
# PART 3 — FULL RAG PIPELINE
# ══════════════════════════════════════════════════════════

def rag(question: str, top_k: int = TOP_K) -> dict:
    """
    Full RAG pipeline — one call, one answer.

    Args:
        question: user's natural language question
        top_k:    number of chunks to retrieve

    Returns:
        {
            "question": "what is the IBAN number?",
            "answer":   "The IBAN number is SE12 3456 7890 1234 5678",
            "sources":  [{"filename": "1.pdf", "page": 1, ...}]
        }
    """
    # Step 1 — retrieve relevant chunks
    chunks = hybrid_search(question, top_k=top_k)

    if not chunks:
        return {
            "question": question,
            "answer":   "No relevant documents found.",
            "sources":  []
        }

    # Step 2 — build prompt
    messages = build_prompt(question, chunks)

    # Step 3 — ask LLM
    answer = ask_llm(messages)

    # Step 4 — return answer + sources
    sources = [
        {
            "filename": c["filename"],
            "page":     c["page"],
            "section":  c["section"],
            "score":    round(c.get("score", 0), 3)
        }
        for c in chunks
    ]

    return {
        "question": question,
        "answer":   answer,
        "sources":  sources
    }


# ══════════════════════════════════════════════════════════
# MAIN — test with sample questions
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    questions = [
        "What is the IBAN number?",
        "What are the payment terms?",
        "How much is the total invoice amount?",
        "What happens if I pay late?",
        "Who is this invoice billed to?",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        result = rag(question)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Source: {result['sources'][0]['filename']} "
              f"page {result['sources'][0]['page']} "
              f"(score {result['sources'][0]['score']})")
"""
rag.py

Connects search and LLM together into a full RAG pipeline.

Flow:
    question → search → build_prompt → LLM → answer

Main function:
    answer = rag("what is the IBAN number?")
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from search import hybrid_search

load_dotenv()

# ── Config ──────────────────────────────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_K        = 3   # how many chunks to retrieve

# ── OpenAI client ───────────────────────────────────────────
openai_client = OpenAI()


# ══════════════════════════════════════════════════════════
# PART 1 — BUILD PROMPT
# ══════════════════════════════════════════════════════════

def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a readable context string.

    Each chunk is labelled with its source file and page
    so the LLM knows where the information came from.

    Example output:
        [Source: 1.pdf, page 1]
        INVOICE # INV-2024-001
        Date: Jul 20, 2026
        ...
    """
    context_parts = []

    for chunk in chunks:
        source = f"[Source: {chunk['filename']}, page {chunk['page']}]"
        context_parts.append(f"{source}\n{chunk['text']}")

    return "\n\n---\n\n".join(context_parts)


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    """
    Build the message list to send to the LLM.

    Returns a list of messages:
    - system message: instructions + retrieved context
    - user message: the question

    Why keep them separate?
    - System = instructions and context (what the LLM knows)
    - User   = the question (what the user wants)
    Mixing them breaks role separation and multi-turn patterns.
    """
    context = build_context(chunks)

    system_message = f"""You are a helpful financial document assistant.
Answer the user's question using ONLY the information from the context below.
If the answer is not in the context, say "I don't have that information in the provided documents."
Be concise and precise. For numbers and amounts, be exact.

CONTEXT:
{context}""".strip()

    return [
        {"role": "system",  "content": system_message},
        {"role": "user",    "content": question}
    ]


# ══════════════════════════════════════════════════════════
# PART 2 — CALL LLM
# ══════════════════════════════════════════════════════════

def ask_llm(messages: list[dict]) -> str:
    """
    Send messages to OpenAI and return the answer text.
    """
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0   # 0 = deterministic, factual answers
                        # good for finance — no hallucinated numbers
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════
# PART 3 — FULL RAG PIPELINE
# ══════════════════════════════════════════════════════════

def rag(question: str, top_k: int = TOP_K) -> dict:
    """
    Full RAG pipeline — one call, one answer.

    Args:
        question: user's natural language question
        top_k:    number of chunks to retrieve

    Returns:
        {
            "question": "what is the IBAN number?",
            "answer":   "The IBAN number is SE12 3456 7890 1234 5678",
            "sources":  [{"filename": "1.pdf", "page": 1, ...}]
        }
    """
    # Step 1 — retrieve relevant chunks
    chunks = hybrid_search(question, top_k=top_k)

    if not chunks:
        return {
            "question": question,
            "answer":   "No relevant documents found.",
            "sources":  []
        }

    # Step 2 — build prompt
    messages = build_prompt(question, chunks)

    # Step 3 — ask LLM
    answer = ask_llm(messages)

    # Step 4 — return answer + sources
    sources = [
        {
            "filename": c["filename"],
            "page":     c["page"],
            "section":  c["section"],
            "score":    round(c.get("score", 0), 3)
        }
        for c in chunks
    ]

    return {
        "question": question,
        "answer":   answer,
        "sources":  sources
    }


# ══════════════════════════════════════════════════════════
# MAIN — test with sample questions
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    questions = [
        "What is the IBAN number?",
        "What are the payment terms?",
        "How much is the total invoice amount?",
        "What happens if I pay late?",
        "Who is this invoice billed to?",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        result = rag(question)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Source: {result['sources'][0]['filename']} "
              f"page {result['sources'][0]['page']} "
              f"(score {result['sources'][0]['score']})")


# ══════════════════════════════════════════════════════════
# QUERY REWRITING (bonus point)
# ══════════════════════════════════════════════════════════

REWRITE_INSTRUCTIONS = """
You are helping improve a search query for a financial document assistant.
The user asked a question that may be vague or use informal language.
Rewrite it into a clear, specific search query that will find relevant 
information in financial documents like invoices, bank statements, and contracts.

Rules:
- Keep the same intent
- Make it more specific and searchable
- Use financial terminology where appropriate
- Return ONLY the rewritten question, nothing else
""".strip()


def rewrite_query(question: str) -> str:
    """
    Use LLM to rewrite a vague user question into a
    better search query.

    Example:
        "what about the bank?"
        → "What are the bank details, IBAN and BIC number?"

        "the contract thing"
        → "What are the key terms and conditions of the contract?"
    """
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_INSTRUCTIONS},
            {"role": "user",   "content": question}
        ],
        temperature=0
    )
    rewritten = response.choices[0].message.content.strip()
    return rewritten


def rag_with_rewrite(question: str, top_k: int = TOP_K) -> dict:
    """
    RAG pipeline with query rewriting.

    Same as rag() but rewrites the question first
    for better search results.

    Returns:
        {
            "question":          original question,
            "rewritten_question": rewritten question,
            "answer":            LLM answer,
            "sources":           source chunks
        }
    """
    # Step 1 — rewrite the question
    rewritten = rewrite_query(question)
    print(f"  Original:  {question}")
    print(f"  Rewritten: {rewritten}")

    # Step 2 — search with rewritten question
    chunks = hybrid_search(rewritten, top_k=top_k)

    if not chunks:
        return {
            "question":           question,
            "rewritten_question": rewritten,
            "answer":             "No relevant documents found.",
            "sources":            []
        }

    # Step 3 — build prompt using original question
    # (we search with rewritten but answer the original)
    messages = build_prompt(question, chunks)

    # Step 4 — ask LLM
    answer = ask_llm(messages)

    # Step 5 — return everything
    sources = [
        {
            "filename": c["filename"],
            "page":     c["page"],
            "section":  c["section"],
            "score":    round(c.get("score", 0), 3)
        }
        for c in chunks
    ]

    return {
        "question":           question,
        "rewritten_question": rewritten,
        "answer":             answer,
        "sources":            sources
    }


# ══════════════════════════════════════════════════════════
# TEST QUERY REWRITING
# ══════════════════════════════════════════════════════════

def test_rewriting():
    """Test query rewriting with vague questions."""
    vague_questions = [
        "what about the bank?",
        "the contract thing",
        "how much?",
        "payment stuff",
        "when?",
    ]

    print("\n=== QUERY REWRITING TEST ===\n")
    for q in vague_questions:
        rewritten = rewrite_query(q)
        print(f"Original:  {q}")
        print(f"Rewritten: {rewritten}")
        print()
