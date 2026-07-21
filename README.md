# 💼 Personal Finance Document Assistant

A RAG-based assistant that lets you ask natural language questions about your financial PDFs — invoices, bank statements, and contracts. Built as a capstone project for the [DataTalks.Club LLM Zoomc[...]

---

## App Demo
[![Demo](screenshots/demo.webm)](https://github.com/sameena83/finance-assistant-AI/raw/main/screenshots/demo.webm
)

## Live Demo

🚀 https://finance-assistant-ai-uhdbxt4rhhhkq7oddpthx4.streamlit.app/



Deployed on Streamlit Cloud — free, no login required.
## Problem Statement

Managing financial documents is tedious. Finding specific information — payment terms, IBAN numbers, contract deadlines, spending totals — requires manually reading through multiple PDFs.

This assistant lets you ask plain English questions and get precise answers grounded in your actual documents:

> *"When is payment due?"* → "Payment is due within 30 days of invoice date (Net 30)."

> *"What is the IBAN number?"* → "The IBAN is SE12 3456 7890 1234 5678."

> *"When does the contract end?"* → "The contract ends on December 31, 2026."

---

## Screenshots
### Chat Interface
<img width="1512" height="784" alt="Screenshot 2026-07-21 at 12 24 41" src="https://github.com/user-attachments/assets/25b5a0ec-770d-4a27-bb1d-b148587727da" />

### Monitoring Dashboard
<img width="1512" height="784" alt="Screenshot 2026-07-21 at 12 16 33" src="https://github.com/user-attachments/assets/b4c673e9-aa21-4bbb-b8c5-17f7200d7f5b" />

### Dashboard Charts
<img width="1512" height="784" alt="Screenshot 2026-07-21 at 12 16 50" src="https://github.com/user-attachments/assets/8df3cc03-b2a5-456e-a559-a0ae7a5c6b1b" />

### With Query rewriting
<img width="1512" height="784" alt="Screenshot 2026-07-21 at 12 37 48" src="https://github.com/user-attachments/assets/8309594d-9936-43b0-862f-b9ff1cd55cbf" />



---

## Architecture

```
PDFs (invoices, statements, contracts)
        ↓
pdf_to_text.py      — extract text with pdfplumber
        ↓
chunker.py          — split into overlapping chunks (1000 chars, 150 overlap)
        ↓
ingest.py           — embed with OpenAI + store in SQLite
        ↓
search.py           — hybrid search (keyword + semantic)
        ↓
rag.py              — build prompt + call LLM
        ↓
app.py              — Streamlit chat interface
```

---

## Features

- **PDF ingestion** — drag and drop invoices, bank statements, contracts
- **Hybrid search** — combines keyword search and semantic vector search
- **Query rewriting** — LLM rewrites vague questions for better search results
- **Source citations** — every answer shows which document it came from
- **Evaluation pipeline** — hit rate, MRR, and LLM-as-a-judge scoring
- **Monitoring dashboard** — 5 live charts tracking usage and feedback
- **User feedback** — thumbs up/down on every answer
- **Docker ready** — full stack with docker-compose

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI API (gpt-4o-mini) |
| Embeddings | OpenAI text-embedding-3-small |
| PDF extraction | pdfplumber (primary), pypdf (fallback) |
| Keyword search | sqlitesearch (TextSearchIndex) |
| Vector search | SQLite + cosine similarity |
| UI | Streamlit |
| Monitoring | Streamlit dashboard |
| Containerization | Docker + docker-compose |
| Package manager | uv |

---

## Project Structure

```
finance-assistant/
├── data/
│   ├── raw/                    # put your PDFs here
│   ├── extracted/              # extracted text (auto-created)
│   └── ground_truth/           # evaluation results
├── screenshots/                # app screenshots
├── pdf_to_text.py              # PDF text extraction
├── chunker.py                  # fixed-size overlapping chunking
├── ingest.py                   # embed + build search index
├── search.py                   # keyword + semantic + hybrid search
├── rag.py                      # prompt builder + LLM + query rewriting
├── evaluate.py                 # retrieval metrics + LLM-as-a-judge
├── app.py                      # Streamlit chat UI
├── monitor.py                  # monitoring dashboard
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .python-version
├── .env.example
├── .gitignore
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

### 1. Clone the repo

```bash
git clone https://github.com/sameena83/finance-assistant-AI.git
cd finance-assistant-AI
```

### 2. Set up environment

```bash
uv venv
source .venv/bin/activate    # Mac/Linux
# or
.venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
uv add openai pdfplumber pypdf python-dotenv sqlitesearch \
       pandas tqdm streamlit pydantic numpy
```

### 4. Configure API key

```bash
cp .env.example .env
# open .env and add your OPENAI_API_KEY
```

### 5. Add sample PDFs

```bash
cp your_invoice.pdf data/raw/
cp your_bank_statement.pdf data/raw/
cp your_contract.pdf data/raw/
```

### 6. Build the index

```bash
uv run python ingest.py
```

### 7. Start the app

```bash
# Terminal 1 — chat interface
uv run streamlit run app.py

# Terminal 2 — monitoring dashboard
uv run streamlit run monitor.py --server.port 8502
```

Open:
- Chat app: [http://localhost:8501](http://localhost:8501)
- Dashboard: [http://localhost:8502](http://localhost:8502)

---

## Run with Docker

```bash
# copy and fill in your API key
cp .env.example .env

# build and start
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501)

---

## Evaluation

Run the full evaluation pipeline:

```bash
uv run python evaluate.py
```

This will:
1. Generate ground truth questions from your documents (A → Q* pattern)
2. Evaluate retrieval — hit rate and MRR for keyword, semantic, and hybrid search
3. Evaluate answer quality using LLM-as-a-judge

### Sample results (3 documents)

| Method | Hit Rate | MRR |
|---|---|---|
| Keyword | 1.000 | 1.000 |
| Semantic | 1.000 | 1.000 |
| Hybrid | 1.000 | 1.000 |

Answer quality: **5/5 RELEVANT (100%)**

---

## Sample Questions

### Invoice questions
- "What is the total invoice amount?"
- "What is the IBAN number for payment?"
- "Who is this invoice billed to?"
- "What are the payment terms?"
- "What is the VAT number?"

### Bank statement questions
- "What is the closing balance?"
- "How much was received from Sam & Co AB?"
- "What was spent on software subscriptions?"
- "What is the total credit amount?"

### Contract questions
- "When does the contract end?"
- "What is the monthly retainer fee?"
- "What is the termination notice period?"
- "What happens if payment is late?"

---

## Evaluation Criteria Coverage

| Criterion | Implementation |
|---|---|
| Problem description | ✅ Clearly described above |
| Retrieval flow | ✅ Knowledge base + LLM |
| Retrieval evaluation | ✅ Keyword + semantic + hybrid compared |
| LLM evaluation | ✅ LLM-as-a-judge with RELEVANT/PARTLY/NOT scoring |
| Interface | ✅ Streamlit UI |
| Ingestion pipeline | ✅ Python script (ingest.py) |
| Monitoring | ✅ Dashboard with 5 charts + user feedback |
| Containerization | ✅ Full docker-compose |
| Reproducibility | ✅ All steps documented, versions pinned |
| Hybrid search | ✅ Keyword + semantic combined |
| Query rewriting | ✅ LLM rewrites vague questions |

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key | ✅ |
| `OPENAI_MODEL` | Model to use (default: gpt-4o-mini) | Optional |
| `DB_PATH` | SQLite text index path (default: data/finance.db) | Optional |
| `CHUNK_SIZE` | Chunk size in characters (default: 1000) | Optional |
| `CHUNK_OVERLAP` | Overlap between chunks (default: 150) | Optional |

---

## Built as part of

[DataTalks.Club LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp) — a free course on building LLM-powered applications.
