### 💼 Personal Finance Document Assistant

A RAG-based assistant that lets you ask natural language questions about your financial PDFs — invoices, bank statements, and contracts. Built as a capstone project for the DataTalks.Club LLM Zoomcamp 2026.


## Problem Statement

Managing financial documents is tedious. Finding specific information — payment terms, IBAN numbers, contract deadlines, spending totals — requires manually reading through multiple PDFs.

This assistant lets you ask plain English questions and get precise answers grounded in your actual documents:


"When is payment due?" → "Payment is due within 30 days of invoice date (Net 30)."




"What is the IBAN number?" → "The IBAN is SE12 3456 7890 1234 5678."




"When does the contract end?" → "The contract ends on December 31, 2026."
## Stack

- **LLM**: OpenAI API (gpt-4o-mini)
- **PDF extraction**: pdfplumber (primary), pypdf (fallback)
- **Embeddings**: nomic-embed-text / sentence-transformers
- **Search**: sqlitesearch (keyword) + vector search (semantic)
- **UI**: Streamlit
- **Orchestration**: Docker + docker-compose
- **Evaluation**: ground truth generation + hit rate + MRR + LLM-as-a-judge

## Project Structure

```
finance-assistant/
├── data/
│   ├── raw/                    # upload PDFs here
│   ├── extracted/              # text extracted from PDFs
│   └── ground_truth/           # ground_truth.csv for evaluation
├── pdf_to_text.py              # PDF extraction
├── chunker.py                  # fixed-size overlapping chunks
├── ingest.py                   # embed + index chunks
├── search.py                   # build_index() + search_documents()
├── rag.py                      # build_prompt() + LLM call
├── agent.py                    # agentic loop
├── evaluate.py                 # hit rate, MRR, LLM-as-a-judge
├── app.py                      # Streamlit UI
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .python-version
├── .env                        # never commit this
├── .env.example
├── .gitignore
└── README.md
```
## Screenshots
### Chat Interface
<img width="1512" height="784" alt="Screenshot 2026-07-21 at 12 24 41" src="https://github.com/user-attachments/assets/25b5a0ec-770d-4a27-bb1d-b148587727da" />

### Monitoring Dashboard
<img width="1512" height="784" alt="Screenshot 2026-07-21 at 12 16 33" src="https://github.com/user-attachments/assets/b4c673e9-aa21-4bbb-b8c5-17f7200d7f5b" />

### Dashboard Charts
<img width="1512" height="784" alt="Screenshot 2026-07-21 at 12 16 50" src="https://github.com/user-attachments/assets/8df3cc03-b2a5-456e-a559-a0ae7a5c6b1b" />

### With Query rewriting
<img width="1512" height="784" alt="Screenshot 2026-07-21 at 12 37 48" src="https://github.com/user-attachments/assets/8309594d-9936-43b0-862f-b9ff1cd55cbf" />




## Quickstart

### 1. Clone and set up environment

```bash
git clone <your-repo>
cd finance-assistant
uv venv
source .venv/bin/activate
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Add your PDFs

```bash
cp your_invoice.pdf data/raw/
```

### 4. Run locally

```bash
uv run python ingest.py          # build the index
uv run streamlit run app.py      # start the UI
```

### 5. Run with Docker

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501)

## Sample questions

- "How much did I spend on consulting in April 2026?"
- "What are the payment terms on invoice INV-2024-001?"
- "What is the IBAN for Svenska Handelsbanken?"
- "What was the tax amount on the latest invoice?"
- "Summarise the key obligations in this contract."

## Built as part of

[DataTalks.Club LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp)
