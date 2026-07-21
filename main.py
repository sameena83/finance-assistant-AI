"""
main.py

Entry point for the finance assistant pipeline.
Run this to ingest PDFs and start the assistant.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def main():
    print("Finance Assistant — starting pipeline")
    print("Run individual modules:")
    print("  uv run python ingest.py     # index PDFs")
    print("  uv run streamlit run app.py # start UI")


if __name__ == "__main__":
    main()
