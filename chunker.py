"""
chunker.py

Splits extracted text into fixed-size overlapping chunks,
preserving metadata (filename, page number, section).

Each chunk is a dict:
{
    "id":        unique string (filename + page + chunk index),
    "text":      the chunk content,
    "filename":  source PDF filename,
    "page":      page number (1-based),
    "section":   detected section heading (or "unknown"),
    "chunk_index": position of this chunk within the page
}
"""

import re
import hashlib


# Configuration
CHUNK_SIZE    = 1000  # bigger — fits whole invoice
CHUNK_OVERLAP = 150 # overlap between consecutive chunks


# Section heading detection 
# Matches lines like "INVOICE SUMMARY", "Payment Terms:", "## Summary"
HEADING_PATTERN = re.compile(
    r"^(#{1,3}\s+.+|[A-Z][A-Z\s]{3,}:|[A-Z][a-z].{0,40}:)\s*$",
    re.MULTILINE
)

def detect_section(text: str) -> str:
    """Return the last section heading found before this text block, or 'unknown'."""
    matches = HEADING_PATTERN.findall(text)
    if matches:
        return matches[-1].strip().rstrip(":")
    return "unknown"


# Core chunking logic
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping fixed-size chunks.

    Example with chunk_size=20, overlap=5 and text="ABCDEFGHIJKLMNOPQRSTUVWXYZ":
      chunk 1: ABCDEFGHIJKLMNOPQRST   (0..20)
      chunk 2: PQRSTUVWXYZ            (15..26)  ← overlap of 5 chars
    """
    chunks = []
    start  = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary (. ! ?) or newline
        # so chunks don't cut mid-sentence
        if end < len(text):
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind(".\n", start, end),
                text.rfind("\n",  start, end),
            )
            if boundary > start + (chunk_size // 2):
                end = boundary + 1  # include the punctuation

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap  # step back by overlap for next chunk

    return chunks


#  Per-page chunker
def chunk_page(page: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Chunk a single page dict produced by pdf_to_text.py.

    Expected input format:
        {
            "filename": "invoice_march.pdf",
            "page":     1,
            "text":     "INVOICE\nDate: 2024-03-01\n..."
        }

    Returns a list of chunk dicts with full metadata.
    """
    text     = page.get("text", "").strip()
    filename = page.get("filename", "unknown")
    page_num = page.get("page", 0)

    if not text:
        return []

    section    = detect_section(text)
    raw_chunks = chunk_text(text, chunk_size, overlap)
    results    = []

    for i, chunk in enumerate(raw_chunks):
        # Stable unique ID: hash of filename + page + chunk index
        uid = hashlib.md5(f"{filename}-p{page_num}-c{i}".encode()).hexdigest()[:10]

        results.append({
            "id":          uid,
            "text":        chunk,
            "filename":    filename,
            "page":        page_num,
            "section":     section,
            "chunk_index": i,
        })

    return results


# Document-level chunker 
def chunk_document(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Chunk all pages of a document.

    Input: list of page dicts from pdf_to_text.py
    Output: flat list of chunk dicts across all pages
    """
    all_chunks = []

    for page in pages:
        chunks = chunk_page(page, chunk_size, overlap)
        all_chunks.extend(chunks)

    return all_chunks


#  Quick inspection helper 
def print_chunk_summary(chunks: list[dict]) -> None:
    """Print a quick summary of chunks — useful for eyeballing output."""
    print(f"Total chunks: {len(chunks)}")
    print("-" * 60)
    for c in chunks[:3]:   # show first 3 only
        print(f"ID       : {c['id']}")
        print(f"File     : {c['filename']}  page {c['page']}  chunk {c['chunk_index']}")
        print(f"Section  : {c['section']}")
        print(f"Text     : {c['text'][:120]}...")
        print("-" * 60)


#  Main test with a sample 
if __name__ == "__main__":
    # Simulate what pdf_to_text.py produces
    sample_pages = [
        {
            "filename": "invoice_march.pdf",
            "page": 1,
            "text": (
                "INVOICE\n"
                "Date: 2024-03-01\n"
                "Invoice Number: INV-2024-001\n\n"
                "Bill To:\nAccme Corp\n123 Business St\n\n"
                "Payment Terms:\n"
                "Payment is due within 30 days of invoice date. "
                "Late payments are subject to a 2% monthly interest charge. "
                "Please include the invoice number on your payment.\n\n"
                "Items:\n"
                "Consulting services - March 2024: €2,500.00\n"
                "Software licence fee: €500.00\n"
                "Total: €3,000.00\n"
            )
        },
        {
            "filename": "invoice_march.pdf",
            "page": 2,
            "text": (
                "Bank Details:\n"
                "Bank: Svenska Handelsbanken\n"
                "IBAN: SE00 0000 0000 0000 0000\n"
                "BIC: HANDSESS\n\n"
                "Thank you for your business."
            )
        }
    ]

    chunks = chunk_document(sample_pages)
    print_chunk_summary(chunks)
