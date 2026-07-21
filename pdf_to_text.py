"""
pdf_to_text.py

Extracts text from PDF files page by page.
Uses pdfplumber as primary extractor (handles tables and multi-column layouts).
Falls back to pypdf if pdfplumber fails or returns empty text.

Returns a list of page dicts:
[
    {
        "filename": "Invoice_INV-2024-001.pdf",
        "page":     1,
        "text":     "INVOICE\n# INV-2024-001\n..."
    },
    ...
]
"""

import os
import pdfplumber
from pypdf import PdfReader


# Primary extractor pdfplumber
def extract_with_pdfplumber(filepath: str) -> list[dict]:
    """
    Extract text from each page using pdfplumber.
    Better than pypdf for tables and multi-column layouts.
    """
    pages = []
    filename = os.path.basename(filepath)

    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text and text.strip():
                pages.append({
                    "filename": filename,
                    "page":     i + 1,
                    "text":     text.strip()
                })

    return pages


# Fallback extractor  pypdf 
def extract_with_pypdf(filepath: str) -> list[dict]:
    """
    Fallback extractor using pypdf.
    Used when pdfplumber returns empty text (e.g. scanned PDFs).
    Note: pypdf reads text in stream order — may interleave columns.
    """
    pages = []
    filename = os.path.basename(filepath)
    reader = PdfReader(filepath)

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "filename": filename,
                "page":     i + 1,
                "text":     text.strip()
            })

    return pages


#  Main function
def extract_pdf(filepath: str) -> list[dict]:
    """
    Extract text from a PDF file.

    Tries pdfplumber first. If it returns no pages or empty text,
    falls back to pypdf.

    Args:
        filepath: path to the PDF file

    Returns:
        list of page dicts with keys: filename, page, text

    Raises:
        FileNotFoundError: if the PDF does not exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath}")

    print(f"Extracting: {filepath}")

    # Try pdfplumber first
    try:
        pages = extract_with_pdfplumber(filepath)
        if pages:
            print(f"  pdfplumber -> {len(pages)} page(s) extracted")
            return pages
        else:
            print(f"  pdfplumber returned empty -- trying pypdf fallback")
    except Exception as e:
        print(f"  pdfplumber failed ({e}) -- trying pypdf fallback")

    # Fallback to pypdf
    try:
        pages = extract_with_pypdf(filepath)
        if pages:
            print(f"  pypdf -> {len(pages)} page(s) extracted")
            return pages
        else:
            print(f"  Warning: both extractors returned empty for {filepath}")
            return []
    except Exception as e:
        print(f"  pypdf also failed: {e}")
        return []


# Extract all PDFs in a folder
def extract_all(folder: str = "data/raw") -> list[dict]:
    """
    Extract text from all PDFs in a folder.
    Returns a flat list of page dicts across all PDFs.
    """
    all_pages = []

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return []

    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {folder}")
        return []

    print(f"Found {len(pdf_files)} PDF(s) in {folder}")

    for filename in sorted(pdf_files):
        filepath = os.path.join(folder, filename)
        pages = extract_pdf(filepath)
        all_pages.extend(pages)

    print(f"\nTotal pages extracted: {len(all_pages)}")
    return all_pages


# Quick inspection helper
def print_page_summary(pages: list[dict]) -> None:
    """Print a summary of extracted pages."""
    print(f"\nTotal pages: {len(pages)}")
    print("-" * 60)
    for p in pages:
        print(f"File : {p['filename']}  page {p['page']}")
        print(f"Text : {p['text'][:200]}...")
        print("-" * 60)


#  Main test with invoice 
if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/raw/Invoice_INV-2024-001.pdf"
    pages = extract_pdf(filepath)
    print_page_summary(pages)
