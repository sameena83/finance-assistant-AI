from pdf_to_text import extract_pdf
from chunker import chunk_document

pages  = extract_pdf("data/raw/1.pdf")
chunks = chunk_document(pages)

print(f"\nTotal chunks: {len(chunks)}")
print("-" * 60)
for c in chunks:
    print(f"ID      : {c['id']}")
    print(f"Section : {c['section']}")
    print(f"Page    : {c['page']}")
    print(f"Text    : {c['text'][:200]}")
    print("-" * 60)