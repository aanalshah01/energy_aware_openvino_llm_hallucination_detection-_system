import os
from pypdf import PdfReader
from vector_db import KnowledgeBaseManager

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits academic text into overlapping chunks for retrieval."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)
    return chunks

def ingest_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    chunks = chunk_text(full_text)
    kb = KnowledgeBaseManager()
    kb.add_documents(chunks)
    print(f"Successfully ingested '{os.path.basename(pdf_path)}' ({len(chunks)} chunks).")

if __name__ == "__main__":
    # Example: Ingest an uploaded textbook or academic paper PDF
    ingest_pdf("dbms.pdf")
    pass