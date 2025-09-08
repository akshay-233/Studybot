import os
import re
import hashlib
from typing import List, Tuple, Optional
from pypdf import PdfReader
from docx import Document as DocxDocument

def file_to_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md"]:
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            return f.read()
    if ext in [".pdf"]:
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n".join(parts)
    if ext in [".docx"]:
        doc = DocxDocument(path)
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Unsupported file type: {ext}")

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 450, overlap: int = 100) -> List[str]:
    """Roughly chunk by words with overlap to preserve context."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap if chunk_size - overlap > 0 else chunk_size
    return chunks

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]
