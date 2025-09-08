# Personalized Study Bot (RAG + Summarization + Quiz + Adaptive Learning)

An end‑to‑end Streamlit app that:
- Ingests a document (PDF/DOCX/TXT)
- Builds embeddings + FAISS index
- Answers student questions with Retrieval‑Augmented Generation (RAG)
- Summarizes answers for easy understanding
- Generates quizzes from the uploaded content
- Tracks performance in SQLite and retests weak areas with explanations

## ✨ Features
- Local, open‑source models by default (no API keys required)
- Optional OpenAI support for higher‑quality generation (set `OPENAI_API_KEY`)
- MCQ/True‑False/Fill‑in‑the‑Blank quizzes
- Explanations for wrong answers from source chunks
- Retest weak concepts until mastery

## 🧱 Tech
- Streamlit UI
- Sentence-Transformers (`all-MiniLM-L6-v2`) for embeddings
- FAISS for vector search
- FLAN‑T5 for summarization/generation (local; small & fast)
- SQLite for persistence
- Scikit‑learn TF‑IDF for non‑LLM quiz fallback

## 🚀 Quickstart
```bash
# 1) Create and activate a venv (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Use OpenAI for generation
export OPENAI_API_KEY=sk-...   # Windows: set OPENAI_API_KEY=sk-...

# 4) Run
streamlit run app.py
```

## 🗂️ Project Structure
```
studybot/
 ├─ app.py                # Streamlit UI
 ├─ rag.py                # Chunking, embeddings, retrieval
 ├─ quiz.py               # Quiz generation, grading, explanations
 ├─ db.py                 # SQLite schema and helpers
 ├─ utils.py              # File parsing, text cleaning, caching
 ├─ requirements.txt
 ├─ README.md
 └─ data/                 # Uploaded docs + indexes + sqlite db
```

## 🧪 Notes
- PDF/DOCX parsing is best-effort; ensure your docs are text‑selectable (OCR not included).
- First run will download models; allow a minute.
- You can swap models in `rag.py` (embedding) and `utils.py` (generation/summarization).

## ✅ Roadmap
- Multi‑doc collections with tags
- Progress dashboards by concept
- Audio (TTS) explanations
- Flashcards (Leitner system)
```

