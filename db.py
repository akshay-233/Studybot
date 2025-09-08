import os
import sqlite3
from typing import Optional, List, Tuple, Dict, Any

DB_PATH = os.environ.get("DB_PATH","data/studybot.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    doc_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS qa_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS quizzes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    meta TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    quiz_id INTEGER NOT NULL,
    qtype TEXT NOT NULL,
    prompt TEXT NOT NULL,
    options TEXT,
    answer TEXT NOT NULL,
    chunk_ref TEXT
);

CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    quiz_id INTEGER NOT NULL,
    question_id INTEGER NOT NULL,
    user_answer TEXT NOT NULL,
    correct INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(SCHEMA)
    conn.commit()
    conn.close()

def create_session(doc_id: str, doc_path: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO sessions(doc_id, doc_path) VALUES (?,?)",(doc_id, doc_path))
    conn.commit()
    sid = cur.lastrowid
    conn.close()
    return sid

def log_qa(session_id: int, question: str, answer: str, sources: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO qa_logs(session_id, question, answer, sources) VALUES (?,?,?,?)",
                (session_id, question, answer, sources))
    conn.commit()
    conn.close()

def create_quiz(session_id: int, meta: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO quizzes(session_id, meta) VALUES (?,?)",(session_id, meta))
    conn.commit()
    qid = cur.lastrowid
    conn.close()
    return qid

def add_question(quiz_id: int, qtype: str, prompt: str, options: Optional[str], answer: str, chunk_ref: Optional[str]):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO questions(quiz_id, qtype, prompt, options, answer, chunk_ref) VALUES (?,?,?,?,?,?)",
                (quiz_id, qtype, prompt, options, answer, chunk_ref))
    conn.commit()
    conn.close()

def record_attempt(quiz_id: int, question_id: int, user_answer: str, correct: bool):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO attempts(quiz_id, question_id, user_answer, correct) VALUES (?,?,?,?)",
                (quiz_id, question_id, user_answer, int(correct)))
    conn.commit()
    conn.close()

def get_quiz_questions(quiz_id: int) -> List[Dict[str,Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, qtype, prompt, options, answer, chunk_ref FROM questions WHERE quiz_id=?",(quiz_id,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({"id": r[0], "qtype": r[1], "prompt": r[2], "options": r[3], "answer": r[4], "chunk_ref": r[5]})
    return out

def quiz_stats(quiz_id: int) -> Dict[str,Any]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), SUM(correct) FROM attempts WHERE quiz_id=?",(quiz_id,))
    total, correct = cur.fetchone()
    conn.close()
    correct = correct or 0
    acc = (correct/total*100.0) if total else 0.0
    return {"total": total, "correct": correct, "accuracy": acc}
