import os
import io
import json
import streamlit as st
from typing import List, Dict, Any
from rag import ingest, VectorStore
from utils import ensure_dir
import db as dbi
from quiz import generate_mcq_from_chunks, generate_tf_from_chunks, explain_for_wrong

# Optional OpenAI for better generation/summaries
USE_OPENAI = bool(os.environ.get("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"))
if USE_OPENAI:
    from openai import OpenAI
    oai = OpenAI()

st.set_page_config(page_title="Personalized Study Bot", page_icon="🧠", layout="wide")

st.title("🧠 Personalized Study Bot (RAG + Quiz + Adaptive Learning)")

# Sidebar
with st.sidebar:
    st.header("Session")
    st.write("1) Upload a document\n2) Ask questions (RAG)\n3) Generate & take a quiz\n4) Review mistakes → Retest")
    st.markdown("---")
    top_k = st.slider("Top‑k retrieval", 3, 10, 5)
    num_mcq = st.slider("MCQs to generate", 3, 15, 7)
    num_tf = st.slider("True/False to generate", 0, 10, 3)

# Upload
uploaded = st.file_uploader("Upload your study document (PDF/DOCX/TXT)", type=["pdf","docx","txt","md"])
if uploaded:
    path = os.path.join("data", uploaded.name)
    ensure_dir("data")
    with open(path,"wb") as f:
        f.write(uploaded.read())
    st.success(f"Uploaded: {uploaded.name}")
    if st.button("Build Knowledge Base"):
        vs, doc_id, n_chunks = ingest(path, store_dir="data")
        st.session_state["vs_index_path"] = vs.index_path
        st.session_state["doc_id"] = doc_id
        st.session_state["doc_path"] = path
        st.session_state["n_chunks"] = n_chunks
        # DB
        dbi.init_db()
        sid = dbi.create_session(doc_id, path)
        st.session_state["session_id"] = sid
        st.success(f"Indexed {n_chunks} chunks. Session #{sid} created.")

# Load vector store if exists
def load_vs_from_ss() -> VectorStore:
    vs = VectorStore(index_path=st.session_state["vs_index_path"])
    vs.load()
    return vs

# Q&A
st.header("Ask a question")
question = st.text_input("Type your question here")
if st.button("Answer with RAG", disabled=not st.session_state.get("vs_index_path")):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        vs = load_vs_from_ss()
        hits = vs.search(question, k=top_k)
        context = "\n\n".join([f"[Chunk {h['chunk_id']}] {h['text']}" for h in hits])
        if USE_OPENAI:
            prompt = f"You are a helpful tutor. Using ONLY the context, answer clearly and simply.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
            resp = oai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.2)
            answer = resp.choices[0].message.content
        else:
            # Lightweight local fallback: just return the most relevant chunk(s) as a summarized answer.
            # Simple compress by truncation; you can swap to transformers summarization if desired.
            chunks = [h["text"] for h in hits]
            answer = "Here’s the summary from the most relevant sections:\n\n" + "\n\n".join([c[:400] for c in chunks[:2]])
        st.markdown("### Answer")
        st.write(answer)
        srcs = [{"chunk_id": h["chunk_id"], "score": h["score"]} for h in hits]
        with st.expander("Sources"):
            for h in hits:
                st.write(f"Chunk {h['chunk_id']} (score={h['score']:.3f})")
                st.write(h["text"][:300] + ("..." if len(h["text"])>300 else ""))
        if st.session_state.get("session_id"):
            dbi.log_qa(st.session_state["session_id"], question, answer, json.dumps(srcs))

# Quiz generation
st.header("Generate a quiz")
if st.button("Create Quiz", disabled=not st.session_state.get("vs_index_path")):
    vs = load_vs_from_ss()
    chunks = [m["text"] for m in vs.metadata]
    mcq = generate_mcq_from_chunks(chunks, n=num_mcq)
    tfq = generate_tf_from_chunks(chunks, n=num_tf)
    questions = mcq + tfq
    quiz_id = dbi.create_quiz(st.session_state["session_id"], meta=json.dumps({"num_mcq": num_mcq, "num_tf": num_tf}))
    for q in questions:
        options = json.dumps(q.get("options")) if q.get("options") else None
        dbi.add_question(quiz_id, q["qtype"], q["prompt"], options, q["answer"], q.get("chunk_ref"))
    st.session_state["quiz_id"] = quiz_id
    st.success(f"Quiz #{quiz_id} created with {len(questions)} questions. Scroll down to take it.")

# Take quiz
if st.session_state.get("quiz_id"):
    st.subheader(f"Quiz #{st.session_state['quiz_id']}")
    questions = dbi.get_quiz_questions(st.session_state["quiz_id"])
    answers: Dict[int, Any] = {}
    for q in questions:
        st.markdown(f"**Q{q['id']} ({q['qtype']}):** {q['prompt']}")
        if q["qtype"] == "mcq":
            opts = json.loads(q["options"])
            choice = st.radio(f"Choose an answer for Q{q['id']}", opts, key=f"q{q['id']}")
            answers[q["id"]] = choice
        elif q["qtype"] == "tf":
            choice = st.radio(f"Choose an answer for Q{q['id']}", ["True","False"], key=f"q{q['id']}")
            answers[q["id"]] = choice
        else:
            ans = st.text_input(f"Your answer for Q{q['id']}", key=f"q{q['id']}")
            answers[q["id"]] = ans

    if st.button("Submit Quiz"):
        # grade
        correct_count = 0
        vs = load_vs_from_ss()
        for q in questions:
            ua = str(answers.get(q["id"], "")).strip()
            gt = str(q["answer"]).strip()
            is_correct = (ua.lower() == gt.lower())
            if is_correct: correct_count += 1
            dbi.record_attempt(st.session_state["quiz_id"], q["id"], ua, is_correct)

        stats = dbi.quiz_stats(st.session_state["quiz_id"])
        st.success(f"Score: {stats['correct']} / {stats['total']} ({stats['accuracy']:.1f}%)")

        # Explanations for wrong answers
        wrong = [q for q in questions if str(answers.get(q["id"], "")).strip().lower() != str(q["answer"]).strip().lower()]
        if wrong:
            st.markdown("### Review: Explanations for incorrect answers")
            for q in wrong:
                chunk_idx = int(q["chunk_ref"]) if q["chunk_ref"] and q["chunk_ref"].isdigit() else None
                chunk_text = vs.metadata[chunk_idx]["text"] if chunk_idx is not None and chunk_idx < len(vs.metadata) else ""
                exp = explain_for_wrong(chunk_text, q)
                st.info(f"Q{q['id']}: {exp}")

        # Retest option
        if st.button("Retest weak areas"):
            # Create mini-quiz from the same chunk refs
            weak_chunks = []
            for q in wrong:
                try:
                    weak_chunks.append(vs.metadata[int(q["chunk_ref"])]["text"])
                except Exception:
                    pass
            if weak_chunks:
                mcq = generate_mcq_from_chunks(weak_chunks, n=min(5, len(weak_chunks)))
                quiz_id = dbi.create_quiz(st.session_state["session_id"], meta=json.dumps({"retest":"weak_areas"}))
                for q in mcq:
                    options = json.dumps(q.get("options")) if q.get("options") else None
                    dbi.add_question(quiz_id, q["qtype"], q["prompt"], options, q["answer"], q.get("chunk_ref"))
                st.session_state["quiz_id"] = quiz_id
                st.success(f"Retest Quiz #{quiz_id} created. Scroll up to answer.")
            else:
                st.info("No specific weak chunks detected; try generating a new quiz.")
