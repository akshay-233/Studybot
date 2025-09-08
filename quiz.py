import random
import re
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

def _top_terms_per_chunk(chunks: List[str], top_k: int = 8) -> List[List[str]]:
    vect = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1,2))
    X = vect.fit_transform(chunks)
    feature_names = vect.get_feature_names_out()
    results = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        inds = row.toarray().ravel().argsort()[::-1][:top_k]
        terms = [feature_names[j] for j in inds if re.match(r"^[A-Za-z].*", feature_names[j])]
        results.append(terms[:top_k])
    return results

def generate_mcq_from_chunks(chunks: List[str], n: int = 5) -> List[Dict[str,Any]]:
    """Heuristic MCQ generation without LLMs: use TF-IDF terms to create cloze questions."""
    qs = []
    top_terms = _top_terms_per_chunk(chunks, top_k=12)
    for i, chunk in enumerate(chunks):
        if len(qs) >= n: break
        terms = [t for t in top_terms[i] if len(t.split())<=3]
        if not terms: continue
        answer = random.choice(terms)
        sentence = chunk.strip()
        # pick a sentence containing the answer if possible
        sentences = re.split(r'(?<=[.?!])\s+', sentence)
        containing = [s for s in sentences if answer.lower() in s.lower()]
        base = random.choice(containing) if containing else random.choice(sentences[:3] or [chunk[:200]])
        prompt = re.sub(re.escape(answer), "____", base, flags=re.IGNORECASE)
        # distractors
        pool = list({t for tl in top_terms for t in tl if t.lower()!=answer.lower()})
        random.shuffle(pool)
        distractors = [w for w in pool if w.lower()!=answer.lower() and w.lower() not in prompt.lower()][:3]
        options = [answer] + distractors
        random.shuffle(options)
        qs.append({
            "qtype": "mcq",
            "prompt": prompt,
            "options": options,
            "answer": answer,
            "chunk_ref": str(i)
        })
    return qs

def generate_tf_from_chunks(chunks: List[str], n: int = 3) -> List[Dict[str,Any]]:
    qs = []
    for i, chunk in enumerate(chunks):
        if len(qs) >= n: break
        sentence = re.split(r'(?<=[.?!])\s+', chunk.strip())[0] if chunk.strip() else chunk[:150]
        # Flip a factual token by negation
        if " is " in sentence:
            prompt = sentence.replace(" is ", " is not ", 1)
            answer = "False"
        else:
            prompt = sentence
            answer = "True"
        qs.append({"qtype":"tf", "prompt": prompt, "options": ["True","False"], "answer": answer, "chunk_ref": str(i)})
    return qs

def explain_for_wrong(chunk_text: str, question: Dict[str,Any]) -> str:
    base = "Let's clarify. " \
           "Here is the key passage: " + chunk_text[:500] + " ... " \
           "Focus on the terms used in the question. The correct answer is: " + str(question.get("answer"))
    return base
