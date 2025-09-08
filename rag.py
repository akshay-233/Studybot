import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from utils import file_to_text, clean_text, chunk_text, ensure_dir, stable_hash

EMB_MODEL_NAME = os.environ.get("EMB_MODEL_NAME","sentence-transformers/all-MiniLM-L6-v2")

class VectorStore:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.emb_model = SentenceTransformer(EMB_MODEL_NAME)
        self.index = None
        self.metadata: List[Dict[str,Any]] = []
        self.dim = self.emb_model.get_sentence_embedding_dimension()

    def _build_index(self, embeddings: np.ndarray):
        index = faiss.IndexFlatIP(self.dim)
        # normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        self.index = index

    def add_texts(self, chunks: List[str], doc_id: str, path: str):
        vectors = self.emb_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
        if self.index is None:
            self._build_index(vectors.copy())
        else:
            self.index.add(vectors.copy())
        for i, chunk in enumerate(chunks):
            self.metadata.append({"doc_id": doc_id, "path": path, "chunk_id": i, "text": chunk})

    def search(self, query: str, k: int = 5) -> List[Dict[str,Any]]:
        if self.index is None:
            return []
        q = self.emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1: 
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        return results

    def save(self):
        ensure_dir(os.path.dirname(self.index_path))
        # save faiss
        faiss.write_index(self.index, self.index_path)
        # save metadata
        import json
        with open(self.index_path + ".meta.json","w",encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self):
        if not os.path.exists(self.index_path):
            return False
        self.index = faiss.read_index(self.index_path)
        import json
        with open(self.index_path + ".meta.json","r",encoding="utf-8") as f:
            self.metadata = json.load(f)
        return True

def ingest(path: str, store_dir: str = "data") -> Tuple[VectorStore, str, int]:
    text = clean_text(file_to_text(path))
    chunks = chunk_text(text, chunk_size=450, overlap=120)
    doc_id = stable_hash(os.path.basename(path) + str(len(text)))
    index_path = os.path.join(store_dir, f"{doc_id}.faiss")
    vs = VectorStore(index_path=index_path)
    if not vs.load():
        vs.add_texts(chunks, doc_id, path)
        vs.save()
    return vs, doc_id, len(chunks)
