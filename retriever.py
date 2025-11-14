import sqlite3
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

DB_FILE = "memory.db"
EMB_MODEL = "all-MiniLM-L6-v2"
TOP_K = 1

class Retriever:
    def __init__(self, db_file: str = DB_FILE):
        print("Loading index from", db_file, "...")
        self.db_file = db_file

        conn = sqlite3.connect(db_file)
        rows = conn.execute("SELECT source, text, embedding FROM chunks").fetchall()
        conn.close()

        self.sources = [r[0] for r in rows]
        self.texts   = [r[1] for r in rows]
        vectors      = [pickle.loads(r[2]) for r in rows]

        if vectors:
            self.vectors = np.vstack(vectors)
            self.vectors = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10)
        else:
            self.vectors = np.zeros((0, 0))

        self.model = SentenceTransformer(EMB_MODEL)

    def retrieve(self, query: str, top_k: int = TOP_K):
        if self.vectors.size == 0:
            return []

        q = self.model.encode([query], convert_to_numpy=True)[0]
        q = q / (np.linalg.norm(q) + 1e-10) 

        sims = self.vectors.dot(q)

        idxs = np.argsort(-sims)[:top_k]

        return [
            {
                "score": float(sims[i]),
                "text": self.texts[i],
                "source": self.sources[i]
            }
            for i in idxs
        ]

if __name__ == "__main__":
    r = Retriever()
    q = input("Query> ")
    for out in r.retrieve(q, top_k=3):
        print(f"\nScore: {out['score']:.3f}  Source: {out['source']}")
        print(out["text"])
