import sqlite3
import pickle
from sentence_transformers import SentenceTransformer
from utils import load_docs, chunk_text

DB_FILE = "memory.db"
EMB_MODEL = "all-MiniLM-L6-v2"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY,source TEXT NOT NULL,text TEXT NOT NULL,embedding BLOB NOT NULL)"""
INSERT_SQL = "INSERT INTO chunks (source, text, embedding) VALUES (?, ?, ?)"

def ensure_db(path: str = DB_FILE):
    with sqlite3.connect(path) as conn:
        conn.execute(CREATE_SQL)

def build(db_path: str = DB_FILE, model_name: str = EMB_MODEL):
    docs = load_docs("data")
    if not docs:
        raise SystemExit("No docs found in data/. Add .txt/.md files and retry.")

    ensure_db(db_path)
    model = SentenceTransformer(model_name)

    total = 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM chunks")

        for fname, text in docs:
            chunks = chunk_text(text)
            if not chunks:
                continue

            vectors = model.encode(chunks,show_progress_bar=True,convert_to_numpy=True)

            for chunk, vec in zip(chunks, vectors):
                conn.execute(INSERT_SQL,
                    (fname, chunk, pickle.dumps(vec, protocol=pickle.HIGHEST_PROTOCOL)))
                total += 1

        conn.commit()

    print(f"Saved {total} chunks to {db_path}")

if __name__ == "__main__":
    build()
