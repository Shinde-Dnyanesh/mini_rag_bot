from pathlib import Path
import numpy as np
import re
from typing import List, Tuple

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    text = " ".join(text.split())        
    if len(text) <= chunk_size:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks, current = [], ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) + 1 <= chunk_size:
            current = (current + " " + s).strip()
        else:
            chunks.append(current)
            current = s

    if current:
        chunks.append(current)

    return chunks

def load_docs(data_dir: str = "data") -> List[Tuple[str, str]]:
    p = Path(data_dir)
    if not p.exists():
        return []

    out = []
    for f in sorted(p.iterdir()):
        if f.suffix.lower() in (".txt", ".md"):
            out.append((f.name, f.read_text(encoding="utf-8")))
    return out


def cosine_similarity_vec(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m.dot(q)