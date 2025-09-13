import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_PATH = Path(__file__).parent / "data" / "companies.csv"

# Load dataset
companies = pd.read_csv(DATA_PATH)

# Build embeddings and FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(companies["blurb"].tolist(), convert_to_numpy=True)
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

app = FastAPI()


def _search(query: str, k: int):
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    scores, ids = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        row = companies.iloc[idx]
        results.append({
            "id": int(row["id"]),
            "name": row["name"],
            "blurb": row["blurb"],
            "sector": row["sector"],
            "score": float(score),
        })
    return results


@app.get("/search")
def search(q: str, k: int = 5):
    return {"results": _search(q, k)}


@app.get("/summary")
def summary(q: str):
    neighbors = _search(q, 3)
    pieces = [f"{n['name']} ({n['sector']})" for n in neighbors]
    text = f"Companies similar to '{q}' include " + ", ".join(pieces) + "."
    return {"summary": text, "neighbors": neighbors}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
