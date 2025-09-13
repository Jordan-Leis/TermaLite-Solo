# ThemaLite-Solo

ThemaLite-Solo is a minimal FastAPI service that demonstrates vector search and summarisation
using a small in-repository dataset. It loads company blurbs, embeds them with a
Sentence Transformer model and performs nearest neighbour search with FAISS.

## Quickstart

```bash
pip install -r requirements.txt
python app.py
```

### Search
```bash
curl "http://127.0.0.1:8000/search?q=fintech compliance&k=5"
```

### Summary
```bash
curl "http://127.0.0.1:8000/summary?q=enterprise AI for retail"
```
