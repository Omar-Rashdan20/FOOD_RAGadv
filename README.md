# Food RAG — Production Hybrid RAG System

A production-grade food recommendation API built on Advanced RAG principles: hybrid BM25 + vector search, cross-encoder reranking, MMR diversity, semantic caching, and a full RAGAS-style evaluation framework — all exposed via a FastAPI server.

---

## Project Structure

```
food-rag/
├── api/
│   ├── app.py          # FastAPI app + lifespan
│   ├── routes.py       # /health, /recommend, /eval
│   └── schemas.py      # Pydantic request/response models
├── src/
│   ├── cache.py        # 3-layer semantic cache
│   ├── config.py       # Settings from .env
│   ├── data_loader.py  # Dataset loading + normalization
│   ├── evaluator.py    # RAGAS-style eval metrics
│   ├── filters.py      # NL query parser
│   ├── hybrid_retriever.py  # BM25 + ChromaDB + RRF
│   ├── query_transformer.py # Multi-query, HyDE, step-back, routing
│   ├── rag_pipeline.py # Pipeline orchestrator
│   ├── reranker.py     # Cross-encoder + MMR
│   ├── utils.py        # Shared helpers
│   └── vector_store.py # ChromaDB interface
├── eval/
│   └── test_samples.json
├── data/
│   └── FoodDataSet.json   # (bring your own)
├── main.py
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
```

---

## Running the API Server

```bash
# Via CLI flag
python main.py --serve

# Via uvicorn directly
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs available at `http://localhost:8000/docs`.

---

## API Endpoints

### `GET /health`
Returns pipeline readiness and cache stats.

### `POST /recommend`
```json
{
  "query": "healthy Indian food under 400 calories, no nuts",
  "n_results": 5,
  "use_cache": true
}
```

### `DELETE /recommend/cache`
Clears both exact and semantic cache layers.

### `POST /eval`
```json
[
  {
    "query": "vegan Indian food under 400 calories",
    "ground_truth_answer": "Dal Tadka is a great choice.",
    "relevant_doc_ids": ["food_042"]
  }
]
```

---

## CLI Usage

```bash
# Single query
python main.py "spicy Thai noodles under 500 calories"

# Interactive mode
python main.py

# Rebuild ChromaDB index
python main.py --rebuild-index "healthy salad"

# Disable cross-encoder (faster)
python main.py --no-cross-encoder "light pasta"

# Run evaluation
python main.py --eval eval/test_samples.json

# Cache stats
python main.py --cache-stats "italian under 500 cal"
```

---

## Evaluation Targets

| Metric            | Target |
|-------------------|--------|
| Recall@10         | ≥ 0.80 |
| Precision@10      | ≥ 0.60 |
| MRR               | ≥ 0.70 |
| Hit Rate          | ≥ 0.90 |
| Faithfulness      | ≥ 0.90 |
| Answer Relevancy  | ≥ 0.85 |
| Context Precision | ≥ 0.70 |
| Context Recall    | ≥ 0.80 |
| Answer Correctness| ≥ 0.70 |

---

## Architecture

```
POST /recommend
      │
      ▼
Query Router ──► CLARIFICATION / REJECTION / GENERATION
      │
      ▼
Multi-Query Transform (4 variants + HyDE + step-back)
      │
      ▼
Hybrid Retrieval
  ├── Dense: ChromaDB (cosine) × each variant
  ├── Sparse: BM25 × each variant
  └── RRF Fusion (k=60)
      │
      ▼
Cross-encoder Reranker
  ├── ms-marco-MiniLM-L-6-v2
  ├── Metadata boost (cuisine, calories, dietary, mood)
  ├── Allergen penalty
  └── MMR diversity (λ=0.6)
      │
      ▼
3-Layer Cache
  ├── Layer 1: SHA-256 exact match
  └── Layer 2: Cosine similarity ≥ 0.92
      │
      ▼
Grounded Generation (Gemini)
```
