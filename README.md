# Food RAG - Production Hybrid RAG System

A production-grade food recommendation API built with hybrid BM25 + vector search, metadata filtering, cross-encoder reranking, semantic caching, and RAGAS-style evaluation. The system uses a local Ollama model for grounded answer generation and LLM-as-judge generation evaluation.

---
##Demo: 
 [Link](https://www.linkedin.com/posts/omar-rashdan-64a475282_ai-rag-llm-ugcPost-7458951337712472065-hF6C?utm_source=share&utm_medium=member_desktop&rcm=ACoAAETErFoBium0uFL-HSaG10fR0-6OHAO8NXU)
---

## Project Structure

```text
food-rag/
|-- api/
|   |-- app.py          # FastAPI app + lifespan
|   |-- routes.py       # /health, /recommend, /eval
|   `-- schemas.py      # Pydantic request/response models
|-- data/
|   |-- FoodDataSet.json
|   `-- eval_data/
|       `-- test_samples_50.json
|-- scripts/
|   `-- preprocess_usda_fooddata.py
|-- src/
|   |-- cache.py        # Exact + semantic cache
|   |-- config.py       # Settings from .env
|   |-- data_loader.py  # Dataset loading + normalization
|   |-- evaluator.py    # Retrieval + generation eval metrics
|   |-- filters.py      # Nutrition/query filter parser
|   |-- hybrid_retriever.py  # BM25 + ChromaDB + RRF
|   |-- query_transformer.py # Router, multi-query, step-back
|   |-- rag_pipeline.py # Pipeline orchestrator
|   |-- reranker.py     # Cross-encoder + MMR
|   `-- vector_store.py # ChromaDB interface
|-- main.py
|-- requirements.txt
`-- .env.example
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2:3b
```

Make sure Ollama is running before starting the API:

```bash
ollama serve
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Dataset

The app expects the production dataset here by default:

```text
data/FoodDataSet.json
```

Default `.env` setting:

```env
DATASET_PATH=./data/FoodDataSet.json
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b
```

Each food record should support RAG fields such as:

```json
{
  "id": "2727567",
  "food_name": "Chicken thigh",
  "aliases": ["chicken thigh"],
  "semantic_text": "Food Name: Chicken thigh...",
  "metadata": {
    "calories": 221,
    "protein_g": 18.6,
    "low_sodium": true,
    "high_protein": true
  },
  "tags": ["high-protein", "low-sodium"]
}
```

---

## USDA Preprocessing

If you need to regenerate the dataset from a USDA-style JSON file:

```bash
python scripts/preprocess_usda_fooddata.py --input "C:/path/to/FoodData_Central_foundation_food_json_2026-04-30.json"
```

Output:

```text
data/FoodDataSet.json
```

The preprocessor normalizes food names, extracts nutrition fields, computes missing calories, generates nutrition tags, creates aliases, builds semantic text, and removes duplicates.

---

## Running The API

```bash
python main.py --serve
```

Or run directly with reload:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### `GET /health`

Returns pipeline readiness and cache stats.

### `POST /recommend`

```json
{
  "query": "healthy food under 300 calories with low sodium",
  "n_results": 5,
  "use_cache": true
}
```

Example query ideas:

```json
{
  "query": "high protein low fat foods",
  "n_results": 3,
  "use_cache": true
}
```

```json
{
  "query": "foods for muscle recovery",
  "n_results": 3,
  "use_cache": true
}
```

### `DELETE /recommend/cache`

Clears recommendation cache, semantic cache, and route cache.

### `POST /eval`

```json
[
  {
    "query": "show me protein rich food for a post workout meal",
    "ground_truth_answer": "Chicken thigh, pollock, and pine nuts are relevant high-protein options for post-workout recovery.",
    "relevant_doc_ids": ["2727567", "333476", "2346392"]
  }
]
```

---

## CLI Usage

```bash
# Single query
python main.py "healthy food under 300 calories with low sodium"

# Interactive mode
python main.py

# Rebuild ChromaDB index
python main.py --rebuild-index "high protein low fat foods"

# Disable cross-encoder for faster testing
python main.py --no-cross-encoder "foods rich in potassium"

# Run evaluation
python main.py --eval data/eval_data/test_samples_50.json

# Cache stats
python main.py --cache-stats "low sodium healthy snacks"
```

---

## Query Routing

The router chooses one strategy per query:

| Strategy | Used for |
| --- | --- |
| `NORMAL_RETRIEVAL` | Exact food names, direct ingredient searches, simple nutrition lookups |
| `MULTI_QUERY` | Nutrition constraints, calorie filtering, diet searches, ingredient combinations |
| `MULTI_QUERY_STEPBACK` | Goal-based or recommendation-style queries like muscle recovery or digestion |
| `CLARIFICATION` | Queries too vague to retrieve reliably |
| `REJECTION` | Non-food topics |

Examples:

```text
apple pie calories -> NORMAL_RETRIEVAL
healthy foods under 300 calories -> MULTI_QUERY
foods for muscle recovery -> MULTI_QUERY_STEPBACK
protein -> CLARIFICATION
weather tomorrow -> REJECTION
```

---

## Evaluation

Retrieval metrics:

- Recall@K
- Precision@K
- MRR
- Hit Rate

Generation metrics use LLM-as-judge:

- Faithfulness
- Answer relevancy
- Context precision
- Context recall
- Answer correctness

Run:

```bash
python main.py --eval data/eval_data/test_samples_50.json
```

---

## Architecture

```text
POST /recommend
      |
      v
Query Router
      |
      +--> CLARIFICATION / REJECTION
      |
      +--> NORMAL_RETRIEVAL
      +--> MULTI_QUERY
      +--> MULTI_QUERY_STEPBACK
      |
      v
Hybrid Retrieval
  - Dense: ChromaDB vector search
  - Sparse: BM25 keyword search
  - RRF fusion
      |
      v
Cross-encoder Reranker + MMR Diversity
      |
      v
Grounded Generation with Ollama llama3.2:3b
      |
      v
Recommendation Cache + Semantic Cache + Route Cache
```

---

## Metadata Filtering

The dataset metadata is optimized for filtering:

```python
where = {
    "$and": [
        {"calories": {"$lte": 300}},
        {"high_protein": {"$eq": True}},
        {"low_sodium": {"$eq": True}},
    ]
}
```

Supported nutrition-aware searches include:

- healthy food under 300 calories
- high protein low fat foods
- vegan foods rich in iron
- foods rich in potassium
- low sodium healthy snacks
- diabetic-friendly foods
- foods for muscle recovery
