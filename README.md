# Multimodal Personalized Recommendation Engine

A production-style recommender trained on **real Amazon shopping data**. A PyTorch **two-tower** model learns user and item embeddings from interaction histories plus **CLIP image and text features**. **FAISS** retrieves candidates in under a millisecond. A **LightGBM LambdaRank** model reranks them using collaborative, content and popularity signals. Training runs are tracked in **MLflow**. Serving is a torch-free **FastAPI** service (ONNX Runtime + FAISS + LightGBM) behind a React app. The app personalizes in real time, explains each recommendation, and monitors drift.

**Stack:** PyTorch · Hugging Face CLIP (ViT-B/32) · FAISS (HNSW) · LightGBM LambdaRank · ONNX / ONNX Runtime · MLflow · FastAPI · PostgreSQL · SQLAlchemy · JWT (Argon2) · Prometheus metrics · React + Vite

**Data:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab), *Video Games* category, official 5-core leave-last-out split: **94,762 customers · 25,612 products · 814,586 interactions**, with product titles, descriptions, features and images (25,606 product photos).

## Results

All numbers come from `backend/artifacts/evaluation.json`, produced by `python -m scripts.evaluate`. Every one of the **94,762 test users** is ranked against the **full 25,612-item catalog** (no sampled negatives), with items already in their history excluded. Each user has one held-out item, so Recall@10 equals HitRate@10.

| Model | Recall@10 | NDCG@10 (95% CI) | Recall@20 | MRR | Coverage@10 |
|---|---|---|---|---|---|
| Popularity | 0.0249 | 0.0127 (0.0122–0.0133) | 0.0379 | 0.0100 | 0.1% |
| CLIP-kNN (content only, no training) | 0.0219 | 0.0111 (0.0106–0.0117) | 0.0351 | 0.0088 | 60.3% |
| BPR-MF | 0.0413 | 0.0221 (0.0213–0.0229) | 0.0628 | 0.0177 | 58.2% |
| ItemKNN-CF (best baseline) | 0.0703 | 0.0397 (0.0387–0.0407) | 0.0988 | 0.0323 | 94.9% |
| **Two-tower, multimodal (retrieval only)** | 0.0938 | 0.0513 (0.0502–0.0524) | 0.1375 | 0.0414 | 47.1% |
| **Two-tower + LambdaRank (full pipeline)** | **0.1099** | **0.0625** (0.0612–0.0638) | **0.1550** | **0.0512** | 48.0% |

- **+57.6% NDCG@10 and +56.3% Recall@10 over the best baseline** (ItemKNN-CF). The paired bootstrap 95% CI for the NDCG@10 gain is +0.0218 to +0.0239, so the lift is well clear of noise.
- **Reranking adds +22% NDCG@10 on top of retrieval** (0.0513 → 0.0625). The target is in the top 100 FAISS candidates for 29.0% of users, which is the upper bound on what the reranker can find.
- **CLIP features help.** Retrieval NDCG@10 is 0.0513 with CLIP text + image features vs 0.0472 for an ID-only two-tower (+8.6%, same architecture and training schedule). CLIP features alone, with no learning, are weak (0.0111). The gain comes from combining content features with learned collaborative signal.

**Serving latency** (in-process, CPU, 500 replayed test users, real serving stack):

| Stage | p50 | p95 | p99 |
|---|---|---|---|
| User tower (ONNX Runtime) | 0.2 ms | 0.3 ms | — |
| FAISS HNSW top-200 | 0.4 ms | 0.5 ms | — |
| Features + LambdaRank + SHAP reasons | — | 8.6 ms | — |
| **End to end** | **8.5 ms** | **9.4 ms** | **9.8 ms** |
| CLIP semantic search (text encoder + 2 FAISS searches) | 6.8 ms | 8.8 ms | 11.0 ms |

End-to-end p95 latency is about 10x under the 100 ms target. FAISS HNSW (M=32, efSearch=256) returns 99.9% of the exact top-100.

These are offline results on public data, and live CTR in the app comes from the traffic simulator, not from real users.

## Architecture

```
                 ┌─────────────── offline (PyTorch) ────────────────┐
PostgreSQL ──►   │ CLIP ViT-B/32 ─► text + image embeddings (cached)│
products         │        │                                         │
customers        │ Two-tower model                                  │
interactions     │   item tower: ID emb (+ID dropout) + MLP(CLIP)   │
                 │               + platform emb                     │
                 │   user tower: causal Transformer over history    │
                 │   loss: sampled softmax, mixed negatives, logQ   │
                 │        │                                         │
                 │ LightGBM LambdaRank on validation candidates     │
                 │        │                                         │
                 │ ONNX export (verified vs torch) ─ FAISS build ─► │──► artifacts/runs/<version>
                 │ MLflow: params · curves · metrics · artifacts    │        └── current ─► serving
                 └──────────────────────────────────────────────────┘

                 ┌──────────────── online (no PyTorch) ─────────────┐
React ──► FastAPI│ history (Postgres) ─► user tower (ONNX Runtime)  │
 For you         │   ─► FAISS HNSW top-200 ─► 17 ranking features   │
 Catalog/search  │   ─► LambdaRank ─► SHAP reasons + "because you…" │
 Evaluation      │ CLIP text (ONNX) ─► FAISS ─► semantic search     │
 Monitoring      │ events ─► Postgres (next request is personalized)│
                 │ impressions ─► CTR · PSI drift · /metrics        │
                 └──────────────────────────────────────────────────┘
```

### Design decisions worth asking about

- **The user tower is a function of history, not a user-ID lookup.** A shopper created five seconds ago gets personalized recommendations after their first click, with no retraining. Recommendations update after every event. The spec's original "fall back to popularity until the next retrain" behavior is used only when a shopper has no history at all.
- **Content and identity in the item tower, with ID dropout.** 15% of the time the ID embedding is dropped during training, so the CLIP pathway has to carry signal on its own. That makes the content path useful for long-tail items. The ablation measures what each modality adds.
- **Sampled softmax with mixed negatives and logQ correction.** In-batch negatives follow the popularity distribution, so without correction the model learns to under-rank popular items. Shared uniform negatives plus logQ correction fix that (Yi et al., 2019; Yang et al., 2020).
- **Honest offline evaluation.** Every model ranks against the full catalog of 25,612 items, not a sampled subset, so metrics aren't inflated. The ranker is trained on validation-stage targets the retriever never saw. Catalog `average_rating` and `rating_number` are excluded from the ranker because they come from a 2023 crawl that counts reviews written after the evaluation cutoff. Bootstrap confidence intervals are reported.
- **Torch-free serving.** The API runs ONNX Runtime + FAISS + LightGBM. Each ONNX export is checked against PyTorch, with the maximum absolute difference recorded in the manifest. On macOS the PyTorch and FAISS wheels each bundle an OpenMP runtime and crash in one process, so the split is a requirement here. It also gives a smaller serving footprint.
- **Length-bucketed training.** The median history is 4 items, but sequences are padded to 50. Batching sequences of similar length and trimming the padding made an epoch about 3x faster, and turned CPU training from impractical into roughly 45 s per epoch.

## Run it

Ports and database are independent of Project 1 (API **8001**, UI **5174**, database `recommendation_db`).

```bash
# 1. Database
createdb recommendation_db            # or create a dedicated role; see backend/.env.example

# 2. Backend
cd backend
python -m venv ../../venvs/multimodal-env && source ../../venvs/multimodal-env/bin/activate
pip install -r requirements.txt
cp .env.example .env                   # set DATABASE_URL, JWT_SECRET, DEMO_EMAIL/DEMO_PASSWORD

# 3. Data: downloads Amazon Reviews 2023 (~550 MB), loads Postgres via COPY, fetches ~25k images (~1 GB)
python -m scripts.ingest

# 4. Features + training (first run downloads CLIP weights)
python -m app.ml.features              # ~9 min on Apple MPS, cached afterwards
python -m app.ml.train                 # ~20 min on CPU: two-tower + LambdaRank + ONNX + FAISS, promoted to serving
for v in id_only id_text content_only; do python -m app.ml.train --variant $v; done   # ablations

# 5. Offline evaluation (baselines, CIs, ablations, serving latency)
python -m scripts.evaluate

# 6. Serve
uvicorn app.main:app --port 8001
cd ../frontend && cp .env.example .env && npm install && npm run dev     # http://localhost:5174
```

Other tools:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # experiment tracking UI (from backend/)
python -m scripts.simulate --shoppers 40 --platform "Nintendo Switch"   # live traffic -> CTR + drift
python -m scripts.drift                # PSI report; exits 2 above threshold (for a scheduled job)
python -m pytest -q                    # unit + black-box API tests
```

A `Makefile` at the repo root wraps these (`make pipeline`, `make api`, `make frontend`, `make test`). The reported model was trained for 40 epochs; the default is now 20 epochs with patience 3, since validation NDCG flattens around epoch 25 of the 40-epoch schedule.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/auth/login` | JWT login |
| GET | `/api/catalog/products` | browse/filter/sort the catalog |
| GET | `/api/catalog/products/{id}` | product detail + FAISS "more like this" |
| GET | `/api/catalog/search?q=` | CLIP text→image+text semantic search |
| GET/POST | `/api/customers` | sample shoppers / create a new live shopper |
| GET | `/api/recommendations/{customer_id}?k=&rerank=` | personalized top-k with reasons and per-stage latency; logs impressions |
| POST | `/api/events` | view / click / add_to_cart / purchase feedback (attributed via `request_id`) |
| GET | `/api/evaluation` | offline report + training curves + ranker importances |
| GET | `/api/monitoring` | PSI drift + online CTR / conversion |
| GET | `/metrics` | Prometheus latency histograms and counters |

## Project layout

```
backend/
  app/
    config.py database.py models.py schemas.py security.py routes.py main.py monitoring.py
    ml/
      data.py        Postgres -> sequences, leave-last-out split
      features.py    CLIP text + image embeddings (cached by ASIN)
      towers.py      two-tower model, causal user tower, mixed-negative logQ loss
      baselines.py   Popularity, ItemKNN-CF, BPR-MF, CLIP-kNN
      ranking.py     vectorised LTR features + LightGBM LambdaRank
      train.py       training pipeline + MLflow + promotion
      export.py      ONNX export with numerical verification
      index.py       FAISS HNSW item index + CLIP search index (torch-free)
      inference.py   online service: ONNX -> FAISS -> LambdaRank -> explanations
      metrics.py     Recall/NDCG/MRR, coverage, diversity, novelty
  scripts/  ingest.py evaluate.py benchmark.py drift.py simulate.py
  tests/    test_ml.py (unit) test_api.py (black-box integration)
frontend/   React + Vite: For you · Catalog · Evaluation · Monitoring
```

## Limitations

- Reviews stand in for purchases. Live events of every type are treated as positive implicit feedback, without weighting by event type.
- The serving model is the one trained on the training split. The user tower always reads each shopper's latest history, but item embeddings and ranker statistics only refresh on retraining.
- Leave-last-out evaluates one held-out item per user, so Recall@10 equals HitRate@10.
- The PSI threshold (0.2) is a monitoring heuristic, not a statistical test or an automatic retraining trigger.
