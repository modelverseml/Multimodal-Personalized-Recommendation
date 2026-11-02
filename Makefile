# Run from the repo root. Override PY to point at your virtualenv's python.
PY ?= ../venvs/multimodal-env/bin/python
BACKEND = cd backend &&

.PHONY: ingest features train ablations evaluate pipeline api frontend mlflow test drift simulate

ingest:      ## download Amazon Reviews 2023, load Postgres, fetch images
	$(BACKEND) ../$(PY) -m scripts.ingest
features:    ## CLIP text + image embeddings (cached)
	$(BACKEND) ../$(PY) -m app.ml.features
train:       ## two-tower + LambdaRank, promote to serving
	$(BACKEND) ../$(PY) -m app.ml.train
ablations:   ## optional, slow (~20 min each): modality ablations, tracked in MLflow
	$(BACKEND) for v in id_only id_text content_only; do ../$(PY) -m app.ml.train --variant $$v; done
evaluate:    ## test-set metrics, baselines, latency -> artifacts/evaluation.json
	$(BACKEND) ../$(PY) -m scripts.evaluate
pipeline: ingest features train evaluate

api:
	$(BACKEND) ../$(PY) -m uvicorn app.main:app --port 8001
frontend:
	cd frontend && npm run dev
mlflow:
	$(BACKEND) ../$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
test:
	$(BACKEND) ../$(PY) -m pytest -q
drift:
	$(BACKEND) ../$(PY) -m scripts.drift
simulate:
	$(BACKEND) ../$(PY) -m scripts.simulate --shoppers 40 --platform "Nintendo Switch"
