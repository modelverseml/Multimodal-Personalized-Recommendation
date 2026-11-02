import os
import threading
from contextlib import asynccontextmanager

# Must precede any OpenMP-linked import (FAISS, LightGBM); see app/ml/inference.py.
os.environ.setdefault("OMP_NUM_THREADS", "1")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

from app.config import settings
from app.database import Base, engine
from app.ml.inference import ArtifactsMissing, get_recommender
from app.routes import router


def warm_up():
    try:
        service = get_recommender()
    except ArtifactsMissing as exc:
        print(f"[startup] {exc}")
        return
    # First calls pay for ONNX Runtime / LightGBM initialisation; do it here
    # rather than on a user's first request.
    sample = [int(p) for p in service.product_ids[:5]]
    service.recommend(sample)
    service.search("warm up")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    # Load model artifacts off the event loop so /health answers immediately.
    threading.Thread(target=warm_up, daemon=True).start()
    yield


app = FastAPI(title="Multimodal Recommendation API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
app.include_router(router, prefix="/api")
app.mount("/metrics", make_asgi_app())
settings.image_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=settings.image_dir), name="images")


@app.get("/health")
def health():
    return {"status": "ok"}
