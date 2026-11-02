"""Online recommendation service (no PyTorch at serving time).

Request path, all in-process on CPU:
  history -> user tower (ONNX Runtime, causal transformer over item vectors)
          -> FAISS HNSW top-C candidate retrieval
          -> vectorised feature computation + LightGBM LambdaRank
          -> per-feature SHAP contributions for the top-k (explanations)
Because the user tower is a function of history (no per-user weights), a new
event changes the very next recommendation; no retraining needed.

The serving stack (ONNX Runtime + FAISS + LightGBM) deliberately excludes
PyTorch: on macOS the PyTorch and FAISS wheels each bundle an OpenMP runtime
that cannot coexist in one process. ONNX also gives a smaller, faster-starting
serving footprint. Within the process, FAISS and LightGBM are both kept
single-threaded for the same reason (see Recommender.__init__ and ranking.py).
"""
import json
import os
import threading
import time
from pathlib import Path

# FAISS and LightGBM link different OpenMP runtimes; a multithreaded parallel
# region in one can deadlock or crash on worker threads owned by the other.
# omp_set_num_threads() is per-thread, and requests run on FastAPI worker
# threads, so the limit must be process-wide and set before OpenMP starts.
# Online work is one query at a time, so nothing is lost; ONNX Runtime uses
# its own thread pool and is unaffected.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from app.config import settings
from app.ml import index as faiss_index
from app.ml.ranking import FEATURE_LABELS, FEATURES, LGB_THREADS, FeatureStore

HIDDEN_REASONS = {"history_length", "has_image", "retrieval_rank", "two_tower_score", "price"}


class ArtifactsMissing(RuntimeError):
    pass


def _session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class Recommender:
    def __init__(self, path: Path | None = None):
        faiss_index.faiss.omp_set_num_threads(1)  # see OMP_NUM_THREADS note at the top
        self.path = path or settings.artifact_dir / "current"
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise ArtifactsMissing(f"No trained model at {self.path}. Run `python -m app.ml.train` first.")

        self.manifest = json.loads(manifest_path.read_text())
        self.version = self.manifest["model_version"]
        self.max_len = self.manifest["tower_config"]["max_len"]
        self.product_ids = np.load(self.path / "product_ids.npy")
        self.index_of = {int(pid): i + 1 for i, pid in enumerate(self.product_ids)}
        self.item_vecs = np.load(self.path / "item_vecs.npy").astype(np.float32)
        self.user_tower = _session(self.path / "user_tower.onnx")
        self.items = faiss_index.load(self.path / "items.hnsw")
        self.search_image = faiss_index.load(self.path / "search_image.flat")
        self.search_text = faiss_index.load(self.path / "search_text.flat")
        self.store = FeatureStore.load(self.path / "features")
        self.ranker = lgb.Booster(model_file=str(self.path / "ranker.txt"))
        trending = self.store.recent_popularity + 0.25 * self.store.popularity
        trending[0] = -np.inf
        self.trending = np.argsort(-trending)
        self._clip = None
        self._clip_lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def to_index(self, product_ids: list[int]) -> list[int]:
        return [self.index_of[p] for p in product_ids if p in self.index_of]

    def to_product(self, idx) -> int:
        return int(self.product_ids[int(idx) - 1])

    def padded(self, hist_idx: list[int]) -> np.ndarray:
        hist = hist_idx[-self.max_len:]
        out = np.zeros((1, self.max_len), np.int64)
        if hist:
            out[0, -len(hist):] = hist
        return out

    def user_vector(self, padded: np.ndarray) -> np.ndarray:
        """padded: (B, max_len) left-padded histories -> (B, d). The ONNX graph
        takes one history of length 2..max_len, so trim leading padding but
        keep at least two positions (extra padding is masked, so it is a no-op)."""
        out = []
        for row in padded:
            width = min(max(int((row > 0).sum()), 2), self.max_len)
            hist = row[None, -width:]
            out.append(self.user_tower.run(None, {"item_vecs": self.item_vecs[hist], "pad": hist == 0})[0])
        return np.concatenate(out)

    # ------------------------------------------------------------------ API
    def recommend(self, history: list[int], k: int = 12, rerank: bool = True,
                  candidates: int | None = None) -> dict:
        """history: product ids, oldest first. Returns ranked product ids with
        scores, explanations and per-stage latency."""
        t_start = time.perf_counter()
        timings = {}
        hist_idx = self.to_index(history)
        seen = set(hist_idx)
        if not hist_idx:
            picks = [i for i in self.trending[: k + len(seen)] if i not in seen][:k]
            timings["total_ms"] = (time.perf_counter() - t_start) * 1000
            return {"strategy": "trending", "timings": timings, "items": [
                {"product_id": self.to_product(i), "score": float(self.store.recent_popularity[i]),
                 "retrieval_rank": None, "reasons": [{"feature": "recent_popularity",
                                                       "label": FEATURE_LABELS["recent_popularity"],
                                                       "contribution": None}],
                 "anchor_product_id": None}
                for i in picks]}

        padded = self.padded(hist_idx)
        t0 = time.perf_counter()
        user = self.user_vector(padded)
        timings["user_tower_ms"] = (time.perf_counter() - t0) * 1000

        n_cand = candidates or settings.RETRIEVAL_CANDIDATES
        t0 = time.perf_counter()
        scores, ids = self.items.search(user, n_cand + len(seen))
        keep = [(i, s) for i, s in zip(ids[0], scores[0]) if i > 0 and i not in seen][:n_cand]
        cands = np.array([i for i, _ in keep], dtype=np.int64)
        cand_scores = np.array([s for _, s in keep], dtype=np.float32)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        reasons = [[] for _ in cands]
        if rerank and len(cands):
            t0 = time.perf_counter()
            feats = self.store.compute(padded, cands[None], cand_scores[None])[0]
            final = self.ranker.predict(feats, num_threads=LGB_THREADS)
            order = np.argsort(-final)[:k]
            contrib = self.ranker.predict(feats[order], pred_contrib=True, num_threads=LGB_THREADS)[:, :-1]
            for row, c in enumerate(contrib):
                top = [j for j in np.argsort(-c) if c[j] > 0 and FEATURES[j] not in HIDDEN_REASONS][:3]
                reasons[order[row]] = [{"feature": FEATURES[j], "label": FEATURE_LABELS[FEATURES[j]],
                                        "contribution": float(c[j])} for j in top]
            timings["ranking_ms"] = (time.perf_counter() - t0) * 1000
            strategy = "two_tower+ltr"
        else:
            order = np.arange(min(k, len(cands)))
            final = cand_scores
            strategy = "two_tower"

        # Anchor: the history item whose vector is closest to each recommendation.
        hist_arr = np.array(hist_idx[-self.max_len:])
        sims = self.item_vecs[cands[order]] @ self.item_vecs[hist_arr].T
        anchors = hist_arr[sims.argmax(1)]

        timings["total_ms"] = (time.perf_counter() - t_start) * 1000
        return {"strategy": strategy, "timings": timings, "items": [
            {"product_id": self.to_product(cands[j]), "score": float(final[j]),
             "retrieval_rank": int(j) + 1, "reasons": reasons[j],
             "anchor_product_id": self.to_product(a)}
            for j, a in zip(order, anchors)]}

    def similar(self, product_id: int, k: int = 12) -> list[dict]:
        idx = self.index_of.get(product_id)
        if idx is None:
            return []
        scores, ids = self.items.search(self.item_vecs[idx][None], k + 1)
        return [{"product_id": self.to_product(i), "score": float(s)}
                for i, s in zip(ids[0], scores[0]) if i > 0 and i != idx][:k]

    def _clip_text(self):
        with self._clip_lock:
            if self._clip is None:
                tokenizer = Tokenizer.from_file(str(self.path / "clip_tokenizer" / "tokenizer.json"))
                tokenizer.enable_truncation(77)
                self._clip = (tokenizer, _session(self.path / "clip_text.onnx"))
        return self._clip

    def search(self, query: str, k: int = 24, depth: int = 200, rrf_k: int = 60) -> tuple[list[dict], float]:
        """Text-to-product search in CLIP space: rank products by how well their
        image and their listing text match the query, fused with reciprocal
        rank fusion (Cormack et al., 2009)."""
        tokenizer, session = self._clip_text()
        t0 = time.perf_counter()
        ids = np.array([tokenizer.encode(query).ids], dtype=np.int64)
        q = session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0].astype(np.float32)
        fused: dict[int, float] = {}
        for index in (self.search_image, self.search_text):
            _, hits = index.search(q, depth)
            for rank, i in enumerate(hits[0]):
                if i > 0:
                    fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + rank + 1)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        ms = (time.perf_counter() - t0) * 1000
        return [{"product_id": self.to_product(i), "score": s} for i, s in top], ms


_service: Recommender | None = None
_lock = threading.Lock()


def get_recommender() -> Recommender:
    global _service
    with _lock:
        if _service is None:
            _service = Recommender()
    return _service


def reload_recommender() -> Recommender:
    global _service
    with _lock:
        _service = None
    return get_recommender()
