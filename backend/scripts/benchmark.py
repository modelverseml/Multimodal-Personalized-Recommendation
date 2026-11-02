"""Serving latency benchmark on the real (torch-free) serving stack.

    python -m scripts.benchmark [--samples 1000] [--out artifacts/latency.json]

Replays test users' train+valid histories through Recommender.recommend()
exactly as the API does (ONNX user tower -> FAISS HNSW -> LightGBM), with and
without the reranker, and measures FAISS HNSW recall/latency against exact
inner-product search. Runs in its own process because the serving stack
cannot share a process with PyTorch on macOS (see app/ml/inference.py).
"""
import argparse
import json
import time

import faiss
import numpy as np

from app.config import settings
from app.ml import index as faiss_index
from app.ml.data import build_split, load_catalog, load_events
from app.ml.inference import Recommender
from app.ml.metrics import percentiles


def latency(service: Recommender, histories: list[list[int]]) -> dict:
    stages: dict[str, list[float]] = {}
    for rerank in (True, False):
        key = "with_ranker" if rerank else "retrieval_only"
        for h in histories[:20]:  # warm-up
            service.recommend(h, rerank=rerank)
        for h in histories:
            for stage, ms in service.recommend(h, rerank=rerank)["timings"].items():
                stages.setdefault(f"{key}.{stage}", []).append(ms)
    return {stage: percentiles(v) for stage, v in stages.items()}


def faiss_vs_exact(service: Recommender, user_vecs: np.ndarray, k: int = 100) -> dict:
    exact = faiss.IndexFlatIP(service.item_vecs.shape[1])
    exact.add(np.ascontiguousarray(service.item_vecs[1:]))
    results, times = {}, {}
    for name, index, offset in (("exact_flat", exact, 1), ("hnsw", service.items, 0)):
        ids_all, t_all = [], []
        for v in user_vecs:
            t0 = time.perf_counter()
            _, ids = index.search(v[None], k)
            t_all.append((time.perf_counter() - t0) * 1000)
            ids_all.append(ids[0] + offset)
        results[name], times[name] = ids_all, percentiles(t_all)
    overlap = np.mean([len(set(a) & set(b)) / k for a, b in zip(results["hnsw"], results["exact_flat"])])
    return {
        "k": k,
        "hnsw_recall_vs_exact": float(overlap),
        "exact_flat": times["exact_flat"],
        "hnsw": times["hnsw"],
        "hnsw_params": {"M": faiss_index.HNSW_M, "efConstruction": faiss_index.EF_CONSTRUCTION,
                        "efSearch": faiss_index.EF_SEARCH},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--out", default=str(settings.artifact_dir / "latency.json"))
    args = parser.parse_args()

    catalog = load_catalog()
    split = build_split(load_events(catalog))
    users = np.flatnonzero(split.test > 0)
    rng = np.random.default_rng(1)
    users = rng.choice(users, min(args.samples, len(users)), replace=False)
    hist = split.history("test")
    histories = [[int(catalog.product_ids[i - 1]) for i in hist[u]] for u in users]

    service = Recommender()
    report = {"samples": len(histories), "latency": latency(service, histories)}
    user_vecs = np.concatenate([service.user_vector(service.padded(service.to_index(h))) for h in histories])
    report["faiss"] = faiss_vs_exact(service, user_vecs)
    search_ms = [service.search(q)[1] for q in ["racing wheel", "cute platformer", "horror game", "retro rpg"] * 10]
    report["semantic_search"] = percentiles(search_ms[4:])
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    total = report["latency"]["with_ranker.total_ms"]
    print(f"end-to-end p50 {total['p50_ms']:.1f} ms, p95 {total['p95_ms']:.1f} ms, p99 {total['p99_ms']:.1f} ms; "
          f"HNSW recall@100 vs exact {report['faiss']['hnsw_recall_vs_exact']:.3f}")


if __name__ == "__main__":
    main()
