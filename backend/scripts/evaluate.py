"""Offline evaluation on the held-out test interactions.

    python -m scripts.evaluate [--users 20000] [--latency-samples 1000]

Protocol: leave-last-out on the official 5-core split. For each test user the
model sees train + valid history and must rank the single held-out item
against the *entire* catalog (history items excluded). Recall@K equals HitRate@K
with one relevant item per user.

Compares Popularity, ItemKNN-CF, BPR-MF and a zero-training CLIP content
baseline against the two-tower retriever and the two-tower + LambdaRank
pipeline; reports bootstrap 95% CIs, beyond-accuracy metrics, ablations, FAISS
HNSW recall vs exact search, and end-to-end serving latency.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# This process uses PyTorch and LightGBM together, and on macOS each bundles its
# own OpenMP runtime; their worker threads crash when both run parallel regions.
# One OpenMP thread avoids that. Matrix products still go through BLAS
# (Accelerate), so evaluation stays fast. torch is also imported first.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import lightgbm as lgb
import mlflow
import numpy as np

from app.config import BACKEND_DIR, settings
from app.ml.baselines import BPRMF, ClipKNN, ItemKNN, Popularity
from app.ml.data import item_counts, pad_left
from app.ml.metrics import beyond_accuracy, percentiles, ranking_metrics, target_ranks
from app.ml.towers import TowerConfig
from app.ml.train import (
    build_feature_store, build_model, candidate_features, mlflow_metrics, prepare, rerank_top, retrieve,
)

K = 20


def top_from_scores(score_fn, histories, k=K, batch=2048, device="cpu"):
    tops = []
    for s in range(0, len(histories), batch):
        chunk = histories[s:s + batch]
        scores = torch.as_tensor(np.asarray(score_fn(chunk, s), dtype=np.float32), device=device)
        scores[:, 0] = -torch.inf
        seen = torch.as_tensor(pad_left(chunk, max(len(h) for h in chunk)), device=device)
        scores.scatter_(1, seen, -torch.inf)
        scores[:, 0] = -torch.inf
        tops.append(scores.topk(k, dim=1).indices.cpu().numpy())
    return np.concatenate(tops)


def bootstrap_ci(per_user: np.ndarray, n: int = 1000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [per_user[rng.integers(0, len(per_user), len(per_user))].mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarise(name, top, targets, ctx, counts, seconds):
    ranks = target_ranks(top, targets)
    ndcg_user = np.where((ranks >= 0) & (ranks < 10), 1.0 / np.log2(np.maximum(ranks, 0) + 2), 0.0)
    row = {
        "model": name,
        **ranking_metrics(top, targets, ks=(10, 20)),
        **beyond_accuracy(top, ctx.catalog.n_items, ctx.catalog.category_codes, counts),
        "ndcg@10_ci95": bootstrap_ci(ndcg_user),
        "seconds": round(seconds, 1),
    }
    print(f"  {name:<32} recall@10 {row['recall@10']:.4f}  ndcg@10 {row['ndcg@10']:.4f}  "
          f"recall@20 {row['recall@20']:.4f}  coverage@10 {row['coverage@10']:.3f}  ({seconds:.0f}s)")
    return row, ndcg_user


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=0, help="evaluate on a random sample (0 = all test users)")
    parser.add_argument("--latency-samples", type=int, default=1000)
    parser.add_argument("--skip-bpr", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")
    ctx = prepare()
    current = settings.artifact_dir / "current"
    manifest = json.loads((current / "manifest.json").read_text())

    split = ctx.split
    users = np.flatnonzero(split.test > 0)
    if args.users:
        users = np.sort(np.random.default_rng(0).choice(users, args.users, replace=False))
    histories_all = split.history("test")
    histories = [histories_all[u] for u in users]
    targets = split.test[users]
    counts = item_counts(histories_all, ctx.catalog.n_items)
    n = ctx.catalog.n_items
    print(f"evaluating {len(users):,} test users over {n:,} items")

    results, per_user = [], {}

    def run(name, fn):
        t0 = time.perf_counter()
        top = fn()
        row, nd = summarise(name, top, targets, ctx, counts, time.perf_counter() - t0)
        results.append(row)
        per_user[name] = nd
        return top

    pop = Popularity(histories_all, n)
    run("Popularity", lambda: top_from_scores(lambda h, s: pop.scores(h), histories))
    knn = ItemKNN(histories_all, n)
    run("ItemKNN-CF", lambda: top_from_scores(lambda h, s: knn.scores(h), histories))
    if not args.skip_bpr:
        def bpr_top():
            bpr = BPRMF(histories_all, n, device=str(device))
            return top_from_scores(lambda h, s: bpr.scores_for_users(users[s:s + len(h)]), histories)
        run("BPR-MF", bpr_top)
    clip = ClipKNN(ctx.clip_text, ctx.clip_image)
    run("CLIP-kNN (content only)", lambda: top_from_scores(lambda h, s: clip.scores(h), histories))

    # Two-tower retrieval (exact search, i.e. what FAISS approximates).
    tower_cfg = TowerConfig(**manifest["tower_config"])
    model = build_model(ctx, tower_cfg)
    model.load_state_dict(torch.load(current / "model.pt", map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    cands_box = {}

    def two_tower():
        cands, _ = retrieve(model, histories, 100, device)
        cands_box["cands"] = cands
        return cands[:, :K]
    run("Two-Tower (multimodal)", two_tower)
    candidate_recall = float((cands_box["cands"] == targets[:, None]).any(1).mean())

    def with_ranker():
        stamps = [t[: len(h)] for t, h in zip(split.timestamps, histories_all)]
        store = build_feature_store(ctx, model.all_item_vectors().cpu().numpy(), histories_all, stamps)
        ranker = lgb.Booster(model_file=str(current / "ranker.txt"))
        cands, feats = candidate_features(model, store, histories, 100, device)
        return rerank_top(ranker, cands, feats, K)
    run("Two-Tower + LambdaRank", with_ranker)

    # Ablations: retrieval-stage metrics of each two-tower variant on the same test protocol.
    ablations = []
    for variant in ("id_only", "id_text", "content_only"):
        path = settings.artifact_dir / "ablation" / f"{variant}.json"
        if path.exists():
            data = json.loads(path.read_text())
            ablations.append({"variant": variant, **data["test"], "best_epoch": data["best_epoch"],
                              "run_id": data["run_id"]})
    full = next(r for r in results if r["model"] == "Two-Tower (multimodal)")
    ablations.append({"variant": "full", **{k: full[k] for k in ("recall@10", "ndcg@10", "recall@20", "ndcg@20", "mrr")},
                      "best_epoch": manifest["best_epoch"], "run_id": manifest["run_id"]})

    best_baseline = max((r for r in results if not r["model"].startswith("Two-Tower")), key=lambda r: r["ndcg@10"])
    ours = next(r for r in results if r["model"] == "Two-Tower + LambdaRank")
    diff = per_user["Two-Tower + LambdaRank"] - per_user[best_baseline["model"]]
    lift = {
        "baseline": best_baseline["model"],
        "ndcg@10_relative_lift": ours["ndcg@10"] / best_baseline["ndcg@10"] - 1,
        "recall@10_relative_lift": ours["recall@10"] / best_baseline["recall@10"] - 1,
        "ndcg@10_diff_ci95": bootstrap_ci(diff),
    }

    # The serving stack (ONNX Runtime + FAISS + LightGBM) cannot share a process
    # with PyTorch on macOS, so the latency benchmark runs as its own process.
    print("serving latency benchmark (separate process, real serving stack)")
    latency_path = settings.artifact_dir / "latency.json"
    subprocess.run([sys.executable, "-m", "scripts.benchmark", "--samples", str(args.latency_samples),
                    "--out", str(latency_path)], check=True, cwd=BACKEND_DIR)
    bench = json.loads(latency_path.read_text())
    latency, faiss_stats = bench["latency"], bench["faiss"]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": manifest["model_version"],
        "run_id": manifest["run_id"],
        "dataset": manifest["dataset"],
        "protocol": "leave-last-out, full-catalog ranking, history items excluded",
        "n_users": int(len(users)),
        "n_items": int(n),
        "results": results,
        "candidate_recall@100": candidate_recall,
        "lift_vs_best_baseline": lift,
        "ablations": ablations,
        "latency": latency,
        "faiss": faiss_stats,
        "semantic_search": bench["semantic_search"],
    }
    out = settings.artifact_dir / "evaluation.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    print(f"lift vs {best_baseline['model']}: NDCG@10 {lift['ndcg@10_relative_lift']:+.1%}, "
          f"Recall@10 {lift['recall@10_relative_lift']:+.1%}")

    mlflow.set_tracking_uri(settings.mlflow_uri)
    with mlflow.start_run(run_id=manifest["run_id"]):
        for r in results:
            slug = r["model"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
            mlflow.log_metrics(mlflow_metrics({m: r[m] for m in ("recall@10", "ndcg@10", "recall@20", "mrr")}, f"test_{slug}_"))
        mlflow.log_metrics({
            "test_latency_p95_ms": latency["with_ranker.total_ms"]["p95_ms"],
            "test_latency_p99_ms": latency["with_ranker.total_ms"]["p99_ms"],
            "test_hnsw_recall_vs_exact": faiss_stats["hnsw_recall_vs_exact"],
            "test_ndcg10_lift_vs_best_baseline": lift["ndcg@10_relative_lift"],
        })
        mlflow.log_artifact(str(out), artifact_path="evaluation")


if __name__ == "__main__":
    main()
