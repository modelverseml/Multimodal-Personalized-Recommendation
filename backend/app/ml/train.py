"""End-to-end training pipeline.

    python -m app.ml.train                    # full multimodal model + ranker, promoted to serving
    python -m app.ml.train --variant id_only  # ablation run (tracked in MLflow, not promoted)

Steps:
 1. Load catalog + interaction log from PostgreSQL; leave-last-out split.
 2. Load/compute CLIP text + image embeddings (cached by ASIN).
 3. Train the two-tower model with mixed-negative sampled softmax; early-stop on
    validation NDCG@10 (full-catalog ranking).
 4. Train a LightGBM LambdaRank reranker on validation-stage candidates.
 5. Build FAISS indexes and serving feature store from all interactions.
 6. Log params, curves, metrics and artifacts to MLflow; write a versioned
    artifact directory and point artifacts/current at it.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import numpy as np
import torch

from app.config import BACKEND_DIR, settings
from app.ml.baselines import item_knn_matrix
from app.ml.data import Catalog, Split, build_split, item_counts, load_catalog, load_events, pad_left
from app.ml.export import export_clip_text, export_user_tower
from app.ml.features import best_device, build_item_features
from app.ml.metrics import ranking_metrics
from app.ml.ranking import FEATURES, LGB_THREADS, FeatureStore, train_ranker
from app.ml.towers import TowerConfig, TwoTower, sampled_softmax_loss

VARIANTS = {
    "full": dict(use_id=True, use_text=True, use_image=True),
    "id_only": dict(use_id=True, use_text=False, use_image=False),
    "id_text": dict(use_id=True, use_text=True, use_image=False),
    "content_only": dict(use_id=False, use_text=True, use_image=True),
}


@dataclass
class TrainConfig:
    variant: str = "full"
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    temperature: float = 0.05
    n_uniform: int = 4096
    patience: int = 3
    seed: int = 42
    candidates: int = 100
    ranker_users: int = 30000


@dataclass
class Context:
    catalog: Catalog
    split: Split
    clip_text: np.ndarray   # (n+1, 512), row 0 zeros
    clip_image: np.ndarray
    has_image: np.ndarray


def prepare(include_live: bool = False) -> Context:
    t0 = time.perf_counter()
    catalog = load_catalog()
    events = load_events(catalog, include_live=include_live)
    split = build_split(events)
    clip = build_item_features(catalog.products)
    pad = lambda m: np.vstack([np.zeros((1, m.shape[1]), m.dtype), m])
    print(f"loaded {catalog.n_items:,} items, {len(split.customer_ids):,} customers, "
          f"{len(events):,} events in {time.perf_counter() - t0:.1f}s")
    return Context(catalog, split, pad(clip["text"]), pad(clip["image"]),
                   np.concatenate([[False], clip["has_image"]]))


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(ctx: Context, tower_cfg: TowerConfig) -> TwoTower:
    return TwoTower(
        tower_cfg,
        torch.as_tensor(ctx.catalog.category_codes),
        torch.as_tensor(ctx.clip_text) if tower_cfg.use_text else None,
        torch.as_tensor(ctx.clip_image) if tower_cfg.use_image else None,
    )


@torch.no_grad()
def retrieve(model: TwoTower, histories: list[np.ndarray], k: int, device,
             batch: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by inner product over the full catalog, excluding padding
    and items already in the history. Returns (indices, scores)."""
    model.eval()
    item_vecs = model.all_item_vectors()
    max_len = model.cfg.max_len
    # Process users sorted by history length so each batch is padded only to
    # its own longest history; results are scattered back to input order.
    order = np.argsort([len(h) for h in histories], kind="stable")
    tops = np.zeros((len(histories), k), np.int64)
    scores = np.zeros((len(histories), k), np.float32)
    for s in range(0, len(order), batch):
        rows = order[s:s + batch]
        chunk = [histories[i] for i in rows]
        longest = max(len(h) for h in chunk) or 1
        hist = torch.as_tensor(pad_left(chunk, min(max_len, longest)), device=device)
        users = model.user_vector(hist)
        logits = users @ item_vecs.T
        seen = torch.as_tensor(pad_left(chunk, longest), device=device)
        logits.scatter_(1, seen, -torch.inf)
        logits[:, 0] = -torch.inf
        val, idx = logits.topk(k, dim=1)
        tops[rows] = idx.cpu().numpy()
        scores[rows] = val.cpu().numpy()
    return tops, scores


def eval_users(split: Split, stage: str) -> np.ndarray:
    return np.flatnonzero(split.targets(stage) > 0)


def evaluate_retrieval(model, ctx: Context, stage: str, device, k: int = 20) -> dict[str, float]:
    users = eval_users(ctx.split, stage)
    hist = ctx.split.history(stage)
    top, _ = retrieve(model, [hist[u] for u in users], k, device)
    return ranking_metrics(top, ctx.split.targets(stage)[users], ks=(10, 20))


def length_buckets(lengths: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Batches of similar-length sequences, in random order. Noise on the sort
    key reshuffles which sequences share a batch from epoch to epoch."""
    order = np.argsort(lengths + np.random.uniform(0, 2, len(lengths)), kind="stable")
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    np.random.shuffle(batches)
    return batches


def mlflow_metrics(metrics: dict, prefix: str = "") -> dict:
    """MLflow metric names cannot contain '@'."""
    return {f"{prefix}{k}".replace("@", "_at_"): v for k, v in metrics.items()}


def train_two_tower(ctx: Context, cfg: TrainConfig, device, log_mlflow: bool = True):
    seed_everything(cfg.seed)
    tower_cfg = TowerConfig(
        n_items=ctx.catalog.n_items,
        n_categories=len(ctx.catalog.category_names) - 1,
        clip_dim=ctx.clip_text.shape[1],
        max_len=settings.MAX_HISTORY,
        **VARIANTS[cfg.variant],
    )
    model = build_model(ctx, tower_cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    counts = item_counts(ctx.split.train, ctx.catalog.n_items) + 1.0
    counts[0] = 0
    log_pop = torch.as_tensor(np.log(counts / counts.sum()), dtype=torch.float32, device=device)

    seqs = [s for s in ctx.split.train if len(s) >= 2]
    padded = pad_left(seqs, tower_cfg.max_len + 1)
    lengths = np.minimum([len(s) for s in seqs], tower_cfg.max_len + 1)
    steps = (len(padded) + cfg.batch_size - 1) // cfg.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps, pct_start=0.1)
    print(f"[{cfg.variant}] {sum(p.numel() for p in params):,} trainable params, "
          f"{len(seqs):,} training sequences, device={device}")

    best, best_state, history, bad = -1.0, None, [], 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        start = time.perf_counter()
        total = 0.0
        for rows in length_buckets(lengths, cfg.batch_size):
            # Median history is ~4 items: trimming each batch to its own longest
            # sequence avoids spending most compute on padding.
            batch = torch.as_tensor(padded[rows][:, -lengths[rows].max():], device=device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            states = model.encode_history(inp)
            valid = (inp > 0) & (tgt > 0)
            loss = sampled_softmax_loss(
                states[valid], tgt[valid], model.item_tower, log_pop, cfg.n_uniform, cfg.temperature,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            total += loss.item()

        metrics = evaluate_retrieval(model, ctx, "valid", device)
        row = {"epoch": epoch, "loss": total / steps, "seconds": time.perf_counter() - start, **metrics}
        history.append(row)
        print(f"  epoch {epoch:2d} loss {row['loss']:.4f}  val recall@10 {metrics['recall@10']:.4f} "
              f"ndcg@10 {metrics['ndcg@10']:.4f}  ({row['seconds']:.0f}s)")
        if log_mlflow:
            mlflow.log_metrics({"train_loss": row["loss"], **mlflow_metrics(metrics, "val_")}, step=epoch)

        if metrics["ndcg@10"] > best:
            best, bad = metrics["ndcg@10"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"  early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, tower_cfg, history


def build_feature_store(ctx: Context, item_vecs: np.ndarray, sequences: list[np.ndarray],
                        stamps: list[np.ndarray]) -> FeatureStore:
    """Feature tables computed from the given interaction sequences only, so
    offline features never see the interactions being predicted."""
    n = ctx.catalog.n_items
    counts = item_counts(sequences, n)
    all_ts = np.concatenate(stamps)
    cutoff = np.quantile(all_ts, 0.9)
    recent = [s[t >= cutoff] for s, t in zip(sequences, stamps)]
    p = ctx.catalog.products
    with_pad = lambda a, fill=0.0: np.concatenate([[fill], a]).astype(np.float32)
    return FeatureStore(
        item_vecs=item_vecs.astype(np.float32),
        clip_text=ctx.clip_text,
        clip_image=ctx.clip_image,
        has_image=ctx.has_image.astype(np.float32),
        cooc=item_knn_matrix(sequences, n),
        category=ctx.catalog.category_codes.astype(np.int64),
        store=ctx.catalog.store_codes.astype(np.int64),
        popularity=np.log1p(counts).astype(np.float32),
        recent_popularity=np.log1p(item_counts(recent, n)).astype(np.float32),
        price=with_pad(np.log1p(p["price"].to_numpy(dtype=float)), np.nan),
    )


def candidate_features(model, store: FeatureStore, histories: list[np.ndarray], k: int, device,
                       batch: int = 512):
    """Retrieve top-k candidates and compute ranking features for each user."""
    cands, scores = retrieve(model, histories, k, device)
    padded = pad_left(histories, settings.MAX_HISTORY)
    feats = np.concatenate([
        store.compute(padded[s:s + batch], cands[s:s + batch], scores[s:s + batch])
        for s in range(0, len(histories), batch)
    ])
    return cands, feats


def rerank_top(ranker, cands: np.ndarray, feats: np.ndarray, k: int) -> np.ndarray:
    b, c, f = feats.shape
    scores = ranker.predict(feats.reshape(-1, f), num_threads=LGB_THREADS).reshape(b, c)
    order = np.argsort(-scores, axis=1)[:, :k]
    return np.take_along_axis(cands, order, 1)


def fit_ranker(model, ctx: Context, cfg: TrainConfig, device):
    """Train LambdaRank on validation-stage candidates (history = train split),
    so the ranker learns from targets the retrieval model never trained on."""
    rng = np.random.default_rng(cfg.seed)
    users = eval_users(ctx.split, "valid")
    users = rng.permutation(users)[: cfg.ranker_users]
    hist = ctx.split.history("valid")
    train_stamps = [t[: len(h)] for t, h in zip(ctx.split.timestamps, ctx.split.train)]
    item_vecs = model.all_item_vectors().cpu().numpy()
    store = build_feature_store(ctx, item_vecs, ctx.split.train, train_stamps)

    t0 = time.perf_counter()
    cands, feats = candidate_features(model, store, [hist[u] for u in users], cfg.candidates, device)
    labels = (cands == ctx.split.valid[users][:, None]).astype(np.int32)
    print(f"ranker features for {len(users):,} users in {time.perf_counter() - t0:.0f}s; "
          f"retrieval recall@{cfg.candidates} = {labels.any(1).mean():.4f}")

    n_hold = len(users) // 10
    hold, fit = np.arange(n_hold), np.arange(n_hold, len(users))
    fit = fit[labels[fit].any(1)]                      # groups without a positive carry no signal
    hold_pos = hold[labels[hold].any(1)]
    flat = lambda idx: (feats[idx].reshape(-1, len(FEATURES)), labels[idx].ravel(),
                        np.full(len(idx), cfg.candidates))
    ranker = train_ranker(*flat(fit), *flat(hold_pos), seed=cfg.seed)

    # Held-out users (including those whose target was not retrieved).
    targets = ctx.split.valid[users[hold]]
    before = ranking_metrics(cands[hold][:, :20], targets, ks=(10, 20))
    after = ranking_metrics(rerank_top(ranker, cands[hold], feats[hold], 20), targets, ks=(10, 20))
    print(f"ranker on held-out valid users: ndcg@10 {before['ndcg@10']:.4f} -> {after['ndcg@10']:.4f}, "
          f"recall@10 {before['recall@10']:.4f} -> {after['recall@10']:.4f}")
    importance = dict(zip(FEATURES, ranker.booster_.feature_importance("gain").round(1).tolist()))
    return ranker, {"retrieval": before, "reranked": after, "best_iteration": int(ranker.best_iteration_ or 0)}, importance


def promote(run_dir, ctx: Context, model: TwoTower, tower_cfg: TowerConfig, ranker, manifest: dict) -> None:
    """Write the serving bundle. Serving features use *all* interactions."""
    run_dir.mkdir(parents=True, exist_ok=True)
    item_vecs = model.all_item_vectors().cpu().numpy()
    torch.save(model.state_dict(), run_dir / "model.pt")
    np.save(run_dir / "item_vecs.npy", item_vecs)
    np.save(run_dir / "product_ids.npy", ctx.catalog.product_ids)
    ranker.booster_.save_model(str(run_dir / "ranker.txt"))

    full = ctx.split.history("test")
    full = [np.append(h, t) if t else h for h, t in zip(full, ctx.split.test)]
    store = build_feature_store(ctx, item_vecs, full, ctx.split.timestamps)
    store.save(run_dir / "features")

    # Serving runs without PyTorch: export the online models to ONNX, verified
    # against torch, and build FAISS indexes in a separate (torch-free) process.
    manifest["onnx_max_abs_diff"] = {
        "user_tower": export_user_tower(model.user_tower, run_dir / "user_tower.onnx"),
        "clip_text": export_clip_text(settings.CLIP_MODEL, run_dir),
    }
    print(f"ONNX export verified: {manifest['onnx_max_abs_diff']}")
    subprocess.run([sys.executable, "-m", "app.ml.index", str(run_dir)], check=True, cwd=BACKEND_DIR)

    # Reference distributions for drift monitoring.
    cats = ctx.catalog.category_codes[np.concatenate(full)]
    names = ctx.catalog.category_names
    cat_counts = np.bincount(cats, minlength=len(names))
    pop = item_counts(full, ctx.catalog.n_items)
    manifest["reference"] = {
        "category_distribution": {names[i]: int(c) for i, c in enumerate(cat_counts) if i and c},
        "popularity_deciles": np.quantile(pop[1:], np.linspace(0.1, 0.9, 9)).tolist(),
    }
    manifest["tower_config"] = tower_cfg.to_dict()
    manifest["category_names"] = names
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    current = settings.artifact_dir / "current"
    if current.is_symlink() or current.exists():
        current.unlink() if current.is_symlink() else shutil.rmtree(current)
    current.symlink_to(run_dir.relative_to(settings.artifact_dir), target_is_directory=True)
    print(f"promoted {run_dir.name} -> {current}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--no-promote", action="store_true", help="track in MLflow but do not update serving")
    # Length-bucketed batches have many distinct shapes; on Apple MPS that
    # defeats the allocator cache, so CPU is both faster and leaner here.
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda | auto")
    parser.add_argument("--include-live", action="store_true",
                        help="also train on live app events (production retrain; not for benchmarking)")
    args = parser.parse_args()
    cfg = TrainConfig(variant=args.variant, epochs=args.epochs)
    promote_run = args.variant == "full" and not args.no_promote

    device = best_device() if args.device == "auto" else torch.device(args.device)
    ctx = prepare(include_live=args.include_live)
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"two-tower-{cfg.variant}") as run:
        mlflow.set_tags({"dataset": f"amazon-reviews-2023/{settings.DATASET_CATEGORY}/5core-last-out",
                         "include_live": str(args.include_live),
                         "variant": cfg.variant, "clip_model": settings.CLIP_MODEL})
        mlflow.log_params({**asdict(cfg), "n_items": ctx.catalog.n_items,
                           "n_customers": len(ctx.split.customer_ids), "device": str(device)})
        t0 = time.perf_counter()
        model, tower_cfg, history = train_two_tower(ctx, cfg, device)
        mlflow.log_params({f"tower_{k}": v for k, v in tower_cfg.to_dict().items() if k not in ("n_items",)})
        best = max(history, key=lambda r: r["ndcg@10"])
        mlflow.log_metrics(mlflow_metrics({k: v for k, v in best.items() if "@" in k or k == "mrr"}, "best_val_"))
        mlflow.log_metric("train_seconds", time.perf_counter() - t0)

        manifest = {
            "run_id": run.info.run_id,
            "model_version": f"{cfg.variant}-{run.info.run_id[:8]}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": f"Amazon Reviews 2023 / {settings.DATASET_CATEGORY} (5-core)",
            "clip_model": settings.CLIP_MODEL,
            "n_items": ctx.catalog.n_items,
            "n_customers": int(len(ctx.split.customer_ids)),
            "train_config": asdict(cfg),
            "best_epoch": best["epoch"],
            "validation": {k: v for k, v in best.items() if "@" in k or k == "mrr"},
            "history": history,
        }

        if promote_run:
            ranker, ranker_metrics, importance = fit_ranker(model, ctx, cfg, device)
            mlflow.log_metrics(mlflow_metrics(ranker_metrics["reranked"], "ranker_val_"))
            manifest["ranker"] = {**ranker_metrics, "feature_importance": importance, "features": FEATURES}
            run_dir = settings.artifact_dir / "runs" / manifest["model_version"]
            promote(run_dir, ctx, model, tower_cfg, ranker, manifest)
            for name in ("manifest.json", "ranker.txt", "model.pt"):
                mlflow.log_artifact(str(run_dir / name), artifact_path="serving")
        else:
            out = settings.artifact_dir / "ablation"
            out.mkdir(parents=True, exist_ok=True)
            test = evaluate_retrieval(model, ctx, "test", torch.device(device))
            mlflow.log_metrics(mlflow_metrics(test, "test_"))
            result = {**manifest, "test": test}
            (out / f"{cfg.variant}.json").write_text(json.dumps(result, indent=2))
            print(f"[{cfg.variant}] test {json.dumps({k: round(v, 4) for k, v in test.items()})}")


if __name__ == "__main__":
    main()
