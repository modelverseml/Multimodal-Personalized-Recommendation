"""Offline ranking metrics. All metrics are computed by full ranking over the
whole catalog (no sampled negatives), which avoids the inflated numbers that
sampled metrics produce (Krichene & Rendle, 2020)."""
import numpy as np


def target_ranks(top: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """0-based rank of each target within its top-k list, or -1 if absent."""
    hits = top == targets[:, None]
    return np.where(hits.any(1), hits.argmax(1), -1)


def ranking_metrics(top: np.ndarray, targets: np.ndarray, ks=(10, 20)) -> dict[str, float]:
    ranks = target_ranks(top, targets)
    out = {}
    for k in ks:
        hit = (ranks >= 0) & (ranks < k)
        out[f"recall@{k}"] = float(hit.mean())
        gain = np.where(hit, 1.0 / np.log2(np.maximum(ranks, 0) + 2), 0.0)
        out[f"ndcg@{k}"] = float(gain.mean())
    out["mrr"] = float(np.where(ranks >= 0, 1.0 / (np.maximum(ranks, 0) + 1), 0.0).mean())
    return out


def beyond_accuracy(top: np.ndarray, n_items: int, category_codes: np.ndarray,
                    item_counts: np.ndarray, k: int = 10) -> dict[str, float]:
    top = top[:, :k]
    coverage = len(np.unique(top)) / n_items
    cats = category_codes[top]
    diversity = np.mean([len(np.unique(row)) for row in cats]) / k
    p = item_counts / item_counts.sum()
    novelty = float(np.mean(-np.log2(np.maximum(p[top], 1e-12))))
    return {f"coverage@{k}": float(coverage), f"category_diversity@{k}": float(diversity), "novelty": novelty}


def percentiles(samples_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }
