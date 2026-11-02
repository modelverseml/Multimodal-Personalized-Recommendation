"""Production monitoring: distribution drift (PSI) and online engagement.

PSI compares the category mix (and item-popularity mix) of recent live events
against the training reference saved in the model manifest. The 0.2 cutoff is
a common monitoring heuristic, not a statistical test or an automatic
retraining trigger.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Impression, Interaction, Product

EPS = 1e-4


def psi(expected: dict[str, float], actual: dict[str, float]) -> tuple[float, list[dict]]:
    keys = sorted(set(expected) | set(actual))
    e = np.array([expected.get(k, 0.0) for k in keys], dtype=float)
    a = np.array([actual.get(k, 0.0) for k in keys], dtype=float)
    e = np.maximum(e / max(e.sum(), EPS), EPS)
    a = np.maximum(a / max(a.sum(), EPS), EPS)
    terms = (a - e) * np.log(a / e)
    rows = [{"bucket": k, "expected": float(x), "actual": float(y), "psi": float(t)}
            for k, x, y, t in zip(keys, e, a, terms)]
    rows.sort(key=lambda r: -abs(r["psi"]))
    return float(terms.sum()), rows


def drift_report(db: Session, manifest: dict, days: int = 7, min_events: int = 50) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    live = db.execute(
        select(Product.category, func.count())
        .join(Interaction, Interaction.product_id == Product.id)
        .where(Interaction.source == "live", Interaction.created_at >= since)
        .group_by(Product.category)
    ).all()
    actual = {c: float(n) for c, n in live}
    n_live = int(sum(actual.values()))
    reference = manifest.get("reference", {})
    expected = reference.get("category_distribution", {})

    report = {
        "window_days": days,
        "live_events": n_live,
        "threshold": settings.DRIFT_PSI_THRESHOLD,
        "model_version": manifest.get("model_version"),
    }
    if n_live < min_events:
        return {**report, "status": "insufficient_data", "psi": None, "buckets": [],
                "message": f"Need at least {min_events} live events in the last {days} days (have {n_live})."}

    value, rows = psi(expected, actual)
    status = "drift" if value >= settings.DRIFT_PSI_THRESHOLD else "warning" if value >= 0.1 else "stable"
    return {**report, "status": status, "psi": value, "buckets": rows[:12]}


def engagement(db: Session, days: int = 7) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    impressions = db.scalar(select(func.count()).select_from(Impression).where(Impression.created_at >= since)) or 0
    requests = db.scalar(
        select(func.count(func.distinct(Impression.request_id))).where(Impression.created_at >= since)
    ) or 0
    attributed = db.execute(
        select(Interaction.event_type, func.count())
        .where(Interaction.source == "live", Interaction.request_id.is_not(None), Interaction.created_at >= since)
        .group_by(Interaction.event_type)
    ).all()
    counts = {e: n for e, n in attributed}
    clicks = counts.get("click", 0)
    return {
        "window_days": days,
        "requests": requests,
        "impressions": impressions,
        "attributed_events": counts,
        "ctr": clicks / impressions if impressions else None,
        "conversion_rate": counts.get("purchase", 0) / impressions if impressions else None,
    }
