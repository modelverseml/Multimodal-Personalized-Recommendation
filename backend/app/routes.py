import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from prometheus_client import Counter, Histogram
from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

from app import monitoring
from app.config import settings
from app.database import get_db
from app.ml.data import POSITIVE_EVENTS
from app.ml.inference import ArtifactsMissing, Recommender, get_recommender, reload_recommender
from app.models import AppUser, Customer, Impression, Interaction, Product
from app.schemas import (
    CustomerCreate, CustomerDetail, CustomerOut, EventCreate, EventOut, HistoryEvent, LoginRequest,
    ProductDetail, ProductOut, ProductPage, ProductWithSimilar, Recommendation, RecommendationResponse,
    ScoredProduct, SearchResult, TokenOut, UserOut,
)
from app.security import create_token, current_user, verify_password

router = APIRouter()
auth = [Depends(current_user)]

LATENCY = Histogram(
    "recommendation_latency_seconds", "Recommendation latency by stage", ["stage"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
REQUESTS = Counter("recommendation_requests_total", "Recommendation requests", ["strategy"])
EVENTS = Counter("feedback_events_total", "Feedback events received", ["event_type"])


def recommender() -> Recommender:
    try:
        return get_recommender()
    except ArtifactsMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def products_by_id(db: Session, ids) -> dict[int, Product]:
    ids = list(set(ids))
    if not ids:
        return {}
    return {p.id: p for p in db.scalars(select(Product).where(Product.id.in_(ids)))}


def customer_history(db: Session, customer_id: int, limit: int) -> list[int]:
    """Most recent positive interactions, oldest first, each product kept at
    its latest position (a click followed by a purchase counts once)."""
    rows = db.execute(
        select(Interaction.product_id)
        .where(Interaction.customer_id == customer_id, Interaction.event_type.in_(POSITIVE_EVENTS))
        .order_by(Interaction.created_at.desc(), Interaction.id.desc())
        .limit(limit * 2)
    ).scalars().all()
    seen, ordered = set(), []
    for pid in rows:
        if pid not in seen:
            seen.add(pid)
            ordered.append(pid)
    return ordered[:limit][::-1]


# ------------------------------------------------------------------ auth
@router.post("/auth/login", response_model=TokenOut)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(AppUser).where(AppUser.email == body.email))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return TokenOut(access_token=create_token(user), user=UserOut.model_validate(user))


@router.get("/auth/me", response_model=UserOut)
def me(user: AppUser = Depends(current_user)):
    return user


# ------------------------------------------------------------------ catalog
@router.get("/catalog/categories", dependencies=auth)
def categories(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Product.category, func.count()).group_by(Product.category).order_by(func.count().desc())
    ).all()
    return [{"name": c, "count": n} for c, n in rows]


@router.get("/catalog/products", response_model=ProductPage, dependencies=auth)
def list_products(
    q: str | None = None,
    category: str | None = None,
    sort: str = Query("popular", pattern="^(popular|rating|price_asc|price_desc)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=96),
    db: Session = Depends(get_db),
):
    stmt = select(Product)
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(Product.title.ilike(like), Product.store.ilike(like)))
    if category:
        stmt = stmt.where(Product.category == category)
    total = db.scalar(select(func.count()).select_from(stmt.subquery()))
    order = {
        "popular": [Product.rating_number.desc()],
        "rating": [Product.average_rating.desc().nulls_last(), Product.rating_number.desc()],
        "price_asc": [Product.price.asc().nulls_last()],
        "price_desc": [Product.price.desc().nulls_last()],
    }[sort]
    items = db.scalars(stmt.order_by(*order, Product.id).offset((page - 1) * page_size).limit(page_size)).all()
    return ProductPage(items=items, total=total, page=page, page_size=page_size)


@router.get("/catalog/products/{product_id}", response_model=ProductWithSimilar, dependencies=auth)
def product_detail(product_id: int, k: int = Query(12, le=48), db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    similar = []
    try:
        hits = get_recommender().similar(product_id, k)
        lookup = products_by_id(db, [h["product_id"] for h in hits])
        similar = [ScoredProduct(product=lookup[h["product_id"]], score=h["score"])
                   for h in hits if h["product_id"] in lookup]
    except ArtifactsMissing:
        pass
    return ProductWithSimilar(product=ProductDetail.model_validate(product), similar=similar)


@router.get("/catalog/search", response_model=SearchResult, dependencies=auth)
def semantic_search(q: str = Query(..., min_length=2, max_length=200), k: int = Query(24, le=96),
                    db: Session = Depends(get_db)):
    hits, ms = recommender().search(q, k)
    lookup = products_by_id(db, [h["product_id"] for h in hits])
    return SearchResult(query=q, latency_ms=ms, items=[
        ScoredProduct(product=lookup[h["product_id"]], score=h["score"]) for h in hits if h["product_id"] in lookup
    ])


# ------------------------------------------------------------------ customers
@router.get("/customers", response_model=list[CustomerOut], dependencies=auth)
def list_customers(
    q: str | None = None,
    min_history: int = Query(8, ge=0),
    limit: int = Query(24, le=100),
    db: Session = Depends(get_db),
):
    """Live shoppers first, then a stable pseudo-random sample of dataset shoppers."""
    params = {"min_history": min_history, "limit": limit, "q": f"%{q}%" if q else None}
    rows = db.execute(text("""
        WITH counts AS (
            SELECT customer_id, COUNT(*) AS n FROM interactions GROUP BY customer_id
        ),
        picked AS (
            SELECT c.id, c.external_id, c.display_name, c.source, COALESCE(n, 0) AS n, c.created_at
            FROM customers c
            LEFT JOIN counts ON counts.customer_id = c.id
            WHERE (c.source = 'live' OR COALESCE(n, 0) >= :min_history)
              AND (CAST(:q AS TEXT) IS NULL OR c.display_name ILIKE :q OR c.external_id ILIKE :q
                   OR CAST(c.id AS TEXT) = TRIM(BOTH '%' FROM :q))
            ORDER BY (c.source = 'live') DESC, c.created_at DESC, md5(c.id::text)
            LIMIT :limit
        )
        -- Top category only for the returned page, not the whole table.
        SELECT picked.id, picked.external_id, picked.display_name, picked.source, picked.n, t.category
        FROM picked
        LEFT JOIN LATERAL (
            SELECT p.category FROM interactions i JOIN products p ON p.id = i.product_id
            WHERE i.customer_id = picked.id
            GROUP BY p.category ORDER BY COUNT(*) DESC LIMIT 1
        ) t ON true
        ORDER BY (picked.source = 'live') DESC, picked.created_at DESC, md5(picked.id::text)
    """), params).all()
    return [CustomerOut(id=r[0], external_id=r[1], display_name=r[2], source=r[3],
                        interaction_count=r[4], top_category=r[5]) for r in rows]


@router.post("/customers", response_model=CustomerOut, dependencies=auth)
def create_customer(body: CustomerCreate, db: Session = Depends(get_db)):
    customer = Customer(external_id=f"live-{uuid.uuid4().hex[:12]}", display_name="", source="live")
    db.add(customer)
    db.flush()
    customer.display_name = body.display_name or f"New shopper {customer.id}"
    db.commit()
    return CustomerOut(id=customer.id, external_id=customer.external_id, display_name=customer.display_name,
                       source=customer.source, interaction_count=0)


@router.get("/customers/{customer_id}", response_model=CustomerDetail, dependencies=auth)
def customer_detail(customer_id: int, limit: int = Query(50, le=200), db: Session = Depends(get_db)):
    customer = db.get(Customer, customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    events = db.execute(
        select(Interaction, Product)
        .join(Product, Product.id == Interaction.product_id)
        .where(Interaction.customer_id == customer_id)
        .order_by(Interaction.created_at.desc(), Interaction.id.desc())
        .limit(limit)
    ).all()
    total = db.scalar(select(func.count()).where(Interaction.customer_id == customer_id))
    mix: dict[str, float] = {}
    for _, p in events:
        mix[p.category] = mix.get(p.category, 0) + 1
    n = sum(mix.values()) or 1
    mix = dict(sorted(((k, v / n) for k, v in mix.items()), key=lambda kv: -kv[1]))
    return CustomerDetail(
        id=customer.id, external_id=customer.external_id, display_name=customer.display_name,
        source=customer.source, interaction_count=total, top_category=next(iter(mix), None),
        category_mix=mix,
        history=[HistoryEvent(product=p, event_type=i.event_type, rating=i.rating, created_at=i.created_at)
                 for i, p in events],
    )


# ------------------------------------------------------------------ recommendations
@router.get("/recommendations/{customer_id}", response_model=RecommendationResponse, dependencies=auth)
def recommend(
    customer_id: int,
    k: int = Query(12, ge=1, le=50),
    rerank: bool = True,
    db: Session = Depends(get_db),
):
    service = recommender()
    if db.get(Customer, customer_id) is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    history = customer_history(db, customer_id, settings.MAX_HISTORY)
    result = service.recommend(history, k=k, rerank=rerank)
    for stage, ms in result["timings"].items():
        LATENCY.labels(stage.removesuffix("_ms")).observe(ms / 1000)
    REQUESTS.labels(result["strategy"]).inc()

    ids = [r["product_id"] for r in result["items"]] + [r["anchor_product_id"] for r in result["items"]
                                                        if r["anchor_product_id"]]
    lookup = products_by_id(db, ids)
    request_id = str(uuid.uuid4())
    items = []
    for pos, r in enumerate(result["items"], start=1):
        if r["product_id"] not in lookup:
            continue
        items.append(Recommendation(
            position=pos, product=lookup[r["product_id"]], score=r["score"],
            retrieval_rank=r["retrieval_rank"], reasons=r["reasons"],
            anchor=lookup.get(r["anchor_product_id"]),
        ))
        db.add(Impression(request_id=request_id, customer_id=customer_id, product_id=r["product_id"],
                          position=pos, score=r["score"], strategy=result["strategy"],
                          model_version=service.version))
    db.commit()
    return RecommendationResponse(
        request_id=request_id, customer_id=customer_id, model_version=service.version,
        strategy=result["strategy"], history_length=len(history), timings=result["timings"], items=items,
    )


@router.post("/events", response_model=EventOut, dependencies=auth)
def record_event(body: EventCreate, db: Session = Depends(get_db)):
    if db.get(Customer, body.customer_id) is None or db.get(Product, body.product_id) is None:
        raise HTTPException(status_code=404, detail="Unknown customer or product")
    event = Interaction(customer_id=body.customer_id, product_id=body.product_id, event_type=body.event_type,
                        source="live", request_id=body.request_id)
    db.add(event)
    db.commit()
    EVENTS.labels(body.event_type).inc()
    return EventOut(id=event.id, customer_id=event.customer_id, product_id=event.product_id,
                    event_type=event.event_type, created_at=event.created_at)


# ------------------------------------------------------------------ model + evaluation
@router.get("/model", dependencies=auth)
def model_info():
    m = recommender().manifest
    return {k: m.get(k) for k in ("run_id", "model_version", "created_at", "dataset", "clip_model",
                                  "n_items", "n_customers", "best_epoch", "validation", "train_config",
                                  "tower_config")}


@router.post("/model/reload", dependencies=auth)
def model_reload():
    try:
        service = reload_recommender()
    except ArtifactsMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"model_version": service.version}


@router.get("/evaluation", dependencies=auth)
def evaluation():
    path = settings.artifact_dir / "evaluation.json"
    report = json.loads(path.read_text()) if path.exists() else None
    manifest = {}
    try:
        manifest = get_recommender().manifest
    except ArtifactsMissing:
        pass
    return {
        "report": report,
        "training": {
            "model_version": manifest.get("model_version"),
            "run_id": manifest.get("run_id"),
            "history": manifest.get("history", []),
            "ranker": manifest.get("ranker"),
        },
    }


@router.get("/monitoring", dependencies=auth)
def monitoring_report(days: int = Query(7, ge=1, le=90), db: Session = Depends(get_db)):
    manifest = recommender().manifest
    return {
        "drift": monitoring.drift_report(db, manifest, days=days),
        "engagement": monitoring.engagement(db, days=days),
    }
