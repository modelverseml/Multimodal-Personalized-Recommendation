"""Load the Amazon Reviews 2023 benchmark into PostgreSQL and fetch product images.

Usage (from backend/):
    python -m scripts.ingest                 # download + load + images
    python -m scripts.ingest --skip-images

Data: McAuley Lab, "Bridging Language and Items for Retrieval and
Recommendation" (Hou et al., 2024). We use the official 5-core
leave-last-out split so offline metrics are comparable with the literature.
Interactions are inserted train -> valid -> test, so ordering by
(created_at, id) reproduces the official split exactly, even on timestamp ties.
"""
import argparse
import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psycopg
from sqlalchemy import select

from app.config import settings
from app.database import Base, SessionLocal, engine
from app.models import AppUser
from app.security import hash_password

HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"
SPLITS = ("train", "valid", "test")


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.rename(dest)


def fetch_raw(category: str) -> dict[str, Path]:
    raw = settings.raw_dir
    files = {}
    for split in SPLITS:
        name = f"{category}.{split}.csv"
        files[split] = raw / name
        download(f"{HF_BASE}/benchmark/5core/last_out_w_his/{name}", files[split])
    files["meta"] = raw / f"meta_{category}.jsonl"
    download(f"{HF_BASE}/raw/meta_categories/meta_{category}.jsonl", files["meta"])
    return files


def clean(text) -> str:
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    return " ".join(str(text or "").split())


def parse_category(categories: list[str], main_category: str | None) -> tuple[str, str]:
    """Platform-level category (e.g. 'Nintendo Switch') plus the remaining path."""
    path = [c for c in categories if c and c != "Video Games"]
    if path and path[0] == "Legacy Systems" and len(path) > 1:
        path = path[1:]
    if not path:
        return (main_category or "Other"), ""
    return path[0], " > ".join(path[1:])


def parse_price(value) -> float | None:
    try:
        price = float(str(value).replace("$", "").replace(",", ""))
        return price if price > 0 else None
    except (TypeError, ValueError):
        return None


def load_rows(files: dict[str, Path]):
    interactions = []
    for split in SPLITS:
        with open(files[split], newline="") as f:
            for row in csv.DictReader(f):
                interactions.append((row["user_id"], row["parent_asin"], float(row["rating"]), int(row["timestamp"])))
    needed = {asin for _, asin, _, _ in interactions}

    products = {}
    with open(files["meta"]) as f:
        for line in f:
            item = json.loads(line)
            asin = item["parent_asin"]
            if asin not in needed:
                continue
            images = item.get("images") or []
            main = next((im for im in images if im.get("variant") == "MAIN"), images[0] if images else {})
            category, subcategory = parse_category(item.get("categories") or [], item.get("main_category"))
            products[asin] = {
                "asin": asin,
                "title": clean(item.get("title")) or asin,
                "description": clean(item.get("description"))[:4000],
                "features": clean(item.get("features"))[:4000],
                "category": category[:128],
                "subcategory": subcategory[:255],
                "store": clean(item.get("store"))[:255] or None,
                "price": parse_price(item.get("price")),
                "average_rating": item.get("average_rating"),
                "rating_number": int(item.get("rating_number") or 0),
                "image_url": main.get("large") or main.get("hi_res"),
            }
    missing = needed - products.keys()
    if missing:
        print(f"warning: {len(missing)} items have no metadata; dropping their interactions")
        interactions = [r for r in interactions if r[1] in products]
    return products, interactions


def bulk_load(products: dict, interactions: list) -> None:
    print("recreating schema")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    product_ids = {asin: i for i, asin in enumerate(products, start=1)}
    customer_ids: dict[str, int] = {}
    for user, *_ in interactions:
        customer_ids.setdefault(user, len(customer_ids) + 1)

    dsn = settings.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
    now = datetime.now(timezone.utc)
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cols = ["id", "asin", "title", "description", "features", "category", "subcategory",
                "store", "price", "average_rating", "rating_number", "image_url", "created_at"]
        with cur.copy(f"COPY products ({', '.join(cols)}) FROM STDIN") as copy:
            for asin, p in products.items():
                copy.write_row([product_ids[asin]] + [p[c] for c in cols[1:-1]] + [now])
        print(f"products: {len(products):,}")

        with cur.copy("COPY customers (id, external_id, display_name, source, created_at) FROM STDIN") as copy:
            for ext, cid in customer_ids.items():
                copy.write_row([cid, ext, f"Shopper {cid}", "dataset", now])
        print(f"customers: {len(customer_ids):,}")

        with cur.copy(
            "COPY interactions (id, customer_id, product_id, event_type, rating, source, created_at) FROM STDIN"
        ) as copy:
            for i, (user, asin, rating, ts) in enumerate(interactions, start=1):
                copy.write_row([
                    i, customer_ids[user], product_ids[asin], "review", rating, "dataset",
                    datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                ])
        print(f"interactions: {len(interactions):,}")

        for table in ("products", "customers", "interactions"):
            cur.execute(f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), (SELECT MAX(id) FROM {table}))")
        cur.execute("ANALYZE")
        conn.commit()


def ensure_demo_user() -> None:
    if not settings.DEMO_EMAIL or not settings.DEMO_PASSWORD:
        print("DEMO_EMAIL / DEMO_PASSWORD not set; skipping demo account")
        return
    with SessionLocal() as db:
        user = db.scalar(select(AppUser).where(AppUser.email == settings.DEMO_EMAIL))
        if user is None:
            user = AppUser(email=settings.DEMO_EMAIL, role="admin", password_hash="")
            db.add(user)
        user.password_hash = hash_password(settings.DEMO_PASSWORD)
        db.commit()
    print(f"demo account: {settings.DEMO_EMAIL}")


async def download_images(products: dict, concurrency: int = 48) -> None:
    out = settings.image_dir
    out.mkdir(parents=True, exist_ok=True)
    todo = [(a, p["image_url"]) for a, p in products.items()
            if p["image_url"] and not (out / f"{a}.jpg").exists()]
    print(f"images to download: {len(todo):,}")
    sem = asyncio.Semaphore(concurrency)
    done = failed = 0

    async def fetch(client: httpx.AsyncClient, asin: str, url: str):
        nonlocal done, failed
        async with sem:
            for attempt in range(3):
                try:
                    r = await client.get(url)
                    r.raise_for_status()
                    (out / f"{asin}.jpg").write_bytes(r.content)
                    break
                except httpx.HTTPError:
                    if attempt == 2:
                        failed += 1
                    await asyncio.sleep(1 + attempt)
            done += 1
            if done % 2000 == 0:
                print(f"  {done:,}/{len(todo):,} images")

    async with httpx.AsyncClient(timeout=30, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0 (research; recsys demo)"}) as client:
        await asyncio.gather(*(fetch(client, a, u) for a, u in todo))
    print(f"images done ({failed} failed)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=settings.DATASET_CATEGORY)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--images-only", action="store_true")
    args = parser.parse_args()

    files = fetch_raw(args.category)
    products, interactions = load_rows(files)
    if not args.images_only:
        bulk_load(products, interactions)
        ensure_demo_user()
    if not args.skip_images:
        asyncio.run(download_images(products))


if __name__ == "__main__":
    main()
