"""Black-box integration tests: start the real API (uvicorn) in a subprocess
and exercise it over HTTP. A subprocess keeps the torch-free serving stack
isolated from the PyTorch imports in test_ml.py (see app/ml/inference.py).
Skipped when the database, trained artifacts or demo credentials are missing."""
import socket
import subprocess
import sys
import time

import httpx
import pytest
from sqlalchemy import text

from app.config import BACKEND_DIR, settings
from app.database import engine

try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1 FROM products LIMIT 1"))
    DB_OK = True
except Exception:
    DB_OK = False

pytestmark = pytest.mark.skipif(
    not DB_OK or not (settings.artifact_dir / "current" / "manifest.json").exists() or not settings.DEMO_PASSWORD,
    reason="needs ingested database, trained artifacts and demo credentials",
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"],
        cwd=BACKEND_DIR,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def client(base_url):
    c = httpx.Client(base_url=f"{base_url}/api", timeout=60)
    token = c.post("/auth/login", json={"email": settings.DEMO_EMAIL, "password": settings.DEMO_PASSWORD})
    assert token.status_code == 200
    c.headers["Authorization"] = f"Bearer {token.json()['access_token']}"
    yield c
    c.close()


def test_auth_required_and_bad_login(base_url):
    assert httpx.get(f"{base_url}/api/catalog/products").status_code in (401, 403)
    bad = httpx.post(f"{base_url}/api/auth/login", json={"email": settings.DEMO_EMAIL, "password": "wrong"})
    assert bad.status_code == 401


def test_catalog_and_similar_items(client):
    page = client.get("/catalog/products", params={"page_size": 5}).json()
    assert page["total"] > 1000 and len(page["items"]) == 5
    detail = client.get(f"/catalog/products/{page['items'][0]['id']}").json()
    assert len(detail["similar"]) > 0


def test_recommendations_exclude_history_and_log_impressions(client):
    customer = client.get("/customers", params={"min_history": 10, "limit": 1}).json()[0]
    detail = client.get(f"/customers/{customer['id']}", params={"limit": 200}).json()
    seen = {h["product"]["id"] for h in detail["history"]}
    recs = client.get(f"/recommendations/{customer['id']}", params={"k": 10}).json()
    assert recs["strategy"] == "two_tower+ltr"
    assert len(recs["items"]) == 10
    assert not seen & {r["product"]["id"] for r in recs["items"]}
    assert recs["timings"]["total_ms"] < 100
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM impressions WHERE request_id = :r"),
                         {"r": recs["request_id"]}).scalar()
    assert n == 10


def test_new_shopper_cold_start_then_realtime_personalisation(client):
    shopper = client.post("/customers", json={"display_name": "pytest shopper"}).json()
    cold = client.get(f"/recommendations/{shopper['id']}").json()
    assert cold["strategy"] == "trending"

    switch = client.get("/catalog/products", params={"category": "Nintendo Switch", "page_size": 3}).json()["items"]
    for p in switch:
        r = client.post("/events", json={"customer_id": shopper["id"], "product_id": p["id"], "event_type": "click"})
        assert r.status_code == 200
    warm = client.get(f"/recommendations/{shopper['id']}", params={"k": 12}).json()
    assert warm["strategy"] == "two_tower+ltr" and warm["history_length"] == 3
    share = sum(r["product"]["category"] == "Nintendo Switch" for r in warm["items"]) / len(warm["items"])
    assert share >= 0.5, share

    with engine.begin() as conn:  # keep the demo database tidy
        conn.execute(text("DELETE FROM customers WHERE id = :id"), {"id": shopper["id"]})


def test_semantic_search(client):
    res = client.get("/catalog/search", params={"q": "racing wheel controller", "k": 10}).json()
    assert len(res["items"]) == 10
    titles = " ".join(i["product"]["title"].lower() for i in res["items"])
    assert "wheel" in titles or "racing" in titles


def test_api_process_is_torch_free():
    out = subprocess.run(
        [sys.executable, "-c", "import sys, app.main; from app.ml.inference import Recommender; "
                               "print('torch' in sys.modules)"],
        cwd=BACKEND_DIR, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False"
