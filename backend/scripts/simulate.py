"""Simulate live shoppers against the running API.

    python -m scripts.simulate --shoppers 40 --platform "Nintendo Switch"

Each simulated shopper starts cold, browses a few catalog items, then
repeatedly requests recommendations and clicks/buys the ones that match their
preferred platform. This exercises the real-time path (history -> user tower
-> FAISS -> ranker), populates impressions and attributed clicks for online
CTR, and (with a skewed platform mix) produces measurable PSI drift.
"""
import argparse
import random

import httpx

from app.config import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8001/api")
    parser.add_argument("--shoppers", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--platform", default="Nintendo Switch")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    client = httpx.Client(base_url=args.api, timeout=30)
    token = client.post("/auth/login", json={"email": settings.DEMO_EMAIL, "password": settings.DEMO_PASSWORD})
    token.raise_for_status()
    client.headers["Authorization"] = f"Bearer {token.json()['access_token']}"

    seed_items = client.get("/catalog/products", params={"category": args.platform, "page_size": 96}).json()["items"]
    clicks = impressions = 0
    latencies = []
    for s in range(args.shoppers):
        shopper = client.post("/customers", json={"display_name": f"Sim {args.platform} fan {s + 1}"}).json()
        for item in rng.sample(seed_items, 3):
            client.post("/events", json={"customer_id": shopper["id"], "product_id": item["id"], "event_type": "click"})
        for _ in range(args.rounds):
            recs = client.get(f"/recommendations/{shopper['id']}", params={"k": 12}).json()
            latencies.append(recs["timings"]["total_ms"])
            impressions += len(recs["items"])
            for rec in recs["items"]:
                likes = rec["product"]["category"] == args.platform
                if rng.random() < (0.35 if likes else 0.03):
                    clicks += 1
                    event = "purchase" if rng.random() < 0.2 else "click"
                    client.post("/events", json={"customer_id": shopper["id"], "product_id": rec["product"]["id"],
                                                 "event_type": event, "request_id": recs["request_id"]})
    latencies.sort()
    print(f"{args.shoppers} shoppers, {impressions} impressions, {clicks} clicks "
          f"(CTR {clicks / max(impressions, 1):.1%}); server p50 {latencies[len(latencies) // 2]:.1f} ms, "
          f"p95 {latencies[int(len(latencies) * 0.95)]:.1f} ms")


if __name__ == "__main__":
    main()
