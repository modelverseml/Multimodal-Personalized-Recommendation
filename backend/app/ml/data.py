"""Load the catalog and interaction log from PostgreSQL into training arrays.

Item indices are 1-based (0 is padding) and follow product id order, so the
same mapping is reproducible at serving time from the saved product id list.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.database import engine

POSITIVE_EVENTS = ("click", "add_to_cart", "purchase", "review")


@dataclass
class Catalog:
    products: pd.DataFrame            # one row per item, ordered by item index - 1
    product_ids: np.ndarray           # item index - 1 -> products.id
    category_codes: np.ndarray        # (n_items + 1,) category code per item index, 0 for padding
    category_names: list[str]
    store_codes: np.ndarray           # (n_items + 1,)

    @property
    def n_items(self) -> int:
        return len(self.product_ids)

    def index_of(self) -> dict[int, int]:
        return {int(pid): i + 1 for i, pid in enumerate(self.product_ids)}


@dataclass
class Split:
    """Leave-last-out split over chronologically ordered sequences."""
    customer_ids: np.ndarray
    train: list[np.ndarray]           # history used for training
    valid: np.ndarray                 # second-to-last item per user (0 if none)
    test: np.ndarray                  # last item per user (0 if none)
    timestamps: list[np.ndarray] = field(default_factory=list)

    def history(self, stage: str) -> list[np.ndarray]:
        """History visible when predicting the given stage's target."""
        if stage == "valid":
            return self.train
        if stage == "test":
            return [np.append(h, v) if v else h for h, v in zip(self.train, self.valid)]
        raise ValueError(stage)

    def targets(self, stage: str) -> np.ndarray:
        return self.valid if stage == "valid" else self.test


def load_catalog() -> Catalog:
    products = pd.read_sql(
        text("""
            SELECT id, asin, title, description, features, category, subcategory,
                   store, price, average_rating, rating_number, image_url
            FROM products ORDER BY id
        """),
        engine,
    )
    categories = sorted(products["category"].fillna("Other").unique())
    cat_lookup = {c: i + 1 for i, c in enumerate(categories)}
    stores = products["store"].fillna("").str.lower()
    store_lookup = {s: i + 1 for i, s in enumerate(sorted(set(stores) - {""}))}
    return Catalog(
        products=products,
        product_ids=products["id"].to_numpy(),
        category_codes=np.concatenate([[0], products["category"].fillna("Other").map(cat_lookup).to_numpy()]),
        category_names=["<pad>"] + categories,
        store_codes=np.concatenate([[0], stores.map(lambda s: store_lookup.get(s, 0)).to_numpy()]),
    )


def load_events(catalog: Catalog, include_live: bool = False) -> pd.DataFrame:
    """Positive interactions in chronological order. Offline benchmarking uses
    only the dataset's events so results are reproducible regardless of what
    has been clicked in the app; production retraining can opt in to live events."""
    sources = ["dataset", "live"] if include_live else ["dataset"]
    events = pd.read_sql(
        text("""
            SELECT customer_id, product_id, created_at
            FROM interactions
            WHERE event_type = ANY(:types) AND source = ANY(:sources)
            ORDER BY customer_id, created_at, id
        """),
        engine,
        params={"types": list(POSITIVE_EVENTS), "sources": sources},
    )
    index = catalog.index_of()
    events["item"] = events["product_id"].map(index)
    return events.dropna(subset=["item"]).astype({"item": np.int64})


def build_split(events: pd.DataFrame, min_len: int = 3) -> Split:
    customer_ids, train, valid, test, stamps = [], [], [], [], []
    ts = events["created_at"].astype("int64").to_numpy() // 10**9
    items = events["item"].to_numpy()
    users = events["customer_id"].to_numpy()
    boundaries = np.flatnonzero(np.diff(users)) + 1
    for start, end in zip(np.r_[0, boundaries], np.r_[boundaries, len(users)]):
        seq = items[start:end]
        customer_ids.append(users[start])
        stamps.append(ts[start:end])
        if len(seq) >= min_len:
            train.append(seq[:-2])
            valid.append(seq[-2])
            test.append(seq[-1])
        else:
            train.append(seq)
            valid.append(0)
            test.append(0)
    return Split(np.array(customer_ids), train, np.array(valid), np.array(test), stamps)


def item_counts(sequences: list[np.ndarray], n_items: int) -> np.ndarray:
    counts = np.zeros(n_items + 1, dtype=np.float64)
    if sequences:
        np.add.at(counts, np.concatenate(sequences), 1)
    return counts


def pad_left(sequences: list[np.ndarray], max_len: int) -> np.ndarray:
    """Left-pad (most recent item last) and truncate to the last max_len items."""
    out = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq = seq[-max_len:]
        if len(seq):
            out[i, -len(seq):] = seq
    return out
