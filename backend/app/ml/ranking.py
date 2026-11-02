"""Second-stage learning-to-rank.

FAISS returns the top candidates by two-tower similarity. A LightGBM
LambdaRank model then re-scores those candidates with richer features the
retrieval model cannot express cheaply: collaborative co-occurrence,
platform/brand affinity, recency-aware popularity, and CLIP visual/text
similarity to the customer's recent items. Features are computed in vectorised
batches so the same code path serves offline training and online requests.
"""
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp

FEATURES = [
    "two_tower_score", "retrieval_rank", "last_item_sim", "max_history_sim",
    "clip_text_sim", "clip_image_sim", "clip_image_max",
    "cooc_score", "cooc_last_item",
    "category_affinity", "same_category_as_last", "brand_affinity",
    "popularity", "recent_popularity", "price", "history_length", "has_image",
]
# Deliberately excluded: the catalog's average_rating / rating_number. They come
# from a 2023 crawl that counts reviews written *after* the evaluation cutoff,
# so they would leak future popularity into offline metrics.

# Human-readable phrasing for explanation chips in the UI.
FEATURE_LABELS = {
    "two_tower_score": "Matches your taste profile",
    "retrieval_rank": "Strong embedding match",
    "last_item_sim": "Similar to what you just viewed",
    "max_history_sim": "Similar to an item you liked",
    "clip_text_sim": "Similar descriptions to your items",
    "clip_image_sim": "Visually similar to your items",
    "clip_image_max": "Looks like something you liked",
    "cooc_score": "Customers with your history bought this",
    "cooc_last_item": "Often bought with your last item",
    "category_affinity": "On a platform you play",
    "same_category_as_last": "Same platform as your last item",
    "brand_affinity": "From a brand you buy",
    "popularity": "Popular overall",
    "recent_popularity": "Trending recently",
    "price": "Price point",
    "history_length": "Based on your activity",
    "has_image": "Has product imagery",
}

RECENT = 10

# On macOS, PyTorch and LightGBM each bundle their own OpenMP runtime, and
# running both multithreaded in one process segfaults. LightGBM is therefore
# kept single-threaded everywhere (~60 ms/tree here; serving scores only a few
# hundred rows), while torch keeps its threads.
LGB_THREADS = 1


@dataclass
class FeatureStore:
    item_vecs: np.ndarray        # (n+1, d) two-tower item vectors
    clip_text: np.ndarray        # (n+1, 512), row 0 zeros
    clip_image: np.ndarray       # (n+1, 512)
    has_image: np.ndarray        # (n+1,)
    cooc: sp.csr_matrix          # (n+1, n+1) ItemKNN similarity
    category: np.ndarray         # (n+1,) category codes
    store: np.ndarray            # (n+1,) brand codes
    popularity: np.ndarray       # (n+1,) log1p counts
    recent_popularity: np.ndarray
    price: np.ndarray            # log1p, NaN if unknown

    def save(self, path):
        sp.save_npz(str(path) + ".cooc.npz", self.cooc)
        np.savez(path, **{k: v for k, v in self.__dict__.items() if k != "cooc"})

    @classmethod
    def load(cls, path):
        data = np.load(str(path) + ".npz")
        return cls(cooc=sp.load_npz(str(path) + ".cooc.npz"), **{k: data[k] for k in data.files})

    def compute(self, hist: np.ndarray, cands: np.ndarray, tt_scores: np.ndarray) -> np.ndarray:
        """hist: (B, L) left-padded item indices; cands, tt_scores: (B, C).
        Returns (B, C, len(FEATURES)) float32."""
        b, c = cands.shape
        n = len(self.category)
        mask = hist > 0
        hist_len = mask.sum(1)
        recent = hist[:, -RECENT:]
        recent_mask = (recent > 0)[..., None]
        last = hist[:, -1]

        cand_vecs = self.item_vecs[cands]                                       # (B, C, d)
        last_sim = np.einsum("bcd,bd->bc", cand_vecs, self.item_vecs[last])
        rec_vecs = self.item_vecs[recent]                                       # (B, R, d)
        max_hist_sim = np.einsum("bcd,brd->bcr", cand_vecs, rec_vecs)
        max_hist_sim = np.where(recent_mask.transpose(0, 2, 1), max_hist_sim, -1).max(2)

        def profile(table):
            vecs = (table[recent] * recent_mask).sum(1)
            return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)

        text_sim = np.einsum("bcd,bd->bc", self.clip_text[cands], profile(self.clip_text))
        cand_img = self.clip_image[cands]
        img_sim = np.einsum("bcd,bd->bc", cand_img, profile(self.clip_image))
        img_max = np.einsum("bcd,brd->bcr", cand_img, self.clip_image[recent])
        img_max = np.where(recent_mask.transpose(0, 2, 1), img_max, -1).max(2)

        rows = np.repeat(np.arange(b), recent.shape[1])
        h_recent = sp.csr_matrix(
            (recent_mask.ravel().astype(np.float32), (rows, recent.ravel())), shape=(b, n),
        )
        cooc_all = (h_recent @ self.cooc).toarray()
        cooc_score = np.take_along_axis(cooc_all, cands, 1)
        cooc_last = np.take_along_axis(self.cooc[last].toarray(), cands, 1)

        def affinity(codes):
            k = codes.max() + 1
            counts = np.zeros((b, k), np.float32)
            np.add.at(counts, (np.repeat(np.arange(b), hist.shape[1]), codes[hist].ravel()), mask.ravel())
            counts[:, 0] = 0
            counts /= np.maximum(hist_len, 1)[:, None]
            return np.take_along_axis(counts, codes[cands], 1)

        cat_aff = affinity(self.category)
        brand_aff = affinity(self.store)
        same_cat = (self.category[cands] == self.category[last][:, None]).astype(np.float32)

        rank = np.broadcast_to(np.arange(c, dtype=np.float32), (b, c))
        feats = [
            tt_scores, rank, last_sim, max_hist_sim, text_sim, img_sim, img_max,
            cooc_score, cooc_last, cat_aff, same_cat, brand_aff,
            self.popularity[cands], self.recent_popularity[cands], self.price[cands],
            np.broadcast_to(np.log1p(hist_len)[:, None], (b, c)), self.has_image[cands],
        ]
        return np.stack(feats, axis=-1).astype(np.float32)


def train_ranker(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 x_val: np.ndarray, y_val: np.ndarray, groups_val: np.ndarray, seed: int = 0) -> lgb.LGBMRanker:
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        lambdarank_truncation_level=20,
        random_state=seed,
        n_jobs=LGB_THREADS,
        verbose=-1,
    )
    ranker.fit(
        x, y, group=groups,
        eval_set=[(x_val, y_val)], eval_group=[groups_val], eval_at=[10],
        feature_name=FEATURES,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return ranker
