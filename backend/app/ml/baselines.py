"""Reference recommenders the two-tower model has to beat.

Every scorer maps a batch of histories to dense scores over item indices
0..n_items (column 0 is padding and is masked by the evaluator).
"""
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F

from app.ml.data import item_counts


def history_matrix(histories: list[np.ndarray], n_items: int, recent: int | None = None) -> sp.csr_matrix:
    rows, cols = [], []
    for i, h in enumerate(histories):
        h = h[-recent:] if recent else h
        rows.append(np.full(len(h), i))
        cols.append(h)
    rows = np.concatenate(rows) if rows else np.array([], int)
    cols = np.concatenate(cols) if cols else np.array([], int)
    m = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(histories), n_items + 1))
    m.data[:] = 1.0  # binarise repeats
    return m


def item_knn_matrix(train: list[np.ndarray], n_items: int, topk: int = 100, shrink: float = 10.0) -> sp.csr_matrix:
    """Item-item cosine similarity on the binary user-item matrix, pruned to
    the top-k neighbours per item (classic item-based CF, Sarwar et al. 2001)."""
    x = history_matrix(train, n_items)
    co = (x.T @ x).tocsr().astype(np.float32)
    co.setdiag(0)
    co.eliminate_zeros()
    deg = np.asarray(x.sum(0)).ravel()
    co = co.tocoo()
    co.data = co.data / (np.sqrt(deg[co.row] * deg[co.col]) + shrink)
    co = co.tocsr()
    rows, cols, vals = [], [], []
    for i in range(co.shape[0]):
        start, end = co.indptr[i], co.indptr[i + 1]
        if start == end:
            continue
        data, idx = co.data[start:end], co.indices[start:end]
        if len(data) > topk:
            keep = np.argpartition(-data, topk)[:topk]
            data, idx = data[keep], idx[keep]
        rows.append(np.full(len(idx), i))
        cols.append(idx)
        vals.append(data)
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=co.shape,
    )


class Popularity:
    name = "Popularity"

    def __init__(self, train: list[np.ndarray], n_items: int):
        self.counts = item_counts(train, n_items).astype(np.float32)

    def scores(self, histories: list[np.ndarray]) -> np.ndarray:
        return np.tile(self.counts, (len(histories), 1))


class ItemKNN:
    name = "ItemKNN-CF"

    def __init__(self, train: list[np.ndarray], n_items: int, topk: int = 100):
        self.n_items = n_items
        self.sim = item_knn_matrix(train, n_items, topk)

    def scores(self, histories: list[np.ndarray]) -> np.ndarray:
        return (history_matrix(histories, self.n_items) @ self.sim).toarray()


class ClipKNN:
    """Zero-training content baseline: cosine between candidate CLIP features
    and the mean CLIP features of the customer's 10 most recent items."""
    name = "CLIP-kNN (content)"

    def __init__(self, clip_text: np.ndarray, clip_image: np.ndarray, recent: int = 10):
        both = np.concatenate([clip_text, clip_image], axis=1)
        norms = np.linalg.norm(both, axis=1, keepdims=True)
        self.items = (both / np.maximum(norms, 1e-8)).astype(np.float32)
        self.recent = recent

    def scores(self, histories: list[np.ndarray]) -> np.ndarray:
        profile = np.stack([self.items[h[-self.recent:]].mean(0) for h in histories])
        return profile @ self.items.T


class BPRMF:
    """Matrix factorisation with Bayesian Personalised Ranking loss (Rendle et al., 2009).
    Transductive: it can only score users that were present at training time."""
    name = "BPR-MF"

    def __init__(self, train: list[np.ndarray], n_items: int, dim: int = 64, epochs: int = 30,
                 lr: float = 5e-3, reg: float = 1e-5, device: str = "cpu", seed: int = 0):
        torch.manual_seed(seed)
        self.device = device
        users = np.concatenate([np.full(len(h), u) for u, h in enumerate(train)])
        items = np.concatenate(train)
        self.user = torch.nn.Embedding(len(train), dim).to(device)
        self.item = torch.nn.Embedding(n_items + 1, dim).to(device)
        torch.nn.init.normal_(self.user.weight, std=0.05)
        torch.nn.init.normal_(self.item.weight, std=0.05)
        opt = torch.optim.Adam(list(self.user.parameters()) + list(self.item.parameters()), lr=lr)
        u_all = torch.as_tensor(users, device=device)
        i_all = torch.as_tensor(items, device=device)
        batch = 8192
        for _ in range(epochs):
            perm = torch.randperm(len(u_all), device=device)
            for s in range(0, len(perm), batch):
                idx = perm[s:s + batch]
                u, i = u_all[idx], i_all[idx]
                j = torch.randint(1, n_items + 1, (len(idx),), device=device)
                eu, ei, ej = self.user(u), self.item(i), self.item(j)
                loss = -F.logsigmoid((eu * ei).sum(-1) - (eu * ej).sum(-1)).mean()
                loss = loss + reg * (eu.pow(2).sum() + ei.pow(2).sum() + ej.pow(2).sum()) / len(idx)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.user_matrix = self.user.weight.detach().cpu().numpy()
        self.item_matrix = self.item.weight.detach().cpu().numpy()

    def scores_for_users(self, user_rows: np.ndarray) -> np.ndarray:
        return self.user_matrix[user_rows] @ self.item_matrix.T
