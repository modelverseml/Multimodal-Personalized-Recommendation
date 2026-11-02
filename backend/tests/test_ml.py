import numpy as np
import scipy.sparse as sp
import torch

from app.ml.baselines import ItemKNN, Popularity
from app.ml.data import build_split, pad_left
from app.ml.metrics import ranking_metrics, target_ranks
from app.ml.ranking import FEATURES, FeatureStore
from app.ml.towers import TowerConfig, TwoTower, sampled_softmax_loss
from app.monitoring import psi

N_ITEMS, N_CATS, CLIP = 30, 3, 8


def tiny_model(**flags):
    cfg = TowerConfig(n_items=N_ITEMS, n_categories=N_CATS, clip_dim=CLIP, dim=16, max_len=6,
                      n_layers=1, n_heads=2, **flags)
    cats = torch.randint(1, N_CATS + 1, (N_ITEMS + 1,))
    cats[0] = 0
    clip = torch.nn.functional.normalize(torch.randn(N_ITEMS + 1, CLIP), dim=-1)
    return TwoTower(cfg, cats, clip if cfg.use_text else None, clip if cfg.use_image else None)


def test_metrics_single_target():
    top = np.array([[5, 3, 9], [1, 2, 4], [7, 8, 6]])
    targets = np.array([3, 4, 0])
    assert target_ranks(top, targets).tolist() == [1, 2, -1]
    m = ranking_metrics(top, targets, ks=(1, 3))
    assert m["recall@1"] == 0
    assert np.isclose(m["recall@3"], 2 / 3)
    assert np.isclose(m["ndcg@3"], (1 / np.log2(3) + 1 / np.log2(4)) / 3)


def test_leave_last_out_split_and_padding():
    import pandas as pd
    events = pd.DataFrame({
        "customer_id": [1, 1, 1, 1, 2, 2],
        "item": [4, 5, 6, 7, 8, 9],
        "created_at": pd.to_datetime(range(6), unit="s", utc=True),
    })
    split = build_split(events)
    assert split.train[0].tolist() == [4, 5] and split.valid[0] == 6 and split.test[0] == 7
    assert split.train[1].tolist() == [8, 9] and split.test[1] == 0  # too short to evaluate
    assert split.history("test")[0].tolist() == [4, 5, 6]
    assert pad_left([np.array([1, 2, 3])], 2).tolist() == [[2, 3]]
    assert pad_left([np.array([1])], 3).tolist() == [[0, 0, 1]]


def test_user_tower_is_causal_and_finite_with_padding():
    model = tiny_model().eval()
    hist = torch.tensor([[0, 0, 0, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
    states = model.encode_history(hist)
    assert torch.isfinite(states).all()
    assert torch.allclose(states.norm(dim=-1), torch.ones_like(states[..., 0]), atol=1e-5)
    # Changing a later item must not change earlier positions (causality).
    changed = hist.clone()
    changed[1, -1] = 9
    again = model.encode_history(changed)
    assert torch.allclose(states[1, :-1], again[1, :-1], atol=1e-5)
    assert not torch.allclose(states[1, -1], again[1, -1])


def test_sampled_softmax_trains_and_variants_build():
    torch.manual_seed(0)
    for flags in ({}, {"use_text": False, "use_image": False}, {"use_id": False}):
        model = tiny_model(**flags)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        seqs = torch.randint(1, N_ITEMS + 1, (16, 7))
        log_pop = torch.full((N_ITEMS + 1,), -np.log(N_ITEMS))
        losses = []
        for _ in range(30):
            states = model.encode_history(seqs[:, :-1])
            loss = sampled_softmax_loss(states.reshape(-1, 16), seqs[:, 1:].reshape(-1),
                                        model.item_tower, log_pop, n_uniform=8, temperature=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert np.isfinite(losses).all() and losses[-1] < losses[0]


def test_baselines_and_feature_store_shapes():
    train = [np.array([1, 2, 3]), np.array([2, 3, 4]), np.array([1, 3, 5])]
    knn = ItemKNN(train, 6)
    assert knn.scores([np.array([2])]).shape == (1, 7)
    assert knn.scores([np.array([2])])[0, 3] > 0  # 3 co-occurs with 2
    assert Popularity(train, 6).scores([train[0]])[0, 3] == 3

    n, d = 6, 4
    rng = np.random.default_rng(0)
    store = FeatureStore(
        item_vecs=rng.normal(size=(n + 1, d)).astype(np.float32),
        clip_text=rng.normal(size=(n + 1, d)).astype(np.float32),
        clip_image=rng.normal(size=(n + 1, d)).astype(np.float32),
        has_image=np.ones(n + 1, np.float32), cooc=knn.sim,
        category=np.array([0, 1, 1, 2, 2, 1, 2]), store=np.array([0, 1, 0, 2, 1, 1, 0]),
        popularity=np.ones(n + 1, np.float32), recent_popularity=np.ones(n + 1, np.float32),
        price=np.full(n + 1, np.nan, np.float32),
    )
    hist = pad_left([np.array([1, 2]), np.array([3])], 5)
    cands = np.array([[3, 4, 5], [1, 2, 6]])
    feats = store.compute(hist, cands, np.zeros((2, 3), np.float32))
    assert feats.shape == (2, 3, len(FEATURES))
    cat_aff = feats[..., FEATURES.index("category_affinity")]
    assert np.isclose(cat_aff[0, 0], 0.0) and np.isclose(cat_aff[0, 2], 1.0)  # history all category 1


def test_psi():
    value, _ = psi({"a": 50, "b": 50}, {"a": 50, "b": 50})
    assert abs(value) < 1e-9
    shifted, rows = psi({"a": 90, "b": 10}, {"a": 10, "b": 90})
    assert shifted > 0.2 and rows[0]["bucket"] in {"a", "b"}
