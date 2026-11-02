"""Two-tower retrieval model.

Item tower:  ID embedding + MLP over frozen CLIP (text ++ image) + platform embedding.
             ID dropout during training forces the content path to carry signal,
             so items with few interactions still land in a sensible place.
User tower:  causal self-attention over the item-tower vectors of the customer's
             recent history. The user tower has no per-user parameters; it is a
             function of history, so a brand-new shopper is personalised from
             their first click without retraining.
Both towers emit L2-normalised vectors, so retrieval is maximum inner product
search, which is what FAISS indexes.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TowerConfig:
    n_items: int
    n_categories: int
    clip_dim: int = 512
    dim: int = 128
    max_len: int = 50
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.2
    id_dropout: float = 0.15
    use_id: bool = True
    use_text: bool = True
    use_image: bool = True

    def to_dict(self):
        return asdict(self)


class ItemTower(nn.Module):
    def __init__(self, cfg: TowerConfig, clip_text: torch.Tensor | None, clip_image: torch.Tensor | None):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.category = nn.Embedding(cfg.n_categories + 1, d, padding_idx=0)
        self.id_emb = nn.Embedding(cfg.n_items + 1, d, padding_idx=0) if cfg.use_id else None
        if self.id_emb is not None:
            nn.init.normal_(self.id_emb.weight, std=0.02)

        content = []
        if cfg.use_text:
            content.append(clip_text)
        if cfg.use_image:
            content.append(clip_image)
        if content:
            # Frozen pretrained features; row 0 is the padding item.
            self.register_buffer("clip", torch.cat(content, dim=1).float(), persistent=True)
            self.content = nn.Sequential(
                nn.Linear(self.clip.shape[1], 2 * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(2 * d, d),
            )
        else:
            self.clip = None
            self.content = None
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        h = self.category(self.category_codes[items])
        if self.id_emb is not None:
            ident = self.id_emb(items)
            if self.training and self.cfg.id_dropout > 0 and self.content is not None:
                keep = (torch.rand(items.shape, device=items.device) > self.cfg.id_dropout).unsqueeze(-1)
                ident = ident * keep
            h = h + ident
        if self.content is not None:
            h = h + self.content(self.clip[items])
        return F.normalize(self.out(h), dim=-1)


class UserTower(nn.Module):
    def __init__(self, cfg: TowerConfig):
        super().__init__()
        self.cfg = cfg
        self.position = nn.Embedding(cfg.max_len, cfg.dim)
        layer = nn.TransformerEncoderLayer(
            cfg.dim, cfg.n_heads, 2 * cfg.dim, cfg.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def attention_mask(self, pad: torch.Tensor) -> torch.Tensor:
        """Causal mask that also hides padding keys. Every query may always
        attend to itself, so left-padded rows never produce all-masked (NaN) rows."""
        b, length = pad.shape
        causal = torch.ones(length, length, dtype=torch.bool, device=pad.device).tril()
        allowed = causal.unsqueeze(0) & ~pad.unsqueeze(1)
        allowed |= torch.eye(length, dtype=torch.bool, device=pad.device).unsqueeze(0)
        blocked = ~allowed
        return blocked.repeat_interleave(self.cfg.n_heads, dim=0)

    def forward(self, item_vecs: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """item_vecs: (B, L, d) left-padded history; returns (B, L, d) states,
        where position t summarises the history up to and including t."""
        length = item_vecs.shape[1]
        pos = torch.arange(self.cfg.max_len - length, self.cfg.max_len, device=item_vecs.device)
        h = self.dropout(item_vecs * (self.cfg.dim ** 0.5) + self.position(pos))
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        h = self.encoder(h, mask=self.attention_mask(pad))
        return F.normalize(self.norm(h), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, cfg: TowerConfig, category_codes: torch.Tensor,
                 clip_text: torch.Tensor | None = None, clip_image: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        self.item_tower = ItemTower(cfg, clip_text, clip_image)
        self.item_tower.register_buffer("category_codes", category_codes.long(), persistent=True)
        self.user_tower = UserTower(cfg)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        """history: (B, L) left-padded item indices -> (B, L, d) user states."""
        pad = history == 0
        vecs = self.item_tower(history)
        return self.user_tower(vecs, pad)

    def user_vector(self, history: torch.Tensor) -> torch.Tensor:
        return self.encode_history(history)[:, -1]

    @torch.no_grad()
    def all_item_vectors(self, batch: int = 8192) -> torch.Tensor:
        """(n_items + 1, d) matrix; row 0 (padding) is zeroed."""
        device = next(self.parameters()).device
        idx = torch.arange(self.cfg.n_items + 1, device=device)
        vecs = torch.cat([self.item_tower(idx[i:i + batch]) for i in range(0, len(idx), batch)])
        vecs[0] = 0
        return vecs


def sampled_softmax_loss(
    queries: torch.Tensor, positives: torch.Tensor, item_tower: ItemTower,
    log_pop: torch.Tensor, n_uniform: int, temperature: float,
) -> torch.Tensor:
    """Mixed negative sampling: in-batch positives (popularity-distributed) plus
    shared uniform negatives, with logQ correction so popular items are not
    over-penalised for appearing as negatives more often (Yi et al., 2019;
    Yang et al., 2020)."""
    unique, labels = torch.unique(positives, return_inverse=True)
    n_items = item_tower.cfg.n_items
    uniform = torch.randint(1, n_items + 1, (n_uniform,), device=queries.device)
    candidates = torch.cat([unique, uniform])
    logits = queries @ item_tower(candidates).T / temperature

    # Expected number of times each candidate is drawn as a negative.
    expected = len(positives) * log_pop[candidates].exp() + n_uniform / n_items
    logits = logits - expected.log()
    # A uniform draw that equals the row's own positive is not a negative.
    collision = uniform.unsqueeze(0) == positives.unsqueeze(1)
    in_batch = torch.zeros(len(positives), len(unique), dtype=torch.bool, device=queries.device)
    logits = logits.masked_fill(torch.cat([in_batch, collision], dim=1), float("-inf"))
    return F.cross_entropy(logits, labels)
