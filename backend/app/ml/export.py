"""Export the online models to ONNX so serving does not depend on PyTorch.

- user_tower.onnx: (item_vecs [1, L, d], pad [1, L]) -> user vector [1, d]
  Item vectors are precomputed, so the item tower never runs online.
- clip_text.onnx:  (input_ids, attention_mask) -> L2-normalised CLIP text embedding
  for semantic search.

Each export is checked against the PyTorch module on fresh inputs and the max
absolute difference is recorded in the manifest.

Shapes: serving always scores one request at a time, so batch is static (1).
Sequence length is dynamic with a minimum of 2, because torch.export
specialises size-0/1 dimensions; callers left-pad shorter inputs (padding is
masked out of attention, so the result is unchanged).
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.nn import functional as F

from app.ml.towers import UserTower

OPSET = 18


class _UserVector(torch.nn.Module):
    def __init__(self, tower: UserTower):
        super().__init__()
        self.tower = tower

    def forward(self, item_vecs, pad):
        return self.tower(item_vecs, pad)[:, -1]


class _ClipText(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, input_ids, attention_mask):
        return F.normalize(self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask), dim=-1)


def export_user_tower(tower: UserTower, path: Path) -> float:
    module = _UserVector(tower.cpu().eval()).eval()
    d, max_len = tower.cfg.dim, tower.cfg.max_len
    vecs = F.normalize(torch.randn(1, 8, d), dim=-1)
    pad = torch.zeros(1, 8, dtype=torch.bool)
    pad[0, :5] = True
    length = torch.export.Dim("length", min=2, max=max_len)
    torch.onnx.export(
        module, (vecs, pad), str(path), input_names=["item_vecs", "pad"], output_names=["user"],
        dynamic_shapes=({1: length}, {1: length}), dynamo=True, opset_version=OPSET,
        external_data=False, verbose=False,
    )
    session = ort.InferenceSession(str(path))
    worst = 0.0
    for n, n_pad in ((2, 1), (2, 0), (7, 3), (max_len, 0), (max_len, 40)):
        v = F.normalize(torch.randn(1, n, d), dim=-1)
        p = torch.zeros(1, n, dtype=torch.bool)
        p[0, :n_pad] = True
        with torch.no_grad():
            ref = module(v, p).numpy()
        out = session.run(None, {"item_vecs": v.numpy(), "pad": p.numpy()})[0]
        worst = max(worst, float(np.abs(ref - out).max()))
    return worst


def export_clip_text(model_name: str, out_dir: Path) -> float:
    from transformers import CLIPModel, CLIPTokenizerFast

    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
    tokenizer.save_pretrained(out_dir / "clip_tokenizer")
    module = _ClipText(CLIPModel.from_pretrained(model_name).eval()).eval()
    sample = tokenizer(["cozy farming game for kids"], return_tensors="pt")
    length = torch.export.Dim("length", min=2, max=77)
    path = out_dir / "clip_text.onnx"
    torch.onnx.export(
        module, (sample["input_ids"], sample["attention_mask"]), str(path),
        input_names=["input_ids", "attention_mask"], output_names=["embeds"],
        dynamic_shapes=({1: length}, {1: length}), dynamo=True, opset_version=OPSET,
        external_data=False, verbose=False,
    )
    session = ort.InferenceSession(str(path))
    worst = 0.0
    for query in ("rpg", "dark horror survival game", "retro pixel-art role playing game " * 12):
        tokens = tokenizer([query], truncation=True, max_length=77, return_tensors="pt")
        with torch.no_grad():
            ref = module(tokens["input_ids"], tokens["attention_mask"]).numpy()
        out = session.run(None, {"input_ids": tokens["input_ids"].numpy(),
                                 "attention_mask": tokens["attention_mask"].numpy()})[0]
        worst = max(worst, float(np.abs(ref - out).max()))
    return worst
