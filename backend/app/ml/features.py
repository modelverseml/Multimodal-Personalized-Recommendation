"""CLIP text and image embeddings for every catalog item.

Both modalities are encoded with the same pretrained CLIP model, so they share
an embedding space; that is what makes text-to-image semantic search work.
Embeddings are cached per ASIN, so re-running only encodes new products.
Targets the Transformers 4.x API (get_*_features return projected tensors).
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.config import settings


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def product_text(row) -> str:
    parts = [row["title"], f"Platform: {row['category']}."]
    if row.get("subcategory"):
        parts.append(f"{row['subcategory']}.")
    if row.get("store"):
        parts.append(f"By {row['store']}.")
    parts.append(row.get("features") or "")
    parts.append(row.get("description") or "")
    # CLIP's text encoder sees at most 77 tokens; the tokenizer truncates.
    return " ".join(p for p in parts if p)[:1000]


def cache_path() -> Path:
    slug = settings.CLIP_MODEL.replace("/", "__")
    return settings.artifact_dir / "features" / f"clip_{slug}.npz"


class ClipEncoder:
    def __init__(self, model_name: str | None = None, device: torch.device | None = None):
        self.device = device or best_device()
        name = model_name or settings.CLIP_MODEL
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(name)
        self.dim = self.model.config.projection_dim

    @torch.inference_mode()
    def encode_text(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            tokens = self.processor.tokenizer(
                texts[i:i + batch_size], padding=True, truncation=True,
                max_length=77, return_tensors="pt",
            ).to(self.device)
            feats = self.model.get_text_features(**tokens)
            out.append(torch.nn.functional.normalize(feats, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    @torch.inference_mode()
    def encode_images(self, paths: list[Path | None], batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """Returns (embeddings, has_image). Missing/corrupt images get a zero vector."""
        emb = np.zeros((len(paths), self.dim), np.float32)
        ok = np.zeros(len(paths), bool)

        def load(path):
            if path is None or not Path(path).exists():
                return None
            try:
                with Image.open(path) as im:
                    return self.processor.image_processor(im.convert("RGB"), return_tensors="pt")["pixel_values"][0]
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            for i in range(0, len(paths), batch_size):
                pixels = list(pool.map(load, paths[i:i + batch_size]))
                idx = [j for j, p in enumerate(pixels) if p is not None]
                if not idx:
                    continue
                batch = torch.stack([pixels[j] for j in idx]).to(self.device)
                feats = torch.nn.functional.normalize(self.model.get_image_features(pixel_values=batch), dim=-1)
                rows = np.array(idx) + i
                emb[rows] = feats.float().cpu().numpy()
                ok[rows] = True
                if (i // batch_size) % 40 == 0:
                    print(f"  images {i + len(idx):,}/{len(paths):,}")
        return emb, ok


def build_item_features(products: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return text/image CLIP matrices aligned with `products` rows."""
    path = cache_path()
    cached = {}
    if path.exists():
        data = np.load(path, allow_pickle=False)
        for a, t, im, ok in zip(data["asin"], data["text"], data["image"], data["has_image"]):
            cached[str(a)] = (t, im, ok)

    todo = products[~products["asin"].isin(cached.keys())]
    if len(todo):
        print(f"encoding {len(todo):,} items with {settings.CLIP_MODEL}")
        start = time.perf_counter()
        encoder = ClipEncoder()
        texts = [product_text(r) for r in todo.to_dict("records")]
        text_emb = encoder.encode_text(texts)
        paths = [settings.image_dir / f"{a}.jpg" for a in todo["asin"]]
        image_emb, has_image = encoder.encode_images(paths)
        for a, t, im, ok in zip(todo["asin"], text_emb, image_emb, has_image):
            cached[a] = (t, im, ok)
        print(f"CLIP encoding took {time.perf_counter() - start:.1f}s on {encoder.device}")

        path.parent.mkdir(parents=True, exist_ok=True)
        asins = list(cached)
        np.savez(
            path,
            asin=np.array(asins),
            text=np.stack([cached[a][0] for a in asins]),
            image=np.stack([cached[a][1] for a in asins]),
            has_image=np.array([cached[a][2] for a in asins]),
        )

    rows = [cached[a] for a in products["asin"]]
    return {
        "text": np.stack([r[0] for r in rows]).astype(np.float32),
        "image": np.stack([r[1] for r in rows]).astype(np.float32),
        "has_image": np.array([r[2] for r in rows]),
    }


if __name__ == "__main__":
    from app.ml.data import load_catalog

    catalog = load_catalog()
    feats = build_item_features(catalog.products)
    print({k: v.shape for k, v in feats.items()}, "image coverage:", feats["has_image"].mean().round(4))
