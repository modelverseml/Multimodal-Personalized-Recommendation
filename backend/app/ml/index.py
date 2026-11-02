"""FAISS indexes.

- Item index: HNSW over two-tower item vectors (inner product == cosine since
  vectors are L2-normalised). Used for candidate retrieval and "similar items".
- Search indexes: exact inner product over CLIP image embeddings and, separately,
  CLIP text embeddings. A query is matched against both what a product looks
  like and what its listing says, and the two rankings are merged with
  reciprocal rank fusion (text-text and text-image cosines live on different
  scales, so their raw scores should not be added).

This module must not import PyTorch: on macOS the FAISS and PyTorch wheels
each bundle an OpenMP runtime and cannot share a process. Training therefore
builds indexes by running `python -m app.ml.index <run_dir>` as a child process.
"""
import sys
from pathlib import Path

import faiss
import numpy as np

HNSW_M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 256


def build_item_index(item_vecs: np.ndarray) -> faiss.Index:
    """item_vecs: (n+1, d) with row 0 = padding. FAISS ids are item indices."""
    d = item_vecs.shape[1]
    hnsw = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = EF_CONSTRUCTION
    index = faiss.IndexIDMap2(hnsw)
    ids = np.arange(1, len(item_vecs), dtype=np.int64)
    index.add_with_ids(np.ascontiguousarray(item_vecs[1:], dtype=np.float32), ids)
    set_ef_search(index, EF_SEARCH)
    return index


def set_ef_search(index: faiss.Index, ef: int) -> None:
    faiss.downcast_index(index.index).hnsw.efSearch = ef


def build_flat_index(vectors: np.ndarray, ids: np.ndarray) -> faiss.Index:
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
    index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), ids.astype(np.int64))
    return index


def build_search_indexes(clip_text: np.ndarray, clip_image: np.ndarray,
                         has_image: np.ndarray) -> tuple[faiss.Index, faiss.Index]:
    """(image_index, text_index); rows of the inputs are item indices, row 0 is padding.
    Items without a photo are simply absent from the image index."""
    ids = np.arange(len(clip_text))
    with_image = ids[(ids > 0) & has_image]
    return build_flat_index(clip_image[with_image], with_image), build_flat_index(clip_text[1:], ids[1:])


def save(index: faiss.Index, path) -> None:
    faiss.write_index(index, str(path))


def load(path) -> faiss.Index:
    return faiss.read_index(str(path))


def build_all(run_dir: Path) -> None:
    item_vecs = np.load(run_dir / "item_vecs.npy")
    feats = np.load(run_dir / "features.npz")
    save(build_item_index(item_vecs), run_dir / "items.hnsw")
    image_index, text_index = build_search_indexes(feats["clip_text"], feats["clip_image"],
                                                   feats["has_image"].astype(bool))
    save(image_index, run_dir / "search_image.flat")
    save(text_index, run_dir / "search_text.flat")
    print(f"built FAISS indexes for {len(item_vecs) - 1:,} items in {run_dir}")


if __name__ == "__main__":
    build_all(Path(sys.argv[1]))
