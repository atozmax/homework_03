#!/usr/bin/env python
"""predict.py — Self-contained embedding inference.

MUST implement exactly 4 functions:
    load_bundle()    → (model, tokenizer)
    embed(texts)     → np.ndarray shape (N, 384)
    similarity(a, b) → float
    info()           → dict

The 7-step pipeline in embed():
    1. Tokenize (padding=True, truncation=True, max_length=256, return_tensors="pt")
    2. Move tensors to device
    3. Forward pass under torch.no_grad()
    4. Mean-pool weighted by attention mask: sum(H * mask) / sum(mask).clamp(min=1e-9)
    5. L2 normalize: F.normalize(pooled, p=2, dim=1)
    6. Move to CPU, convert to numpy float32
    7. Return

DO NOT import sentence_transformers. Use raw transformers only.
"""
from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple

# HINT: from transformers import AutoModel, AutoTokenizer
# HINT: import torch, torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BUNDLE_DIR = os.getenv("BUNDLE_DIR", os.path.join(os.path.dirname(__file__), "model"))
MAX_SEQ_LEN = 256
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# TODO: Implement these 4 functions
# ---------------------------------------------------------------------------

def load_bundle(bundle_dir: str | None = None) -> Tuple:
    """Load model and tokenizer from the bundle directory.

    Args:
        bundle_dir: Path to bundle/model/. Defaults to BUNDLE_DIR env var.

    Returns:
        (model, tokenizer) tuple. model is in eval mode on the correct device.
        tokenizer is loaded from the same directory.

    HINT: AutoModel.from_pretrained(bundle_dir), AutoTokenizer.from_pretrained(bundle_dir)
    """
    # TODO: implement
    # HINT: set torch.manual_seed(0) for determinism
    # HINT: model.eval(), model.to(device)
    raise NotImplementedError("TODO: load_bundle()")


def embed(texts: List[str]) -> np.ndarray:
    """Embed a list of texts into a (N, 384) float32 numpy array.

    The 7-step pipeline:
        1. tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        2. move to device
        3. model(**encoded) under torch.no_grad()
        4. mean-pool: sum(last_hidden * mask) / sum(mask).clamp(min=1e-9)
        5. L2 normalize: F.normalize(pooled, p=2, dim=1)
        6. detach().cpu().numpy().astype(np.float32)
        7. return

    Args:
        texts: List of strings to embed. Can be empty.

    Returns:
        np.ndarray of shape (len(texts), 384), dtype float32.
        For empty input, returns shape (0, 384).

    HINT: if not texts: return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    """
    # TODO: implement the 7-step pipeline
    # HINT: mask = encoded["attention_mask"].unsqueeze(-1).float()
    # HINT: summed = (last_hidden * mask).sum(dim=1)
    # HINT: counts = mask.sum(dim=1).clamp(min=1e-9)
    # HINT: pooled = summed / counts
    raise NotImplementedError("TODO: embed()")


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        a: First embedding vector (384,).
        b: Second embedding vector (384,).

    Returns:
        float: Cosine similarity in [-1, 1].

    HINT: For L2-normalized vectors, cosine = dot product
    HINT: float((a * b).sum()) — OR use np.dot(a, b)
    """
    # TODO: implement
    raise NotImplementedError("TODO: similarity()")


def info() -> dict:
    """Return metadata about the loaded bundle.

    Returns:
        dict with keys: model_name, embedding_dim, max_seq_len, device,
        framework, deterministic, bundle_dir.

    HINT: return {"model_name": "sentence-transformers/all-MiniLM-L6-v2", ...}
    """
    # TODO: implement
    # HINT: check if model is loaded, if not call load_bundle() first
    raise NotImplementedError("TODO: info()")


# ---------------------------------------------------------------------------
# CLI entry point (used by scripts/gen_manifest.py and for testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser(description="Bundle embed CLI")
    p.add_argument("--text", action="append", default=[], help="repeatable text input")
    p.add_argument("--texts-file", help="JSON list of strings")
    p.add_argument("--out", help="optional .npy output path")
    p.add_argument("--info", action="store_true", help="print info and exit")
    args = p.parse_args()

    if args.info:
        print(json.dumps(info(), indent=2, default=str))
        raise SystemExit(0)

    texts: list[str] = list(args.text)
    if args.texts_file:
        with open(args.texts_file, encoding="utf-8") as f:
            texts.extend(json.load(f))

    if not texts:
        print("ERROR: provide --text or --texts-file", file=sys.stderr)
        raise SystemExit(2)

    emb = embed(texts)
    if args.out:
        np.save(args.out, emb)
        print(f"Saved {emb.shape} to {args.out}")
    else:
        print(json.dumps([[round(float(x), 6) for x in row] for row in emb]))
