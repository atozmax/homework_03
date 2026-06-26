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

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BUNDLE_DIR = os.getenv("BUNDLE_DIR", os.path.join(os.path.dirname(__file__), "model"))
MAX_SEQ_LEN = 256
EMBEDDING_DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_tokenizer = None
_device = None
_bundle_dir = None

# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def load_bundle(bundle_dir: str | None = None) -> Tuple:
    """Load model and tokenizer from the bundle directory."""
    global _model, _tokenizer, _device, _bundle_dir

    path = bundle_dir or BUNDLE_DIR
    torch.manual_seed(0)

    model = AutoModel.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    _model = model
    _tokenizer = tokenizer
    _device = device
    _bundle_dir = path

    return model, tokenizer


def _ensure_loaded() -> None:
    if _model is None or _tokenizer is None:
        load_bundle()


def embed(texts: List[str]) -> np.ndarray:
    """Embed a list of texts into a (N, 384) float32 numpy array."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    _ensure_loaded()

    encoded = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )
    encoded = {k: v.to(_device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = _model(**encoded)
        last_hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        normalized = F.normalize(pooled, p=2, dim=1)

    return normalized.detach().cpu().numpy().astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    return float(np.dot(a, b))


def info() -> dict:
    """Return metadata about the loaded bundle."""
    _ensure_loaded()
    return {
        "model_name": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "max_seq_len": MAX_SEQ_LEN,
        "device": str(_device).replace("torch.", ""),
        "framework": f"torch {torch.__version__}",
        "deterministic": True,
        "bundle_dir": _bundle_dir,
    }


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
