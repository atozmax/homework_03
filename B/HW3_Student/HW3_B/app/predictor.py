"""app.predictor — thin wrapper over the bundle's BundlePredictor.

This module exists so the FastAPI layer doesn't import from `predict` directly
(safer refactoring — if the bundle's module name ever changes, only this file
needs to change).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


# TODO: implement embed_texts(predictor, texts) -> np.ndarray
# This function takes a BundlePredictor instance and a list of strings,
# and returns a (B, 384) float32 numpy array of L2-normalized embeddings.
# HINT: call predictor.embed(list(texts)) to get the raw embeddings
# HINT: verify that arr.shape[1] == 384 (expected embedding dimension)
# HINT: the bundle's embed() already returns L2-normalized vectors
# HINT: if texts is empty, return np.zeros((0, 384), dtype=np.float32)


# TODO: (bonus) implement cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray
# Return (a.shape[0], b.shape[0]) cosine similarity matrix.
# HINT: L2-normalize each row of a and b, then compute a_n @ b_n.T
