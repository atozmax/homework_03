"""app.client_qdrant — Qdrant client (read-only API key)."""
from __future__ import annotations

import time
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from . import config


# TODO: create a singleton QdrantClient instance
# HINT: use a module-level variable _client: Optional[QdrantClient] = None
# HINT: implement get_client() -> QdrantClient that lazily creates the client
# HINT: QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None, timeout=10.0)

# TODO: implement ping() -> bool
# Try to list collections to verify connectivity. Return True if reachable, False otherwise.
# HINT: call get_client().get_collections()

# TODO: implement vector_count(collection: str) -> Optional[int]
# Return the number of vectors in a collection, or None on error.
# HINT: get_client().get_collection(collection_name=collection).vectors_count

# TODO: implement search(collection, vector, top_k, lang, primary, exclude_neutral) -> List[models.ScoredPoint]
# Run an ANN search with optional payload filters.
# HINT: build a qdrant_client.http.models.Filter with must/must_not conditions
# HINT: must conditions for lang and primary (if provided)
# HINT: must_not condition for primary="neutral" if exclude_neutral is True
# HINT: call get_client().search(collection_name=..., query_vector=..., limit=..., query_filter=..., with_payload=True, with_vectors=False)

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY or None,
            timeout=10.0,
        )
    return _client


def ping() -> bool:
    try:
        get_client().get_collections()
        return True
    except Exception:
        return False


def vector_count(collection: str) -> Optional[int]:
    try:
        info = get_client().get_collection(collection_name=collection)
        return info.vectors_count
    except Exception:
        return None


def search(
    collection: str,
    vector: List[float],
    top_k: int,
    lang: Optional[str] = None,
    primary: Optional[str] = None,
    exclude_neutral: bool = True,
) -> List[models.ScoredPoint]:
    must: List[models.FieldCondition] = []
    must_not: List[models.FieldCondition] = []
    if lang is not None:
        must.append(
            models.FieldCondition(key="lang", match=models.MatchValue(value=lang))
        )
    if primary is not None:
        must.append(
            models.FieldCondition(key="primary", match=models.MatchValue(value=primary))
        )
    if exclude_neutral:
        must_not.append(
            models.FieldCondition(
                key="primary", match=models.MatchValue(value="neutral")
            )
        )
    query_filter = None
    if must or must_not:
        query_filter = models.Filter(
            must=must or None,
            must_not=must_not or None,
        )
    return get_client().search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )
