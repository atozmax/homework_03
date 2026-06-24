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
