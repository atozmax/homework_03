"""app.search — hybrid (Qdrant ANN + PG audit) search orchestration."""
from __future__ import annotations

import time
from typing import List, Optional

from . import client_pg, client_qdrant
from . import config
from .schemas import SearchHit


# TODO: implement hybrid_search(query_vector, top_k, lang, primary, exclude_neutral) -> tuple[List[SearchHit], float]
# This function orchestrates a hybrid search:
#   1. Call client_qdrant.search() to get Qdrant ANN hits
#   2. Extract IDs from Qdrant results
#   3. Call client_pg.fetch_corpus_hits() to get the source-of-truth rows from Postgres
#   4. Zip Qdrant hits with PG rows to build SearchHit objects
#   5. Return (hits, took_ms) where took_ms is elapsed time in milliseconds
#
# HINT: qdr_hits = client_qdrant.search(collection=config.QDRANT_COLLECTION, vector=query_vector, top_k=top_k, lang=lang, primary=primary, exclude_neutral=exclude_neutral)
# HINT: ids = [str(h.id) for h in qdr_hits]
# HINT: pg_rows = client_pg.fetch_corpus_hits(ids)
# HINT: SearchHit(id=str(h.id), score=float(h.score), text=row["text"], primary=row["primary_label"], labels=list(row["labels"]), lang=row["lang"], source=row["source"])
# HINT: use time.perf_counter() to measure elapsed time
# HINT: return empty list if no Qdrant hits: [], (time.perf_counter() - t0) * 1000.0
